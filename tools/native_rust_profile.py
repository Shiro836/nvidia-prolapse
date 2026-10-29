#!/usr/bin/env python3
"""Explicit local Rust profile/artifact identities for source-native build gates."""
import hashlib
import json
from pathlib import Path

PROFILES = ('dev', 'release')
TARGETS = {'host': None, 'android': 'x86_64-linux-android', 'android32': 'i686-linux-android'}
# Crates NVK's kernel-driver backend over the RM (the nvkmd-rm static library linked into
# libvulkan_nouveau) is built from, on top of the RM's.
NVKMD_CRATES = ('nvkmd-rm', 'nvkmd-rm-logic', 'nvkmd-sys', 'nv-ffi')


def rust_library(root, target, profile='dev'):
    if profile not in PROFILES or target not in TARGETS:
        raise ValueError(f'unsupported Rust target/profile: {target}/{profile}')
    path = Path(root) / 'build/nv-rm-rust'
    if TARGETS[target]:
        path /= TARGETS[target]
    return path / ('debug' if profile == 'dev' else 'release') / 'libnv_bionic_native.so'


def nvk_dir(root, target):
    """The NVK build dir of `target`."""
    if target not in TARGETS:
        raise ValueError(f'unsupported NVK target: {target}')
    return Path(root) / ('build/nvk-' + target)


def metadata_path(root, target):
    return nvk_dir(root, target) / 'rust-build.json'


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def source_snapshot(root):
    root = Path(root)
    crate, mesa = root / 'rm', root / 'mesa'
    crates = [crate, root / 'rm-abi', root / 'rm-core', root / 'rm-sys', root / 'video']
    crates += [root / name for name in NVKMD_CRATES]
    paths = sorted(path for member in crates for path in (member / 'src').rglob('*.rs'))
    paths += [member / 'Cargo.toml' for member in crates]
    paths += [root / 'Cargo.toml', root / 'Cargo.lock', mesa / 'nv_rm.h',
              mesa / 'nvkmd_rm.h', mesa / 'build.py',
              mesa / 'build-compat.patch', mesa / 'rm-integration.patch',
              root / 'tools/native_rust_profile.py']
    if not any(path.suffix == '.rs' for path in paths):
        raise RuntimeError(f'missing Rust source inputs under {crate / "src"}')
    return {str(path.relative_to(root)): identity(path)['sha256'] for path in paths}


def write_metadata(root, target, profile, command, compiler, flag_environment, sources_before):
    if source_snapshot(root) != sources_before:
        raise RuntimeError(f"Source inputs changed while building {target}; rebuild before gating/staging")
    record = {'format': 2, 'target': target, 'rust_profile': profile,
              'cargo_command': list(map(str, command)), 'rustc': compiler,
              'flag_environment': flag_environment, 'source_sha256': sources_before,
              'rust': identity(rust_library(root, target, profile)),
              'nvk': identity(nvk_dir(root, target) / 'src/nouveau/vulkan/libvulkan_nouveau.so')}
    path = metadata_path(root, target)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)
    return record


def verify_metadata(root, target, profile='dev'):
    path = metadata_path(root, target)
    remedy = f'rebuild mesa/build.py --target {target} --rust-profile {profile}'
    try:
        record = json.loads(path.read_text())
        # Format 1 records also describe builds with the C nvkmd backend: never accepted.
        recorded = (record.get('format'), record.get('target'), record.get('rust_profile'))
        if recorded != (2, target, profile):
            raise ValueError('recorded format/target/profile differs from requested selection')
        if record.get('source_sha256') != source_snapshot(root):
            raise ValueError('source inputs changed since successful build')
        expected = {'rust': rust_library(root, target, profile),
                    'nvk': nvk_dir(root, target) / 'src/nouveau/vulkan/libvulkan_nouveau.so'}
        for name, artifact in expected.items():
            if record.get(name) != identity(artifact):
                raise ValueError(f'{name} artifact path/hash differs from the successful build')
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise RuntimeError(f'Rust build provenance unavailable/mismatched at {path}: {error}; {remedy}') from error
    return record
