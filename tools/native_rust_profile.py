#!/usr/bin/env python3
"""Explicit local Rust profile/artifact identities for source-native build gates."""
import hashlib
import json
from pathlib import Path

PROFILES = ('dev', 'release')
TARGETS = {'host': None, 'android': 'x86_64-linux-android', 'android32': 'i686-linux-android'}
# NVK's kernel-driver backend over the RM: nvkmd_rm.c, or the Rust nvkmd-rm static library
# (mesa/build.py --nvkmd), each in its own build dir.
BACKENDS = ('c', 'rust')
# Sources the Rust backend's archive is built from, on top of the RM's.
RUST_BACKEND_CRATES = ('nvkmd-rm', 'nvkmd-rm-logic', 'nvkmd-sys', 'nv-ffi')


def rust_library(root, target, profile='dev'):
    if profile not in PROFILES or target not in TARGETS:
        raise ValueError(f'unsupported Rust target/profile: {target}/{profile}')
    path = Path(root) / 'build/nv-rm-rust'
    if TARGETS[target]:
        path /= TARGETS[target]
    return path / ('debug' if profile == 'dev' else 'release') / 'libnv_bionic_native.so'


def nvk_dir(root, target, nvkmd='c'):
    """The NVK build dir of `target` with the `nvkmd` backend."""
    if target not in TARGETS or nvkmd not in BACKENDS:
        raise ValueError(f'unsupported NVK target/backend: {target}/{nvkmd}')
    return Path(root) / ('build/nvk-' + target + ('-rust' if nvkmd == 'rust' else ''))


def metadata_path(root, target, nvkmd='c'):
    return nvk_dir(root, target, nvkmd) / 'rust-build.json'


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def source_snapshot(root, nvkmd='c'):
    root = Path(root)
    crate, mesa = root / 'rm', root / 'mesa'
    crates = [crate, root / 'rm-abi', root / 'rm-core', root / 'rm-sys', root / 'video']
    if nvkmd == 'rust':
        crates += [root / name for name in RUST_BACKEND_CRATES]
    paths = sorted(path for member in crates for path in (member / 'src').rglob('*.rs'))
    paths += [member / 'Cargo.toml' for member in crates]
    paths += [root / 'Cargo.toml', root / 'Cargo.lock', mesa / 'nv_rm.h',
              mesa / 'nvkmd_rm.h', mesa / 'nvkmd_rm.c', mesa / 'build.py',
              mesa / 'build-compat.patch', mesa / 'rm-integration.patch',
              root / 'tools/native_rust_profile.py']
    if not any(path.suffix == '.rs' for path in paths):
        raise RuntimeError(f'missing Rust source inputs under {crate / "src"}')
    return {str(path.relative_to(root)): identity(path)['sha256'] for path in paths}


def write_metadata(root, target, profile, command, compiler, flag_environment, sources_before, nvkmd='c'):
    if source_snapshot(root, nvkmd) != sources_before:
        raise RuntimeError(f"Source inputs changed while building {target}; rebuild before gating/staging")
    record = {'format': 1, 'target': target, 'rust_profile': profile, 'nvkmd': nvkmd,
              'cargo_command': list(map(str, command)), 'rustc': compiler,
              'flag_environment': flag_environment, 'source_sha256': sources_before,
              'rust': identity(rust_library(root, target, profile)),
              'nvk': identity(nvk_dir(root, target, nvkmd) / 'src/nouveau/vulkan/libvulkan_nouveau.so')}
    path = metadata_path(root, target, nvkmd)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)
    return record


def verify_metadata(root, target, profile='dev', nvkmd='c'):
    path = metadata_path(root, target, nvkmd)
    remedy = f'rebuild mesa/build.py --target {target} --rust-profile {profile} --nvkmd {nvkmd}'
    try:
        record = json.loads(path.read_text())
        # A record without a backend is a C build.
        recorded = (record.get('format'), record.get('target'), record.get('rust_profile'),
                    record.get('nvkmd', 'c'))
        if recorded != (1, target, profile, nvkmd):
            raise ValueError('recorded target/profile/backend differs from requested selection')
        if record.get('source_sha256') != source_snapshot(root, nvkmd):
            raise ValueError('source inputs changed since successful build')
        expected = {'rust': rust_library(root, target, profile),
                    'nvk': nvk_dir(root, target, nvkmd) / 'src/nouveau/vulkan/libvulkan_nouveau.so'}
        for name, artifact in expected.items():
            if record.get(name) != identity(artifact):
                raise ValueError(f'{name} artifact path/hash differs from the successful build')
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise RuntimeError(f'Rust build provenance unavailable/mismatched at {path}: {error}; {remedy}') from error
    return record
