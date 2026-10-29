#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate nvkmd-sys/src/<target>.rs: bindings of the Mesa headers the nvkmd backend implements and calls.

    mesa/gen-nvkmd-bindings.py           write the bindings of every target with a configured build
    mesa/gen-nvkmd-bindings.py --check   exit 1 when a committed file differs from what the build generates

The input is nvkmd-sys/wrapper.h, compiled with the exact include paths and defines of nvkmd.c in each
target's NVK build (build/nvk-<target>/compile_commands.json): layouts depend on the target's pointer
width and on configuration defines such as HAVE_FUTEX. Android targets add the NDK sysroot, as Mesa's
own bindgen runs do (mesa/build.py BINDGEN_EXTRA_CLANG_ARGS). bindgen is the one Mesa's Rust build pins
(build/nak-tools, installed by mesa/compiler/build.py --tools-only; BINDGEN=<path> overrides it); the
version is checked so that every checkout generates the same bytes.

bindgen's other inputs are pinned too, so that a difference means stale bindings and nothing else:
libclang and the clang whose builtin headers and search paths bindgen adds are the pinned NDK's
(clang-sys otherwise picks the newest libclang.so and the first clang on PATH), rustfmt is the
toolchain's from rust-toolchain.toml, and BINDGEN_EXTRA_CLANG_ARGS* from the caller are dropped.
The host target still reads the host's libc and libdrm headers: those are the headers the host
NVK build compiles against.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "nvkmd-sys"
WRAPPER = CRATE / "wrapper.h"
BINDGEN_VERSION = "0.72.1"  # mesa/compiler/build.py
RUST_TARGET = "1.99.1"  # rust-toolchain.toml
NDK = ROOT / "build/android-work/ndk/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64"

# build dir -> (output module, clang target, Android triple or None)
TARGETS = {
    "host": ("x86_64_linux_gnu", "x86_64-unknown-linux-gnu", None),
    "android": ("x86_64_linux_android", "x86_64-linux-android", "x86_64-linux-android"),
    "android32": ("i686_linux_android", "i686-linux-android", "i686-linux-android"),
}

# What the backend implements (the nvkmd vtables, vk_sync_type), fills (nv_device_info), recovers its
# objects from (nvk_device, vk_device's loss hooks) and calls (the functions below), plus the entries the
# offline harness (nvkmd-harness) drives it through (the probe, Mesa's memory wrappers). Everything these
# reach is generated too; vk_device's other members are left opaque.
ALLOWLIST_TYPES = [
    "nvkmd_.*", "vk_sync", "vk_sync_type", "vk_sync_wait", "vk_sync_signal", "vk_sync_features",
    "vk_sync_wait_flags", "vk_sync_flags", "nv_device_info", "nv_zcull_device_info", "util_vma_heap",
    "_drmDevice", "_drmVersion", "nvkmd_sys_.*",
]
ALLOWLIST_FUNCTIONS = [
    "nvkmd_mem_init", "nvkmd_dev_alloc_va", "nvkmd_va_bind_mem", "nvkmd_va_free", "nvkmd_va_unbind",
    "vk_sync_wait_many", "vk_sync_signal", "util_vma_heap_init", "util_vma_heap_finish",
    "util_vma_heap_alloc", "util_vma_heap_alloc_addr", "util_vma_heap_free",
    "util_cache_granularity", "__vk_errorf", "_vk_device_set_lost", "drmGetVersion", "drmFreeVersion",
    "nvkmd_try_create_pdev_for_drm", "nvkmd_dev_init", "nvkmd_dev_finish", "nvkmd_mem_finish",
    "nvkmd_dev_info_init_shared_mem_sizes", "open", "close",
    "fstat", "nvkmd_dev_alloc_mem", "nvkmd_dev_alloc_tiled_mem", "nvkmd_dev_import_dma_buf",
    "nvkmd_mem_map", "nvkmd_mem_unmap", "nvkmd_mem_unref",
]
# libdrm's device-record constants (xf86drm.h) and the open flags of the render node (fcntl.h): macros.
ALLOWLIST_VARS = ["DRM_NODE_RENDER", "DRM_NODE_MAX", "DRM_BUS_PCI", "O_RDWR", "O_CLOEXEC"]
OPAQUE_TYPES = ["vk_device", "vk_object_base", "vk_physical_device", "vk_instance", "list_head"]


def nvkmd_flags(build):
    """Include paths and defines of nvkmd.c in the NVK build dir `build`, as absolute clang arguments."""
    database = build / "compile_commands.json"
    if not database.exists():
        return None
    for entry in json.loads(database.read_text()):
        if entry["file"].endswith("src/nouveau/vulkan/nvkmd/nvkmd.c"):
            break
    else:
        sys.exit(f"gen-nvkmd-bindings: nvkmd.c is not in {database}")
    directory = Path(entry["directory"])
    args = entry["arguments"] if "arguments" in entry else shlex.split(entry["command"])
    flags = []
    for arg in args[1:]:
        if arg.startswith("-I"):
            flags.append("-I" + str((directory / arg[2:]).resolve()))
        elif arg.startswith(("-D", "-U", "-std=")):
            flags.append(arg)
    return flags


def pinned_env():
    """The environment for bindgen: every input that is not the headers comes from a pinned source."""
    libclang = NDK / "lib/libclang.so"
    clang = NDK / "bin/clang"
    for path in (libclang, clang):
        if not path.exists():
            sys.exit(f"gen-nvkmd-bindings: no {path}; run mesa/build.py once to download the NDK")
    env = {name: value for name, value in os.environ.items() if not name.startswith("BINDGEN_EXTRA_CLANG_ARGS")}
    env["LIBCLANG_PATH"] = str(libclang.parent)
    env["CLANG_PATH"] = str(clang)
    env["RUSTUP_TOOLCHAIN"] = RUST_TARGET
    return env


def generate(tool, target, build):
    module, clang_target, triple = TARGETS[target]
    flags = nvkmd_flags(build)
    if flags is None:
        return None
    command = [str(tool), str(WRAPPER), "--rust-target", RUST_TARGET, "--rust-edition", "2024",
               "--use-core", "--no-doc-comments", "--no-prepend-enum-name", "--default-enum-style", "consts",
               "--raw-line", "// SPDX-License-Identifier: MIT",
               "--raw-line", f"// Generated by mesa/gen-nvkmd-bindings.py for {clang_target}; do not edit."]
    for name in ALLOWLIST_TYPES:
        command += ["--allowlist-type", name]
    for name in ALLOWLIST_FUNCTIONS:
        command += ["--allowlist-function", name]
    for name in ALLOWLIST_VARS:
        command += ["--allowlist-var", name]
    for name in OPAQUE_TYPES:
        command += ["--opaque-type", name]
    command += ["--", f"--target={clang_target}"] + flags
    if triple:
        command += [f"--sysroot={NDK}/sysroot", "-isystem", f"{NDK}/sysroot/usr/include/{triple}",
                    "-D__ANDROID_API__=33"]
    run = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=pinned_env())
    if run.returncode != 0:
        sys.exit(f"gen-nvkmd-bindings: bindgen failed for {target}:\n{run.stderr}")
    return module, run.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    parser.add_argument("--target", choices=sorted(TARGETS), action="append",
                        help="only this build (repeatable; default: every configured build)")
    parser.add_argument("--build", type=Path,
                        help="the NVK build dir of the single --target (default build/nvk-<target>)")
    args = parser.parse_args()
    if args.build and len(args.target or []) != 1:
        parser.error("--build needs exactly one --target")
    tool = Path(os.environ.get("BINDGEN", ROOT / "build/nak-tools/bin/bindgen"))
    if not tool.exists():
        sys.exit(f"gen-nvkmd-bindings: no {tool}; run mesa/compiler/build.py --tools-only --download-deps")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout.split()
    if version != ["bindgen", BINDGEN_VERSION]:
        sys.exit(f"gen-nvkmd-bindings: need bindgen {BINDGEN_VERSION}, found {' '.join(version)}")
    stale = []
    for target in args.target or sorted(TARGETS):
        build = args.build or ROOT / f"build/nvk-{target}"
        result = generate(tool, target, build)
        if result is None:
            if args.target:
                sys.exit(f"gen-nvkmd-bindings: {build} is not configured")
            continue
        module, text = result
        path = CRATE / f"src/{module}.rs"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(text)
    if stale:
        sys.exit("gen-nvkmd-bindings: stale " + ", ".join(stale) + "; run mesa/gen-nvkmd-bindings.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
