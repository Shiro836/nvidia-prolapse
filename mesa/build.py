#!/usr/bin/env python3
"""Build full pinned Mesa NVK for the host and Android (no deployment)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
import native_rust_profile as profiles
import cc_launcher  # NATIVE_CC_LAUNCHER (ccache) in front of the C/C++ compilers
from fetch import fetch, retry_run  # pinned downloads: mirrors, retries, verified before kept
import android_libelf

PIN = "ddaf85e353442d9d7f192fdcf1823e652e3f2ee3"  # tag mesa-26.2.5 (2026-10-14)
SOURCE = ROOT / "build/mesa-rm-source"
TOOLS = ROOT / "build/nak-tools/bin"
DEPS = ROOT / "build/nvk-host-deps"
NDK = ROOT / "build/android-work/ndk/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64"
ANDROID = ROOT / "build/android-work/deps"


def ninja_load():
    """`-l N` when NINJA_LOAD is set (packaging/native/build-native.sh builds targets concurrently: each ninja
    keeps -j but stops starting jobs while the load average is above N)."""
    return ["-l", os.environ["NINJA_LOAD"]] if os.environ.get("NINJA_LOAD") else []

def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)

def setup_run(command, args, **kwargs):
    """meson setup; with --download-deps it may fetch wraps (crates for NAK), so retry like other downloads."""
    if getattr(args, "download_deps", False):
        retry_run(command, **kwargs)
    else:
        run(command, **kwargs)

def apply(patch):
    reverse = subprocess.run(["git", "-C", str(SOURCE), "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if reverse.returncode:
        run(["git", "-C", SOURCE, "apply", "--check", patch])
        run(["git", "-C", SOURCE, "apply", patch])

def host_dependencies(download):
    """Host-only tools: verify archive hashes before local extraction."""
    archives = DEPS / "downloads"
    archives.mkdir(parents=True, exist_ok=True)
    prefix = DEPS / "root"
    prefix.mkdir(exist_ok=True)
    items = json.loads((HERE / "host-build-deps.json").read_text())
    # extracted once per pin set: concurrent target builds run host tools (mesa_clc) from this prefix, so
    # it is never rewritten under them
    stamp = prefix / ".extracted"
    want = "".join(sorted(f"{i['filename']} {i['sha256']}\n" for i in items))
    if stamp.is_file() and stamp.read_text() == want:
        return
    for item in items:
        path = archives / item["filename"]
        if not path.exists():
            if not download:
                raise SystemExit(f"Missing {path}; use --download-deps for local build tools")
            # the Arch Linux Archive keeps every package version; a live mirror only while it is current
            name = path.name.rsplit("-", 3)[0]
            fetch([f"https://archive.archlinux.org/packages/{name[0]}/{name}/{path.name}",
                   f"https://geo.mirror.pkgbuild.com/extra/os/x86_64/{path.name}"], path, item["sha256"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path}")
        # Limit extraction to usr/: package manager metadata is not needed.
        run(["bsdtar", "-xf", path, "-C", prefix, "usr"])
    for directory in ("lib/pkgconfig", "share/pkgconfig"):
        for pc in (prefix / "usr" / directory).glob("*.pc"):
            text = pc.read_text()
            text = text.replace("prefix=/usr", f"prefix={prefix}/usr")
            text = text.replace("datarootdir=/usr", f"datarootdir={prefix}/usr")
            text = text.replace("datadir=/usr", f"datadir={prefix}/usr")
            pc.write_text(text)
    stamp.write_text(want)

def android_cross_file(bits=64, out=None, download=False):
    """Meson cross file + pkg-config dir for an Android ABI: build/nvk-<target>.ini and
    build/nvk-<target>-pkgconfig (out=None), or both under `out`."""
    target = "android" if bits == 64 else "android32"
    triple = "x86_64-linux-android" if bits == 64 else "i686-linux-android"
    cpu = "x86_64" if bits == 64 else "x86"
    drm_cpu = "x86_64" if bits == 64 else "x86"
    pkg = ROOT / f"build/nvk-{target}-pkgconfig" if out is None else out / "pkgconfig"
    pkg.mkdir(parents=True, exist_ok=True)
    runtime = ANDROID / "image-rootfs/system/lib64" if bits == 64 else ROOT / "build/nvk-android32/runtime"
    headers = {
        "cutils": ["aosp_system_core/libcutils/include", "aosp_system_core/libsystem/include", "aosp_system_core/libprocessgroup/include"],
        "hardware": ["aosp_libhardware/include", "aosp_system_core/libsystem/include", "aosp_frameworks_native/vulkan/include", "aosp_frameworks_native/libs/nativebase/include", "aosp_frameworks_native/libs/nativewindow/include", "aosp_frameworks_native/libs/arect/include"],
        "log": ["android_system_logging/liblog/include"],
        "sync": ["aosp_system_core/libsync/include"],
        "nativewindow": ["aosp_frameworks_native/libs/nativewindow/include", "aosp_frameworks_native/libs/nativebase/include", "aosp_frameworks_native/libs/arect/include"],
    }
    for name, dirs in headers.items():
        for directory in dirs:
            if not (ANDROID / directory).is_dir():
                raise SystemExit(f"Missing existing pinned platform headers: {ANDROID / directory}")
        (pkg / f"{name}.pc").write_text(f"Name: {name}\nDescription: Android13 platform {name}\nVersion: 33\nLibs: -L{runtime} -l{name}\nCflags: " + " ".join(f"-I{ANDROID / d}" for d in dirs) + "\n")
    # Mesa >= 26: NVK requires libelf (static, upstream Android CI recipe).
    android_libelf.install_pc(android_libelf.build(ROOT, NDK / "bin", triple, download), pkg)
    drm = ROOT / "build/libdrm"
    (pkg / "libdrm.pc").write_text(f"Name: libdrm\nDescription: Source-built Android libdrm\nVersion: 2.4.134\nLibs: -L{drm}/build-android33-{drm_cpu} -ldrm\nCflags: -I{drm}/drm -I{drm}/drm/include/drm\n")
    cross = ROOT / f"build/nvk-{target}.ini" if out is None else out / f"{target}.ini"
    cross.write_text(f"""[binaries]
c = {cc_launcher.cross_binary(f'{NDK}/bin/{triple}33-clang')}
cpp = {cc_launcher.cross_binary(f'{NDK}/bin/{triple}33-clang++')}
ar = '{NDK}/bin/llvm-ar'
strip = '{NDK}/bin/llvm-strip'
rust = ['rustc', '--target={triple}']
pkg-config = 'pkg-config'
[properties]
needs_exe_wrapper = true
pkg_config_libdir = '{pkg}'
[built-in options]
c_args = ['-fPIC']
cpp_args = ['-fPIC']
cpp_link_args = ['-static-libstdc++', '-Wl,-rpath-link,{runtime}']
[host_machine]
system = 'android'
cpu_family = '{cpu}'
cpu = '{cpu}'
endian = 'little'
""")
    return cross

# The NVK patch stack, in order.
NVK_PATCHES = (
    "build-compat.patch",
    "rm-integration.patch",
    # Generic NVK hardware behaviour (any backend): PASCAL_A clear bracket.
    "pascal-a-clear.patch",
    # NV12/P010 dma-buf import: 26.2 has upstream !40473 (multi-planar LINEAR
    # modifiers, plane offsets, multi-plane tiled shadows); these two
    # upstream main fixes are not in 26.2.x: ecfea8d9 (modifier list on a
    # YCbCr image picks LINEAR without nil) and bfb90f05 (MEMORY_PLANE_i ->
    # plane i), cherry-picked unchanged.
    "nvk-upstream-ecfea8d9-ycbcr-modifier.patch",
    "nvk-upstream-bfb90f05-memory-plane-aspects.patch",
    # Client plane layouts that cannot be bound fail vkCreateImage loudly
    # (upstream asserts).
    "nvk-explicit-plane-layouts.patch",
    # GLES through ANGLE keeps NaN-preserving FMin/FMax/FClamp (Mesa 26
    # folds max(NaN, 0) to NaN; GTA SA car paint black), driconf by engine.
    "nvk-angle-nan-preserve-min-max.patch",
    # Shader disk cache in an explicit per-application directory
    # (create_with_path): nvrm's program cache and, on Android, NVK's
    # pipeline cache in the application's code_cache; never redirected
    # into the EGL blob cache (keep-file-backend); self-healing after a
    # damaged item (evict-invalid).
    "mesa-disk-cache-create-with-path.patch",
    "mesa-disk-cache-keep-file-backend.patch",
    "mesa-disk-cache-evict-invalid-item.patch",
    "nvk-android-app-disk-cache.patch",
    # Shader programs published as the proprietary driver does: copy-engine
    # uploads into an unmapped heap, an upload count, and an instruction-
    # cache invalidate after the upload wait in the first submission after
    # the count moved; programs in whole 256 B prefetch blocks (Ampere B+).
    "nvk-shader-upload-publish.patch",
    "nvk-shader-prefetch-in-range.patch",
    # Fermi..Volta: block-linear DRM-modifier images get no host-visible
    # VRAM (BAR1) memory type, as the proprietary driver offers none; BAR1
    # does not return the modifier's byte layout there.
    "nvk-prevolta-modifier-memory-types.patch",
    # A 3D view of a 2D image (invalid usage NVIDIA's driver accepts) is
    # described as a depth-1 3D image instead of tripping nil's assert.
    "nvk-3d-view-of-2d-image.patch",
    # Colour-attachment output -> fragment reads drain with
    # PIXEL_SHADER_BARRIER instead of WAIT_FOR_IDLE, and QMDs from Turing
    # on carry CWD_MEMBAR_TYPE = L1_MEMBAR, as NVIDIA's driver does.
    "nvk-pixel-shader-barrier.patch",
    "nak-qmd-cwd-membar.patch",
    # Ampere B to Blackwell B (not GA100): compute -> compute barriers
    # order through QMD dependencies (an empty join QMD per barrier, host
    # dependency decrements) instead of WAIT_FOR_IDLE, as NVIDIA's driver
    # does.
    "nak-qmd-dependency.patch",
    "nvk-cs-join-qmd.patch",
    # A compute-only queue family whose queues run on their own
    # subcontext in graphics queue 0's channel group (AMPERE_COMPUTE_B+,
    # backends with shared contexts), as the proprietary driver exposes
    # and places its compute queues; its transfers and clears run as
    # compute work there.
    "nvk-compute-queue-family.patch",
    # Every compute context starts with the proprietary driver's compute
    # initialiser words where they are public (SPA/QMD version, SKED
    # cache sequence with IGNORE_VEID, CWD slot count on GB20x); SKED
    # cache invalidates then happen only in that WFI-ALL form.
    "nvk-stock-compute-init.patch",
    # QMD V05 (Blackwell): every dispatch carries the proprietary
    # driver's occupancy limits; without them a full-GPU grid and the
    # graphics work of another queue in the same channel group
    # time-slice instead of running together.
    "nak-qmd-occupancy.patch",
    # The same occupancy limits in the AMPERE_COMPUTE_B..Hopper QMD
    # layouts, CTA_LAUNCH_QUEUE / QMD_GROUP_ID by the recording queue
    # family, and the compute-only family's indirect dispatch sizing the
    # MME data FIFO through the compute class.
    "nak-qmd-launch-fields.patch",
    # Turing..Ada compute contexts start with the proprietary driver's
    # SCG setup (SET_SCG_CONTROL from the GR shader pipe counts, the SM
    # SCG control sequence) and every Turing+ one with its SCG compute
    # scheduling parameters: without them a compute-only queue's work
    # time-slices with graphics work instead of running beside it.
    "nvk-compute-scg-init.patch",
    # Conditional rendering on the compute-only queue: its channel loads
    # the conditional-rendering macro, sizes the macro data FIFO through
    # the compute class, and the condition sets the compute class's own
    # render enable.
    "nvk-compute-queue-cond-render.patch",
    # NAK loop codegen. Blackwell ALU instructions take no constant-buffer operands
    # (NAK opt_copy_prop.rs), so a bindless cbuf value used in divergent
    # control flow is read from the uniform register it was loaded into
    # instead of being re-loaded into a GPR in every block;
    "nak-blackwell-ldcx-uniform-reads.patch",
    # convergent loop-invariant ALU results and cbuf loads leave loops
    # entered in uniform control flow (uniform registers, no GPR cost),
    # as NVIDIA's compiler keeps them in uniform registers loaded once;
    "nak-licm-convergent.patch",
    # a divergent loop without a continue gets no reconvergence barrier
    # per iteration, only the one around the loop that its breaks sync
    # on, which is the shape NVIDIA's compiler emits for such loops.
    "nak-loop-continue-sync.patch",
    # Blackwell: texture fetches (TEX/TLD/TLD4) all signal one counting
    # scoreboard and their results are awaited with DEPBAR.LE, as
    # NVIDIA's compiler does, so more than five fetches can be in flight.
    "nak-blackwell-counted-tex-scoreboard.patch",
    # Blackwell: a uniform bindless handle reaches TEX/TLD/TLD4 as an
    # even uniform register pair (texture index, sampler index) built
    # once, instead of a GPR copy in every fetch's source vector, as
    # NVIDIA's compiler encodes it.
    "nak-blackwell-uniform-tex-handle.patch",
    # Volta/Turing: the SKED QMD cache is shared by the channel group's
    # subcontexts, so it is not invalidated per command buffer; instead
    # each command buffer counts its dispatches (QMD RELEASE0 reduction
    # ADD to a launch counter), waits for them at its end and drops
    # every QMD it allocated from the cache with a PCAS INVALIDATE, as
    # the proprietary driver does below AMPERE_A.
    "nvk-turing-sked-qmd-invalidate.patch",
    # Volta..Ampere A compute-only queues: their compute class has no
    # MME, so indirect dispatch, conditional rendering and compute
    # invocation statistics there do without NVK's macros (the indirect
    # buffer fetched as push-buffer data into the QMD, as the proprietary
    # driver does on its Turing and Volta compute queues), and the
    # compute-only family is offered from Volta on.
    "nvk-compute-queue-no-mme.patch",
    # vkCmdCopyMemoryIndirectKHR on a queue without a copy engine (the
    # compute-only family): one copy kernel per command that reads the
    # command and moves its bytes, as the proprietary driver records it,
    # instead of copy-engine methods the channel has no object for. Each
    # launch releases the command buffer's launch counter (QMD RELEASE0,
    # every QMD layout from Volta), kept in GPU-only device-memory blocks
    # (NVKMD_MEM_POOL_BLOCK); the barrier after the copy and the end of the
    # command buffer wait for that counter instead of idling the engine.
    "nvk-compute-queue-copy-memory-indirect.patch",
)

NVK_PATCH_LIST = [HERE / name for name in NVK_PATCHES]

def prepared_stamp(source, inputs):
    """Content hash of what a source preparation applies (patches, copied files): written into the worktree
    after --prepare / a full run, checked by --skip-prepare (concurrent target builds share one worktree)."""
    h = hashlib.sha256(PIN.encode())
    for path in sorted(Path(p) for p in inputs):
        for f in (sorted(x for x in path.rglob("*") if x.is_file()) if path.is_dir() else [path]):
            h.update(str(f.relative_to(HERE)).encode() + b"\0" + f.read_bytes())
    return Path(source) / ".nv-prepared", h.hexdigest()

def check_prepared(source, inputs, what):
    stamp, digest = prepared_stamp(source, inputs)
    if not stamp.is_file() or stamp.read_text().strip() != digest:
        raise SystemExit(f"{source} is not prepared for this checkout ({what}); run build.py --prepare-only first")

def mark_prepared(source, inputs):
    stamp, digest = prepared_stamp(source, inputs)
    stamp.write_text(digest + "\n")

NVK_PREP_INPUTS = NVK_PATCH_LIST + [HERE / n for n in ("nv_rm.h", "nvkmd_rm.h", "nvkmd_rm.c")]

def write_ndk_zlib_pc(pkgconfig_dir):
    """zlib.pc for the NDK sysroot zlib (an NDK stable library, libz.so in
    every image's /system/lib*); libzstd is not in the NDK."""
    ndk_zlib = (NDK / "sysroot/usr/include/zlib.h").read_text()
    zlib_version = next(line.split('"')[1].split("-")[0] for line in ndk_zlib.splitlines()
                        if line.startswith("#define ZLIB_VERSION "))
    (pkgconfig_dir / "zlib.pc").write_text(
        f"Name: zlib\nDescription: NDK sysroot zlib\nVersion: {zlib_version}\nLibs: -lz\nCflags:\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("host", "android", "android32", "both"), default="both")
    parser.add_argument("--rust-profile", choices=profiles.PROFILES, default="dev",
                        help="Rust Cargo profile (dev keeps the existing debug build)")
    parser.add_argument("--download-deps", action="store_true")
    parser.add_argument("-j", type=int, default=12)
    prep = parser.add_mutually_exclusive_group()
    prep.add_argument("--prepare-only", action="store_true",
                      help="apply the patch stack to the Mesa worktree and stop")
    prep.add_argument("--skip-prepare", action="store_true",
                      help="build a target from a worktree an earlier --prepare-only/full run prepared for this "
                           "checkout (lets several targets build concurrently from one worktree)")
    args = parser.parse_args()
    sources_before = profiles.source_snapshot(ROOT)
    for tool in ("bindgen", "cbindgen"):
        if not (TOOLS / tool).exists():
            parser.error(f"Missing {tool}; run mesa/compiler/build.py --tools-only --download-deps first")
    if not SOURCE.exists():
        run(["git", "-C", ROOT / "build/mesa-git", "worktree", "add", "--detach", SOURCE, PIN])
        run(["git", "-C", SOURCE, "sparse-checkout", "disable"])
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        parser.error(f"Mesa revision mismatch: {actual}")
    if args.skip_prepare:
        check_prepared(SOURCE, NVK_PREP_INPUTS, "NVK")
    else:
        prepare_nvk_source(args.download_deps)
        if args.prepare_only:
            print(f"Prepared {SOURCE}")
            return
    rm_source = SOURCE / "src/nouveau/vulkan/nvkmd/rm"
    host_clc = ROOT / "build/nvk-host/src/compiler/clc"
    host_vtn = ROOT / "build/nvk-host/src/compiler/spirv"
    env = dict(os.environ)
    env["PATH"] = ":".join(map(str, (host_clc, host_vtn, TOOLS))) + ":" + env["PATH"]
    env["PKG_CONFIG_PATH"] = f"{DEPS}/root/usr/lib/pkgconfig:{DEPS}/root/usr/share/pkgconfig"
    env["LD_LIBRARY_PATH"] = f"{DEPS}/root/usr/lib"
    cc_launcher.host_env(env)
    common = ["-Dvulkan-drivers=nouveau", "-Dgallium-drivers=", "-Dglx=disabled", "-Degl=disabled", "-Dgbm=disabled", "-Dopengl=false", "-Dgles1=disabled", "-Dgles2=disabled", "-Dbuild-tests=false", "-Dvideo-codecs=", "-Dvalgrind=disabled", "-Dlibunwind=disabled", "-Dbuildtype=release"]
    # Android consumes source-built host mesa_clc and vtn_bindgen2.
    targets = ["host", "android"] if args.target == "both" else [args.target]
    if args.target.startswith("android") and not (host_clc / "mesa_clc").exists():
        targets.insert(0, "host")
    for target in targets:
        build = ROOT / f"build/nvk-{target}"
        options = list(common)
        target_env = dict(env)
        rust_build = ROOT / "build/nv-rm-rust"
        rust_command = ["cargo", "build", "--offline", "--no-default-features", "--target-dir", rust_build,
                        "--manifest-path", ROOT / "rm/Cargo.toml"]
        if args.rust_profile == "release":
            rust_command.append("--release")
        rust_lib = profiles.rust_library(ROOT, target, args.rust_profile).parent
        bits = 32 if target == "android32" else 64
        triple = "i686-linux-android" if bits == 32 else "x86_64-linux-android"
        runtime = ROOT / "build/nvk-android32/runtime" if bits == 32 else ANDROID / "image-rootfs/system/lib64"
        if target.startswith("android"):
            rust_command += ["--target", triple]
            target_env["CARGO_TARGET_" + triple.upper().replace("-", "_") + "_LINKER"] = str(NDK / f"bin/{triple}33-clang")
        run(rust_command, env=target_env)
        linker_args = [f"-L{rust_lib}", f"-Wl,-rpath,{rust_lib}"]
        options += ["-Dc_link_args=" + json.dumps(linker_args)]
        if target == "host":
            options += ["-Dplatforms=", "-Dinstall-mesa-clc=true", "-Dcpp_link_args=" + json.dumps(linker_args)]
        else:
            # Persistent pipeline cache (nvk-android-app-disk-cache.patch) with
            # the NDK's zlib as compressor.
            cross = android_cross_file(bits, download=args.download_deps)
            write_ndk_zlib_pc(ROOT / f"build/nvk-{target}-pkgconfig")
            options += ["--cross-file", cross, "-Dplatforms=android", "-Dplatform-sdk-version=33", "-Dandroid-libbacktrace=disabled", "-Dllvm=disabled", "-Dmesa-clc=system", "-Dexpat=disabled", "-Dzstd=disabled", "-Dzlib=enabled", "-Dshader-cache=enabled"]
            options += ["-Dcpp_link_args=" + json.dumps(linker_args + ["-static-libstdc++", f"-Wl,-rpath-link,{runtime}"])]
            target_env.pop("PKG_CONFIG_PATH", None)
            target_env["BINDGEN_EXTRA_CLANG_ARGS"] = f"--sysroot={NDK}/sysroot -isystem {NDK}/sysroot/usr/include/{triple} -D__ANDROID_API__=33"
        setup = cc_launcher.setup_command(build, SOURCE)
        setup += options + ["--wrap-mode=" + ("default" if args.download_deps else "nodownload")]
        setup_run(setup, args, env=target_env)
        cc_launcher.record(build)
        artifacts = ["src/nouveau/vulkan/libvulkan_nouveau.so"]
        if target == "host":
            artifacts += ["src/compiler/clc/mesa_clc", "src/compiler/spirv/vtn_bindgen2"]
        run(["ninja", "-C", build, *artifacts, "-j", args.j, *ninja_load()], env=target_env)
        compiler = "cc" if target == "host" else NDK / f"bin/{triple}33-clang"
        run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", HERE / "inspect.c", "-ldl", "-o", build / "inspect"])
        run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", HERE / "abi-check.c", "-o", build / "abi-check"])
        for name in ("nv_rm.h", "nvkmd_rm.h", "nvkmd_rm.c"):
            if hashlib.sha256((rm_source / name).read_bytes()).digest() != hashlib.sha256((HERE / name).read_bytes()).digest():
                raise SystemExit(f"Compiled Mesa bridge copy differs from canonical {name}; rebuild")
        profiles.write_metadata(ROOT, target, args.rust_profile, rust_command,
                                subprocess.check_output(["rustc", "--version"], text=True).strip(),
                                {key: target_env[key] for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")
                                 if key in target_env}, sources_before)
        print(f"Built full {target} NVK (Rust {args.rust_profile}): {build}/src/nouveau/vulkan/libvulkan_nouveau.so")

def prepare_nvk_source(download):
    stamp = SOURCE / ".nv-prepared"
    if stamp.exists():
        stamp.unlink()
    patches = NVK_PATCH_LIST
    # Patches touching the same file only reverse-check on top of the stack:
    # unwind the applied ones newest first, then apply the whole stack in order.
    for patch in reversed(patches):
        if subprocess.run(["git", "-C", str(SOURCE), "apply", "--reverse", "--check", str(patch)],
                          capture_output=True).returncode == 0:
            run(["git", "-C", SOURCE, "apply", "--reverse", patch])
    for patch in patches:
        apply(patch)
    rm_source = SOURCE / "src/nouveau/vulkan/nvkmd/rm"
    rm_source.mkdir(parents=True, exist_ok=True)
    for name in ("nv_rm.h", "nvkmd_rm.h", "nvkmd_rm.c"):
        shutil.copy2(HERE / name, rm_source / name)
    cache = SOURCE / "subprojects/packagecache"
    cache.mkdir(exist_ok=True)
    for archive in (ROOT / "build/mesa-nak-source/subprojects/packagecache").glob("*.tar.gz"):
        if not (cache / archive.name).exists():
            shutil.copy2(archive, cache / archive.name)
    host_dependencies(download)
    mark_prepared(SOURCE, NVK_PREP_INPUTS)


if __name__ == "__main__":
    main()
