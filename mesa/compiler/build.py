#!/usr/bin/env python3
"""Build the source compiler bridge using pinned Mesa and local NDK r29."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "ddaf85e353442d9d7f192fdcf1823e652e3f2ee3"  # tag mesa-26.2.5 (2026-10-14)
SOURCE = ROOT / "build/mesa-nak-source"
TOOLS = ROOT / "build/nak-tools"
NDK = ROOT / "build/android-work/ndk/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64"
sys.path.insert(0, str(HERE.parent))
import android_libelf  # Mesa >= 26: NVK (nouveau vulkan) requires libelf
sys.path.insert(0, str(ROOT / "tools"))
from fetch import retry_run  # noqa: E402  (network commands: retries with backoff)
import cc_launcher  # noqa: E402  (NATIVE_CC_LAUNCHER: ccache in front of the C/C++ compilers)


def ninja_load():
    """`-l N` when NINJA_LOAD is set (packaging/native/build-native.sh builds targets concurrently: each ninja
    keeps -j but stops starting jobs while the load average is above N)."""
    return ["-l", os.environ["NINJA_LOAD"]] if os.environ.get("NINJA_LOAD") else []

def run(args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("host", "android", "both"), default="both")
    parser.add_argument("--download-deps", action="store_true")
    parser.add_argument("-j", type=int, default=12)
    parser.add_argument("--tools-only", action="store_true",
                        help="install the pinned bindgen/cbindgen (build/nak-tools) and stop: all the NVK builds "
                             "need from here, so they can start while NAK builds")
    args = parser.parse_args()
    env = dict(os.environ, PATH=str(TOOLS / "bin") + ":" + os.environ["PATH"])
    for tool, package, version in [("bindgen", "bindgen-cli", "0.72.1"), ("cbindgen", "cbindgen", "0.29.2")]:
        if not (TOOLS / "bin" / tool).exists():
            if not args.download_deps:
                parser.error(f"Missing {tool}; rerun with --download-deps to build its pinned version locally")
            retry_run(["cargo", "install", package, "--version", version, "--locked", "--root", TOOLS])
    if args.tools_only:
        return
    if not SOURCE.exists():
        run(["git", "-C", ROOT / "build/mesa-git", "worktree", "add", "--detach", SOURCE, PIN])
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        parser.error(f"Wrong Mesa revision {actual}; expected {PIN}")
    patch = HERE / "mesa-compiler-only.patch"
    already = subprocess.run(["git", "-C", str(SOURCE), "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if already.returncode:
        run(["git", "-C", SOURCE, "apply", "--check", patch])
        run(["git", "-C", SOURCE, "apply", patch])
    bridge = SOURCE / "src/nouveau/native-bridge"
    bridge.mkdir(exist_ok=True)
    for file in ("nv_nak.h", "nv_nak.c", "check.c", "meson.build"):
        shutil.copy2(HERE / file, bridge / file)

    common = ["-Dvulkan-drivers=nouveau", "-Dgallium-drivers=", "-Dplatforms=", "-Dglx=disabled",
              "-Degl=disabled", "-Dgbm=disabled", "-Dopengl=false", "-Dgles1=disabled", "-Dgles2=disabled",
              "-Dllvm=disabled", "-Dbuild-tests=false", "-Dvideo-codecs=", "-Dvalgrind=disabled",
              "-Dlibunwind=disabled", "-Dbuildtype=release"]
    targets = ("host", "android") if args.target == "both" else (args.target,)
    for target in targets:
        build = ROOT / f"build/nak-{target}"
        options = list(common)
        target_env = dict(env)
        cc_launcher.host_env(target_env)
        if target == "android":
            pkg = ROOT / "build/nak-android-pkgconfig"
            pkg.mkdir(exist_ok=True)
            android_libelf.install_pc(android_libelf.build(ROOT, NDK / "bin", "x86_64-linux-android",
                                                           args.download_deps), pkg)
            drm = ROOT / "build/libdrm"
            (pkg / "libdrm.pc").write_text(f"Name: libdrm\nDescription: source-built Android libdrm\nVersion: 2.4.134\nLibs: -L{drm}/build-android33-x86_64 -ldrm\nCflags: -I{drm}/drm -I{drm}/drm/include/drm\n")
            cross = ROOT / "build/nak-android.ini"
            cross.write_text(f"""[binaries]
c = {cc_launcher.cross_binary(f'{NDK}/bin/x86_64-linux-android33-clang')}
cpp = {cc_launcher.cross_binary(f'{NDK}/bin/x86_64-linux-android33-clang++')}
ar = '{NDK}/bin/llvm-ar'
strip = '{NDK}/bin/llvm-strip'
rust = ['rustc', '--target=x86_64-linux-android']
pkg-config = 'pkg-config'
[properties]
needs_exe_wrapper = true
pkg_config_libdir = '{pkg}'
[built-in options]
c_args = ['-fPIC']
cpp_args = ['-fPIC']
cpp_link_args = ['-static-libstdc++']
[host_machine]
system = 'android'
cpu_family = 'x86_64'
cpu = 'x86_64'
endian = 'little'
""")
            options += ["--cross-file", cross, "-Dexpat=disabled", "-Dzstd=disabled", "-Dzlib=disabled", "-Dshader-cache=disabled"]
            target_env["BINDGEN_EXTRA_CLANG_ARGS"] = f"--sysroot={NDK}/sysroot -isystem {NDK}/sysroot/usr/include/x86_64-linux-android -D__ANDROID_API__=33"
        setup = cc_launcher.setup_command(build, SOURCE)
        setup += options + ["--wrap-mode=" + ("default" if args.download_deps else "nodownload")]
        if args.download_deps:      # setup may fetch wraps (NAK's crates)
            retry_run(setup, env=target_env)
        else:
            run(setup, env=target_env)
        cc_launcher.record(build)
        run(["ninja", "-C", build, "src/nouveau/native-bridge/libnv_native_nak.so", "src/nouveau/native-bridge/nv-nak-check", "-j", args.j, *ninja_load()], env=target_env)
        print(f"Built {target}: {build}/src/nouveau/native-bridge/libnv_native_nak.so")

if __name__ == "__main__":
    main()
