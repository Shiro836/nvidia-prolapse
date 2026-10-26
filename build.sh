#!/usr/bin/env bash
# build.sh — build every driver component for the host and Android (x86_64 + i686), in dependency order.
# All inputs are pinned in the component recipes (android/pins.env, mesa/build.py PIN, */Makefile,
# media/vndk/build.sh). Outputs go to build/ and the components' out/ directories. Network is needed on
# the first run (pinned downloads, checksum-verified by each recipe). Rust profile: release.
#
#   ./build.sh                  everything
#   ANDROID_IMAGES=<dir>        system.img/vendor.img to read link stubs from (default: android/provision.sh's choice)
#   JOBS=<n>                    ninja -j (default: CPU count)
#
# Host requirements (not provisioned; install them from the distribution):
#   programs   git python3 make cc c++ meson ninja pkg-config rustup (cargo; rust-toolchain.toml selects
#              the pinned toolchain and Android targets) debugfs (e2fsprogs) readelf nm (binutils) unzip
#              bison flex glslangValidator llvm-config
#   libraries  for the host NVK build: LLVM + clang (libclang-cpp), SPIRV-LLVM-Translator, SPIRV-Tools,
#              libdrm, libelf, zlib, zstd, expat, libudev, libdisplay-info (meson names any
#              missing one)
#   python     mako, packaging, PyYAML (Mesa's meson checks them)
#   tests      spirv-val (tests/vulkan/build.py), strace (tests/vulkan/capture.py)
set -euo pipefail
REPO=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
cd "$REPO"
PY=${PYTHON:-python3}
JOBS=${JOBS:-$(nproc)}
say() { printf '\n== %s\n' "$*"; }
die() { echo "build: $*" >&2; exit 1; }
for tool in git "$PY" make cc c++ meson ninja pkg-config rustup cargo debugfs readelf nm unzip bison flex \
        glslangValidator llvm-config; do
    command -v "$tool" >/dev/null || die "host program $tool is missing (see the host requirements at the top of build.sh)"
done

say "Android inputs (NDK r29, AOSP headers, image link stubs)"
android/provision.sh

say "Mesa object store build/mesa-git (worktrees are taken from it at the pins)"
MESA_URL=https://gitlab.freedesktop.org/mesa/mesa.git
NVK_PIN=$(sed -n 's/^PIN = "\([0-9a-f]\{40\}\)".*/\1/p' mesa/build.py)
HAL_PIN=$(sed -n 's/^MESA_SHA := \([0-9a-f]\{40\}\)$/\1/p' hal/Makefile)
GRALLOC_PIN=$(sed -n 's/^MESA_REV *:= \([0-9a-f]\{40\}\)$/\1/p' gralloc/Makefile)
[ -n "$NVK_PIN" ] && [ -n "$HAL_PIN" ] && [ -n "$GRALLOC_PIN" ] || die "could not read the Mesa pins"
[ -d build/mesa-git/.git ] || git init -q build/mesa-git
for pin in "$NVK_PIN" "$HAL_PIN" "$GRALLOC_PIN"; do
    git -C build/mesa-git cat-file -e "$pin^{commit}" 2>/dev/null \
        || "$PY" tools/fetch.py retry -- git -C build/mesa-git fetch -q "$MESA_URL" "$pin"
done

SYSIMG=$(ls build/android-work/deps/dl/system-img/*.img 2>/dev/null || true)
VENIMG=$(ls build/android-work/deps/dl/vendor-img/*.img 2>/dev/null || true)
if [ -z "$SYSIMG" ] || [ -z "$VENIMG" ]; then
    IMGDIR=${ANDROID_IMAGES:-$(sed -n 's/^images_path *= *//p' /var/lib/waydroid/waydroid.cfg 2>/dev/null | head -1)}
    SYSIMG=${IMGDIR:+$IMGDIR/system.img} VENIMG=${IMGDIR:+$IMGDIR/vendor.img}
fi
[ -n "$SYSIMG" ] && [ -r "$SYSIMG" ] && [ -r "$VENIMG" ] || die "no system/vendor image (android/provision.sh imagelibs, or ANDROID_IMAGES)"
SYSIMG=$(realpath -- "$SYSIMG") VENIMG=$(realpath -- "$VENIMG")
export WAYDROID_SYSTEM_IMG=$SYSIMG WAYDROID_VENDOR_IMG=$VENIMG    # media/vndk/build.sh reads these

say "bionic libdrm (both ABIs)"
make -C libdrm
make -C libdrm ARCH=x86

say "Android runtimes for the host tests (read-only from the image): i686 closure, x86_64 bionic"
"$PY" mesa/prepare-runtime32.py --image "$SYSIMG"
"$PY" mesa/prepare-runtime64.py --image "$SYSIMG"

say "NVK + the Rust RM backend (host, x86_64, i686)"
"$PY" mesa/compiler/build.py --tools-only --download-deps
"$PY" mesa/build.py --target host --download-deps -j "$JOBS" --rust-profile release
"$PY" mesa/build.py --target android --download-deps --skip-prepare -j "$JOBS" --rust-profile release
"$PY" mesa/build.py --target android32 --download-deps --skip-prepare -j "$JOBS" --rust-profile release

say "Vulkan HAL over NVK (+ its Android test programs)"
"$PY" hal/build-nvk.py
"$PY" hal/build-nvk-tests.py

say "gralloc backend over NVK + minigbm"
"$PY" gralloc/build-nvk.py --target all
make -C gralloc minigbm-fetch
make -C gralloc minigbm SYSTEM_IMG="$SYSIMG" VENDOR_IMG="$VENIMG"

say "Codec2 NVDEC decoder (vendor libraries, host unit tests, service)"
media/vndk/build.sh
make -C media test
NV_VIDEO_LIB_DIR=$REPO/build/nv-rm-rust/x86_64-linux-android/release media/vndk/build.sh service

say "standalone NAK bridge (host + x86_64)"
"$PY" mesa/compiler/build.py --target both --download-deps -j "$JOBS"
echo "build: done"
