#!/usr/bin/env bash
# android/provision.sh [stage...] — fetch the Android build inputs (NDK, AOSP header clones, link stubs
# from the target image) from the pins in android/pins.env.
#
#   stages: ndk clones imagelibs   (default: all, in that order)
#
# Work tree: $ANDROID_WORK (default <repo>/build/android-work, gitignored):
#   ndk/                     the pinned NDK (r29)
#   deps/                    AOSP header clones, image link stubs (deps/image-rootfs: unmodified libraries
#                            read from the pinned system/vendor images; linked against, never shipped)
# Every download is sha256-checked against pins.env; a missing or wrong pin fails loudly.
# Each stage is idempotent (a .provisioned stamp holds the pin it was made from).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
# shellcheck source=pins.env
source "$HERE/pins.env"
WORK=${ANDROID_WORK:-$REPO/build/android-work}
D=$WORK/deps
NDK=$WORK/ndk/android-ndk-$NDK_VERSION
API=30
TARGET=x86_64-linux-android
mkdir -p "$D/dl"

die() { echo "provision: $*" >&2; exit 1; }
STAGES=("$@"); [ $# -gt 0 ] || STAGES=(ndk clones imagelibs)
has_stage() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# fetch URL SHA256 OUT [ALT_URL...]: download once (tools/fetch.py: alternative sources, retries with backoff,
# stalled or trickling sources given up, kept only when the hash matches), verify always
fetch() {
    local url=$1 sha=$2 out=$3; shift 3
    [ -n "$sha" ] || die "no sha256 pin for $url in pins.env"
    python3 "$REPO/tools/fetch.py" --out "$out" --sha256 "$sha" "$url" "$@" || die "download failed: $url $*"
    echo "$sha  $out" | sha256sum -c --quiet - || die "sha256 mismatch for $out (pin $sha)"
}
net() { python3 "$REPO/tools/fetch.py" retry -- "$@"; }   # a network command, retried with backoff
# git_pin DIR URL SHA: shallow fetch of exactly one commit
git_pin() {
    local dir=$1 url=$2 sha=$3
    if [ -e "$dir/.provisioned" ] && [ "$(cat "$dir/.provisioned")" = "$sha" ]; then return 0; fi
    rm -rf "$dir"; git init -q "$dir"
    net git -C "$dir" fetch -q --depth 1 "$url" "$sha"
    git -C "$dir" checkout -q FETCH_HEAD
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "$dir: HEAD is not $sha"
    echo "$sha" > "$dir/.provisioned"
    echo "  $(basename "$dir"): $sha"
}

if has_stage ndk; then
    echo "== stage ndk ($NDK_VERSION)"
    if [ ! -x "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" ]; then
        fetch "$NDK_URL" "$NDK_ZIP_SHA256" "$WORK/dl-ndk-$NDK_VERSION.zip"
        mkdir -p "$WORK/ndk"; unzip -q "$WORK/dl-ndk-$NDK_VERSION.zip" -d "$WORK/ndk"
    fi
    grep -q "Pkg.ReleaseName = $NDK_VERSION" "$NDK/source.properties" || die "$NDK is not $NDK_VERSION"
fi
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64

if has_stage clones; then
    echo "== stage clones"
    git_pin "$D/aosp_fmq"                "$AOSP_BASE/system/libfmq"            "$AOSP_FMQ_SHA"
    git_pin "$D/aosp_fmtlib"             "$AOSP_BASE/external/fmtlib"          "$AOSP_FMTLIB_SHA"
    git_pin "$D/aosp_frameworks_native"  "$AOSP_BASE/frameworks/native"        "$AOSP_FRAMEWORKS_NATIVE_SHA"
    git_pin "$D/aosp_hardware_interfaces" "$AOSP_BASE/hardware/interfaces"     "$AOSP_HARDWARE_INTERFACES_SHA"
    git_pin "$D/aosp_hidl"               "$AOSP_BASE/system/tools/hidl"        "$AOSP_HIDL_SHA"
    git_pin "$D/aosp_libbase"            "$AOSP_BASE/system/libbase"           "$AOSP_LIBBASE_SHA"
    git_pin "$D/aosp_libhardware"        "$AOSP_BASE/hardware/libhardware"     "$AOSP_LIBHARDWARE_SHA"
    git_pin "$D/aosp_libhidl"            "$AOSP_BASE/system/libhidl"           "$AOSP_LIBHIDL_SHA"
    git_pin "$D/aosp_libhwbinder"        "$AOSP_BASE/system/libhwbinder"       "$AOSP_LIBHWBINDER_SHA"
    git_pin "$D/aosp_system_core"        "$AOSP_BASE/system/core"              "$AOSP_SYSTEM_CORE_SHA"
    git_pin "$D/android_system_logging"  "$SYSTEM_LOGGING_URL"                 "$SYSTEM_LOGGING_SHA"
fi

if has_stage imagelibs; then
    echo "== stage imagelibs"
    R=$D/image-rootfs
    if [ ! -e "$R/.provisioned" ]; then
        IMGDIR=${ANDROID_IMAGES:-}
        if [ -z "$IMGDIR" ] && [ -r /var/lib/waydroid/waydroid.cfg ]; then
            IMGDIR=$(sed -n 's/^images_path *= *//p' /var/lib/waydroid/waydroid.cfg | head -1)
        fi
        if [ -n "$IMGDIR" ] && [ -r "$IMGDIR/system.img" ] && [ -r "$IMGDIR/vendor.img" ]; then
            SYSIMG=$IMGDIR/system.img VENIMG=$IMGDIR/vendor.img
            echo "  link stubs from the local images in $IMGDIR"
        else
            for part in system vendor; do
                u=IMG_${part^^}_URL s=IMG_${part^^}_SHA256 a=IMG_${part^^}_URL_ALT
                fetch "${!u}" "${!s}" "$D/dl/$part.zip" "${!a}"
                rm -rf "$D/dl/$part-img"; unzip -oq "$D/dl/$part.zip" -d "$D/dl/$part-img"
            done
            SYSIMG=$(ls "$D"/dl/system-img/*.img) VENIMG=$(ls "$D"/dl/vendor-img/*.img)
            echo "  link stubs from the pinned OTA images"
        fi
        rm -rf "$R"; mkdir -p "$R/system" "$R/apex/com.android.vndk.v33" "$R/vendor"
        debugfs -R "rdump /system/lib64 $R/system" "$SYSIMG" 2>&1 | grep -v -e '^debugfs' -e 'Operation not permitted while changing ownership' || true
        debugfs -R "rdump /system/apex/com.android.vndk.v33/lib64 $R/apex/com.android.vndk.v33" "$SYSIMG" 2>&1 | grep -v -e '^debugfs' -e 'Operation not permitted while changing ownership' || true
        debugfs -R "rdump /lib64 $R/vendor" "$VENIMG" 2>&1 | grep -v -e '^debugfs' -e 'Operation not permitted while changing ownership' || true
        for lib in "$R/system/lib64/libhidlbase.so"; do
            [ -f "$lib" ] || die "expected $lib missing after extraction"
        done
        touch "$R/.provisioned"
    fi
    echo "  $(find "$R" -name '*.so' | wc -l) shared objects"
fi
echo "== provisioned: $D"
