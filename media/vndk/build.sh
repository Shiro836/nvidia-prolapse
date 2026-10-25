#!/usr/bin/env bash
# media/vndk/build.sh — build the vendor-variant Codec2 runtime libraries that the
# LineageOS 20 (Android 13) x86_64 Waydroid image lacks in its VNDK apex and vendor.img:
#   android.hardware.media.c2@1.0/1.1/1.2.so (hidl-gen c++-sources), libstagefright_bufferpool@2.0.1.so,
#   libcodec2_vndk.so, libcodec2_hidl_plugin.so (vendor-only dep of the next three, AOSP target.vendor),
#   libcodec2_hidl@1.0/1.1/1.2.so (server side).
# Standalone with the Android toolchain (NDK r29 + platform-libc++ std::__1 override, hidl-gen,
# pinned AOSP header clones): run android/provision.sh first. Every source is pinned by SHA below.
# Links with -Wl,-z,defs against .so files dumped read-only from the local Waydroid images
# (VNDK apex com.android.vndk.current/lib64, /system/lib64 for LLNDK); those are never shipped.
#
# Usage: media/vndk/build.sh [stage...]   stages: fetch hidl build verify (default all),
#        service (explicit; NV_VIDEO_LIB_DIR=<dir of the x86_64 Android libnv_bionic_native.so>)
# Output: <repo>/build/media-c2/out/lib64/*.so (unstripped) + out/VERIFY.txt
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
# shellcheck source=../../android/pins.env
source "$REPO/android/pins.env"
HD=$REPO/build/android-work/deps
W=$REPO/build/media-c2
OUT=$W/out/lib64
BIN=$REPO/build/android-work/ndk/android-ndk-$NDK_VERSION/toolchains/llvm/prebuilt/linux-x86_64/bin
# API 33 = the image's platform (Android 13). libc/libm/libdl resolve against the NDK API-33 stubs (the
# image's /system/lib64/libc.so is a symlink into the runtime apex); everything else against image copies.
CXX=$BIN/x86_64-linux-android33-clang++
# compiler launcher for the compile steps (packaging/native/build-native.sh: ccache; empty = none, tools/cc_launcher.py)
read -r -a CCL <<< "${NATIVE_CC_LAUNCHER:-}"
READELF=$BIN/llvm-readelf
# hidl-gen built from system/tools/hidl + Waydroid's patch (stage fetch), not a stock one: see WD_*.
HG=$W/deps/hidl-gen-build/hidl-gen
die() { echo "media/vndk/build.sh: $*" >&2; exit 1; }

# ---- pins ----
AV_URL=https://android.googlesource.com/platform/frameworks/av
AV_SHA=95be9bad234d69f4a8ded5ee72b60315b1353098            # tm-qpr3
AV_DIRS=(media/bufferpool media/codec2/core media/codec2/vndk media/codec2/hidl
         media/codec2/components/base media/libstagefright/xmlparser
         media/libstagefright/foundation/include media/libstagefright/bqhelper/include
         media/libstagefright/include/media/stagefright)
LIBION_URL=https://android.googlesource.com/platform/system/memory/libion
LIBION_SHA=fe13bd22eec8204d6102c45fcb1c1dce667585a2        # android-13.0.0_r79
LIBDMABUFHEAP_URL=https://android.googlesource.com/platform/system/memory/libdmabufheap
LIBDMABUFHEAP_SHA=666e7028fde4814cdeb581b8ddeeb4ec185b9cd2 # android-13.0.0_r79
# The image's libhidlbase is NOT stock AOSP: Waydroid patches system/libhwbinder, system/libhidl and hidl-gen
# (system/tools/hidl) for host_hwbinder. That changes the C++ ABI every HIDL interface library compiles against:
# IBinder gains `virtual bool isHostHwBinder()` after remoteBinder() (BHwBinder::onTransact moves from vtable
# slot 11 to 12), Parcel gains `mutable bool mIsHost` (sizeof grows; generated proxies call Parcel(bool)), and
# generated proxies call IInterface::isHostHwBinder / ProcessState::self(bool). Code generated and compiled
# against stock headers runs with a short vtable and undersized Parcels against the image's libhidlbase.
# Pin: lineage-20 at the image's build date (2026-04-03); the three patch sets are unchanged since bc829a2.
WD_VENDOR_URL=https://github.com/waydroid/android_vendor_waydroid
WD_VENDOR_SHA=dd9c63616ca0076dd5b3b487104459e5bec20a5f    # lineage-20, 2026-04-02
WD_PATCHES=waydroid-patches/base-patches-33/system
WD_TREES=("aosp_libhwbinder libhwbinder $AOSP_LIBHWBINDER_SHA" "aosp_libhidl libhidl $AOSP_LIBHIDL_SHA"
          "aosp_hidl tools/hidl $AOSP_HIDL_SHA")
SYSIMG=${WAYDROID_SYSTEM_IMG:-/var/lib/waydroid/images/system.img}

STAGES=("$@"); [ $# -gt 0 ] || STAGES=(fetch hidl build verify)
has_stage() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

net() { python3 "$REPO/tools/fetch.py" retry -- "$@"; }   # a network command, retried with backoff
git_pin() {  # DIR URL SHA [sparse dirs...]
    local dir=$1 url=$2 sha=$3; shift 3
    local want="$sha${*:+ $*}"
    if [ -e "$dir/.provisioned" ] && [ "$(cat "$dir/.provisioned")" = "$want" ]; then return 0; fi
    if [ $# -gt 0 ] && [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
        git -C "$dir" sparse-checkout set "$@"            # existing sparse clone at the pin: widen it
    else
        rm -rf "$dir"; git init -q "$dir"
        if [ $# -gt 0 ]; then
            git -C "$dir" sparse-checkout set "$@"
            net git -C "$dir" fetch -q --depth 1 --filter=blob:none "$url" "$sha"
        else
            net git -C "$dir" fetch -q --depth 1 "$url" "$sha"
        fi
        git -C "$dir" checkout -q FETCH_HEAD
    fi
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "$dir: HEAD is not $sha"
    echo "$want" > "$dir/.provisioned"
    echo "  $(basename "$dir"): $sha"
}

[ -x "$CXX" ] || die "NDK $NDK_VERSION missing: run android/provision.sh"
for d in aosp_libhidl aosp_libhwbinder aosp_fmq aosp_system_core aosp_libbase android_system_logging \
         aosp_frameworks_native aosp_hardware_interfaces aosp_libhardware aosp_hidl aosp_fmtlib; do
    [ -e "$HD/$d/.provisioned" ] || die "$HD/$d missing: run android/provision.sh clones"
done
[ "$(cat "$HD/aosp_hardware_interfaces/.provisioned")" = "$AOSP_HARDWARE_INTERFACES_SHA" ] \
    || die "aosp_hardware_interfaces is not at $AOSP_HARDWARE_INTERFACES_SHA"

if has_stage fetch; then
    echo "== stage fetch"
    mkdir -p "$W/deps"
    git_pin "$W/av" "$AV_URL" "$AV_SHA" "${AV_DIRS[@]}"
    git_pin "$W/deps/libion" "$LIBION_URL" "$LIBION_SHA"
    git_pin "$W/deps/libdmabufheap" "$LIBDMABUFHEAP_URL" "$LIBDMABUFHEAP_SHA"
    git_pin "$W/deps/vendor_waydroid" "$WD_VENDOR_URL" "$WD_VENDOR_SHA" "$WD_PATCHES"
    # $W/deps/wd: the pinned AOSP clones (android/provision.sh) of libhwbinder/libhidl/hidl + Waydroid's patches (the sources
    # the image's libhidlbase and its hidl-gen were built from), plus links to the unpatched trees hidl-gen's
    # CMakeLists expects next to them (its WNV layout).
    WD=$W/deps/wd
    want="$WD_VENDOR_SHA"
    for t in "${WD_TREES[@]}"; do
        read -r tree sub sha <<< "$t"
        [ "$(cat "$HD/$tree/.provisioned")" = "$sha" ] || die "$HD/$tree is not at $sha"
        want+=" $tree=$sha:$(cat "$W/deps/vendor_waydroid/$WD_PATCHES/$sub"/*.patch | sha256sum | cut -c1-16)"
    done
    if [ ! -e "$WD/.provisioned" ] || [ "$(cat "$WD/.provisioned")" != "$want" ]; then
        rm -rf "$WD" "$W/deps/hidl-gen-build"; mkdir -p "$WD"
        for t in "${WD_TREES[@]}"; do
            read -r tree sub sha <<< "$t"
            mkdir -p "$WD/$tree"; git -C "$HD/$tree" archive "$sha" | tar -x -C "$WD/$tree"
            git -C "$WD/$tree" init -q && git -C "$WD/$tree" add -A
            git -C "$WD/$tree" -c user.name=c2 -c user.email=c2@build commit -qm "base $sha"
            for pf in "$W/deps/vendor_waydroid/$WD_PATCHES/$sub"/*.patch; do
                git -C "$WD/$tree" apply --whitespace=nowarn "$pf" || die "waydroid patch does not apply: $pf"
                git -C "$WD/$tree" add -A
                git -C "$WD/$tree" diff --cached --quiet && die "waydroid patch changed nothing: $pf"
                git -C "$WD/$tree" -c user.name=c2 -c user.email=c2@build commit -qm "$(basename "$pf")"
            done
            echo "  $tree: $sha + $(ls "$W/deps/vendor_waydroid/$WD_PATCHES/$sub"/*.patch | wc -l) waydroid patches"
        done
        for l in aosp_system_core aosp_libbase android_system_logging aosp_fmtlib; do ln -s "$HD/$l" "$WD/$l"; done
        grep -q 'virtual bool *isHostHwBinder' "$WD/aosp_libhwbinder/include/hwbinder/IBinder.h" \
            || die "patched IBinder.h lacks isHostHwBinder()"
        echo "$want" > "$WD/.provisioned"
    fi
    if [ ! -x "$HG" ]; then
        cmake -S "$REPO/android/hidl-gen" -B "$W/deps/hidl-gen-build" -DCMAKE_BUILD_TYPE=Release -DWNV="$WD" >/dev/null
        cmake --build "$W/deps/hidl-gen-build" -j"$(nproc)" >/dev/null
        echo "  hidl-gen (waydroid-patched) built"
    fi
    # link stubs: the image's VNDK apex (vendor processes resolve VNDK libs here) and /system/lib64 (LLNDK)
    [ -r "$SYSIMG" ] || die "$SYSIMG not readable (set WAYDROID_SYSTEM_IMG)"
    IMG=$W/image
    if [ ! -e "$IMG/.provisioned" ] || [ "$(cat "$IMG/.provisioned")" != "$(stat -c %Y:%s "$SYSIMG")" ]; then
        rm -rf "$IMG"; mkdir -p "$IMG/vndk" "$IMG/system"
        debugfs -R "rdump /system/apex/com.android.vndk.current/lib64 $IMG/vndk" "$SYSIMG" 2>&1 \
            | grep -v -e '^debugfs' -e 'Operation not permitted' || true
        debugfs -R "rdump /system/lib64 $IMG/system" "$SYSIMG" 2>&1 \
            | grep -v -e '^debugfs' -e 'Operation not permitted' || true
        [ -f "$IMG/vndk/lib64/libcodec2.so" ] || die "image VNDK apex has no libcodec2.so"
        stat -c %Y:%s "$SYSIMG" > "$IMG/.provisioned"
    fi
fi
IMG=$W/image
VNDK=$IMG/vndk/lib64
WD=$W/deps/wd
[ -e "$WD/.provisioned" ] && [ -x "$HG" ] || die "$WD / waydroid-patched hidl-gen missing: run stage fetch"
LLNDK=$IMG/system/lib64

HO=$W/hidl-out
if has_stage hidl; then
    echo "== stage hidl"
    rm -rf "$HO"; mkdir -p "$HO/headers" "$HO/src"
    ROOTS=(-r "android.hidl:$WD/aosp_libhidl/transport" -r "android.hardware:$HD/aosp_hardware_interfaces")
    for pkg in android.hidl.base@1.0 android.hidl.manager@1.0 android.hidl.manager@1.1 android.hidl.manager@1.2 \
               android.hidl.safe_union@1.0 android.hidl.memory@1.0 android.hidl.token@1.0 \
               android.hardware.graphics.common@1.0 android.hardware.graphics.common@1.1 \
               android.hardware.graphics.common@1.2 android.hardware.graphics.bufferqueue@1.0 \
               android.hardware.graphics.bufferqueue@2.0 android.hardware.graphics.mapper@4.0 \
               android.hardware.media@1.0 android.hardware.media.omx@1.0 android.hardware.media.bufferpool@2.0 \
               android.hardware.media.c2@1.0 android.hardware.media.c2@1.1 android.hardware.media.c2@1.2; do
        "$HG" -o "$HO/headers" -L c++-headers "${ROOTS[@]}" "$pkg"
    done
    for v in 1.0 1.1 1.2; do
        "$HG" -o "$HO/src/c2@$v" -L c++-sources "${ROOTS[@]}" "android.hardware.media.c2@$v"
    done
    # NDK-backend AIDL headers for gralloctypes/Gralloc4.h that android/aidl-stubs lacks. No aidl
    # compiler in the toolchain: emit the C++ shape the NDK backend generates (enum class over the
    # @Backing type, plain-field parcelable) from the pinned graphics/common .aidl. Only the type layout
    # and enumerator values cross the libgralloctypes ABI; no parcel code is used by these modules.
    AG=$HO/aidl/aidl/android/hardware/graphics/common; mkdir -p "$AG"
    ADIR=$HD/aosp_hardware_interfaces/graphics/common/aidl/android/hardware/graphics/common
    for e in Dataspace PlaneLayoutComponentType StandardMetadataType; do
        f=$ADIR/$e.aidl
        bt=$(sed -n 's/.*@Backing(type="\([a-z]*\)").*/\1/p' "$f")
        case $bt in int) ct=int32_t;; long) ct=int64_t;; byte) ct=int8_t;; *) die "$e: backing '$bt'";; esac
        { echo "// generated by media/vndk/build.sh from $e.aidl @ $AOSP_HARDWARE_INTERFACES_SHA"
          echo "#pragma once"; echo "#include <cstdint>"
          echo "namespace aidl::android::hardware::graphics::common {"
          echo "enum class $e : $ct {"
          sed -n "/^enum $e {/,/^}/p" "$f" | sed '1d;$d' | sed 's://.*$::' | grep -v '^\s*$' \
              | grep -v '^\s*/\?\*'
          echo "};"; echo "}"; } > "$AG/$e.h"
    done
    grep -q '^parcelable Rect {' "$ADIR/Rect.aidl" || die "Rect.aidl shape changed"
    { echo "// generated by media/vndk/build.sh from Rect.aidl @ $AOSP_HARDWARE_INTERFACES_SHA"
      echo "#pragma once"; echo "#include <cstdint>"
      echo "namespace aidl::android::hardware::graphics::common {"
      echo "struct Rect {"
      sed -n '/^parcelable Rect {/,/^}/p' "$ADIR/Rect.aidl" | sed '1d;$d' | sed 's/^\s*int /  int32_t /;s/;/ = 0;/'
      echo "};"; echo "}"; } > "$AG/Rect.h"
    echo "  $(find "$HO/headers" -name '*.h' | wc -l) headers, $(find "$HO/src" -name '*.cpp' | wc -l) sources"
fi

# Patched source tree: av at AV_SHA (sparse dirs) + patches/*.patch, rebuilt every run (idempotent).
AV=$W/src/media
prepare_src() {
    rm -rf "$W/src"; mkdir -p "$W/src"
    git -C "$W/av" archive "$AV_SHA" "${AV_DIRS[@]}" | tar -x -C "$W/src"
    mkdir -p "$W/src/frameworks_native"
    git -C "$HD/aosp_frameworks_native" archive "$AOSP_FRAMEWORKS_NATIVE_SHA" libs/ui/include \
        | tar -x -C "$W/src/frameworks_native"
    git -C "$W/src" init -q && git -C "$W/src" add -A
    git -C "$W/src" -c user.name=c2 -c user.email=c2@build commit -qm "base $AV_SHA"
    : > "$W/out/PATCHES"; echo "frameworks/av $AV_SHA, frameworks/native libs/ui/include $AOSP_FRAMEWORKS_NATIVE_SHA" >> "$W/out/PATCHES"
    for pf in "$HERE"/patches/[0-9][0-9][0-9][0-9]-*.patch; do
        [ -e "$pf" ] || continue
        git -C "$W/src" apply --whitespace=nowarn "$pf" || die "patch does not apply: $pf"
        git -C "$W/src" add -A
        git -C "$W/src" diff --cached --quiet && die "patch changed nothing: $pf"
        git -C "$W/src" -c user.name=c2 -c user.email=c2@build commit -qm "$(basename "$pf")"
        echo "$(sha256sum "$pf" | cut -c1-16)  $(basename "$pf")" >> "$W/out/PATCHES"
    done
}
INC_COMMON=(
  -I"$HO/headers"
  -I"$WD/aosp_libhidl/transport/include" -I"$WD/aosp_libhidl/base/include"
  -I"$WD/aosp_libhwbinder/include" -I"$HD/aosp_fmq/include" -I"$HD/aosp_fmq/base"
  -I"$HD/aosp_system_core/libcutils/include" -I"$HD/aosp_system_core/libutils/include"
  -I"$HD/aosp_system_core/libsystem/include" -I"$HD/aosp_system_core/libsync/include"
  -I"$HD/android_system_logging/liblog/include" -I"$HD/aosp_libbase/include"
)
INC_C2=(
  -I"$AV/codec2/core/include" -I"$AV/codec2/vndk/include" -I"$AV/codec2/vndk/internal"
  -I"$AV/bufferpool/2.0/include"
  -I"$AV/libstagefright/foundation/include" -I"$AV/libstagefright/bqhelper/include"
  -I"$AV/libstagefright/include"
  -I"$W/src/frameworks_native/libs/ui/include" -I"$HD/aosp_frameworks_native/libs/nativewindow/include"
  -I"$HD/aosp_frameworks_native/libs/nativebase/include" -I"$HD/aosp_frameworks_native/libs/arect/include"
  -I"$HD/aosp_frameworks_native/libs/math/include" -I"$HD/aosp_frameworks_native/libs/gralloc/types/include"
  -I"$HD/aosp_frameworks_native/libs/binder/include" -I"$HD/aosp_frameworks_native/headers/media_plugin"
  -I"$HD/aosp_frameworks_native/libs/gui/include"
  -I"$HD/aosp_libhardware/include"
  -I"$W/deps/libion/include" -I"$W/deps/libion/kernel-headers"
  -I"$W/deps/libdmabufheap/include"
  -I"$HO/aidl" -I"$REPO/android/aidl-stubs"
)
# Soong (build/soong cc/config/global.go + x86_64_device.go @ android-13.0.0_r79) flags for a vendor-variant
# cc_library: commonGlobalCflags, deviceGlobalCflags, commonGlobalCppflags, noOverrideGlobalCflags, plus the
# vendor-variant defines (-D__ANDROID_VNDK__ -D__ANDROID_VENDOR__), no RTTI (Soong default rtti: false),
# then the module cflags (per module below). Deviations: -isystem libcxx-override (NDK headers configured
# for the platform libc++ namespace std::__1), NDK r29 clang instead of clang-r450784d (14.0.6), so
# warnings that clang 14 did not have are listed in NEWCLANG_NOERR, -g dropped (size), no -nostdlibinc
# (NDK sysroot provides bionic headers).
SOONG_CFLAGS=(-DANDROID -fmessage-length=0 -W -Wall -Wno-unused -Winit-self -Wpointer-arith
  -Wunreachable-code-loop-increment -no-canonical-prefixes -DNDEBUG -UDEBUG -fno-exceptions -Wno-multichar
  -O2 -fno-strict-aliasing -Werror=date-time -Werror=pragma-pack -Werror=pragma-pack-suspicious-include
  -Werror=string-plus-int -Werror=unreachable-code-loop-increment -D__compiler_offsetof=__builtin_offsetof
  -faddrsig -fcommon -Werror=int-conversion -Wno-reserved-id-macro -Wno-sign-compare
  -Wno-defaulted-function-deleted -Wno-inconsistent-missing-override -Wno-c99-designator -Wno-gnu-designator
  -Wno-gnu-folding-constant -Wunguarded-availability -D__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__
  -ffp-contract=off -ftrivial-auto-var-init=zero
  -ffunction-sections -fdata-sections -fno-short-enums -funwind-tables -fstack-protector-strong
  -Wa,--noexecstack -D_FORTIFY_SOURCE=2 -Wstrict-aliasing=2 -Werror=return-type -Werror=non-virtual-dtor
  -Werror=address -Werror=sequence-point -Werror=format-security -Werror=implicit-function-declaration
  -Wsign-promo -Wimplicit-fallthrough -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS -Wno-gnu-include-next
  -fvisibility-inlines-hidden -fPIC
  -Werror=bool-operation -Werror=implicit-int-float-conversion -Werror=int-in-bool-context
  -Werror=int-to-pointer-cast -Werror=pointer-to-int-cast -Werror=string-compare -Werror=xor-used-as-pow
  -Wno-void-pointer-to-enum-cast -Wno-void-pointer-to-int-cast -Wno-pointer-to-int-cast
  -Werror=fortify-source -Werror=address-of-temporary -Wno-tautological-constant-compare
  -Wno-tautological-type-limit-compare -Wno-reorder-init-list -Wno-implicit-int-float-conversion
  -Wno-sizeof-array-div -Wno-tautological-overlap-compare -Wno-deprecated-copy -Wno-range-loop-construct
  -Wno-misleading-indentation -Wno-zero-as-null-pointer-constant -Wno-deprecated-anon-enum-enum-conversion
  -Wno-string-compare -Wno-pessimizing-move -Wno-non-c-typedef-for-linkage -Wno-align-mismatch
  -Wno-error=unused-but-set-variable -Wno-error=unused-but-set-parameter)
# invalid-specialization (clang 20+): codec2/core C2Enum.h specializes std::underlying_type/is_enum.
# missing-designated-field-initializers (clang 19+, split out of -Wmissing-field-initializers that -W enables;
# clang 14 did not warn on designated initializers): codec2/hidl/1.x/utils/types.cpp.
NEWCLANG_NOERR=(-Wno-invalid-specialization -Wno-missing-designated-field-initializers)
# Android 13 bionic defines PAGE_SIZE 4096 on x86_64 (libfmq headers use it); NDK r29 hides it unless:
# NDK r29 libc++ marks std::iterator deprecated (libhidl HidlSupport.h derives from it); diagnostics only:
NDK_COMPAT=(-D__BIONIC_DEPRECATED_PAGE_SIZE_MACRO -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS)
CXXFLAGS=(-std=gnu++17 -isystem "$REPO/android/libcxx-override" "${SOONG_CFLAGS[@]}"
          -D__ANDROID_VNDK__ -D__ANDROID_VENDOR__ -fno-rtti "-ffile-prefix-map=$W/=" "${NDK_COMPAT[@]}"
          -Wall -Werror "${NEWCLANG_NOERR[@]}")   # -Wall -Werror: every module here (hidl_defaults / its bp)
LDLIBS_BASE=(-nostdlib++ "$W/obj/libcxx_compat.a" -Wl,--exclude-libs,libcxx_compat.a -L"$OUT" -L"$VNDK" -L"$LLNDK" -Wl,-z,defs
  -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--build-id=md5 -Wl,--fatal-warnings
  -Wl,--no-undefined-version -Wl,--icf=safe)

compile_link() {  # NAME "extra cflags" "srcs..." -- link libs...
    local name=$1 extra=$2; shift 2
    local srcs=() libs=() o objs=() obj=$W/obj/$name
    while [ $# -gt 0 ] && [ "$1" != -- ]; do srcs+=("$1"); shift; done; shift
    libs=("$@")
    rm -rf "$obj"; mkdir -p "$obj"
    for s in "${srcs[@]}"; do
        o=$obj/$(echo "${s#"$W"/}" | tr / _).o
        # shellcheck disable=SC2086
        "${CCL[@]}" "$CXX" "${CXXFLAGS[@]}" $extra "${INC_COMMON[@]}" "${INC_C2[@]}" -c "$s" -o "$o" \
            || die "compile failed: $s ($name)"
        objs+=("$o")
    done
    "$CXX" -shared -o "$OUT/$name.so" "${objs[@]}" \
        "${LDLIBS_BASE[@]}" "${libs[@]}" -lc++ -Wl,-soname,"$name.so" \
        || die "link failed: $name"
    echo "  built $name.so"
}

HIDL_LIBS=(-lhidlbase -llog -lutils -lcutils)
if has_stage build; then
    echo "== stage build"
    mkdir -p "$OUT" "$W/obj"
    prepare_src
    # android/libcxx_compat.cpp: symbols NDK r29 libc++ headers expect from the dylib that the
    # image's libc++.so (Android 13) lacks. Linked from an archive with --exclude-libs so each library
    # keeps a local copy and none of them exports std:: symbols the platform libc++ owns.
    "${CCL[@]}" "$CXX" "${CXXFLAGS[@]}" -c "$REPO/android/libcxx_compat.cpp" -o "$W/obj/libcxx_compat.o"
    rm -f "$W/obj/libcxx_compat.a"; "$BIN/llvm-ar" rcs "$W/obj/libcxx_compat.a" "$W/obj/libcxx_compat.o"
    [ -f "$VNDK/libcodec2.so" ] || die "image stubs missing: run stage fetch"
    # 1. HIDL interface libraries (hidl_interface → cc_library, hidl_defaults: -Wall -Werror)
    C2DEPS=(-l:android.hardware.graphics.bufferqueue@1.0.so -l:android.hardware.graphics.bufferqueue@2.0.so
            -l:android.hardware.graphics.common@1.0.so -l:android.hardware.graphics.common@1.1.so
            -l:android.hardware.graphics.common@1.2.so -l:android.hardware.media.bufferpool@2.0.so
            -l:android.hardware.media.omx@1.0.so -l:android.hardware.media@1.0.so
            -l:android.hidl.safe_union@1.0.so)
    compile_link android.hardware.media.c2@1.0 "" $(find "$HO/src/c2@1.0" -name '*.cpp' | sort) -- \
        "${C2DEPS[@]}" "${HIDL_LIBS[@]}"
    compile_link android.hardware.media.c2@1.1 "" $(find "$HO/src/c2@1.1" -name '*.cpp' | sort) -- \
        "${C2DEPS[@]}" -l:android.hardware.media.c2@1.0.so "${HIDL_LIBS[@]}"
    compile_link android.hardware.media.c2@1.2 "" $(find "$HO/src/c2@1.2" -name '*.cpp' | sort) -- \
        "${C2DEPS[@]}" -l:android.hardware.media.c2@1.0.so -l:android.hardware.media.c2@1.1.so "${HIDL_LIBS[@]}"
    # 2. libstagefright_bufferpool@2.0.1 (media/bufferpool/2.0/Android.bp)
    BP=$AV/bufferpool/2.0
    compile_link libstagefright_bufferpool@2.0.1 "-DBUFFERPOOL_CLONE_HANDLES" \
        "$BP/Accessor.cpp" "$BP/AccessorImpl.cpp" "$BP/BufferPoolClient.cpp" "$BP/BufferStatus.cpp" \
        "$BP/ClientManager.cpp" "$BP/Connection.cpp" "$BP/Observer.cpp" -- \
        -lcutils -lfmq -lhidlbase -llog -lutils -l:android.hardware.media.bufferpool@2.0.so
    # 3. libcodec2_vndk (media/codec2/vndk/Android.bp)
    V=$AV/codec2/vndk
    compile_link libcodec2_vndk "" \
        "$V/C2AllocatorBlob.cpp" "$V/C2AllocatorIon.cpp" "$V/C2AllocatorGralloc.cpp" "$V/C2Buffer.cpp" \
        "$V/C2Config.cpp" "$V/C2DmaBufAllocator.cpp" "$V/C2Fence.cpp" "$V/C2PlatformStorePluginLoader.cpp" \
        "$V/C2Store.cpp" "$V/platform/C2BqBuffer.cpp" "$V/platform/C2SurfaceSyncObj.cpp" "$V/types.cpp" \
        "$V/util/C2Debug.cpp" "$V/util/C2InterfaceHelper.cpp" "$V/util/C2InterfaceUtils.cpp" \
        "$V/util/C2ParamUtils.cpp" -- \
        -l:android.hardware.graphics.bufferqueue@2.0.so -l:android.hardware.graphics.common@1.2.so \
        -l:android.hardware.media.bufferpool@2.0.so -lbase -lcutils -ldl -ldmabufheap -lfmq -lgralloctypes \
        -lhidlbase -lion -llog -lnativewindow -lstagefright_foundation -l:libstagefright_bufferpool@2.0.1.so \
        -lui -lutils
    # 4. libcodec2_hidl_plugin (media/codec2/hidl/plugin/Android.bp, vendor: true)
    P=$AV/codec2/hidl/plugin
    compile_link libcodec2_hidl_plugin "-I$P/include -I$P/internal" \
        "$P/DefaultFilterPlugin.cpp" "$P/FilterWrapper.cpp" -- \
        -lbase -lcodec2 -lcodec2_vndk -llog -lutils
    # 5. libcodec2_hidl@1.x server utils (media/codec2/hidl/1.x/utils/Android.bp, target.vendor libs)
    prev=()
    for v in 1.0 1.1 1.2; do
        U=$AV/codec2/hidl/$v/utils
        c2ifs=()
        for w in 1.0 1.1 1.2; do c2ifs+=(-l:android.hardware.media.c2@$w.so); [ "$w" = "$v" ] && break; done
        hidlinc=""
        for w in 1.0 1.1 1.2; do hidlinc+=" -I$AV/codec2/hidl/$w/utils/include"; [ "$w" = "$v" ] && break; done
        compile_link "libcodec2_hidl@$v" "$hidlinc -I$P/include -I$P/internal" \
            "$U/Component.cpp" "$U/ComponentInterface.cpp" "$U/ComponentStore.cpp" "$U/Configurable.cpp" \
            "$U/InputBufferManager.cpp" "$U/InputSurface.cpp" "$U/InputSurfaceConnection.cpp" "$U/types.cpp" -- \
            -l:android.hardware.graphics.bufferqueue@1.0.so -l:android.hardware.graphics.bufferqueue@2.0.so \
            -l:android.hardware.graphics.common@1.0.so -l:android.hardware.media@1.0.so \
            -l:android.hardware.media.bufferpool@2.0.so "${c2ifs[@]}" -l:android.hardware.media.omx@1.0.so \
            -lbase -lcodec2 "${prev[@]}" -lcodec2_vndk -lcutils -lhidlbase -llog \
            -l:libstagefright_bufferpool@2.0.1.so -lstagefright_bufferqueue_helper -lcodec2_hidl_plugin \
            -lui -lutils
        prev+=(-l:libcodec2_hidl@$v.so)
    done
fi

# ---- stage service (explicit: `build.sh service`) ------------------------------------------------
# android.hardware.media.c2@1.2-service-nvprolapse: media/ core + c2/ front + service main,
# with the same Soong-derived flags. Links the libraries built above, image stubs, vendor.img's
# libhwminijail.so (link-time copy), and the nv_video backend: NV_VIDEO_LIB_DIR = directory holding the
# x86_64 Android build of libnv_bionic_native.so (DT_NEEDED only; it is staged in /vendor_extra/lib64).
VENDORIMG=${WAYDROID_VENDOR_IMG:-/var/lib/waydroid/images/vendor.img}
if has_stage service; then
    echo "== stage service"
    [ -n "${NV_VIDEO_LIB_DIR:-}" ] && [ -f "$NV_VIDEO_LIB_DIR/libnv_bionic_native.so" ] \
        || die "NV_VIDEO_LIB_DIR must hold the x86_64-linux-android libnv_bionic_native.so"
    [ -f "$OUT/libcodec2_hidl@1.2.so" ] || die "run the build stage first"
    "$READELF" -h "$NV_VIDEO_LIB_DIR/libnv_bionic_native.so" | grep -q 'X86-64' || die "libnv_bionic_native.so is not x86_64"
    MV=$W/image/vendor; mkdir -p "$MV"
    debugfs -R "dump /lib64/libhwminijail.so $MV/libhwminijail.so" "$VENDORIMG" >/dev/null 2>&1
    [ -s "$MV/libhwminijail.so" ] || die "cannot read /lib64/libhwminijail.so from $VENDORIMG"
    M=$REPO/media
    SINC=(-I"$M/core" -I"$M/c2" -I"$REPO/rm/video"
          -I"$AV/codec2/hidl/1.0/utils/include" -I"$AV/codec2/hidl/1.1/utils/include"
          -I"$AV/codec2/hidl/1.2/utils/include" -I"$AV/codec2/hidl/plugin/include"
          -I"$AV/codec2/components/base/include" -isystem "$REPO/build/minigbm/cros_gralloc"
          -isystem "$REPO/build/minigbm" -I"$WD/aosp_libhidl/minijail/include")
    obj=$W/obj/service; rm -rf "$obj"; mkdir -p "$obj" "$W/out/bin"
    objs=()
    for s in "$M/core/nvc2_decoder.cpp" "$M/core/nvc2_caps.cpp" "$M/c2/NvC2Component.cpp" "$M/c2/service.cpp" \
             "$AV/codec2/components/base/SimpleC2Interface.cpp"; do
        o=$obj/$(basename "$s").o
        "${CCL[@]}" "$CXX" "${CXXFLAGS[@]}" "${INC_COMMON[@]}" "${INC_C2[@]}" "${SINC[@]}" -c "$s" -o "$o" \
            || die "compile failed: $s"
        objs+=("$o")
    done
    BINOUT=$W/out/bin/android.hardware.media.c2@1.2-service-nvprolapse
    "$CXX" -pie -o "$BINOUT" "${objs[@]}" "${LDLIBS_BASE[@]}" -L"$MV" -L"$NV_VIDEO_LIB_DIR" \
        -l:libcodec2_hidl@1.2.so -l:libcodec2_hidl@1.1.so -l:libcodec2_hidl@1.0.so \
        -l:android.hardware.media.c2@1.2.so -l:android.hardware.media.c2@1.1.so -l:android.hardware.media.c2@1.0.so \
        -lcodec2_vndk -lcodec2 -lstagefright_foundation -lhidlbase -lbinder -lutils -lcutils -llog -lbase \
        -lhwminijail -lnv_bionic_native -lc++ || die "link failed: service"
    echo "  built $BINOUT"
    fail=0
    for nd in $("$READELF" -d "$BINOUT" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
        if [ -f "$VNDK/$nd" ]; then where=vndk-apex
        elif [ -f "$OUT/$nd" ]; then where=built
        elif [ "$nd" = libhwminijail.so ]; then where=vendor.img
        elif [ "$nd" = libnv_bionic_native.so ]; then where=vendor_extra
        elif { [ -e "$LLNDK/$nd" ] || [ -L "$LLNDK/$nd" ]; } && case $nd in libc.so|libm.so|libdl.so|liblog.so|libnativewindow.so) true;; *) false;; esac; then where=llndk
        else where=MISSING; fail=1; fi
        echo "  NEEDED $nd -> $where"
    done
    if grep -q '__ndk1' <<< "$("$BIN/llvm-nm" -D "$BINOUT")"; then echo "  std::__ndk1 reference in the service"; fail=1; fi
    [ $fail = 0 ] || die "service verification failed"
fi

if has_stage verify; then
    echo "== stage verify"
    V=$W/out/VERIFY.txt; : > "$V"
    fail=0
    for so in "$OUT"/*.so; do
        n=$(basename "$so")
        soname=$("$READELF" -d "$so" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')
        echo "$n SONAME=$soname" >> "$V"
        [ "$soname" = "$n" ] || { echo "SONAME mismatch: $n" >> "$V"; fail=1; }
        for nd in $("$READELF" -d "$so" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
            if [ -f "$VNDK/$nd" ]; then where=vndk-apex
            elif [ -f "$OUT/$nd" ]; then where=built
            elif { [ -f "$LLNDK/$nd" ] || [ -L "$LLNDK/$nd" ]; } && case $nd in libc.so|libm.so|libdl.so|liblog.so|libnativewindow.so|libsync.so|libvndksupport.so|libandroid_net.so|libbinder_ndk.so|libmediandk.so|libvulkan.so|libEGL.so|libGLESv1_CM.so|libGLESv2.so|libGLESv3.so|libselinux.so) true;; *) false;; esac; then where=llndk
            else where=MISSING; fail=1; fi
            echo "  NEEDED $nd -> $where" >> "$V"
        done
    done
    NM=$BIN/llvm-nm
    for so in "$OUT"/*.so; do   # C++ ABI: platform libc++ namespace only, never the NDK's
        if grep -q '__ndk1' <<< "$("$NM" -D "$so")"; then echo "std::__ndk1 reference: $(basename "$so")" >> "$V"; fail=1; fi
    done
    for sym in _ZN7android8hardware5media2c24V1_25utils14ComponentStoreC1ERKNSt3__110shared_ptrI16C2ComponentStoreEE \
               _ZN7android8hardware5media2c24V1_25utils14ComponentStoreC2ERKNSt3__110shared_ptrI16C2ComponentStoreEE; do
        if grep -q " T $sym\$" <<< "$("$NM" -D --defined-only "$OUT/libcodec2_hidl@1.2.so")"; then
            echo "libcodec2_hidl@1.2.so exports $sym" >> "$V"
        else echo "libcodec2_hidl@1.2.so MISSING export $sym" >> "$V"; fail=1; fi
    done
    # Informational: exported (non-std) symbols vs the image's core-variant build of the same module.
    for so in "$OUT"/*.so; do
        n=$(basename "$so"); ref=$LLNDK/$n
        [ -f "$ref" ] || { echo "export diff $n: no core-variant reference in the image" >> "$V"; continue; }
        exp() { "$NM" -D --defined-only "$1" | awk '$2 ~ /[TDBVR]/ {print $3}' | grep -v '^_ZN\?K\?St3__1' | sort -u; }
        miss=$(comm -23 <(exp "$ref") <(exp "$so") | tr '\n' ' ')
        echo "export diff $n vs image /system/lib64: missing=[${miss}] extra=$(comm -13 <(exp "$ref") <(exp "$so") | wc -l)" >> "$V"
    done
    # Gate: the generated interface libraries import exactly the libhidlbase (hwbinder/hidl transport) ABI that
    # the image's own build of the same .hal imports (Waydroid host_hwbinder entry points included). A stock-AOSP
    # hidl-gen/libhwbinder build imports Parcel() / ProcessState::self() instead and crashes in BHwBinder::transact.
    for so in "$OUT"/android.hardware.*.so; do
        n=$(basename "$so"); ref=$LLNDK/$n
        [ -f "$ref" ] || { echo "hidlbase import diff $n: no reference in the image" >> "$V"; fail=1; continue; }
        imp() { "$NM" -D -u "$1" | awk '{print $2}' | grep '^_ZN\?K\?7android8hardware[0-9]' | grep -v '^_ZN7android8hardware5media' | sort -u; }
        # only-in-image (<) fails: an entry point the image's generated code uses that ours does not (stock
        # Parcel()/ProcessState::self() in place of the host_hwbinder ones). only-in-ours (>) is reported: a
        # header-inline helper clang 14 folded away (logAlwaysFatal via hidl_vec's size check); -z defs already
        # proved the image's libhidlbase defines it.
        lt=$(comm -23 <(imp "$ref") <(imp "$so") | tr '\n' ' '); gt=$(comm -13 <(imp "$ref") <(imp "$so") | tr '\n' ' ')
        echo "hidlbase import diff $n vs image: only-image=[${lt}] only-ours=[${gt}]" >> "$V"
        [ -z "$lt" ] || fail=1
    done
    cat "$V"
    [ $fail = 0 ] || die "verification failed (see $V)"
fi
