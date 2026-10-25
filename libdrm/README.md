# libdrm/ — libdrm.so.2 for bionic, from pinned upstream

Mesa's NVK and the gralloc backend link libdrm. Android's own libdrm is built unversioned (`libdrm.so`,
upstream `Android.bp`), so the driver ships its own: the upstream core library (no vendor helper libraries,
no tests, no docs), cross-built with the NDK for Android API 33, x86_64 and i686, installed as the real
file `libdrm.so.2` (`DT_SONAME libdrm.so.2`, `DT_NEEDED` bionic `libc.so` only, linked `-z defs`).

## Pin

`https://gitlab.freedesktop.org/mesa/drm.git` tag `libdrm-2.4.134` = commit
`e984d448b8b17aab853369e6c203e53719f46de1`. Unmodified: `make` refuses a clone at another commit or with
local changes. Upstream's `meson.build` links an unversioned `libdrm.so` when it detects Android; the cross
files (`x86_64-android33.ini`, `x86-android33.ini`) set the SONAME to `libdrm.so.2` with a linker flag.

## Build and check

    make             # clone into ../build/libdrm/drm if needed, meson + ninja, out/libdrm.so.2, check
    make ARCH=x86    # the same for i686 -> out32/libdrm.so.2
    make check       # artifact check only
    make clean

`make check` (`check-libdrm.py`) reads the artifact with `llvm-readelf` and fails on: the wrong ELF class or
machine, a `DT_SONAME` other than `libdrm.so.2`, a `DT_NEEDED` outside `libc.so`/`libm.so`/`libdl.so`
(a glibc SONAME is named as such), `TEXTREL`, or an exported `drm*` set that differs from the host's
`/usr/lib/libdrm.so.2` when the host has the pinned version (another host version prints the difference).
