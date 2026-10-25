# libgbm_mesa_wrapper.so (NVIDIA native gralloc backend for minigbm gbm_mesa)
#
#   make host      -> out/host/libgbm_mesa_wrapper.so    (glibc, for alloc-test)
#   make android   -> out/android/libgbm_mesa_wrapper.so   (NDK r30, x86_64, API 33)
#   make android32 -> out/android32/libgbm_mesa_wrapper.so (NDK r30, i686, API 33)
#   make check     -> verify both Android builds' exports / SONAME / DT_NEEDED / ELF class
#   make test      -> build and run test/alloc-test against the host build
#   make check-header -> pinned include/gbm_mesa_wrapper.h == build/minigbm copy
#
# The 32-bit build serves the image's 32-bit processes (mediaserver and its software
# codecs, 32-bit apps) as a *mapper* backend: minigbm's gralloc mapper imports and mmaps
# dma-bufs in-process and needs no driver for that. Allocation always happens in the
# 64-bit allocator service. A 32-bit process that did try to allocate would hit the lazy
# ICD setup and fail loudly (no 32-bit NVIDIA closure is shipped).

NDK      ?= /opt/android-ndk
NDK_BIN  := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin
NDK_CC   ?= $(NDK_BIN)/x86_64-linux-android33-clang
NDK_CC32 ?= $(NDK_BIN)/i686-linux-android33-clang
READELF  ?= $(NDK_BIN)/llvm-readelf
HOST_CC  ?= gcc
MINIGBM  ?= ../build/minigbm

SONAME   := libgbm_mesa_wrapper.so
SRC      := nvidia_gbm_wrapper.c
HDRS     := include/gbm_mesa_wrapper.h include/gbm_mesa_wrapper_map_flags.h include/gbm_mesa_wrapper_layout.h

WARN     := -Wall -Wextra -Werror
# Path-independent output: the same sources give the same bytes (debug info included) from any
# checkout path, e.g. the repo and packaging/arch/src/ (the package build).
HERE       := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PREFIX_MAP := -ffile-prefix-map=$(HERE)=gralloc
COMMON   := -std=c11 -O2 -g $(PREFIX_MAP) -fPIC -fvisibility=hidden $(WARN) -Iinclude
# -z nodelete: minigbm dlclose()s the wrapper when its driver object dies;
# the process-wide VkDevice must not be unloaded under live dma-bufs.
LDCOMMON := -shared -Wl,-soname,$(SONAME) -Wl,-z,defs -Wl,-z,nodelete

# host: Vulkan headers from /usr/include/vulkan, drm_fourcc.h from libdrm
HOST_CFLAGS  := $(COMMON) -I/usr/include/libdrm
HOST_LDLIBS  := -ldl -lpthread
# android: Vulkan + <drm/drm_fourcc.h> + <linux/dma-buf.h> from the NDK sysroot.
# _FILE_OFFSET_BITS=64: on ILP32 bionic this makes lseek/mmap take 64-bit offsets
# (dma-buf sizes are measured with lseek(SEEK_END)); a no-op on 64-bit.
ANDROID_CFLAGS := $(COMMON) -D_FILE_OFFSET_BITS=64
ANDROID_LDLIBS := -llog -ldl

# bionic libraries the Android build may depend on; the ICD and libnvglibc
# are dlopen()ed at run time, never DT_NEEDED
ALLOWED_NEEDED := libc.so libdl.so liblog.so libm.so

all: host android android32

host: out/host/$(SONAME)
android: out/android/$(SONAME)
android32: out/android32/$(SONAME)

out/host/$(SONAME): $(SRC) $(HDRS) Makefile
	@mkdir -p $(@D)
	$(HOST_CC) $(HOST_CFLAGS) $(LDCOMMON) -o $@ $(SRC) $(HOST_LDLIBS)

out/android/$(SONAME): $(SRC) $(HDRS) Makefile
	@mkdir -p $(@D)
	$(NDK_CC) $(ANDROID_CFLAGS) $(LDCOMMON) -o $@ $(SRC) $(ANDROID_LDLIBS)

out/android32/$(SONAME): $(SRC) $(HDRS) Makefile
	@mkdir -p $(@D)
	$(NDK_CC32) $(ANDROID_CFLAGS) $(LDCOMMON) -o $@ $(SRC) $(ANDROID_LDLIBS)

out/host/alloc-test: test/alloc-test.c $(HDRS) Makefile
	@mkdir -p $(@D)
	$(HOST_CC) -std=c11 -O1 -g $(PREFIX_MAP) $(WARN) -Iinclude -I/usr/include/libdrm -o $@ test/alloc-test.c -lvulkan -ldl -lpthread

# 32-bit host build of the backend (i686 glibc, lib32-nvidia-utils): the lock path of the image's
# 32-bit processes, tested on the host with lib32 Vulkan/EGL/GLES
out/host32/$(SONAME): $(SRC) $(HDRS) Makefile
	@mkdir -p $(@D)
	$(HOST_CC) -m32 -D_FILE_OFFSET_BITS=64 $(HOST_CFLAGS) $(LDCOMMON) -o $@ $(SRC) $(HOST_LDLIBS)

LOCKTEST_CFLAGS := -std=c11 -O1 -g $(PREFIX_MAP) $(WARN) -Iinclude -I/usr/include/libdrm
out/host/lock-test: test/lock-test.c $(HDRS) Makefile
	@mkdir -p $(@D)
	$(HOST_CC) $(LOCKTEST_CFLAGS) -o $@ test/lock-test.c -lvulkan -lEGL -lGLESv2 -ldl
out/host32/lock-test: test/lock-test.c $(HDRS) Makefile
	@mkdir -p $(@D)
	$(HOST_CC) -m32 -D_FILE_OFFSET_BITS=64 $(LOCKTEST_CFLAGS) -o $@ test/lock-test.c -L/usr/lib32 \
	    -lvulkan -lEGL -lGLESv2 -ldl

# alloc-test (layout rule, LINEAR CPU buffers, formats, leaks), then lock-test for CPU-mapped
# render targets (block-linear + GPU staging copy) in both host ABIs. About 2 minutes.
test: out/host/$(SONAME) out/host/alloc-test out/host/lock-test out/host32/$(SONAME) out/host32/lock-test
	DISPLAY= WAYLAND_DISPLAY= ./out/host/alloc-test ./out/host/$(SONAME)
	DISPLAY= WAYLAND_DISPLAY= ./out/host/lock-test ./out/host/$(SONAME)
	DISPLAY= WAYLAND_DISPLAY= ./out/host32/lock-test ./out/host32/$(SONAME)

check: out/android/$(SONAME) out/android32/$(SONAME)
	@set -e; for pair in out/android/$(SONAME):ELF64 out/android32/$(SONAME):ELF32; do \
	lib=$${pair%%:*}; class=$${pair##*:}; echo "== $$lib"; \
	$(READELF) -h $$lib | grep -q "Class:.*$$class" || { echo "FAIL: $$lib is not $$class"; exit 1; }; \
	exports=$$($(READELF) --dyn-syms --wide $$lib | awk '$$1 ~ /^[0-9]+:$$/ && $$7 != "UND" && $$5 != "LOCAL" && $$8 != "" {print $$8}' | sort -u); \
	echo "exports: $$exports"; \
	[ "$$exports" = "$$(printf 'get_gbm_alloc_flags_ops\nget_gbm_layout_ops\nget_gbm_map_flags_ops\nget_gbm_ops')" ] || { echo "FAIL: dynamic exports must be exactly get_gbm_ops + get_gbm_map_flags_ops + get_gbm_layout_ops + get_gbm_alloc_flags_ops"; exit 1; }; \
	soname=$$($(READELF) -d $$lib | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p'); \
	echo "SONAME: $$soname"; \
	[ "$$soname" = "$(SONAME)" ] || { echo "FAIL: SONAME"; exit 1; }; \
	needed=$$($(READELF) -d $$lib | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); \
	echo "NEEDED:" $$needed; \
	for n in $$needed; do \
		case " $(ALLOWED_NEEDED) " in *" $$n "*) ;; *) echo "FAIL: DT_NEEDED $$n is not a bionic library"; exit 1;; esac; \
	done; \
	$(READELF) -d $$lib | grep -q 'FLAGS_1.*NODELETE' || { echo "FAIL: DF_1_NODELETE missing"; exit 1; }; \
	done; echo "check: OK"

# ---------------------------------------------------------------------------------------------------
# libminigbm_gralloc_gbm_mesa.so (x86_64): minigbm's gbm_mesa core, pinned 16844ba + our patch
# (minigbm/0001-*.patch: CPU usage no longer forces LINEAR on GPU render targets). Only the 64-bit
# library is rebuilt: the allocator service is 64-bit and is the only caller of bo_create; every
# mapper (64- and 32-bit) works with either library because our map path is chosen by the buffer's
# modifier, not by its usage.
#
# minigbm is C + C++ (Soong: c++17, no exceptions) and its C++ API (cros_gralloc_driver, std::function)
# is called by the image's own mapper/allocator, so it is compiled against the platform libc++'s
# headers (AOSP android13-release external/libcxx, std::__1, never the NDK's std::__ndk1) and linked
# with -z defs against link-time copies of the guest's own libraries (read from the images with
# debugfs, never shipped). Third-party trees are pinned below; `make minigbm-pins` checks them.
MINIGBM_REV  := 16844babfbd83d7c206ca22256257cbf26ca7de1
A13SRC       ?= ../build/android13-src
LIBCXX       ?= ../build/android13-libcxx
FWNATIVE     ?= ../build/p7/src/fw-native
LIBDRM_SRC   ?= ../build/libdrm/drm
MESA         ?= ../build/mesa-git
MINIGBM_PINS := $(A13SRC)/system-core:83366e7e7a1556ab6cff117673bb9f17b973de99 \
                $(A13SRC)/system-logging:6bfb380b168c7127d79977a0e931d6e5e469d887 \
                $(A13SRC)/hardware-libhardware:bbd07bd2e60752a665ebc9b0915241f3c1636516 \
                $(LIBCXX):d387ee3a2765e94899e3eb445870e7f2dc1eb2b1 \
                $(FWNATIVE):2fe7eddf185ecb63223eeadc58804f9bf6cbd426 \
                $(MINIGBM):$(MINIGBM_REV)
MESA_REV     := ddaf85e353442d9d7f192fdcf1823e652e3f2ee3
LIBDRM_TAG   := libdrm-2.4.134
SYSTEM_IMG   ?= /var/lib/waydroid/images/system.img
VENDOR_IMG   ?= /var/lib/waydroid/images/vendor.img
MG_PATCHES   := $(sort $(wildcard minigbm/*.patch))
MG_SRC       := out/minigbm-src
MG_LIB       := libminigbm_gralloc_gbm_mesa.so
SYSLIBS      := out/syslibs/lib64
SYSLIBS32    := out/syslibs/lib
# vendor processes see libc++/libcutils from the VNDK APEX, the rest are LLNDK (/system) or vendor
SYSLIB_SRC   := system:/system/apex/com.android.vndk.current/lib64/libc++.so \
                system:/system/apex/com.android.vndk.current/lib64/libcutils.so \
                system:/system/lib64/liblog.so system:/system/lib64/libsync.so \
                system:/system/lib64/libnativewindow.so vendor:/lib64/libdrm.so
SYSLIB_SRC32 := system:/system/apex/com.android.vndk.current/lib/libc++.so \
                system:/system/apex/com.android.vndk.current/lib/libcutils.so \
                system:/system/lib/liblog.so system:/system/lib/libsync.so \
                system:/system/lib/libnativewindow.so vendor:/lib/libdrm.so
MG_C_SRCS    := amdgpu.c drv.c drv_array_helpers.c drv_helpers.c dumb_driver.c i915.c mediatek.c msm.c \
                rockchip.c vc4.c virtgpu.c virtgpu_cross_domain.c virtgpu_virgl.c
MG_CXX_SRCS  := cros_gralloc/cros_gralloc_buffer.cc cros_gralloc/cros_gralloc_helpers.cc \
                cros_gralloc/cros_gralloc_driver.cc gbm_mesa_driver/gbm_mesa_internals.cpp \
                gbm_mesa_driver/gbm_mesa_driver.cpp
NDK_CXX      ?= $(NDK_BIN)/x86_64-linux-android33-clang++
NDK_CXX32    ?= $(NDK_BIN)/i686-linux-android33-clang++
MG_INC       := -I$(MG_SRC) -I$(MG_SRC)/gbm_mesa_driver -Iout/minigbm-gen \
                -isystem $(A13SRC)/system-core/libcutils/include -isystem $(A13SRC)/system-core/libsystem/include \
                -isystem $(A13SRC)/system-core/libsync/include -isystem $(A13SRC)/system-logging/liblog/include \
                -isystem $(A13SRC)/hardware-libhardware/include \
                -isystem $(FWNATIVE)/libs/nativewindow/include -isystem $(FWNATIVE)/libs/nativebase/include \
                -isystem $(LIBDRM_SRC) -isystem $(LIBDRM_SRC)/include/drm
# Soong minigbm_defaults + DRV_EXTERNAL (gbm_mesa_driver/Android.bp); vendor module, API 33
MG_DEFS      := -D_GNU_SOURCE=1 -D_FILE_OFFSET_BITS=64 -DANDROID_API_LEVEL=33 -DDRV_EXTERNAL -D__ANDROID_VNDK__
MG_CFLAGS    := -O2 -g -fPIC $(PREFIX_MAP) -Wall -Wno-unused-parameter -Wno-reorder-init-list -Wno-invalid-offsetof $(MG_DEFS) $(MG_INC)
MG_CXXFLAGS  := $(MG_CFLAGS) -std=c++17 -fno-exceptions -fno-rtti -nostdinc++ -isystem $(LIBCXX)/include

# Clone the pinned trees minigbm-pins checks (shallow fetch of the exact commit; an existing clone at
# another commit is an error, never reset). packaging/native/build-native.sh runs this before `minigbm`.
MINIGBM_SOURCES := $(A13SRC)/system-core=https://android.googlesource.com/platform/system/core \
                   $(A13SRC)/system-logging=https://android.googlesource.com/platform/system/logging \
                   $(A13SRC)/hardware-libhardware=https://android.googlesource.com/platform/hardware/libhardware \
                   $(LIBCXX)=https://android.googlesource.com/platform/external/libcxx \
                   $(FWNATIVE)=https://android.googlesource.com/platform/frameworks/native \
                   $(MINIGBM)=https://github.com/waydroid/android_external_minigbm.git
minigbm-fetch:
	@set -e; for p in $(MINIGBM_PINS); do d=$${p%%:*}; want=$${p##*:}; url=; \
	    for s in $(MINIGBM_SOURCES); do [ "$${s%%=*}" = "$$d" ] && url=$${s#*=}; done; \
	    [ -n "$$url" ] || { echo "minigbm-fetch: no source URL for $$d" >&2; exit 1; }; \
	    if [ -d $$d/.git ]; then have=$$(git -C $$d rev-parse HEAD); \
	        [ "$$have" = "$$want" ] || { echo "minigbm-fetch: $$d is at $$have, want $$want (move it aside)" >&2; exit 1; }; \
	    else mkdir -p $$d && git -C $$d init -q && python3 ../tools/fetch.py retry -- git -C $$d fetch -q --depth 1 $$url $$want && \
	        git -C $$d checkout -q FETCH_HEAD; echo "minigbm-fetch: $$d at $$want"; fi; done

minigbm-pins:
	@set -e; for p in $(MINIGBM_PINS); do d=$${p%%:*}; want=$${p##*:}; \
	    have=$$(git -C $$d rev-parse HEAD 2>/dev/null || echo missing); \
	    [ "$$have" = "$$want" ] || { echo "minigbm-pins: $$d is at $$have, want $$want" >&2; exit 1; }; done; \
	git -C $(MESA) cat-file -e $(MESA_REV):src/gbm/main/gbm.h || { echo "minigbm-pins: $(MESA) lacks $(MESA_REV)" >&2; exit 1; }; \
	git -C $(LIBDRM_SRC) describe --tags --exact-match HEAD | grep -qx $(LIBDRM_TAG) || { echo "minigbm-pins: $(LIBDRM_SRC) is not $(LIBDRM_TAG)" >&2; exit 1; }; \
	echo "minigbm-pins: OK"

$(MG_SRC)/.stamp: $(MG_PATCHES) Makefile | minigbm-pins
	@rm -rf $(MG_SRC) out/minigbm-gen && mkdir -p $(MG_SRC) out/minigbm-gen
	git -C $(MINIGBM) archive $(MINIGBM_REV) | tar -x -C $(MG_SRC)
	@set -e; for p in $(MG_PATCHES); do echo "patch $$p"; patch -s -d $(MG_SRC) -p1 < $$p; done
	@# minigbm includes <gbm.h> (Mesa): the header only, from the pinned Mesa object store
	git -C $(MESA) show $(MESA_REV):src/gbm/main/gbm.h > out/minigbm-gen/gbm.h
	@touch $@

# $(1) = output dir, $(2) = image paths (img:path)
define dump_syslibs
	@mkdir -p $(1)
	@set -e; for s in $(2); do img=$${s%%:*}; p=$${s#*:}; \
	    case $$img in system) f=$(SYSTEM_IMG);; *) f=$(VENDOR_IMG);; esac; \
	    debugfs -R "dump $$p $(1)/$$(basename $$p)" $$f >/dev/null 2>&1; \
	    [ -s $(1)/$$(basename $$p) ] || { echo "cannot read $$p from $$f" >&2; exit 1; }; done; touch $(1)/.stamp
endef
$(SYSLIBS)/.stamp:
	$(call dump_syslibs,$(SYSLIBS),$(SYSLIB_SRC))
$(SYSLIBS32)/.stamp:
	$(call dump_syslibs,$(SYSLIBS32),$(SYSLIB_SRC32))

# $(1) = C compiler, $(2) = C++ compiler, $(3) = object dir, $(4) = syslibs dir
define build_minigbm
	@rm -rf $(3) && mkdir -p $(3) $(@D)
	@set -e; for s in $(MG_C_SRCS); do echo "  CC  $$s"; \
	    $(1) -std=gnu11 $(MG_CFLAGS) -c $(MG_SRC)/$$s -o $(3)/$$(basename $$s).o; done
	@set -e; for s in $(MG_CXX_SRCS); do echo "  CXX $$s"; \
	    $(2) $(MG_CXXFLAGS) -c $(MG_SRC)/$$s -o $(3)/$$(basename $$s).o; done
	$(2) -shared -nostdlib++ -Wl,-soname,$(MG_LIB) -Wl,-z,defs -o $@ \
	    $(3)/*.o -L$(4) -lcutils -ldrm -lnativewindow -lsync -llog -lc++ -ldl -lm
	@echo "built $@ from minigbm $(MINIGBM_REV) + $(notdir $(MG_PATCHES))"
endef
out/android/$(MG_LIB): $(MG_SRC)/.stamp $(SYSLIBS)/.stamp
	$(call build_minigbm,$(NDK_CC),$(NDK_CXX),out/minigbm-obj,$(SYSLIBS))
# 32-bit (i686): the image's 32-bit mapper and gralloc0 module NEED it; with our patch 0002 their
# CPU locks pass the lock's access flags to the backend like the 64-bit ones
out/android32/$(MG_LIB): $(MG_SRC)/.stamp $(SYSLIBS32)/.stamp
	$(call build_minigbm,$(NDK_CC32),$(NDK_CXX32),out/minigbm-obj32,$(SYSLIBS32))

minigbm: out/android/$(MG_LIB) out/android32/$(MG_LIB)

# The rebuilt library must export every symbol the image's own consumers of it import from it (the
# 64-bit mapper, allocator service and gralloc0 module, the only vendor ELF files that NEED it,
# measured over the whole vendor.img), import nothing the image's libraries lack (-z defs above), and
# never reference the NDK's libc++ ABI. Weak template instantiations that only the image's build
# happened to emit are not part of that contract.
MG_CONSUMERS := /lib64/hw/android.hardware.graphics.mapper@4.0-impl.minigbm_gbm_mesa.so \
                /bin/hw/android.hardware.graphics.allocator@4.0-service.minigbm_gbm_mesa \
                /lib64/hw/gralloc.minigbm_gbm_mesa.so
MG_CONSUMERS32 := /lib/hw/android.hardware.graphics.mapper@4.0-impl.minigbm_gbm_mesa.so \
                /lib/hw/gralloc.minigbm_gbm_mesa.so
# $(1) = our library, $(2) = the image's library path, $(3) = consumers, $(4) = work dir
define check_minigbm
	@rm -rf $(4) && mkdir -p $(4)
	@set -e; for p in $(2) $(3); do \
	    debugfs -R "dump $$p $(4)/$$(basename $$p)" $(VENDOR_IMG) >/dev/null 2>&1; \
	    [ -s $(4)/$$(basename $$p) ] || { echo "cannot read $$p from $(VENDOR_IMG)" >&2; exit 1; }; done
	@set -e; exp() { $(READELF) --dyn-syms --wide $$1 | awk '$$1 ~ /^[0-9]+:$$/ && $$7 != "UND" && $$5 != "LOCAL" && $$8 != "" {print $$8}' | sort -u; }; \
	und() { $(READELF) --dyn-syms --wide $$1 | awk '$$1 ~ /^[0-9]+:$$/ && $$7 == "UND" && $$8 != "" {sub(/@.*/, "", $$8); print $$8}' | sort -u; }; \
	exp $(4)/$(MG_LIB) > $(4)/image.exp; exp $(1) > $(4)/ours.exp; \
	for c in $(notdir $(3)); do und $(4)/$$c; done | sort -u | comm -12 - $(4)/image.exp > $(4)/needed.sym; \
	missing=$$(comm -23 $(4)/needed.sym $(4)/ours.exp); \
	echo "$(1): exports: image $$(wc -l < $(4)/image.exp), ours $$(wc -l < $(4)/ours.exp); imported from it by the image's consumers: $$(wc -l < $(4)/needed.sym)"; \
	[ -z "$$missing" ] || { echo "FAIL: exports the image's consumers need, missing from ours:"; echo "$$missing"; exit 1; }; \
	$(READELF) -h $(1) | grep -q "Class:.*$(5)" || { echo "FAIL: $(1) is not $(5)"; exit 1; }; \
	echo "NEEDED: $$($(READELF) -d $(1) | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')"; \
	if $(READELF) --dyn-syms --wide $(1) | grep -q '__ndk1'; then echo "FAIL: NDK libc++ ABI (std::__ndk1) referenced"; exit 1; fi
endef
check-minigbm: out/android/$(MG_LIB) out/android32/$(MG_LIB)
	$(call check_minigbm,out/android/$(MG_LIB),/lib64/$(MG_LIB),$(MG_CONSUMERS),out/minigbm-check,ELF64)
	$(call check_minigbm,out/android32/$(MG_LIB),/lib/$(MG_LIB),$(MG_CONSUMERS32),out/minigbm-check32,ELF32)
	@echo "check-minigbm: OK"

# import-lock-test: minigbm's gbm_mesa core (drv.c + gbm_mesa_driver, the layer cros_gralloc's
# retain/lock call; no cros_gralloc/HIDL) compiled for the host from the same patched tree, with
# the host backend (clang++ like Soong; libstdc++ needs the three -include headers that the platform
# libc++ pulls in transitively). Allocates GPU/codec-only NV12/P010 in one process, imports and CPU-locks them
# in another (minigbm/0005). The test supplies property_get; minigbm logs to stderr off Android.
HOSTMG_C   := drv.c drv_helpers.c drv_array_helpers.c
HOSTMG_CXX := gbm_mesa_driver/gbm_mesa_internals.cpp gbm_mesa_driver/gbm_mesa_driver.cpp
HOSTMG_SRC ?= $(MG_SRC)
HOSTMG_INC  = -I$(HOSTMG_SRC) -I$(HOSTMG_SRC)/gbm_mesa_driver -Iout/minigbm-gen \
              -isystem $(A13SRC)/system-core/libcutils/include -isystem $(A13SRC)/system-logging/liblog/include \
              -I/usr/include/libdrm
HOSTMG_FLAGS = -O1 -g $(PREFIX_MAP) -Wall -Wno-unused-parameter -D_GNU_SOURCE -DDRV_EXTERNAL $(HOSTMG_INC)
out/host/import-lock-test: test/import-lock-test.c $(MG_SRC)/.stamp Makefile
	@rm -rf out/host/hostmg-obj && mkdir -p out/host/hostmg-obj
	@set -e; for s in $(HOSTMG_C); do $(HOST_CC) -std=gnu11 $(HOSTMG_FLAGS) -c $(HOSTMG_SRC)/$$s -o out/host/hostmg-obj/$$(basename $$s).o; done
	@set -e; for s in $(HOSTMG_CXX); do clang++ -std=c++17 -fno-exceptions -fno-rtti -Wno-reorder-init-list -include utility -include functional -include cassert $(HOSTMG_FLAGS) -c $(HOSTMG_SRC)/$$s -o out/host/hostmg-obj/$$(basename $$s).o; done
	$(HOST_CC) -std=gnu11 $(HOSTMG_FLAGS) -Wextra -Werror -c test/import-lock-test.c -o out/host/hostmg-obj/import-lock-test.o
	g++ -o $@ out/host/hostmg-obj/*.o -ldrm -ldl -lpthread

import-lock-test: out/host/$(SONAME) out/host/import-lock-test
	DISPLAY= WAYLAND_DISPLAY= LD_LIBRARY_PATH=$(CURDIR)/out/host ./out/host/import-lock-test

# guest-import-lock: the same check in the guest through cros_gralloc_driver of a libminigbm given
# by path (push it and the binary to /data/local/tmp; no vendor mount is touched)
out/android/guest-import-lock: test/guest-import-lock.c Makefile
	@mkdir -p $(@D)
	$(NDK_CC) -std=gnu11 -O1 -g $(PREFIX_MAP) $(WARN) -o $@ test/guest-import-lock.c -landroid -lnativewindow -ldl

check-header: $(MG_SRC)/.stamp
	@tail -n +11 include/gbm_mesa_wrapper.h | cmp - $(MINIGBM)/gbm_mesa_driver/gbm_mesa_wrapper.h && \
	git -C $(MINIGBM) rev-parse HEAD | grep -qx 16844babfbd83d7c206ca22256257cbf26ca7de1 && \
	tail -n +8 include/gbm_mesa_wrapper_map_flags.h | cmp - $(MG_SRC)/gbm_mesa_driver/gbm_mesa_wrapper_map_flags.h && \
	tail -n +8 include/gbm_mesa_wrapper_layout.h | cmp - $(MG_SRC)/gbm_mesa_driver/gbm_mesa_wrapper_layout.h && \
	tail -n +8 include/gbm_mesa_wrapper_alloc_flags.h | cmp - $(MG_SRC)/gbm_mesa_driver/gbm_mesa_wrapper_alloc_flags.h && \
	echo "check-header: OK (identical to minigbm 16844ba / to the patched tree)"

clean:
	rm -rf out

.PHONY: minigbm-fetch all host android android32 test import-lock-test check check-header clean minigbm minigbm-pins check-minigbm
