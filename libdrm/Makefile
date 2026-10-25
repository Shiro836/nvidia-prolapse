# libdrm.so.2 for bionic, built from pinned upstream (README.md says why the driver needs it).
#   make          clone (if needed) + configure + build + copy to out/libdrm.so.2 + check
#   make ARCH=x86 the same for 32-bit x86 (i686) -> out32/libdrm.so.2, checked against the lib32 driver
#   make check    artifact check only (ELF class/machine, SONAME, bionic-only NEEDED, no TEXTREL)
#   make clean    remove out/ and the meson build dir (the clone stays)

NDK      ?= /opt/android-ndk
TC       := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64
READELF  := $(TC)/bin/llvm-readelf
PYTHON   := python3 -I
ARCH     ?= x86_64

HERE     := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TOP      := $(abspath $(HERE)/..)
URL      := https://gitlab.freedesktop.org/mesa/drm.git
VERSION  := 2.4.134
TAG      := libdrm-$(VERSION)
SHA      := e984d448b8b17aab853369e6c203e53719f46de1
SRC      := $(TOP)/build/libdrm/drm
ifeq ($(ARCH),x86_64)
BLD      := $(TOP)/build/libdrm/build-android33-x86_64
OUT      := $(HERE)/out
CROSS    := $(HERE)/x86_64-android33.ini
TRIPLE   := x86_64-linux-android33
HOSTDRM  ?= /usr/lib/libdrm.so.2
else ifeq ($(ARCH),x86)
BLD      := $(TOP)/build/libdrm/build-android33-x86
OUT      := $(HERE)/out32
CROSS    := $(HERE)/x86-android33.ini
TRIPLE   := i686-linux-android33
HOSTDRM  ?= /usr/lib32/libdrm.so.2
else
$(error ARCH must be x86_64 or x86, not '$(ARCH)')
endif

# Only the core library: every GPU-vendor helper library, test and doc is off.
MESON_OPTS := --buildtype=release -Ddefault_library=shared \
    -Dintel=disabled -Dradeon=disabled -Damdgpu=disabled -Dnouveau=disabled \
    -Dvmwgfx=disabled -Domap=disabled -Dexynos=disabled -Dfreedreno=disabled \
    -Dfreedreno-kgsl=false -Dtegra=disabled -Dvc4=disabled -Detnaviv=disabled \
    -Dcairo-tests=disabled -Dman-pages=disabled -Dvalgrind=disabled \
    -Dudev=false -Dtests=false -Dinstall-test-programs=false

.PHONY: all check clean source FORCE
all: $(OUT)/.checked

# The clone must be the pinned commit with no local change (principle: everything from
# pinned sources, no patches). A clone at another commit is an error, never reused.
source:
	@set -e; if [ ! -d "$(SRC)/.git" ]; then \
	    mkdir -p "$(dir $(SRC))"; \
	    echo "libdrm: cloning $(URL) $(TAG)"; \
	    python3 $(TOP)/tools/fetch.py retry --cleanup "$(SRC)" -- \
	        git clone -q -c advice.detachedHead=false --depth 1 --branch $(TAG) $(URL) "$(SRC)"; fi; \
	h=$$(git -C "$(SRC)" rev-parse HEAD); \
	if [ "$$h" != "$(SHA)" ]; then echo "libdrm: $(SRC) is at $$h, not $(TAG) = $(SHA); remove it and re-run make" >&2; exit 1; fi; \
	if [ -n "$$(git -C "$(SRC)" status --porcelain)" ]; then echo "libdrm: $(SRC) has local changes; this build uses pristine $(TAG) (git -C $(SRC) status)" >&2; exit 1; fi

$(BLD)/ndk.ini: FORCE
	@mkdir -p $(BLD)
	@[ -x "$(TC)/bin/$(TRIPLE)-clang" ] || { echo "libdrm: no NDK clang under $(TC) (set NDK=)" >&2; exit 1; }
	@printf "[constants]\nndk = '%s'\n" "$(NDK)" > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv $@.tmp $@; fi

# Any change to the cross file, the options or the NDK path reconfigures from scratch.
$(BLD)/build.ninja: $(BLD)/ndk.ini $(CROSS) $(HERE)/Makefile | source
	rm -rf $(BLD)/meson-private $(BLD)/meson-info $(BLD)/meson-logs $(BLD)/build.ninja
	meson setup $(BLD) $(SRC) --cross-file $(BLD)/ndk.ini --cross-file $(CROSS) $(MESON_OPTS)

$(BLD)/libdrm.so: $(BLD)/build.ninja FORCE | source
	ninja -C $(BLD) libdrm.so

# The real file under the SONAME, no symlink: it lands in /vendor_extra/lib64 as is.
$(OUT)/libdrm.so.2: $(BLD)/libdrm.so
	@mkdir -p $(OUT)
	cp -f $< $@.tmp && mv -f $@.tmp $@

# No NVIDIA userspace loads this libdrm (NVK and the gralloc backend do), so there is no census of
# NVIDIA's libraries here.
CHECK = $(PYTHON) $(HERE)/check-libdrm.py --readelf $(READELF) --allow-needed libc.so,libm.so,libdl.so \
	    --arch $(ARCH) --host-libdrm $(HOSTDRM) --pinned-version $(VERSION) \
	    $(OUT)/libdrm.so.2

$(OUT)/.checked: $(OUT)/libdrm.so.2 $(HERE)/check-libdrm.py FORCE
	@rm -f $@
	@$(CHECK)
	@touch $@

check: $(OUT)/libdrm.so.2
	@$(CHECK)

clean:
	rm -rf $(OUT) $(BLD)
