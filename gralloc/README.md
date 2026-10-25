# gralloc: NVIDIA-native allocation backend for minigbm

`libgbm_mesa_wrapper.so` for NVIDIA native mode. The guest keeps its stock
gralloc, minigbm `gbm_mesa` (`ro.hardware.gralloc=minigbm_gbm_mesa`, HIDL
allocator/mapper 4.0, pinned at `build/minigbm`, commit 16844ba). That backend
does not allocate by itself. It dlopens `libgbm_mesa_wrapper.so` and calls
`get_gbm_ops()` (`gbm_mesa_driver/gbm_mesa_internals.cpp:231-245`). This
directory builds that library. It allocates **inside the calling process
through NVIDIA's own Vulkan ICD**, so nothing goes over a socket to the host.

The baseline (sibling repo `waydroid-nvidia`, `src/minigbm-vtest/vtest_wrapper.c`) replaces
the same library and forwards every allocation to a host process. This one
does the allocation work itself (exportable `VkImage` + DRM modifier, the
logic of the baseline's host-side `vtest_gpu_alloc.c`) in the guest process.

## How it is loaded

- minigbm: `dlopen("libgbm_mesa_wrapper.so")` → `get_gbm_ops()`. That is the
  only exported symbol, and the library is built `-fvisibility=hidden`.
- The ICD: `dlopen("libEGL_nvidia.so.0", RTLD_NOW|RTLD_LOCAL)` by soname, then
  `vk_icdNegotiateLoaderICDInterfaceVersion` (v5) and
  `vk_icdGetInstanceProcAddr`. There is no libvulkan and no HAL in between. In
  the guest the soname resolves from the vendor and sphal namespaces
  (`/vendor_extra/lib64`, see `runbooks/native-mode.md`). On the host it is
  `/usr/lib/libEGL_nvidia.so.0`. The ICD and the libnvglibc shims are
  dlopened, never `DT_NEEDED`.
- The Vulkan device is created on the **first `alloc` or the first CPU lock of a
  block-linear buffer**, not in `dev_create`. The setup is thread-safe and runs
  once per process. minigbm also calls `dev_create` in every mapper (app)
  process that locks a CPU buffer (`gbm_mesa_internals.cpp:413, 453`); a
  process that only locks LINEAR buffers needs only `mmap` and never loads the
  NVIDIA driver on our behalf. A failed setup is sticky and is reported again,
  with its remedy, on every later `alloc` or block-linear lock.
- minigbm's own gbm_mesa core (`libminigbm_gralloc_gbm_mesa.so`, 64- and 32-bit) is
  rebuilt from the pinned minigbm with six patches (`minigbm/`: 0001 "Layout rule",
  0002 lock access flags, see "Lock/unlock", 0003 FP16/1010102 render targets, see "Format whitelist",
  0004 NV12/P010 decoder output layout, 0005 wrapper import on the first CPU map, so any process
  can lock a buffer allocated without CPU usage, e.g. CCodec copying decoder output; host test
  `make import-lock-test`, guest test `test/guest-import-lock.c`, 0006 CPU-cached memory for buffers the
  CPU reads often, see "Memory placement") and bound over the image's files; the
  mapper and allocator binaries stay the image's.
- Exports: `get_gbm_ops` (minigbm's ABI), `get_gbm_map_flags_ops` (the optional
  map/unmap-with-access-flags ABI our patch 0002 adds, `include/gbm_mesa_wrapper_map_flags.h`,
  a copy of the patched tree's header) and `get_gbm_layout_ops` (patch 0004,
  `include/gbm_mesa_wrapper_layout.h`): `plane_layout()` places the planes of a NV12/P010 buffer,
  which minigbm lays out on its R8 carrier, where the driver can bind each plane (the alignment
  and size of that plane of a DISJOINT image of the same pitches) and returns the buffer size.
  NVIDIA 615 binds plane by plane with 64 KiB granularity (a CbCr plane off a page samples zeros,
  the image needs more than the packed layout); NVK's rule leaves the packed layout unchanged.
  And `get_gbm_alloc_flags_ops` (patch 0006, `include/gbm_mesa_wrapper_alloc_flags.h`): `alloc()` with
  `GBM_ALLOC_FLAG_CPU_READ_OFTEN`, see "Memory placement".
- The library is linked `-z nodelete`. minigbm `dlclose()`s the wrapper when
  its driver object dies, and the process-wide `VkDevice` must not be
  unloaded while buffers are alive.

## GPU selection

The GPU is the physical device with `vendorID == 0x10de`. minigbm passes the
first `/dev/dri/renderD*` it globbed, from any vendor. With one NVIDIA GPU
that fd is ignored. With several, the backend picks the one whose
`VK_EXT_physical_device_drm` render major:minor equals the fd's `st_rdev`,
which compares device numbers, not paths. If none matches, it fails and names
the remedy: set `gralloc.gbm.device`, which minigbm honours.

The device enables exactly `VK_KHR_external_memory_fd`,
`VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier` and
`VK_KHR_image_format_list`, with one queue. `VK_KHR_format_feature_flags2`
must be supported, because `VkDrmFormatModifierPropertiesList2EXT` is queried
through it, but it is not enabled.

## Format whitelist (`get_gbm_format`)

This is the same list and mapping as the baseline (`vtest_get_gbm_format`,
`drm_format_to_vk`):

| DRM | VkFormat | sRGB view |
|---|---|---|
| R8 | R8_UNORM | |
| RGB565 | R5G6B5_UNORM_PACK16 | |
| XRGB8888, ARGB8888 | B8G8R8A8_UNORM | B8G8R8A8_SRGB |
| XBGR8888, ABGR8888 | R8G8B8A8_UNORM | R8G8B8A8_SRGB |
| ABGR2101010 | A2B10G10R10_UNORM_PACK32 | |
| ABGR16161616F | R16G16B16A16_SFLOAT | |

minigbm's gbm_mesa glue registers render/scanout combinations only for the 8888 formats and RGB565; patch
`minigbm/0003-gbm_mesa-FP16-and-1010102-render-targets-from-the-wrapper.patch` (2026-10-11) adds ABGR16161616F
(`HAL_PIXEL_FORMAT_RGBA_FP16`) and ABGR2101010 (`RGBA_1010102`), allocated only by a wrapper whose `get_gbm_format`
accepts them (else a loud EINVAL, never the R8 carrier). Without it the allocator refused FP16 window buffers ("Unsupported
combination ... usage 0xb00") and `eglCreateWindowSurface` on NVIDIA's EGL_EXT_pixel_format_float configs failed with
EGL_BAD_NATIVE_WINDOW (dEQP-EGL `wide_color.window_fp16_*`; pass with the patch).

Every other format returns 0. That is minigbm's contract
(`gbm_mesa_internals.cpp:326, 342-364`): minigbm then lays the buffer out
itself and asks us for a 4096-wide LINEAR R8 carrier with `force_linear`.

Formats that have an sRGB twin are allocated `MUTABLE_FORMAT` with a
`VkImageFormatListCreateInfo{UNORM, SRGB}`. Importers such as ANGLE and the
HAL create sRGB views, and this keeps that use in spec.

## Layout rule: LINEAR only for CPU buffers the GPU does not render into

Changed in G7d: before, every buffer with CPU usage was LINEAR. NVIDIA's EGL
imports LINEAR dma-bufs external-only (`eglQueryDmaBufModifiersEXT` external_only=1; a LINEAR EGLImage on
`GL_TEXTURE_2D` or a renderbuffer gives GL_INVALID_OPERATION, host-verified on stock EGL), so SurfaceFlinger
could not render its screenshot buffer (usage 0xb33 = SW_READ/WRITE_OFTEN | HW_RENDER | HW_TEXTURE |
HW_COMPOSER, SurfaceFlinger.cpp:6611-6616) on NVIDIA's GLES and aborted.

| Gralloc usage (minigbm `BO_USE_*`) | minigbm passes (with our patch) | Layout | CPU lock |
|---|---|---|---|
| any CPU bit (`SW_READ_*`/`SW_WRITE_*`) **and** `HW_RENDER`/`HW_2D` (`BO_USE_RENDERING`) or `FRONT_BUFFER` (`BO_USE_FRONT_RENDERING`) | `needs_map_stride`, not `force_linear` | block-linear | GPU copy through a linear staging buffer |
| CPU bits, no GPU render usage (e.g. `HW_TEXTURE`, `HW_COMPOSER`, video/camera) | `needs_map_stride` + `force_linear` | LINEAR | direct `mmap` |
| no CPU bits | neither | block-linear | none (minigbm never maps it) |
| a format outside the whitelist (YUV, BLOB, ...): minigbm's R8 carrier | `force_linear` (gbm_mesa_internals.cpp:350) | LINEAR | direct `mmap` |

Why "render" is the dividing line: a buffer the GPU only samples works LINEAR, because SurfaceFlinger/Skia bind
every non-output AHB as `GL_TEXTURE_EXTERNAL_OES` (`GrAHardwareBufferUtils::GetBackendFormat` returns the external
target for every GL format, android13 Skia :70-95; RenderEngine creates input-layer textures with
`isRenderable=false`, SkiaGLRenderEngine.cpp:1047, and only the output buffer with `true`, :793), which NVIDIA
supports for LINEAR, and Vulkan imports LINEAR fine. Only a colour attachment needs a non-LINEAR layout.

Measured usages on this image (`dumpsys SurfaceFlinger`, "Imported gralloc buffers"): screenshot 0xb33 (now
block-linear), app/system windows 0xb00 and FramebufferSurface 0x1b00 (no CPU: block-linear as before), the
software video decoder's window buffers in Gallery 0x920 = SW_WRITE_OFTEN | HW_TEXTURE | HW_COMPOSER, RGB565
(LINEAR, direct mmap: the P4 video path is unchanged, no copy per frame), a 29x37 `bbq-adapter` buffer 0x933
(LINEAR).

The decision needs the usage, which the wrapper ABI does not carry: `alloc_args` has only `force_linear`,
`needs_map_stride`, `use_scanout` (`include/gbm_mesa_wrapper.h`), and stock minigbm sets the first two together
for every CPU buffer (gbm_mesa_internals.cpp:327-328). The 0xb33 screenshot buffer and the 0x920 video buffer
look identical to the backend (CPU + scanout). So the decision is made where minigbm makes it, in a small
pinned patch to minigbm's gbm_mesa glue (`minigbm/0001-gbm_mesa-CPU-usage-does-not-force-LINEAR-on-render-targets.patch`):
`force_linear = SW && !(RENDERING|FRONT_RENDERING)`, `needs_map_stride = SW`. The backend then answers:
LINEAR exactly when `force_linear`; with an unpatched minigbm (both flags together) it behaves as before.
`needs_map_stride` without `force_linear` is the contract Mesa's own `gbm_bo_map` serves for tiled buffers.

- **Non-LINEAR buffers** get NVIDIA block-linear modifiers only. The candidates are the device's single-plane,
  non-LINEAR modifiers whose features include COLOR_ATTACHMENT, SAMPLED, TRANSFER_SRC and TRANSFER_DST, kept only
  if `vkGetPhysicalDeviceImageFormatProperties2` accepts them as exportable dma-bufs at this size. The driver
  picks one from the list. The compositor (KWin, via the hwcomposer) can only display block-linear dma-bufs.
- **LINEAR buffers** get `DRM_FORMAT_MOD_LINEAR` only. NVIDIA's LINEAR modifier does not list COLOR_ATTACHMENT (it
  lists `LINEAR_COLOR_ATTACHMENT_NV`, `VK_NV_linear_color_attachment`), so COLOR_ATTACHMENT and STORAGE usage are
  requested only where every candidate supports them.
- If no modifier qualifies, the allocation fails and the log names the device modifiers. Nothing substitutes
  another layout.
- Memory is always a dedicated allocation (block-linear reports DEDICATED_ONLY). Block-linear buffers use a
  DEVICE_LOCAL type; LINEAR buffers see "Memory placement".
- After export, the `VkImage` is destroyed and the `VkDeviceMemory` freed. The dma-buf fd keeps the allocation
  alive, which `test/alloc-test` verifies.
- Results: `out_fd` is the dma-buf (`O_CLOEXEC`), `out_modifier` the modifier the driver chose, `out_stride` the
  plane-0 `rowPitch` (the offset must be 0).

#### Memory placement

LINEAR buffers get the first memory type the image allows that is HOST_VISIBLE|HOST_COHERENT, or DEVICE_LOCAL
when none is (NVIDIA's dma-buf mmap works for both). On NVK that is VRAM behind the BAR
(DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT); on NVIDIA's own driver uncached system memory. Both are fine for the
CPU writing and the GPU reading, and very slow for CPU reads, because the dma-buf mmap keeps the allocation's
caching (open-gpu-kernel-modules `nv-dmabuf.c` `nv_dma_buf_mmap`).

A buffer the CPU reads often and that is not scanned out (minigbm `BO_USE_SW_READ_OFTEN` without
`BO_USE_SCANOUT`, passed by patch 0006 as `GBM_ALLOC_FLAG_CPU_READ_OFTEN`, the rule minigbm's amdgpu backend
uses for GTT USWC) goes to a HOST_VISIBLE|HOST_COHERENT|HOST_CACHED type; there is no other choice for it, and
no such type is a loud failure. Example: hardware decoder output that a Codec2 client without a Surface copies
out on the CPU (the decoder fetches its blocks with CPU_READ).

Measured (RTX 5090, 615.71.09, 4096x1224 R8 LINEAR, full CPU read of the mapping): NVK BAR 22 MB/s, NVIDIA
uncached system memory 92 MB/s, cached system memory 9-11 GB/s on both. Allocation 0.2-0.3 ms either way.
In the guest, ByteBuffer-mode H.264 decode (c2.nvprolapse.avc.decoder) went from 31 to 1065 fps at
1920x816 and from 25 to 971 fps at 1920x1080; AOSP's software decoder from 36 to 491 / 26 to 381 fps.

## Map stride

minigbm reads `map_stride` only for CPU buffers and falls back to `strides[0]` when it is 0 (drv.c:734-738); the
result becomes the handle's `pixel_stride`, computed once in the allocator, the stride every CPU user of a lock
assumes. Mapper processes never see `map_stride` (`gbm_mesa_bo_import` leaves it 0); they use `pixel_stride` and
`strides[0]` (gralloc4 plane layouts).
- LINEAR: `out_map_stride = 0` ("same as stride"); the mmap is direct, so the CPU stride is `rowPitch` =
  `strides[0]`. A non-zero value would be wrong on minigbm's R8-carrier path, where `strides[0]` is minigbm's own
  layout of the real format.
- block-linear + `needs_map_stride`: `out_map_stride = rowPitch`, and the staging copy a lock returns has rows of
  exactly `rowPitch` bytes (the importer's stride, which minigbm passes from `strides[0]`). So `map_stride`,
  `pixel_stride`, `strides[0]` and the staging pitch are equal by construction, in the allocator and in every
  mapper. Alloc fails loudly unless `rowPitch` is a whole number of pixels >= width x bpp (measured: NVIDIA pads
  block-linear rows to 64 bytes, e.g. 1001x37 RGBA8 -> 4032, 333x17 RGB565 -> 704, 257x129 FP16 -> 2112).

## Lock/unlock of block-linear buffers (GPU copy through linear staging)

The design of minigbm's amdgpu backend (amdgpu.c `amdgpu_map_bo`/`amdgpu_unmap_bo`: an SDMA copy to and from a
linear GTT buffer), done with NVIDIA's Vulkan ICD in the locking process:

- **map**: import the dma-buf as a `VkImage` with its explicit modifier and stride (dedicated import), copy it into
  a linear staging `VkBuffer` (`bufferRowLength` = stride / bpp), wait for the copy on a `VkFence`, return the
  staging pointer. The image is acquired from `VK_QUEUE_FAMILY_EXTERNAL` in `GENERAL` (contents preserved) and
  released back the same way; a buffer barrier makes the copy visible to the host.
- **unmap**: copy the staging buffer back into the image, wait, return, **only when the mapping had WRITE
  access**. A read-only lock never writes the buffer: a GPU write that lands while it is held survives, and CPU
  writes into a read view are dropped.
- **Access flags** (changed 2026-10-11): `gbm_ops.map/unmap` carry none, so minigbm patch
  `minigbm/0002-gbm_mesa-pass-the-lock-access-flags-to-the-wrapper.patch` passes `BO_MAP_READ`/`BO_MAP_WRITE`
  (minigbm derives them from the lock usage, `cros_gralloc_convert_map_usage`) through the new optional
  `get_gbm_map_flags_ops()` (version 1; a wrapper without it keeps the old ABI, one with another version fails
  driver init loudly). Rules: READ: copy in at map, no copy-back. WRITE (with or without READ): copy in at map
  too, copy back at unmap. A WRITE-only lock must start from the buffer's contents because the copy-back covers
  the whole buffer and the contract keeps what the lock does not write: "The content of the buffer outside of the
  specified area is NOT modified by this call" (hardware/libhardware android13 bbd07bd2 `gralloc.h:198-202`,
  `lock()`; the same in `AHardwareBuffer_lock`, frameworks/native 2fe7eddf
  `libs/nativewindow/include/android/hardware_buffer.h:432-435`); IMapper 4.0 `lock()` (hardware/interfaces
  android13-release `graphics/mapper/4.0/IMapper.hal:190-251`) makes only content *outside* the access region
  unreliable for reading and never declares a write lock's contents undefined. **Nested locks**:
  `cros_gralloc_buffer::lock` reuses the held mapping; when the nested lock adds access (WRITE while READ is held)
  the patch widens the mapping's flags (`drv_bo_widen_map_flags`) and unmap gets the widened flags. The plain
  `gbm_ops.map/unmap` (an unpatched core) are READ|WRITE. Both the 64- and 32-bit cores are rebuilt with the patch,
  so every guest process passes flags. LINEAR buffers bracket the mmap with `DMA_BUF_IOCTL_SYNC` in the lock's
  direction(s).
- **Staging memory** is a HOST_VISIBLE|HOST_CACHED system-memory type (coherent preferred, else flush/invalidate
  around the copies); none means a loud failure, never an uncached or BAR mapping. The tiled VRAM itself is never
  mapped for the CPU (CPU reads of BAR are unusable, measured below).
- **Synchronisation, all explicit.** NVIDIA takes no part in dma-buf implicit sync (`../egl/FENCE-CONTRACT.md`
  §2.5), so nothing relies on it. Before map: minigbm's lock waits for the acquire fence on the CPU before it calls
  `bo_map` (cros_gralloc_driver.cc:446-450), so the producer's work is complete when our copy is submitted. After
  unmap: minigbm's unlock always returns release fence -1 (cros_gralloc_driver.cc:493, "may access immediately"),
  so the write-back completes before `unmap` returns (CPU wait on the fence). A lock without a fence is ordered
  against nothing, as the gralloc contract says.
- **D1 cannot apply.** The device enables neither `VK_KHR_external_semaphore_fd` nor `VK_KHR_external_fence_fd`,
  so it can neither import a SYNC_FD wait (the only operation that advances a queue timeline without starting it)
  nor export a sync_fd (the only place a poisoned value is observed); its copies signal a plain `VkFence`. That is
  the "Which devices" rule of `../hal/AHB-CONTRACT.md` §8.2, so no queue-timeline start is needed.
- **Processes that lock without having allocated** (every mapper: apps, SurfaceFlinger, screencap, 32-bit
  processes): the process-wide Vulkan device is created lazily on the first lock of a block-linear buffer (the
  same once-only, mutex-guarded setup as `alloc`; GPU chosen by the importing gbm device's render node). A failure
  is sticky and named in the log; map returns `MAP_FAILED` with ENODEV. Measured first lock: 120-170 ms on the host
  (ICD load + device), ~66 ms in the guest (`../egl` `sfimage --cpu-check`, ICD already loaded for EGL). A process
  that only locks LINEAR buffers still never loads the driver on our behalf.
- **Lifetime**: one copy context per process (command pool, command buffer, fence; made on first use, kept with
  the process-wide device). Per buffer, the imported image and the staging buffer are made at the first lock and
  kept until `free` (the per-buffer staging cache: one buffer's size of system memory per CPU-locked block-linear
  buffer in that process). A copy that does not complete within 10 s marks the copy context unusable (its command
  buffer may still be in flight) and every later lock fails with that reason. A second `map` of the same bo
  without `unmap` fails with EBUSY (minigbm maps a bo once and refcounts, drv.c:498-523). An `unmap` that cannot
  write back logs that the CPU's changes were not written back.
- **Cost** (host, RTX 5090, 2560x1412 RGBA8 = 13.8 MiB, `test/lock-test` timing, median of 30): lock 0.56 ms, unlock
  0.60 ms, CPU read of the staging copy 0.36 ms. A ReBAR (DEVICE_LOCAL|HOST_VISIBLE) mapping of the same size: CPU
  write 0.64 ms, CPU read 1207 ms. CPU detiling straight into VRAM would replace the 0.6 ms write-back with at least
  the 0.64 ms BAR write plus the detiling and cannot serve reads: not worth building.

## import / map (LINEAR)

- `import` duplicates the fd (`F_DUPFD_CLOEXEC`), records the size with `lseek(SEEK_END)`, the modifier, stride and
  the importing device's render node. `free` closes the duplicate (and releases a block-linear buffer's Vulkan
  objects).
- `map` of a LINEAR buffer calls `mmap(PROT_READ|PROT_WRITE, MAP_SHARED)` on the whole dma-buf, bracketed by
  `DMA_BUF_IOCTL_SYNC` START|RW, and `unmap` issues END|RW and then `munmap`. minigbm's gbm_mesa backend has no
  `bo_invalidate` or `bo_flush`, and it calls map once per lock and unmap once per unlock.

## Logging

On Android the backend logs with `__android_log_print` under the tag
`gralloc.nvidia`. On the host it writes to stderr. `NVGRALLOC_TRACE=1` traces
every operation (the result of each alloc includes its modifier, stride, size,
memory type and usage). On Android the property `debug.nvgralloc.trace=1` does
the same, because the allocator service is started by init.

## 32-bit build (mapper only)

The image's 32-bit processes (mediaserver and its software codecs, 32-bit
apps) run minigbm's 32-bit mapper, which `dlopen`s `/vendor/lib/libgbm_mesa_wrapper.so`
to `import` and `map` buffers in-process. `make android32` builds this library
for i686; native mode binds it over the vendor.img file. Its LINEAR import/map
path is plain `fcntl`/`lseek`/`mmap`/`DMA_BUF_IOCTL_SYNC` on the dma-buf and
loads no NVIDIA library; a lock of a block-linear buffer brings up the 32-bit
ICD (P6 closure in `/vendor_extra/lib`) for the staging copy, tested on the
host (`out/host32`) and in the guest (`sfimage32 --cpu-check`). Allocation
happens only in the 64-bit allocator service; the 32-bit
`libminigbm_gralloc_gbm_mesa.so` is still rebuilt with our patches (bound over
`/vendor/lib/`), so 32-bit locks pass their access flags (`sfimage32 --cpu-check`). Measured 2026-10-10 without this
bind: mediaserver (x86) maps NVIDIA dma-bufs through the stock Mesa wrapper on
`gralloc.gbm.device` and dies with SIGBUS in `SoftwareRenderer::render`;
Gallery reports "Can't play this video".

## Build and test

```
make host        # out/host/libgbm_mesa_wrapper.so (gcc, /usr/include/vulkan, libdrm drm_fourcc.h)
make android     # out/android/libgbm_mesa_wrapper.so (NDK r30 x86_64-linux-android33-clang)
make android32   # out/android32/libgbm_mesa_wrapper.so (NDK r30 i686-linux-android33-clang)
make check       # both Android builds: ELF class, exports == the four get_gbm_*ops, SONAME, DT_NEEDED ⊆ {libc,libdl,liblog,libm}.so, NODELETE
make check-header  # include/gbm_mesa_wrapper.h == build/minigbm (16844ba)
make test        # host: alloc-test, lock-test (64-bit) and lock-test (32-bit build, out/host32), ~30 s
make minigbm     # out/android{,32}/libminigbm_gralloc_gbm_mesa.so: minigbm 16844ba + minigbm/*.patch (x86_64, i686)
make check-minigbm  # both ABIs: exports cover every symbol the image's mapper/allocator/gralloc0 import from it,
                 # -z defs against the guest's own libs, no std::__ndk1
make check-header   # also: include/gbm_mesa_wrapper_map_flags.h == the patched tree's copy
make minigbm-pins   # the pinned header trees (below) are at their recorded commits
```

`libminigbm_gralloc_gbm_mesa.so` is C and C++ whose C++ API (`cros_gralloc_driver`,
`std::function`) the image's own mapper and allocator call, so it is compiled with the NDK
clang against the platform libc++ headers (AOSP `android13-release` external/libcxx
d387ee3, `std::__1`), never the NDK's `std::__ndk1`, with Soong's flags for the module
(`-fno-exceptions`, `DRV_EXTERNAL`, c++17), and linked `-z defs` against link-time copies of the
guest's libraries read from the images with debugfs (VNDK `libc++`/`libcutils`, LLNDK
`liblog`/`libsync`/`libnativewindow`, vendor `libdrm`; never shipped). Header trees, pinned
sparse clones in `build/`: AOSP `android13-release` system/core 83366e7e (libcutils,
libsystem, libsync), system/logging 6bfb380b (liblog), hardware/libhardware bbd07bd2,
frameworks/native 2fe7eddf (nativewindow, nativebase), libdrm 2.4.134, Mesa ddaf85e3 (mesa-26.2.5)
(`gbm.h` only). Measured: the image's library exports 162 symbols, the image's consumers
import 37 of them, ours exports all 37.

`include/gbm_mesa_wrapper.h` is a pinned copy of the minigbm ABI header. Its
provenance is in the file.

`test/alloc-test` opens the render node whose sysfs PCI vendor is 0x10de and
drives the library only through `get_gbm_ops()`. It verifies every buffer on
the GPU through a separate `VkDevice` created via the system Vulkan loader,
importing with `VkImageDrmFormatModifierExplicitCreateInfoEXT` from the
reported stride and modifier. It covers:

- concurrent first allocations;
- a block-linear 1280×800 buffer, cleared and pattern-uploaded through one
  import and read back through another;
- LINEAR CPU buffers (64×64 RGB565, 1001×37 ABGR8888, 333×17 RGB565): CPU
  writes are verified on the GPU and GPU clears through `map`;
- the same for buffers allocated with `GBM_ALLOC_FLAG_CPU_READ_OFTEN` (cached memory; 1001×37 ABGR8888,
  333×17 RGB565, the 4096×574 R8 carrier of a 1920×816 YV12 decoder block), and an unknown flag refused;
- `map` of the block-linear buffer from above through the staging copy (pattern exact);
- the whitelist, every whitelisted format in both layouts, and the R8 carrier;
- 200 alloc/free cycles of 1920×1080. These check fds, VmRSS, the GPU's
  `memory.used` and system MemAvailable, and each measurement is first shown
  to see live buffers.

`test/lock-test` (64-bit and, from `out/host32`, 32-bit) covers the access flags (per format/size: a READ lock
keeps a GPU write made by an independent Vulkan device between its map and unmap and drops CPU scribbles; a
WRITE-only lock of a rect keeps everything else; a READ map unmapped with READ|WRITE is written back; controls:
READ|WRITE and the plain `gbm_ops` map/unmap overwrite that same GPU write, so the test can see a copy-back; bad
flags give EINVAL; a LINEAR buffer with WRITE then READ locks) and CPU-mapped render targets:
a buffer allocated as minigbm passes usage 0xb33 (`needs_map_stride`, not `force_linear`,
scanout) must be block-linear with `map_stride == stride`; then CPU write through
map/unmap (checked on the GPU through an independent Vulkan import), GL rendering into it
(stock EGL/GLES on the NVIDIA device: dma-buf EGLImage as `GL_TEXTURE_2D` FBO, scissored
clear + shader draw) read back through map, Vulkan rendering (render pass,
`vkCmdClearAttachments`) read back through map, and a separate exec'd process that never
allocated (lazy ICD bring-up) checking all of it and CPU-writing a rectangle, which the
parent checks on the GPU and through `glReadPixels`. Exact pixels; ABGR8888 2560x1412,
1001x37, 1x1; RGB565 1280x720, 333x17; ABGR16161616F 257x129, 64x64. Then lock/unlock
timings at 2560x1412 with a ReBAR comparison, and a leak check (200 x alloc/import/map/
write/unmap/free of 1080p + 200 locks of one buffer: fds, VmRSS, GPU memory.used,
MemAvailable). Against the pre-2026-10-11 backend it reproduces the G7d failure
(LINEAR, `glEGLImageTargetTexture2DOES` 0x502).

Each test exits 0 only when every check passes.
