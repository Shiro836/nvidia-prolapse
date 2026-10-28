# Offline Mesa transport regression

Run from the repository root after configuring the pinned host Mesa build:

```sh
python3 mesa/tests/run-pdev-preflight.py
```

The harness compiles the canonical adapter and selected actual pinned Mesa
runtime sources. Fake RM/DRM calls check arguments; unexpected device access
and DRM syncobj create/wait/destroy operations abort. It requires no GPU and
does not establish runtime support on any driver.

Coverage includes selected NVIDIA node checks, partial initialization cleanup,
absent legacy syncobj capabilities, wait deadlines/ANY/ALL/pending, a simulated
6.251-second valid dependency, maintenance and submit-thread loss, ring retries,
sync reset/move, FD forwarding, and destruction without infinite pre-drain.
`submit-on-demand.inc` drives the patched `vk_queue.c` in THREADED_ON_DEMAND
mode with a minimal timeline sync: `driver_submit` asserts that every wait it
receives is already pending (fails under the former IMMEDIATE mode), a
same-thread wait-before-signal and the work behind it wait in the submit
thread until the host signal, order is kept, and a drained queue submits on
the caller again, and a binary signal of a deferred submission is reserved so
a binary wait submitted behind it does not block vkQueueSubmit.
VA teardown hands every binding to one `nv_rm_va_free` (no synchronous unbind):
a deferred release keeps the range reserved until the backend's callback runs,
a failed call publishes loss and retains the range, and an application rebind
over a live binding unbinds immediately.

Void sync destruction uses the historical `nv_rm_sync_destroy_deferred` ABI
(immediate caller-side disposal since the caller-side backend); the synchronous
entrypoint is a harness tripwire. Cleanup after loss is checked, and an injected
disposal failure must log a diagnostic, publish Mesa/Rust loss and wake the
eventfd through the installed nonblocking loss hook. Rust CPU and hardware tests
separately exercise slot lifetime across retire, detach-on-wait, reset, move
and import.

`submit-fusion.inc` exercises the actual adapter's combined push/wait/signal
call. Exact mock expectations cover empty, signal-only, wait-only and
0/1/1000/1001/2000-entry submissions, continuation boundaries, first/final
ring-pressure retries, failed waits, rejected batches, and an accepted prefix
followed by failure. Signals appear only on the final chunk and waits only on
the first accepted chunk. Invalid signal types, allocation failure, and an
oversized continuation chain reject before any submission. Bind-only contexts
continue to signal through the CPU path. This checks C-to-Rust call contracts,
not GPU execution or Rust ownership internals.

Loss notification checks invoke the actual Mesa device and queue setters through
the adapter's installed callback. They require loss publication before the wake,
exercise the callback under a submit mutex, and verify hook removal before the
backend is freed. Scripted waits require one blocking backend call, reject status
polling, preserve caller deadlines, and give loss precedence over racing success.
The threaded loss tests use real eventfd wakeups; no GPU or device nodes are used.

`sync-properties.inc` invokes the actual public Mesa fence/semaphore property
queries. It asserts these masks, rather than deriving them from a second model:

| Object | Requested handle | Compatible / export-from-imported | Features |
| --- | --- | --- | --- |
| Fence | OPAQUE_FD | OPAQUE_FD only | Import + export |
| Fence | SYNC_FD | SYNC_FD only | Import + export |
| Binary semaphore | OPAQUE_FD | OPAQUE_FD only | Import + export |
| Binary semaphore | SYNC_FD | SYNC_FD only | Import + export |
| Timeline semaphore | OPAQUE_FD | OPAQUE_FD only | Import + export |
| Timeline semaphore | SYNC_FD | None | None |

`sync-timeline-mode.c` compiles the pinned `vk_device.c` and exposes its static
selector only to this test. It verifies that the two adapter types provide
exactly one timeline implementation and select native timeline mode. Opaque
sharing and sync-file copy transfer use separate callback sets while retaining
the same RM completion operations. Cross-handle conversion is not advertised.

The runner stores its executable and objects under
`build/gpu-coverage/pdev-preflight/`. Actual GPU probe results are separate.

`pascal-a-clear.c` (build line in its header) compiles the patched
`nvk_pascal_a.h` from `build/mesa-rm-source` and checks the PASCAL_A clear
bracket words `0x800103d8` / `0x800103e0, 0x800003d8` and that only class
0xc097 gets them.

## Perf capture (Rust RM backend; off by default)

`NV_RM_CAPTURE=<dir>` (host) or `setprop debug.nvrm.capture <dir>` + `debug.nvrm.capture.proc <cmdline substring>`
(guest) arms per-submission GPU timestamps (two host semaphore releases per kick) and a VA index in the matching
process; `echo 'ts frames=300' > <dir>/go` (or `push frames=30`, which also copies every pushbuffer) records the next
N presents to `<dir>/cap-<pid>-<dev>-<n>.bin`. Decode with `rm/tools/nvcap.py CAP --driver nvk|nvrm [--json]`
(per-frame kicks, GPFIFO entries, bytes, draws, method histogram, WFIs, invalidates, CE/I2M, cbuf binds/loads,
GPU busy/idle) and `rm/tools/nvcap_timeline.py CAP FRAME N` (per-kick CPU submit vs GPU start/end).

## vk-barrier-order.c (pipeline-barrier ordering: colour -> sampled, compute -> compute, compute -> fragment)

    T=build/barrier-order/test; mkdir -p $T; S=mesa/tests
    for x in QUAD:vert:quad FILL:frag:fill SAMPLE:frag:sample SSBO:frag:ssbo PRODUCE:comp:produce CONSUME:comp:consume; do
      IFS=: read D ST NM <<< "$x"
      glslangValidator -V -S $ST -D$D $S/vk-barrier-order.glsl --vn vk_barrier_order_$NM -o $T/vk-barrier-order.$NM.h
    done
    cc -std=c11 -Wall -Wextra -Werror -I$T $S/vk-barrier-order.c -lvulkan -o $T/vk-barrier-order
    E="VK_DRIVER_FILES=$PWD/build/nvk-host/native-rm-icd.json VK_LOADER_LAYERS_DISABLE=~implicit~"
    for spin in 0 300 2000 20000; do env $E $T/vk-barrier-order 40 $spin; done
    env $E $T/vk-barrier-order 5 2000 unordered                            # negative control: must FAIL
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json $T/vk-barrier-order 10 2000   # stock reference
    NVK_DEBUG=push_dump env $E $T/vk-barrier-order 1 2000 2>&1 | grep -c PIXEL_SHADER_BARRIER   # 12 on Blackwell B

Per round, with a producer that spins SPIN iterations before writing: colour (draw into C, barrier
COLOR_ATTACHMENT_OUTPUT/WRITE -> FRAGMENT_SHADER/SAMPLED_READ as image barrier, memory barrier in GENERAL, and
image barrier with BY_REGION, then a draw sampling C), compute (dispatch writes A, CS storage barrier, dispatch
reads A), cs-gfx (dispatch writes A, barrier to FRAGMENT_SHADER/STORAGE_READ, draw reads A per pixel). 12 rounds per
command buffer, every result compared exactly. PASS = 0 mismatches. `unordered` replaces the three barriers by
TOP -> BOTTOM without accesses: colour and compute then fail on both drivers (stale previous-round values);
cs-gfx passes even unordered on this host (the compute -> 3D switch orders it), so it only guards the
CS -> FS case against corruption, not ordering. On Blackwell B the colour barrier is PIXEL_SHADER_BARRIER
(words 0x80000378, 0x800003dd = method 0x0f74, then INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI) instead of
WAIT_FOR_IDLE, as NVIDIA's driver emits it (libnvidia-eglcore.so 615.78.08 E 0xc65e10 / E 0xc64090).
10-24 (RTX 5090): 480 rounds per case at spin 0 / 300 / 2000 / 20000, all PASS; stock PASS; unordered FAIL on
both (colour 60/60 and compute 57-60/60 bad rounds). 0 Xid.

## vk-cs-join-reuse.c (compute -> compute barriers on re-executed and split command buffers)

    T=build/cs-join-reuse; mkdir -p $T; S=mesa/tests
    glslangValidator -V -S comp $S/vk-cs-join-reuse.glsl --vn vk_cs_join_reuse_inc -o $T/vk-cs-join-reuse.inc.h
    cc -std=c11 -Wall -Wextra -Werror -I$T $S/vk-cs-join-reuse.c -lvulkan -o $T/vk-cs-join-reuse
    E="VK_DRIVER_FILES=$PWD/build/nvk-host/native-rm-icd.json VK_LOADER_LAYERS_DISABLE=~implicit~"
    for spin in 0 2000 20000; do for m in reuse double clear split; do env $E $T/vk-cs-join-reuse 20 $spin $m; done; done
    env $E $T/vk-cs-join-reuse 5 2000 unordered                     # negative control: must FAIL
    NVK_DEBUG=push_dump env $E $T/vk-cs-join-reuse 1 2000 2>&1 | grep -c SET_INLINE_QMD_ADDRESS_A   # 40 on Blackwell B

16 epochs x 3 dispatches, each dispatch a read-spin-write increment of its slice of a host-visible
buffer, a COMPUTE storage write/read -> COMPUTE storage read/write memory barrier after every epoch;
every element must equal the epoch count after each execution. `reuse` records once without
ONE_TIME_SUBMIT and submits 20 times; `double` submits the SIMULTANEOUS_USE command buffer twice per
vkQueueSubmit; `clear` adds a vkCmdClearColorImage (3D work) inside every epoch; `split` puts epoch 0
in a first command buffer and starts the second with a one-workgroup dispatch and the barrier, so
the barrier has to order the first command buffer's work too. From AMPERE_COMPUTE_B to Blackwell B
(not GA100) these barriers become join / link QMDs loaded inline (SET_INLINE_QMD_ADDRESS_A, with
INLINE_SIZE_INLINE_384 for the 384-byte V04 / V05 QMDs, then LOAD_INLINE_QMD_DATA) and SEND_SIGNALING_PCAS2_B DECREMENT_DEPENDENCE, as NVIDIA's driver orders
compute work (libnvidia-eglcore.so 615.78.08 E 0xd35a90 / E 0xd345d0 / E 0xd33fb0 / E 0xd36060);
the first barrier of a command buffer and every barrier after 0x24 chained dispatches + QMDs keep
the method path.
10-24 (RTX 5090, Blackwell B): reuse / double / clear / split at spin 0, 2000, 20000, 20 executions each, all
PASS; unordered FAIL (5/5 executions). Push dump of one execution: 40 inline QMDs, 76 dependency decrements,
3 INVALIDATE_SHADER_CACHES{FLUSH_DATA} + 1 WAIT_FOR_IDLE (first barrier, two chain-limit fallbacks, the
pending last barrier at vkEndCommandBuffer) where NVK without the join QMDs emits 16 flush-form invalidates.
vk-barrier-order on the same build: 40 rounds at spin 0 / 300 / 2000 / 20000 PASS, unordered FAIL; its push
dump goes from 90 to 78 WAIT_FOR_IDLE and 37 to 25 flush-form invalidates (the compute -> compute barrier of
each round). 0 Xid.
10-25 (RTX 5090, under load, the class gate widened to c7c0..cec0): reuse / double / clear / split at spin 0,
2000, 20000 PASS, unordered FAIL, 40 inline QMDs; vk-barrier-order 4 spins PASS, unordered FAIL. 0 Xid.
10-25, driver 580.178.04: RTX 3060 (GA106, AMPERE_COMPUTE_B) and RTX 4060 (AD107, ADA_COMPUTE_A): every
mode and spin PASS, 40 inline QMDs (256-byte V03, no INLINE_384), 76 dependency decrements, 1 WAIT_FOR_IDLE
per execution; the unordered control passes on both GPUs with NVK (it fails with NVIDIA's driver there),
so on these GPUs it does not show that the barriers do the ordering. RTX 2060 (TU106): join path off as
intended (0 inline QMDs), every mode PASS. vk-barrier-order: 4 spins PASS on all three, unordered FAIL.
0 Xid.

## nak-qmd-membar.c (CWD_MEMBAR_TYPE in the QMDs of the built NVK)

    M=build/mesa-rm-source; B=build/nvk-host; S=mesa/tests
    cc -std=gnu11 -DHAVE_FUNC_ATTRIBUTE_PACKED -DHAVE_ENDIAN_H -DUTIL_ARCH_LITTLE_ENDIAN=1 -DUTIL_ARCH_BIG_ENDIAN=0 \
       -I$M/src/nouveau/compiler -I$M/src/nouveau/headers -I$M/src/compiler -I$M/src/compiler/nir -I$B/src/compiler/nir \
       -I$M/include -I$M/src -I$B/src/compiler -I$B/src $S/nak-qmd-membar.c -ldl -o build/nak-qmd-membar
    build/nak-qmd-membar $B/src/nouveau/vulkan/libvulkan_nouveau.so

PASS = CWD_MEMBAR_TYPE 3 (L1_MEMBAR) from Turing to Blackwell B (word 11 for c5c0..c9c0, word 35 for cbc0,
word 19 for cdc0 / cec0) and 0 on Volta (c3c0), as NVIDIA's driver builds its QMDs (libnvidia-eglcore.so
615.78.08 QMD template ELF 0xaa12a0, tails ELF 0xaa1338 / 0xaa14c7 / 0xaa13d3). 10-25: PASS.

## nak-qmd-dependency.c (QMD dependency field setters of the built NVK, every layout they serve)

    M=build/mesa-rm-source; B=build/nvk-host; S=mesa/tests
    cc -std=gnu11 -DHAVE_FUNC_ATTRIBUTE_PACKED -DHAVE_ENDIAN_H -DUTIL_ARCH_LITTLE_ENDIAN=1 -DUTIL_ARCH_BIG_ENDIAN=0 \
       -I$M/src/nouveau/compiler -I$M/src/nouveau/headers -I$M/src/compiler -I$M/src/compiler/nir -I$B/src/compiler/nir \
       -I$M/include -I$M/src -I$B/src/compiler -I$B/src $S/nak-qmd-dependency.c -ldl -o build/nak-qmd-dependency
    build/nak-qmd-dependency $B/src/nouveau/vulkan/libvulkan_nouveau.so

Counter, DEPENDENT_QMD0/1 and invalidates set on a zeroed QMD of c7c0, c9c0 (V03_00), cbc0 (V04_00), cdc0 and
cec0 (V05_00); every word must equal the bits NVIDIA's driver writes for that layout (list in the file header;
the V03 second dependent pointer goes to word 9). PASS = all five classes "ok". 10-25: PASS.

## vk-zcull-depth.c (depth-tested rendering with ZCULL active on the RM backend)

    T=build/zcull/test; mkdir -p $T; S=mesa/tests
    glslangValidator -V $S/vk-zcull-depth.vert --vn vk_zcull_depth_vert -o $T/vk-zcull-depth.vert.h
    glslangValidator -V $S/vk-zcull-depth.frag --vn vk_zcull_depth_frag -o $T/vk-zcull-depth.frag.h
    cc -std=c11 -Wall -Wextra -Werror -I$T $S/vk-zcull-depth.c -lvulkan -o $T/vk-zcull-depth
    VK_DRIVER_FILES=$PWD/build/nvk-host/native-rm-icd.json VK_LOADER_LAYERS_DISABLE=~implicit~ $T/vk-zcull-depth 40
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json $T/vk-zcull-depth 10   # stock reference

300x200 colour + D32_SFLOAT, two depth images per run (alternating frames): depth attachment +
TRANSFER_SRC (NVK allocates a ZCULL save plane, nvk_image.c:1142-1155) and + TRANSFER_DST (no plane,
ZCULL without save/restore). Per frame: clear 1.0 + LESS, LOAD + LESS (LOAD_ZCULL with a plane; one
quad hidden everywhere), readback; clear 0.0 + GREATER, LOAD + GREATER, readback. Every colour and
depth value is compared exactly with a CPU model. PASS = 0 mismatches. ZCULL is in use when the
"plane" depth image is larger than the "no plane" one, and `NVK_DEBUG=push_dump` shows
SET_ZCULL_STORAGE_*, CLEAR_ZCULL_REGION, LOAD_ZCULL and STORE_ZCULL.
2026-10-24, RTX 5090, 615.78.08: NVK with ZCULL info from GR_GET_ZCULL_INFO = PASS 40 frames (plane
image 375,424 B vs 327,680 B; push dump of 2 frames: 6 SET_ZCULL_STORAGE_A, 4 CLEAR_ZCULL_REGION, 2
LOAD_ZCULL, 4 STORE_ZCULL); stock 615.78.08 PASS; NVK without ZCULL info PASS (both images 327,680 B).
RM values on GB202: width/height align 1360/32, pixels per aliquot 10880, aliquots 5120, region
byte multiplier 352, region/subregion header 384/2304, 16 subregions aligned 1360x64; GRAPHICS_ZCULL
context buffer 0x1b9600 B, alignment 0x1000.

## vk-nan-minmax.c (FMax/FMin/FClamp with a NaN operand under ANGLE; GTA SA black car paint)

    T=build/mesa-tests; mkdir -p $T
    glslangValidator -V mesa/tests/vk-nan-minmax.comp \
      --vn vk_nan_minmax_spv -o $T/vk-nan-minmax.spv.h
    cc -std=c11 -Wall -Wextra -Werror -I$T mesa/tests/vk-nan-minmax.c \
      -lvulkan -lm -o $T/vk-nan-minmax
    E="VK_DRIVER_FILES=$PWD/build/nvk-host/native-rm-icd.json VK_LOADER_LAYERS_DISABLE=~implicit~ \
       DRIRC_CONFIGDIR=$PWD/build/mesa-rm-source/src/nouveau/vulkan"
    env $E $T/vk-nan-minmax ANGLE maxnum          # PASS = driconf entry active
    env $E $T/vk-nan-minmax nvk-nan-test report   # other engines keep upstream semantics
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json $T/vk-nan-minmax ANGLE maxnum   # stock reference

The host build reads driconf from files (expat), hence DRIRC_CONFIGDIR; Android builds embed
00-nvk-defaults.conf (`-Dexpat=disabled` -> driconf_static.h). Compute dispatch of
`max(pow(d,10),0)*2`, `clamp(pow(d,10),0,1)`, `min(-pow(d,10),0)` for d = -0.5, 0.5, -2, 0.
2026-10-16, RTX 5090: NVK 26.2.5 unpatched = NaN for max/min at d < 0 (clamp 0); stock 615.71.09 =
0 for every engine name; patched NVK with engine "ANGLE" = PASS, engine "nvk-nan-test" or "ANGLEx" =
NaN (upstream), `nvk_force_nan_preserve_min_max=true` in the environment = PASS for any engine.

## vk-shader-reuse.c (shader heap address reuse; shader publication order)

Each iteration creates a graphics pipeline from one of 8 vertex and 12 fragment shaders (exact
unrolled integer chains of different lengths, generated by `vk-shader-reuse-gen.py`, needs glslc)
through a fresh VkPipelineCache, draws one full-screen triangle into R32_UINT, waits, checks every
pixel against the CPU value and destroys the pipeline and the cache, so the next pipeline's headers
and code land on recycled heap ranges. Stale program headers show as Xid 13 "Shader Program Header"
(VK_ERROR_DEVICE_LOST), stale code as wrong pixels. The test tears everything down at the end
(device destruction included). Host:

    T=build/shader-reuse-test; mkdir -p $T
    python3 mesa/tests/vk-shader-reuse-gen.py $T
    cc -std=c11 -O2 -Wall -Wextra -Werror -I$T \
      mesa/tests/vk-shader-reuse.c -lvulkan -o $T/vk-shader-reuse
    VK_DRIVER_FILES=build/nvk-host/native-rm-icd.json $T/vk-shader-reuse 3000 1        # idle between frames
    VK_DRIVER_FILES=build/nvk-host/native-rm-icd.json $T/vk-shader-reuse 2000 5 8 256  # 8 frames in flight, 256x256

Arguments: ITERATIONS SEED FRAMES SIZE. With FRAMES > 1 a slot's pipeline is destroyed only after its
fence signalled while the other frames still execute, so ranges are recycled under a busy queue.
It guards the recycled-range path; it does not reproduce the rare Xid 13 SPH fault (no deterministic
reproducer exists).

Publication order (`nvk-shader-upload-publish.patch`, `nvk-shader-prefetch-in-range.patch`): capture
the stream and check it with `rm/tools/nvcap_publish.py`:

    mkdir -p $T/cap && echo "push frames=1" > $T/cap/go
    NV_RM_CAPTURE=$T/cap VK_DRIVER_FILES=build/nvk-host/native-rm-icd.json $T/vk-shader-reuse 300 7 8 64
    python3 rm/tools/nvcap_publish.py $T/cap/cap-*.bin

2026-10-19, GB202: with both patches 902 submits, 343 copy-engine copies, 300 instruction
invalidates, 600 program-address uses, 0 errors (every use after its copy, the acquire and the
invalidate; 256 B aligned; prefetch inside the copied range). Without them (same pin): no copies, no
instruction invalidate, 0x80-aligned addresses, 8 of 8 uses flagged.

Android: build the same source with the NDK's `x86_64-linux-android33-clang ... -lvulkan` and run it
in the guest.

`vk-async-compute.c` (build lines in its header, shaders in
`vk-async-compute.glsl`) runs a graphics queue and a compute-only-family queue
of one device on a real GPU: `overlap` times a busy draw and a busy dispatch
alone and together (overlap factor), `pingpong` chains 16 dependent
dispatches across the two queues with a timeline semaphore (in step order and
with the compute steps submitted first) and with binary semaphores, the
compute queue's steps dispatched indirectly, `teardown` destroys the device
with both queues busy, `solo` runs a device with only a compute-family queue.
Every result is checked; exit 77 means the device has no compute-only family.

`vk-compute-queue-compressed.c` (build line in its header) runs only on a compute-only-family queue: BC1 and BC7
2D images (64x64, 4 levels, 3 layers) are filled by vkCmdCopyBufferToImage, copied to a second image by
vkCmdCopyImage and read back by vkCmdCopyImageToBuffer, every block compared. That queue has no copy engine, so
these copies are compute work, and a view that reinterprets a compressed level as blocks gets its layer stride
from that single level: layers past the first need one view per layer. Exit 77 = no compute-only family.
10-25 (RTX 5090): PASS on NVK + Rust RM and on the stock driver. 0 Xid.

## nvrm (gallium GL/GLES) tests

Build nvrm for the host after NVK's host build (`python3 mesa/build.py --gl --target host
--rust-profile release`, see ../ANDROID-GL.md; `./build.sh` does it). Each program's header has its
cases and arguments. Host build and run, from the repository root:

    B=build/nvrm-host
    cc -std=c11 -Wall -Wextra -Werror -Ibuild/mesa-gl-source/include mesa/tests/gl-g0a-clear.c \
       -L$B/src/egl -lEGL -L$B/src/mesa/glapi/es2api -lGLESv2 \
       -Wl,-rpath,$PWD/$B/src/egl:$PWD/$B/src/mesa/glapi/es2api -o gl-g0a-clear
    LIBGL_DRIVERS_PATH=$PWD/$B/src/gallium/targets/dri ./gl-g0a-clear

The loader maps the nvidia-drm render node to nvrm by itself; no driver override is needed. A
reference comes from the same source linked against the host's Mesa through glvnd on llvmpipe
(`cc ... -lEGL -lGLESv2`, run with
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1`);
the `--gl` build has no llvmpipe (LLVM disabled). `NVRM_DEBUG_SHADERS=1` prints st/mesa's NIR and
NAK's assembly.

| Program | Checks |
|---|---|
| `gl-g0a-clear.c` | surfaceless EGL + GLES2 context, glClear readback exact; GL_VENDOR names nvrm |
| `gl-g0b-triangle.c` | NAK-compiled triangle and indexed quad vs a reference image (`--flat`: 0 mismatching pixels; gradients may differ by 1 LSB) |
| `gl-lod-switchover.c` | min/mag switch-over point of explicit-LOD vertex lookups |
| `gl-skia-shaders.c` | SurfaceFlinger RenderEngine (Skia) shader features vs llvmpipe |
| `gl-dmabuf.c` | dma-buf import/export and native fences, NVK as the second party (`-DNVRM_NO_VULKAN` for the EGL-only part) |
| `gl-yuv.c` | NV12/P010 EGLImage import + samplerExternalOES vs a CPU reference |
| `gl-es31-compute.c`, `gl-es31-helper-loads.c`, `gl-cs-sysvals.c` | GLES 3.1 compute, SSBOs, atomics, indirect dispatch, compute system values, helper-invocation loads |
| `gl-es31-b.c` | GLES 3.1 images, memory barriers, no-attachment FBOs, gather, stencil texturing, MS fetch |
| `gl-es31-mc.c`, `gl-mc-vis.c`, `gl-polygon-mode.c`, `gl-shader-heap.c` | the GLES 3.1 feature set of Minecraft Bedrock's renderer: clip control, depth clamp, copy_image, GL_NV_polygon_mode, shader heap growth |
| `gl-linear-zs.c`, `gl-zero-vb.c`, `gl-border-swizzle.c` | LINEAR colour targets with depth / odd pitches, empty vertex and index ranges, border colour with swizzles |
| `gl-gs.c` | geometry shaders, cube map arrays |
| `gl-narrow-args.c` | narrow (char/short) GL arguments with garbage upper bits (`mesa-glapi-narrow-args.patch`) |
| `gl-shader-async.c` | asynchronous shader compile seen from the GL API, shared shader objects |
| `gl-texpool-rebind.c` | texture/sampler pool rebinds while an earlier batch still samples |
| `gl-capture-late-arm.c`, `gl-perf-frame.c`, `gl-perf-mc.c` | perf capture armed late, per-frame placement/synchronisation cost, a Minecraft-shaped frame (directional, never timings on a busy GPU) |

Android: `gl-android-smoke.py` (../ANDROID-GL.md) builds and runs the triangle, Skia, dma-buf and
YUV programs on both bionic ABIs.

dEQP-GLES2/GLES3 on the host build: `deqp-gles2-run.py <deqp binary> <caselist> <outdir> [nvrm|llvmpipe]`
runs a case list in batches and survives crashes (a crash or watchdog timeout marks the running case
and resumes after it), writing `results.json`. dEQP is built from VK-GL-CTS with the EGL surfaceless
platform (`cmake -DDEQP_TARGET=surfaceless`); pass `DEQP_CONFIG=rgba8888d24s8ms0`. Keep case lists in
dEQP's own (tree) order: the resume logic assumes file order = execution order. `NVRM_GL_DIR_SUFFIX=-x`
makes both `build.py --gl` and the runner use build/mesa-gl-source-x and build/nvrm-host-x.
