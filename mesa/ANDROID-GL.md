# nvrm on Android: build, host smoke test, guest integration contract

nvrm is a gallium GL/GLES driver over NAK, nil and the Rust RM backend. It is opt-in beside ANGLE:
a guest selects it with `ro.hardware.egl=nvrm`.

## Build

    [NVRM_GL_DIR_SUFFIX=-x] python3 mesa/build.py --gl --target android|android32|both|host --rust-profile release

`./build.sh` runs it for both Android ABIs and the host after NVK.

- Same Mesa pin and NVK patch stack as the NVK build, plus `gallium-nvrm.patch` (meson and loader
  hookup), the GL-only Mesa fixes (`mesa-oes-texture-stencil8.patch`, `mesa-nv-polygon-mode.patch`,
  `mesa-glapi-narrow-args.patch`), `nak-panic-message.patch` and `android-egl-suffix.patch`, with
  `gallium-nvrm/` copied to `src/gallium/drivers/nvrm`.
  Own worktree `build/mesa-gl-source<sfx>`. Android: no Cargo build of its own; nvrm links NVK's Rust
  output `build/nv-rm-rust/<triple>/<profile>/libnv_bionic_native.so` (`profiles.rust_library`) and
  refuses unless `profiles.verify_metadata` says NVK's build is current and was built from this
  checkout's `nv_rm.h` (build NVK first: `build.py --target android|android32 --rust-profile release`).
  Build dirs `build/nvrm-android<sfx>` / `build/nvrm-android32<sfx>`, cross file and pkg-config in
  `build/nvrm-<target>-cross<sfx>/`. NVK's cross files, `build/nvk-*` and `build/nv-rm-rust` are not
  written. The host build (`build/nvrm-host<sfx>`) has its own Cargo target dir `build/nvrm-rust<sfx>`.
- Toolchain and sysroot are the NVK Android recipe: NDK r29 `<triple>33-clang`, SDK 33, platform
  headers from `build/android-work/deps` (`android/provision.sh`), image libraries for linking
  (`image-rootfs/system/lib64`, `build/nvk-android32/runtime`), the source-built `build/libdrm`.
- Mesa options: `-Dplatforms=android -Dplatform-sdk-version=33 -Dgallium-drivers=nvrm
  -Dvulkan-drivers=nouveau` (NAK/nil only; libvulkan is not built) `-Degl=enabled -Dgles1=enabled
  -Dgles2=enabled -Degl-lib-suffix=_nvrm -Dgles-lib-suffix=_nvrm -Dandroid-libbacktrace=disabled
  -Dllvm=disabled -Dexpat/zstd=disabled -Dzlib=enabled -Dshader-cache=enabled` (zlib = the NDK
  sysroot's libz, a `zlib.pc` written into the GL cross pkg-config dir).
  `android.hardware.graphics.mapper` is not in the pkg-config dir, so u_gralloc has the cros,
  libdrm, qcom and fallback backends (no imapper4). The cros backend is the one used
  (`strings libEGL_nvrm.so | grep "CrOS Gralloc"`).
- `android-egl-suffix.patch`: on Android Mesa always names libgallium `libgallium_dri.so`; an image
  that ships its own Mesa has `/vendor/lib{,64}/libgallium_dri.so`, and the sphal namespace resolves
  DT_NEEDED by soname. With the patch the library is `libgallium_dri<egl-lib-suffix>.so` =
  `libgallium_dri_nvrm.so`.
- Output per ABI: `src/egl/libEGL_nvrm.so`, `src/mesa/glapi/es1api/libGLESv1_CM_nvrm.so`,
  `src/mesa/glapi/es2api/libGLESv2_nvrm.so`, `src/gallium/targets/dri/libgallium_dri_nvrm.so`, plus
  `nvrm-build.json` (sha256 of each and of the `libnv_bionic_native.so` it linked against, this
  checkout's HEAD and dirty flag as `driver_source_head` / `driver_source_dirty`, Mesa pin). The
  libraries carry build-tree RUNPATHs, as NVK's libvulkan_nouveau does, and debug info.

## Host smoke test (bionic, no container)

Runner: `mesa/tests/gl-android-smoke.py [--target android|android32|both] OUT` builds the test
programs and the llvmpipe references, runs them per ABI through the image's bionic linker and prints
ANDROID SMOKE PASS/FAIL. It covers `tests/gl-g0b-triangle.c --es3` (flat scene byte-identical to
llvmpipe, gradient scene reported), `tests/gl-skia-shaders.c`, `tests/gl-dmabuf.c` (built without
Vulkan) and `tests/gl-yuv.c`.

By hand, `gl-g0b-triangle.c --es3` (ES 3.0 context; fails unless GL_VERSION is "OpenGL ES 3.x"),
built with the NDK against the `_nvrm` libraries directly (no Android EGL loader on the host) and
run with the bionic linker as `tests/vulkan/run.py` does:

    x86_64: env -i LD_LIBRARY_PATH=<rust>:<B>/src/egl:<B>/src/mesa/glapi/es2api:<B>/src/gallium/targets/dri:\
              build/bionic-runtime64/lib64:build/nvk-android/runtime:build/android-work/deps/image-rootfs/system/lib64 \
            build/bionic-runtime64/linker64 ./gl-g0b-triangle-android --es3 --flat nvrm out.rgba ref-flat.rgba
    i686:   env -i LD_LIBRARY_PATH=<rust32>:<B32>/...:build/nvk-android32/runtime \
            unshare --user --map-root-user --pid --fork build/nvk-android32/runtime/linker \
            ./gl-g0b-triangle-android32 --es3 --flat nvrm out.rgba ref-flat.rgba

The reference comes from the host build of the same file on llvmpipe
(`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1`).

## Guest integration contract

Files (per ABI; `<lib>` = lib64 for x86_64, lib for i686):

| Guest path | Source | Why there |
|---|---|---|
| `/vendor/<lib>/egl/libEGL_nvrm.so`, `libGLESv1_CM_nvrm.so`, `libGLESv2_nvrm.so` | `build/nvrm-<target>/src/...` | Android 13's EGL loader opens `lib{EGL,GLESv1_CM,GLESv2}_<ro.hardware.egl>.so` only from `/vendor/<lib>/egl` or `/system/<lib>/egl` (frameworks/native opengl/libs/EGL/Loader.cpp), by absolute path into sphal. Ship all three. |
| `/vendor_extra/<lib>/libgallium_dri_nvrm.so` | `build/nvrm-<target>/src/gallium/targets/dri/` | DT_NEEDED of all three; on the target image the sphal search and permitted paths include `/vendor_extra/${LIB}`. Not `/vendor/<lib>`, where an image's own Mesa lives. |
| `/vendor_extra/<lib>/libnv_bionic_native.so`, `libdrm.so.2` | the same files NVK uses | DT_NEEDED of libgallium_dri_nvrm; one copy per ABI, shared with libvulkan_nouveau. A stager must refuse unless `nvrm-build.json` `rust.sha256` equals the staged file (true by construction: nvrm links NVK's file). |
| image libraries | `libcutils libhardware liblog libsync libnativewindow libm libdl libc` | already reachable from sphal: an image's own libEGL_mesa.so and libvulkan_nouveau.so need the same set. |

Properties and configuration:
- `ro.hardware.egl=nvrm` selects nvrm; anything else (e.g. `angle`) leaves it unused.
- Device selection: platform_android walks the DRM render nodes (optionally `drm.gpu.vendor_name`)
  and the patched loader maps the kernel name `nvidia-drm` to `nvrm`. With only the NVIDIA render
  node in the container no property is needed; if one is set it must be `nvidia-drm`.
- Gralloc: `ro.hardware.gralloc=minigbm_gbm_mesa`. u_gralloc AUTO picks cros: `hw_get_module(GRALLOC)`
  loads the image's `gralloc.minigbm_gbm_mesa.so` (module name "CrOS Gralloc", which needs
  `libminigbm_gralloc_gbm_mesa.so` built from `gralloc/`) and `perform(GET_BUFFER_INFO)` gives fourcc,
  fds, modifier, offsets and strides. Those buffers are allocated by the minigbm backend through NVK
  (block-linear NVIDIA modifiers or LINEAR).
- Shader cache: built with `-Dshader-cache=enabled` and zlib. nvrm opens one Mesa disk_cache
  (multi-file) per screen at `dirname($TMPDIR)/code_cache/nvrm_shader_cache` (the application's code
  cache; the framework sets TMPDIR before any app code runs) through `disk_cache_create_with_path`
  (mesa-disk-cache-create-with-path.patch). It holds st/mesa's GLSL/NIR items and nvrm's NAK binaries
  (nvrm_shader_cache.c). libEGL's EGL_ANDROID_blob_cache callbacks are ignored by such a cache
  (mesa-disk-cache-keep-file-backend.patch). Processes without an app code cache (isolated,
  system_server, SurfaceFlinger, tools) log once and keep compiled shaders in memory; their Mesa cache
  stays disabled and serves libEGL's in-memory blob cache. With the cache, NUM_PROGRAM_BINARY_FORMATS
  is 1.

## Driver capabilities

`tests/gl-android-smoke.py` is the capability gate: per ABI it runs `tests/gl-dmabuf.c` (no Vulkan)
through the bionic linker, prints the EGL extension string, and fails unless the extensions below
exist and round-trip.

1. **dma-buf import/export.** `nvrm_handle.c`: `resource_from_handle` (FD only; plane 0, offset 0,
   explicit modifier required: NVIDIA memory has no implicit layout), `resource_get_handle` /
   `resource_get_param` (FD, stride, offset, modifier; KMS/flink refused), `resource_create_with_modifiers`,
   `query_dmabuf_modifiers` / `is_dmabuf_modifier_supported` from nil's list (= NVK's): RGBA/RGBX/BGRA/
   FP16/1010102 = 6 uncompressed block-linear `0x03000000006060{10..15}` + LINEAR; RGB565 = LINEAR only
   (nil has no 16-bit GOB modifier encoding, NVK the same). None external-only: LINEAR imports sample
   through a nil pitch TIC and render through LAYOUT_PITCH (pitch must be a 128 B multiple, else a
   named abort at bind time; NVK's LINEAR images are 128 B aligned). Imports = `nv_rm_import` + VA
   with the modifier's PTE kind (as NVK's nvkmd backend); exports = `nv_rm_export`, uncompressed like NVK's.
2. **EGL_ANDROID_native_fence_sync / EGL_KHR_wait_sync.** `caps.native_fence_fd`;
   `fence_get_fd` = `nv_rm_context_fence` (semaphore sync_file for the batch token; an empty batch
   when a fence is requested with no work), `create_fence_fd` dups the sync_file,
   `fence_server_sync` queues it for `nv_rm_submit_waits` on the next batch. Flushes are
   synchronous (the CPU waits for each batch), so fds are already signalled when handed out.
3. **Window-buffer render targets for the minigbm format list** (gl-dmabuf test 6: Vulkan allocates
   from GL's modifier list as the minigbm backend does, GL renders, Vulkan reads back exact):
   RGBA_8888, RGBX_8888, BGRA_8888, RGBA_FP16, RGBA_1010102 (block-linear), RGB_565 (LINEAR, pitch 128).
4. **YUV (NV12, P010) EGLImage import + samplerExternalOES.** SurfaceFlinger composites video layers
   this way. nvrm reports no YUV texture format, so dri2 imports per plane (NV12 = R8 + R8G8,
   P010 = R16 + R16G16, chained through `pipe_resource::next`) and st/mesa samples the planes and
   converts (nir_lower_tex CSC: BT601/709/2020 from EGL_YUV_COLOR_SPACE_HINT_EXT, limited/full from
   EGL_SAMPLE_RANGE_HINT_EXT; on Android Mesa takes them from the gralloc dataspace).
   `query_dmabuf_modifiers(NV12/P010)` = LINEAR only, external-only. Plane rules = NVK's for a plane
   of a LINEAR DRM-modifier image (nvk_image.c explicit plane offsets, nil `new_linear`): LINEAR
   modifier, pitch a 32 B multiple >= the plane row, offset a multiple of the plane's nil alignment
   (32 B; 256 B on Kepler), plane inside the dma-buf; each plane imports the dma-buf (kindless, kind 0
   VA) and is addressed at VA + offset (`nvrm_resource::plane_offset_B`); sampler-only (colour-target
   bind and export of a plane refused). The minigbm 0004 layout (pitch ALIGN(bytes, 256), CbCr at
   pitch*even(rows)) satisfies this, same fd or separate fds. Gate: `tests/gl-yuv.c` (host and both
   bionic ABIs in gl-android-smoke.py).
   P010 colour accuracy is Mesa's: the plane is sampled as UNORM16 and the 8-bit-defined ranges are
   applied (no 10-bit rescale), up to 3 LSB (8-bit) off strict 10-bit H.273 for limited range; all
   gallium drivers that lower P010 share it.

Refused (loudly): YUV formats other than NV12/P010 (YV12/I420, NV21, P012/P016, packed YUYV...),
block-linear YUV, offsets on non-YUV imports, mipmapped/array/MSAA/depth exports, KMS handles.

## Shader failures and shader dumps on Android (logcat, tag MESA)

- A shader NAK cannot compile still aborts (there is no fallback compiler), but first logs, one
  logcat line each: `nvrm: NAK failed to compile the <stage> shader; NAK panic:`, NAK's panic
  location and message (`nak-panic-message.patch`: NAK's catch_unwind keeps the message of the last
  panic per thread, `nak_compile_shader_panic_message()`; NAK's own stderr print is lost on Android),
  then `nvrm: NIR handed to NAK:` and that NIR (a clone taken right before `nak_compile_shader`, since
  NAK lowers in place). Long lines are split at 1000 bytes.
- Opt-in dump of every shader (st/mesa NIR + NAK assembly): `NVRM_DEBUG_SHADERS=1` on the host; on
  Android Mesa's `os_get_option` falls back to properties (util/os_misc.c `os_get_android_option`:
  lowercase, `_` to `.`, `mesa.` prefix, tried as `debug.`/`vendor.`/plain), so
  `setprop debug.mesa.nvrm.debug.shaders 1` enables it for every process including SurfaceFlinger.
  It is read per shader creation, so it applies to shaders created after the setprop. `logcat -s MESA`.
- SurfaceFlinger's RenderEngine (Skia) samples gralloc buffers through `samplerExternalOES`
  (OES_EGL_image_external is always exposed by Mesa); NAK has no external sampler dimension
  (GLSL_SAMPLER_DIM_EXTERNAL), so nvrm lowers it to 2D (`nvrm_texture.c`), covered by
  `tests/gl-skia-shaders.c` in the Android smoke.
