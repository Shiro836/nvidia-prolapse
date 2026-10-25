# VK_ANDROID_external_memory_android_hardware_buffer: the consumer contract

What the two consumers that matter in the guest actually call, in what order, with which
structures. Measured from their source, not inferred from symptoms. Written before the
implementation (`src/ahb.c`, `src/imgxlate.c`); the implementation follows it.

Sources (fetched 2026-10-10, line numbers refer to these exact revisions):

- **ANGLE** `c1a25085dd9e4a8cd6c72be278c0b4bdf6ce2824` (the build in the guest),
  `https://chromium.googlesource.com/angle/angle/+/c1a25085dd9e4a8cd6c72be278c0b4bdf6ce2824/<path>`.
  Abbreviations: `HBS` = `src/libANGLE/renderer/vulkan/android/HardwareBufferImageSiblingVkAndroid.cpp`,
  `VKH` = `src/libANGLE/renderer/vulkan/vk_helpers.cpp`, `VKU` = `.../vk_utils.cpp`,
  `VKR` = `.../vk_renderer.cpp`, `VKB` = `.../vk_barrier_data.cpp`, `VKW` = `.../vk_wrapper.h`,
  `VAU` = `.../android/vk_android_utils.cpp`, `CAPS` = `.../vk_caps_utils.cpp`.
- **Skia** (Android 13 framework copy), `https://android.googlesource.com/platform/external/skia/+/refs/heads/android13-release/<path>`:
  `GAU` = `src/gpu/GrAHardwareBufferUtils.cpp`, `GVC` = `src/gpu/vk/GrVkCaps.cpp`,
  `GVI` = `src/gpu/vk/GrVkImage.cpp`, `GVM` = `src/gpu/vk/GrVkMemory.cpp`.
- **HWUI** (Android 13), `https://android.googlesource.com/platform/frameworks/base/+/refs/heads/android13-release/<path>`:
  `VKS` = `libs/hwui/renderthread/VulkanSurface.cpp`, `VKM` = `libs/hwui/renderthread/VulkanManager.cpp`.
- Mesa 25.2.0 `src/vulkan/runtime/vk_android.c` (`build/mesa-git`), the semantic reference for a
  dma-buf driver.
- The Vulkan spec text is cited as "spec" where neither consumer pins a behaviour.

## 1. Enabling

| Consumer | Condition | Where |
|---|---|---|
| ANGLE | feature `supportsAndroidHardwareBuffer` = device lists **both** `VK_ANDROID_external_memory_android_hardware_buffer` and `VK_EXT_queue_family_foreign`; then enables both and loads `vkGetAndroidHardwareBufferPropertiesANDROID` + `vkGetMemoryAndroidHardwareBufferANDROID` | VKR:5666-5670, 3895-3899, 4565; VKU:1306-1307 |
| ANGLE | `supportsExternalFormatResolve` only with `VK_ANDROID_external_format_resolve`'s feature (not exposed here) → `VkAndroidHardwareBufferFormatResolvePropertiesANDROID` is never chained | VKR:6723-6729; HBS:239-246, 343-350 |
| Skia | `fSupportsAndroidHWBExternalMemory` = extension at spec version **≥ 2** + external memory + bindMemory2 (queue_family_foreign deliberately not required) | GVC:353-362 |
| HWUI | enables **every** device extension the driver lists (`mDeviceExtensions` = all), fatal without `VK_KHR_swapchain` (the loader adds it) | VKM:214-224 |

Consequence: the HAL must list the extension (spec version 5) and `VK_EXT_queue_family_foreign`;
every listed extension must survive `vkCreateDevice` (HWUI/dEQP enable all of them).

## 2. Import sequence: ANGLE (`eglCreateImageKHR(EGL_NATIVE_BUFFER_ANDROID)`)

`HardwareBufferImageSiblingVkAndroid::ValidateHardwareBuffer` then `::initImpl`:

1. `vkGetAndroidHardwareBufferPropertiesANDROID(device, ahb, &VkAndroidHardwareBufferPropertiesANDROID{pNext = VkAndroidHardwareBufferFormatPropertiesANDROID})`
   (HBS:234-254, again at HBS:333-354). Fields used:
   - `format`: if not `UNDEFINED`, ANGLE uses its own format tables for that VkFormat
     (`formatHasNecessaryFormatSupport`, HBS:174-202) and the image is **not** "external";
   - `formatFeatures`: only for the external-format decision: needs `SAMPLED_IMAGE`, plus
     `COLOR_ATTACHMENT` if the AHB has `GPU_FRAMEBUFFER` (HBS:165-172);
   - `externalFormat`: `ASSERT(externalFormat != 0)` (HBS:270, quoting the spec "must not be zero");
   - suggested Ycbcr fields only when external or YUV (HBS:515-541);
   - `allocationSize`, `memoryTypeBits`: become the memory requirements (HBS:585-588).
2. `AHardwareBuffer_acquire` (HBS:331), `AHardwareBuffer_describe` (HBS:414).
3. `vkCreateImage` (via `ImageHelper::initExternal`, VKH:5898-6067) with:
   - `tiling = VK_IMAGE_TILING_OPTIMAL` unconditionally (HBS:24-74; the comment names NVIDIA:
     linear + `INPUT_ATTACHMENT` is unsupported there);
   - `usage` = `TRANSFER_SRC|TRANSFER_DST` (non-external) | `SAMPLED|INPUT_ATTACHMENT` for
     `GPU_SAMPLED_IMAGE` | `COLOR_ATTACHMENT` (or `DEPTH_STENCIL`) for `GPU_FRAMEBUFFER` (HBS:78-107);
   - `flags` = `CUBE_COMPATIBLE` for `GPU_CUBE_MAP`, `PROTECTED` for `PROTECTED_CONTENT` (HBS:111-126),
     `MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXT` only with that extension (absent on this ICD, HBS:479-485),
     plus `MUTABLE_FORMAT` **with** a `VkImageFormatListCreateInfo{actual, sRGB/linear twin}` when
     both formats have the same features (`DeriveCreateInfoPNext`, ColorspaceOverrides, VKH:6082-6141; HBS:556-566);
   - `pNext` = [`VkImageFormatListCreateInfo` →] `VkExternalMemoryImageCreateInfo{handleTypes = AHB}`
     → `VkExternalFormatANDROID{externalFormat = 0}` (non-zero only on the external path) (HBS:449-453, 370-372);
   - `mipLevels` = full chain only with `GPU_MIPMAP_COMPLETE`, else 1 (HBS:459-461); `arrayLayers` = AHB layers (HBS:468);
   - `sharingMode = EXCLUSIVE` (VKH:6019); `initialLayout` = `ExternalPreInitialized` → **`VK_IMAGE_LAYOUT_UNDEFINED`** (VKH:6022, VKB:427-438).
4. **No** `vkGetImageMemoryRequirements*` on this image. Requirements come from step 1 with
   `alignment = 0` (HBS:585-588).
5. `vkAllocateMemory` with `pNext = VkMemoryDedicatedAllocateInfo{image, buffer = NULL}` →
   `VkImportAndroidHardwareBufferInfoANDROID{buffer}` (HBS:574-583), `allocationSize` =
   `allocationSize` from step 1, memory type: first type in `memoryTypeBits` with `DEVICE_LOCAL`,
   else (external memory) the first type in `memoryTypeBits` at all (VKU:85-100, 382-438).
6. **`vkBindImageMemory`** (version 1, offset 0) for single-plane images (VKU:140-154, VKW:1599-1603).
7. Ownership: the image starts on `kForeignDeviceQueueIndex` (HBS:594-596, VKH:6535-6576); ANGLE
   records barriers with `srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT` when acquiring and
   `dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT` when releasing (`releaseToForeign`, VKH:7355-7378).
   Barriers go through `vkCmdPipelineBarrier` or events (`vkCmdSetEvent`/`vkCmdWaitEvents`, with
   image barriers); with synchronization2 the `2` variants.
8. Image views (`vkCreateImageView`), then any transfer/clear/blit/resolve command, descriptor and
   framebuffer use go through views or the image handle.
9. Teardown: image released (garbage-collected) and memory freed; `AHardwareBuffer_release` in
   `onDestroy` (HBS:629-638).

`vkGetMemoryAndroidHardwareBufferANDROID` is loaded but never called by ANGLE (only VKU:1190, 1307).
ANGLE never calls `vkGetPhysicalDeviceImageFormatProperties2` with the AHB handle type.

### ANGLE buffers (`GL_EXT_external_buffer`, advertised on Android ≥ 26, CAPS:557)

`BufferHelper::initExternal` (VKH:4827-4865) + `InitAndroidExternalMemory` (VAU):
`vkCreateBuffer` with `VkExternalMemoryBufferCreateInfo{AHB}`; the AHB must be
`AHARDWAREBUFFER_FORMAT_BLOB`; `vkGetAndroidHardwareBufferPropertiesANDROID` (no format struct);
`vkAllocateMemory` with `VkImportAndroidHardwareBufferInfoANDROID` → `VkMemoryDedicatedAllocateInfo{buffer}`;
then `vkBindBufferMemory`. So **AHB import for VkBuffer is used** (rarely).

## 3. Import sequence: Skia / HWUI (`debug.hwui.renderer=skiavk`)

HWUI's `VulkanSurface` does not use `VK_KHR_swapchain` for the window: it dequeues buffers from
the `ANativeWindow` and wraps each as `SkSurface::MakeFromAHardwareBuffer` (VKS:434).

1. Window setup: `vkGetPhysicalDeviceImageFormatProperties2` with
   `VkPhysicalDeviceExternalImageFormatInfo{AHB}`, `tiling = OPTIMAL`,
   `usage = COLOR_ATTACHMENT|SAMPLED|TRANSFER_SRC|TRANSFER_DST`, `flags = 0`, output chain
   **only** `VkAndroidHardwareBufferUsageANDROID` (no `VkExternalImageFormatProperties`); failure is
   fatal for the surface; `androidHardwareBufferUsage` is OR'ed into the window's buffer usage (VKS:225-268).
2. Per buffer (`GrAHardwareBufferUtils::MakeVulkanBackendTexture`, GAU:320-520):
   - known AHB formats map to a VkFormat **without** asking the driver (GAU:96-114); only unknown
     formats query and then require `format == UNDEFINED` (external) (GAU:115-160);
   - `vkGetAndroidHardwareBufferPropertiesANDROID` with `VkAndroidHardwareBufferFormatPropertiesANDROID` (GAU:343-353);
     for known formats asserts `formatFeatures ⊇ SAMPLED_IMAGE|TRANSFER_SRC|TRANSFER_DST` (debug, GAU:366-371)
     and `format == hwbFormatProps.format` (GAU:380);
   - `vkCreateImage`: `VkExternalMemoryImageCreateInfo{AHB}` → `VkExternalFormatANDROID{0}`, format = the
     known VkFormat, `usage = SAMPLED|TRANSFER_SRC|TRANSFER_DST` (+`COLOR_ATTACHMENT` if renderable),
     `tiling = OPTIMAL`, `flags = 0` (or `PROTECTED`), 1 mip, 1 layer, `EXCLUSIVE`, `initialLayout = UNDEFINED` (GAU:381-422);
   - memory type: **must** have `DEVICE_LOCAL` among `memoryTypeBits`, else the import is abandoned (GAU:425-450);
   - `vkAllocateMemory` with `VkMemoryDedicatedAllocateInfo{image}` → `VkImportAndroidHardwareBufferInfoANDROID`,
     `allocationSize` from the properties (GAU:452-475);
   - **`vkBindImageMemory2`** (one bind, offset 0) (GAU:477-489);
   - image state: `OPTIMAL`, layout `UNDEFINED`, queue family **`VK_QUEUE_FAMILY_EXTERNAL`** (GAU:494-507);
     barriers transfer ownership from/to EXTERNAL or FOREIGN (GVI:390-412).
3. Skia's own `vkGetImageMemoryRequirements` is only for images it allocates itself (GVM:58-66) and in a
   testing-only path (GrVkGpu.cpp:1958); `vkGetImageSubresourceLayout` only for linear images it owns
   (GrVkGpu.cpp:755-770). Neither touches an AHB image.

## 4. What the HAL must provide (derived)

| Item | Requirement | From |
|---|---|---|
| Extension list | `VK_ANDROID_external_memory_android_hardware_buffer` spec 5, `VK_EXT_queue_family_foreign` | §1 |
| Properties query | `allocationSize`, `memoryTypeBits` (with a `DEVICE_LOCAL` type for VRAM buffers, or Skia fails), `format` = the equivalence-table VkFormat, **`externalFormat != 0`**, `formatFeatures` ⊇ SAMPLED/TRANSFER for usable formats; Ycbcr suggestions | §2.1, §3.2 |
| Image create | accept `OPTIMAL` (both), any usage incl. `INPUT_ATTACHMENT`, `MUTABLE_FORMAT` + format list, `CUBE_COMPATIBLE`, layers; `VkExternalFormatANDROID{0}` chained; `initialLayout UNDEFINED` | §2.3, §3.2 |
| Memory | dedicated import via `VkImportAndroidHardwareBufferInfoANDROID`; `alignment` is not asked; no `vkGetImageMemoryRequirements` before allocation | §2.4-5, §3.2 |
| Bind | **both** `vkBindImageMemory` (ANGLE) and `vkBindImageMemory2` (Skia) | §2.6, §3.2 |
| Commands | barriers with `FOREIGN`/`EXTERNAL` queue families (PipelineBarrier, events, sync2), views, transfers on the app's image handle | §2.7-8, §3.2 |
| Image format query | AHB handle type with `OPTIMAL`, output `VkAndroidHardwareBufferUsageANDROID` possibly without `VkExternalImageFormatProperties` | §3.1 |
| Buffers | `vkCreateBuffer` with the AHB handle type, BLOB AHB import with dedicated buffer | §2 (buffers) |
| Export | not used by either consumer (`vkGetMemoryAndroidHardwareBufferANDROID` never called) | §2 |

## 5. Implementation decisions this contract forces (and deviations from the first design)

- `externalFormat` is **non-zero**: like Mesa (vk_android.c:760, 842) it carries the VkFormat value.
  `VkExternalFormatANDROID{externalFormat = that value}` is accepted in `vkCreateImage` (format
  must then be `UNDEFINED`, spec) and translated back to the real format; image views of such an
  image with `format = UNDEFINED` get the real format; `vkCreateSamplerYcbcrConversion` with it gets
  `format` set and the struct removed. Design said `externalFormat = 0`; that contradicts the spec
  and ANGLE's assert.
- The image's tiling (`OPTIMAL`) is advisory: the buffer's layout decides. The HAL realises the
  image at `vkAllocateMemory`/bind time with `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` and the
  buffer's explicit modifier/stride/offset.
- `vkBindImageMemory` (v1) must be intercepted, not only `vkBindImageMemory2`.
- Unsupported AHB contents (implementation-defined layouts we cannot express, multi-plane other than
  NV12/P010, §9) fail the properties query with `VK_ERROR_INVALID_EXTERNAL_HANDLE` (the only failure code the
  spec allows for that command besides OOM), with the format named in the log. Design said
  `FORMAT_NOT_SUPPORTED`, which the spec does not list for this command.
- `memoryTypeBits` is the ICD's `vkGetMemoryFdPropertiesKHR` answer for the dma-buf. For a
  sysmem (CPU-mappable, LINEAR) gralloc buffer that answer has no `DEVICE_LOCAL` type
  (measured 0x9 = types 0 and 3), so Skia (GAU:425-450) refuses such buffers while ANGLE accepts
  them (VKU:431-438). That is the ICD's truth; it is reported, not papered over.
- AHB import for `VkBuffer` is implemented (ANGLE uses it); AHB **export** without a dedicated
  image is refused loudly (no consumer uses it, §2).
- `formatFeatures` is the ICD's `drmFormatModifierTilingFeatures` of the buffer's actual modifier
  (block-linear RGBA8 on 615.71.09: `0x1dd83`, LINEAR: `0x1dd03`, measured). The spec says the
  field "must include `MIDPOINT_CHROMA_SAMPLES` or `COSITED_CHROMA_SAMPLES`"; Mesa adds
  `MIDPOINT` unconditionally. This HAL does not: the ICD reports neither bit for any RGB format
  and modifier, and neither consumer takes the external-format path for a buffer whose `format` is
  defined (HBS:364-365, GAU:96-114). Claiming the bit would invite a Ycbcr conversion the ICD
  does not support.
- `vkGetImageMemoryRequirements*` on an AHB image before it has memory is invalid usage
  (VUID-VkImageMemoryRequirementsInfo2-image-01897) and neither consumer does it (§2.4, §3.3).
  It is answered from a probe image with the modifiers the gralloc would pick for the image's
  tiling, with a warning. For an `OPTIMAL` image backed by a CPU (LINEAR, sysmem) buffer the
  probe cannot know the buffer will be sysmem, so its `memoryTypeBits` (0x3) only overlap the
  buffer's (0x9) in type 0.

## 6. Swapchain-bound app images and loader-object private data (added 2026-10-10, from the G3 wsi run)

Sources: `build/android13/fn-vulkan/libvulkan/swapchain.cpp` (`S`), `api_gen.cpp` (`AG`),
`include/vulkan/vk_android_native_buffer.h` (`NB`), CTS `vktWsiSwapchainTests.cpp:1298-1420, 748-870`,
Mesa `vk_object.c:232-257`.

| Fact | Where |
|---|---|
| `vkCreateImage` with `VkImageSwapchainCreateInfoKHR{swapchain != NULL}` reaches the driver **unchanged**; the handle is the loader's own object | AG:1145 (plain dispatch), no interception in S |
| `vkBindImageMemory2[KHR]` with `VkBindImageMemorySwapchainInfoKHR` is rewritten by the loader: it chains a `VkNativeBufferANDROID` for the gralloc buffer of `imageIndex` **in front of** the app's chain (`native_buffer.pNext = app chain`, `bind.pNext = &native_buffer`), `memory` stays `VK_NULL_HANDLE` | S:1989-2085 (`InterceptBindImageMemory2`, 2024-2043); NB:42-51 (spec version 8) |
| No loader notification when such an image or its swapchain dies | S (none) |
| `vkSet/GetPrivateData[EXT]` on `VK_OBJECT_TYPE_SWAPCHAIN_KHR` / `SURFACE_KHR` reach the driver with the loader's handles; the AOSP 13 loader has no private-data code at all | `grep PrivateData` over S, driver.cpp, api.cpp = empty |
| Mesa stores private data of swapchain/surface objects in the driver on Android for exactly this reason | vk_object.c:232-257 |

Measured consequence with NVIDIA's ICD: both reach eglcore with a foreign pointer and SIGSEGV
(`nvhal_CreateImage+511` for the create; a direct libdeqp→eglcore frame for the private data).

HAL behaviour (src/anb.c `nvhal_CreateImage`/`bind2`, src/ahb.c `nvhal_sc_*`, src/imgxlate.c `set_pd`/`get_pd`):
- A create with a non-null swapchain returns a HAL-owned deferred wrapper (kind `NVHAL_WRAP_SWAPCHAIN`), the
  ICD is not called; the real image is created at the swapchain bind through the ANB import core
  (`nvhal_anb_import`: explicit modifier from the cros handle, dedicated dma-buf import, bind). Memory
  requirements before the bind come from an OPTIMAL-tiling probe (approximation; the bind never uses them).
- Every other combination (native buffer on a plain/AHB image, swapchain info on a plain image, deferred image
  bound without both structs or with `memory != NULL`, a second bind, `vkBindImageMemory` on a deferred image)
  is `VK_ERROR_INVALID_EXTERNAL_HANDLE` with the condition in the log.
- The image-translating entry points (`NVHAL_IC_WRAP`) are installed whenever ANB **or** AHB is enabled; before
  this fix they were AHB-only, so an ANB-only device (dEQP's) would have passed wrapper pointers to the ICD.
- Private data for swapchain/surface handles lives in a per-device HAL table keyed (type, handle, slot);
  `vkDestroyPrivateDataSlot[EXT]` drops the slot's values (the ICD recycles slot handles); never-set reads 0.
  Known limit (shared with Mesa): a later swapchain at a recycled address reads stale values until the slot or
  device is destroyed.
- Not covered: `vkGetDeviceImage{Memory,SparseMemory}Requirements` / `vkGetDeviceImageSubresourceLayout` with
  `VkImageSwapchainCreateInfoKHR` in the create info, and debug naming/tagging of swapchain objects, still reach
  the ICD (not measured).
Host tests: `test/anb-roundtrip.c` (image C section, pixel round trip through the swapchain bind),
`test/loader-pd.c` (private data; its control run shows the raw ICD dying with SIGSEGV on the same call).

## 7. ASTC LDR and ETC2/EAC emulation (ASTC added 2026-10-10, risk R8; ETC2/EAC 2026-10-18, 7.8)

Sources: `src/astc.c` (`A`), `src/cmdstate.c` (`CS`), `src/imgxlate.c` (`X`), `src/anb.c`, `src/ahb.c`,
`src/device.c`, `src/hal.c`; dEQP `external/vulkancts/modules/vulkan/api/vktApiFeatureInfo.cpp` (`FI`, CTS
vulkan-cts-1.4.6.2); Mesa `mesa-25.2.0` (`M`). Design reference: Mesa `src/vulkan/runtime/vk_texcompress_astc.c`
(anv/radv) and the baseline's `vn_astc_emu` (`patches/mesa/0003`).

### 7.1 ICD facts measured on the way (615.71.09, RTX 5090)

| Fact | Evidence |
|---|---|
| `textureCompressionASTC_LDR = 0`; every ASTC format reports 0 features, yet `vkCreateImage(ASTC_4x4, OPTIMAL)` returns `VK_SUCCESS` and `vkGetImageMemoryRequirements` size 0 (Unity/VMA then fails: Endfield) | `scratchpad/ef/probe/astcprobe.c`, host ICD and guest HAL |
| The backing images accept the sparse flags: `R8G8B8A8_SRGB` usage `SAMPLED\|STORAGE\|TRANSFER_*` + `MUTABLE\|EXTENDED_USAGE\|SPARSE_BINDING\|SPARSE_ALIASED` -> 0; `R32G32B32A32_UINT` `SAMPLED\|TRANSFER_*` + `SPARSE_BINDING\|SPARSE_ALIASED` -> 0 | `scratchpad/astc/ifpprobe.c` |
| `R8G8B8A8_SRGB` has no STORAGE without `EXTENDED_USAGE` (-11) | `scratchpad/astc/icdfacts.c` |
| Queue families 0xf, 0xc (transfer only), 0xe, video; `minTexelBufferOffsetAlignment` 16; `sparseBinding` 1, `sparseResidencyAliased` 1 | same |

### 7.2 What the HAL advertises (only when the ICD lacks ASTC and the probe A:70 passes)

| Query | Answer | Where |
|---|---|---|
| Probe prerequisites | `R8G8B8A8_UINT` STORAGE, `R32G32B32A32_UINT` SAMPLED+TRANSFER, the 4 LUT texel-buffer formats, `shaderStorageImageWriteWithoutFormat`, 12x12x4 workgroups, a compute queue; else one error "(ASTC gap, R8)" and nothing advertised | A:70-137 |
| `vkGetPhysicalDeviceFeatures[2][KHR]` | `textureCompressionASTC_LDR = VK_TRUE` | A:194-217 |
| `vkGetPhysicalDeviceFormatProperties[2][KHR]` (28 2D LDR formats) | optimal = `R8G8B8A8_{UNORM,SRGB}` features & `SAMPLED\|BLIT_SRC\|FILTER_LINEAR\|TRANSFER_SRC\|TRANSFER_DST\|FILTER_MINMAX` = 0x1d401; linear/buffer 0; `VkFormatProperties3` same; DRM modifier lists empty | A:52, A:219-275 |
| `vkGetPhysicalDeviceImageFormatProperties[2]` | 2D, OPTIMAL, usage ⊆ `SAMPLED\|TRANSFER_*`, flags ⊆ `MUTABLE\|CUBE\|EXTENDED_USAGE` + `SPARSE_BINDING` (if `sparseBinding`) + `SPARSE_ALIASED` (if also `sparseResidencyAliased`); answer = the ICD's for the RGBA8 backing with `STORAGE`, `MUTABLE\|EXTENDED_USAGE` added; `sampleCounts = 1`; external memory refused | A:57-63, A:281-397, ahb.c:1243 |
| Why the sparse flags | FI:4705-4737 offers `SPARSE_BINDING\|RESIDENCY\|ALIASED` for sampled usage per device feature; FI:4753 `isRequiredImageParameterCombination` exempts compressed formats only from other usages (FI:4766), 1D/3D and `SPARSE_RESIDENCY` (FI:4788-4791); a refusal fails at FI:5092. Measured: 896 of 1,428 required combinations refused before, 0 after; guest `image_format_properties.2d.optimal.astc*` 28/28 after | |
| `vkCreateDevice` | `textureCompressionASTC_LDR` stripped before the ICD, `shaderStorageImageWriteWithoutFormat` enabled (decoder output has no format qualifier); API 1.0 devices also get `VK_KHR_maintenance2` | A:399-427, device.c:428-441 |

### 7.3 What an app ASTC image is

| Part | Create info | Where |
|---|---|---|
| The app's `VkImage` = an ICD image | format `R8G8B8A8_UNORM`/`_SRGB` (per ASTC UNORM/SRGB, Mesa `vk_texcompress_astc_emulation_format`), usage += `STORAGE`, flags += `MUTABLE_FORMAT\|EXTENDED_USAGE`; an app format list is mapped to RGBA8 + `R8G8B8A8_UINT` | A:656-733 |
| One block image per mip *m* | `R32G32B32A32_UINT` (one texel per 128-bit block), extent `ceil(max(1,w>>m)/bw) x ceil(max(1,h>>m)/bh)`, 1 mip, same layers/sharing/initial layout, usage `SAMPLED\|TRANSFER_*`, the app's sparse flags | A:724-729 |
| Memory requirements | parts laid out in order, each at its own alignment (sparse: all at the largest = sparse block size, total rounded up); `size = end of last part`, `alignment = max`, `memoryTypeBits = AND`; dedicated requirements answered false/false. 1920x1080 ASTC_4x4: 11,796,480 B vs RGBA8 alone 8,847,360 B (host and guest) | A:736-755, A:874-916 |
| Binding | `vkBindImageMemory`: RGBA8 at offset, block *m* at offset + off[1+m]; `vkBindImageMemory2`: block binds appended; `vkQueueBindSparse`: each opaque bind split over the parts (resource and memory offsets shifted, padding skipped) | A:918-954, A:1012-1090, anb.c:585, ahb.c:1118, X:1048 |
| Dedicated allocation | `VkMemoryDedicatedAllocateInfo{image}` / NV variant removed (the memory also backs the block images) | A:988-1010, ahb.c:837 |

### 7.4 Intercepted entry points

| Entry point | Behaviour | Where |
|---|---|---|
| `vkCreateImage` / `vkDestroyImage` | ASTC → parts above; destroy frees block images, cached views and descriptor sets | anb.c:455, 477; A:775-872 |
| `vkCreateImageView` | ASTC view format → RGBA8 UNORM/SRGB; `VkImageViewUsageCreateInfo{app usage}` added (the image's STORAGE is the HAL's); `VkImageViewASTCDecodeModeEXT` dropped | X:350, A:956-986 |
| `vkGetImageMemoryRequirements[2]`, `vkGetDeviceImage{Memory,SparseMemory}Requirements`, `vkGetDeviceImageSubresourceLayout` | combined requirements; sparse 0; layout zero (OPTIMAL) | X:212, 231, 628, 649, 675 |
| `vkCmdCopyBufferToImage[2]` | regions in blocks → block image, then decode (7.5) | X:760, 818; A:1315 |
| `vkCmdCopyImageToBuffer[2]` | from the block image: compressed data round-trips bit-exact | X:768, 819; A:1340 |
| `vkCmdCopyImage[2]` | ASTC→ASTC: texels RGBA8→RGBA8 + blocks; ASTC→128-bit: blocks; 128-bit→ASTC: blocks then decode (extent = source texels, spec) | X:744, 816; A:1353 |
| `vkCmdPipelineBarrier[2]`, `vkCmdWaitEvents[2]`, `vkCmdSetEvent2` | each image barrier on an ASTC image is repeated for its block images of the covered mips (layouts, queue ownership) | X:858, 874, 887; A:1143-1144 |
| Compute-state shadow (`vkBeginCommandBuffer`, `vkCreateCommandPool`, `vkCmdBindPipeline`, `vkCmdBindShadersEXT`, `vkCmdBindDescriptorSets[2]`, `vkCmdPushDescriptorSet[2]`, `vkCmdPushConstants[2]`, descriptor-buffer offsets/samplers, conditional rendering) | records the COMPUTE bind point's pipeline/shader, live set-binding calls (deep copies) and push constants since the first compute-stage push; replayed after every decode. Handed out by GDPA only on emulating devices | CS:100, 268, 305-352, 626; hal.c:531 |
| Command-buffer lookup | per-thread cache keyed by a generation counter (every insert/delete bumps it) | X:28-40 |

### 7.5 The decode (Mesa `vk_texcompress_astc` usage, in the app's command buffer)

- Barrier TRANSFER→COMPUTE: block image and RGBA8 subresources from the copy's layout to GENERAL; bind the
  decoder pipeline (specialization: block size, `DECODE_8BIT = true`), set 0 (storage `R8G8B8A8_UINT` view,
  sampled block view, 5 LUTs, partition table), push `{blk_start, texel_end, 0}` (20 B), dispatch
  `ceil(bx/2) x ceil(by/2) x layers`; barrier COMPUTE→TRANSFER back to the app's layout (so the app's next barrier,
  whose first scope is its transfer, chains); then restore the app's compute state. A:1222-1313, CS:305.
- Views and descriptor sets cached per (image, mip, layer range); pipelines per block size; LUT buffer
  HOST_VISIBLE, built once per device. A:471-610, A:1171-1220.

### 7.6 Refused loudly (`VK_ERROR_FORMAT_NOT_SUPPORTED` or a dropped command, error text "(ASTC gap, R8)")

| Case | Why | Where |
|---|---|---|
| LINEAR/DRM tiling, 1D/3D, MSAA, sparse residency, alias, protected, disjoint, block-texel-view, external memory, storage/attachment/host-transfer usage | no ASTC memory layout to offer; texels live decoded in RGBA8 | A:281-294, A:656-722 |
| View with a non-ASTC format | block-texel views not emulated | A:956-967 |
| Device-address, indirect, host-image copies; `vkTransitionImageLayout`; descriptor-heap writes | regions unknown at record time / no host path | X:375, 397, 793, 826-841, 1075 |
| Upload on a queue family without COMPUTE | blocks copied, decode impossible (dEQP uploads on the universal queue, vktApiCopiesAndBlittingUtil.cpp:1412-1424) | A:1265-1271 |
| Decode inside conditional rendering; app state set via update templates, descriptor heaps or dynamic pipeline layouts | dispatch obeys the predicate / state not replayable: error at the decode | A:1272-1275, CS:482, 576 |

### 7.7 Provenance, tests, measurements

- Mesa commit `0b97dcdd9424fce130054e1972a610b765644003` (Makefile:33), files read with `git show` from
  `build/mesa-git`'s object store. Shader: `glslangValidator -V -S comp -x` = M `src/vulkan/runtime/meson.build:283-287`,
  validated with `spirv-val --target-env vulkan1.0` (Makefile:53-55), glslang version in `out/gen/astc_spv.version`.
  LUTs: `tools/astc-luts-gen.cpp` over M `src/util/texcompress_astc_luts.cpp` (wrapper order/formats of
  `texcompress_astc_luts_wrap.cpp`) → `out/gen/nvhal_astc_luts.c`.
- Void-extent blocks with reserved bits 10-11 ≠ 11b: M `astc_decoder.glsl:1229` returns the error colour; M
  `src/mesa/main/texcompress_astc.cpp` (`decode_void_extent`) and dEQP `framework/common/tcuAstcUtil.cpp:694`
  ignore the bits. Encoders write 11b; `test/astc-roundtrip.c` does not generate such blocks.
- `test/astc-roundtrip.c` (reference: Mesa's CPU decoder, `test/astc-ref.cpp`): queries incl. the FI loop (1,428
  combinations), refusals, 8 pixel cases (5 block sizes, UNORM/SRGB, NPOT, mips, layers, 2 sparse-bound), raw
  readback, ASTC↔ASTC/128-bit copies, blit, compute state across the upload; API 1.3 and 1.0. SRGB exact, UNORM
  ±1 (CPU UNORM16 vs GPU decode_unorm8 rounding).
- Overhead per `vkCmdPushConstants` (2M calls, host): graphics-only 5.8 → 8.6 ns, compute stage 7.6 → 13.3 ns.
- Guest (2026-10-10): astcprobe as 7.2/7.3; ASTC caselist 34,522 cases 0 fail; `dEQP-VK.api.*` 327,798 cases
  0 fail / 4 QualityWarning; Endfield past GfxDevice init.

### 7.8 ETC2/EAC (added 2026-10-18)

The same machinery emulates the 10 ETC2/EAC formats (`VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK` ... `VK_FORMAT_EAC_R11G11_SNORM_BLOCK`).
Neither driver mode has them on desktop GPUs (NVIDIA 615: `textureCompressionETC2 = 0`; NVK `nvk_physical_device.c`:
`textureCompressionETC2 = false`), while Android's Vulkan baseline (`VP_ANDROID_baseline_2021`) requires the feature, and
Chrome's Skia compositor refuses its ETC1 tab thumbnails without `ETC2_R8G8B8_UNORM` (white tab strip). ASTC and ETC2 are
probed and decided independently (`nvhal_astc_pd.state` / `.etc2_state`).

| Item | Behaviour | Where |
|---|---|---|
| Probe | `R8G8B8A8_UINT`, `R16_UINT`, `R16G16_UINT` STORAGE; `R32G32_UINT` and `R32G32B32A32_UINT` SAMPLED+TRANSFER; `shaderStorageImageWriteWithoutFormat`; 8x8 workgroups; a compute queue; each decoded format has SAMPLED\|BLIT_SRC\|FILTER_LINEAR; else one error "(ETC2 gap)" and nothing advertised | A `probe_etc2` |
| Advertised | `textureCompressionETC2 = 1`; optimal features = the decoded format's & (SAMPLED\|BLIT_SRC\|FILTER_LINEAR\|TRANSFER_SRC\|TRANSFER_DST\|FILTER_MINMAX), no linear/buffer features, no host image transfer; image format properties as for ASTC (7.2) | A `fmt_features`, `ifp_check` |
| Decoded image (the app's handle) | Mesa `vk_texcompress_etc2_emulation_format`: RGB8/RGB8A1/RGBA8 -> `R8G8B8A8_UNORM`/`_SRGB`, R11 -> `R16_UNORM`/`_SNORM`, RG11 -> `R16G16_UNORM`/`_SNORM` | A `efmt_get` |
| Block images | one per mip, one texel per 4x4 block: `R32G32_UINT` (RGB8, RGB8A1, R11: 64-bit blocks), `R32G32B32A32_UINT` (RGBA8, RG11) | A `efmt_get` |
| Decoder | `src/etc2_decoder.comp` (written from the Khronos Data Format Specification's ETC2 chapter), one invocation per texel, 8x8 workgroups, specialization KIND (RGB8, RGB8A1, RGBA8, R11, RG11) x SIGNED; writes through `R8G8B8A8_UINT` / `R16_UINT` / `R16G16_UINT` views; push `{start, end}` (16 B); same barriers and compute-state restore as 7.5 | A `etc2_pipeline`, `record_decode` |
| Views | formats of the image's decode class only: ETC2 UNORM/SRGB twins (same bytes); EAC UNORM and SNORM decode to different values, so an R11 UNORM image with an R11 SNORM view (or format list entry) is refused, like block-texel views | A `build_cis`, `nvhal_astc_CreateImageView` |
| Copies | as ASTC (7.4); between two emulated images the decoded texels are copied only within one decode class, otherwise the destination's blocks are decoded (e.g. ETC2 RGBA8 blocks copied into an EAC RG11 image decode as RG11) | A `nvhal_astc_cmd_copy_image` |

Test: `test/etc2-roundtrip.c` (reference: Mesa's CPU decoder `src/mesa/main/texcompress_etc.c` at the pinned commit,
`test/etc2-ref.c` + stand-in headers `test/etc2-ref/`): queries incl. the dEQP image_format_properties loop, refusals,
12 pixel cases (all 10 formats, NPOT sizes with partial edge blocks, mip chains to 1x1, layers, 2 sparse-bound),
blocks steered so that every ETC2 mode (individual, differential, T, H, planar, punchthrough transparent) and every EAC
corner (multiplier 0, signed base -128) occurs; every texel must match bit for bit (RGBA8 bytes, R11/RG11 as 16-bit
UNORM/SNORM), raw readback bit-exact, same-format / 64-/128-bit UINT / cross-class copies, blit; API 1.3 and 1.0.
Host (RTX 5090, 615.71.09): 1,279,739 texels bit-exact per run, 0 differences, 64- and 32-bit.

## 8. Queue-timeline start at vkCreateDevice (added 2026-10-11; NVIDIA defect D1)

The defect and the proof are in `../egl/FENCE-CONTRACT.md` §3.5 and §6a; this section is the HAL's half of the
implemented contract. Code: `src/device.c` (`start_queue_timelines`, `start_one_queue`), called from
`nvhal_CreateDevice` before the device is recorded and returned.

### 8.1 What the driver does (measured, 615.71.09, RTX 5090, raw ICD, `test/vk-first-wait.c`, 24 rounds per mode)

| Fact | Evidence |
|---|---|
| One timeline per **VkQueue** (family, index), created with the device | ioctl log: `SEMSURF_CTX_CREATE` count at vkCreateDevice + vkGetDeviceQueue = 3 + number of queues (1 queue → 4, 2 → 5, 4 → 7, 16 → 19) |
| A start on one queue does not cover another queue of the same family | `mq-warm-g0-use-g1` 24/24 early, `mq-warm-g0-use-g15` 24/24 early; control `mq-warm-g1-use-g1` 0/24 |
| ... nor a queue of another family | `mq-warm-g0-use-c0` 24/24 early (graphics family 0, compute family 2); `mq-fresh-c0` 24/24 |
| Starting every queue covers all of them | `mq-warm-all-use-c0`, `mq-warm-all-use-g1` (graphics x2 + compute + transfer): 0/24 |
| The start is per queue, not per thread | `warm-thread` (start in another thread, which is joined): 0/24 |
| `vkQueueWaitIdle` / `vkDeviceWaitIdle` after the start do not reset it | `warm-qidle`, `warm-didle`: 0/24 |
| `vkGetDeviceQueue2` with flags 0 returns the same VkQueue as `vkGetDeviceQueue` | `warm-q2`: same handle 24/24, 0/24 early. No family of this GPU has `VK_QUEUE_PROTECTED_BIT` and `protectedMemory` is false, so no queue with flags ≠ 0 can be created here (not measurable) |
| A semaphore export starts a timeline like a fence export | `sem-warm`, `sem-warm-sem-export`: 0/24 |
| Unstarted fresh queue, first op an imported-fd wait | `sem-export`, `fence-export`, `hal-seq`, `mq-fresh-g1`: 24/24 early, the work ending 10.5–12.6 ms after the exported fd was seen signalled |

### 8.2 The contract

- **Where**: inside `vkCreateDevice`, after the ICD created the device and before the HAL returns it. No app thread
  can hold the VkDevice or any of its queues yet, so no app wait can precede the start, and nothing races it.
  `vkGetDeviceQueue{,2}` were rejected as the place: they return `void` (a failure could not be reported), any app
  thread may call them at any time, and the timelines exist from device creation on.
- **Which queues**: every queue of every `VkDeviceQueueCreateInfo` (`queueCount` of them per family; a create info
  with `flags ≠ 0` is fetched with `vkGetDeviceQueue2` and those flags, as the spec requires).
- **What**: per queue, one empty `vkQueueSubmit` (one `VkSubmitInfo`, nothing in it) with an exportable fence,
  `vkGetFenceFdKHR(SYNC_FD)`, `poll()` on that fd (≤ 5 s) so the fence is idle before `vkDestroyFence`, `close`.
  Without `VK_KHR_external_fence_fd` enabled: the same with a binary semaphore the submit signals and
  `vkGetSemaphoreFdKHR(SYNC_FD)`. Both are fence creations with `V == 0` (§6a): `started := 1`, forced release.
- **Which devices**: those that enable `VK_KHR_external_semaphore_fd` or `VK_KHR_external_fence_fd` (after the HAL's
  own additions; `VK_ANDROID_native_buffer` adds both). A device with neither can neither import a SYNC_FD wait (the
  only operation that advances V without `started`) nor export a sync_fd (the only place a poisoned value would be
  observed), so it is left alone.
- **Failure is loud**: any `VkResult` error, an exported fd of −1 (the driver would have skipped the fence creation
  that sets `started`), a missing entry point, or a sync_fd that does not signal within 5 s fails `vkCreateDevice`
  with that error (or `VK_ERROR_INITIALIZATION_FAILED`), logged with the family/index and the reason; the device is
  destroyed. There is no fallback.
- `NVHAL_TRACE=1` / `debug.nvhal.trace` logs per device: "N queue timeline(s) started (fence export, X ms)".

### 8.3 Cost (host, 200 device creations each, `vk-first-wait --bench-create`, the HAL's own timer)

| Queues created | Start cost: median / p90 / max | vkCreateDevice + vkGetDeviceQueue (whole), ICD vs HAL before vs HAL after, median |
|---|---|---|
| 1 (graphics) | 0.069 / 0.261 / 0.407 ms | 63.3 / 61.8 / 63.5 ms (medians of two runs each, averaged) |
| 4 (graphics x2, compute, transfer) | 0.244 / 0.406 / 1.618 ms | 121.6 / 123.4 / 124.8 ms |
| 16 (graphics) | 1.887 / 2.332 / 8.600 ms | 615–630 / 537–636 / 494–522 ms (two runs; the spread between runs is larger than the start) |

The start is below the run-to-run spread of device creation itself (the ICD spends ~20–37 ms per created queue).

### 8.4 Tests

- `make hal-fence-check` (`HAL_FENCE_ROUNDS`, default 20; measured with 200): every `vk-first-wait` mode through the HAL, plus the
  semaphore-only start (`--no-fence-ext --no-anb`). Pass bar: 0 early in every mode.
- `make driver-check`: the same modes through the **raw ICD** (`DRIVER_CHECK_ROUNDS`, default 10). It detects the
  driver defect and therefore fails on 615.71.09; it passes only on a driver that keeps the contract.
- `../egl` host stress, `vk-write` / `vk-read` consumers (`../egl/README.md` "Tests"): a Vulkan compositor through
  this HAL on the window's dma-bufs.

## 9. NV12/P010 (decoder output) import (added 2026-10-14)

Sources: Mesa `src/vulkan/runtime/vk_android.c` `get_ahb_buffer_format_properties2` (pin 0b97dcdd, lines 720-852),
`src/util/u_gralloc/u_gralloc_imapper4_api.cpp` `mapper4_get_buffer_color_info` (dataspace mapping), minigbm
16844ba `cros_gralloc/gralloc4/CrosGralloc4Metadata.h` + `CrosGralloc4Mapper.cc:1033-1070` (metadata at the start of
the reserved region), gralloc patch `minigbm/0004`, NVK patch `mesa/nvk-ycbcr-linear-modifiers.patch`.

| Item | Behaviour | Where |
|---|---|---|
| Buffer | cros handle fourcc NV12/P010, `num_planes` 2, modifier LINEAR, `fds[1]` the same dma-buf as `fds[0]` (fstat dev/ino), both planes inside the dma-buf; else `VK_ERROR_INVALID_EXTERNAL_HANDLE` with the facts | ahb.c `gbuf_get` |
| Properties | `format` UNDEFINED (not in the equivalence table; Skia requires it, GAU:115-160), `externalFormat` = VkFormat (`G8_B8R8_2PLANE_420_UNORM` / `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16`), `formatFeatures` = the driver's LINEAR 2-plane modifier features (NVK: 0xcfd001, incl. MIDPOINT/COSITED, YCBCR linear filter, separate reconstruction), refused if SAMPLED or a chroma location is missing | `ahb_format_props` |
| Suggested conversion | gralloc4 Dataspace (int32 at offset 1028 of the reserved region, read-only mmap): standard BT709/BT601*/BT2020*, range FULL else narrow (Mesa's mapping) plus the legacy 0x101-0x104 values; unset/unknown: **BT.709 narrow** (Mesa defaults to BT.601; video decoder output is usually HD); chroma MIDPOINT both (minigbm stores no ChromaSiting); components identity | `yuv_suggest` |
| Image | `VkExternalFormatANDROID` (or the VkFormat itself) accepted; realised with `VkImageDrmFormatModifierExplicitCreateInfoEXT` LINEAR, 2 plane layouts from the handle; INPUT_ATTACHMENT (ANGLE asks for it) dropped with the existing LINEAR warning | `parse_ci`, `realise` |
| Ycbcr conversion | `VkExternalFormatANDROID` -> format, struct removed (existing path) | `create_ycbcr` |
| Export | a YUV AHB is never allocated from Vulkan: loud `VK_ERROR_OUT_OF_DEVICE_MEMORY` | `nvhal_AllocateMemory` |
| Driver layout rule | Each plane must sit where the ICD can bind it like a plane of a DISJOINT image of the same pitches: offset a multiple of that plane's `VkMemoryRequirements.alignment`, its required size inside the dma-buf, planes in order without overlap; else `vkAllocateMemory` `VK_ERROR_INVALID_EXTERNAL_HANDLE` with the numbers. NVK: 32 B alignment, exact sizes (its LINEAR import rule, NIL `new_linear`). Stock 615 (measured 2026-10-15, RTX 5090): per plane pitch*rows rounded up to 64 KiB, 64 KiB alignment; the non-disjoint image needs the sum of those, independent of the explicit offsets. Stock binds each plane as its own mapping: a CbCr offset off a 4 KiB page makes RM refuse the plane's mapping node (kernel `intermapRegisterDmaMapping: Failed to insert new mapping node`) while `vkBindImageMemory` succeeds and the plane samples zeros; an import whose dma-buf is smaller than the requirement fails `VK_ERROR_OUT_OF_DEVICE_MEMORY`, closes the fd and leaves later exportable allocations of the device failing. The HAL never imports with less than the requirement | `yuv_layout_bindable`, `nvhal_AllocateMemory` |
| Gralloc layout | minigbm 0004 packed layout (pitch ALIGN(bytes, 256), CbCr at pitch*even(H)), then the wrapper's `get_gbm_layout_ops()->plane_layout()` moves each plane to the driver's per-plane alignment and sizes the buffer (stock: 854x480 NV12 CbCr 524288, 786432 B; NVK: unchanged, 491520 / 737280 B) | gralloc `nvg_plane_layout` |

Test: `test/ahb-yuv.c` (host, fake gralloc mirroring gbm_mesa + 0004 + the wrapper's plane_layout; compute shader samples every luma texel through
the conversion, CPU reference of the spec's reconstruction + model conversion, tolerance 2/255).

