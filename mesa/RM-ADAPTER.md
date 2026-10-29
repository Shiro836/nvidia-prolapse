# Mesa NVK on the Rust RM transport

Mesa owns the Vulkan implementation, shader compiler, image layout, descriptors,
MME programs, command generation and Vulkan object lifetimes. The adapter replaces
only NVK's existing `nvkmd` kernel transport. This preserves mature graphics code
instead of expanding the small experimental Rust Vulkan frontend into a second
implementation of every Vulkan feature.

Pinned source: Mesa `ddaf85e353442d9d7f192fdcf1823e652e3f2ee3` (mesa-26.2.5).
The interface is read directly from `src/nouveau/vulkan/nvkmd/nvkmd.h`, with
Nouveau's existing `nvkmd/nouveau/` implementation as the contract reference.

`nv_rm.h` is the narrow C ABI. The Rust backend owns one selected physical device,
physical allocations, GPU mappings and persistent channels behind one mutex and
runs every call on the caller (no worker; see SYNCHRONIZATION.md). C passes owned
IDs and packet ranges; no `Rc` object escapes that mutex. Mesa's C
adapter owns the DRM render fd, synchronization objects and its VA allocator.

The adapter deliberately uses the NVIDIA DRM render node selected from the
libdrm enumeration and verified by vendor and driver name. It passes the same
PCI identity into Rust RM enumeration. No fixed `/dev/dri/renderD*` is selected.

Memory has two separate lifetimes: the physical allocation and each GPU VA
mapping. NVK allocates an initial plain mapping for every memory object, then
may bind aliases with an image-specific PTE kind. The adapter records independent
mapping IDs so it can unbind aliases without freeing the physical memory.
VA teardown (`va_free`) hands all mapping ids of a reservation to
`nv_rm_va_free` with a release callback; the backend unmaps and calls it to
return the range to the heap only after the GPU work outstanding at free time
retired, so a range is never reused while mapped and no free waits on the GPU
(SYNCHRONIZATION.md, "Resource reclamation").
The C VA allocator owns `[16 GiB, 1 TiB)`; Rust's queue, completion and other
internal allocations remain below 16 GiB. Allocation flags requesting GART use
real cached system memory; they are not silently backed by uncached VRAM.

A channel persists across submissions because NVK initializes graphics and
compute state once (`nvk_queue_init_context_state`) and relies on that state in
later submissions. Recreating the channel for each packet would lose shader
local-memory windows, macro programs and graphics state. Engine masks use NVK's
copy/2D/3D/M2MF/compute bits without translation.

`nvkmd_ctx_exec` accumulates GPU packet addresses and sizes. The Rust submit call
executes the actual NVK packet streams in order and returns only after hardware
completion. Incomplete-method packet fragments remain in the same submission.
No opcode replacement or shader fixture recognition occurs.

`nvkmd_ctx_wait` waits on Mesa's real DRM sync objects outside the Rust worker,
allowing another queue to submit the work that satisfies a dependency. Mesa's
submission threads keep these waits off the application thread. Signal calls
first flush work to completion, then signal the DRM sync object. This initial
transport is synchronous on the GPU worker; it is correct ordering with limited
parallelism, not an asynchronous kernel submission claim.

The stock NVK capabilities must match the transport. `rm-integration.patch` gates
sparse residency, external memory and fixed CPU mapping features; unsupported
paths return named errors. Conformance version is cleared for this unvalidated
transport. DMA-buf import/export uses measured RM/NVKMS/PRIME sharing: both
VRAM and cached system memory passed cross-client aliases, re-export and source
destruction on host and bionic. Blackwell dedicated modifier allocations use
ordinary physical backing with a GENERIC_MEMORY-kind VA, while explicit DRM
modifiers and stride carry the image geometry to consumers. Only uncompressed
2D modifiers are accepted (GOB height log2 0..5); legacy implicit GL tiling is
unsupported. Android presentation remains a separate pixel gate; buffer sharing
alone does not verify it.

The backend-only Rust build disables `prototype-vulkan`. This matters because
NVK exports `vk_icdGetInstanceProcAddr` rather than `vkGetInstanceProcAddr`; a
prototype Vulkan symbol in a dependency could make a direct dlopen test select
the wrong implementation. The RM backend exports only its C transport API.

Source anchors:

- `nvkmd/nvkmd.h`: physical/device/memory/VA/context operation tables.
- `nvkmd/nvkmd.c`: memory refcounts, mapping semantics and buffer range assertions.
- `nvkmd/nouveau/nvkmd_nouveau_mem.c`: each memory object receives a default VA;
  image aliases are separate VA bindings.
- `nvkmd/nouveau/nvkmd_nouveau_ctx.c`: incomplete packet grouping, synchronization
  and submit/flush semantics.
- `nvk_queue.c`: persistent queue state, upload dependency and submit ordering.
- `winsys/nouveau_context.c`: engine class selection by low class byte.
- `vulkan/runtime/vk_sync.h`, `vk_drm_syncobj.c`, `vk_queue.c`: synchronization
  features, DRM timeline implementation and threaded submit behavior.

The native gralloc build (`NVG_SOURCE_RM`) prefers block-linear images where NIL
can describe their GOB layout. For Blackwell R8 and RGB565, NIL has no block-linear
DRM encoding yet; LINEAR is listed independently and selected only if its queried
format features include the requested color attachment, sampling and copy usage.
The default proprietary-driver gralloc build retains its existing layout rule.
`gralloc/out/host/lock-test LIB --source-linear` checks these two formats with
independent Vulkan imports, rendering, CPU access and a separate mapper process.

Current image transport measurements are narrower than presentation support:
RGBA8 and RGBA16F native Vulkan cross-process pixel checks work. The bounded
modifier gate also passed stock EGL import and rendering at four geometries;
modifier selection now respects the image's actual GOB height so consumers see
the layout NIL really allocated. The bounded LINEAR gate passed R8/RGB565 at
four geometries (960415 pixels at each verification stage), including rendering
and cross-process CPU/GPU access. The earlier full lock stress gate found a
persistent-channel submission timeout after the rendering checks. Completion
sequencing, system-memory tail packets and doorbell ordering were corrected,
then 1030 empty completions crossed the 1024-entry ring successfully. The full
source stress rerun passed all rendering checks, 200 allocation/import/map/free
cycles and 200 repeated locks: zero errors, fd71→71, RSS +4 KiB. Lock/unlock
medians were 0.620/0.646 ms at 2560×1412. A live Android desktop remains a
separate integration gate.

Reverse sharing is not yet symmetric. A bounded stock-allocator → NVK-reader
test imported an RGBA8 1001×37 dma-buf (modifier `0x0300000000606012`, pitch 4032,
allocation 262144 bytes), but its initial NVK GPU-VA bind failed with RM status
`0x51` (`NV_ERR_NO_MEMORY`) inside `vkGetMemoryFdPropertiesKHR`, before a GPU
submission. Stock allocation and its CPU staging copy succeeded.
Native allocations →
stock EGL and native → native sharing passed; this result does not invalidate
those directions or establish the reverse failure's cause.

The next diagnostic is to query the imported RM memory's physical attributes
before mapping. Matching public OGKM `ctrl0041.h` defines GET_MEM_PAGE_SIZE
`0x410118` (one output `NvU64`) and GET_SURFACE_INFO `0x410110` (16-byte list
descriptor; 8-byte `{index,data}` entries for attributes 1, size low/high 2/3,
physical attributes 8 and address space 9). The current mapper explicitly asks
for 4-KiB pages. `virt_mem_allocator_gm107.c:738–761` shows RM can override that
for compressible physical memory, so page size, alignment and underlying kind
must be measured before changing mapping flags. No retry or fallback is added.
