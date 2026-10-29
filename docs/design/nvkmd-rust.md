# nvkmd backend in Rust (design)

Status: design, not started. Replaces `mesa/nvkmd_rm.c` (1,353 lines) with a Rust crate that
implements NVK's `nvkmd` vtable and calls the RM through its C ABI. The RM stays one shared
instance per process in `libnv_bionic_native.so`, the router for every driver that uses it: NVK,
the gallium-nvrm GL driver and the Codec2 decoder. Pinned Mesa: `ddaf85e` (26.2.5) plus the patch stack in `mesa/`
(`rm-integration.patch` adds `has_sparse`, `has_rm_compression`, the `notify_lost` hook,
`vk_sync_type::reserve_signal`; the compute-queue patches add `max_shared_ctx`,
`create_shared_ctx`, `NVKMD_MEM_POOL_BLOCK`). Line numbers below are `nvkmd_rm.c` at 833571a
unless a file is named.

## 1. Inventory

### 1.1 Entry points NVK reaches, and what `nvkmd_rm.c` does

Callers are NVK files in `src/nouveau/vulkan/` (through the `nvkmd.h` inline wrappers or the
`nvkmd.c` core functions, which keep refcounts, `map_cnt`, the `mems` list and VM logging).

| vtable entry | NVK callers | `nvkmd_rm.c` behaviour | `nv_rm_*` |
|---|---|---|---|
| `nvkmd_try_create_pdev_for_drm` → `nvkmd_rm_try_create_pdev` (416) | nvk_physical_device.c | PCI + NVIDIA vendor + render node, else `INCOMPATIBLE_DRIVER`; opens render node `O_CLOEXEC`; `drmGetVersion` name must be `nvidia-drm`; `fstat` for `render_dev`; opens RM once to read `NvRmInfo`, closes it; fills `nv_device_info` (incl. `nv_device_info_init_shared_mem_sizes`, `nc_atom_size_B = util_cache_granularity()`, ZCULL when `has_zcull_info`), `nvkmd_info` (`has_dma_buf`, `has_alloc_tiled` for kind 0x06/0xfe, `has_rm_compression` for 0x06, `max_shared_ctx`, `sm_isa_version`, shader pipe counts), `bind_align_B = 4096`, two sync types | `open`, `close` |
| pdev `destroy` (326) | nvk_physical_device.c | close fd, free | — |
| pdev `get_vram_used` (333) | nvk_physical_device.c | returns 0 (`has_get_vram_used` stays false) | — |
| pdev `get_drm_primary_fd` | nvk_wsi.c | not set (NULL → -1) | — |
| pdev `create_dev` (360) | nvk_device.c | reopens render node, opens RM, VA window `[1<<34, 1<<40)`, `mems` list + mutex, `va_mutex` + `util_vma_heap`; when `log->device` is a logical device: installs `check_status`, `notify_lost`, `notify_lost_data = rm` | `open` (+`close` on failure) |
| dev `destroy` (563) | nvk_device.c | clears the three hooks, closes RM (runs deferred VA releases into the heap), closes fd, `util_vma_heap_finish` | `close` |
| dev `get_gpu_timestamp` (585) | nvk_device.c | 0 + stderr line on failure | `timestamp` |
| dev `get_drm_fd` (594) | nvk_device.c | render node fd | — |
| dev `alloc_mem` / `alloc_tiled_mem` (861, 872, policy in 769) | nvk_device_memory.c, nvk_queue.c, nvk_mem_arena.c, nvk_device.c, nvk_cmd_draw.c (+ `alloc_mapped_mem` users) | placement: GART / VRAM / POOL_BLOCK (exactly 2 MiB, kind 0, not GART/mappable/shared, align 2 MiB) / VRAM_COMPRESSIBLE (kind 0, VRAM, not mappable/shared, generic kind 0x06, 64 KiB multiple); VRAM rounded to 64 KiB size and alignment; tiled: kind must be the generic kind, tile `& ~0x70 == 0`, `tile >> 4 <= 5`, pre-Turing mappable block-linear VRAM refused; `nvkmd_mem_init(flags | COHERENT, size, 4096)`; own VA via `nvkmd_dev_alloc_va` + `nvkmd_va_bind_mem`; rollback frees and publishes loss if the free fails | `alloc`, `free` |
| dev `import_dma_buf` (904) | nvk_device_memory.c | borrows the fd; flags `LOCAL|SHARED|CAN_MAP|COHERENT`; own VA, align 4096 | `import`, `free` |
| dev `alloc_va` (599) | nvk_buffer.c, nvk_image.c, nvk_mem_arena.c | `SPARSE` unsupported; size ≤ window−4095, rounded to 4 KiB, align ≥ 4 KiB power of two; two heaps as `nvkmd_nouveau` (`nvkmd_nouveau_dev.c:51-58`): `[16 GiB, 512 GiB)` ordinary, `[512 GiB, 1 TiB)` for `REPLAY`; `ALLOC_FIXED` only with `REPLAY` (else unsupported), inside the replay range via `util_vma_heap_alloc_addr` (outside or taken: `INVALID_OPAQUE_CAPTURE_ADDRESS`), else `util_vma_heap_alloc`; heaps under `va_mutex`; a released range returns to the heap of its address | — |
| dev `create_ctx` / `create_shared_ctx` (1326, 1334, 1285) | nvk_queue.c, nvk_upload_queue.c | `BIND` engine → no channel; else context (shared: share must be a channel ctx of the same dev) | `context_create`, `context_create_shared` |
| mem `free` (950) | via `nvkmd_mem_unref` (10 files) | `nvkmd_va_free`, RM free (loss on failure), destroy `map_mutex` | `free` |
| mem `map` (962) | via `nvkmd_mem_map` (nvk_heap.c, nvk_host_copy.c, nvk_device_memory.c, nvk_mem_arena.c) | fixed maps unsupported; maps the shared lazy CPU mapping; `DEVICE_LOST` or RM-lost → loss; OOM statuses pass through, rest `MEMORY_MAP_FAILED`; NULL address → unmap + `MEMORY_MAP_FAILED` | `map`, `unmap`, `is_lost` |
| mem `unmap` (998) | via `nvkmd_mem_unmap` | loss on failure | `unmap` |
| mem `overmap` (1009) | nvk_device_memory.c | unsupported (`has_overmap` false) | — |
| mem `sync_to_gpu` / `sync_from_gpu` (1031) | via `nvkmd_mem_sync_*` | abort: every allocation is `COHERENT`, so `nvkmd.c` never calls them | — |
| mem `export_dma_buf` (1020) | nvk_device_memory.c | | `export` |
| mem `log_handle` (1038) | nvkmd.c (NVK_DEBUG_VM) | allocation id, truncated to 32 bits | — |
| va `free` (733) | nvk_mem_arena.c, nvk_buffer.c, nvk_image.c | hands every binding to one deferred VA free; the range returns to the heap from `rm_va_release` (719) on whichever thread retires it; loss on failure (range stays reserved); stack array of 16 ids, calloc above | `va_free` |
| va `bind_mem` (679) | via `nvkmd_va_bind_mem` | range checks, unbind overlap first, bind with the VA's kind, append binding | `bind` |
| va `unbind` (645) | via `nvkmd_va_unbind` / ctx `bind` | partial overlap unsupported; immediate unbind of covered bindings in list order | `unbind` |
| ctx `destroy` (1261) | nvk_queue.c, nvk_upload_queue.c | discards unsubmitted pushes, bounded context destroy (stderr on failure) | `context_destroy` |
| ctx `wait` (1155) | nvk_queue.c | flushes pending pushes first; bind ctx: CPU `vk_sync_wait_many(COMPLETE, UINT64_MAX)`; else records `(id, value)` (skips timeline value 0; non-RM sync unsupported) | — (deferred to submit) |
| ctx `exec` (1186) | nvk_queue.c, nvk_mem_stream.c | validates 4-byte alignment, `0 < size < 8 MiB`; appends `NvRmExec{addr,size,flags}` | — |
| ctx `bind` (1205) | nvk_buffer.c, nvk_image.c | flush, then `nvkmd_va_bind_mem`/`nvkmd_va_unbind` per bind | — |
| ctx `signal` (1225) | nvk_mem_stream.c, nvk_queue.c | bind ctx: flush + `vk_sync_signal` each; else builds points (calloc per call), flush with signals | — |
| ctx `flush` (1136) / `sync` (1142) | not called by NVK at this pin (only from the other ctx entries) | flush: validates every 1000-entry chunk boundary (no `INCOMPLETE` last entry) first, empty submit when only waits/signals, submits chunks, signals on the final chunk; on `VK_NOT_READY` waits for the oldest token (`UINT64_MAX`) and retries; keeps only the unaccepted suffix. sync: flush + wait last token | `submit_batch`, `wait` |

### 1.2 The two `vk_sync_type`s (284-323)

`rm_sync_type` (binary + timeline, opaque fd import/export) and `rm_sync_file_type` (binary,
sync_file import/export); both copied into the pdev and listed in `sync_types`. Payload
`struct nvkmd_rm_sync { struct vk_sync base; uint64_t id; }` (the id is the RM's payload
reference). The type check used by ctx `wait`/`signal` is `sync->type->init == rm_sync_init`.

| op | behaviour | `nv_rm_*` |
|---|---|---|
| `init` | flags map 1:1 (static asserts 142-144) | `sync_create` |
| `finish` | drop payload ref; failure → stderr + `vk_device_set_lost` | `sync_destroy_deferred` |
| `signal`, `get_value` (timeline type only), `reset`, `move` (shared flag of src) | pass-through | `sync_signal`, `sync_get_value`, `sync_reset`, `sync_move` |
| `reserve_signal` | `*reserved = ok && done` | `sync_reserve` |
| `wait_many` | `rm_wait_status` before and after; stack array of 16 points (calloc above); deadline → timeout via `os_time_get_nano`; 2 → `VK_TIMEOUT` | `sync_wait_many` |
| `import_opaque_fd` / `export_opaque_fd` | borrow / return owned fd (`opaque = 1`) | `sync_import_fd`, `sync_export_fd` |
| `import_sync_file` / `export_sync_file` | same, `opaque = 0`, export passes the render fd | same |

### 1.3 Status handling

- `rm_result` (73): 0 → `VK_SUCCESS`, else the negative status *is* the `VkResult`, logged with
  `vk_errorf` ("Native Rust RM %s failed"). The mapping from RM errors to statuses lives in
  `rm/src/device/capi.rs` `vk_status` and `Entry` (per Vulkan's per-command error lists).
- `unsupported` (80): `VK_ERROR_FEATURE_NOT_PRESENT`, named.
- `rm_mem_lost` (92): void cleanup paths: stderr, then `vk_device_set_lost` when a logical
  device exists, else `nv_rm_mark_lost`; returns `DEVICE_LOST`.
- `rm_wait_status` (105): Mesa's lost counter (`vk_device_is_lost_no_report`), then
  `nv_rm_is_lost` → `vk_device_set_lost`. `rm_check_status` (339) does the same through
  `nv_rm_check_status`. `rm_notify_lost` (353) mirrors Mesa-side loss (`vk_device.c` and
  `vk_queue.c` both call `vk_device_notify_lost`) into `nv_rm_mark_lost`.
- Since 833571a the RM side publishes loss itself whenever an entry returns
  `VK_ERROR_DEVICE_LOST` (the `guarded` wrapper in `rm/src/device/capi.rs`), so Mesa sees it on its next
  status or wait query.

### 1.4 What else uses `nv_rm.h`

`nvkmd_rm.c` uses 30 of the 39 entries. The other nine (`sync_destroy`, `sync_status_many`,
`submit_sync`, `context_signal`, `submit_waits`, `submit`, `poll`, `context_fence`, `exec`) are
used by the gallium-nvrm GL driver (18 entries in all) and by tests (`rm/src/device_test_ffi.rs`,
`mesa/bridge-check.c`). The same library exports `nv_video_*` for Codec2. Every entry stays
exported: the port changes who calls the ABI, not the ABI.

## 2. Shape of the Rust side

### 2.1 Crates (RUST.md §1)

| crate | role | unsafe |
|---|---|---|
| `nv-rm-abi` | the RM's C ABI as Rust: `#[repr(C)]` structs (`Info`, `Allocation`, `SyncPoint`, `Push`), typed handles (`#[repr(transparent)]` `MemoryId`, `MappingId`, `ContextId`, `SyncId`, `Token` over `u64`; the device as an opaque `NvRmDevice`), the `VaRelease` callback type, and one `extern "C"` declaration per entry. cbindgen generates `mesa/nv_rm.h` from it | none (types + `extern` decls) |
| `nvkmd-sys` | bindgen output of the pinned, patched headers: `nvkmd.h`, `vk_sync.h`, `nv_device_info.h`, `util/vma.h`, `xf86drm.h` (`drmDevice`, `drmVersion`), the called functions of `vk_log.h`/`vk_device.h`; `vk_device`, `vk_object_base` and `nvk_device` opaque, with the members the backend reaches (`nvk_device.vk`/`.nvkmd`, `vk_device._lost.lost`/`notify_lost`/`notify_lost_data`/`check_status`, `vk_object_base.device`) as generated offsets and member types (`nvkmd-sys/wrapper.h`). Generated per target, committed. No logic | none (types + `extern` decls) |
| `nvkmd-rm-logic` | pure policy, `#![forbid(unsafe_code)]`: placement/size/alignment choice, tiled-kind validation, VA request validation, binding-list overlap rules, push validation and the 1000-entry continuation chunking with accepted-prefix retention, wait filtering, `Info` → device/KMD info derivation | forbidden |
| `nvkmd-rm` | boundary, `staticlib`: the ops tables and `vk_sync_type`s, `extern "C"` entries, object recovery, calls into Mesa C and into `libnv_bionic_native` through `nv-rm-abi` | all of it, in one module `ffi` |

Bindings are generated by `mesa/gen-nvkmd-bindings.py` with the bindgen version Mesa's build
already pins, from the exact compile flags of `nvkmd.c` in each target's
`compile_commands.json` (layouts depend on `HAVE_FUTEX` and pointer width), for
`x86_64-linux-android`, `i686-linux-android` and the host. bindgen's generated layout
assertions are the compile-time asserts of §4 on every target; `mesa/build.py` regenerates
after `meson setup` and fails when the committed file differs (stale bindings after a pin
bump cannot build). Committing them keeps `cargo test --workspace` independent of a Mesa build.

`nv_rm.h` is generated, never hand-written: `mesa/gen-rm-header.py` runs the cbindgen Mesa's
build pins over `nv-rm-abi` and the header is committed with a generated banner;
`mesa/build.py` regenerates it and fails on a difference. The `rm` crate's exported definitions are pinned to the declarations at
compile time (`const _: nv_rm_abi::SyncSignal = capi::nv_rm_sync_signal;` per entry, the fn
pointer type aliases living next to the declarations), so a definition and its declaration
cannot drift. C consumers (gallium-nvrm, the C backend until it is deleted, the C tests)
include the generated header.

### 2.2 Objects across the boundary

Each NVK-visible object is a `#[repr(C)]` Rust struct whose first field is the bindgen base
struct, allocated in Rust and handed to Mesa as `&mut base`:

| object | Rust owner | created / destroyed | holds |
|---|---|---|---|
| `RmPdev` | `Box`, raw in Mesa | `try_create_pdev` / pdev `destroy` (the only `Box::from_raw`) | render fd (`OwnedFd`, `O_CLOEXEC`), `CString` render path, `Info`, the two `vk_sync_type`s and the NULL-terminated `sync_types` array |
| `RmDev` | `Box`, raw in Mesa | `create_dev` / dev `destroy` | `RmDevice` (owns the `NvRmDevice` handle; `Drop` = `nv_rm_close`), `Option<NonNull<vk_device>>` (logical device only), render fd, `Mutex<VaHeaps>` (the ordinary and the replay `util_vma_heap`, initialised in place) |
| `RmMem` | `Box`; lifetime is Mesa's refcount (`nvkmd_mem_unref` → `free`) | alloc / import / `free` | the `NvRmAllocation` (id, size); the RM free runs once, in `free` or in the creation rollback |
| `RmVa` | `Box` | `alloc_va` / va `free` | the bound ranges and their `MappingId`s as two parallel `Vec`s in bind order (C list order, so unbind order is preserved); `free` hands the id array to `nv_rm_va_free` as it is, with no per-call bookkeeping allocation |
| `RmCtx` | `Box` | `create_ctx` / ctx `destroy` | `Option<rm::ContextId>` (None = bind ctx), `Vec<Push>`, `Vec<WaitPoint>`, `Vec<SignalPoint>` scratch (capacity kept), last `Token` |
| `RmSync` | in place in Mesa's `vk_sync` allocation (`size = size_of::<RmSync>()`) | `init` writes the `SyncId` from `nv_rm_sync_create`; `finish` passes it to `nv_rm_sync_destroy_deferred` | the `SyncId` |

Recovering our type from a Mesa pointer checks the ops pointer first (the Rust form of
`NVKMD_DECL_SUBCLASS`'s assert, nvkmd.h:399-406) and returns `Err`/the entry's failure value
on mismatch; for syncs the check is `sync.type_` against the two types stored in our pdev.
Device-from-`vk_device` (`rm_sync_device`, 125) stays `container_of(vk, nvk_device, vk)->nvkmd`
using bindgen offsets.

Destroy order is written out: dev `destroy` clears `notify_lost`/`notify_lost_data`/
`check_status`, closes the RM device (which may still release VA ranges into the heap through
the `VaRelease` callback, whose data pointer is that mutex: `nv_rm_close` runs every outstanding
release before it returns), closes the fd, then finishes the heaps (as 563-583). Mesa's inline-only helpers (`simple_mtx_init/destroy`, `list_inithead`) get three
small non-inline functions in `nvkmd.c` next to the existing `nvkmd_mem_init`:
`nvkmd_dev_init`, `nvkmd_dev_finish`, `nvkmd_mem_finish`, and
`nvkmd_dev_info_init_shared_mem_sizes` for `nv_device_info.h`'s inline shared-memory table
(~25 C lines in `rm-integration.patch`; forcing interface: Mesa's `static inline` mutex/list API in
`util/simple_mtx.h`, `util/list.h` and `nv_device_info.h`).

Calls into Mesa C from Rust: `__vk_errorf`, `_vk_device_set_lost` (variadic, always with a
literal `"%s"` format), `nvkmd_mem_init`, `nvkmd_dev_alloc_va`, `nvkmd_va_bind_mem`,
`nvkmd_va_free`, `nvkmd_va_unbind`, `vk_sync_wait_many`, `vk_sync_signal`, `util_vma_heap_*`,
`nv_device_info_init_shared_mem_sizes`, `util_cache_granularity`, `drmGetVersion`/
`drmFreeVersion`. Mesa builds with hidden visibility, so these resolve only at static link
time: the crate must be linked into `libvulkan_nouveau.so` (a staticlib), it cannot be a
separate cdylib. The RM entries are the opposite case: they are dynamic symbols of
`libnv_bionic_native.so`, which `libvulkan_nouveau.so` keeps linking (`-lnv_bionic_native`), so
every driver in the process reaches the same RM instance. Mesa's lost counter (`vk_device_is_lost_no_report`, inline:
`p_atomic_read(&device->_lost.lost) > 0`) is read as `AtomicI32::from_ptr` at the bindgen
offset with `Acquire`.

### 2.3 Where `unsafe` lives

Only `nvkmd-rm/src/ffi/`: the `extern "C"` entries and `vk_sync_type` callbacks, pointer
validation and object recovery, the in-place sync payload, `extern` calls into Mesa/libdrm and
the RM ABI, the fallible `Box` helper below, and `unsafe impl Send/Sync` for `RmDev` (raw `vk_device` pointer:
Vulkan requires the device to outlive every nvkmd call, and Mesa's notify hook contract permits
calls from any thread). Every entry runs inside the shared panic guard (moved from `rm::ffi`
into a small `nv-ffi` crate so the RM and the backend use one implementation, each linking its
own copy; the guard reports through a sink type each library supplies, `nv_ffi::PanicLog`). Failure values: the entry's `VkResult`
(`VK_ERROR_DEVICE_LOST` for a panic, device marked failed); void callbacks that have a logical
device publish loss through `vk_device_set_lost`, the same channel their RM-failure path uses
today; the rest (`pdev destroy`, `get_vram_used`, mem `sync_*`) abort with the message.

C returns `VK_ERROR_OUT_OF_HOST_MEMORY` when `calloc` fails; `Box::new` aborts and
`Box::try_new` is unstable on 1.99.1 (checked). A `try_box` helper in `ffi` (`alloc` + null
check) keeps the C result; app-sized arrays use `Vec::try_reserve`.

### 2.4 The RM side of the ABI

The RM crate keeps the device logic in `rm::device` (`rm/src/device.rs`: `Bridge`, `Core`,
`Inner`, submit, sync, reclaim) and the exported entries in its child module `rm::device::capi`
(`rm/src/device/capi.rs`, implementing the `nv-rm-abi` declarations); as a child, `capi` reads
the device's private items without widening them. Next, the entry bodies move behind a Rust API
of the device (typed handles, `GpuVa`/`ByteSize`, `Result<T, rm::Error>`, sync payloads as
references instead of `Arc::into_raw` integers, which removes most of the `unsafe` in
`sync_ref`/`submit`), leaving `capi` with pointer validation, the guard and the status mapping.
The exported symbol list, struct layouts and statuses stay as they are: loss publication stays
in the entry guard (833571a), the `Error → VkResult` mapping per entry stays in `capi.rs`, and
the VA release stays the `VaRelease` callback (called once per range, after the RM unmapped it).

The backend calls the ABI directly (no trait in between): production links
`libnv_bionic_native.so`; the differential harness (§4.1) links the same fake `nv_rm_*`
functions it gives the C backend. The `nv-rm-abi` declarations are the written list of what the
backend needs from the RM.

## 3. Build integration

NIL and NAK show how Mesa links Rust into NVK: a Rust `static_library` with `rust_abi: 'c'`
(`src/nouveau/nil/meson.build:78-86`, `src/nouveau/compiler/meson.build:92-120`), bindings via
`rust.bindgen` (`nil/meson.build:41-69`, `compiler/meson.build:43-73`), reached by NVK through
`idep_nil`/`idep_nak` (`vulkan/meson.build:143-144`). Both staticlibs already link into
`libvulkan_nouveau.so` together (checked in the Android build: two `crate-type staticlib`
targets). Our crate depends on Cargo packages (`nv-rm-abi`, `nv-ffi`), so Cargo builds it (as
it builds `rm` today in `mesa/build.py:344-357`); meson only links the archive:

- `rm-integration.patch` adds a meson option `nvk-rm-backend` (`c` | `rust`) and
  `nvk-rm-rust-lib` (path). `c`: `nvkmd/rm/nvkmd_rm.c` compiled as now. `rust`: no C source;
  `libvulkan_nouveau` gets `link_args: [<path>/libnvkmd_rm.a, '-Wl,--exclude-libs,libnvkmd_rm.a']`
  so nothing from our archive (Rust std, `nv-ffi`) enters the dynamic symbol table.
- Both backends keep `-lnv_bionic_native` on `libvulkan_nouveau`: the archive holds the
  backend and `nv-rm-abi`'s declarations only, never the `rm` crate, so a process has one RM
  (one RM client, fork check and logger) whichever of NVK, gallium-nvrm and Codec2 it loads.
- `mesa/build.py --nvkmd {c,rust}` (default `c` until the gate), builds `-p nvkmd-rm
  --target <triple>` first (into `build/nvkmd-rm-rust`), passes the archive path, and uses
  separate build dirs (`build/nvk-<target>` for `c`, `build/nvk-<target>-rust` for `rust`) so
  both builds coexist for A/B; the host build that provides `mesa_clc` to the Android builds
  stays the `c` one. `tests/vulkan/run.py` and `capture.py` take the same switch.
- Link check (all three targets; lld for Android, GNU ld for the host): the archive's Rust std
  objects coexist with NAK's and NIL's only when the archive is built without LTO and by the
  same rustc as Mesa's Rust: it then carries the same std objects, and the linker takes each
  once (one `panic_unwind` runtime, since Mesa builds its Rust with the default unwind
  strategy). With the workspace's thin LTO, both linkers fail on duplicate std symbols, so the
  archive uses the `release-staticlib` profile (`lto = false`). Mesa's hidden functions resolve
  at the static link, the dynamic symbol table stays that of the C build, and the backend's
  symbols are local. One consequence: the Rust std in `libvulkan_nouveau.so` is one copy shared
  by NAK, NIL and the backend, so process-wide std state there (the panic hook) is shared too.
- Symbol tests: `nm -D --defined-only libvulkan_nouveau.so` of the two builds is identical
  (the C build already leaks ~2,300 Rust symbols from NAK/NIL; ours adds none), and its
  undefined `nv_rm_*` set under the Rust backend is a subset of the C one. The only global C
  symbol in `libnvkmd_rm.a` is `nvkmd_rm_try_create_pdev` (`create_shared_ctx` is reached
  through the ops table only; no NVK file calls `nvkmd_rm_create_shared_ctx`).
  `libnv_bionic_native.so` exports exactly the generated header's entries plus `nv_video_*`.
- Load time (§13): no constructors, no threads; the RM's lazily started watcher and the
  `pthread_atfork` registration behave as in `libnv_bionic_native.so` today.

## 4. Verification

### 4.1 Differential, entry level: the offline fake-RM harness

`nvkmd-harness` (workspace crate, binary behind its `mesa` feature) drives a backend only through
Mesa's public nvkmd interface: `nvkmd_try_create_pdev_for_drm`, the ops tables, the `vk_sync_type`s
and the hooks a backend installs in a `vk_device`. Static-inline Mesa helpers it needs have Rust
equivalents in `nvkmd-harness/src/mesa.rs` (`vk_device_is_lost_no_report` as an atomic load,
`list_inithead` of the stand-in device's queue list). `mesa/tests/run-nvkmd-harness.py` links it
once per backend against that backend's host NVK build (`build/nvk-host`, `build/nvk-host-rust`),
with libc's render-node and `calloc`/`free` calls, libdrm and Mesa's diagnostic sinks replaced
through `--wrap` (C variadic definitions are stable Rust) and the `nv_rm_*` entries defined by a
fake RM that records every call. Injected failures (`open`, `fstat`, the version query, a chosen
`calloc`, RM statuses, values wider than Mesa's fields) cover error paths no GPU run reaches;
diagnostics are compared by content class (the harness checks a word of Mesa's message).

Gate per step: every suite passes on both backends and the call log (every RM, render-node and
libdrm call with arguments and result, plus each case's outcome and the filled device info and
sync type shapes) is byte-identical between them, suite by suite. Suites: `pdev`, `info`, `loss`,
`memory-api`, `memory-cleanup`, `sync-api`, `wait-loss`, `submit-fusion`.

The C preflight harness (`mesa/tests/pdev-preflight.c`, which includes `nvkmd_rm.c`) has no call
log and builds backend-private objects directly (`nvkmd_rm_mem`, `nvkmd_rm_ctx`, `nvkmd_rm_sync`
with chosen ids, the pending push array), so its scenarios are re-expressed through the public
interface rather than replayed: objects come from `alloc_mem`/`create_ctx`/`vk_sync_init`, state
is observed through later calls. Its backend suites move into the Rust harness with the steps
that port their entries (`memory-api`, `memory-cleanup`: 6; `sync-api`, `wait-loss`: 7;
`submit-fusion`: 8); its tests of Mesa's runtime and our patches to it, which reach no backend
entry (`submit-on-demand`, `sync-properties`, the queue part of `loss-notification`, the
`nvk_mem_stream` and sync-provider checks), stay a C test of the Mesa build. The C backend's part
of it is deleted once the last backend suite has moved.

### 4.2 Differential, system level: capture

`tests/vulkan/capture.py record` on each build, then `capture.py diff` C vs Rust: the ioctl
sequence and every pushbuffer byte for the 42 cases (same build self-diff is 0/42 today). Gate:
0/42. VA addresses appear in pushbuffers, so the first port keeps `util_vma_heap` (called
through FFI) to get identical addresses by construction. `drmGetVersion` stays the libdrm call
for the same reason (its two `DRM_IOCTL_VERSION` calls are in the trace). Non-ioctl syscalls
(`fstat` vs `statx` from `std::fs`) are not compared and may differ.

### 4.3 Per-entry unit and ABI tests

- `nvkmd-rm-logic`: table tests per policy function with expected values taken from the C
  branches (each test names the `nvkmd_rm.c` line it pins): placement for every flag/kind/size
  combination incl. 64 KiB rounding and the pool block, tiled-kind acceptance, `alloc_va`
  bounds incl. `size > window - 4095` and fixed-address alignment, partial-overlap detection,
  chunking at 999/1000/1001 with trailing `INCOMPLETE` runs and prefix retention after a
  failure at chunk k. Full Miri (forbid-unsafe crate).
- `nvkmd-rm` boundary: ops tables have the same NULL/non-NULL slots as the C tables;
  `vk_sync_type` `size`/`features` equal the C values; object recovery rejects a foreign ops
  pointer; in-place payload init/finish and `try_box` run under Miri over heap-allocated Mesa
  structs (syscall-free).
- ABI: bindgen layout asserts compile for all three targets (`cargo check --target
  i686-linux-android` in the gate); `mesa/build.py` fails on stale bindings; the generated
  `nv_rm.h` is compiled with `mesa/abi-check.c` on both Android targets and matches the
  `nv-rm-abi` layouts; the symbol-list tests of §3.

### 4.4 Hot path: CPU per submit ≤ C (RUST.md §14)

The per-frame path is ctx `exec`/`wait`/`signal` → `flush` → `submit_batch`, sync
`wait_many`, mem `map`. The port allocates nothing per call there (scratch `Vec`s with kept
capacity; C `calloc`s the signal array on every `signal` call, 1245). Measured by a new
`tests/vulkan/native-vulkan-submit-cost.c`: 200,000 `vkQueueSubmit2` of one tiny command buffer
with one already-signalled timeline wait and one signal, then the same with
`vkWaitSemaphores` and `vkMapMemory`/`vkUnmapMemory` loops; reports
`CLOCK_THREAD_CPUTIME_ID` of the submitting thread and process CPU (`getrusage`, includes
Mesa's submit thread) per operation. Five interleaved runs per backend on an idle GPU (no other
GPU clients, `nvidia-smi` utilisation 0 before each run). Pass: Rust median ≤ C median + the
spread of the C runs. Numbers go into the switch commit; a frame-level check on a shipped
workload (CPU time per frame) follows on the device.

### 4.5 Regression and live

After the switch builds: tests/vulkan `run.py --target all` on both backends, the dEQP-VK
sample against the C-backend build of the same commit (expected changes: none), live Waydroid
checks, 0 new `NVRM: Xid`.

## 5. Steps

Each step is one commit (or a short series) that passes the workspace gate, the pdev-preflight
suites on the C backend, and the checks named. Estimates are focused working hours.

| # | step | checks | est. |
|---|---|---|---|
| 0 | rebase onto the in-flight status-mapping and teardown-after-loss changes in `rm/src/device.rs`/`gpu.rs` | — | — |
| 1 | `nv-rm-abi` + generated `nv_rm.h` (cbindgen, typed handles, stale check), replacing the hand-written header; `rm` entries pinned to the declarations (done) | generated header compiles in `nvkmd_rm.c`, gallium-nvrm, `bridge-check.c`, `abi-check.c` on both Android targets; `nm -D` of `libnv_bionic_native.so` identical; capture 0/42 | 3 h |
| 1b | `rm`: split `mesa.rs` into `rm::device` and `rm::device::capi` (§2.4; done as a pure move), then the entry bodies behind a device API. Pure refactor | rm tests + Miri on touched unsafe modules, pdev-preflight, capture 0/42, `nm -D` identical | 5 h |
| 2 | `nvkmd-sys`: generator script, committed bindings for 3 targets, stale check in `build.py` | layout asserts on 3 targets, regen is a no-op | 3 h |
| 3 | build plumbing: meson option, `build.py --nvkmd`, link spike with an archive holding only the guard (three Rust staticlibs in one `.so`, `--exclude-libs`) | both Android targets link, `nm -D` unchanged | 2 h |
| 4 | `nvkmd-rm-logic` | unit tables, full Miri | 4 h |
| 5 | `nvkmd-rm`: pdev + dev entries, panic guard, `try_box`; `nvkmd-harness` | harness `pdev`/`info`/`loss` on both, call logs identical | 5 h |
| 6 | memory + VA entries (heap via `util_vma_heap`, `VaRelease`) | memory-api, memory-cleanup scenarios ported to the harness; call logs | 5 h |
| 7 | sync types | sync-api, wait-loss scenarios ported; call logs | 5 h |
| 8 | ctx entries (hot path) | submit-fusion scenarios ported; call logs | 5 h |
| 9 | switch gate: both backends on 3 targets | capture 0/42, run.py all, `nm -D`, submit-cost bench (idle GPU), Miri, then dEQP sample + live | 4 h + regression run |
| 10 | after the gate: delete `nvkmd_rm.c`/`nvkmd_rm.h` and the `c` option; every `nv_rm_*` entry stays (gallium-nvrm, Codec2's library); the harness keeps the C fake (or a Rust one) | gate, capture 0/42 | 2 h |
| 11 | separate later commits, each with its own test: Rust VA heap (differential vs `util_vma_heap` on recorded alloc/free sequences), DRM version query by ioctl in Rust, bugs found during 1–8 | own tests | — |

About 48 h before the regression run. Steps 1b-4 are independent of each other after step 1.

### Risks

- **fd lifetimes.** Imports borrow (`BorrowedFd` scoped to the call; `nvk_AllocateMemory`
  closes the app fd only after success); exports become raw only on the success return, so no
  error path leaks or double-closes; render-node fds are `OwnedFd` opened `O_CLOEXEC` (std
  does). The sync_file watcher's dup lifetime stays inside `rm`, untouched.
- **fork/zygote (§13).** A pdev made in a zygote holds an fd and plain data only; every RM
  device records its process (`ProcessBound`, one atomic load per call) and a call from a forked
  child fails with `VK_ERROR_DEVICE_LOST` without touching the parent's objects, as through the
  C ABI today. Sync payloads and contexts reach the RM only through their device, so they
  inherit that check.
- **Loss publication (833571a).** Stays in the RM's entry guard, so every consumer of the ABI
  gets it; step 7 keeps both reads in `rm_wait_status` (Mesa's counter first, then the RM flag)
  in the same order.
- **Teardown after loss** is being changed in `rm` now; the dev `destroy` port calls the same
  close path and inherits the fix, so step 0 is a hard prerequisite.
- **Three Rust std copies in `libvulkan_nouveau.so`** (NAK, NIL, ours; the RM's is in its own
  library, as today): works for two today; step 3 proves three before anything depends on it.
- **ABI drift between the RM and its users.** One source (`nv-rm-abi`) for both sides: the RM's
  definitions are pinned to it at compile time, C users include the header generated from it,
  and the stale-header test fails a change that is not regenerated.
- **Panics into C:** every callback is guarded; release keeps `panic = "unwind"`.
- **Header drift on a Mesa pin bump:** stale bindings fail the build (§2.1); the `nvk_device`
  offset is the one layout from a non-nvkmd header and is pinned by the same check.
- **Lock order** stays RM `Inner` → VA heap mutex; the heap lock is never held across an RM
  call (the `VaHeaps` lock is taken per heap operation; the release callback takes it).

## 6. Open questions

- **Q2: RUST.md §4 for void callbacks.** §4 says interfaces with no error return abort on a
  panic. Several nvkmd void callbacks (mem `free`/`unmap`, va `free`, sync `finish`, ctx
  `destroy`) report RM failures today by publishing device loss. Proposal: treat device loss as
  their failure channel for panics too (§2.3), abort only where no device exists. Needs a
  one-line RUST.md clarification if accepted.
