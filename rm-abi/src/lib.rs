// SPDX-License-Identifier: MIT
//! The RM backend's C ABI, defined once in Rust.
//!
//! `libnv_bionic_native.so` (the `nv-bionic-native` crate) exports these
//! entries; NVK's nvkmd backend, the gallium GL driver and the tests call them.
//! `mesa/nv_rm.h` is generated from this crate by `tools/gen-rm-header.py`
//! (cbindgen), so C callers and Rust callers see the same declarations. Each
//! entry has a function-pointer type generic over the device type: the
//! declarations below and the exporting crate's definitions are both assigned
//! to it at compile time, so neither can drift from the other.
//!
//! Raw declarations only, no logic. Contract of every entry (unless its
//! documentation says otherwise): all calls are thread-safe and run on the
//! calling thread (no worker); the device pointer is NULL or a live
//! `nv_rm_open` device kept alive for the whole call; an output pointer is
//! NULL or writable, and an input array readable for its count; a descriptor
//! argument is open for the call; a sync id is 0 or a live `nv_rm_sync_create`
//! payload. Entries return 0 on success and a negative `VkResult` on failure,
//! with a diagnostic on stderr (GNU) or in logcat, tag "nvk-rm" (Android).
//!
//! Resource frees never wait for GPU work: pending work may be gated on a
//! foreign fence whose producer is the freeing thread. `nv_rm_free` and
//! `nv_rm_va_free` instead defer the real RM free/unmap (and the VA range's
//! return to the caller) until every queue's work submitted before the call
//! has retired; with nothing outstanding they act immediately.
//!
//! Context, memory and mapping ids are numeric handles the backend validates;
//! sync ids are backend payload references.
use core::ffi::{c_char, c_int, c_void};
use core::marker::{PhantomData, PhantomPinned};

/// An RM device made by [`nv_rm_open`]; opaque to callers.
#[repr(C)]
#[derive(Debug)]
pub struct NvRmDevice {
    _opaque: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// An allocation made by [`nv_rm_alloc`] or [`nv_rm_import`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NvRmMemoryId(u64);
impl NvRmMemoryId {
    /// The handle with this raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
    /// The raw value C sees.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
/// A GPU mapping made by [`nv_rm_bind`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NvRmMappingId(u64);
impl NvRmMappingId {
    /// The handle with this raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
    /// The raw value C sees.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
/// A context (one channel) made by [`nv_rm_context_create`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NvRmContextId(u64);
impl NvRmContextId {
    /// The handle with this raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
    /// The raw value C sees.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
/// A synchronization payload made by [`nv_rm_sync_create`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NvRmSyncId(u64);
impl NvRmSyncId {
    /// The handle with this raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
    /// The raw value C sees.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
/// A context's GPU completion point, returned by submission.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NvRmToken(u64);
impl NvRmToken {
    /// The handle with this raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
    /// The raw value C sees.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// `NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS` field order (ctrl2080gr.h), which is
/// also the order of Mesa's `nv_zcull_device_info` (nv_device_info.h:17-30).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    missing_docs,
    reason = "fields are the RM control's, named as in ctrl2080gr.h"
)]
pub struct NvRmZcullInfo {
    pub width_align_pixels: u32,
    pub height_align_pixels: u32,
    pub pixel_squares_by_aliquots: u32,
    pub aliquot_total: u32,
    pub region_byte_multiplier: u32,
    pub region_header_size: u32,
    pub subregion_header_size: u32,
    pub subregion_count: u32,
    pub subregion_width_align_pixels: u32,
    pub subregion_height_align_pixels: u32,
}

/// What [`nv_rm_open`] reports about the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[expect(
    missing_docs,
    reason = "identity, class and count fields are named after their RM queries"
)]
pub struct NvRmInfo {
    pub vendor_id: u32,
    pub device_id: u32,
    pub chipset: u32,
    pub sm: u32,
    pub pci_domain: u32,
    pub pci_bus: u32,
    pub pci_device: u32,
    pub pci_function: u32,
    pub gpc_count: u32,
    pub tpc_count: u32,
    pub mp_per_tpc: u32,
    pub max_warps_per_mp: u32,
    pub cls_copy: u32,
    pub cls_eng2d: u32,
    pub cls_eng3d: u32,
    pub cls_m2mf: u32,
    pub cls_compute: u32,
    /// NIL uncompressed color PTE kind: Fermi 0xfe, Turing+ 6.
    pub generic_kind: u32,
    pub vram_size: u64,
    pub bar_size: u64,
    /// The GPU name, NUL-terminated.
    pub name: [c_char; 64],
    /// ZCULL region geometry (`GR_GET_ZCULL_INFO`). Meaningful only when
    /// `has_zcull_info` is 1.
    pub zcull: NvRmZcullInfo,
    /// The `GRAPHICS_ZCULL` context-switch buffer size.
    pub zcull_ctxsw_size: u32,
    /// The `GRAPHICS_ZCULL` context-switch buffer alignment.
    pub zcull_ctxsw_align: u32,
    /// 0 when the RM query failed or returned values outside the supported
    /// range; the reason is logged at open.
    pub has_zcull_info: u32,
    /// RM's SM ISA code (`NV2080_CTRL_GR_INFO_INDEX_SM_VERSION`, e.g. 0xa04 on
    /// `GB20x`); `sm` is the shader model decimal derived from it.
    pub sm_isa: u32,
    /// `NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT`: an input of the compute
    /// contexts' `SET_SCG_CONTROL`.
    pub shader_pipe_count: u32,
    /// `NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT` (the TPC count, as
    /// open-gpu-kernel-modules `nv_gpu_ops.c` reads it).
    pub shader_pipe_sub_count: u32,
    /// Contexts that can join another context's channel group
    /// ([`nv_rm_context_create_shared`]): from Volta on RM's subcontext count
    /// (GR info `MAX_SUBCONTEXT_COUNT`) less VEID 0; before Volta, where
    /// joining channels share the group's single GR context, the group's 32
    /// channels less one (gm107 `dev_ram.h` `NV_RAMRL_ENTRY_TSG_LENGTH_MAX`).
    pub max_shared_ctx: u32,
    /// Zero.
    pub reserved: u32,
}

/// An allocation made by [`nv_rm_alloc`] or [`nv_rm_import`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvRmAllocation {
    /// The allocation's handle.
    pub id: NvRmMemoryId,
    /// Its size in bytes.
    pub size: u64,
    /// Reserved zero; CPU maps are acquired separately.
    pub cpu_address: usize,
}

/// One push buffer segment to execute.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(
    non_snake_case,
    reason = "`size_B` is the C field name Mesa uses for byte sizes"
)]
pub struct NvRmExec {
    /// GPU VA of the segment.
    pub address: u64,
    /// Size in bytes.
    pub size_B: u32,
    /// `NV_RM_EXEC_*`.
    pub flags: u32,
}

/// One (payload, value) pair of a wait or signal list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvRmSyncPoint {
    /// The payload.
    pub sync: NvRmSyncId,
    /// The timeline value (ignored for binary payloads).
    pub value: u64,
}

/// [`nv_rm_sync_create`]: a timeline payload.
pub const NV_RM_SYNC_TIMELINE: u32 = 0x1;
/// [`nv_rm_sync_create`]: the payload may be exported.
pub const NV_RM_SYNC_SHAREABLE: u32 = 0x2;
/// [`nv_rm_sync_create`]: the payload is shared with other processes.
pub const NV_RM_SYNC_SHARED: u32 = 0x4;
/// [`nv_rm_sync_status_many`], [`nv_rm_sync_wait_many`]: a submitted, not yet
/// executed signal counts as ready.
pub const NV_RM_SYNC_WAIT_PENDING: u32 = 0x1;
/// [`nv_rm_sync_status_many`], [`nv_rm_sync_wait_many`]: any point instead of
/// all points.
pub const NV_RM_SYNC_WAIT_ANY: u32 = 0x2;
/// [`NvRmExec::flags`]: no prefetch of the segment.
pub const NV_RM_EXEC_NO_PREFETCH: u32 = 0x1;
/// [`NvRmExec::flags`]: the segment continues in the next one.
pub const NV_RM_EXEC_INCOMPLETE: u32 = 0x2;
/// Engine bit (the same bits as Mesa's `nvkmd_engines`).
pub const NV_RM_ENGINE_COPY: u32 = 0x1;
/// Engine bit.
pub const NV_RM_ENGINE_2D: u32 = 0x2;
/// Engine bit.
pub const NV_RM_ENGINE_3D: u32 = 0x4;
/// Engine bit.
pub const NV_RM_ENGINE_M2MF: u32 = 0x8;
/// Engine bit.
pub const NV_RM_ENGINE_COMPUTE: u32 = 0x10;
/// [`nv_rm_alloc`] placement: device memory.
pub const NV_RM_PLACEMENT_VRAM: u32 = 0;
/// [`nv_rm_alloc`] placement: system memory.
pub const NV_RM_PLACEMENT_GART: u32 = 1;
/// Device-local, never CPU-mapped or exported: RM is asked for compression and
/// big pages (`NVOS32` `COMPR ANY | PAGE_SIZE BIG`), as the stock driver does. RM
/// decides whether compression is granted; the allocation's own VA keeps the
/// requested (uncompressed) kind. Size must be a 64 KiB multiple; the
/// alignment is raised to 64 KiB.
pub const NV_RM_PLACEMENT_VRAM_COMPRESSIBLE: u32 = 2;
/// One GPU-internal pool block: 2 MiB of device memory in one physically
/// contiguous huge page, GPU-cached, uncompressed, never CPU-mapped or
/// exported (the NVOS32 request the proprietary driver uses for the memory
/// holding its command buffers' QMDs and launch counters). Size must be
/// exactly 2 MiB and the kind 0; [`nv_rm_map`] fails on it, and it binds whole
/// at a 2 MiB aligned VA.
pub const NV_RM_PLACEMENT_VRAM_POOL_BLOCK: u32 = 3;
/// Start of the VA window the caller owns, `[16 GiB, 1 TiB)`; the backend's
/// internal mappings stay below it.
pub const NV_RM_VA_START: u64 = 0x0004_0000_0000;
/// End of the caller's VA window.
pub const NV_RM_VA_END: u64 = 0x0100_0000_0000;

/// Returns a VA range to the caller's allocator. Called at most once per
/// successful [`nv_rm_va_free`], only after RM unmapped every listed mapping,
/// on whichever thread reclaims it (that call, a later backend call, or
/// [`nv_rm_close`], which runs all outstanding releases before returning).
/// Never called if an unmap failed (device lost; the range stays reserved).
/// The backend may hold its resource lock while calling it: it may take only
/// the caller's VA-heap lock, which must never be held across backend calls.
pub type NvRmVaRelease = Option<unsafe extern "C" fn(data: *mut c_void, address: u64, size: u64)>;

unsafe extern "C" {
    /// Opens the GPU at a PCI address: `*out` receives the device, `*info`
    /// what RM reports about it.
    pub fn nv_rm_open(
        domain: u32,
        bus: u32,
        device: u32,
        function: u32,
        out: *mut *mut NvRmDevice,
        info: *mut NvRmInfo,
    ) -> i32;
    /// Nonblocking snapshot: 0 = live, 1 = lost (also for NULL). One atomic
    /// load. The caller keeps the device alive for this call, as with other
    /// entries.
    pub fn nv_rm_is_lost(device: *const NvRmDevice) -> i32;
    /// CPU-only software-loss publication: atomics, CPU waiter notification
    /// and an eventfd wake. May take the waiters' condition-variable lock
    /// only; never waits for GPU progress, takes submission/resource locks or
    /// issues RM calls. The caller retains the device throughout the call.
    pub fn nv_rm_mark_lost(device: *const NvRmDevice);
}
/// The type of [`nv_rm_open`], over the device type `D`.
pub type NvRmOpen<D> = unsafe extern "C" fn(u32, u32, u32, u32, *mut *mut D, *mut NvRmInfo) -> i32;
const _: NvRmOpen<NvRmDevice> = nv_rm_open;
/// The type of [`nv_rm_is_lost`], over the device type `D`.
pub type NvRmIsLost<D> = unsafe extern "C" fn(*const D) -> i32;
const _: NvRmIsLost<NvRmDevice> = nv_rm_is_lost;
/// The type of [`nv_rm_mark_lost`], over the device type `D`.
pub type NvRmMarkLost<D> = unsafe extern "C" fn(*const D);
const _: NvRmMarkLost<NvRmDevice> = nv_rm_mark_lost;

unsafe extern "C" {
    /// Destroys a device made by [`nv_rm_open`]; NULL is ignored. No call on
    /// it may be in flight or follow.
    pub fn nv_rm_close(device: *mut NvRmDevice);
    /// `pte_kind` 0: plain memory (RM picks the layout). Non-zero: uncompressed
    /// block-linear memory of that kind (the device's generic kind only, VRAM
    /// or GART); RM must grant exactly that kind. `tile_mode` must be 0.
    pub fn nv_rm_alloc(
        device: *mut NvRmDevice,
        size: u64,
        alignment: u64,
        placement: u32,
        pte_kind: u32,
        tile_mode: u32,
        out: *mut NvRmAllocation,
    ) -> i32;
    /// Separate internal/client map references share one lazy mapping.
    pub fn nv_rm_map(device: *mut NvRmDevice, memory: NvRmMemoryId, address: *mut usize) -> i32;
    /// Drops one map reference.
    pub fn nv_rm_unmap(device: *mut NvRmDevice, memory: NvRmMemoryId) -> i32;
    /// Drops the allocation id now (CPU mapping must be released). The RM
    /// object is freed after the work submitted before this call retires.
    pub fn nv_rm_free(device: *mut NvRmDevice, memory: NvRmMemoryId) -> i32;
    /// Maps `[memory_offset, memory_offset + size)` of `memory` at `address`
    /// with `pte_kind`; `*mapping` receives the mapping.
    pub fn nv_rm_bind(
        device: *mut NvRmDevice,
        memory: NvRmMemoryId,
        memory_offset: u64,
        size: u64,
        address: u64,
        pte_kind: u32,
        mapping: *mut NvRmMappingId,
    ) -> i32;
    /// Immediate RM unmap, no GPU wait: only for application-ordered binds,
    /// where Vulkan forbids unbinding/rebinding memory still used by pending
    /// work.
    pub fn nv_rm_unbind(device: *mut NvRmDevice, mapping: NvRmMappingId) -> i32;
    /// Frees a VA reservation `[address, address + size)` and all its mappings
    /// (ids from [`nv_rm_bind`], each inside the range). On success the backend
    /// owns the mappings and calls `release(data, address, size)` once,
    /// deferred as described in the crate documentation (immediately if
    /// `count` is 0 or nothing is outstanding). On failure nothing was taken:
    /// the caller keeps the range reserved.
    pub fn nv_rm_va_free(
        device: *mut NvRmDevice,
        mappings: *const NvRmMappingId,
        count: u32,
        address: u64,
        size: u64,
        release: NvRmVaRelease,
        data: *mut c_void,
    ) -> i32;
    /// Creates a context with its own channel group for the `NV_RM_ENGINE_*`
    /// bits in `engines`.
    pub fn nv_rm_context_create(
        device: *mut NvRmDevice,
        engines: u32,
        context: *mut NvRmContextId,
    ) -> i32;
    /// Like [`nv_rm_context_create`], but the new context's channel joins the
    /// TSG of context `share` on the lowest free subcontext (VEID) instead of
    /// getting a TSG of its own, so work on both contexts can run on the GR
    /// engine at the same time. Only the compute engine (16) is supported, on
    /// Volta and later; anything else fails with a named reason. The TSG
    /// lives until the last context in it is destroyed.
    pub fn nv_rm_context_create_shared(
        device: *mut NvRmDevice,
        engines: u32,
        share: NvRmContextId,
        context: *mut NvRmContextId,
    ) -> i32;
    /// Destroys a context; bounded even when its work never completes.
    pub fn nv_rm_context_destroy(device: *mut NvRmDevice, context: NvRmContextId) -> i32;
    /// All pushes (including an incomplete push and its following
    /// continuation) execute on the persistent channel in order. Success
    /// means full GPU completion.
    pub fn nv_rm_exec(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        pushes: *const NvRmExec,
        count: u32,
    ) -> i32;
    /// Async submission: 0 means queued, `*token` names GPU completion; 1
    /// (`VK_NOT_READY`) means no work/waits were queued and `*token` names the
    /// oldest batch to wait before retrying. Other results are negative
    /// errors. Borrowed wait fds remain owned by the caller; accepted waits
    /// retain owned descriptor duplicates and cancellable userspace gates, not
    /// kernel WAIT callbacks. Empty batches are valid and issue a completion.
    pub fn nv_rm_submit_waits(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        pushes: *const NvRmExec,
        count: u32,
        drm_fd: c_int,
        wait_fds: *const c_int,
        wait_count: u32,
        token: *mut NvRmToken,
    ) -> i32;
    /// [`nv_rm_submit_waits`] without waits.
    pub fn nv_rm_submit(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        pushes: *const NvRmExec,
        count: u32,
        token: *mut NvRmToken,
    ) -> i32;
    /// Nonblocking completion query; `*complete` is 0 or 1.
    pub fn nv_rm_poll(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        token: NvRmToken,
        complete: *mut u32,
    ) -> i32;
    /// Waits on the caller. `u64::MAX` is infinite, zero polls. Returns 0 when
    /// complete, 2 (`VK_TIMEOUT`) at deadline, negative on error.
    pub fn nv_rm_wait(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        token: NvRmToken,
        timeout_ns: u64,
    ) -> i32;
    /// Exports a real pending GPU fence; the returned fd belongs to the caller.
    pub fn nv_rm_context_fence(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        token: NvRmToken,
        drm_fd: c_int,
        sync_fd: *mut c_int,
    ) -> i32;
    /// Polls channel and retained incoming dependency errors after Vulkan
    /// waits. Returns `VK_SUCCESS` or `VK_ERROR_DEVICE_LOST`, preserving
    /// producer failures.
    pub fn nv_rm_check_status(device: *mut NvRmDevice) -> i32;
    /// The GPU timestamp.
    pub fn nv_rm_timestamp(device: *mut NvRmDevice, timestamp: *mut u64) -> i32;
    /// Exports an allocation as a dma-buf; the fd belongs to the caller.
    pub fn nv_rm_export(
        device: *mut NvRmDevice,
        memory: NvRmMemoryId,
        drm_fd: c_int,
        dma_buf_fd: *mut c_int,
    ) -> i32;
    /// Imports a dma-buf (borrowed) as an allocation.
    pub fn nv_rm_import(
        device: *mut NvRmDevice,
        drm_fd: c_int,
        dma_buf_fd: c_int,
        out: *mut NvRmAllocation,
    ) -> i32;
    /// Creates a payload (`NV_RM_SYNC_*` flags). Sync identities belong to
    /// this device. Payloads accepted by submission are retained independently
    /// of identity reset/move/destruction. Mutation is transactional: a failed
    /// create/reset/move/import leaves live state intact.
    pub fn nv_rm_sync_create(
        device: *mut NvRmDevice,
        flags: u32,
        initial_value: u64,
        sync: *mut NvRmSyncId,
    ) -> i32;
    /// Destroys a payload.
    pub fn nv_rm_sync_destroy(device: *mut NvRmDevice, sync: NvRmSyncId) -> i32;
    /// Historical name kept for ABI stability: disposal is immediate on the
    /// caller. Accepted submissions keep their own payload slot references
    /// until retired.
    pub fn nv_rm_sync_destroy_deferred(device: *mut NvRmDevice, sync: NvRmSyncId) -> i32;
    /// Resets a binary payload.
    pub fn nv_rm_sync_reset(device: *mut NvRmDevice, sync: NvRmSyncId) -> i32;
    /// Moves a binary payload from `src` to `dst`.
    pub fn nv_rm_sync_move(
        device: *mut NvRmDevice,
        dst: NvRmSyncId,
        src: NvRmSyncId,
        shared_source: u32,
    ) -> i32;
    /// Binds a deferred (submit-thread) binary signal: `dst` becomes the
    /// handle the deferred batch signals, on `src`'s current slot; `src`
    /// reports it pending. `*reserved` = 0: `src` is not reservable and
    /// nothing changed.
    pub fn nv_rm_sync_reserve(
        device: *mut NvRmDevice,
        dst: NvRmSyncId,
        src: NvRmSyncId,
        reserved: *mut u32,
    ) -> i32;
    /// Host signal (a binary payload ignores `value`).
    pub fn nv_rm_sync_signal(device: *mut NvRmDevice, sync: NvRmSyncId, value: u64) -> i32;
    /// A timeline payload's value.
    pub fn nv_rm_sync_get_value(device: *mut NvRmDevice, sync: NvRmSyncId, value: *mut u64) -> i32;
    /// Nonblocking status query; `*ready` is 0 or 1. `flags` select
    /// pending/any.
    pub fn nv_rm_sync_status_many(
        device: *mut NvRmDevice,
        points: *const NvRmSyncPoint,
        count: u32,
        flags: u32,
        ready: *mut u32,
    ) -> i32;
    /// Returns 0 on completion, 2 on caller timeout, negative on failure.
    /// Infinite waits use `u64::MAX`. Sleeps on the caller: one waiting thread
    /// polls the RM event descriptor and wakes the others; the deadline is the
    /// caller's only.
    pub fn nv_rm_sync_wait_many(
        device: *mut NvRmDevice,
        points: *const NvRmSyncPoint,
        count: u32,
        flags: u32,
        timeout_ns: u64,
    ) -> i32;
    /// Import borrows `fd`. `opaque` selects `OPAQUE_FD` instead of `SYNC_FD`.
    pub fn nv_rm_sync_import_fd(
        device: *mut NvRmDevice,
        sync: NvRmSyncId,
        fd: c_int,
        opaque: u32,
    ) -> i32;
    /// Export returns an owned fd. Only explicit export needs a kernel export
    /// fence.
    pub fn nv_rm_sync_export_fd(
        device: *mut NvRmDevice,
        sync: NvRmSyncId,
        opaque: u32,
        drm_fd: c_int,
        fd: *mut c_int,
    ) -> i32;
    /// Returns `VK_NOT_READY` with the oldest token on ring pressure; no
    /// waits, pushes or signals are accepted in that case. Success retains all
    /// payloads and returns the actual GPU completion token for this batch.
    pub fn nv_rm_submit_batch(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        pushes: *const NvRmExec,
        count: u32,
        waits: *const NvRmSyncPoint,
        wait_count: u32,
        signals: *const NvRmSyncPoint,
        signal_count: u32,
        token: *mut NvRmToken,
    ) -> i32;
    /// [`nv_rm_submit_batch`] without signals.
    pub fn nv_rm_submit_sync(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        pushes: *const NvRmExec,
        count: u32,
        waits: *const NvRmSyncPoint,
        wait_count: u32,
        token: *mut NvRmToken,
    ) -> i32;
    /// [`nv_rm_submit_batch`] with signals only.
    pub fn nv_rm_context_signal(
        device: *mut NvRmDevice,
        context: NvRmContextId,
        signals: *const NvRmSyncPoint,
        count: u32,
        token: *mut NvRmToken,
    ) -> i32;

}

/// The type of [`nv_rm_close`], over the device type `D`.
pub type NvRmClose<D> = unsafe extern "C" fn(*mut D);
const _: NvRmClose<NvRmDevice> = nv_rm_close;
/// The type of [`nv_rm_alloc`], over the device type `D`.
pub type NvRmAlloc<D> =
    unsafe extern "C" fn(*mut D, u64, u64, u32, u32, u32, *mut NvRmAllocation) -> i32;
const _: NvRmAlloc<NvRmDevice> = nv_rm_alloc;
/// The type of [`nv_rm_map`], over the device type `D`.
pub type NvRmMap<D> = unsafe extern "C" fn(*mut D, NvRmMemoryId, *mut usize) -> i32;
const _: NvRmMap<NvRmDevice> = nv_rm_map;
/// The type of [`nv_rm_unmap`], over the device type `D`.
pub type NvRmUnmap<D> = unsafe extern "C" fn(*mut D, NvRmMemoryId) -> i32;
const _: NvRmUnmap<NvRmDevice> = nv_rm_unmap;
/// The type of [`nv_rm_free`], over the device type `D`.
pub type NvRmFree<D> = unsafe extern "C" fn(*mut D, NvRmMemoryId) -> i32;
const _: NvRmFree<NvRmDevice> = nv_rm_free;
/// The type of [`nv_rm_bind`], over the device type `D`.
pub type NvRmBind<D> =
    unsafe extern "C" fn(*mut D, NvRmMemoryId, u64, u64, u64, u32, *mut NvRmMappingId) -> i32;
const _: NvRmBind<NvRmDevice> = nv_rm_bind;
/// The type of [`nv_rm_unbind`], over the device type `D`.
pub type NvRmUnbind<D> = unsafe extern "C" fn(*mut D, NvRmMappingId) -> i32;
const _: NvRmUnbind<NvRmDevice> = nv_rm_unbind;
/// The type of [`nv_rm_va_free`], over the device type `D`.
pub type NvRmVaFree<D> = unsafe extern "C" fn(
    *mut D,
    *const NvRmMappingId,
    u32,
    u64,
    u64,
    NvRmVaRelease,
    *mut c_void,
) -> i32;
const _: NvRmVaFree<NvRmDevice> = nv_rm_va_free;
/// The type of [`nv_rm_context_create`], over the device type `D`.
pub type NvRmContextCreate<D> = unsafe extern "C" fn(*mut D, u32, *mut NvRmContextId) -> i32;
const _: NvRmContextCreate<NvRmDevice> = nv_rm_context_create;
/// The type of [`nv_rm_context_create_shared`], over the device type `D`.
pub type NvRmContextCreateShared<D> =
    unsafe extern "C" fn(*mut D, u32, NvRmContextId, *mut NvRmContextId) -> i32;
const _: NvRmContextCreateShared<NvRmDevice> = nv_rm_context_create_shared;
/// The type of [`nv_rm_context_destroy`], over the device type `D`.
pub type NvRmContextDestroy<D> = unsafe extern "C" fn(*mut D, NvRmContextId) -> i32;
const _: NvRmContextDestroy<NvRmDevice> = nv_rm_context_destroy;
/// The type of [`nv_rm_exec`], over the device type `D`.
pub type NvRmExecFn<D> = unsafe extern "C" fn(*mut D, NvRmContextId, *const NvRmExec, u32) -> i32;
const _: NvRmExecFn<NvRmDevice> = nv_rm_exec;
/// The type of [`nv_rm_submit_waits`], over the device type `D`.
pub type NvRmSubmitWaits<D> = unsafe extern "C" fn(
    *mut D,
    NvRmContextId,
    *const NvRmExec,
    u32,
    c_int,
    *const c_int,
    u32,
    *mut NvRmToken,
) -> i32;
const _: NvRmSubmitWaits<NvRmDevice> = nv_rm_submit_waits;
/// The type of [`nv_rm_submit`], over the device type `D`.
pub type NvRmSubmit<D> =
    unsafe extern "C" fn(*mut D, NvRmContextId, *const NvRmExec, u32, *mut NvRmToken) -> i32;
const _: NvRmSubmit<NvRmDevice> = nv_rm_submit;
/// The type of [`nv_rm_poll`], over the device type `D`.
pub type NvRmPoll<D> = unsafe extern "C" fn(*mut D, NvRmContextId, NvRmToken, *mut u32) -> i32;
const _: NvRmPoll<NvRmDevice> = nv_rm_poll;
/// The type of [`nv_rm_wait`], over the device type `D`.
pub type NvRmWait<D> = unsafe extern "C" fn(*mut D, NvRmContextId, NvRmToken, u64) -> i32;
const _: NvRmWait<NvRmDevice> = nv_rm_wait;
/// The type of [`nv_rm_context_fence`], over the device type `D`.
pub type NvRmContextFence<D> =
    unsafe extern "C" fn(*mut D, NvRmContextId, NvRmToken, c_int, *mut c_int) -> i32;
const _: NvRmContextFence<NvRmDevice> = nv_rm_context_fence;
/// The type of [`nv_rm_check_status`], over the device type `D`.
pub type NvRmCheckStatus<D> = unsafe extern "C" fn(*mut D) -> i32;
const _: NvRmCheckStatus<NvRmDevice> = nv_rm_check_status;
/// The type of [`nv_rm_timestamp`], over the device type `D`.
pub type NvRmTimestamp<D> = unsafe extern "C" fn(*mut D, *mut u64) -> i32;
const _: NvRmTimestamp<NvRmDevice> = nv_rm_timestamp;
/// The type of [`nv_rm_export`], over the device type `D`.
pub type NvRmExport<D> = unsafe extern "C" fn(*mut D, NvRmMemoryId, c_int, *mut c_int) -> i32;
const _: NvRmExport<NvRmDevice> = nv_rm_export;
/// The type of [`nv_rm_import`], over the device type `D`.
pub type NvRmImport<D> = unsafe extern "C" fn(*mut D, c_int, c_int, *mut NvRmAllocation) -> i32;
const _: NvRmImport<NvRmDevice> = nv_rm_import;
/// The type of [`nv_rm_sync_create`], over the device type `D`.
pub type NvRmSyncCreate<D> = unsafe extern "C" fn(*mut D, u32, u64, *mut NvRmSyncId) -> i32;
const _: NvRmSyncCreate<NvRmDevice> = nv_rm_sync_create;
/// The type of [`nv_rm_sync_destroy`], over the device type `D`.
pub type NvRmSyncDestroy<D> = unsafe extern "C" fn(*mut D, NvRmSyncId) -> i32;
const _: NvRmSyncDestroy<NvRmDevice> = nv_rm_sync_destroy;
/// The type of [`nv_rm_sync_destroy_deferred`], over the device type `D`.
pub type NvRmSyncDestroyDeferred<D> = unsafe extern "C" fn(*mut D, NvRmSyncId) -> i32;
const _: NvRmSyncDestroyDeferred<NvRmDevice> = nv_rm_sync_destroy_deferred;
/// The type of [`nv_rm_sync_reset`], over the device type `D`.
pub type NvRmSyncReset<D> = unsafe extern "C" fn(*mut D, NvRmSyncId) -> i32;
const _: NvRmSyncReset<NvRmDevice> = nv_rm_sync_reset;
/// The type of [`nv_rm_sync_move`], over the device type `D`.
pub type NvRmSyncMove<D> = unsafe extern "C" fn(*mut D, NvRmSyncId, NvRmSyncId, u32) -> i32;
const _: NvRmSyncMove<NvRmDevice> = nv_rm_sync_move;
/// The type of [`nv_rm_sync_reserve`], over the device type `D`.
pub type NvRmSyncReserve<D> = unsafe extern "C" fn(*mut D, NvRmSyncId, NvRmSyncId, *mut u32) -> i32;
const _: NvRmSyncReserve<NvRmDevice> = nv_rm_sync_reserve;
/// The type of [`nv_rm_sync_signal`], over the device type `D`.
pub type NvRmSyncSignal<D> = unsafe extern "C" fn(*mut D, NvRmSyncId, u64) -> i32;
const _: NvRmSyncSignal<NvRmDevice> = nv_rm_sync_signal;
/// The type of [`nv_rm_sync_get_value`], over the device type `D`.
pub type NvRmSyncGetValue<D> = unsafe extern "C" fn(*mut D, NvRmSyncId, *mut u64) -> i32;
const _: NvRmSyncGetValue<NvRmDevice> = nv_rm_sync_get_value;
/// The type of [`nv_rm_sync_status_many`], over the device type `D`.
pub type NvRmSyncStatusMany<D> =
    unsafe extern "C" fn(*mut D, *const NvRmSyncPoint, u32, u32, *mut u32) -> i32;
const _: NvRmSyncStatusMany<NvRmDevice> = nv_rm_sync_status_many;
/// The type of [`nv_rm_sync_wait_many`], over the device type `D`.
pub type NvRmSyncWaitMany<D> =
    unsafe extern "C" fn(*mut D, *const NvRmSyncPoint, u32, u32, u64) -> i32;
const _: NvRmSyncWaitMany<NvRmDevice> = nv_rm_sync_wait_many;
/// The type of [`nv_rm_sync_import_fd`], over the device type `D`.
pub type NvRmSyncImportFd<D> = unsafe extern "C" fn(*mut D, NvRmSyncId, c_int, u32) -> i32;
const _: NvRmSyncImportFd<NvRmDevice> = nv_rm_sync_import_fd;
/// The type of [`nv_rm_sync_export_fd`], over the device type `D`.
pub type NvRmSyncExportFd<D> =
    unsafe extern "C" fn(*mut D, NvRmSyncId, u32, c_int, *mut c_int) -> i32;
const _: NvRmSyncExportFd<NvRmDevice> = nv_rm_sync_export_fd;
/// The type of [`nv_rm_submit_batch`], over the device type `D`.
pub type NvRmSubmitBatch<D> = unsafe extern "C" fn(
    *mut D,
    NvRmContextId,
    *const NvRmExec,
    u32,
    *const NvRmSyncPoint,
    u32,
    *const NvRmSyncPoint,
    u32,
    *mut NvRmToken,
) -> i32;
const _: NvRmSubmitBatch<NvRmDevice> = nv_rm_submit_batch;
/// The type of [`nv_rm_submit_sync`], over the device type `D`.
pub type NvRmSubmitSync<D> = unsafe extern "C" fn(
    *mut D,
    NvRmContextId,
    *const NvRmExec,
    u32,
    *const NvRmSyncPoint,
    u32,
    *mut NvRmToken,
) -> i32;
const _: NvRmSubmitSync<NvRmDevice> = nv_rm_submit_sync;

/// The type of [`nv_rm_context_signal`], over the device type `D`.
pub type NvRmContextSignal<D> =
    unsafe extern "C" fn(*mut D, NvRmContextId, *const NvRmSyncPoint, u32, *mut NvRmToken) -> i32;
const _: NvRmContextSignal<NvRmDevice> = nv_rm_context_signal;

// The layouts C sees (mesa/abi-check.c prints the same values from the
// generated header): checked at compile time on every target the crate is
// built for, the Android ones included.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<NvRmInfo>() == 224);
    assert!(offset_of!(NvRmInfo, generic_kind) == 68);
    assert!(offset_of!(NvRmInfo, vram_size) == 72);
    assert!(offset_of!(NvRmInfo, bar_size) == 80);
    assert!(offset_of!(NvRmInfo, name) == 88);
    assert!(offset_of!(NvRmInfo, zcull) == 152);
    assert!(offset_of!(NvRmInfo, zcull_ctxsw_size) == 192);
    assert!(offset_of!(NvRmInfo, has_zcull_info) == 200);
    assert!(offset_of!(NvRmInfo, sm_isa) == 204);
    assert!(offset_of!(NvRmInfo, shader_pipe_count) == 208);
    assert!(offset_of!(NvRmInfo, shader_pipe_sub_count) == 212);
    assert!(offset_of!(NvRmInfo, max_shared_ctx) == 216);
    assert!(size_of::<NvRmZcullInfo>() == 40);
    assert!(size_of::<NvRmExec>() == 16);
    assert!(offset_of!(NvRmExec, size_B) == 8);
    assert!(offset_of!(NvRmExec, flags) == 12);
    assert!(size_of::<NvRmSyncPoint>() == 16);
    assert!(offset_of!(NvRmSyncPoint, value) == 8);
    assert!(offset_of!(NvRmAllocation, size) == 8);
    assert!(offset_of!(NvRmAllocation, cpu_address) == 16);
    assert!(
        size_of::<NvRmAllocation>() == 16 + size_of::<usize>().next_multiple_of(align_of::<u64>())
    );
    assert!(size_of::<NvRmSyncId>() == 8 && align_of::<NvRmSyncId>() == align_of::<u64>());
};
