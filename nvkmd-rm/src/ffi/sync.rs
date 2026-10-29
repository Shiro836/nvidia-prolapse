// SPDX-License-Identifier: MIT
//! The two `vk_sync_type`s every physical device lists: a binary/timeline type
//! with opaque-fd import and export, and a binary type with `sync_file` import
//! and export. Separate
//! types because opaque shared state and sync-file copies have different
//! lifetime contracts; Mesa's external-handle queries then report same-type
//! compatibility only. Both run the same RM payload operations.
//!
//! Each sync object is Mesa's allocation of `size` bytes ([`RmSync`]): Mesa's
//! `vk_sync` followed by the RM payload id, which `init` writes and every
//! later callback only reads. The device of a callback is the KMD device of
//! the NVK device whose `vk` member Mesa passes.
//!
//! Host operations (create, signal, reset, move, import, export) go to the RM
//! without a liveness check of their own: the RM decides what still succeeds
//! on a lost device (`vkSignalSemaphore` has no `VK_ERROR_DEVICE_LOST`). CPU
//! waits check loss before and after the RM's wait, Mesa's loss count first
//! and then the RM's flag, so loss published while the caller slept is
//! returned even when the wait also completed.
//!
//! Threading: Mesa may wait on one sync from several threads; they only read
//! the id. `init` writes it before the sync is visible to other threads.
use std::ffi::c_int;
use std::mem::{MaybeUninit, offset_of};
use std::ptr::NonNull;

use nv_rm_abi::{
    NV_RM_SYNC_SHAREABLE, NV_RM_SYNC_SHARED, NV_RM_SYNC_TIMELINE, NV_RM_SYNC_WAIT_ANY,
    NV_RM_SYNC_WAIT_PENDING, NvRmSyncId, NvRmSyncPoint, nv_rm_is_lost, nv_rm_sync_create,
    nv_rm_sync_destroy_deferred, nv_rm_sync_export_fd, nv_rm_sync_get_value, nv_rm_sync_import_fd,
    nv_rm_sync_move, nv_rm_sync_reserve, nv_rm_sync_reset, nv_rm_sync_signal, nv_rm_sync_wait_many,
};
use nv_rm_core::conv::usize_from_u32;
use nvkmd_rm_logic::sync::{WaitStatus, clock_reading, reserved, shared_source, wait_timeout};
use nvkmd_sys::{
    VK_ERROR_DEVICE_LOST, VK_ERROR_OUT_OF_HOST_MEMORY, VK_SUCCESS, VK_SYNC_FEATURE_BINARY,
    VK_SYNC_FEATURE_CPU_RESET, VK_SYNC_FEATURE_CPU_SIGNAL, VK_SYNC_FEATURE_CPU_WAIT,
    VK_SYNC_FEATURE_GPU_WAIT, VK_SYNC_FEATURE_TIMELINE, VK_SYNC_FEATURE_WAIT_ANY,
    VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL, VK_SYNC_FEATURE_WAIT_PENDING, VK_SYNC_IS_SHAREABLE,
    VK_SYNC_IS_SHARED, VK_SYNC_IS_TIMELINE, VK_SYNC_WAIT_ANY, VK_SYNC_WAIT_PENDING, VK_TIMEOUT,
    VkResult, os_time_get_nano, vk_device, vk_object_base, vk_sync, vk_sync_features,
    vk_sync_flags, vk_sync_type, vk_sync_wait, vk_sync_wait_flags,
};

use super::dev::{RmDev, mesa_is_lost};
use super::{Diag, diag, rm_result, set_lost, try_alloc_zeroed_slice, vk_error};

// Mesa's sync and wait flags are passed to the RM unchanged (`nv_rm.h`).
const _: () = assert!(VK_SYNC_IS_TIMELINE == NV_RM_SYNC_TIMELINE);
const _: () = assert!(VK_SYNC_IS_SHAREABLE == NV_RM_SYNC_SHAREABLE);
const _: () = assert!(VK_SYNC_IS_SHARED == NV_RM_SYNC_SHARED);
const _: () = assert!(VK_SYNC_WAIT_PENDING == NV_RM_SYNC_WAIT_PENDING);
const _: () = assert!(VK_SYNC_WAIT_ANY == NV_RM_SYNC_WAIT_ANY);

/// The object Mesa allocates in place for each sync of these types: Mesa's
/// `vk_sync` and the RM's payload id.
#[repr(C)]
struct RmSync {
    base: vk_sync,
    id: NvRmSyncId,
}

/// Features of both types: CPU and GPU waits (any, pending, before signal),
/// CPU reset and signal.
const COMMON_FEATURES: vk_sync_features = VK_SYNC_FEATURE_BINARY
    | VK_SYNC_FEATURE_GPU_WAIT
    | VK_SYNC_FEATURE_CPU_WAIT
    | VK_SYNC_FEATURE_CPU_RESET
    | VK_SYNC_FEATURE_CPU_SIGNAL
    | VK_SYNC_FEATURE_WAIT_PENDING
    | VK_SYNC_FEATURE_WAIT_ANY
    | VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL;

/// The binary/timeline type with opaque-fd import and export.
pub(super) const SYNC_TYPE: vk_sync_type = vk_sync_type {
    size: size_of::<RmSync>(),
    features: COMMON_FEATURES | VK_SYNC_FEATURE_TIMELINE,
    init: Some(sync_init),
    finish: Some(sync_finish),
    signal: Some(sync_signal),
    signal_many: None,
    get_value: Some(sync_get_value),
    reset: Some(sync_reset),
    reset_many: None,
    move_: Some(sync_move),
    reserve_signal: Some(sync_reserve_signal),
    wait: None,
    wait_many: Some(sync_wait_many),
    import_opaque_fd: Some(sync_import_opaque_fd),
    export_opaque_fd: Some(sync_export_opaque_fd),
    import_win32_handle: None,
    export_win32_handle: None,
    set_win32_export_params: None,
    import_sync_file: None,
    export_sync_file: None,
};

/// The binary type with `sync_file` import and export.
pub(super) const SYNC_FILE_TYPE: vk_sync_type = vk_sync_type {
    size: size_of::<RmSync>(),
    features: COMMON_FEATURES,
    init: Some(sync_init),
    finish: Some(sync_finish),
    signal: Some(sync_signal),
    signal_many: None,
    get_value: None,
    reset: Some(sync_reset),
    reset_many: None,
    move_: Some(sync_move),
    reserve_signal: Some(sync_reserve_signal),
    wait: None,
    wait_many: Some(sync_wait_many),
    import_opaque_fd: None,
    export_opaque_fd: None,
    import_win32_handle: None,
    export_win32_handle: None,
    set_win32_export_params: None,
    import_sync_file: Some(sync_import_sync_file),
    export_sync_file: Some(sync_export_sync_file),
};

/// An import or export handle: an opaque fd (the payload's shared state) or
/// a `sync_file` (a copy of its current fence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handle {
    OpaqueFd,
    SyncFile,
}

impl Handle {
    /// The RM's `opaque` argument (`nv_rm.h` `nv_rm_sync_import_fd`).
    const fn opaque(self) -> u32 {
        match self {
            Self::OpaqueFd => 1,
            Self::SyncFile => 0,
        }
    }
}

/// A sync callback's logical device, KMD device and sync.
struct Call<'a> {
    vk: NonNull<vk_device>,
    dev: &'a RmDev,
    sync: NonNull<RmSync>,
}

impl Call<'_> {
    /// The device as the log object of the callback's diagnostics.
    fn log(&self) -> *mut vk_object_base {
        self.vk.as_ptr().cast()
    }

    /// The sync's RM payload.
    fn id(&self) -> NvRmSyncId {
        // SAFETY: a live sync of our types (`sync_of`); `id` is written only
        // by `init`, before other threads see the sync.
        unsafe { id_of(self.sync).read() }
    }

    /// The sync's Mesa flags (`VK_SYNC_IS_*`).
    fn flags(&self) -> vk_sync_flags {
        // SAFETY: a live sync; Mesa writes its flags only in calls Vulkan
        // externally synchronises with this one (the owning semaphore or
        // fence).
        unsafe { member::<vk_sync_flags>(self.sync.cast(), offset_of!(vk_sync, flags)).read() }
    }
}

/// The member at `offset` of the object at `base`, as a typed pointer (no
/// reference is made: Mesa owns the object).
fn member<T>(base: NonNull<vk_sync>, offset: usize) -> *mut T {
    base.as_ptr().wrapping_byte_add(offset).cast()
}

/// The payload id member of a sync of our types.
fn id_of(sync: NonNull<RmSync>) -> *mut NvRmSyncId {
    member(sync.cast(), offset_of!(RmSync, id))
}

/// The device of a sync callback.
///
/// # Safety
/// `vk` is the live `vk` member of an `nvk_device` (Mesa calls a sync type
/// with the device the sync was made on).
unsafe fn device<'a>(vk: *mut vk_device) -> (NonNull<vk_device>, &'a RmDev) {
    let vk = NonNull::new(vk).expect("Mesa passes the sync's device");
    // SAFETY: a live NVK device (caller), whose KMD device lives as long.
    let dev = unsafe { RmDev::from_vk(vk) }.expect("an NVK device's KMD device is ours");
    (vk, dev)
}

/// The sync of a callback, checked to be of a type `dev` lists.
///
/// # Safety
/// `sync` is NULL or a live `vk_sync`.
unsafe fn sync_of(dev: &RmDev, sync: *mut vk_sync) -> NonNull<RmSync> {
    let sync = NonNull::new(sync).expect("Mesa passes a sync");
    // SAFETY: a live vk_sync (caller); Mesa sets its type before any callback.
    let sync_type =
        unsafe { member::<*const vk_sync_type>(sync, offset_of!(vk_sync, type_)).read() };
    assert!(
        dev.pdev().lists_sync_type(sync_type),
        "Mesa calls a sync type's callbacks with syncs of that type"
    );
    // A sync of our types is the first field of an allocation of
    // `vk_sync_type.size` = size_of::<RmSync>() bytes.
    sync.cast()
}

/// The RM payload behind a sync a context waits on or signals, or `None`
/// when the sync is of a type `dev` does not list (a type check, not an
/// assertion, since any `vk_sync`
/// reaches a queue).
///
/// # Safety
/// `sync` is a live `vk_sync`.
pub(super) unsafe fn payload(dev: &RmDev, sync: NonNull<vk_sync>) -> Option<NvRmSyncId> {
    // SAFETY: a live vk_sync (caller); Mesa sets its type before handing it out.
    let sync_type =
        unsafe { member::<*const vk_sync_type>(sync, offset_of!(vk_sync, type_)).read() };
    if !dev.pdev().lists_sync_type(sync_type) {
        return None;
    }
    // SAFETY: a sync of our types is the first field of an allocation of
    // `size_of::<RmSync>()` bytes, its id written by `init` before the sync
    // reached a queue.
    Some(unsafe { id_of(sync.cast()).read() })
}

/// Whether a live sync of any type is a timeline (`VK_SYNC_IS_TIMELINE`).
///
/// # Safety
/// `sync` is a live `vk_sync`.
pub(super) unsafe fn is_timeline(sync: NonNull<vk_sync>) -> bool {
    // SAFETY: a live vk_sync (caller); Mesa writes its flags only in calls
    // Vulkan externally synchronises with a queue submission.
    let flags = unsafe { member::<vk_sync_flags>(sync, offset_of!(vk_sync, flags)).read() };
    flags & VK_SYNC_IS_TIMELINE != 0
}

/// A callback's device and sync.
///
/// # Safety
/// As [`device`] and [`sync_of`].
unsafe fn call<'a>(vk: *mut vk_device, sync: *mut vk_sync) -> Call<'a> {
    // SAFETY: this function's contract.
    let (vk, dev) = unsafe { device(vk) };
    // SAFETY: this function's contract.
    let sync = unsafe { sync_of(dev, sync) };
    Call { vk, dev, sync }
}

/// Runs a callback that returns a `VkResult` inside the panic guard; a
/// panic is the device's loss to the caller.
fn guarded(entry: &'static str, body: impl FnOnce() -> VkResult) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(entry, None, || VK_ERROR_DEVICE_LOST, body)
}

/// The loss check of every wait: Mesa's loss count, then the RM's loss flag,
/// which becomes Mesa's loss. One atomic load each, no lock.
///
/// # Safety
/// `vk` is the live logical device of `dev`.
unsafe fn wait_status(vk: NonNull<vk_device>, dev: &RmDev) -> VkResult {
    // SAFETY: a live logical device (caller).
    if unsafe { mesa_is_lost(vk) } {
        return VK_ERROR_DEVICE_LOST;
    }
    // SAFETY: the device's live RM device.
    if unsafe { nv_rm_is_lost(dev.rm()) } != 0 {
        // SAFETY: a live logical device (caller).
        return unsafe {
            set_lost(
                vk,
                format_args!(
                    "Native Rust RM maintenance reported a failed GPU dependency or channel"
                ),
            )
        };
    }
    VK_SUCCESS
}

/// `init`: a new RM payload with the sync's flags and initial value.
///
/// # Safety
/// Mesa's `vk_sync_type.init` contract: `vk` the device, `sync` a zeroed
/// allocation of this type's size with type and flags set.
unsafe extern "C" fn sync_init(vk: *mut vk_device, sync: *mut vk_sync, initial: u64) -> VkResult {
    guarded("vk_sync_type.init", || {
        // SAFETY: the contract.
        let call = unsafe { call(vk, sync) };
        // SAFETY: the device's live RM device; the RM writes the sync's own
        // id member, which this backend passes for it.
        let status =
            unsafe { nv_rm_sync_create(call.dev.rm(), call.flags(), initial, id_of(call.sync)) };
        rm_result(call.log(), status, "create sync payload")
    })
}

/// `finish`: drops this payload reference; accepted GPU submissions keep
/// their own until they retire. A failure cannot be returned and is the
/// device's loss; Mesa's loss hook only
/// publishes and wakes, so this may run under Mesa's locks.
///
/// # Safety
/// Mesa's `vk_sync_type.finish` contract: a sync `init` made, not used again.
unsafe extern "C" fn sync_finish(vk: *mut vk_device, sync: *mut vk_sync) {
    nv_ffi::guard_or_abort::<Diag, ()>("vk_sync_type.finish", || {
        // SAFETY: the contract.
        let call = unsafe { call(vk, sync) };
        // SAFETY: the device's live RM device.
        let status = unsafe { nv_rm_sync_destroy_deferred(call.dev.rm(), call.id()) };
        if status != VK_SUCCESS {
            diag(format_args!(
                "Native Rust RM sync disposal failed: {status}"
            ));
            // SAFETY: the live logical device.
            unsafe {
                set_lost(
                    call.vk,
                    format_args!("Native Rust RM sync disposal failed: {status}"),
                );
            }
        }
    });
}

/// `signal`: a host signal (a binary payload ignores `value`).
///
/// # Safety
/// Mesa's `vk_sync_type.signal` contract.
unsafe extern "C" fn sync_signal(vk: *mut vk_device, sync: *mut vk_sync, value: u64) -> VkResult {
    guarded("vk_sync_type.signal", || {
        // SAFETY: the contract.
        let call = unsafe { call(vk, sync) };
        // SAFETY: the device's live RM device.
        let status = unsafe { nv_rm_sync_signal(call.dev.rm(), call.id(), value) };
        rm_result(call.log(), status, "signal sync payload")
    })
}

/// `get_value`: a timeline payload's current value.
///
/// # Safety
/// Mesa's `vk_sync_type.get_value` contract: `value` writable.
unsafe extern "C" fn sync_get_value(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    value: *mut u64,
) -> VkResult {
    guarded("vk_sync_type.get_value", || {
        let value = NonNull::new(value).expect("Mesa passes a writable value");
        // SAFETY: the contract.
        let call = unsafe { call(vk, sync) };
        // SAFETY: the device's live RM device; `value` is writable (contract).
        let status = unsafe { nv_rm_sync_get_value(call.dev.rm(), call.id(), value.as_ptr()) };
        rm_result(call.log(), status, "query sync value")
    })
}

/// `reset`: a binary payload back to unsignalled.
///
/// # Safety
/// Mesa's `vk_sync_type.reset` contract.
unsafe extern "C" fn sync_reset(vk: *mut vk_device, sync: *mut vk_sync) -> VkResult {
    guarded("vk_sync_type.reset", || {
        // SAFETY: the contract.
        let call = unsafe { call(vk, sync) };
        // SAFETY: the device's live RM device.
        let status = unsafe { nv_rm_sync_reset(call.dev.rm(), call.id()) };
        rm_result(call.log(), status, "reset sync payload")
    })
}

/// `move`: `src`'s binary payload into `dst`, telling the RM whether `src`
/// is shared with another process.
///
/// # Safety
/// Mesa's `vk_sync_type.move` contract: two live syncs of this device.
unsafe extern "C" fn sync_move(
    vk: *mut vk_device,
    dst: *mut vk_sync,
    src: *mut vk_sync,
) -> VkResult {
    guarded("vk_sync_type.move", || {
        // SAFETY: the contract.
        let dst = unsafe { call(vk, dst) };
        // SAFETY: the contract.
        let src = unsafe { call(vk, src) };
        let shared = shared_source(src.flags() & VK_SYNC_IS_SHARED != 0);
        // SAFETY: the device's live RM device.
        let status = unsafe { nv_rm_sync_move(dst.dev.rm(), dst.id(), src.id(), shared) };
        rm_result(dst.log(), status, "move sync payload")
    })
}

/// `reserve_signal`: binds `src`'s deferred binary signal to `dst`; `*out`
/// reports whether the RM did (false on any failure).
///
/// # Safety
/// Mesa's `vk_sync_type.reserve_signal` contract: `out` writable.
unsafe extern "C" fn sync_reserve_signal(
    vk: *mut vk_device,
    dst: *mut vk_sync,
    src: *mut vk_sync,
    out: *mut bool,
) -> VkResult {
    guarded("vk_sync_type.reserve_signal", || {
        let out = NonNull::new(out).expect("Mesa passes a writable answer");
        // SAFETY: the contract.
        let dst = unsafe { call(vk, dst) };
        // SAFETY: the contract.
        let src = unsafe { call(vk, src) };
        let mut done = 0;
        // SAFETY: the device's live RM device; `done` is a writable local.
        let status = unsafe { nv_rm_sync_reserve(dst.dev.rm(), dst.id(), src.id(), &raw mut done) };
        let result = rm_result(dst.log(), status, "reserve binary signal");
        // SAFETY: writable (contract).
        unsafe { out.write(reserved(result == VK_SUCCESS, done)) };
        result
    })
}

/// Waits of up to this many points use a stack array: CPU waits on fences
/// and semaphores are small, and their path allocates nothing.
const STACK_POINTS: usize = 16;

/// `wait_many`: waits on the caller's thread until the points complete (all,
/// or any), the caller's deadline passes, or the device is lost. Loss is
/// checked before the RM wait and again after it. The deadline is the caller's: a host timeline signal
/// may take arbitrarily long, and elapsed time is not loss.
///
/// # Safety
/// Mesa's `vk_sync_type.wait_many` contract: `waits` readable for `count`
/// entries, each a live sync of this device.
unsafe extern "C" fn sync_wait_many(
    vk: *mut vk_device,
    count: u32,
    waits: *const vk_sync_wait,
    flags: vk_sync_wait_flags,
    deadline: u64,
) -> VkResult {
    guarded("vk_sync_type.wait_many", || {
        // SAFETY: the contract.
        let (vk, dev) = unsafe { device(vk) };
        // SAFETY: the sync's live logical device.
        if unsafe { wait_status(vk, dev) } != VK_SUCCESS {
            return VK_ERROR_DEVICE_LOST;
        }
        if count == 0 {
            return VK_SUCCESS;
        }
        let waits = NonNull::new(waits.cast_mut()).expect("Mesa passes its waits");
        let len = usize_from_u32(count);
        // SAFETY: readable for `count` entries (contract), not written during
        // the call.
        let waits = unsafe { core::slice::from_raw_parts(waits.as_ptr(), len) };
        let mut stack = [MaybeUninit::<NvRmSyncPoint>::uninit(); STACK_POINTS];
        let mut heap;
        let points: &mut [MaybeUninit<NvRmSyncPoint>] = if len <= STACK_POINTS {
            &mut stack[..len]
        } else {
            let Some(storage) = try_alloc_zeroed_slice::<NvRmSyncPoint>(len) else {
                return vk_error(vk.as_ptr().cast(), VK_ERROR_OUT_OF_HOST_MEMORY);
            };
            heap = storage;
            &mut heap
        };
        for (point, wait) in points.iter_mut().zip(waits) {
            // SAFETY: a live sync of this device (contract).
            let sync = unsafe { sync_of(dev, wait.sync) };
            let call = Call { vk, dev, sync };
            point.write(NvRmSyncPoint {
                sync: call.id(),
                value: wait.wait_value,
            });
        }
        let timeout = wait_timeout(deadline, || {
            // SAFETY: Mesa's clock (util/os_time.c), no preconditions.
            clock_reading(unsafe { os_time_get_nano() })
        });
        // SAFETY: the device's live RM device; `points` holds `count` entries,
        // every one written above.
        let status = unsafe {
            nv_rm_sync_wait_many(
                dev.rm(),
                points.as_ptr().cast::<NvRmSyncPoint>(),
                count,
                flags,
                timeout,
            )
        };
        // SAFETY: the sync's live logical device.
        if unsafe { wait_status(vk, dev) } != VK_SUCCESS {
            return VK_ERROR_DEVICE_LOST;
        }
        match WaitStatus::from_rm(status) {
            WaitStatus::Complete => VK_SUCCESS,
            WaitStatus::Timeout => VK_TIMEOUT,
            WaitStatus::Failed(failed) => {
                rm_result(vk.as_ptr().cast(), failed, "wait sync completion")
            }
        }
    })
}

/// Import: the RM takes the handle's fence state; `fd` is borrowed.
///
/// # Safety
/// Mesa's `vk_sync_type.import_*` contract: `fd` open for the call.
unsafe fn import(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    fd: c_int,
    handle: Handle,
    operation: &str,
) -> VkResult {
    // SAFETY: the contract.
    let call = unsafe { call(vk, sync) };
    // SAFETY: the device's live RM device; `fd` is open for the call.
    let status = unsafe { nv_rm_sync_import_fd(call.dev.rm(), call.id(), fd, handle.opaque()) };
    rm_result(call.log(), status, operation)
}

/// Export: a new descriptor the caller owns, written to `out` only on
/// success. The RM exports through the KMD device's render node.
///
/// # Safety
/// Mesa's `vk_sync_type.export_*` contract: `out` writable.
unsafe fn export(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    out: *mut c_int,
    handle: Handle,
    operation: &str,
) -> VkResult {
    let out = NonNull::new(out).expect("Mesa passes a writable descriptor");
    // SAFETY: the contract.
    let call = unsafe { call(vk, sync) };
    let mut fd = -1;
    // SAFETY: the device's live RM device; `fd` is a writable local.
    let status = unsafe {
        nv_rm_sync_export_fd(
            call.dev.rm(),
            call.id(),
            handle.opaque(),
            call.dev.drm_fd(),
            &raw mut fd,
        )
    };
    let result = rm_result(call.log(), status, operation);
    if result == VK_SUCCESS {
        // SAFETY: writable (contract).
        unsafe { out.write(fd) };
    }
    result
}

unsafe extern "C" fn sync_import_opaque_fd(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    fd: c_int,
) -> VkResult {
    guarded("vk_sync_type.import_opaque_fd", || {
        // SAFETY: Mesa's import contract.
        unsafe { import(vk, sync, fd, Handle::OpaqueFd, "import opaque sync fd") }
    })
}

unsafe extern "C" fn sync_export_opaque_fd(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    out: *mut c_int,
) -> VkResult {
    guarded("vk_sync_type.export_opaque_fd", || {
        // SAFETY: Mesa's export contract.
        unsafe { export(vk, sync, out, Handle::OpaqueFd, "export opaque sync fd") }
    })
}

unsafe extern "C" fn sync_import_sync_file(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    fd: c_int,
) -> VkResult {
    guarded("vk_sync_type.import_sync_file", || {
        // SAFETY: Mesa's import contract.
        unsafe { import(vk, sync, fd, Handle::SyncFile, "import sync file") }
    })
}

unsafe extern "C" fn sync_export_sync_file(
    vk: *mut vk_device,
    sync: *mut vk_sync,
    out: *mut c_int,
) -> VkResult {
    guarded("vk_sync_type.export_sync_file", || {
        // SAFETY: Mesa's export contract.
        unsafe { export(vk, sync, out, Handle::SyncFile, "export sync file") }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timelines_are_read_from_the_flags() {
        let mut sync = vk_sync {
            type_: core::ptr::null(),
            flags: VK_SYNC_IS_TIMELINE,
        };
        // SAFETY: a live vk_sync.
        assert!(unsafe { is_timeline(NonNull::from(&mut sync)) });
        sync.flags = 0;
        // SAFETY: as above.
        assert!(!unsafe { is_timeline(NonNull::from(&mut sync)) });
    }

    #[test]
    fn the_payload_id_follows_the_base() {
        let mut sync = RmSync {
            base: vk_sync {
                type_: core::ptr::null(),
                flags: 0,
            },
            id: NvRmSyncId::new(42),
        };
        let sync = NonNull::from(&mut sync);
        // SAFETY: the id member of a live RmSync.
        assert_eq!(unsafe { id_of(sync).read() }, NvRmSyncId::new(42));
    }
}
