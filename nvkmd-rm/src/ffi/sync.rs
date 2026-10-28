// SPDX-License-Identifier: MIT
//! The two `vk_sync_type`s every physical device lists: a binary/timeline type
//! with opaque-fd import and export, and a binary type with `sync_file` import
//! and export (`nvkmd_rm.c` `rm_sync_type`, `rm_sync_file_type`). Separate
//! types because opaque shared state and sync-file copies have different
//! lifetime contracts; Mesa's external-handle queries then report same-type
//! compatibility only.
//!
//! The payload operations over the RM are not served by this backend yet:
//! `init` fails `VK_ERROR_FEATURE_NOT_PRESENT`, so no payload exists and no
//! other callback can be reached with one. The callbacks the C backend sets are
//! set here too, so the types' shapes (features, callback slots, payload size)
//! are what Mesa's queries see.
use std::ffi::c_int;

use nv_rm_abi::NvRmSyncId;
use nvkmd_sys::{
    VK_SYNC_FEATURE_BINARY, VK_SYNC_FEATURE_CPU_RESET, VK_SYNC_FEATURE_CPU_SIGNAL,
    VK_SYNC_FEATURE_CPU_WAIT, VK_SYNC_FEATURE_GPU_WAIT, VK_SYNC_FEATURE_TIMELINE,
    VK_SYNC_FEATURE_WAIT_ANY, VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL, VK_SYNC_FEATURE_WAIT_PENDING,
    VkResult, vk_device, vk_object_base, vk_sync, vk_sync_features, vk_sync_type, vk_sync_wait,
    vk_sync_wait_flags,
};

use super::{Diag, unsupported};

/// The payload Mesa allocates in place for each sync object of these types:
/// Mesa's `vk_sync` and the RM's payload id (`nvkmd_rm.c` `nvkmd_rm_sync`).
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
    import_opaque_fd: Some(sync_import_fd),
    export_opaque_fd: Some(sync_export_fd),
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
    import_sync_file: Some(sync_import_fd),
    export_sync_file: Some(sync_export_fd),
};

/// The device as the log object of a sync callback (its `vk_object_base` is
/// its first member, `vk_device.h`).
fn log_of(device: *mut vk_device) -> *mut vk_object_base {
    device.cast()
}

/// Every payload operation: not served, see the module docs.
fn not_served(device: *mut vk_device, entry: &'static str) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        entry,
        None,
        || nvkmd_sys::VK_ERROR_DEVICE_LOST,
        || unsupported(log_of(device), "RM synchronization payloads"),
    )
}

unsafe extern "C" fn sync_init(
    device: *mut vk_device,
    _sync: *mut vk_sync,
    _initial: u64,
) -> VkResult {
    not_served(device, "vk_sync_type.init")
}

/// Unreachable while `init` fails: Mesa finishes only initialised syncs.
unsafe extern "C" fn sync_finish(_device: *mut vk_device, _sync: *mut vk_sync) {
    nv_ffi::guard_or_abort::<Diag, ()>("vk_sync_type.finish", || {
        panic!("finish of a sync payload this backend never created");
    });
}

unsafe extern "C" fn sync_signal(
    device: *mut vk_device,
    _sync: *mut vk_sync,
    _value: u64,
) -> VkResult {
    not_served(device, "vk_sync_type.signal")
}

unsafe extern "C" fn sync_get_value(
    device: *mut vk_device,
    _sync: *mut vk_sync,
    _value: *mut u64,
) -> VkResult {
    not_served(device, "vk_sync_type.get_value")
}

unsafe extern "C" fn sync_reset(device: *mut vk_device, _sync: *mut vk_sync) -> VkResult {
    not_served(device, "vk_sync_type.reset")
}

unsafe extern "C" fn sync_move(
    device: *mut vk_device,
    _dst: *mut vk_sync,
    _src: *mut vk_sync,
) -> VkResult {
    not_served(device, "vk_sync_type.move")
}

unsafe extern "C" fn sync_reserve_signal(
    device: *mut vk_device,
    _dst: *mut vk_sync,
    _src: *mut vk_sync,
    _reserved: *mut bool,
) -> VkResult {
    not_served(device, "vk_sync_type.reserve_signal")
}

unsafe extern "C" fn sync_wait_many(
    device: *mut vk_device,
    _count: u32,
    _waits: *const vk_sync_wait,
    _flags: vk_sync_wait_flags,
    _deadline: u64,
) -> VkResult {
    not_served(device, "vk_sync_type.wait_many")
}

unsafe extern "C" fn sync_import_fd(
    device: *mut vk_device,
    _sync: *mut vk_sync,
    _fd: c_int,
) -> VkResult {
    not_served(device, "vk_sync_type import")
}

unsafe extern "C" fn sync_export_fd(
    device: *mut vk_device,
    _sync: *mut vk_sync,
    _fd: *mut c_int,
) -> VkResult {
    not_served(device, "vk_sync_type export")
}
