// SPDX-License-Identifier: MIT
//! The C side of the backend: the entry NVK calls (`nvkmd.c`
//! `nvkmd_try_create_pdev_for_drm` → `nvkmd_rm_try_create_pdev`), the ops
//! tables and `vk_sync_type`s it hands to Mesa, and the calls into Mesa,
//! libc, libdrm and the RM. Every entry validates its pointers once and runs
//! its body inside the shared panic guard (`nv-ffi`).
//!
//! Objects NVK sees are `#[repr(C)]` structs whose first field is Mesa's base
//! struct, allocated here and handed out as a pointer to that base; an entry
//! recovers its object after checking the base's ops pointer is ours. Mesa
//! serialises calls per object as its own backends require (`nvkmd.h`); the
//! entries take no lock of their own.
use std::alloc::Layout;
use std::ffi::{CStr, CString, c_int};
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use nvkmd_sys::{
    __vk_errorf, _vk_device_set_lost, VK_ERROR_FEATURE_NOT_PRESENT, VK_SUCCESS, VkResult,
    vk_device, vk_object_base,
};

mod ctx;
mod dev;
mod memory;
mod object;
mod pdev;
mod sync;
mod va;

/// Platform log tag of this backend's diagnostics.
const TAG: &CStr = c"nvkmd-rm";

/// Panic reports go to the driver's shared sink under [`TAG`].
struct Diag;
impl nv_ffi::PanicLog for Diag {
    fn write(message: fmt::Arguments<'_>) {
        nv_ffi::diag::write(TAG, message);
    }
}

/// A diagnostic line on the driver's sink, for failures no Vulkan call
/// returns (void callbacks).
fn diag(message: fmt::Arguments<'_>) {
    nv_ffi::diag::write(TAG, message);
}

/// Mesa's `vk_errorf` (`vk_log.h`): reports `message` against the log object
/// (NULL during enumeration) and returns `result`.
#[track_caller]
fn vk_errorf(log: *mut vk_object_base, result: VkResult, message: fmt::Arguments<'_>) -> VkResult {
    let text = message.to_string().replace('\0', "\u{fffd}");
    let text = CString::new(text).expect("interior NULs were replaced");
    // SAFETY: `log` is NULL or the caller's live log object (each entry's
    // contract); "%s" reads exactly one NUL-terminated string, `text`, which
    // outlives the call.
    unsafe {
        __vk_errorf(
            log.cast(),
            result,
            TAG.as_ptr(),
            line(),
            c"%s".as_ptr(),
            text.as_ptr(),
        )
    }
}

/// Mesa's `vk_device_set_lost` (`vk_device.h`): counts the logical device's
/// loss, runs its `notify_lost` hook on the first one and reports `message`;
/// returns `VK_ERROR_DEVICE_LOST`.
///
/// # Safety
/// `vk` is a live logical device.
#[track_caller]
unsafe fn set_lost(vk: NonNull<vk_device>, message: fmt::Arguments<'_>) -> VkResult {
    let text = message.to_string().replace('\0', "\u{fffd}");
    let text = CString::new(text).expect("interior NULs were replaced");
    // SAFETY: a live logical device (caller); "%s" reads exactly one
    // NUL-terminated string, `text`, which outlives the call.
    unsafe {
        _vk_device_set_lost(
            vk.as_ptr(),
            TAG.as_ptr(),
            line(),
            c"%s".as_ptr(),
            text.as_ptr(),
        )
    }
}

/// Mesa's `vk_error` (`vk_log.h`): reports `result` without a message.
#[track_caller]
fn vk_error(log: *mut vk_object_base, result: VkResult) -> VkResult {
    // SAFETY: `log` is NULL or the caller's live log object; a NULL format
    // reads no arguments.
    unsafe { __vk_errorf(log.cast(), result, TAG.as_ptr(), line(), core::ptr::null()) }
}

/// The line Mesa's diagnostics carry next to [`TAG`]: the reporting call's.
#[track_caller]
fn line() -> c_int {
    c_int::try_from(std::panic::Location::caller().line()).unwrap_or(c_int::MAX)
}

/// An RM status as a `VkResult`: 0 is success, anything else is the
/// `VkResult` itself (the RM maps its errors per Vulkan command, `nv_rm.h`),
/// reported against `log` naming `operation`.
#[track_caller]
fn rm_result(log: *mut vk_object_base, status: i32, operation: &str) -> VkResult {
    if status == VK_SUCCESS {
        return VK_SUCCESS;
    }
    vk_errorf(
        log,
        status,
        format_args!("Native Rust RM {operation} failed"),
    )
}

/// `VK_ERROR_FEATURE_NOT_PRESENT` for a request this backend does not serve.
#[track_caller]
fn unsupported(log: *mut vk_object_base, what: &str) -> VkResult {
    vk_errorf(
        log,
        VK_ERROR_FEATURE_NOT_PRESENT,
        format_args!("Native Rust RM transport does not implement {what}"),
    )
}

/// Zeroed heap storage for one `T`, or `None` when the allocator is out of
/// memory. Rust's system allocator serves a zeroed request with `calloc`,
/// as NVK's backends allocate their objects (`nvkmd_nouveau_*.c`), so host
/// memory exhaustion is `VK_ERROR_OUT_OF_HOST_MEMORY` instead of an abort.
/// (`Box::try_new` is not stable.)
fn try_alloc_zeroed<T>() -> Option<Box<MaybeUninit<T>>> {
    const { assert!(size_of::<T>() != 0, "backend objects are not zero-sized") };
    let layout = Layout::new::<T>();
    // SAFETY: `layout` has a non-zero size (asserted above).
    let storage = unsafe { std::alloc::alloc_zeroed(layout) };
    let storage = NonNull::new(storage.cast::<MaybeUninit<T>>())?;
    // SAFETY: allocated by the global allocator with `T`'s layout, so the box
    // owns it and frees it with the same layout; `MaybeUninit` needs no
    // initialisation.
    Some(unsafe { Box::from_raw(storage.as_ptr()) })
}

/// Zeroed heap storage for `count` values of `T`, or `None` when the
/// allocator is out of memory or the size overflows. Served by `calloc`, as
/// [`try_alloc_zeroed`], for app-sized arrays (wait lists).
fn try_alloc_zeroed_slice<T>(count: usize) -> Option<Box<[MaybeUninit<T>]>> {
    const { assert!(size_of::<T>() != 0, "backend arrays are not zero-sized") };
    if count == 0 {
        return Some(Box::new([]));
    }
    let layout = Layout::array::<T>(count).ok()?;
    // SAFETY: `layout` has a non-zero size (non-zero `T`, `count` > 0).
    let storage = unsafe { std::alloc::alloc_zeroed(layout) };
    let storage = NonNull::new(storage.cast::<MaybeUninit<T>>())?;
    let slice = core::ptr::slice_from_raw_parts_mut(storage.as_ptr(), count);
    // SAFETY: allocated by the global allocator with the layout of
    // `[T; count]`, so the box owns it and frees it with the same layout;
    // `MaybeUninit` needs no initialisation.
    Some(unsafe { Box::from_raw(slice) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_objects_are_zeroed_and_freed() {
        let object = try_alloc_zeroed::<[u64; 4]>().expect("host memory");
        // SAFETY: all-zero bytes are a valid `[u64; 4]`.
        assert_eq!(*unsafe { object.assume_init() }, [0; 4]);
    }

    #[test]
    fn arrays_are_zeroed_and_sized() {
        assert_eq!(
            try_alloc_zeroed_slice::<u64>(0).map(|slice| slice.len()),
            Some(0)
        );
        let array = try_alloc_zeroed_slice::<u32>(37).expect("host memory");
        assert_eq!(array.len(), 37);
        let zero = |value: &MaybeUninit<u32>| {
            // SAFETY: all-zero bytes are a valid `u32`.
            unsafe { value.assume_init() == 0 }
        };
        assert!(array.iter().all(zero));
    }

    #[test]
    fn oversized_arrays_are_refused() {
        assert!(try_alloc_zeroed_slice::<u64>(usize::MAX / 4).is_none());
    }
}
