// SPDX-License-Identifier: MIT
//! The C entry points NVK calls (`nvkmd.c` `nvkmd_try_create_pdev_for_drm`).
//! Each validates its pointers once and runs its body inside the shared panic
//! guard. Stateless; callable from any thread.
use nvkmd_sys::{
    _drmDevice, VK_ERROR_DEVICE_LOST, VK_ERROR_INCOMPATIBLE_DRIVER, VkResult, nvk_debug,
    nvkmd_pdev, vk_object_base,
};

/// Platform log tag of this backend's diagnostics.
const TAG: &core::ffi::CStr = c"nvkmd-rm";

/// Panic reports go to the driver's shared sink under [`TAG`].
struct Diag;
impl nv_ffi::PanicLog for Diag {
    fn write(message: core::fmt::Arguments<'_>) {
        nv_ffi::diag::write(TAG, message);
    }
}

/// NVK's probe of one DRM device (`nvkmd_rm.h`). This build creates no
/// physical device: every device is `VK_ERROR_INCOMPATIBLE_DRIVER`, with a
/// diagnostic naming the build option, and `*out` is NULL.
///
/// Ownership: `drm` and `log` are borrowed for the call and not read here;
/// `out` must be NULL or writable. Callable concurrently.
///
/// # Safety
/// `out` is NULL or valid for one pointer write.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nvkmd_rm_try_create_pdev(
    _drm: *mut _drmDevice,
    _log: *mut vk_object_base,
    _debug: nvk_debug,
    out: *mut *mut nvkmd_pdev,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_rm_try_create_pdev",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            if out.is_null() {
                nv_ffi::diag::write(TAG, format_args!("nvkmd_rm_try_create_pdev: NULL output"));
                return VK_ERROR_INCOMPATIBLE_DRIVER;
            }
            // SAFETY: `out` is non-NULL (checked above) and writable for one
            // pointer by this function's contract (NVK passes its local).
            unsafe { out.write(core::ptr::null_mut()) };
            nv_ffi::diag::write(
                TAG,
                format_args!(
                    "nvkmd Rust backend: physical devices are not implemented in this build; \
                     build Mesa with nvk-rm-backend=c"
                ),
            );
            VK_ERROR_INCOMPATIBLE_DRIVER
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_device_is_refused_with_a_null_device() {
        let mut out = core::ptr::NonNull::<nvkmd_pdev>::dangling().as_ptr();
        // SAFETY: `out` is a live local, writable for one pointer.
        let status = unsafe {
            nvkmd_rm_try_create_pdev(
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                &raw mut out,
            )
        };
        assert_eq!(status, VK_ERROR_INCOMPATIBLE_DRIVER);
        assert!(out.is_null());
        // SAFETY: a NULL output is accepted and never written.
        let status = unsafe {
            nvkmd_rm_try_create_pdev(
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(status, VK_ERROR_INCOMPATIBLE_DRIVER);
    }
}
