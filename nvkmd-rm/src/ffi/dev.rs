// SPDX-License-Identifier: MIT
//! The KMD device (`nvkmd_dev`): its own render node and RM device, the VA
//! window NVK allocates from, and, for a logical Vulkan device, the loss hooks
//! between Mesa and the RM (`nvkmd_rm.c` `rm_dev_create`, `rm_dev_destroy`).
//!
//! Loss crosses in both directions: Mesa's loss of the logical device reaches
//! the RM through `vk_device.notify_lost` (`nv_rm_mark_lost`, which only
//! publishes and wakes: it may run under Mesa's submit mutex), and the RM's
//! through `vk_device.check_status`. The hooks are cleared before the RM device
//! closes, so nothing calls into a closing device.
//!
//! The memory, VA and context entries are not served by this backend yet; they
//! fail `VK_ERROR_FEATURE_NOT_PRESENT`.
use std::ffi::c_int;
use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Mutex;

use nv_rm_abi::{
    NV_RM_VA_END, NV_RM_VA_START, NvRmDevice, NvRmInfo, nv_rm_check_status, nv_rm_close,
    nv_rm_mark_lost, nv_rm_open, nv_rm_timestamp,
};
use nvkmd_sys::{
    _vk_device_set_lost, NVKMD_SYS_NVK_DEVICE_NVKMD, NVKMD_SYS_NVK_DEVICE_VK,
    NVKMD_SYS_VK_DEVICE_CHECK_STATUS, NVKMD_SYS_VK_DEVICE_NOTIFY_LOST,
    NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA, NVKMD_SYS_VK_OBJECT_BASE_DEVICE, VK_ERROR_DEVICE_LOST,
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_OUT_OF_HOST_MEMORY, VK_SUCCESS, VkResult, nvkmd_ctx,
    nvkmd_dev, nvkmd_dev_finish, nvkmd_dev_init, nvkmd_dev_ops, nvkmd_engines, nvkmd_mem,
    nvkmd_mem_flags, nvkmd_pdev, nvkmd_sys_layout, nvkmd_sys_vk_device_check_status,
    nvkmd_sys_vk_device_notify_lost, nvkmd_va, nvkmd_va_flags, util_vma_heap, util_vma_heap_finish,
    util_vma_heap_init, vk_device, vk_object_base,
};

use super::pdev::{RmPdev, open_render_node};
use super::{Diag, diag, rm_result, try_alloc_zeroed, unsupported, vk_error};

/// A KMD device. Mesa holds `&base`; [`dev_destroy`] frees it.
#[repr(C)]
pub(super) struct RmDev {
    base: nvkmd_dev,
    rm: NonNull<NvRmDevice>,
    /// The logical device whose loss hooks point here; `None` for the
    /// device the physical-device probe makes.
    vk: Option<NonNull<vk_device>>,
    /// The render node; `None` only during `dev_destroy`, which closes it
    /// before finishing the heap.
    fd: Option<OwnedFd>,
    /// NVK's VA window `[NV_RM_VA_START, NV_RM_VA_END)` (Mesa's
    /// `util_vma_heap`, which points into itself: initialised in place).
    heap: Mutex<util_vma_heap>,
}

static DEV_OPS: nvkmd_dev_ops = nvkmd_dev_ops {
    destroy: Some(dev_destroy),
    get_gpu_timestamp: Some(dev_timestamp),
    get_drm_fd: Some(dev_drm_fd),
    alloc_mem: Some(dev_alloc_mem),
    alloc_tiled_mem: Some(dev_alloc_tiled_mem),
    import_dma_buf: Some(dev_import_dma_buf),
    alloc_va: Some(dev_alloc_va),
    create_ctx: Some(dev_create_ctx),
    create_shared_ctx: Some(dev_create_shared_ctx),
};

impl RmDev {
    /// Our device behind Mesa's pointer, or `None` for NULL or another
    /// backend's.
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_dev`.
    unsafe fn from_base<'a>(base: *mut nvkmd_dev) -> Option<&'a Self> {
        // SAFETY: NULL or a live nvkmd_dev (caller).
        let dev = unsafe { base.as_ref() }?;
        if !core::ptr::eq(dev.ops, &raw const DEV_OPS) {
            return None;
        }
        // SAFETY: a dev with our ops is the first field of a live `RmDev`.
        Some(unsafe { &*base.cast::<Self>() })
    }
}

fn offset(member: nvkmd_sys_layout) -> usize {
    nv_rm_core::conv::usize_from_u32(member)
}

/// Writes the member at `member` of the live `device`.
///
/// # Safety
/// `device` is live and not accessed concurrently; `member` is one of the
/// `NVKMD_SYS_VK_DEVICE_*` offsets of a member of type `T`.
unsafe fn write_member<T>(device: NonNull<vk_device>, member: nvkmd_sys_layout, value: T) {
    // SAFETY: the member lies inside the live device (Mesa's layout, caller).
    let slot = unsafe { device.byte_add(offset(member)) }.cast::<T>();
    // SAFETY: an aligned member of type `T`, not accessed concurrently (caller).
    unsafe { slot.write(value) };
}

/// The physical device's `create_dev`: a KMD device on its own render node
/// and RM device; with a logical device as `log`, the loss hooks installed.
///
/// # Safety
/// `pdev` is a live physical device; `log` NULL or a live Vulkan object;
/// `out` writable for one pointer.
pub(super) unsafe extern "C" fn dev_create(
    pdev: *mut nvkmd_pdev,
    log: *mut vk_object_base,
    out: *mut *mut nvkmd_dev,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_pdev.create_dev",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            if out.is_null() {
                diag(format_args!("nvkmd_pdev.create_dev: NULL output"));
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            // SAFETY: `out` is writable (contract).
            unsafe { out.write(core::ptr::null_mut()) };
            // SAFETY: a live nvkmd_pdev (contract).
            let Some(pdev) = (unsafe { RmPdev::from_base(pdev) }) else {
                diag(format_args!(
                    "nvkmd_pdev.create_dev: not a device of this backend"
                ));
                return VK_ERROR_INITIALIZATION_FAILED;
            };
            // SAFETY: `log` is NULL or live (contract).
            match unsafe { create(pdev, log) } {
                Ok(dev) => {
                    // SAFETY: `out` is writable (above).
                    unsafe { out.write(dev) };
                    VK_SUCCESS
                }
                Err(result) => result,
            }
        },
    )
}

/// The logical device of a log object, if it has one (`vk_object_base.device`).
///
/// # Safety
/// `log` is NULL or a live Vulkan object.
unsafe fn logical_device(log: *mut vk_object_base) -> Option<NonNull<vk_device>> {
    let log = NonNull::new(log)?;
    // SAFETY: `device` lies inside the live object (Mesa's layout).
    let slot =
        unsafe { log.byte_add(offset(NVKMD_SYS_VK_OBJECT_BASE_DEVICE)) }.cast::<*mut vk_device>();
    // SAFETY: an aligned pointer member of a live object.
    NonNull::new(unsafe { slot.read() })
}

/// `create_dev` after validation; the call order and failure values are
/// `nvkmd_rm.c:359-408`.
///
/// # Safety
/// `log` is NULL or a live Vulkan object; when it has a logical device, that
/// device outlives the KMD device (Vulkan: a device outlives its children).
unsafe fn create(pdev: &RmPdev, log: *mut vk_object_base) -> Result<*mut nvkmd_dev, VkResult> {
    let Some(storage) = try_alloc_zeroed::<RmDev>() else {
        return Err(vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY));
    };
    let Some(fd) = open_render_node(&pdev.render_path) else {
        drop(storage);
        return Err(vk_error(log, VK_ERROR_INITIALIZATION_FAILED));
    };
    let info = &pdev.info;
    let mut rm: *mut NvRmDevice = core::ptr::null_mut();
    // SAFETY: `NvRmInfo` is a C struct of integers; all-zero is valid.
    let mut reported: NvRmInfo = unsafe { core::mem::zeroed() };
    // SAFETY: both outputs are writable locals.
    let status = unsafe {
        nv_rm_open(
            info.pci_domain,
            info.pci_bus,
            info.pci_device,
            info.pci_function,
            &raw mut rm,
            &raw mut reported,
        )
    };
    let result = rm_result(log, status, "open device");
    let rm = match (result, NonNull::new(rm)) {
        (VK_SUCCESS, Some(rm)) => rm,
        (failed, made) => {
            // A failed open may still have made a device: it is closed before
            // the render node, as every teardown here orders them.
            if let Some(made) = made {
                // SAFETY: a device nv_rm_open made, closed once.
                unsafe { nv_rm_close(made.as_ptr()) };
            }
            drop(fd);
            drop(storage);
            return Err(if failed == VK_SUCCESS {
                VK_ERROR_INITIALIZATION_FAILED
            } else {
                failed
            });
        }
    };
    // SAFETY: `log` is NULL or live (contract).
    let vk = unsafe { logical_device(log) };
    // SAFETY: all-zero is a valid `nvkmd_dev` (pointers, integers, the list
    // and mutex before their init below).
    let base: nvkmd_dev = unsafe { core::mem::zeroed() };
    // SAFETY: all-zero is a valid `util_vma_heap` before its init below.
    let heap: util_vma_heap = unsafe { core::mem::zeroed() };
    let mut dev = Box::write(
        storage,
        RmDev {
            base,
            rm,
            vk,
            fd: Some(fd),
            heap: Mutex::new(heap),
        },
    );
    dev.base.ops = &raw const DEV_OPS;
    dev.base.pdev = pdev.base_ptr();
    dev.base.va_start = NV_RM_VA_START;
    dev.base.va_end = NV_RM_VA_END;
    // SAFETY: the device's list and mutex, in their final place.
    unsafe { nvkmd_dev_init(&raw mut dev.base) };
    {
        let heap = dev
            .heap
            .get_mut()
            .expect("a mutex no thread has seen is not poisoned");
        // SAFETY: the heap in its final place; the window is NVK's VA range.
        unsafe { util_vma_heap_init(heap, NV_RM_VA_START, NV_RM_VA_END - NV_RM_VA_START) };
    }
    let dev = Box::into_raw(dev);
    if let Some(vk) = vk {
        // SAFETY: a live logical device that outlives this KMD device; its
        // hooks are written once here and cleared in `dev_destroy`.
        unsafe { install_hooks(vk, rm) };
    }
    Ok(dev.cast())
}

/// Points the logical device's loss hooks at this backend: Mesa's loss goes
/// to `nv_rm_mark_lost(rm)`, Mesa's status checks ask the RM.
///
/// # Safety
/// `vk` is live and not used by another thread during the call.
unsafe fn install_hooks(vk: NonNull<vk_device>, rm: NonNull<NvRmDevice>) {
    let notify: nvkmd_sys_vk_device_notify_lost = Some(notify_lost);
    let check: nvkmd_sys_vk_device_check_status = Some(check_status);
    let data: *mut core::ffi::c_void = rm.as_ptr().cast();
    // SAFETY: the members of the live device, of these types (offsets and
    // types from Mesa's header, nvkmd-sys); not accessed concurrently (caller).
    unsafe { write_member(vk, NVKMD_SYS_VK_DEVICE_CHECK_STATUS, check) };
    // SAFETY: as above.
    unsafe { write_member(vk, NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA, data) };
    // SAFETY: as above.
    unsafe { write_member(vk, NVKMD_SYS_VK_DEVICE_NOTIFY_LOST, notify) };
}

/// Clears the hooks [`install_hooks`] set.
///
/// # Safety
/// As [`install_hooks`].
unsafe fn clear_hooks(vk: NonNull<vk_device>) {
    let notify: nvkmd_sys_vk_device_notify_lost = None;
    let check: nvkmd_sys_vk_device_check_status = None;
    // SAFETY: the members of the live device, of these types; not accessed
    // concurrently (caller).
    unsafe { write_member(vk, NVKMD_SYS_VK_DEVICE_NOTIFY_LOST, notify) };
    // SAFETY: as above.
    unsafe {
        write_member(
            vk,
            NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA,
            core::ptr::null_mut::<core::ffi::c_void>(),
        );
    }
    // SAFETY: as above.
    unsafe { write_member(vk, NVKMD_SYS_VK_DEVICE_CHECK_STATUS, check) };
}

/// `vk_device.notify_lost`: Mesa counted a loss of the logical device;
/// publish it to the RM. May run under Mesa's submit mutex: the RM call only
/// publishes and wakes (`nv_rm.h` `nv_rm_mark_lost`).
unsafe extern "C" fn notify_lost(_vk: *mut vk_device, data: *mut core::ffi::c_void) {
    nv_ffi::guard_or_abort::<Diag, ()>("vk_device.notify_lost", || {
        // SAFETY: `data` is the RM device the hook was installed with, live
        // until the hook is cleared in `dev_destroy`.
        unsafe { nv_rm_mark_lost(data.cast::<NvRmDevice>()) };
    });
}

/// `vk_device.check_status`: the RM's view of the device; a failed GPU
/// dependency or channel is the logical device's loss.
unsafe extern "C" fn check_status(vk: *mut vk_device) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "vk_device.check_status",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            let Some(vk) = NonNull::new(vk) else {
                return VK_ERROR_DEVICE_LOST;
            };
            // SAFETY: Mesa calls the hook on the live logical device of an
            // `nvk_device` (`vk` is its first member, offset checked by nvkmd-sys).
            let nvk = unsafe { vk.byte_sub(offset(NVKMD_SYS_NVK_DEVICE_VK)) };
            // SAFETY: the `nvkmd` member lies inside that live nvk_device.
            let slot = unsafe { nvk.byte_add(offset(NVKMD_SYS_NVK_DEVICE_NVKMD)) }
                .cast::<*mut nvkmd_dev>();
            // SAFETY: an aligned pointer member NVK set at device creation.
            let kmd = unsafe { slot.read() };
            // SAFETY: NULL or the device's live KMD device.
            let Some(dev) = (unsafe { RmDev::from_base(kmd) }) else {
                return VK_ERROR_DEVICE_LOST;
            };
            // SAFETY: the device's live RM device.
            if unsafe { nv_rm_check_status(dev.rm.as_ptr()) } == VK_SUCCESS {
                return VK_SUCCESS;
            }
            // SAFETY: the live logical device; "%s" reads one string.
            unsafe {
                _vk_device_set_lost(
                    vk.as_ptr(),
                    super::TAG.as_ptr(),
                    0,
                    c"%s".as_ptr(),
                    c"Native Rust RM reported a failed GPU dependency or channel".as_ptr(),
                )
            }
        },
    )
}

/// The device's `destroy`, in `nvkmd_rm.c:562-583` order: hooks cleared, RM
/// device closed (it may still return VA ranges to the heap), render node
/// closed, heap and lists finished, storage freed.
///
/// # Safety
/// `base` is a device `create_dev` returned, not used again.
unsafe extern "C" fn dev_destroy(base: *mut nvkmd_dev) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_dev.destroy", || {
        // SAFETY: a live nvkmd_dev (contract).
        let dev =
            unsafe { RmDev::from_base(base) }.expect("Mesa destroys its own backend's devices");
        // SAFETY: made by Box::into_raw in `create`; this is its one destroy.
        let mut dev = unsafe { Box::from_raw(core::ptr::from_ref(dev).cast_mut()) };
        if let Some(vk) = dev.vk {
            // SAFETY: NVK destroys the KMD device before its logical device,
            // with no other call on it in flight (vkDestroyDevice is externally
            // synchronised).
            unsafe { clear_hooks(vk) };
        }
        // SAFETY: the device's RM device, closed once.
        unsafe { nv_rm_close(dev.rm.as_ptr()) };
        drop(dev.fd.take());
        // The heap and the device's list point into themselves: finished in
        // place, before the storage is freed.
        let heap = dev
            .heap
            .get_mut()
            .expect("the heap lock is never held across a panic");
        // SAFETY: the heap `create` initialised; nothing allocates from it now.
        unsafe { util_vma_heap_finish(heap) };
        // SAFETY: the list and mutex `create` initialised.
        unsafe { nvkmd_dev_finish(&raw mut dev.base) };
        drop(dev);
    });
}

/// The device's `get_gpu_timestamp`: 0 when the RM cannot read it, with a
/// diagnostic (Mesa's interface has no error return).
unsafe extern "C" fn dev_timestamp(base: *mut nvkmd_dev) -> u64 {
    nv_ffi::guard_or_abort::<Diag, u64>("nvkmd_dev.get_gpu_timestamp", || {
        // SAFETY: a live nvkmd_dev (contract).
        let dev = unsafe { RmDev::from_base(base) }.expect("Mesa calls its own backend's devices");
        let mut timestamp = 0;
        // SAFETY: the device's live RM device; `timestamp` is writable.
        let status = unsafe { nv_rm_timestamp(dev.rm.as_ptr(), &raw mut timestamp) };
        if status != VK_SUCCESS {
            diag(format_args!("Native Rust RM timestamp failed: {status}"));
        }
        timestamp
    })
}

/// The device's `get_drm_fd`: its render node.
unsafe extern "C" fn dev_drm_fd(base: *mut nvkmd_dev) -> c_int {
    nv_ffi::guard_or_abort::<Diag, c_int>("nvkmd_dev.get_drm_fd", || {
        // SAFETY: a live nvkmd_dev (contract).
        let dev = unsafe { RmDev::from_base(base) }.expect("Mesa calls its own backend's devices");
        dev.fd.as_ref().map_or(-1, AsRawFd::as_raw_fd)
    })
}

/// The memory, VA and context entries: not served (module docs).
fn not_served(log: *mut vk_object_base, entry: &'static str, what: &str) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        entry,
        None,
        || VK_ERROR_DEVICE_LOST,
        || unsupported(log, what),
    )
}

unsafe extern "C" fn dev_alloc_mem(
    _dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    _size: u64,
    _align: u64,
    _flags: nvkmd_mem_flags,
    _out: *mut *mut nvkmd_mem,
) -> VkResult {
    not_served(log, "nvkmd_dev.alloc_mem", "memory allocation")
}

unsafe extern "C" fn dev_alloc_tiled_mem(
    _dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    _size: u64,
    _align: u64,
    _pte_kind: u8,
    _tile_mode: u16,
    _flags: nvkmd_mem_flags,
    _out: *mut *mut nvkmd_mem,
) -> VkResult {
    not_served(log, "nvkmd_dev.alloc_tiled_mem", "memory allocation")
}

unsafe extern "C" fn dev_import_dma_buf(
    _dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    _fd: c_int,
    _out: *mut *mut nvkmd_mem,
) -> VkResult {
    not_served(log, "nvkmd_dev.import_dma_buf", "dma-buf import")
}

unsafe extern "C" fn dev_alloc_va(
    _dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    _flags: nvkmd_va_flags,
    _pte_kind: u8,
    _size: u64,
    _align: u64,
    _fixed: u64,
    _out: *mut *mut nvkmd_va,
) -> VkResult {
    not_served(log, "nvkmd_dev.alloc_va", "VA allocation")
}

unsafe extern "C" fn dev_create_ctx(
    _dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    _engines: nvkmd_engines,
    _out: *mut *mut nvkmd_ctx,
) -> VkResult {
    not_served(log, "nvkmd_dev.create_ctx", "contexts")
}

unsafe extern "C" fn dev_create_shared_ctx(
    _dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    _engines: nvkmd_engines,
    _share: *mut nvkmd_ctx,
    _out: *mut *mut nvkmd_ctx,
) -> VkResult {
    not_served(log, "nvkmd_dev.create_shared_ctx", "contexts")
}
