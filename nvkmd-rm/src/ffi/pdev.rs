// SPDX-License-Identifier: MIT
//! The physical device: NVK's probe of one DRM device
//! ([`nvkmd_rm_try_create_pdev`]), what it reports to NVK, and its ops table.
//!
//! The probe accepts a PCI device of NVIDIA with a render node driven by
//! `nvidia-drm`, opens the RM once to read what it reports, and closes it
//! again; the logical devices made later reopen both (`create_dev`). The call
//! sequence (render node open, version query, stat, RM open/close) and the
//! results of every failure are those of `nvkmd_rm.c` `nvkmd_rm_try_create_pdev`.
use std::ffi::{CStr, CString, c_int};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use nv_rm_abi::{NvRmDevice, NvRmInfo, nv_rm_close, nv_rm_open};
use nv_rm_core::conv::usize_from_u32;
use nvkmd_rm_logic::info::{BIND_ALIGN, device_info, kmd_info};
use nvkmd_sys::{
    _drmDevice, DRM_BUS_PCI, DRM_NODE_RENDER, NV_DEVICE_TYPE_DIS, O_CLOEXEC, O_RDWR,
    VK_ERROR_DEVICE_LOST, VK_ERROR_INCOMPATIBLE_DRIVER, VK_ERROR_INITIALIZATION_FAILED,
    VK_ERROR_OUT_OF_HOST_MEMORY, VK_SUCCESS, VkResult, dev_t, drmFreeVersion, drmGetVersion, fstat,
    nv_device_info, nv_zcull_device_info, nvk_debug, nvkmd_dev_info_init_shared_mem_sizes,
    nvkmd_info, nvkmd_pdev, nvkmd_pdev__bindgen_ty_1, nvkmd_pdev_ops, open, stat,
    util_cache_granularity, vk_object_base, vk_sync_type,
};

use super::sync::{SYNC_FILE_TYPE, SYNC_TYPE};
use super::{Diag, dev, rm_result, try_alloc_zeroed, vk_error, vk_errorf};

/// PCI vendor id of NVIDIA.
const NVIDIA_VENDOR_ID: u16 = 0x10de;
/// The DRM driver name of NVIDIA's kernel modules.
const NVIDIA_DRM: &CStr = c"nvidia-drm";

/// A physical device. Mesa holds `&base`; [`pdev_destroy`] frees it.
#[repr(C)]
pub(super) struct RmPdev {
    base: nvkmd_pdev,
    /// What the RM reported at the probe; the PCI address reopens the device.
    pub(super) info: NvRmInfo,
    /// The render node, reopened by every logical device.
    pub(super) render_path: CString,
    /// The render node, open for the physical device's lifetime.
    fd: OwnedFd,
    sync_types: [vk_sync_type; 2],
    /// `base.sync_types`: the two types above, NULL-terminated.
    sync_type_list: [*const vk_sync_type; 3],
}

pub(super) static PDEV_OPS: nvkmd_pdev_ops = nvkmd_pdev_ops {
    destroy: Some(pdev_destroy),
    get_vram_used: Some(pdev_vram_used),
    get_drm_primary_fd: None,
    create_dev: Some(dev::dev_create),
};

impl RmPdev {
    /// Our physical device behind Mesa's pointer, or `None` when `base` is
    /// NULL or another backend's (its ops are not ours).
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_pdev`.
    pub(super) unsafe fn from_base<'a>(base: *mut nvkmd_pdev) -> Option<&'a Self> {
        // SAFETY: NULL or a live nvkmd_pdev (caller).
        let pdev = unsafe { base.as_ref() }?;
        if !core::ptr::eq(pdev.ops, &raw const PDEV_OPS) {
            return None;
        }
        // SAFETY: a pdev with our ops was made by `nvkmd_rm_try_create_pdev`
        // as the first field of a live, `#[repr(C)]` `RmPdev`.
        Some(unsafe { &*base.cast::<Self>() })
    }

    pub(super) fn base_ptr(&self) -> *mut nvkmd_pdev {
        (&raw const self.base).cast_mut()
    }
}

/// A value narrowed to the width of the Mesa field it is reported in.
fn narrow<T: TryFrom<u64>>(value: u64) -> Result<T, T::Error> {
    T::try_from(value)
}

/// NVK's probe of one DRM device (`nvkmd_rm.h`): `VK_SUCCESS` and the physical
/// device in `*out`, or `VK_ERROR_INCOMPATIBLE_DRIVER` for a device this
/// backend does not drive (Mesa's enumeration skips those, `vk_instance.c`
/// `enumerate_drm_physical_devices_locked`), or the error that stopped an
/// NVIDIA device from being probed; `*out` is NULL then.
///
/// Ownership: `drm` and `log` are borrowed for the call; the device returned
/// belongs to Mesa until its `destroy`. Callable concurrently.
///
/// # Safety
/// `drm` is NULL or a complete libdrm device record; `log` is NULL or a live
/// Vulkan object; `out` is NULL or writable for one pointer.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nvkmd_rm_try_create_pdev(
    drm: *mut _drmDevice,
    log: *mut vk_object_base,
    debug: nvk_debug,
    out: *mut *mut nvkmd_pdev,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_rm_try_create_pdev",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            if out.is_null() {
                super::diag(format_args!("nvkmd_rm_try_create_pdev: NULL output"));
                return VK_ERROR_INCOMPATIBLE_DRIVER;
            }
            // SAFETY: `out` is non-NULL and writable (contract).
            unsafe { out.write(core::ptr::null_mut()) };
            // SAFETY: NULL or a complete libdrm record (contract).
            let Some(drm) = (unsafe { drm.as_ref() }) else {
                return VK_ERROR_INCOMPATIBLE_DRIVER;
            };
            // SAFETY: as above.
            match unsafe { probe(drm, log, debug) } {
                Ok(pdev) => {
                    // SAFETY: `out` is writable (above).
                    unsafe { out.write(pdev) };
                    VK_SUCCESS
                }
                Err(result) => result,
            }
        },
    )
}

/// The render node of `drm` when it is an NVIDIA PCI device that has one.
///
/// # Safety
/// `drm` is a complete libdrm device record.
unsafe fn nvidia_render_node(drm: &_drmDevice) -> Option<&CStr> {
    if u32::try_from(drm.bustype) != Ok(DRM_BUS_PCI) {
        return None;
    }
    // SAFETY: a PCI record's device-info union holds its PCI member (libdrm).
    let pci = unsafe { drm.deviceinfo.pci };
    // SAFETY: NULL or the record's PCI device info, live with the record.
    let pci = unsafe { pci.as_ref() }?;
    if pci.vendor_id != NVIDIA_VENDOR_ID || drm.available_nodes & (1 << DRM_NODE_RENDER) == 0 {
        return None;
    }
    // SAFETY: `nodes` has DRM_NODE_MAX entries (libdrm).
    let slot = unsafe { drm.nodes.add(usize_from_u32(DRM_NODE_RENDER)) };
    // SAFETY: an entry of the record's node array.
    let node = unsafe { slot.read() };
    // SAFETY: an available node's path is a NUL-terminated string the record
    // owns for its lifetime.
    (!node.is_null()).then(|| unsafe { CStr::from_ptr(node) })
}

/// Opens `path` read-write, close-on-exec (§13), through libc's `open` as the
/// C backend does.
pub(super) fn open_render_node(path: &CStr) -> Option<OwnedFd> {
    let flags = c_int::try_from(O_RDWR | O_CLOEXEC).expect("open flags fit int");
    // SAFETY: `path` is NUL-terminated; no O_CREAT, so no mode argument.
    let fd = unsafe { open(path.as_ptr(), flags) };
    // SAFETY: a non-negative result is a descriptor this call opened and
    // nothing else owns.
    (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Whether the render node is driven by `nvidia-drm` (libdrm's version query).
fn is_nvidia_drm(fd: &OwnedFd) -> bool {
    // SAFETY: `fd` is an open descriptor for the call.
    let version = unsafe { drmGetVersion(fd.as_raw_fd()) };
    // SAFETY: NULL or a record libdrm allocated, live until drmFreeVersion.
    let Some(record) = (unsafe { version.as_ref() }) else {
        return false;
    };
    // SAFETY: a non-NULL name is a NUL-terminated string of the record.
    let native = !record.name.is_null() && unsafe { CStr::from_ptr(record.name) } == NVIDIA_DRM;
    // SAFETY: the record came from drmGetVersion and is freed once.
    unsafe { drmFreeVersion(version) };
    native
}

/// The probe after the output was cleared. Failure values and the order of
/// render node, libdrm and RM calls are those of `nvkmd_rm.c:416-560`.
///
/// # Safety
/// `drm` is a complete libdrm device record; `log` is NULL or a live Vulkan
/// object.
unsafe fn probe(
    drm: &_drmDevice,
    log: *mut vk_object_base,
    debug: nvk_debug,
) -> Result<*mut nvkmd_pdev, VkResult> {
    // SAFETY: contract.
    let path = unsafe { nvidia_render_node(drm) }.ok_or(VK_ERROR_INCOMPATIBLE_DRIVER)?;
    let fd = open_render_node(path).ok_or(VK_ERROR_INCOMPATIBLE_DRIVER)?;
    if !is_nvidia_drm(&fd) {
        return Err(VK_ERROR_INCOMPATIBLE_DRIVER);
    }
    let Some(storage) = try_alloc_zeroed::<RmPdev>() else {
        drop(fd);
        return Err(vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY));
    };
    // SAFETY: `stat` is a C struct of integers; all-zero is a valid value.
    let mut st: stat = unsafe { core::mem::zeroed() };
    // SAFETY: `fd` is open; `st` is writable.
    if unsafe { fstat(fd.as_raw_fd(), &raw mut st) } != 0 {
        let error = std::io::Error::last_os_error();
        return Err(vk_errorf(
            log,
            VK_ERROR_INITIALIZATION_FAILED,
            format_args!("Native Rust RM render-node stat failed: {error}"),
        ));
    }
    // SAFETY: a PCI record's bus-info union holds its PCI member (libdrm;
    // the bus type was checked above).
    let bus = unsafe { drm.businfo.pci };
    // SAFETY: NULL or the record's PCI bus info, live with the record.
    let bus = unsafe { bus.as_ref() }.ok_or(VK_ERROR_INCOMPATIBLE_DRIVER)?;
    let mut rm: *mut NvRmDevice = core::ptr::null_mut();
    // SAFETY: `NvRmInfo` is a C struct of integers; all-zero is valid.
    let mut info: NvRmInfo = unsafe { core::mem::zeroed() };
    // SAFETY: both outputs are writable locals.
    let status = unsafe {
        nv_rm_open(
            u32::from(bus.domain),
            u32::from(bus.bus),
            u32::from(bus.dev),
            u32::from(bus.func),
            &raw mut rm,
            &raw mut info,
        )
    };
    let result = rm_result(log, status, "query device");
    // The probe only reads what the RM reports: its device is closed on every
    // path, a failed open's included when it made one (`nv_rm.h`: NULL is
    // ignored, so success closes unconditionally).
    if result != VK_SUCCESS {
        if !rm.is_null() {
            // SAFETY: a device nv_rm_open made, closed once.
            unsafe { nv_rm_close(rm) };
        }
        return Err(result);
    }
    // SAFETY: as above.
    unsafe { nv_rm_close(rm) };
    let pdev = describe(&info, log, debug, &st)?;
    let mut storage = Box::write(
        storage,
        RmPdev {
            base: pdev,
            info,
            render_path: path.to_owned(),
            fd,
            sync_types: [SYNC_TYPE, SYNC_FILE_TYPE],
            sync_type_list: [core::ptr::null(); 3],
        },
    );
    storage.sync_type_list = [
        &raw const storage.sync_types[0],
        &raw const storage.sync_types[1],
        core::ptr::null(),
    ];
    storage.base.sync_types = storage.sync_type_list.as_ptr();
    Ok(Box::into_raw(storage).cast())
}

/// What NVK learns about the device (`nv_device_info`, `nvkmd_info`), from
/// what the RM reported.
///
/// A value that does not fit its Mesa field is refused with
/// `VK_ERROR_INCOMPATIBLE_DRIVER`, as NVK refuses devices it cannot drive
/// (`nvk_physical_device.c` `nvk_create_drm_physical_device`, pre-Kepler and
/// non-conformant devices): Mesa's enumeration then skips this device and
/// keeps the others (`vk_instance.c` `enumerate_drm_physical_devices_locked`),
/// where a truncated value would describe a different device than the RM
/// opened.
fn describe(
    info: &NvRmInfo,
    log: *mut vk_object_base,
    debug: nvk_debug,
    st: &stat,
) -> Result<nvkmd_pdev, VkResult> {
    let refuse = |what: &dyn std::fmt::Display| {
        vk_errorf(
            log,
            VK_ERROR_INCOMPATIBLE_DRIVER,
            format_args!("Native Rust RM device not representable: {what}"),
        )
    };
    let device = device_info(info).map_err(|error| refuse(&error))?;
    let kmd = kmd_info(info);
    // SAFETY: `nv_device_info` is a C struct of integers, arrays and bools;
    // all-zero is valid and is what the C designated initialiser leaves in
    // every field not named.
    let mut dev_info: nv_device_info = unsafe { core::mem::zeroed() };
    dev_info.type_ = NV_DEVICE_TYPE_DIS;
    dev_info.device_id = device.device_id;
    dev_info.chipset = device.chipset;
    for (slot, byte) in dev_info.device_name.iter_mut().zip(device.device_name) {
        *slot = std::ffi::c_char::from_ne_bytes([byte]);
    }
    for (slot, byte) in dev_info.chipset_name.iter_mut().zip(device.chipset_name) {
        *slot = std::ffi::c_char::from_ne_bytes([byte]);
    }
    dev_info.pci.domain = device.pci.domain;
    dev_info.pci.bus = device.pci.bus;
    dev_info.pci.dev = device.pci.dev;
    dev_info.pci.func = device.pci.func;
    dev_info.sm = device.sm;
    dev_info.gpc_count = device.gpc_count;
    dev_info.tpc_count = device.tpc_count;
    dev_info.mp_per_tpc = device.mp_per_tpc;
    dev_info.max_warps_per_mp = device.max_warps_per_mp;
    dev_info.cls_copy = device.classes.copy;
    dev_info.cls_eng2d = device.classes.eng2d;
    dev_info.cls_eng3d = device.classes.eng3d;
    dev_info.cls_m2mf = device.classes.m2mf;
    dev_info.cls_compute = device.classes.compute;
    dev_info.vram_size_B = device.vram_size.get();
    dev_info.bar_size_B = device.bar_size.get();
    // SAFETY: a live, filled nv_device_info; Mesa's helper reads `sm` and
    // writes the shared memory configurations.
    unsafe { nvkmd_dev_info_init_shared_mem_sizes(&raw mut dev_info) };
    // SAFETY: no arguments; Mesa's CPU cache line query.
    let granularity = unsafe { util_cache_granularity() };
    dev_info.nc_atom_size_B = u16::try_from(granularity).map_err(|error| {
        vk_errorf(
            log,
            VK_ERROR_INITIALIZATION_FAILED,
            format_args!(
                "CPU cache granularity {granularity} does not fit nv_device_info: {error}"
            ),
        )
    })?;
    if let Some(zcull) = device.zcull {
        let g = zcull.geometry;
        dev_info.zcull_info = nv_zcull_device_info {
            width_align_pixels: g.width_align_pixels,
            height_align_pixels: g.height_align_pixels,
            pixel_squares_by_aliquots: g.pixel_squares_by_aliquots,
            aliquot_total: g.aliquot_total,
            zcull_region_byte_multiplier: g.region_byte_multiplier,
            zcull_region_header_size: g.region_header_size,
            zcull_subregion_header_size: g.subregion_header_size,
            subregion_count: g.subregion_count,
            subregion_width_align_pixels: g.subregion_width_align_pixels,
            subregion_height_align_pixels: g.subregion_height_align_pixels,
            ctxsw_size: zcull.ctxsw_size,
            ctxsw_align: zcull.ctxsw_align,
        };
        dev_info.has_zcull_info = true;
    }
    let render_dev: dev_t = narrow(st.st_rdev).map_err(|error| refuse(&error))?;
    Ok(nvkmd_pdev {
        ops: &raw const PDEV_OPS,
        debug_flags: debug,
        dev_info,
        kmd_info: nvkmd_info {
            has_dma_buf: kmd.has_dma_buf,
            has_sparse: false,
            has_get_vram_used: false,
            has_alloc_tiled: kmd.has_alloc_tiled,
            has_map_fixed: false,
            has_overmap: false,
            has_compression: false,
            has_rm_compression: kmd.has_rm_compression,
            max_shared_ctx: kmd.max_shared_ctx,
            sm_isa_version: kmd.sm_isa_version,
            shader_pipe_count: kmd.shader_pipe_count,
            shader_pipe_sub_count: kmd.shader_pipe_sub_count,
        },
        bind_align_B: u32::try_from(BIND_ALIGN.get()).expect("the bind alignment is 4 KiB"),
        drm: nvkmd_pdev__bindgen_ty_1 {
            render_dev,
            primary_dev: 0,
        },
        sync_types: core::ptr::null(),
    })
}

/// The physical device's `destroy`: closes the render node and frees it.
///
/// # Safety
/// `base` is a device [`nvkmd_rm_try_create_pdev`] returned, not used again.
unsafe extern "C" fn pdev_destroy(base: *mut nvkmd_pdev) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_pdev.destroy", || {
        // SAFETY: a live nvkmd_pdev (contract).
        unsafe { RmPdev::from_base(base) }.expect("Mesa destroys its own backend's devices");
        // SAFETY: a pdev with our ops is the first field of an `RmPdev` that
        // `probe` leaked with Box::into_raw; Mesa's pointer carries that
        // allocation's provenance, and this is its one destroy.
        drop(unsafe { Box::from_raw(base.cast::<RmPdev>()) });
    });
}

/// Device memory in use: not reported (`has_get_vram_used` is false), 0.
unsafe extern "C" fn pdev_vram_used(_base: *mut nvkmd_pdev) -> u64 {
    nv_ffi::guard_or_abort::<Diag, u64>("nvkmd_pdev.get_vram_used", || 0)
}
