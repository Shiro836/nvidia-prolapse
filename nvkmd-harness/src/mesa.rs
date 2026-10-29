// SPDX-License-Identifier: MIT
//! Handles over Mesa's public nvkmd interface: the DRM device NVK probes, the
//! physical and KMD devices made through `nvkmd_try_create_pdev_for_drm` and
//! the ops tables, the memory and VA reservations made through Mesa's
//! `nvkmd_dev_*` wrappers, and a stand-in logical `vk_device` that holds the
//! loss state and hooks the backend installs. The suites reach the backend
//! only through these, never through its own symbols or structures.
//!
//! Requests Mesa's wrappers would reject with an assertion (a bind outside
//! the memory, a fixed CPU map) go to the backend's ops table directly, as
//! the backend must refuse them on its own.
//!
//! Single-threaded: the suites run on one thread.
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicI32, Ordering};

use nvkmd_sys::{
    _drmDevice, _drmPciBusInfo, _drmPciDeviceInfo, _vk_device_set_lost, DRM_BUS_PCI, DRM_NODE_MAX,
    DRM_NODE_RENDER, NVKMD_SYS_VK_DEVICE_LOST, NVKMD_SYS_VK_DEVICE_QUEUES,
    NVKMD_SYS_VK_OBJECT_BASE_DEVICE, VK_SUCCESS, VkResult, nvkmd_dev, nvkmd_dev_alloc_mem,
    nvkmd_dev_alloc_tiled_mem, nvkmd_dev_alloc_va, nvkmd_dev_import_dma_buf, nvkmd_mem,
    nvkmd_mem_map, nvkmd_mem_ops, nvkmd_mem_unmap, nvkmd_mem_unref, nvkmd_pdev, nvkmd_sys_layout,
    nvkmd_try_create_pdev_for_drm, nvkmd_va, nvkmd_va_free, nvkmd_va_ops, vk_device,
    vk_object_base, vk_sync_type,
};

use crate::fake;
use nv_rm_core::conv::usize_from_u32;

/// PCI vendor id of NVIDIA (`nvkmd_rm.c` `NVIDIA_VENDOR_ID`).
const NVIDIA_VENDOR_ID: u16 = 0x10de;

fn offset(member: nvkmd_sys_layout) -> usize {
    usize_from_u32(member)
}

/// Entries of `drmDevice.nodes`.
const NODES: usize = usize_from_u32(DRM_NODE_MAX);

/// The DRM device NVK hands the backend: a PCI device of NVIDIA at 1:2:3.4
/// with only a render node, [`fake::RENDER_PATH`].
struct DrmFixture {
    nodes: [*mut c_char; NODES],
    pci: _drmPciDeviceInfo,
    bus: _drmPciBusInfo,
    device: _drmDevice,
}

impl DrmFixture {
    #[expect(
        clippy::unnecessary_box_returns,
        reason = "the record points into itself; the box keeps those addresses stable"
    )]
    fn new() -> Box<Self> {
        // SAFETY: all-zero is a valid `_drmDevice` (integers and NULL pointers).
        let device: _drmDevice = unsafe { core::mem::zeroed() };
        let mut fixture = Box::new(Self {
            nodes: [core::ptr::null_mut(); NODES],
            pci: _drmPciDeviceInfo {
                vendor_id: NVIDIA_VENDOR_ID,
                device_id: 0,
                subvendor_id: 0,
                subdevice_id: 0,
                revision_id: 0,
            },
            bus: _drmPciBusInfo {
                domain: 1,
                bus: 2,
                dev: 3,
                func: 4,
            },
            device,
        });
        fixture.nodes[usize_from_u32(DRM_NODE_RENDER)] = fake::RENDER_PATH.as_ptr().cast_mut();
        fixture.device.nodes = fixture.nodes.as_mut_ptr();
        fixture.device.available_nodes = 1 << DRM_NODE_RENDER;
        fixture.device.bustype = i32::try_from(DRM_BUS_PCI).expect("DRM_BUS_PCI is 0");
        fixture.device.deviceinfo.pci = &raw mut fixture.pci;
        fixture.device.businfo.pci = &raw mut fixture.bus;
        fixture
    }
}

/// A stand-in logical device: zeroed, with an empty queue list and its object
/// base naming itself, which is all `vk_device_set_lost` and the backend's
/// loss hooks touch. Heap-allocated so the address the backend keeps stays
/// valid; accessed only through raw pointers because the backend writes it.
pub(crate) struct VkDevice(NonNull<vk_device>);

impl VkDevice {
    pub(crate) fn new() -> Self {
        let device = NonNull::from(Box::leak(Box::new(vk_device {
            _bindgen_opaque_blob: [0; size_of::<vk_device>() / 8],
        })));
        // SAFETY: the queue list head lies inside the allocation (offset from
        // Mesa's header, nvkmd-sys).
        let queues = unsafe { device.as_ptr().byte_add(offset(NVKMD_SYS_VK_DEVICE_QUEUES)) }
            .cast::<[*mut u8; 2]>();
        // SAFETY: `queues` is in bounds and aligned (Mesa's layout); an empty
        // list points to itself (util/list.h list_inithead).
        unsafe { queues.write([queues.cast(), queues.cast()]) };
        // SAFETY: `vk_object_base.device` lies inside the allocation (the
        // base is the first member of vk_device).
        let owner = unsafe {
            device
                .as_ptr()
                .byte_add(offset(NVKMD_SYS_VK_OBJECT_BASE_DEVICE))
        }
        .cast::<*mut vk_device>();
        // SAFETY: in bounds and pointer-aligned (Mesa's layout).
        unsafe { owner.write(device.as_ptr()) };
        Self(device)
    }

    pub(crate) fn as_ptr(&self) -> *mut vk_device {
        self.0.as_ptr()
    }

    /// The device as the log object of a backend call.
    pub(crate) fn log_object(&self) -> *mut vk_object_base {
        self.0.as_ptr().cast()
    }

    /// Mesa's `vk_device_set_lost`.
    pub(crate) fn set_lost(&self, message: &CStr) -> VkResult {
        // SAFETY: the device is live and holds what vk_device_set_lost reads
        // (loss state, empty queue list, hooks the backend installed or NULL);
        // "%s" consumes the one string argument.
        unsafe {
            _vk_device_set_lost(
                self.as_ptr(),
                c"harness".as_ptr(),
                0,
                c"%s".as_ptr(),
                message.as_ptr(),
            )
        }
    }

    /// Mesa's `vk_device_is_lost_no_report`.
    pub(crate) fn is_lost(&self) -> bool {
        is_lost(self.as_ptr())
    }

    /// Clears Mesa's loss count, as a fresh scenario on the same device.
    pub(crate) fn clear_lost(&self) {
        lost_counter(self.as_ptr()).store(0, Ordering::Release);
    }

    /// Whether the pointer member at `member` (one of the loss hooks) is set.
    pub(crate) fn has(&self, member: nvkmd_sys_layout) -> bool {
        !pointer_member_is_null(self.as_ptr(), member)
    }
}

impl Drop for VkDevice {
    fn drop(&mut self) {
        // SAFETY: allocated by Box in `new`; the backend dropped every pointer
        // to it when its KMD device was destroyed (the suites destroy that first).
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

fn lost_counter(device: *mut vk_device) -> &'static AtomicI32 {
    // SAFETY: `_lost.lost` lies inside the live device (Mesa's layout).
    let counter = unsafe { device.byte_add(offset(NVKMD_SYS_VK_DEVICE_LOST)) };
    // SAFETY: an aligned `int` that Mesa only accesses atomically
    // (p_atomic_inc/p_atomic_read), live while the suite holds the device.
    unsafe { AtomicI32::from_ptr(counter.cast()) }
}

/// Mesa's `vk_device_is_lost_no_report` (`vk_device.h`, inline): the loss count
/// is positive. `Acquire` pairs with Mesa's atomic increment.
pub(crate) fn is_lost(device: *mut vk_device) -> bool {
    lost_counter(device).load(Ordering::Acquire) > 0
}

/// Whether the pointer-sized member at `member` of the live `device` is NULL.
pub(crate) fn pointer_member_is_null(device: *mut vk_device, member: nvkmd_sys_layout) -> bool {
    // SAFETY: the member lies inside the live device (Mesa's layout).
    let slot = unsafe { device.byte_add(offset(member)) }.cast::<*const u8>();
    // SAFETY: an aligned pointer-sized member, not written concurrently (the
    // suites are single-threaded).
    unsafe { slot.read() }.is_null()
}

/// A physical device the backend created.
pub(crate) struct Pdev(NonNull<nvkmd_pdev>);

/// NVK's probe (`nvkmd_try_create_pdev_for_drm`) of the fixture's DRM device.
/// On failure the backend must leave the output NULL.
pub(crate) fn try_create_pdev() -> Result<Pdev, VkResult> {
    let mut fixture = DrmFixture::new();
    let mut out = NonNull::<nvkmd_pdev>::dangling().as_ptr();
    // SAFETY: the fixture is a complete DRM device record that outlives the
    // call; `out` is a writable local; no log object (enumeration has none).
    let result = unsafe {
        nvkmd_try_create_pdev_for_drm(
            &raw mut fixture.device,
            core::ptr::null_mut(),
            0,
            &raw mut out,
        )
    };
    if result != VK_SUCCESS {
        assert!(out.is_null(), "a failed probe returns no device");
        return Err(result);
    }
    Ok(Pdev(
        NonNull::new(out).expect("a successful probe returns its device"),
    ))
}

impl Pdev {
    pub(crate) fn get(&self) -> &nvkmd_pdev {
        // SAFETY: a live device the backend created; the backend does not
        // write it after creation.
        unsafe { self.0.as_ref() }
    }

    /// The sync types the device lists (NULL-terminated array).
    pub(crate) fn sync_types(&self) -> Vec<&vk_sync_type> {
        let mut types = Vec::new();
        let list = self.get().sync_types;
        assert!(!list.is_null(), "a physical device lists its sync types");
        for index in 0.. {
            // SAFETY: the list is NULL-terminated; `index` stops at the NULL.
            let slot = unsafe { list.add(index) };
            // SAFETY: `slot` is an entry of the live list (above).
            let entry = unsafe { slot.read() };
            // SAFETY: a non-NULL entry points to a type the device owns.
            match unsafe { entry.as_ref() } {
                Some(entry) => types.push(entry),
                None => break,
            }
        }
        types
    }

    /// The physical device's `create_dev`, with `log` as the log object (a
    /// logical device, or none during probing).
    pub(crate) fn create_dev(&self, log: Option<&VkDevice>) -> Result<Dev, VkResult> {
        let create = self
            .ops()
            .create_dev
            .expect("every physical device creates devices");
        let mut out = NonNull::<nvkmd_dev>::dangling().as_ptr();
        let log = log.map_or(core::ptr::null_mut(), VkDevice::log_object);
        // SAFETY: the backend's own entry on its own live device; `out` is a
        // writable local; `log` is NULL or a live stand-in device.
        let result = unsafe { create(self.0.as_ptr(), log, &raw mut out) };
        if result != VK_SUCCESS {
            assert!(out.is_null(), "a failed create_dev returns no device");
            return Err(result);
        }
        Ok(Dev(
            NonNull::new(out).expect("a successful create_dev returns its device")
        ))
    }

    /// The physical device's `destroy`.
    pub(crate) fn destroy(self) {
        let destroy = self
            .ops()
            .destroy
            .expect("every physical device has destroy");
        // SAFETY: the backend's own entry; the device is not used afterwards
        // (this consumes the handle).
        unsafe { destroy(self.0.as_ptr()) };
    }

    fn ops(&self) -> &nvkmd_sys::nvkmd_pdev_ops {
        // SAFETY: a created device points to the backend's static ops table.
        unsafe { self.get().ops.as_ref() }.expect("a created physical device has ops")
    }
}

/// A KMD device the backend created.
pub(crate) struct Dev(NonNull<nvkmd_dev>);

fn log_of(log: Option<&VkDevice>) -> *mut vk_object_base {
    log.map_or(core::ptr::null_mut(), VkDevice::log_object)
}

/// A call that hands out an object through `out`: the object on success, the
/// result otherwise, when the backend must have left `out` alone or NULL.
fn made<T>(result: VkResult, out: *mut T) -> Result<NonNull<T>, VkResult> {
    if result == VK_SUCCESS {
        return Ok(NonNull::new(out).expect("a successful call returns its object"));
    }
    Err(result)
}

impl Dev {
    /// The device's `destroy`.
    pub(crate) fn destroy(self) {
        // SAFETY: a live device the backend created.
        let ops = unsafe { self.0.as_ref() }.ops;
        // SAFETY: a created device points to the backend's static ops table.
        let ops = unsafe { ops.as_ref() }.expect("a created device has ops");
        let destroy = ops.destroy.expect("every device has destroy");
        // SAFETY: the backend's own entry; the device is not used afterwards.
        unsafe { destroy(self.0.as_ptr()) };
    }

    /// Mesa's `nvkmd_dev_alloc_mem`.
    pub(crate) fn alloc_mem(
        &self,
        log: Option<&VkDevice>,
        size: u64,
        alignment: u64,
        flags: u32,
    ) -> Result<Mem, VkResult> {
        let mut out = core::ptr::null_mut();
        // SAFETY: Mesa's wrapper over the backend's entry on its live device;
        // `out` is a writable local; `log` NULL or a live stand-in device.
        let result = unsafe {
            nvkmd_dev_alloc_mem(
                self.0.as_ptr(),
                log_of(log),
                size,
                alignment,
                flags,
                &raw mut out,
            )
        };
        made(result, out).map(Mem)
    }

    /// Mesa's `nvkmd_dev_alloc_tiled_mem`.
    pub(crate) fn alloc_tiled_mem(
        &self,
        log: Option<&VkDevice>,
        size: u64,
        alignment: u64,
        pte_kind: u8,
        tile_mode: u16,
        flags: u32,
    ) -> Result<Mem, VkResult> {
        let mut out = core::ptr::null_mut();
        // SAFETY: as `alloc_mem`.
        let result = unsafe {
            nvkmd_dev_alloc_tiled_mem(
                self.0.as_ptr(),
                log_of(log),
                size,
                alignment,
                pte_kind,
                tile_mode,
                flags,
                &raw mut out,
            )
        };
        made(result, out).map(Mem)
    }

    /// Mesa's `nvkmd_dev_import_dma_buf`.
    pub(crate) fn import_dma_buf(
        &self,
        log: Option<&VkDevice>,
        fd: c_int,
    ) -> Result<Mem, VkResult> {
        let mut out = core::ptr::null_mut();
        // SAFETY: as `alloc_mem`; the fd is only passed on to the fake RM.
        let result =
            unsafe { nvkmd_dev_import_dma_buf(self.0.as_ptr(), log_of(log), fd, &raw mut out) };
        made(result, out).map(Mem)
    }

    /// Mesa's `nvkmd_dev_alloc_va`.
    pub(crate) fn alloc_va(
        &self,
        log: Option<&VkDevice>,
        flags: u32,
        pte_kind: u8,
        size: u64,
        alignment: u64,
        fixed: u64,
    ) -> Result<Va, VkResult> {
        let mut out = core::ptr::null_mut();
        // SAFETY: as `alloc_mem`.
        let result = unsafe {
            nvkmd_dev_alloc_va(
                self.0.as_ptr(),
                log_of(log),
                flags,
                pte_kind,
                size,
                alignment,
                fixed,
                &raw mut out,
            )
        };
        made(result, out).map(Va)
    }
}

/// Memory the backend allocated or imported; Mesa's reference.
pub(crate) struct Mem(NonNull<nvkmd_mem>);

impl Mem {
    pub(crate) fn get(&self) -> &nvkmd_mem {
        // SAFETY: a live allocation; the suites read it between backend calls.
        unsafe { self.0.as_ref() }
    }

    fn ops(&self) -> &nvkmd_mem_ops {
        // SAFETY: a created allocation points to the backend's static table.
        unsafe { self.get().ops.as_ref() }.expect("a created allocation has ops")
    }

    /// The allocation's own reservation, which the backend made for it.
    pub(crate) fn va(&self) -> Va {
        Va(NonNull::new(self.get().va).expect("every allocation has its VA"))
    }

    /// Mesa's `nvkmd_mem_map` of the internal mapping (reference counted).
    pub(crate) fn map(&self, log: Option<&VkDevice>) -> Result<*mut c_void, VkResult> {
        let mut out = core::ptr::null_mut();
        // SAFETY: Mesa's wrapper on a live allocation; `out` is writable.
        let result = unsafe {
            nvkmd_mem_map(
                self.0.as_ptr(),
                log_of(log),
                nvkmd_sys::NVKMD_MEM_MAP_RDWR,
                core::ptr::null_mut(),
                &raw mut out,
            )
        };
        if result != VK_SUCCESS {
            return Err(result);
        }
        Ok(out)
    }

    /// Mesa's `nvkmd_mem_unmap` of the internal mapping.
    pub(crate) fn unmap(&self) {
        // SAFETY: Mesa's wrapper on a live allocation `map` mapped.
        unsafe { nvkmd_mem_unmap(self.0.as_ptr(), 0) };
    }

    /// The backend's `map` itself, with what it wrote to its output (which
    /// starts as a non-NULL sentinel).
    pub(crate) fn ops_map(
        &self,
        log: Option<&VkDevice>,
        flags: u32,
        fixed: *mut c_void,
    ) -> (VkResult, *mut c_void) {
        let map = self.ops().map.expect("memory maps");
        let mut out = core::ptr::without_provenance_mut(1);
        // SAFETY: the backend's entry on its live allocation; `out` writable.
        let result = unsafe { map(self.0.as_ptr(), log_of(log), flags, fixed, &raw mut out) };
        (result, out)
    }

    /// The backend's `unmap` itself.
    pub(crate) fn ops_unmap(&self, map: *mut c_void) {
        let unmap = self.ops().unmap.expect("memory unmaps");
        // SAFETY: the backend's entry on its live allocation.
        unsafe { unmap(self.0.as_ptr(), 0, map) };
    }

    /// The backend's `export_dma_buf`.
    pub(crate) fn export(&self, log: Option<&VkDevice>) -> Result<c_int, VkResult> {
        let export = self.ops().export_dma_buf.expect("memory exports");
        let mut fd = -1;
        // SAFETY: the backend's entry on its live allocation; `fd` writable.
        let result = unsafe { export(self.0.as_ptr(), log_of(log), &raw mut fd) };
        if result != VK_SUCCESS {
            return Err(result);
        }
        Ok(fd)
    }

    /// The backend's `overmap`.
    pub(crate) fn overmap(&self, log: Option<&VkDevice>) -> VkResult {
        let overmap = self.ops().overmap.expect("the overmap slot is set");
        // SAFETY: the backend's entry on its live allocation.
        unsafe {
            overmap(
                self.0.as_ptr(),
                log_of(log),
                0,
                core::ptr::without_provenance_mut(0x1000),
            )
        }
    }

    /// The backend's `log_handle`.
    pub(crate) fn log_handle(&self) -> u32 {
        let handle = self.ops().log_handle.expect("memory has a log handle");
        // SAFETY: the backend's entry on its live allocation.
        unsafe { handle(self.0.as_ptr()) }
    }

    /// Mesa's `nvkmd_mem_unref` of the last reference: the backend's `free`.
    pub(crate) fn unref(self) {
        // SAFETY: Mesa's wrapper; the suites hold the one reference.
        unsafe { nvkmd_mem_unref(self.0.as_ptr()) };
    }
}

/// A VA reservation the backend made.
pub(crate) struct Va(NonNull<nvkmd_va>);

impl Va {
    pub(crate) fn get(&self) -> &nvkmd_va {
        // SAFETY: a live reservation; the suites read it between backend calls.
        unsafe { self.0.as_ref() }
    }

    fn ops(&self) -> &nvkmd_va_ops {
        // SAFETY: a created reservation points to the backend's static table.
        unsafe { self.get().ops.as_ref() }.expect("a created reservation has ops")
    }

    /// The backend's `bind_mem` (Mesa's `nvkmd_va_bind_mem` asserts the
    /// ranges the backend must check itself).
    pub(crate) fn bind(
        &self,
        log: Option<&VkDevice>,
        offset: u64,
        mem: &Mem,
        mem_offset: u64,
        size: u64,
    ) -> VkResult {
        let bind = self.ops().bind_mem.expect("reservations bind");
        // SAFETY: the backend's entry on its live reservation and allocation.
        unsafe {
            bind(
                self.0.as_ptr(),
                log_of(log),
                offset,
                mem.0.as_ptr(),
                mem_offset,
                size,
            )
        }
    }

    /// The backend's `unbind`.
    pub(crate) fn unbind(&self, log: Option<&VkDevice>, offset: u64, size: u64) -> VkResult {
        let unbind = self.ops().unbind.expect("reservations unbind");
        // SAFETY: the backend's entry on its live reservation.
        unsafe { unbind(self.0.as_ptr(), log_of(log), offset, size) }
    }

    /// Mesa's `nvkmd_va_free`.
    pub(crate) fn free(self) {
        // SAFETY: Mesa's wrapper; the reservation is not used afterwards.
        unsafe { nvkmd_va_free(self.0.as_ptr()) };
    }
}
