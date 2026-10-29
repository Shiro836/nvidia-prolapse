// SPDX-License-Identifier: MIT
//! Memory (`nvkmd_mem`): the device's `alloc_mem`, `alloc_tiled_mem` and
//! `import_dma_buf` entries and the memory ops table (`nvkmd_rm.c`
//! `rm_mem_alloc_kind` … `rm_mem_handle`).
//!
//! Every allocation gets its own VA reservation of its whole size, bound at
//! creation, through Mesa's `nvkmd_dev_alloc_va` and `nvkmd_va_bind_mem` (so
//! NVK's VM logging sees them as for any backend). Every CPU map is coherent
//! (system memory CPU-cached and snooped, device memory write-combined
//! through BAR1), so allocations carry `NVKMD_MEM_COHERENT` and Mesa never
//! asks for cache maintenance.
//!
//! Mesa owns the lifetime: its reference count reaches `free`, and it
//! serialises map and unmap per allocation (`nvkmd.c` `map_mutex`).
use std::ffi::{c_int, c_void};

use nv_rm_abi::{
    NV_RM_PLACEMENT_GART, NV_RM_PLACEMENT_VRAM, NV_RM_PLACEMENT_VRAM_COMPRESSIBLE,
    NV_RM_PLACEMENT_VRAM_POOL_BLOCK, NvRmAllocation, NvRmMemoryId, nv_rm_alloc, nv_rm_export,
    nv_rm_free, nv_rm_import, nv_rm_is_lost, nv_rm_map, nv_rm_unmap,
};
use nv_rm_core::ByteSize;
use nvkmd_rm_logic::memory::{self, MemError, MemFlags, Placement, PteKind, Tiled, TiledError};
use nvkmd_sys::{
    NVKMD_MEM_CAN_MAP, NVKMD_MEM_COHERENT, NVKMD_MEM_LOCAL, NVKMD_MEM_MAP_FIXED, NVKMD_MEM_SHARED,
    VK_ERROR_DEVICE_LOST, VK_ERROR_MEMORY_MAP_FAILED, VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_UNKNOWN, VK_SUCCESS, VkResult, nvkmd_dev,
    nvkmd_dev_alloc_va, nvkmd_mem, nvkmd_mem_finish, nvkmd_mem_flags, nvkmd_mem_init,
    nvkmd_mem_map_flags, nvkmd_mem_ops, nvkmd_va_bind_mem, nvkmd_va_free, vk_object_base,
};

use super::dev::RmDev;
use super::{Diag, diag, rm_result, try_alloc_zeroed, unsupported, vk_error, vk_errorf};

/// An allocation. Mesa holds `&base`; [`mem_free`] frees it.
#[repr(C)]
pub(super) struct RmMem {
    base: nvkmd_mem,
    allocation: NvRmAllocation,
}

static MEM_OPS: nvkmd_mem_ops = nvkmd_mem_ops {
    free: Some(mem_free),
    map: Some(mem_map),
    unmap: Some(mem_unmap),
    overmap: Some(mem_overmap),
    sync_to_gpu: Some(mem_sync),
    sync_from_gpu: Some(mem_sync),
    export_dma_buf: Some(mem_export),
    log_handle: Some(mem_log_handle),
};

/// The page size Mesa binds allocations of this backend in (`bind_align_B`).
const BIND_ALIGN: u32 = 4096;

impl RmMem {
    /// Our allocation behind Mesa's pointer, or `None` for NULL or another
    /// backend's.
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_mem`.
    pub(super) unsafe fn from_base<'a>(base: *mut nvkmd_mem) -> Option<&'a Self> {
        // SAFETY: NULL or a live nvkmd_mem (caller).
        let mem = unsafe { base.as_ref() }?;
        if !core::ptr::eq(mem.ops, &raw const MEM_OPS) {
            return None;
        }
        // SAFETY: a mem with our ops is the first field of a live `RmMem`.
        Some(unsafe { &*base.cast::<Self>() })
    }

    /// Takes back the allocation behind Mesa's pointer, or `None` for NULL or
    /// another backend's.
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_mem` that is not used again.
    unsafe fn owned(base: *mut nvkmd_mem) -> Option<Box<Self>> {
        // SAFETY: NULL or a live nvkmd_mem (caller).
        unsafe { Self::from_base(base) }?;
        // SAFETY: a mem with our ops is the first field of an `RmMem` that
        // `finish` leaked with Box::into_raw; the caller gives it up.
        Some(unsafe { Box::from_raw(base.cast::<Self>()) })
    }

    /// The allocation's RM handle.
    pub(super) fn id(&self) -> NvRmMemoryId {
        self.allocation.id
    }

    /// The allocation's size as Mesa knows it.
    pub(super) fn size(&self) -> ByteSize {
        ByteSize::new(self.base.size_B)
    }

    fn dev(&self) -> &RmDev {
        // SAFETY: every allocation belongs to a live device (Mesa frees
        // allocations before their device).
        unsafe { RmDev::from_base(self.base.dev) }.expect("an allocation's device is ours")
    }
}

/// What a new allocation is: an RM allocation, or a dma-buf import.
#[derive(Clone, Copy)]
enum Source {
    Alloc {
        flags: MemFlags,
        alignment: ByteSize,
        kind: PteKind,
    },
    Import,
}

impl Source {
    /// The rollback step named in the loss a failed rollback publishes.
    fn rollback(self) -> &'static str {
        match self {
            Self::Alloc { .. } => "allocation rollback free",
            Self::Import => "import rollback free",
        }
    }
}

/// The device's `alloc_mem`: plain (kind 0) memory.
///
/// # Safety
/// `dev` is a live device; `log` NULL or a live Vulkan object; `out` writable
/// for one pointer.
pub(super) unsafe extern "C" fn dev_alloc_mem(
    dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    size: u64,
    alignment: u64,
    flags: nvkmd_mem_flags,
    out: *mut *mut nvkmd_mem,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_dev.alloc_mem",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            // SAFETY: as this function's contract.
            unsafe {
                create(dev, log, out, |dev| {
                    alloc(dev, log, size, alignment, flags, PteKind::PITCH)
                })
            }
        },
    )
}

/// The device's `alloc_tiled_mem`: memory of a DRM modifier, served for NIL's
/// uncompressed 2D modifier on the device's generic kind only.
///
/// # Safety
/// As [`dev_alloc_mem`].
pub(super) unsafe extern "C" fn dev_alloc_tiled_mem(
    dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    size: u64,
    alignment: u64,
    pte_kind: u8,
    tile_mode: u16,
    flags: nvkmd_mem_flags,
    out: *mut *mut nvkmd_mem,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_dev.alloc_tiled_mem",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            let tiled = |dev: &RmDev| {
                let kind = PteKind::new(pte_kind);
                let mem_flags = MemFlags::from_bits(flags);
                match memory::tiled(kind, tile_mode, mem_flags, dev.generic_kind()) {
                    Ok(Tiled::Plain) => alloc(dev, log, size, alignment, flags, PteKind::PITCH),
                    Ok(Tiled::Kind(kind)) => alloc(dev, log, size, alignment, flags, kind),
                    Err(error @ TiledError::Unsupported) => {
                        Err(unsupported(log, &error.to_string()))
                    }
                    Err(error @ TiledError::MappableBlockLinearBeforeTuring) => {
                        Err(vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("{error}")))
                    }
                }
            };
            // SAFETY: as this function's contract.
            unsafe { create(dev, log, out, tiled) }
        },
    )
}

/// The device's `import_dma_buf`: the dma-buf `fd` as an allocation. The fd
/// is borrowed; NVK closes it after the import succeeded.
///
/// # Safety
/// As [`dev_alloc_mem`]; `fd` is open for the call.
pub(super) unsafe extern "C" fn dev_import_dma_buf(
    dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    fd: c_int,
    out: *mut *mut nvkmd_mem,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_dev.import_dma_buf",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            // SAFETY: as this function's contract.
            unsafe { create(dev, log, out, |dev| import(dev, log, fd)) }
        },
    )
}

/// Recovers the device, runs `make` and writes the allocation it made to
/// `out`.
///
/// # Safety
/// `dev` is NULL or a live device; `out` NULL or writable for one pointer.
unsafe fn create(
    dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    out: *mut *mut nvkmd_mem,
    make: impl FnOnce(&RmDev) -> Result<*mut nvkmd_mem, VkResult>,
) -> VkResult {
    // SAFETY: NULL or a live nvkmd_dev (caller).
    let Some(dev) = (unsafe { RmDev::from_base(dev) }) else {
        return vk_errorf(
            log,
            VK_ERROR_UNKNOWN,
            format_args!("memory of a foreign device"),
        );
    };
    if out.is_null() {
        return vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("memory: NULL output"));
    }
    match make(dev) {
        Ok(mem) => {
            // SAFETY: `out` is writable (caller, checked non-NULL).
            unsafe { out.write(mem) };
            VK_SUCCESS
        }
        Err(result) => result,
    }
}

/// An RM allocation of `kind` (`nvkmd_rm.c` `rm_mem_alloc_kind`): placement,
/// size and alignment from the policy, then the RM allocation and its VA.
fn alloc(
    dev: &RmDev,
    log: *mut vk_object_base,
    size: u64,
    alignment: u64,
    flags: nvkmd_mem_flags,
    kind: PteKind,
) -> Result<*mut nvkmd_mem, VkResult> {
    let Some(storage) = try_alloc_zeroed::<RmMem>() else {
        return Err(vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY));
    };
    let mem_flags = MemFlags::from_bits(flags);
    let placed = match memory::place(
        ByteSize::new(size),
        ByteSize::new(alignment),
        mem_flags,
        kind,
        dev.generic_kind(),
    ) {
        Ok(placed) => placed,
        Err(error) => {
            drop(storage);
            let result = match error {
                MemError::PoolBlock { .. } => VK_ERROR_UNKNOWN,
                MemError::SizeOverflow { .. } => VK_ERROR_OUT_OF_DEVICE_MEMORY,
            };
            return Err(vk_errorf(log, result, format_args!("{error}")));
        }
    };
    let placement = match placed.placement {
        Placement::Vram => NV_RM_PLACEMENT_VRAM,
        Placement::Gart => NV_RM_PLACEMENT_GART,
        Placement::VramCompressible => NV_RM_PLACEMENT_VRAM_COMPRESSIBLE,
        Placement::VramPoolBlock => NV_RM_PLACEMENT_VRAM_POOL_BLOCK,
    };
    let mut allocation = NvRmAllocation::default();
    // SAFETY: the device's live RM device; `allocation` is a writable local.
    // The RM records a non-zero kind on the physical allocation, so an
    // importer of the memory maps it with that kind (as nouveau's
    // `tile_flags`, `nouveau_bo.c`).
    let status = unsafe {
        nv_rm_alloc(
            dev.rm(),
            placed.size.get(),
            placed.alignment.get(),
            placement,
            u32::from(kind.get()),
            0,
            &raw mut allocation,
        )
    };
    let result = rm_result(log, status, "allocate memory");
    if result != VK_SUCCESS {
        drop(storage);
        return Err(result);
    }
    let source = Source::Alloc {
        flags: mem_flags.union(MemFlags::COHERENT),
        alignment: placed.alignment,
        kind,
    };
    finish(dev, log, storage, allocation, source)
}

/// A dma-buf import (`nvkmd_rm.c` `rm_mem_import`).
fn import(dev: &RmDev, log: *mut vk_object_base, fd: c_int) -> Result<*mut nvkmd_mem, VkResult> {
    let Some(storage) = try_alloc_zeroed::<RmMem>() else {
        return Err(vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY));
    };
    let mut allocation = NvRmAllocation::default();
    // SAFETY: the device's live RM device and render node; `fd` is open for
    // the call (contract) and only borrowed; `allocation` is a writable local.
    let status = unsafe { nv_rm_import(dev.rm(), dev.drm_fd(), fd, &raw mut allocation) };
    let result = rm_result(log, status, "import DMA-buf");
    if result != VK_SUCCESS {
        drop(storage);
        return Err(result);
    }
    finish(dev, log, storage, allocation, Source::Import)
}

/// Makes Mesa's memory object for an RM allocation and gives it a VA of its
/// whole size, bound at offset 0. On failure the RM allocation is freed; if
/// that fails too, the device is lost and the allocation keeps its own
/// failure (`vkAllocateMemory` has no `VK_ERROR_DEVICE_LOST`).
fn finish(
    dev: &RmDev,
    log: *mut vk_object_base,
    storage: Box<core::mem::MaybeUninit<RmMem>>,
    allocation: NvRmAllocation,
    source: Source,
) -> Result<*mut nvkmd_mem, VkResult> {
    // SAFETY: all-zero is a valid `nvkmd_mem` until `nvkmd_mem_init` fills it.
    let base: nvkmd_mem = unsafe { core::mem::zeroed() };
    let mut mem = Box::write(storage, RmMem { base, allocation });
    let (flags, alignment, kind) = match source {
        Source::Alloc {
            flags,
            alignment,
            kind,
        } => (flags.bits(), alignment, kind),
        Source::Import => (
            NVKMD_MEM_LOCAL | NVKMD_MEM_SHARED | NVKMD_MEM_CAN_MAP | NVKMD_MEM_COHERENT,
            ByteSize::new(u64::from(BIND_ALIGN)),
            PteKind::PITCH,
        ),
    };
    let size = allocation.size;
    // SAFETY: the memory object in its final place; the ops are ours.
    unsafe {
        nvkmd_mem_init(
            dev.base_ptr(),
            &raw mut mem.base,
            &raw const MEM_OPS,
            flags,
            size,
            BIND_ALIGN,
        );
    }
    // SAFETY: Mesa's VA entry on our live device; `mem.base.va` is writable.
    let mut result = unsafe {
        nvkmd_dev_alloc_va(
            dev.base_ptr(),
            log,
            0,
            kind.get(),
            size,
            alignment.get(),
            0,
            &raw mut mem.base.va,
        )
    };
    if result == VK_SUCCESS {
        // SAFETY: the reservation just made, of `size` bytes, and this memory
        // of `size` bytes.
        result = unsafe { nvkmd_va_bind_mem(mem.base.va, log, 0, &raw mut mem.base, 0, size) };
        if result == VK_SUCCESS {
            return Ok(Box::into_raw(mem).cast());
        }
        // SAFETY: the reservation made above, freed once.
        unsafe { nvkmd_va_free(mem.base.va) };
    }
    // SAFETY: the device's live RM device; the allocation made above, freed
    // once.
    let status = unsafe { nv_rm_free(dev.rm(), allocation.id) };
    if status != VK_SUCCESS {
        dev.publish_loss(source.rollback(), status);
    }
    // SAFETY: the mutex `nvkmd_mem_init` initialised.
    unsafe { nvkmd_mem_finish(&raw mut mem.base) };
    drop(mem);
    Err(result)
}

/// The allocation's `free` (Mesa's last reference): its VA, then the RM
/// allocation. A failed RM free loses the device.
///
/// # Safety
/// `base` is an allocation of this backend, not used again.
unsafe extern "C" fn mem_free(base: *mut nvkmd_mem) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_mem.free", || {
        // SAFETY: a live allocation, not used again (contract).
        let mut mem = unsafe { RmMem::owned(base) }.expect("Mesa frees its own backend's memory");
        // SAFETY: the allocation's own reservation, freed once.
        unsafe { nvkmd_va_free(mem.base.va) };
        let dev = mem.dev();
        // SAFETY: the device's live RM device; the allocation, freed once.
        let status = unsafe { nv_rm_free(dev.rm(), mem.allocation.id) };
        if status != VK_SUCCESS {
            dev.publish_loss("memory free", status);
        }
        // SAFETY: the mutex `nvkmd_mem_init` initialised.
        unsafe { nvkmd_mem_finish(&raw mut mem.base) };
        drop(mem);
    });
}

/// The allocation's `map`: the RM's shared lazy CPU mapping. A loss is
/// published and reported as a failed map (`vkMapMemory` has no
/// `VK_ERROR_DEVICE_LOST`); out-of-memory statuses pass through.
///
/// # Safety
/// `base` is a live allocation of this backend; `log` NULL or a live Vulkan
/// object; `out` writable for one pointer.
unsafe extern "C" fn mem_map(
    base: *mut nvkmd_mem,
    log: *mut vk_object_base,
    flags: nvkmd_mem_map_flags,
    fixed: *mut c_void,
    out: *mut *mut c_void,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_mem.map",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            if out.is_null() {
                return vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("map: NULL output"));
            }
            // SAFETY: `out` is writable (contract, checked non-NULL).
            unsafe { out.write(core::ptr::null_mut()) };
            if !fixed.is_null() || flags & NVKMD_MEM_MAP_FIXED != 0 {
                return unsupported(log, "fixed CPU mapping");
            }
            // SAFETY: a live allocation (contract).
            let Some(mem) = (unsafe { RmMem::from_base(base) }) else {
                return vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("map of foreign memory"));
            };
            match map(mem, log) {
                Ok(address) => {
                    // SAFETY: `out` is writable (above).
                    unsafe { out.write(address) };
                    VK_SUCCESS
                }
                Err(result) => result,
            }
        },
    )
}

/// `map` after validation (`nvkmd_rm.c` `rm_mem_map`).
fn map(mem: &RmMem, log: *mut vk_object_base) -> Result<*mut c_void, VkResult> {
    let dev = mem.dev();
    let mut address = 0_usize;
    // SAFETY: the device's live RM device; `address` is a writable local.
    let status = unsafe { nv_rm_map(dev.rm(), mem.allocation.id, &raw mut address) };
    if status != VK_SUCCESS {
        // A failed rollback inside the RM can lose the device while returning
        // the original status; the loss flag tells them apart without
        // another RM call.
        // SAFETY: the device's live RM device.
        if status == VK_ERROR_DEVICE_LOST || unsafe { nv_rm_is_lost(dev.rm()) } != 0 {
            dev.publish_loss("CPU map", status);
            return Err(vk_errorf(
                log,
                VK_ERROR_MEMORY_MAP_FAILED,
                format_args!("Native Rust RM CPU map failed: device lost"),
            ));
        }
        let result =
            if status == VK_ERROR_OUT_OF_HOST_MEMORY || status == VK_ERROR_OUT_OF_DEVICE_MEMORY {
                status
            } else {
                VK_ERROR_MEMORY_MAP_FAILED
            };
        return Err(vk_errorf(
            log,
            result,
            format_args!("Native Rust RM CPU map failed: {status}"),
        ));
    }
    if address == 0 {
        // The successful map took a map reference even without an address:
        // drop it here.
        // SAFETY: the device's live RM device; the reference just taken.
        let status = unsafe { nv_rm_unmap(dev.rm(), mem.allocation.id) };
        if status != VK_SUCCESS {
            dev.publish_loss("null CPU map rollback", status);
        }
        return Err(vk_errorf(
            log,
            VK_ERROR_MEMORY_MAP_FAILED,
            format_args!("Native Rust RM CPU map returned a null address"),
        ));
    }
    // The RM's mapping of the allocation: an address in this process, which
    // the RM keeps mapped until the last unmap.
    Ok(core::ptr::with_exposed_provenance_mut(address))
}

/// The allocation's `unmap`: drops one RM map reference. A failure loses the
/// device (the caller's reference is already gone).
///
/// # Safety
/// `base` is a live allocation of this backend that `map` mapped.
unsafe extern "C" fn mem_unmap(
    base: *mut nvkmd_mem,
    _flags: nvkmd_mem_map_flags,
    _map: *mut c_void,
) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_mem.unmap", || {
        // SAFETY: a live allocation (contract).
        let mem = unsafe { RmMem::from_base(base) }.expect("Mesa unmaps its own backend's memory");
        let dev = mem.dev();
        // SAFETY: the device's live RM device; a reference `map` took.
        let status = unsafe { nv_rm_unmap(dev.rm(), mem.allocation.id) };
        if status != VK_SUCCESS {
            dev.publish_loss("CPU unmap", status);
        }
    });
}

/// The allocation's `overmap`: not served (`has_overmap` is false).
unsafe extern "C" fn mem_overmap(
    _base: *mut nvkmd_mem,
    log: *mut vk_object_base,
    _flags: nvkmd_mem_map_flags,
    _map: *mut c_void,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_mem.overmap",
        None,
        || VK_ERROR_DEVICE_LOST,
        || unsupported(log, "CPU overmap"),
    )
}

/// The allocation's `sync_to_gpu` / `sync_from_gpu`. Mesa calls them only for
/// memory without `NVKMD_MEM_COHERENT`, and every allocation of this backend
/// has it: reaching one is a broken contract, and the interface has no error
/// return.
unsafe extern "C" fn mem_sync(base: *mut nvkmd_mem, offset: u64, range: u64) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_mem.sync", || {
        // SAFETY: a live allocation (contract).
        let flags = unsafe { base.as_ref() }.map(|mem| mem.flags);
        diag(format_args!(
            "nvkmd_rm: cache sync requested on coherent memory ({offset:x}+{range:x}, flags {flags:#x?})"
        ));
        std::process::abort();
    });
}

/// The allocation's `export_dma_buf`: a dma-buf fd owned by the caller.
///
/// # Safety
/// `base` is a live allocation of this backend; `log` NULL or a live Vulkan
/// object; `fd` writable.
unsafe extern "C" fn mem_export(
    base: *mut nvkmd_mem,
    log: *mut vk_object_base,
    fd: *mut c_int,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_mem.export_dma_buf",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            // SAFETY: a live allocation (contract).
            let Some(mem) = (unsafe { RmMem::from_base(base) }) else {
                return vk_errorf(
                    log,
                    VK_ERROR_UNKNOWN,
                    format_args!("export of foreign memory"),
                );
            };
            if fd.is_null() {
                return vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("export: NULL output"));
            }
            let dev = mem.dev();
            // SAFETY: the device's live RM device and render node; `fd` is
            // writable (contract, checked non-NULL).
            let status = unsafe { nv_rm_export(dev.rm(), mem.allocation.id, dev.drm_fd(), fd) };
            rm_result(log, status, "export DMA-buf")
        },
    )
}

/// The allocation's `log_handle` (NVK's VM debug log): the RM handle's low 32
/// bits.
unsafe extern "C" fn mem_log_handle(base: *mut nvkmd_mem) -> u32 {
    nv_ffi::guard_or_abort::<Diag, u32>("nvkmd_mem.log_handle", || {
        // SAFETY: a live allocation (contract).
        let mem = unsafe { RmMem::from_base(base) }.expect("Mesa logs its own backend's memory");
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "the log shows the handle's low 32 bits, as Mesa's 32-bit handle field holds"
        )]
        let handle = mem.allocation.id.get() as u32;
        handle
    })
}
