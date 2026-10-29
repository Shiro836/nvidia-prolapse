// SPDX-License-Identifier: MIT
//! GPU virtual address reservations (`nvkmd_va`): the device's two VA heaps,
//! the `alloc_va` entry, the VA ops table (`free`, `bind_mem`, `unbind`) and
//! the release callback through which the RM returns a freed range
//! (`nvkmd_rm.c` `rm_va_alloc` … `rm_va_free`).
//!
//! The heaps are Mesa's `util_vma_heap`, called through its C interface, so
//! the addresses handed out are those of the C backend for the same sequence
//! of requests. Both sit behind one mutex, which is held only around a heap
//! call and never across an RM call; the RM may call [`va_release`] while it
//! holds its own resource lock, so the lock order is RM resource lock → heap
//! lock (`nv_rm.h` `NvRmVaRelease`).
//!
//! A reservation's bindings are kept in bind order, as two parallel lists
//! (ranges for the policy, mapping ids handed to the RM as one array when the
//! reservation is freed). Mesa serialises calls on one reservation.
use std::ffi::c_void;
use std::sync::Mutex;

use nv_rm_abi::{NvRmMappingId, nv_rm_bind, nv_rm_unbind, nv_rm_va_free};
use nv_rm_core::{ByteSize, GpuVa};
use nvkmd_rm_logic::va::{self, Heap, Range, Reservation, Target, VaError, VaFlags};
use nvkmd_sys::{
    VK_ERROR_DEVICE_LOST, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS, VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_UNKNOWN, VK_SUCCESS, VkResult, nvkmd_dev, nvkmd_mem,
    nvkmd_va, nvkmd_va_flags, nvkmd_va_ops, util_vma_heap, util_vma_heap_alloc,
    util_vma_heap_alloc_addr, util_vma_heap_finish, util_vma_heap_free, util_vma_heap_init,
    vk_object_base,
};

use super::dev::RmDev;
use super::memory::RmMem;
use super::object::{self, Object};
use super::{Diag, rm_result, try_alloc_zeroed, unsupported, vk_error, vk_errorf};

/// The window's two heaps (`nvkmd_rm.c` `heap`, `replay_heap`). Each points
/// into itself once initialised: the value is built in its final place and
/// never moved.
pub(super) struct VaHeaps {
    ordinary: util_vma_heap,
    replay: util_vma_heap,
}

impl VaHeaps {
    /// Heaps before [`VaHeaps::init`].
    pub(super) fn zeroed() -> Self {
        // SAFETY: all-zero is a valid `util_vma_heap` (a list head of NULL
        // pointers and integers) until `util_vma_heap_init` fills it.
        let heap = || unsafe { core::mem::zeroed::<util_vma_heap>() };
        Self {
            ordinary: heap(),
            replay: heap(),
        }
    }

    fn heap(&mut self, heap: Heap) -> &mut util_vma_heap {
        match heap {
            Heap::Ordinary => &mut self.ordinary,
            Heap::Replay => &mut self.replay,
        }
    }

    /// Initialises both heaps over their ranges of the window, in place.
    pub(super) fn init(&mut self) {
        for heap in [Heap::Ordinary, Heap::Replay] {
            let (start, size) = heap.range();
            // SAFETY: the heap in its final place (the device's storage).
            unsafe { util_vma_heap_init(self.heap(heap), start.get(), size.get()) };
        }
    }

    /// Releases both heaps' bookkeeping; nothing allocates from them after.
    pub(super) fn finish(&mut self) {
        for heap in [Heap::Ordinary, Heap::Replay] {
            // SAFETY: a heap `init` initialised, finished once.
            unsafe { util_vma_heap_finish(self.heap(heap)) };
        }
    }

    /// A heap-chosen range, or `None` when the heap has no room.
    fn alloc(&mut self, heap: Heap, size: ByteSize, alignment: ByteSize) -> Option<GpuVa> {
        // SAFETY: an initialised heap; size and alignment are non-zero powers
        // of the page (Reservation::new).
        let address = unsafe { util_vma_heap_alloc(self.heap(heap), size.get(), alignment.get()) };
        (address != 0).then_some(GpuVa::new(address))
    }

    /// The range at `address`, or `false` when part of it is taken.
    fn alloc_at(&mut self, heap: Heap, address: GpuVa, size: ByteSize) -> bool {
        // SAFETY: an initialised heap; the range lies inside it
        // (Reservation::fixed_fits).
        unsafe { util_vma_heap_alloc_addr(self.heap(heap), address.get(), size.get()) }
    }

    /// Returns a range to the heap that holds its address.
    fn free(&mut self, address: GpuVa, size: ByteSize) {
        // SAFETY: an initialised heap; the range was allocated from it (the
        // RM returns exactly the ranges `va_free` handed it).
        unsafe { util_vma_heap_free(self.heap(Heap::of(address)), address.get(), size.get()) };
    }
}

/// A reservation. Mesa holds `&base`; [`va_free`] frees it.
#[repr(C)]
pub(super) struct RmVa {
    base: nvkmd_va,
    /// Ranges of the reservation that are bound, in bind order.
    ranges: Vec<Range>,
    /// The RM mapping of each entry of `ranges`.
    mappings: Vec<NvRmMappingId>,
}

static VA_OPS: nvkmd_va_ops = nvkmd_va_ops {
    free: Some(va_free),
    bind_mem: Some(va_bind_mem),
    unbind: Some(va_unbind),
};

// SAFETY: `#[repr(C)]` with its `nvkmd_va` first; only this module hands out
// bases with `VA_OPS`, each inside a boxed `RmVa`.
unsafe impl Object for RmVa {
    type Base = nvkmd_va;
    type Ops = nvkmd_va_ops;
    fn ops(base: &nvkmd_va) -> *const nvkmd_va_ops {
        base.ops
    }
    fn ours() -> *const nvkmd_va_ops {
        &raw const VA_OPS
    }
}

impl RmVa {
    /// Our reservation behind Mesa's pointer, or `None` for NULL or another
    /// backend's.
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_va` that no other call uses
    /// during the borrow (Mesa serialises calls per object).
    unsafe fn from_base<'a>(base: *mut nvkmd_va) -> Option<&'a mut Self> {
        // SAFETY: NULL or a live nvkmd_va used by this call alone (caller).
        unsafe { object::recover_mut(base) }
    }

    /// Takes back the reservation behind Mesa's pointer, or `None` for NULL
    /// or another backend's.
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_va` that is not used again.
    unsafe fn owned(base: *mut nvkmd_va) -> Option<Box<Self>> {
        // SAFETY: NULL or a live nvkmd_va, not used again (caller).
        unsafe { object::take(base) }
    }

    fn dev(&self) -> &RmDev {
        // SAFETY: every reservation belongs to a live device (Mesa frees
        // reservations before their device).
        unsafe { RmDev::from_base(self.base.dev) }.expect("a reservation's device is ours")
    }

    fn size(&self) -> ByteSize {
        ByteSize::new(self.base.size_B)
    }

    /// Unmaps the bindings an unbind of `range` covers, in bind order
    /// (`nvkmd_rm.c` `rm_va_unbind`). On an RM failure the bindings unmapped
    /// before it stay removed and the failed one stays bound.
    fn unbind(&mut self, log: *mut vk_object_base, range: Range) -> VkResult {
        let unbind = match va::unbind(range, self.size(), &self.ranges) {
            Ok(unbind) => unbind,
            Err(error) => return refused(log, error),
        };
        let rm = self.dev().rm();
        let mut index = 0;
        while let Some(binding) = self.ranges.get(index).copied() {
            if !unbind.removes(binding) {
                index += 1;
                continue;
            }
            let mapping = self.mappings[index];
            // SAFETY: the device's live RM device; a mapping its bind returned.
            let status = unsafe { nv_rm_unbind(rm, mapping) };
            let result = rm_result(log, status, "unbind VA");
            if result != VK_SUCCESS {
                return result;
            }
            self.ranges.remove(index);
            self.mappings.remove(index);
        }
        VK_SUCCESS
    }

    /// Binds `[memory_offset, memory_offset + size)` of `memory` at `offset`,
    /// replacing what that range held (`nvkmd_rm.c` `rm_va_bind`).
    fn bind(
        &mut self,
        log: *mut vk_object_base,
        offset: u64,
        memory: &RmMem,
        memory_offset: u64,
        size: ByteSize,
    ) -> VkResult {
        if let Err(error) = va::check_bind(Range::new(memory_offset, size), memory.size()) {
            return refused(log, error);
        }
        let range = Range::new(offset, size);
        let result = self.unbind(log, range);
        if result != VK_SUCCESS {
            return result;
        }
        if self.ranges.try_reserve(1).is_err() || self.mappings.try_reserve(1).is_err() {
            return vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY);
        }
        let address = self.base.addr + offset;
        let mut mapping = NvRmMappingId::default();
        // SAFETY: the device's live RM device and an allocation of it;
        // `mapping` is a writable local.
        let status = unsafe {
            nv_rm_bind(
                self.dev().rm(),
                memory.id(),
                memory_offset,
                size.get(),
                address,
                u32::from(self.base.pte_kind),
                &raw mut mapping,
            )
        };
        let result = rm_result(log, status, "bind VA");
        if result != VK_SUCCESS {
            return result;
        }
        self.ranges.push(range);
        self.mappings.push(mapping);
        VK_SUCCESS
    }
}

/// A refused VA request: `VK_ERROR_OUT_OF_DEVICE_MEMORY` for a size the window
/// cannot hold, `VK_ERROR_FEATURE_NOT_PRESENT` naming the rest.
#[track_caller]
fn refused(log: *mut vk_object_base, error: VaError) -> VkResult {
    match error {
        VaError::Size { .. } => vk_error(log, VK_ERROR_OUT_OF_DEVICE_MEMORY),
        VaError::Sparse
        | VaError::Alignment
        | VaError::FixedNotReplay
        | VaError::BindOutsideMemory
        | VaError::UnbindOutsideReservation
        | VaError::PartialUnbind => unsupported(log, &error.to_string()),
    }
}

/// The device's `alloc_va`: a reservation of `size` bytes from the heap its
/// flags select, or at `fixed` for a replayed capture address.
///
/// # Safety
/// `dev` is a live device; `log` NULL or a live Vulkan object; `out` writable
/// for one pointer.
pub(super) unsafe extern "C" fn dev_alloc_va(
    dev: *mut nvkmd_dev,
    log: *mut vk_object_base,
    flags: nvkmd_va_flags,
    pte_kind: u8,
    size: u64,
    alignment: u64,
    fixed: u64,
    out: *mut *mut nvkmd_va,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_dev.alloc_va",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            // SAFETY: a live nvkmd_dev (contract).
            let Some(dev) = (unsafe { RmDev::from_base(dev) }) else {
                return vk_errorf(
                    log,
                    VK_ERROR_UNKNOWN,
                    format_args!("VA of a foreign device"),
                );
            };
            if out.is_null() {
                return vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("alloc_va: NULL output"));
            }
            match alloc_va(
                dev,
                log,
                VaFlags::from_bits(flags),
                pte_kind,
                size,
                alignment,
                fixed,
            ) {
                Ok(va) => {
                    // SAFETY: `out` is writable (contract, checked non-NULL).
                    unsafe { out.write(va) };
                    VK_SUCCESS
                }
                Err(result) => result,
            }
        },
    )
}

/// `alloc_va` after validation; the order of its checks, its allocation and
/// its results are `nvkmd_rm.c` `rm_va_alloc`.
fn alloc_va(
    dev: &RmDev,
    log: *mut vk_object_base,
    flags: VaFlags,
    pte_kind: u8,
    size: u64,
    alignment: u64,
    fixed: u64,
) -> Result<*mut nvkmd_va, VkResult> {
    let reservation = Reservation::new(ByteSize::new(size), ByteSize::new(alignment), flags)
        .map_err(|error| refused(log, error))?;
    let Some(storage) = try_alloc_zeroed::<RmVa>() else {
        return Err(vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY));
    };
    let fixed = GpuVa::new(fixed);
    let target = match Reservation::target(flags, fixed) {
        Ok(target) => target,
        Err(error) => {
            drop(storage);
            return Err(refused(log, error));
        }
    };
    let address = {
        let mut heaps = dev.heaps();
        match target {
            Target::Heap(heap) => heaps.alloc(heap, reservation.size, reservation.alignment),
            Target::Fixed(address) => (reservation.fixed_fits(address)
                && heaps.alloc_at(Heap::Replay, address, reservation.size))
            .then_some(address),
        }
    };
    let Some(address) = address else {
        drop(storage);
        return Err(match target {
            Target::Fixed(_) => {
                let (start, size) = Heap::Replay.range();
                vk_errorf(
                    log,
                    VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                    format_args!(
                        "Capture address {fixed:#x} outside [{start:#x}, {:#x}) or already in use",
                        start.get() + size.get()
                    ),
                )
            }
            Target::Heap(_) => vk_error(log, VK_ERROR_OUT_OF_DEVICE_MEMORY),
        });
    };
    let va = Box::write(
        storage,
        RmVa {
            base: nvkmd_va {
                ops: &raw const VA_OPS,
                dev: dev.base_ptr(),
                flags: flags.bits(),
                pte_kind,
                addr: address.get(),
                size_B: reservation.size.get(),
            },
            ranges: Vec::new(),
            mappings: Vec::new(),
        },
    );
    Ok(Box::into_raw(va).cast())
}

/// The reservation's `bind_mem`.
///
/// # Safety
/// `va` is a live reservation of this backend; `log` NULL or a live Vulkan
/// object; `memory` a live allocation.
unsafe extern "C" fn va_bind_mem(
    va: *mut nvkmd_va,
    log: *mut vk_object_base,
    offset: u64,
    memory: *mut nvkmd_mem,
    memory_offset: u64,
    size: u64,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_va.bind_mem",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            // SAFETY: a live reservation, used by this call alone (contract).
            let Some(va) = (unsafe { RmVa::from_base(va) }) else {
                return vk_errorf(log, VK_ERROR_UNKNOWN, format_args!("bind of a foreign VA"));
            };
            // SAFETY: a live allocation (contract).
            let Some(memory) = (unsafe { RmMem::from_base(memory) }) else {
                return vk_errorf(
                    log,
                    VK_ERROR_UNKNOWN,
                    format_args!("bind of foreign memory"),
                );
            };
            va.bind(log, offset, memory, memory_offset, ByteSize::new(size))
        },
    )
}

/// The reservation's `unbind`.
///
/// # Safety
/// `va` is a live reservation of this backend; `log` NULL or a live Vulkan
/// object.
unsafe extern "C" fn va_unbind(
    va: *mut nvkmd_va,
    log: *mut vk_object_base,
    offset: u64,
    size: u64,
) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(
        "nvkmd_va.unbind",
        None,
        || VK_ERROR_DEVICE_LOST,
        || {
            // SAFETY: a live reservation, used by this call alone (contract).
            let Some(va) = (unsafe { RmVa::from_base(va) }) else {
                return vk_errorf(
                    log,
                    VK_ERROR_UNKNOWN,
                    format_args!("unbind of a foreign VA"),
                );
            };
            va.unbind(log, Range::new(offset, ByteSize::new(size)))
        },
    )
}

/// The reservation's `free`: every binding goes to one deferred RM VA free,
/// which returns the range to its heap through [`va_release`] once the work
/// that may still use it retired, on whichever thread retires it. It never
/// waits here: that work may wait on a fence this thread must produce
/// (`nvkmd_rm.c` `rm_va_free`). If the RM refuses, the range stays reserved and
/// the device is lost.
///
/// # Safety
/// `va` is a reservation `alloc_va` returned, not used again.
unsafe extern "C" fn va_free(va: *mut nvkmd_va) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_va.free", || {
        // SAFETY: a live reservation, not used again (contract).
        let va = unsafe { RmVa::owned(va) }.expect("Mesa frees its own backend's VAs");
        let dev = va.dev();
        let count = u32::try_from(va.mappings.len())
            .expect("binds within a 1 TiB window at 4 KiB granularity number below 2^32");
        // SAFETY: the device's live RM device; `count` ids at `mappings`, read
        // during the call; `release` and its data (the device's heaps, which
        // `nv_rm_close` outlives no release of) as `NvRmVaRelease` requires.
        let status = unsafe {
            nv_rm_va_free(
                dev.rm(),
                va.mappings.as_ptr(),
                count,
                va.base.addr,
                va.base.size_B,
                Some(va_release),
                core::ptr::from_ref(dev.heaps_lock()).cast_mut().cast(),
            )
        };
        if status != VK_SUCCESS {
            dev.publish_loss("VA free; reservation retained", status);
        }
        drop(va);
    });
}

/// `NvRmVaRelease`: the RM unmapped a freed reservation; its range goes back
/// to the heap of its address. Called at most once per freed reservation, on
/// any thread, before `nv_rm_close` of the device returns.
///
/// # Safety
/// `data` is the heaps lock of a device `va_free` passed, still live.
unsafe extern "C" fn va_release(data: *mut c_void, address: u64, size: u64) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_va release", || {
        // SAFETY: the device's heaps lock (`va_free`), live until the device
        // closes its RM device, which runs every outstanding release first.
        let heaps = unsafe { &*data.cast_const().cast::<Mutex<VaHeaps>>() };
        heaps
            .lock()
            .expect("the heap lock is never held across a panic")
            .free(GpuVa::new(address), ByteSize::new(size));
    });
}
