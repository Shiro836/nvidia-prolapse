// SPDX-License-Identifier: MIT
//! The exported `nv_rm_*` entry points over the device: pointer and argument
//! validation at the C boundary, the shared panic guard, and the mapping of
//! each entry's errors to the `VkResult` its Vulkan commands may return. The
//! declarations and their documentation are the `nv-rm-abi` crate (rm-abi/),
//! which mesa/nv_rm.h is generated from; every definition here is pinned to
//! its declaration at compile time. Threading and locking are the device's.
use super::{
    Allocation, Bridge, Buffer, Info, Inner, NvRmContextId, NvRmMappingId, NvRmMemoryId,
    NvRmSyncId, NvRmToken, ReclaimItem, SyncPoint, VaRelease, context_index, needs_slot, queue_ref,
    sync_pointer, sync_ref,
};
use crate::diag::diag;
use crate::{
    gpu::{Push, SubmitStatus},
    rm::{ByteSize, Error, GpuVa, Mapping, Result},
    syncobj::{Foreign, SyncObj},
};
use nv_rm_core::conv::{u64_from_usize, usize_from_u32};
use std::{
    ffi::c_void,
    os::fd::{BorrowedFd, IntoRawFd},
    sync::{Arc, atomic::Ordering},
};
/// A panic interrupting a call on a Bridge publishes device loss (waking
/// every waiter) and marks the Bridge failed: every later entry point on it
/// returns VK_ERROR_DEVICE_LOST without touching its state (crate::ffi).
impl crate::ffi::Failable for Bridge {
    /// Also true in a forked child: it gets VK_ERROR_DEVICE_LOST and must
    /// open its own device.
    fn failed(&self) -> bool {
        if !self.process.is_current() {
            diag!(
                "nvk Rust RM: device opened by pid {} used in a forked child; open a new device in this process",
                self.process.creator()
            );
            return true;
        }
        self.failed.is_set()
    }
    fn mark_failed(&self) {
        self.failed.set();
        self.core.mark_lost();
    }
}
/// The shared C-boundary guard (crate::ffi::guard) for an entry point on
/// `bridge`, failing with VK_ERROR_DEVICE_LOST.
///
/// An entry point that reports VK_ERROR_DEVICE_LOST publishes the loss on
/// its device before returning: Vulkan treats the logical device as lost
/// from then on (spec "Lost Device"), and Mesa learns it on its next status
/// or wait query (nvkmd_rm.c rm_wait_status, rm_check_status). Publication
/// is CPU-only (Core::mark_lost); stopping submissions follows on the next
/// locked call. The guard's own failure value skips this: that object is
/// already failed, or it belongs to the parent of a forked child, whose
/// wakeup descriptors must not be touched.
pub(super) fn guarded(
    entry: &'static str,
    bridge: Option<&Bridge>,
    f: impl FnOnce() -> i32,
) -> i32 {
    crate::ffi::guard(
        entry,
        bridge.map(|b| -> &dyn crate::ffi::Failable { b }),
        || -4,
        || {
            let status = f();
            if status == -4
                && let Some(bridge) = bridge
            {
                bridge.core.mark_lost();
            }
            status
        },
    )
}
/// The bridge argument of an entry point, or the named error for NULL.
pub(super) fn present(bridge: Option<&Bridge>) -> Result<&Bridge> {
    bridge.ok_or_else(|| Error::Contract("null RM bridge".into()))
}
/// `VkResult` values the entry points return (Vulkan registry `vk.xml`, enum
/// `VkResult`); `nvkmd_rm.c` passes them to Mesa as they are.
pub(super) mod vk {
    pub(super) const ERROR_OUT_OF_HOST_MEMORY: i32 = -1;
    pub(super) const ERROR_OUT_OF_DEVICE_MEMORY: i32 = -2;
    pub(super) const ERROR_INITIALIZATION_FAILED: i32 = -3;
    pub(super) const ERROR_DEVICE_LOST: i32 = -4;
    pub(super) const ERROR_MEMORY_MAP_FAILED: i32 = -5;
    pub(super) const ERROR_TOO_MANY_OBJECTS: i32 = -10;
    pub(super) const ERROR_UNKNOWN: i32 = -13;
    pub(super) const ERROR_INVALID_EXTERNAL_HANDLE: i32 = -1_000_072_003;
}
/// What a failed entry point's error means to the application, before the
/// entry point picks the `VkResult` its Vulkan commands may return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Failure {
    /// A host allocation failed (a syscall returned `ENOMEM`).
    HostMemory,
    /// RM ran out of memory (`Error::is_out_of_device_memory`).
    DeviceMemory,
    /// The device is lost (`Error::is_device_lost`).
    Lost,
    /// The kernel or RM refused the request.
    Refused,
    /// A broken interface contract (`Error::Contract`): the caller's, or an
    /// answer from the kernel or RM outside its contract.
    Contract,
}
impl Failure {
    fn of(error: &Error) -> Self {
        if error.is_device_lost() {
            return Self::Lost;
        }
        if error.is_out_of_device_memory() {
            return Self::DeviceMemory;
        }
        match error {
            Error::Io(_, error) if error.kind() == std::io::ErrorKind::OutOfMemory => {
                Self::HostMemory
            }
            Error::Contract(_) => Self::Contract,
            Error::Io(..)
            | Error::Rm { .. }
            | Error::Wait(_)
            | Error::DeviceLost(_)
            | Error::Sync(_)
            | Error::Channel(_)
            | Error::Memory(_) => Self::Refused,
        }
    }
}
/// The Vulkan commands an entry point serves. A failure may only become a
/// `VkResult` that every one of them lists in its "Return Codes" (`vk.xml`
/// `errorcodes`); where Mesa's NVK maps the same kernel failure, the entry
/// point follows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Entry {
    /// `nv_rm_open`: physical-device probing (`vkEnumeratePhysicalDevices`) and
    /// `vkCreateDevice`. Their common codes are `OUT_OF_HOST_MEMORY`,
    /// `OUT_OF_DEVICE_MEMORY` and `INITIALIZATION_FAILED`;
    /// `vkEnumeratePhysicalDevices` has no `DEVICE_LOST`. NVK reports a device it
    /// cannot open as `INITIALIZATION_FAILED` (nvkmd_nouveau_dev.c:44 at Mesa
    /// 26.2.5).
    Open,
    /// `nv_rm_alloc`: `vkAllocateMemory`, and every command that allocates
    /// driver memory (object creation, command buffer recording, queue
    /// submission). Their common codes are `OUT_OF_HOST_MEMORY` and
    /// `OUT_OF_DEVICE_MEMORY`; `vkAllocateMemory` has no `DEVICE_LOST`. NVK
    /// reports every failed allocation as `OUT_OF_DEVICE_MEMORY`
    /// (nvkmd_nouveau_mem.c:169 at Mesa 26.2.5); a broken bridge contract
    /// is an implementation failure (`VK_ERROR_UNKNOWN`).
    Alloc,
    /// `nv_rm_import`: `vkAllocateMemory` importing a dma-buf (codes as
    /// [`Entry::Alloc`], plus `INVALID_EXTERNAL_HANDLE`). NVK reports a dma-buf
    /// the kernel does not import as `INVALID_EXTERNAL_HANDLE`
    /// (nvkmd_nouveau_mem.c:188 at Mesa 26.2.5); running out of memory and a
    /// lost device are `OUT_OF_DEVICE_MEMORY`, as for [`Entry::Alloc`].
    Import,
    /// `nv_rm_export`: `vkGetMemoryFdKHR` (and the Android hardware buffer
    /// export over it), whose codes are `TOO_MANY_OBJECTS` and
    /// `OUT_OF_HOST_MEMORY`. NVK reports every failed dma-buf export as
    /// `TOO_MANY_OBJECTS` (nvkmd_nouveau_mem.c:293 at Mesa 26.2.5).
    Export,
    /// `nv_rm_map`: `vkMapMemory`, whose codes are `OUT_OF_HOST_MEMORY`,
    /// `OUT_OF_DEVICE_MEMORY` and `MEMORY_MAP_FAILED`. NVK reports a failed CPU
    /// mapping as `MEMORY_MAP_FAILED` (nvkmd_nouveau_mem.c:247 at Mesa 26.2.5),
    /// RM running out of BAR space included; a lost device is published and
    /// reported the same way.
    Map,
    /// `nv_rm_bind`, `nv_rm_unbind`: `vkBindBufferMemory`/`vkBindImageMemory`
    /// (and their `2` forms) and `vkAllocateMemory`, which binds the new memory
    /// to its VA. Their common codes are `OUT_OF_HOST_MEMORY` and
    /// `OUT_OF_DEVICE_MEMORY`, without `DEVICE_LOST`. NVK reports a failed VM
    /// bind as `VK_ERROR_UNKNOWN` (nvkmd_nouveau_va.c:109 at Mesa 26.2.5),
    /// which every command may return; a lost device is published and reported
    /// the same way.
    Bind,
    /// `nv_rm_sync_create`, `nv_rm_sync_signal`, `nv_rm_sync_reset`: CPU-side
    /// payload operations of `vkCreateFence`/`vkCreateSemaphore`,
    /// `vkSignalSemaphore` and `vkResetFences`, none of which lists
    /// `DEVICE_LOST`. They do not check the device state: NVK runs them as DRM
    /// syncobj ioctls that never look at it (vk_drm_syncobj.c:48, :88, :172
    /// at Mesa 26.2.5), and NVIDIA's driver returns `VK_SUCCESS` for a host
    /// signal on a lost device, after which a batch blocked on that value
    /// runs. NVK's payloads report a failed signal or reset as
    /// `VK_ERROR_UNKNOWN` (vk_drm_syncobj.c:148, :179); an error on the way
    /// that reports device loss is published and reported the same way.
    Sync,
    /// `nv_rm_sync_import_fd`, `nv_rm_sync_export_fd`:
    /// `vkImport{Semaphore,Fence}FdKHR` and `vkGet{Semaphore,Fence}FdKHR`, whose
    /// codes are `OUT_OF_HOST_MEMORY` plus `INVALID_EXTERNAL_HANDLE` (import) or
    /// `TOO_MANY_OBJECTS` (export). NVK's payloads report every failed import or
    /// export as `VK_ERROR_UNKNOWN` (vk_drm_syncobj.c:376, :397, :413, :429), RM out of memory
    /// and a lost device included. An import does not check the device state:
    /// NVK's is a DRM syncobj ioctl that never looks at it, and NVIDIA's
    /// driver imports a semaphore fd without a loss check.
    SyncFd,
    /// `nv_rm_context_create`, `nv_rm_context_create_shared`: queue creation
    /// inside `vkCreateDevice`, which lists `INITIALIZATION_FAILED` and
    /// `DEVICE_LOST`. NVK reports a context it cannot create as
    /// `INITIALIZATION_FAILED` (nvkmd_nouveau_dev.c:44 for the device).
    Create,
    /// Every other entry point: submission, waits and status queries of a
    /// created device (`vkQueueSubmit`, `vkWaitForFences`, ...), where a lost
    /// device is `VK_ERROR_DEVICE_LOST` (spec "Lost Device"). Any other failure
    /// is an implementation failure, `VK_ERROR_UNKNOWN`, the one code every
    /// command may return (spec "Error Codes") and the one NVK reports for a
    /// failed submission (nvkmd_nouveau_ctx.c:150 at Mesa 26.2.5); these
    /// commands have no `INITIALIZATION_FAILED`.
    Device,
}
impl Entry {
    /// The `VkResult` this entry point returns for `error`.
    fn status(self, error: &Error) -> i32 {
        self.result(Failure::of(error))
    }
    fn result(self, failure: Failure) -> i32 {
        use Failure::{Contract, DeviceMemory, HostMemory, Lost, Refused};
        match self {
            Self::Open => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Lost | Refused | Contract => vk::ERROR_INITIALIZATION_FAILED,
            },
            Self::Create => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Lost => vk::ERROR_DEVICE_LOST,
                Refused | Contract => vk::ERROR_INITIALIZATION_FAILED,
            },
            Self::Alloc => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory | Lost | Refused => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Contract => vk::ERROR_UNKNOWN,
            },
            Self::Import => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory | Lost => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Refused => vk::ERROR_INVALID_EXTERNAL_HANDLE,
                Contract => vk::ERROR_UNKNOWN,
            },
            Self::Export => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory | Lost | Refused => vk::ERROR_TOO_MANY_OBJECTS,
                Contract => vk::ERROR_UNKNOWN,
            },
            Self::Map => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory | Lost | Refused | Contract => vk::ERROR_MEMORY_MAP_FAILED,
            },
            Self::Bind => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Lost | Refused | Contract => vk::ERROR_UNKNOWN,
            },
            Self::Sync => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Lost | Refused | Contract => vk::ERROR_UNKNOWN,
            },
            Self::SyncFd => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory | Lost | Refused | Contract => vk::ERROR_UNKNOWN,
            },
            Self::Device => match failure {
                HostMemory => vk::ERROR_OUT_OF_HOST_MEMORY,
                DeviceMemory => vk::ERROR_OUT_OF_DEVICE_MEMORY,
                Lost => vk::ERROR_DEVICE_LOST,
                Refused | Contract => vk::ERROR_UNKNOWN,
            },
        }
    }
}
/// The `VkResult` of a failed [`Entry::Device`] entry point (`nv_rm.h`).
pub(super) fn vk_status(error: &Error) -> i32 {
    Entry::Device.status(error)
}
/// The shared C-boundary guard for an entry point whose body returns a
/// `Result`: a failure becomes `entry`'s `VkResult`, and a lost device is
/// published on `bridge` whatever that result is (Mesa learns it on its next
/// status or wait query, as for [`guarded`]). A panic marks `bridge` failed
/// and returns `entry`'s result for a lost device.
pub(super) fn guarded_as(
    entry: Entry,
    name: &'static str,
    bridge: Option<&Bridge>,
    f: impl FnOnce() -> Result<()>,
) -> i32 {
    crate::ffi::guard(
        name,
        bridge.map(|b| -> &dyn crate::ffi::Failable { b }),
        || entry.result(Failure::Lost),
        || match f() {
            Ok(()) => 0,
            Err(error) => {
                diag!("nvk Rust RM: {error}");
                let failure = Failure::of(&error);
                if failure == Failure::Lost
                    && let Some(bridge) = bridge
                {
                    bridge.core.mark_lost();
                }
                entry.result(failure)
            }
        },
    )
}
pub(super) fn report(result: Result<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => {
            diag!("nvk Rust RM: {e}");
            vk_status(&e)
        }
    }
}
/// Write an entry point's result to its output pointer.
///
/// # Safety
/// `p` is NULL (rejected) or valid and aligned for a write of `T`.
pub(super) unsafe fn output<T>(p: *mut T, value: T) -> Result<()> {
    if p.is_null() {
        return Err(Error::Contract("null RM output".into()));
    }
    // SAFETY: non-NULL (checked above), writable and aligned (# Safety).
    unsafe { p.write(value) };
    Ok(())
}
/// An entry point's input array, at most `maximum` elements.
///
/// # Safety
/// When `count` is nonzero, `ptr` is NULL (rejected) or points to `count`
/// initialised, aligned `T`s that nothing writes during `'a`.
pub(super) unsafe fn input<'a, T>(ptr: *const T, count: u32, maximum: u32) -> Result<&'a [T]> {
    if count > maximum || (count != 0 && ptr.is_null()) {
        return Err(Error::Contract("invalid RM input array".into()));
    }
    Ok(if count == 0 {
        &[]
    } else {
        // SAFETY: non-NULL with count <= maximum (checked above); the
        // elements are readable and unchanged for 'a (# Safety).
        unsafe { std::slice::from_raw_parts(ptr, usize_from_u32(count)) }
    })
}
pub(super) fn status_of(result: Result<SubmitStatus>, out: *mut NvRmToken) -> i32 {
    let mut status = 0;
    let code = report((|| {
        let token = match result? {
            SubmitStatus::Submitted(token) => token,
            SubmitStatus::Full(oldest) => {
                status = 1;
                oldest
            }
        };
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, NvRmToken::new(token.get())) }
    })());
    if code == 0 { status } else { code }
}

/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_open(
    domain: u32,
    bus: u32,
    device: u32,
    function: u32,
    out: *mut *mut Bridge,
    info: *mut Info,
) -> i32 {
    guarded_as(Entry::Open, "nv_rm_open", None, move || {
        if out.is_null() || info.is_null() {
            return Err(Error::Contract("null RM open output".into()));
        }
        let (b, i) = Bridge::new([domain, bus, device, function])?;
        // SAFETY: nv_rm.h: `info` is a writable NvRmInfo for the call
        // (non-NULL checked above); Info is its #[repr(C)] layout.
        unsafe { info.write(i) };
        // SAFETY: nv_rm.h: `out` is writable for the call (non-NULL
        // checked above). The Box is released only by nv_rm_close.
        unsafe { out.write(Box::into_raw(Box::new(b))) };
        Ok(())
    })
}
/// Destroys a device made by nv_rm_open.
///
/// # Safety
/// `p` is NULL or a device from nv_rm_open, given back here exactly once,
/// with no call on it in flight and none after.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_close(p: *mut Bridge) {
    crate::ffi::guard_or_abort("nv_rm_close", || {
        if p.is_null() {
            return;
        }
        // SAFETY: nv_rm.h: `p` is a device from nv_rm_open that the caller
        // gives back here exactly once, with no call on it in flight or later.
        let owned = unsafe { Box::from_raw(p) };
        if crate::ffi::Failable::failed(&*owned) {
            // A panic interrupted an earlier call (its state may be
            // inconsistent), or this is a forked child (the RM objects and
            // threads are the parent's): keep it, with every RM object and
            // descriptor it owns, until process exit rather than tear it down.
            diag!("nv_rm_close: device is failed or from another process; its resources are kept");
            std::mem::forget(owned);
            return;
        }
        drop(owned);
    });
}
/// nv_rm.h NV_RM_PLACEMENT_*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Placement {
    /// NV_RM_PLACEMENT_VRAM
    Vram,
    /// NV_RM_PLACEMENT_GART
    Gart,
    /// NV_RM_PLACEMENT_VRAM_COMPRESSIBLE
    VramCompressible,
    /// NV_RM_PLACEMENT_VRAM_POOL_BLOCK
    VramPoolBlock,
}
impl TryFrom<u32> for Placement {
    type Error = Error;
    fn try_from(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::Vram),
            1 => Ok(Self::Gart),
            2 => Ok(Self::VramCompressible),
            3 => Ok(Self::VramPoolBlock),
            _ => Err(Error::Contract(format!("unknown memory placement {v}"))),
        }
    }
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_alloc(
    p: *mut Bridge,
    size: u64,
    alignment: u64,
    placement: u32,
    kind: u32,
    tile: u32,
    out: *mut Allocation,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Alloc, "nv_rm_alloc", object, move || {
        if out.is_null() {
            return Err(Error::Contract("null allocation output".into()));
        }
        let b = present(object)?;
        b.core.live()?;
        // `tile` (nouveau's per-BO tile mode for legacy GL importers) has
        // no RM memory attribute: the block height travels in the DRM
        // modifier. A kind is only meaningful for an uncompressed
        // block-linear VRAM/GART allocation (Device::allocate_block_linear).
        let placement = Placement::try_from(placement)?;
        if tile != 0
            || (kind != 0
                && matches!(
                    placement,
                    Placement::VramCompressible | Placement::VramPoolBlock
                ))
        {
            return Err(Error::Contract(
                    "unsupported memory placement/tiled allocation (tile mode, or a kind on compressible or pool-block memory)".into(),
                ));
        }
        let mut guard = b.lock()?;
        let s = &mut guard.0;
        // Retired frees first: their memory may be what this needs.
        b.reclaim_locked(s, false);
        // A size or alignment beyond the address space is a request no memory
        // can satisfy (VK_ERROR_OUT_OF_DEVICE_MEMORY), not a broken contract.
        let size = usize::try_from(size).map_err(|_overflow| crate::rm::MemoryError::Invalid {
            operation: "allocate memory",
            reason: "size exceeds the address space",
        })?;
        let alignment = usize::try_from(alignment.max(4096)).map_err(|_overflow| {
            crate::rm::MemoryError::Invalid {
                operation: "allocate memory",
                reason: "alignment exceeds the address space",
            }
        })?;
        let memory = match placement {
            Placement::VramCompressible => s.device.allocate_compressible(size, alignment)?,
            Placement::VramPoolBlock => {
                if size != crate::rm::POOL_BLOCK_SIZE {
                    return Err(Error::Contract(format!(
                        "a pool block is {:#x} bytes, not {size:#x}",
                        crate::rm::POOL_BLOCK_SIZE
                    )));
                }
                s.device.allocate_pool_block()?
            }
            Placement::Vram | Placement::Gart if kind != 0 => s.device.allocate_block_linear(
                size,
                alignment,
                placement == Placement::Gart,
                kind,
            )?,
            Placement::Vram | Placement::Gart => {
                s.device
                    .allocate_aligned(size, alignment, false, placement == Placement::Gart)?
            }
        };
        let id = next_id(s)?;
        let a = Allocation {
            id: NvRmMemoryId::new(id),
            size: u64_from_usize(memory.size),
            cpu_address: 0,
        };
        s.buffers.insert(
            id,
            Buffer {
                cpu: s.device.cpu_mapping_slot(),
                memory,
            },
        );
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, a) }
    })
}
pub(super) fn next_id(s: &mut Inner) -> Result<u64> {
    let id = s.next;
    s.next = id
        .checked_add(1)
        .ok_or_else(|| Error::Contract("RM object ID exhausted".into()))?;
    Ok(id)
}
/// Logical maps are independent from the lazy retained allocation mapping.
/// Idle mappings remain reclaimable; allocation free performs real teardown.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_map(
    p: *mut Bridge,
    memory: NvRmMemoryId,
    out: *mut usize,
) -> i32 {
    let memory = memory.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Map, "nv_rm_map", object, move || {
        if out.is_null() {
            return Err(Error::Contract("null CPU map output".into()));
        }
        let b = present(object)?;
        b.core.live()?;
        let mut guard = b.lock()?;
        let s = &mut guard.0;
        b.reclaim_locked(s, false);
        let buffer = s
            .buffers
            .get(&memory)
            .ok_or_else(|| Error::Contract("map unknown RM memory".into()))?;
        let result = buffer
            .cpu
            .acquire(|| buffer.memory.map(), Mapping::cpu_address);
        if s.device.cpu_mapping_failed() {
            b.lose_locked(s);
        }
        let address = result?;
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, address) }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_unmap(p: *mut Bridge, memory: NvRmMemoryId) -> i32 {
    let memory = memory.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_unmap", object, move || {
        report((|| {
            let b = present(object)?;
            // Cleanup remains possible after loss. A failed final teardown keeps
            // both its logical reference and the partially released lease owned.
            let mut guard = b.lock()?;
            let s = &mut guard.0;
            let result = s
                .buffers
                .get(&memory)
                .ok_or_else(|| Error::Contract("unmap unknown RM memory".into()))?
                .cpu
                .release();
            if result.is_err() {
                b.lose_locked(s);
            }
            result
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
///
/// # Panics
/// Only on a broken internal invariant (an id validated under the device
/// lock and then missing); the FFI guard turns that into device loss.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_free(p: *mut Bridge, memory: NvRmMemoryId) -> i32 {
    let memory = memory.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_free", object, move || {
        report((|| {
            let b = present(object)?;
            // Never waits for GPU work (it may be gated on a fence this very
            // thread must produce): the RM object is freed once the work
            // outstanding now has retired (reclaim.rs). Cleanup stays possible
            // after loss; lost queues keep the object until close.
            let mut guard = b.lock()?;
            let s = &mut guard.0;
            let buffer = s
                .buffers
                .get(&memory)
                .ok_or_else(|| Error::Contract("free unknown RM memory".into()))?;
            if buffer.cpu.refs() != 0 {
                return Err(Error::Contract(
                    "free CPU-mapped RM memory; unmap it first".into(),
                ));
            }
            // The CPU VMA is not GPU-visible: evict it now.
            if let Err(error) = buffer.cpu.evict() {
                // Do not remove the buffer or free its RM handle while cleanup
                // is incomplete; explicit retry/teardown retains ownership.
                b.lose_locked(s);
                return Err(error);
            }
            let buffer = s
                .buffers
                .remove(&memory)
                .expect("looked up above under the same lock");
            b.reclaim_or_defer(s, [ReclaimItem::Memory(buffer)])
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_bind(
    p: *mut Bridge,
    memory: NvRmMemoryId,
    offset: u64,
    size: u64,
    address: u64,
    kind: u32,
    out: *mut NvRmMappingId,
) -> i32 {
    let memory = memory.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Bind, "nv_rm_bind", object, move || {
        if out.is_null() {
            return Err(Error::Contract("null bind output".into()));
        }
        let b = present(object)?;
        b.core.live()?;
        if address < 1 << 34 || address.checked_add(size).is_none_or(|n| n > 1 << 40) {
            return Err(Error::Contract("Mesa VA outside reserved range".into()));
        }
        let mut guard = b.lock()?;
        let s = &mut guard.0;
        b.reclaim_locked(s, false);
        let buffer = s
            .buffers
            .get(&memory)
            .ok_or_else(|| Error::Contract("bind unknown RM memory".into()))?;
        let map = s.space.map_range(
            &buffer.memory,
            offset,
            ByteSize::new(size),
            Some(GpuVa::new(address)),
            kind,
        )?;
        if map.address() != GpuVa::new(address) {
            return Err(Error::Contract(
                "RM did not honor fixed mapping address".into(),
            ));
        }
        let id = next_id(s)?;
        s.maps.insert(id, map);
        s.bind_info.insert(id, (address, size, memory, offset));
        if let Some(cap) = s.capture.as_mut() {
            // VA → allocation index for pushbuffer capture; a reused VA
            // replaces every stale overlapping entry.
            let stale: Vec<u64> = cap
                .binds
                .range(..address + size)
                .filter(|&(&start, &(len, _, _))| start + len > address)
                .map(|(&start, _)| start)
                .collect();
            for start in stale {
                cap.binds.remove(&start);
            }
            cap.binds.insert(address, (size, memory, offset));
        }
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, NvRmMappingId::new(id)) }
    })
}
/// Immediate RM unmap, no GPU wait. Only for application-ordered Vulkan
/// binds: Vulkan forbids rebinding/unbinding memory that pending work uses
/// (see nv_rm.h). Driver teardown of a VA uses nv_rm_va_free instead.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_unbind(p: *mut Bridge, mapping: NvRmMappingId) -> i32 {
    let mapping = mapping.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Bind, "nv_rm_unbind", object, move || {
        let b = present(object)?;
        let mut guard = b.lock()?;
        let s = &mut guard.0;
        b.reclaim_locked(s, false);
        let result = s
            .maps
            .get_mut(&mapping)
            .ok_or_else(|| Error::Contract("unbind unknown GPU mapping".into()))?
            .try_unmap();
        if let Err(error) = result {
            // Keep the mapping and typed-VA reservation owned. C must not
            // recycle this address after a failed kernel unmap.
            b.lose_locked(s);
            return Err(Error::DeviceLost(format!(
                "GPU unmap failed; reservation retained: {error}"
            )));
        }
        s.maps.remove(&mapping);
        s.bind_info.remove(&mapping);
        Ok(())
    })
}
/// Release a VA reservation and its mappings without waiting for the GPU:
/// the RM unmaps run and `release(data, address, size)` hands the range back
/// to the caller's heap only once every queue's work submitted before this
/// call has retired (immediately when nothing is outstanding). See nv_rm.h.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
/// `release(data, ..)` stays callable from any thread until nv_rm_close
/// returns.
///
/// # Panics
/// Only on a broken internal invariant (an id validated under the device
/// lock and then missing); the FFI guard turns that into device loss.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_va_free(
    p: *mut Bridge,
    mappings: *const NvRmMappingId,
    count: u32,
    address: u64,
    size: u64,
    release: VaRelease,
    data: *mut c_void,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_va_free", object, move || {
        report((|| {
            let b = present(object)?;
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let ids = unsafe { input(mappings, count, 65536) }?;
            let release =
                release.ok_or_else(|| Error::Contract("null VA release callback".into()))?;
            let end = address
                .checked_add(size)
                .filter(|&end| size != 0 && address >= 1 << 34 && end <= 1 << 40)
                .ok_or_else(|| Error::Contract("VA free outside reserved range".into()))?;
            let mut guard = b.lock()?;
            let s = &mut guard.0;
            // Transactional: validate every id (known, inside this range, no
            // duplicates) before taking any; on error C keeps the range.
            for (i, id) in ids.iter().enumerate() {
                let map = s
                    .maps
                    .get(&id.get())
                    .ok_or_else(|| Error::Contract("VA free of unknown GPU mapping".into()))?;
                if map.address().get() < address
                    || map.address().get().saturating_add(map.length().get()) > end
                    || ids[..i].contains(id)
                {
                    return Err(Error::Contract(
                        "VA free mapping outside its range or repeated".into(),
                    ));
                }
            }
            if ids.is_empty() {
                // Nothing is mapped in the range: no GPU access can reach it.
                b.reclaim_locked(s, false);
                // SAFETY: nv_rm_va_free's contract: `release(data, ..)` is
                // callable until nv_rm_close returns; nothing maps the range.
                unsafe { release(data, address, size) };
                return Ok(());
            }
            // Group: unmaps, then the range release (in order). The scratch
            // vector keeps its capacity: no allocation per free.
            let mut freeing = std::mem::take(&mut s.freeing);
            for id in ids {
                s.bind_info.remove(&id.get());
            }
            freeing.extend(ids.iter().map(|id| {
                ReclaimItem::Mapping(
                    s.maps
                        .remove(&id.get())
                        .expect("every id was validated above under the same lock"),
                )
            }));
            freeing.push(ReclaimItem::Va {
                address,
                size,
                release,
                data,
            });
            let result = b.reclaim_or_defer(s, freeing.drain(..));
            s.freeing = freeing;
            result
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_create(
    p: *mut Bridge,
    engines: u32,
    out: *mut NvRmContextId,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Create, "nv_rm_context_create", object, move || {
        if out.is_null() {
            return Err(Error::Contract("null context output".into()));
        }
        let b = present(object)?;
        b.core.live()?;
        let mut guard = b.lock()?;
        let id = b.create_context(&mut guard.0, engines, None)?;
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, NvRmContextId::new(id)) }
    })
}
/// Create a context whose channel joins `share`'s TSG on the lowest free
/// subcontext (VEID); work on the two contexts can run on the GR engine at
/// the same time. Compute engine only; needs Volta+ subcontexts.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_create_shared(
    p: *mut Bridge,
    engines: u32,
    share: NvRmContextId,
    out: *mut NvRmContextId,
) -> i32 {
    let share = share.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(
        Entry::Create,
        "nv_rm_context_create_shared",
        object,
        move || {
            if out.is_null() {
                return Err(Error::Contract("null context output".into()));
            }
            let b = present(object)?;
            b.core.live()?;
            let mut guard = b.lock()?;
            let id = b.create_context(&mut guard.0, engines, Some(share))?;
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, NvRmContextId::new(id)) }
        },
    )
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
///
/// # Panics
/// Only on a broken internal invariant (an id validated under the device
/// lock and then missing); the FFI guard turns that into device loss.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_destroy(p: *mut Bridge, id: NvRmContextId) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_destroy", object, move || {
        report((|| {
            let b = present(object)?;
            let token = queue_ref(&b.lock()?.0, id)?.last_token();
            // Drain the queue while the device is healthy (bounded, 5 s; stock
            // waits without bound, ELF 0xd68170). A drain that runs out latches
            // the loss; a latched loss fails it at once. The channel is then
            // RM-freed as it is, with no schedule-off first and nothing
            // signalled on the CPU for a batch still blocked in an acquire:
            // the stock driver frees a lost device's channels the same way
            // (vkDestroyDevice ELF 0xca3690 -> 0xa8d4c0), and their RM free
            // stops them. Memory is released only after the free.
            let drained = b
                .wait_token(id, token.get(), 5_000_000_000)
                .unwrap_or_else(|error| {
                    diag!("nvk Rust RM context {id} destroy drain: {error}");
                    false
                });
            if !drained {
                b.lose_locked(&mut b.lock()?.0);
            }
            let mut guard = b.lock()?;
            let s = &mut guard.0;
            let queue = s
                .queues
                .get_mut(
                    context_index(id)
                        .ok_or_else(|| Error::Contract("destroy unknown context".into()))?,
                )
                .and_then(Option::take)
                .ok_or_else(|| Error::Contract("destroy unknown context".into()))?;
            b.check_idle_before_free(id, &queue);
            drop(queue);
            // Groups that waited only on this queue can go now.
            b.reclaim_locked(s, false);
            Ok(())
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_submit_waits(
    p: *mut Bridge,
    id: NvRmContextId,
    pushes: *const Push,
    count: u32,
    drm_fd: i32,
    wait_fds: *const i32,
    wait_count: u32,
    out: *mut NvRmToken,
) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_submit_waits", object, move || {
        let result = (|| {
            if out.is_null() {
                return Err(Error::Contract("null submit token output".into()));
            }
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let pushes = unsafe { input(pushes, count, 1000) }?;
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let fds = unsafe { input(wait_fds, wait_count, 1024) }?;
            if !fds.is_empty() && drm_fd < 0 {
                return Err(Error::Contract("missing dependency DRM fd".into()));
            }
            // Retain borrowed FDs before returning. We own the CPU->GPU gate;
            // no uncancellable NVIDIA DRM WAIT callbacks are installed.
            let mut foreign = Vec::with_capacity(fds.len());
            for &fd in fds {
                if fd < 0 {
                    return Err(Error::Contract("negative dependency fd".into()));
                }
                let status = crate::fences::sync_status(fd)?;
                // SAFETY: nv_rm.h: a descriptor argument is open for the whole call; the borrow ends with it.
                let owned = unsafe { BorrowedFd::borrow_raw(fd) }
                    .try_clone_to_owned()
                    .map_err(|e| Error::Io("retain submit dependency", e))?;
                foreign.push(Arc::new(Foreign::new(owned, status)));
            }
            let b = present(object)?;
            // SAFETY: no sync ids (waits and signals are empty).
            unsafe { b.submit(id, pushes, &[], &foreign, &[]) }
        })();
        status_of(result, out)
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_submit(
    p: *mut Bridge,
    id: NvRmContextId,
    pushes: *const Push,
    count: u32,
    out: *mut NvRmToken,
) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_submit", object, move || {
        // SAFETY: forwards this entry point's own arguments, under its contract.
        unsafe {
            nv_rm_submit_waits(
                p,
                NvRmContextId::new(id),
                pushes,
                count,
                -1,
                std::ptr::null(),
                0,
                out,
            )
        }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_poll(
    p: *mut Bridge,
    id: NvRmContextId,
    token: NvRmToken,
    complete: *mut u32,
) -> i32 {
    let id = id.get();
    let token = token.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_poll", object, move || {
        report((|| {
            if complete.is_null() {
                return Err(Error::Contract("null poll output".into()));
            }
            let b = present(object)?;
            let done = b.wait_token(id, token, 0)?;
            b.reclaim_try();
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(complete, u32::from(done)) }
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_wait(
    p: *mut Bridge,
    id: NvRmContextId,
    token: NvRmToken,
    timeout_ns: u64,
) -> i32 {
    let id = id.get();
    let token = token.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_wait", object, move || {
        let mut status = 0;
        let result = report((|| {
            let b = present(object)?;
            if !b.wait_token(id, token, timeout_ns)? {
                status = 2;
            }
            b.reclaim_try();
            Ok(())
        })());
        if result == 0 { status } else { result }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_fence(
    p: *mut Bridge,
    id: NvRmContextId,
    token: NvRmToken,
    drm_fd: i32,
    out: *mut i32,
) -> i32 {
    let id = id.get();
    let token = token.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_fence", object, move || {
        report((|| {
            if out.is_null() || drm_fd < 0 {
                return Err(Error::Contract("invalid context fence output/fd".into()));
            }
            let b = present(object)?;
            b.core.live()?;
            if b.capture.load(Ordering::Relaxed)
                && let Some(cap) = b.lock()?.0.capture.as_mut()
            {
                cap.mark(id, token, 1);
            }
            let fence = {
                let guard = b.lock()?;
                let queue = queue_ref(&guard.0, id)?;
                // SAFETY: nv_rm.h: `drm_fd` is open for the whole call
                // (non-negative checked above).
                unsafe { queue.context_fence(drm_fd, token) }
            };
            let fd = fence?;
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, fd.into_raw_fd()) }
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_exec(
    p: *mut Bridge,
    id: NvRmContextId,
    pushes: *const Push,
    count: u32,
) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_exec", object, move || {
        loop {
            let mut token = NvRmToken::new(0);
            // SAFETY: this entry point's own arguments, under the same
            // contract; `token` is a live local.
            let result =
                unsafe { nv_rm_submit(p, NvRmContextId::new(id), pushes, count, &raw mut token) };
            if result != 0 && result != 1 {
                return result;
            }
            // Synchronous by contract (nv_rm.h nv_rm_exec), so unbounded like
            // nvkmd_rm.c's context sync; see `deadline` for how loss ends it.
            // SAFETY: this entry point's own device argument.
            let wait = unsafe { nv_rm_wait(p, NvRmContextId::new(id), token, u64::MAX) };
            if wait != 0 {
                return wait;
            }
            if result == 0 {
                return 0;
            }
        }
    })
}
/// Mesa's device-status hook (after Vulkan waits): loss flag plus the
/// throttled authoritative channel scan; foreign/shared producer errors were
/// already published as loss by their observer.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_check_status(p: *mut Bridge) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_check_status", object, move || {
        let result = report((|| {
            let b = present(object)?;
            b.check()?;
            b.reclaim_try();
            Ok(())
        })());
        if result == 0 {
            0
        } else {
            -4 // VK_ERROR_DEVICE_LOST
        }
    })
}
/// Mesa's software loss paths must wake caller waits too. This callback can
/// run under a Mesa queue mutex: atomics, condvar notify and eventfd writes
/// only — never Inner, RM ioctls or GPU waits. Submission stop and shared
/// failure publication follow on the next locked call / watcher / close.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_mark_lost(p: *const Bridge) {
    crate::ffi::guard_or_abort("nv_rm_mark_lost", || {
        // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open
        // device, kept alive by the caller for the whole call.
        if let Some(bridge) = unsafe { p.as_ref() } {
            bridge.core.mark_lost();
        }
    });
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_is_lost(p: *const Bridge) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    // A failed device reads as lost (1), like NULL.
    crate::ffi::guard(
        "nv_rm_is_lost",
        object.map(|b| -> &dyn crate::ffi::Failable { b }),
        || 1,
        move || object.map_or(1, |b| i32::from(b.core.is_lost())),
    )
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_timestamp(p: *mut Bridge, out: *mut u64) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_timestamp", object, move || {
        report((|| {
            let b = present(object)?;
            let guard = b.lock()?;
            let device = &guard.0.device;
            let mut time = 0u64;
            device
                .client
                .control(device.subdevice, 0x20800403, &mut time)?;
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, time) }
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_export(
    p: *mut Bridge,
    memory: NvRmMemoryId,
    drm_fd: i32,
    out: *mut i32,
) -> i32 {
    let memory = memory.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Export, "nv_rm_export", object, move || {
        if drm_fd < 0 || out.is_null() {
            return Err(Error::Contract("invalid external-memory output/fd".into()));
        }
        let b = present(object)?;
        b.core.live()?;
        let guard = b.lock()?;
        let buffer = guard
            .0
            .buffers
            .get(&memory)
            .ok_or_else(|| Error::Contract("export unknown allocation".into()))?;
        // Caller retains fd for the duration of this synchronous call.
        let fd =
                // SAFETY: nv_rm.h: a descriptor argument is open for the whole call; the borrow ends with it.
                crate::external::export(&buffer.memory, unsafe { BorrowedFd::borrow_raw(drm_fd) })?;
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, fd.into_raw_fd()) }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_import(
    p: *mut Bridge,
    drm_fd: i32,
    dma_fd: i32,
    out: *mut Allocation,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Import, "nv_rm_import", object, move || {
        if drm_fd < 0 || out.is_null() {
            return Err(Error::Contract("invalid dma-buf import arguments".into()));
        }
        // The application's handle (VkImportMemoryFdInfoKHR::fd).
        if dma_fd < 0 {
            return Err(crate::rm::MemoryError::ExternalHandle {
                reason: "negative dma-buf descriptor",
                value: u64::from(dma_fd.unsigned_abs()),
            }
            .into());
        }
        let b = present(object)?;
        b.core.live()?;
        let mut guard = b.lock()?;
        let s = &mut guard.0;
        b.reclaim_locked(s, false);
        // SAFETY: nv_rm.h: `drm_fd` is open for the whole call
        // (non-negative checked above); the borrow ends with it.
        let drm = unsafe { BorrowedFd::borrow_raw(drm_fd) };
        // SAFETY: as `drm_fd`, for the dma-buf.
        let dma_buf = unsafe { BorrowedFd::borrow_raw(dma_fd) };
        let memory = crate::external::import(&s.device, drm, dma_buf)?;
        let id = next_id(s)?;
        let result = Allocation {
            id: NvRmMemoryId::new(id),
            size: u64_from_usize(memory.size),
            cpu_address: 0,
        };
        s.buffers.insert(
            id,
            Buffer {
                cpu: s.device.cpu_mapping_slot(),
                memory,
            },
        );
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, result) }
    })
}

// ---- synchronization ABI (all caller-side) ------------------------------
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_create(
    p: *mut Bridge,
    flags: u32,
    initial: u64,
    out: *mut NvRmSyncId,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Sync, "nv_rm_sync_create", object, move || {
        if out.is_null() || flags & !7 != 0 {
            return Err(Error::Contract("invalid sync create arguments".into()));
        }
        // No liveness check: see Entry::Sync.
        let b = present(object)?;
        let slot = b.slot()?;
        let sync = match SyncObj::new(&b.core.pool, flags & 1 != 0, initial, slot) {
            Ok(sync) => sync,
            Err(error) => {
                b.core.pool.release(slot);
                return Err(error);
            }
        };
        let id = u64_from_usize(Arc::into_raw(Arc::new(sync)).expose_provenance());
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, NvRmSyncId::new(id)) }
    })
}
/// # Safety
/// `id` is a live nv_rm_sync_create payload, destroyed exactly once, with no
/// other call using it in flight or after; the device pointer is unused.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_destroy(_p: *mut Bridge, id: NvRmSyncId) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { _p.as_ref() };
    guarded("nv_rm_sync_destroy", object, move || {
        report((|| {
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            unsafe { sync_ref(id) }?;
            // SAFETY: id is the Arc pointer created by nv_rm_sync_create; Mesa
            // destroys each vk_sync exactly once. Accepted submissions own their
            // slot references independently of this payload.
            drop(unsafe { Arc::from_raw(sync_pointer(id)?) });
            Ok(())
        })())
    })
}
/// Kept for ABI stability: destruction is now immediate on the caller (there
/// is no worker queue to defer to); Mesa's void finish callback still uses it.
///
/// # Safety
/// As nv_rm_sync_destroy; `p` is NULL or a live nv_rm_open device.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_destroy_deferred(p: *mut Bridge, id: NvRmSyncId) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_destroy_deferred", object, move || {
        if p.is_null() {
            return report(Err(Error::Contract("null RM bridge".into())));
        }
        // SAFETY: this entry point's own arguments, under the same contract.
        unsafe { nv_rm_sync_destroy(p, id) }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_reset(p: *mut Bridge, id: NvRmSyncId) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Sync, "nv_rm_sync_reset", object, move || {
        // No liveness check: see Entry::Sync.
        let b = present(object)?;
        // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
        let sync = unsafe { sync_ref(id) }?;
        let mut fresh = if sync.needs_fresh_slot()? {
            Some(b.slot()?)
        } else {
            None
        };
        loop {
            match sync.reset(fresh.take()) {
                Ok((unused, changed)) => {
                    if let Some(slot) = unused {
                        b.core.pool.release(slot);
                    }
                    b.broadcast(changed);
                    return Ok(());
                }
                Err(error) if needs_slot(&error) => fresh = Some(b.slot()?),
                Err(error) => return Err(error),
            }
        }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_move(
    p: *mut Bridge,
    dst: NvRmSyncId,
    src: NvRmSyncId,
    _shared_source: u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_move", object, move || {
        report((|| {
            // No liveness check: a payload move is a host payload operation
            // (see Entry::Sync); NVK's runs as DRM syncobj ioctls
            // (vk_drm_syncobj.c:437-468 at Mesa 26.2.5).
            let b = present(object)?;
            if src == dst {
                return Err(Error::Contract("invalid binary sync move".into()));
            }
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            let (source, target) = (unsafe { sync_ref(src) }?, unsafe { sync_ref(dst) }?);
            if source.timeline || target.timeline {
                return Err(Error::Contract("invalid binary sync move".into()));
            }
            let fresh = b.slot()?;
            let (moved, changed) = match source.take_for_move(fresh) {
                Ok(moved) => moved,
                Err(error) => {
                    b.core.pool.release(fresh);
                    return Err(error);
                }
            };
            target.install_moved(moved)?;
            b.broadcast(changed);
            Ok(())
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_signal(
    p: *mut Bridge,
    id: NvRmSyncId,
    value: u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::Sync, "nv_rm_sync_signal", object, move || {
        // No liveness check: see Entry::Sync.
        let b = present(object)?;
        // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
        let sync = unsafe { sync_ref(id) }?;
        let mut fresh = if !sync.timeline && sync.needs_fresh_slot()? {
            Some(b.slot()?)
        } else {
            None
        };
        loop {
            match sync.signal(value, fresh.take()) {
                Ok((unused, changed)) => {
                    if let Some(slot) = unused {
                        b.core.pool.release(slot);
                    }
                    b.broadcast(changed);
                    // Dekker: store above, then wake local waiters if any.
                    b.core.waiters.notify();
                    return Ok(());
                }
                Err(error) if needs_slot(&error) => fresh = Some(b.slot()?),
                Err(error) => return Err(error),
            }
        }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_get_value(
    p: *mut Bridge,
    id: NvRmSyncId,
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_get_value", object, move || {
        report((|| {
            if out.is_null() {
                return Err(Error::Contract("null counter output".into()));
            }
            let b = present(object)?;
            b.check()?;
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            let (value, changed) = unsafe { sync_ref(id) }?.value()?;
            b.broadcast(changed);
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, value) }
        })())
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_status_many(
    p: *mut Bridge,
    points: *const SyncPoint,
    count: u32,
    flags: u32,
    out: *mut u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_status_many", object, move || {
        report((|| {
            if out.is_null() || flags & !3 != 0 {
                return Err(Error::Contract("invalid sync status arguments".into()));
            }
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let points = unsafe { input(points, count, 65536) }?;
            let b = present(object)?;
            b.check()?;
            // SAFETY: nv_rm.h: the points' sync ids are live for the call.
            let ready = unsafe { b.points_ready(points, flags) }?;
            b.reclaim_try();
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, u32::from(ready)) }
        })())
    })
}
/// Returns 0 on completion, 2 on caller timeout, negative on failure. Waits
/// on the calling thread (leader/follower on the device HOST descriptor).
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_wait_many(
    p: *mut Bridge,
    points: *const SyncPoint,
    count: u32,
    flags: u32,
    timeout_ns: u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_wait_many", object, move || {
        let result = (|| {
            if flags & !3 != 0 {
                return Err(Error::Contract("invalid sync wait flags".into()));
            }
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let points = unsafe { input(points, count, 65536) }?;
            let b = present(object)?;
            // SAFETY: nv_rm.h: the points' sync ids are live for the call.
            let ready = unsafe { b.wait_points(points, flags, timeout_ns) }?;
            b.reclaim_try();
            Ok(ready)
        })();
        match result {
            Ok(true) => 0,
            Ok(false) => 2,
            Err(e) => report(Err(e)),
        }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_import_fd(
    p: *mut Bridge,
    id: NvRmSyncId,
    fd: i32,
    opaque: u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::SyncFd, "nv_rm_sync_import_fd", object, move || {
        // No liveness check: see Entry::SyncFd.
        present(object)?;
        if fd < 0 {
            return Err(Error::Contract("invalid synchronization import fd".into()));
        }
        // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
        let sync = unsafe { sync_ref(id) }?;
        // SAFETY: nv_rm.h: a descriptor argument is open for the whole call; the borrow ends with it.
        let fd = unsafe { BorrowedFd::borrow_raw(fd) };
        if opaque != 0 {
            sync.import_opaque(fd)
        } else {
            sync.import_sync_file(fd)
        }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_export_fd(
    p: *mut Bridge,
    id: NvRmSyncId,
    opaque: u32,
    drm_fd: i32,
    out: *mut i32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded_as(Entry::SyncFd, "nv_rm_sync_export_fd", object, move || {
        if out.is_null() {
            return Err(Error::Contract("null synchronization fd output".into()));
        }
        let b = present(object)?;
        b.core.live()?;
        if b.capture.load(Ordering::Relaxed)
            && let Some(cap) = b.lock()?.0.capture.as_mut()
        {
            cap.mark(0, id.get(), 2 + opaque);
        }
        let fd = if opaque != 0 {
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            let shared = unsafe { sync_ref(id) }?.share()?;
            if shared.has_local_work() {
                let mut guard = b.lock()?;
                b.ensure_watcher_host(&mut guard.0)?;
                drop(guard);
                b.core.watcher.publish(&b.core, Arc::clone(&shared))?;
            }
            Some(shared.export()?)
        } else {
            // SAFETY: nv_rm.h: `id` is live and `drm_fd` negative or open
            // for the call.
            unsafe { b.export_sync_file(id, drm_fd) }?
        };
        // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
        unsafe { output(out, fd.map_or(-1, IntoRawFd::into_raw_fd)) }
    })
}
/// Reserve a binary signal for a deferred (submit-thread) submission: `dst`
/// becomes the handle the deferred batch signals, bound to `src`'s current
/// slot; `src` reports the signal as pending. `*reserved` = 0 when `src`
/// cannot be reserved (not an unsignaled native binary without a pending
/// signal), in which case nothing changed.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_reserve(
    p: *mut Bridge,
    dst: NvRmSyncId,
    src: NvRmSyncId,
    reserved: *mut u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_reserve", object, move || {
        report((|| {
            if reserved.is_null() {
                return Err(Error::Contract("null reservation output".into()));
            }
            let b = present(object)?;
            b.core.live()?;
            if src == dst {
                return Err(Error::Contract("invalid binary signal reservation".into()));
            }
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            let (done, changed) =
                unsafe { sync_ref(src) }?.reserve_into(unsafe { sync_ref(dst) }?)?;
            b.broadcast(changed);
            // WAIT_PENDING waiters observe the reservation as a submission.
            std::sync::atomic::fence(Ordering::SeqCst);
            if done && b.core.pending_waiters.load(Ordering::SeqCst) != 0 {
                b.core.waiters.notify();
            }
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(reserved, u32::from(done)) }
        })())
    })
}
/// Submit commands, dependencies and final signal payloads in one transaction.
///
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_submit_batch(
    p: *mut Bridge,
    id: NvRmContextId,
    pushes: *const Push,
    count: u32,
    waits: *const SyncPoint,
    wait_count: u32,
    signals: *const SyncPoint,
    signal_count: u32,
    out: *mut NvRmToken,
) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_submit_batch", object, move || {
        let result = (|| {
            if out.is_null() {
                return Err(Error::Contract("null submit token output".into()));
            }
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let pushes = unsafe { input(pushes, count, 1000) }?;
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let waits = unsafe { input(waits, wait_count, 1024) }?;
            // SAFETY: nv_rm.h: an input array holds `count` readable elements for the call (NULL only with count 0; input() checks both).
            let signals = unsafe { input(signals, signal_count, 65536) }?;
            let b = present(object)?;
            // SAFETY: nv_rm.h: the waits' and signals' sync ids are live for
            // the call.
            unsafe { b.submit(id, pushes, waits, &[], signals) }
        })();
        status_of(result, out)
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_submit_sync(
    p: *mut Bridge,
    id: NvRmContextId,
    pushes: *const Push,
    count: u32,
    waits: *const SyncPoint,
    wait_count: u32,
    out: *mut NvRmToken,
) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_submit_sync", object, move || {
        // SAFETY: forwards this entry point's own arguments, under its contract.
        unsafe {
            nv_rm_submit_batch(
                p,
                NvRmContextId::new(id),
                pushes,
                count,
                waits,
                wait_count,
                std::ptr::null(),
                0,
                out,
            )
        }
    })
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_signal(
    p: *mut Bridge,
    id: NvRmContextId,
    signals: *const SyncPoint,
    count: u32,
    out: *mut NvRmToken,
) -> i32 {
    let id = id.get();
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_signal", object, move || {
        // SAFETY: forwards this entry point's own arguments, under its contract.
        unsafe {
            nv_rm_submit_batch(
                p,
                NvRmContextId::new(id),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                signals,
                count,
                out,
            )
        }
    })
}

// Every exported entry has exactly the type `nv-rm-abi` declares for it, the
// declarations `mesa/nv_rm.h` is generated from.
const _: nv_rm_abi::NvRmOpen<Bridge> = nv_rm_open;
const _: nv_rm_abi::NvRmClose<Bridge> = nv_rm_close;
const _: nv_rm_abi::NvRmAlloc<Bridge> = nv_rm_alloc;
const _: nv_rm_abi::NvRmMap<Bridge> = nv_rm_map;
const _: nv_rm_abi::NvRmUnmap<Bridge> = nv_rm_unmap;
const _: nv_rm_abi::NvRmFree<Bridge> = nv_rm_free;
const _: nv_rm_abi::NvRmBind<Bridge> = nv_rm_bind;
const _: nv_rm_abi::NvRmUnbind<Bridge> = nv_rm_unbind;
const _: nv_rm_abi::NvRmVaFree<Bridge> = nv_rm_va_free;
const _: nv_rm_abi::NvRmContextCreate<Bridge> = nv_rm_context_create;
const _: nv_rm_abi::NvRmContextCreateShared<Bridge> = nv_rm_context_create_shared;
const _: nv_rm_abi::NvRmContextDestroy<Bridge> = nv_rm_context_destroy;
const _: nv_rm_abi::NvRmSubmitWaits<Bridge> = nv_rm_submit_waits;
const _: nv_rm_abi::NvRmSubmit<Bridge> = nv_rm_submit;
const _: nv_rm_abi::NvRmPoll<Bridge> = nv_rm_poll;
const _: nv_rm_abi::NvRmWait<Bridge> = nv_rm_wait;
const _: nv_rm_abi::NvRmContextFence<Bridge> = nv_rm_context_fence;
const _: nv_rm_abi::NvRmExecFn<Bridge> = nv_rm_exec;
const _: nv_rm_abi::NvRmCheckStatus<Bridge> = nv_rm_check_status;
const _: nv_rm_abi::NvRmMarkLost<Bridge> = nv_rm_mark_lost;
const _: nv_rm_abi::NvRmIsLost<Bridge> = nv_rm_is_lost;
const _: nv_rm_abi::NvRmTimestamp<Bridge> = nv_rm_timestamp;
const _: nv_rm_abi::NvRmExport<Bridge> = nv_rm_export;
const _: nv_rm_abi::NvRmImport<Bridge> = nv_rm_import;
const _: nv_rm_abi::NvRmSyncCreate<Bridge> = nv_rm_sync_create;
const _: nv_rm_abi::NvRmSyncDestroy<Bridge> = nv_rm_sync_destroy;
const _: nv_rm_abi::NvRmSyncDestroyDeferred<Bridge> = nv_rm_sync_destroy_deferred;
const _: nv_rm_abi::NvRmSyncReset<Bridge> = nv_rm_sync_reset;
const _: nv_rm_abi::NvRmSyncMove<Bridge> = nv_rm_sync_move;
const _: nv_rm_abi::NvRmSyncSignal<Bridge> = nv_rm_sync_signal;
const _: nv_rm_abi::NvRmSyncGetValue<Bridge> = nv_rm_sync_get_value;
const _: nv_rm_abi::NvRmSyncStatusMany<Bridge> = nv_rm_sync_status_many;
const _: nv_rm_abi::NvRmSyncWaitMany<Bridge> = nv_rm_sync_wait_many;
const _: nv_rm_abi::NvRmSyncImportFd<Bridge> = nv_rm_sync_import_fd;
const _: nv_rm_abi::NvRmSyncExportFd<Bridge> = nv_rm_sync_export_fd;
const _: nv_rm_abi::NvRmSyncReserve<Bridge> = nv_rm_sync_reserve;
const _: nv_rm_abi::NvRmSubmitBatch<Bridge> = nv_rm_submit_batch;
const _: nv_rm_abi::NvRmSubmitSync<Bridge> = nv_rm_submit_sync;
const _: nv_rm_abi::NvRmContextSignal<Bridge> = nv_rm_context_signal;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rm::{ChannelError, ClassId, FenceValue, MemoryError, SyncError, WaitError};
    use std::mem::{offset_of, size_of};
    use std::time::Duration;
    #[test]
    fn status_codes_match_the_c_abi() {
        // nvkmd_rm.c rm_result() returns the status as the VkResult. A failure
        // of a created device that is neither loss nor out of memory is an
        // implementation failure, VK_ERROR_UNKNOWN (-13): submission, waits and
        // status queries have no VK_ERROR_INITIALIZATION_FAILED.
        let io = |errno| Error::Io("op", std::io::Error::from_raw_os_error(errno));
        assert_eq!(vk_status(&Error::DeviceLost("x".into())), -4);
        assert_eq!(vk_status(&io(12)), -1);
        // A failed host allocation in our own code is out of host memory too.
        assert_eq!(
            vk_status(&Error::Io(
                "op",
                std::io::Error::new(std::io::ErrorKind::OutOfMemory, "try_reserve")
            )),
            -1
        );
        assert_eq!(vk_status(&io(22)), -13);
        assert_eq!(vk_status(&Error::Contract("x".into())), -13);
        assert_eq!(
            vk_status(&Error::Rm {
                operation: "op",
                status: 0x1f
            }),
            -13
        );
        let wait = WaitError::Timeout {
            operation: "op",
            object: "x".into(),
            limit: Duration::ZERO,
            elapsed: Duration::ZERO,
        };
        assert_eq!(vk_status(&wait.into()), -13);
        // RM out of memory or of another resource it allocates from:
        // VK_ERROR_OUT_OF_DEVICE_MEMORY (nvstatuscodes.h 0x51, 0x1a).
        for status in [0x51, 0x1a] {
            assert_eq!(
                vk_status(&Error::Rm {
                    operation: "op",
                    status
                }),
                -2,
                "{status:#x}"
            );
        }
        // RM reporting a GPU off the bus, in or needing a reset, or a channel
        // killed by robust channel recovery: device loss (nvstatuscodes.h).
        for status in [0x0f, 0x10, 0x60, 0x62] {
            assert_eq!(
                vk_status(&Error::Rm {
                    operation: "op",
                    status
                }),
                -4,
                "{status:#x}"
            );
        }
        // A poisoned or closed payload and a failed producer are device loss;
        // an operation that does not apply is a refused request.
        for (error, status) in [
            (SyncError::Poisoned { object: "x" }, -4),
            (SyncError::Closed("x"), -4),
            (SyncError::ProducerError { status: -110 }, -4),
            (SyncError::SharedProducerFailed, -4),
            (
                SyncError::Invalid {
                    operation: "x",
                    reason: "y",
                },
                -13,
            ),
            (SyncError::GenerationExhausted { operation: "x" }, -13),
            // The slot pool is GPU-visible memory: full is out of device memory.
            (SyncError::SlotsExhausted { slots: 1 << 20 }, -2),
        ] {
            assert_eq!(vk_status(&error.into()), status);
        }
        // A channel that already ran and stopped working is device loss
        // (VK_ERROR_DEVICE_LOST); a request the channel or RM refused is not.
        for (error, status) in [
            (ChannelError::Lost { operation: "x" }, -4),
            (ChannelError::Stopped, -4),
            (
                ChannelError::Rc {
                    error: 31,
                    put: 0,
                    sequence: 0,
                    host_class: 0xc96f,
                    info16: 0,
                    status: 0xffff,
                    timestamp: 0,
                },
                -4,
            ),
            (ChannelError::Timeline { reason: "x" }, -4),
            (ChannelError::PutReadback { visible: 1, put: 2 }, -4),
            (
                ChannelError::Allocate {
                    class: ClassId::new(0xc96f),
                    source: Box::new(Error::DeviceLost("x".into())),
                },
                -4,
            ),
            (
                ChannelError::Allocate {
                    class: ClassId::new(0xc96f),
                    source: Box::new(Error::Rm {
                        operation: "op",
                        status: 0x51,
                    }),
                },
                -2,
            ),
            (ChannelError::ClassList { reason: "x" }, -13),
            (
                ChannelError::MissingClass {
                    operation: "x",
                    class: ClassId::new(0xc7c0),
                },
                -13,
            ),
            (
                ChannelError::HostClass {
                    class: ClassId::new(0xa06f),
                },
                -13,
            ),
            (
                ChannelError::Unsupported {
                    operation: "x",
                    reason: "y",
                },
                -13,
            ),
            (ChannelError::EngineMask { mask: 0x40 }, -13),
            (ChannelError::SharedGroupEngine { mask: 0x1 }, -13),
            (ChannelError::Group { reason: "x" }, -13),
            (ChannelError::VaHeapCollision { what: "x" }, -13),
            (ChannelError::SemaphoreAddress { address: 4 }, -13),
            (ChannelError::HostMethodsTooLarge { words: 1 }, -13),
            (
                ChannelError::Batch {
                    operation: "x",
                    entries: 0,
                    waits: 0,
                    signals: 0,
                },
                -13,
            ),
            (
                ChannelError::Push {
                    address: 2,
                    size: 4,
                    flags: 0,
                },
                -13,
            ),
            (ChannelError::SequenceExhausted, -13),
            (
                ChannelError::Sem32Range {
                    token: FenceValue::new(1 << 31),
                    retired: FenceValue::new(0),
                },
                -13,
            ),
            (
                ChannelError::Token {
                    operation: "x",
                    token: FenceValue::new(2),
                    issued: FenceValue::new(1),
                },
                -13,
            ),
            (ChannelError::Doorbell, -13),
            (
                ChannelError::ResultMismatch {
                    word: 0,
                    actual: 1,
                    expected: 2,
                },
                -13,
            ),
        ] {
            assert_eq!(vk_status(&error.into()), status);
        }
        // A broken CPU-map cleanup is device loss; every other memory failure
        // is a refused request.
        for (error, status) in [
            (MemoryError::CleanupFailed, -4),
            (
                MemoryError::EvictionFailed(Box::new(Error::Contract("x".into()))),
                -4,
            ),
            (
                MemoryError::RollbackFailed {
                    error: Box::new(Error::Contract("x".into())),
                    cleanup: Box::new(Error::Contract("y".into())),
                },
                -4,
            ),
            (
                MemoryError::BarPressure {
                    used: 1,
                    requested: 2,
                    limit: 3,
                },
                -13,
            ),
            (MemoryError::NotLive { reason: "x" }, -13),
            (
                MemoryError::Invalid {
                    operation: "x",
                    reason: "y",
                },
                -13,
            ),
            (
                MemoryError::BlockLinearKind {
                    kind: 1,
                    generic: 6,
                },
                -13,
            ),
            (
                MemoryError::PoolBlockGrant {
                    attr: 0,
                    size: 0,
                    requested_attr: 1,
                    requested_size: 1,
                    free: None,
                },
                -13,
            ),
            (
                MemoryError::BlockLinearGrant {
                    size: 0,
                    format: 0,
                    attr: 0,
                    kind: 6,
                    free: None,
                },
                -13,
            ),
            (MemoryError::BarAccounting { reason: "x" }, -13),
            (MemoryError::CpuMapping { reason: "x" }, -13),
            (MemoryError::UserdLength { length: 1 }, -13),
            (
                MemoryError::PatternMismatch {
                    word: 0,
                    actual: 1,
                    expected: 2,
                },
                -13,
            ),
            (
                MemoryError::GpuMappingLimits {
                    offset: 0,
                    length: ByteSize::new(1),
                    allocation_size: ByteSize::new(0),
                },
                -13,
            ),
            (
                MemoryError::KindMismatch {
                    chosen: 0,
                    requested: 6,
                },
                -13,
            ),
            (
                MemoryError::OutOfBounds {
                    access: "x",
                    offset: 4,
                    len: 4,
                    size: 4,
                },
                -13,
            ),
            (
                MemoryError::GpuVa {
                    address: GpuVa::new(1),
                },
                -13,
            ),
        ] {
            assert_eq!(vk_status(&error.into()), status);
        }
        assert_eq!(report(Ok(())), 0);
    }
    #[test]
    fn bridge_info_matches_independent_c_oracle() {
        // mesa/abi-check.c: host and Android compile from the public bridge header.
        assert_eq!(size_of::<Info>(), 224);
        assert_eq!(offset_of!(Info, max_shared_ctx), 216);
        assert_eq!(offset_of!(Info, zcull), 152);
        assert_eq!(offset_of!(Info, zcull_ctxsw_size), 192);
        assert_eq!(offset_of!(Info, has_zcull_info), 200);
        assert_eq!(offset_of!(Info, sm_isa), 204);
        assert_eq!(offset_of!(Info, shader_pipe_count), 208);
        assert_eq!(offset_of!(Info, shader_pipe_sub_count), 212);
        assert_eq!(offset_of!(Info, generic_kind), 68);
        assert_eq!(offset_of!(Info, vram_size), 72);
        assert_eq!(offset_of!(Info, bar_size), 80);
        assert_eq!(offset_of!(Info, name), 88);
        assert_eq!(offset_of!(Allocation, cpu_address), 16);
        assert_eq!(size_of::<SyncPoint>(), 16);
    }
    #[test]
    fn allocation_failures_are_out_of_device_memory() {
        // vkAllocateMemory may return OUT_OF_HOST_MEMORY or OUT_OF_DEVICE_MEMORY
        // (not DEVICE_LOST); NVK reports every failed allocation as
        // OUT_OF_DEVICE_MEMORY (nvkmd_nouveau_mem.c:169).
        let alloc = |error: Error| Entry::Alloc.status(&error);
        let rm = |status| Error::Rm {
            operation: "op",
            status,
        };
        assert_eq!(alloc(rm(0x51)), -2);
        assert_eq!(alloc(rm(0x1a)), -2);
        assert_eq!(alloc(rm(0x1f)), -2);
        assert_eq!(alloc(rm(0x0f)), -2);
        assert_eq!(alloc(Error::DeviceLost("x".into())), -2);
        assert_eq!(
            alloc(
                MemoryError::Invalid {
                    operation: "allocate memory",
                    reason: "size exceeds the address space",
                }
                .into()
            ),
            -2
        );
        assert_eq!(
            alloc(Error::Io("op", std::io::Error::from_raw_os_error(12))),
            -1
        );
        assert_eq!(alloc(Error::Contract("null allocation output".into())), -13);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let status = guarded_as(Entry::Alloc, "test_entry", None, || panic!("injected"));
        std::panic::set_hook(hook);
        assert_eq!(status, -2);
    }
    #[test]
    fn import_failures_follow_nvk() {
        // A dma-buf the kernel or RM does not import is INVALID_EXTERNAL_HANDLE
        // (nvkmd_nouveau_mem.c:188); vkAllocateMemory has no DEVICE_LOST.
        let import = |error: Error| Entry::Import.status(&error);
        let handle = || {
            Error::from(MemoryError::ExternalHandle {
                reason: "x",
                value: 1,
            })
        };
        assert_eq!(import(handle()), -1_000_072_003);
        assert_eq!(
            import(Error::Rm {
                operation: "op",
                status: 0x1f
            }),
            -1_000_072_003
        );
        assert_eq!(
            import(Error::Io("op", std::io::Error::from_raw_os_error(9))),
            -1_000_072_003
        );
        assert_eq!(
            import(Error::Io("op", std::io::Error::from_raw_os_error(12))),
            -1
        );
        assert_eq!(
            import(Error::Rm {
                operation: "op",
                status: 0x51
            }),
            -2
        );
        assert_eq!(import(Error::DeviceLost("x".into())), -2);
        assert_eq!(import(Error::Contract("x".into())), -13);
        // SAFETY: a NULL device is rejected before any pointer is used.
        let status = unsafe { nv_rm_import(std::ptr::null_mut(), 0, -1, std::ptr::null_mut()) };
        assert_eq!(status, -13, "a NULL output is a broken bridge contract");
    }
    #[test]
    fn export_failures_follow_nvk() {
        // vkGetMemoryFdKHR may return TOO_MANY_OBJECTS or OUT_OF_HOST_MEMORY;
        // NVK reports every failed export as TOO_MANY_OBJECTS
        // (nvkmd_nouveau_mem.c:293).
        let export = |error: Error| Entry::Export.status(&error);
        for error in [
            Error::Rm {
                operation: "op",
                status: 0x1f,
            },
            Error::Rm {
                operation: "op",
                status: 0x51,
            },
            Error::Io("op", std::io::Error::from_raw_os_error(24)),
            Error::DeviceLost("x".into()),
        ] {
            assert_eq!(export(error), -10);
        }
        assert_eq!(
            export(Error::Io("op", std::io::Error::from_raw_os_error(12))),
            -1
        );
        assert_eq!(export(Error::Contract("x".into())), -13);
    }
    #[test]
    fn map_failures_follow_nvk() {
        // vkMapMemory may return OUT_OF_HOST_MEMORY, OUT_OF_DEVICE_MEMORY or
        // MEMORY_MAP_FAILED (not DEVICE_LOST); NVK reports a failed CPU map as
        // MEMORY_MAP_FAILED (nvkmd_nouveau_mem.c:247).
        let map = |error: Error| Entry::Map.status(&error);
        for error in [
            Error::Rm {
                operation: "RM map memory",
                status: 0x51,
            },
            Error::Rm {
                operation: "RM map memory",
                status: 0x1f,
            },
            MemoryError::BarPressure {
                used: 1,
                requested: 2,
                limit: 2,
            }
            .into(),
            MemoryError::CleanupFailed.into(),
            Error::DeviceLost("x".into()),
            Error::Contract("x".into()),
        ] {
            assert_eq!(map(error), -5);
        }
        assert_eq!(
            map(Error::Io(
                "mmap NVIDIA allocation",
                std::io::Error::from_raw_os_error(12)
            )),
            -1
        );
    }
    #[test]
    fn bind_failures_follow_nvk() {
        // vkBind*Memory and vkAllocateMemory have no DEVICE_LOST; NVK reports a
        // failed VM bind as VK_ERROR_UNKNOWN (nvkmd_nouveau_va.c:109).
        let bind = |error: Error| Entry::Bind.status(&error);
        assert_eq!(bind(Error::DeviceLost("x".into())), -13);
        assert_eq!(
            bind(Error::Rm {
                operation: "op",
                status: 0x1f
            }),
            -13
        );
        assert_eq!(
            bind(
                MemoryError::GpuMappingLimits {
                    offset: 0,
                    length: ByteSize::new(1),
                    allocation_size: ByteSize::new(0),
                }
                .into()
            ),
            -13
        );
        assert_eq!(
            bind(Error::Rm {
                operation: "op",
                status: 0x51
            }),
            -2
        );
        assert_eq!(
            bind(Error::Io("op", std::io::Error::from_raw_os_error(12))),
            -1
        );
    }
    #[test]
    fn sync_failures_follow_mesa() {
        // vkCreateFence, vkSignalSemaphore, vkResetFences and the sync fd
        // import/export commands have no DEVICE_LOST; Mesa's DRM syncobj
        // payloads, which NVK uses, report their failures as VK_ERROR_UNKNOWN
        // (vk_drm_syncobj.c).
        let lost = || Error::from(SyncError::Closed("x"));
        let oom = || Error::Rm {
            operation: "op",
            status: 0x51,
        };
        let refused = || Error::from(SyncError::GenerationExhausted { operation: "x" });
        assert_eq!(Entry::Sync.status(&lost()), -13);
        assert_eq!(Entry::Sync.status(&refused()), -13);
        assert_eq!(Entry::Sync.status(&oom()), -2);
        assert_eq!(Entry::SyncFd.status(&lost()), -13);
        assert_eq!(Entry::SyncFd.status(&refused()), -13);
        assert_eq!(Entry::SyncFd.status(&oom()), -13);
        let host = || Error::Io("op", std::io::Error::from_raw_os_error(12));
        assert_eq!(Entry::Sync.status(&host()), -1);
        assert_eq!(Entry::SyncFd.status(&host()), -1);
    }
    #[test]
    fn queue_creation_keeps_initialization_failed() {
        // Contexts are created inside vkCreateDevice, which lists
        // INITIALIZATION_FAILED and DEVICE_LOST.
        let create = |error: Error| Entry::Create.status(&error);
        assert_eq!(create(Error::Contract("x".into())), -3);
        assert_eq!(create(ChannelError::EngineMask { mask: 0x40 }.into()), -3);
        assert_eq!(create(ChannelError::Stopped.into()), -4);
        assert_eq!(
            create(
                ChannelError::Allocate {
                    class: ClassId::new(0xc96f),
                    source: Box::new(Error::Rm {
                        operation: "op",
                        status: 0x51,
                    }),
                }
                .into()
            ),
            -2
        );
    }
    #[test]
    fn probing_reports_no_device_loss() {
        // vkEnumeratePhysicalDevices may not return VK_ERROR_DEVICE_LOST; a GPU
        // that cannot be opened is VK_ERROR_INITIALIZATION_FAILED, as NVK
        // reports it (nvkmd_nouveau_dev.c:44).
        let open = |error: Error| Entry::Open.status(&error);
        assert_eq!(open(Error::DeviceLost("x".into())), -3);
        assert_eq!(
            open(Error::Rm {
                operation: "op",
                status: 0x0f
            }),
            -3
        );
        assert_eq!(open(ChannelError::Stopped.into()), -3);
        assert_eq!(open(Error::Contract("x".into())), -3);
        assert_eq!(
            open(Error::Io("op", std::io::Error::from_raw_os_error(12))),
            -1
        );
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let status = guarded_as(Entry::Open, "test_entry", None, || panic!("injected"));
        std::panic::set_hook(hook);
        assert_eq!(status, -3);
    }
    #[test]
    fn caller_panic_becomes_device_lost_instead_of_unwinding_into_c() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let status = guarded("test_entry", None, || panic!("injected backend panic"));
        std::panic::set_hook(hook);
        assert_eq!(status, -4);
        assert_eq!(guarded("test_entry", None, || 7), 7);
    }
    #[test]
    fn null_and_invalid_handles_fail_without_touching_state() {
        // SAFETY: NULL is a valid device argument (nv_rm.h).
        assert_eq!(unsafe { nv_rm_is_lost(std::ptr::null()) }, 1);
        // SAFETY: as above.
        unsafe { nv_rm_mark_lost(std::ptr::null()) };
        assert_ne!(
            // SAFETY: a NULL device is rejected before the id is used.
            unsafe { nv_rm_sync_destroy_deferred(std::ptr::null_mut(), NvRmSyncId::new(7)) },
            0
        );
        // SAFETY: as above.
        assert_ne!(
            unsafe { nv_rm_sync_reset(std::ptr::null_mut(), NvRmSyncId::new(7)) },
            0
        );
        // SAFETY: id 0 is rejected before any dereference.
        assert!(unsafe { sync_ref(NvRmSyncId::new(0)) }.is_err());
    }
}
