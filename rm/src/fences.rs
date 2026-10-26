// SPDX-License-Identifier: MIT
//! Native GPU timeline fences, using public RM semaphore-surface + DRM APIs.
//!
//! Sources: cl00da.h, ctrl00da.h, ctrl2080fb.h, nv_drm_common_ioctl.h,
//! nvkms-kapi-sync.c and nvidia-drm-fence.c in matching open kernel modules.
//! The queue must drain before dropping this object; a sync_file retains its
//! dma_fence, but destroying its DRM context errors outstanding fences.
use crate::diag::diag;
use crate::{
    gpu::{AddressSpace, GpuMapping},
    rm::{Device, Error, Mapping, Result},
};
use std::{
    cell::RefCell,
    io,
    mem::size_of,
    os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

#[repr(C, align(8))]
#[derive(Default)]
struct Layout {
    max_submitted: u64,
    monitored: u64,
    stride: u64,
    caps: u32,
    padding: u32,
}
#[repr(C, align(8))]
struct Allocate {
    memory: u32,
    max_submitted_memory: u32,
    flags: u64,
}
#[repr(C)]
struct Bind {
    channel: u32,
    count: u32,
    indices: [u32; 8],
}
#[repr(C, align(8))]
struct Import {
    client: u32,
    surface: u32,
    size: u64,
}
#[repr(C, align(8))]
struct ContextCreate {
    index: u64,
    parameters: u64,
    parameters_size: u64,
    handle: u32,
    padding: u32,
}
#[repr(C, align(8))]
struct FenceCreate {
    context: u32,
    timeout_ms: u32,
    value: u64,
    fd: i32,
    padding: u32,
}
#[cfg(test)]
#[repr(C, align(8))]
struct FenceWait {
    context: u32,
    fd: i32,
    pre: u64,
    post: u64,
}
#[repr(C)]
struct GemClose {
    handle: u32,
    padding: u32,
}
#[repr(C, align(8))]
#[derive(Default)]
struct SyncFileInfo {
    name: [u8; 32],
    status: i32,
    flags: u32,
    count: u32,
    padding: u32,
    entries: u64,
}
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
struct SyncFenceInfo {
    object_name: [u8; 32],
    driver_name: [u8; 32],
    status: i32,
    flags: u32,
    timestamp_ns: u64,
}

const fn request(group: u8, number: u8, bytes: usize, direction: u32) -> u32 {
    (direction << 30) | ((bytes as u32) << 16) | ((group as u32) << 8) | number as u32
}
fn call<T>(
    fd: RawFd,
    group: u8,
    number: u8,
    direction: u32,
    value: &mut T,
    operation: &'static str,
) -> Result<()> {
    // SAFETY: all private callers supply exact initialized public wire structs.
    // Nested pointers remain borrowed until this synchronous ioctl returns.
    let ret = unsafe {
        crate::platform::ioctl(
            fd,
            request(group, number, size_of::<T>(), direction),
            (value as *mut T).cast(),
        )
    };
    if ret < 0 {
        Err(Error::Io(operation, io::Error::last_os_error()))
    } else {
        Ok(())
    }
}
pub(crate) fn sync_status(fd: RawFd) -> Result<i32> {
    sync_status_with(|info, _entries| call(fd, b'>', 4, 3, info, "query sync_file status"))
}

// Linux v4.19 sync_file_ioctl_fence_info's zero-count branch reports only
// dma_fence_is_signaled, hiding errors. v5.10/v6.18 use get_status; verify
// positive summaries using children without guessing which kernel is installed.
// Follow public libsync's two-pass FILE_INFO protocol, without allocating or
// querying children for active/failed summaries. The owned sync_file generation
// is immutable; modern chain unwrapping may remove completed predecessors, so
// accept a smaller returned count, never an oversized or empty result.
fn sync_status_with(
    mut query: impl FnMut(&mut SyncFileInfo, &mut [SyncFenceInfo]) -> Result<()>,
) -> Result<i32> {
    let mut info = SyncFileInfo::default();
    query(&mut info, &mut [])?;
    if info.status <= 0 {
        return Ok(info.status);
    }
    let count = info.count;
    // sync_file_ioctl_fence_info computes its allocation size in __u32.
    // Reject wire-size overflow rather than imposing an arbitrary fence limit.
    let bytes = count.checked_mul(size_of::<SyncFenceInfo>() as u32);
    if count == 0
        || bytes
            .is_none_or(|bytes| usize::try_from(bytes).map_or(true, |n| n > isize::MAX as usize))
    {
        return Err(Error::Contract(format!(
            "sync_file fence count {count} is empty or exceeds kernel/Rust allocation byte limits"
        )));
    }
    let mut inspect = |entries: &mut [SyncFenceInfo]| -> Result<i32> {
        // Reset all input/reserved fields, retaining only the advertised bound.
        let mut details = SyncFileInfo {
            count,
            entries: entries.as_mut_ptr() as u64,
            ..Default::default()
        };
        query(&mut details, entries)?;
        if details.count == 0 || details.count > count {
            return Err(Error::Contract(format!(
                "sync_file fence count changed outside supplied capacity: {count} -> {}",
                details.count
            )));
        }
        let entries = &entries[..details.count as usize];
        // The kernel's full-query summary stops at its first nonpositive child.
        // Inspect every entry so a preceding pending child cannot hide an error.
        if details.status < 0 {
            return Ok(details.status);
        }
        if let Some(failed) = entries.iter().find(|entry| entry.status < 0) {
            return Ok(failed.status);
        }
        Ok(i32::from(
            details.status > 0 && entries.iter().all(|entry| entry.status > 0),
        ))
    };
    if count == 1 {
        return inspect(&mut [SyncFenceInfo::default()]);
    }
    let mut entries = Vec::new();
    entries.try_reserve_exact(count as usize).map_err(|error| {
        Error::Contract(format!("allocate sync_file fence information: {error}"))
    })?;
    entries.resize(count as usize, SyncFenceInfo::default());
    inspect(&mut entries)
}

struct DrmContext {
    fd: OwnedFd,
    handle: u32,
}
impl Drop for DrmContext {
    fn drop(&mut self) {
        let mut close = GemClose {
            handle: self.handle,
            padding: 0,
        };
        if let Err(e) = call(
            self.fd.as_raw_fd(),
            b'd',
            9,
            1,
            &mut close,
            "close DRM semaphore context",
        ) {
            diag!("fence context cleanup: {e}");
        }
    }
}

/// One monotonic CPU64 timeline backed by the queried native64 or sem32 mode.
/// Sem32 reconstruction follows sem_surf.c::_semsurfGetValue exactly. At most
/// 2^31-1 values may be outstanding, enforced by Queue before publishing work.
pub(crate) struct NativeFenceSurface {
    context: RefCell<Option<DrmContext>>,
    #[cfg(test)]
    health: RefCell<Vec<OwnedFd>>,
    device: Rc<Device>,
    handle: u32,
    channel: Option<u32>,
    mapping: Mapping,
    gpu: GpuMapping,
    size: u64,
    mode: PayloadMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayloadMode {
    Native64,
    Tracked32 { maximum_offset: usize },
}

fn payload_mode(layout: &Layout) -> Result<PayloadMode> {
    let stride = layout.stride;
    if stride < 16 || !stride.is_multiple_of(8) || stride > 65536 {
        return Err(Error::Contract("invalid semaphore surface stride".into()));
    }
    let valid_slot = |offset: u64| {
        offset >= 16
            && offset.is_multiple_of(8)
            && offset.checked_add(8).is_some_and(|end| end <= stride)
    };
    if layout.caps & 1 != 0 && !valid_slot(layout.monitored) {
        return Err(Error::Contract("invalid monitored semaphore layout".into()));
    }
    if layout.caps & 2 != 0 {
        return Ok(PayloadMode::Native64);
    }
    if !valid_slot(layout.max_submitted)
        || (layout.caps & 1 != 0 && layout.max_submitted == layout.monitored)
    {
        return Err(Error::Contract("invalid sem32 maxSubmitted slot".into()));
    }
    Ok(PayloadMode::Tracked32 {
        maximum_offset: layout.max_submitted as usize,
    })
}

#[cfg(feature = "diagnostics")]
pub(crate) fn query_layout(device: &Device) -> Result<(u64, u64, u64, u32)> {
    let mut layout = Layout::default();
    device
        .client
        .control(device.subdevice, 0x20801352, &mut layout)?;
    Ok((
        layout.max_submitted,
        layout.monitored,
        layout.stride,
        layout.caps,
    ))
}
#[cfg(feature = "diagnostics")]
pub(crate) fn layout_mode(raw: (u64, u64, u64, u32)) -> Result<String> {
    let layout = Layout {
        max_submitted: raw.0,
        monitored: raw.1,
        stride: raw.2,
        caps: raw.3,
        padding: 0,
    };
    Ok(match payload_mode(&layout)? {
        PayloadMode::Native64 => "native64".into(),
        PayloadMode::Tracked32 { maximum_offset } => format!("tracked32@{maximum_offset}"),
    })
}

/// Public RM reconstruction contract; the producer publishes maximum first.
pub(crate) fn extend_payload(low: u32, maximum: u64) -> Result<u64> {
    let epoch = maximum & !u64::from(u32::MAX);
    let epoch = if low > maximum as u32 {
        epoch
            .checked_sub(1 << 32)
            .ok_or_else(|| Error::Contract("sem32 payload precedes timeline origin".into()))?
    } else {
        epoch
    };
    let value = epoch | u64::from(low);
    if maximum - value > 0x7fff_ffff {
        return Err(Error::Contract(
            "sem32 outstanding distance exceeds half range".into(),
        ));
    }
    Ok(value)
}
pub(crate) fn validate_compat(compat: crate::compat::Compat) -> Result<()> {
    // Freeze all fixed DRM/private payload contracts before creating RM
    // state. Missing legacy fence support is a named research gate.
    for name in [
        "drm_nvidia_semsurf_fence_ctx_create_params",
        "drm_nvidia_semsurf_fence_create_params",
        "drm_nvidia_semsurf_fence_wait_params",
        "NvKmsKapiPrivImportSemaphoreSurfaceParams",
    ] {
        compat.require_same_layout(name)?;
    }
    for (name, value) in [
        ("DRM_NVIDIA_SEMSURF_FENCE_CTX_CREATE", 0x14),
        ("DRM_NVIDIA_SEMSURF_FENCE_CREATE", 0x15),
        ("DRM_NVIDIA_SEMSURF_FENCE_WAIT", 0x16),
    ] {
        compat.require_constant(name, value)?;
    }
    Ok(())
}

impl NativeFenceSurface {
    fn allocate(device: &Rc<Device>, space: &AddressSpace) -> Result<Self> {
        validate_compat(device.client.compat)?;
        let mut layout = Layout::default();
        device
            .client
            .control(device.subdevice, 0x20801352, &mut layout)?;
        let mode = payload_mode(&layout)?;
        let requested = usize::try_from(layout.stride)
            .map_err(|_| Error::Contract("semaphore stride exceeds address space".into()))?;
        let memory = device.allocate_aligned(requested, 4096, false, true)?;
        let mut mapping = memory.map()?;
        for offset in (0..memory.size).step_by(4) {
            mapping.write_u32(offset, 0)?;
        }
        crate::platform::device_full_barrier();
        let gpu = space.map(&memory)?;
        let mut alloc = Allocate {
            memory: memory.handle,
            max_submitted_memory: if mode == PayloadMode::Native64 {
                0
            } else {
                memory.handle
            },
            flags: 0,
        };
        let handle = device.client.allocate(device.subdevice, 0xda, &mut alloc)?;
        // RM initializes the monitored threshold to UINT64_MAX. From this
        // point only RM owns that word; GPU writes payload + timestamp [0,16).
        Ok(Self {
            context: RefCell::new(None),
            #[cfg(test)]
            health: RefCell::new(Vec::new()),
            device: Rc::clone(device),
            handle,
            channel: None,
            mapping,
            gpu,
            size: memory.size as u64,
            mode,
        })
    }

    pub(crate) fn new(device: &Rc<Device>, space: &AddressSpace, channel: u32) -> Result<Self> {
        let mut surface = Self::allocate(device, space)?;
        // Old constructors register engine events themselves. Do not invoke
        // the known unsupported BIND stub or later UNBIND on that contract.
        if device.client.compat.supports_semsurf_bind() {
            let mut bind = Self::binding(channel);
            device.client.control(surface.handle, 0xda0002, &mut bind)?;
            surface.channel = Some(channel);
        }
        Ok(surface)
    }
    fn binding(channel: u32) -> Bind {
        // cl2080_notification.h: GR0/GRAPHICS=12, FIFO_EVENT_MTHD=35.
        Bind {
            channel,
            count: 2,
            indices: [12, 35, 0, 0, 0, 0, 0, 0],
        }
    }
    pub(crate) fn address(&self) -> u64 {
        self.gpu.address()
    }
    pub(crate) fn native64(&self) -> bool {
        self.mode == PayloadMode::Native64
    }
    fn maximum(&self, offset: usize) -> Result<&AtomicU64> {
        let address = self
            .mapping
            .cpu_address()
            .checked_add(offset)
            .ok_or_else(|| Error::Contract("semaphore CPU address overflow".into()))?;
        if address % std::mem::align_of::<AtomicU64>() != 0
            || offset
                .checked_add(8)
                .is_none_or(|end| end > self.size as usize)
        {
            return Err(Error::Contract(
                "unaligned/out-of-bounds cached semaphore atomic".into(),
            ));
        }
        // SAFETY: this word is in owned CPU-cached system memory, not MMIO or
        // GPU payload. RM and userspace both use aligned64 atomic operations;
        // GPU never writes this word. Mapping outlives the borrowed atomic.
        Ok(unsafe { &*(address as *const AtomicU64) })
    }
    pub(crate) fn publish_maximum(&self, token: u64) -> Result<()> {
        if let PayloadMode::Tracked32 { maximum_offset } = self.mode {
            self.maximum(maximum_offset)?
                .fetch_max(token, Ordering::SeqCst);
            crate::platform::device_write_barrier();
        }
        Ok(())
    }
    pub(crate) fn read_completed(&self) -> Result<u64> {
        crate::platform::device_read_barrier();
        if let PayloadMode::Tracked32 { maximum_offset } = self.mode {
            // The low payload MUST be read before the maxSubmitted snapshot.
            let low = self.mapping.read_u32(0)?;
            crate::platform::device_read_barrier();
            return extend_payload(low, self.maximum(maximum_offset)?.load(Ordering::Acquire));
        }
        // Works on LP32 without a tearing64 CPU load. Values never wrap.
        loop {
            let high = self.mapping.read_u32(4)?;
            let low = self.mapping.read_u32(0)?;
            if high == self.mapping.read_u32(4)? {
                crate::platform::device_read_barrier();
                return Ok((u64::from(high) << 32) | u64::from(low));
            }
        }
    }
    fn with_context<T>(
        &self,
        drm_fd: RawFd,
        action: impl FnOnce(&DrmContext) -> Result<T>,
    ) -> Result<T> {
        if drm_fd < 0 {
            return Err(Error::Contract(
                "invalid DRM fd for semaphore timeline".into(),
            ));
        }
        let mut cached = self.context.borrow_mut();
        if cached.is_none() {
            // SAFETY: C/Rust caller lends this live DRM fd for the duration of
            // the call. Duplicate it so its GEM namespace outlives the caller.
            let fd = unsafe { BorrowedFd::borrow_raw(drm_fd) }
                .try_clone_to_owned()
                .map_err(|e| Error::Io("retain DRM semaphore context fd", e))?;
            let params = Import {
                client: self.device.client.root,
                surface: self.handle,
                size: self.size,
            };
            let mut create = ContextCreate {
                index: 0,
                parameters: &params as *const Import as u64,
                parameters_size: size_of::<Import>() as u64,
                handle: 0,
                padding: 0,
            };
            call(
                fd.as_raw_fd(),
                b'd',
                0x54,
                3,
                &mut create,
                "create DRM semaphore fence context",
            )?;
            *cached = Some(DrmContext {
                fd,
                handle: create.handle,
            });
        }
        action(cached.as_ref().expect("DRM context initialized above"))
    }
    /// Legacy kernel-WAIT diagnostic only. Production owns cancellable gates
    /// in the RM worker and never uses exported status as queue health.
    #[cfg(test)]
    pub(crate) fn check_errors(&self) -> Result<()> {
        let mut pending = self.health.borrow_mut();
        let mut index = 0;
        while index < pending.len() {
            match sync_status(pending[index].as_raw_fd())? {
                0 => index += 1,
                status if status > 0 => {
                    pending.swap_remove(index);
                }
                status => {
                    return Err(Error::Contract(format!(
                        "GPU timeline sync_file failed with producer status {status}"
                    )));
                }
            }
        }
        Ok(())
    }
    /// Returns an owned CLOEXEC sync_file. Payload may still be unsignalled.
    /// The matching kernel caps its lifetime at five seconds; zero does not
    /// mean infinite. This is an external fence deadline, not GPU retirement.
    pub(crate) fn context_fence(&self, drm_fd: RawFd, token: u64) -> Result<OwnedFd> {
        if token == 0 {
            return Err(Error::Contract(
                "zero is not a submitted fence token".into(),
            ));
        }
        self.with_context(drm_fd, |ctx| {
            let mut create = FenceCreate {
                context: ctx.handle,
                timeout_ms: 5000,
                value: token,
                fd: -1,
                padding: 0,
            };
            call(
                ctx.fd.as_raw_fd(),
                b'd',
                0x55,
                3,
                &mut create,
                "export GPU semaphore sync_file",
            )?;
            if create.fd < 0 {
                return Err(Error::Contract(
                    "DRM returned invalid semaphore sync_file".into(),
                ));
            }
            // SAFETY: successful CREATE transfers a newly installed fd.
            let fd = unsafe { OwnedFd::from_raw_fd(create.fd) };
            Ok(fd)
        })
    }
    /// Kernel callback advances this timeline to post only after both the
    /// external sync_file and the pre value complete. Input fd stays borrowed.
    #[cfg(test)]
    pub(crate) fn install_wait(
        &self,
        drm_fd: RawFd,
        sync_fd: RawFd,
        pre: u64,
        post: u64,
    ) -> Result<()> {
        self.check_errors()?;
        if pre >= post {
            return Err(Error::Contract("non-monotonic semaphore wait".into()));
        }
        // The matching DRM ioctl incorrectly returns 0 for some bad inputs.
        // Validate the sync_file independently; kernel allocation failures
        // remain detectable only by the queue's completion/error timeout.
        if sync_status(sync_fd)? < 0 {
            return Err(Error::Contract(
                "external sync_file carries producer error".into(),
            ));
        }
        // Retain before installing: an allocation failure must not leave an
        // accepted dependency whose eventual error we cannot observe.
        let retained = unsafe { BorrowedFd::borrow_raw(sync_fd) }
            .try_clone_to_owned()
            .map_err(|e| Error::Io("retain external sync_file health", e))?;
        self.with_context(drm_fd, |ctx| {
            let mut wait = FenceWait {
                context: ctx.handle,
                fd: sync_fd,
                pre,
                post,
            };
            call(
                ctx.fd.as_raw_fd(),
                b'd',
                0x56,
                1,
                &mut wait,
                "install GPU external semaphore wait",
            )
        })?;
        self.health.borrow_mut().push(retained);
        Ok(())
    }
    /// Must run before the bound channel is freed, after queue drain.
    pub(crate) fn unbind(&mut self) -> Result<()> {
        if let Some(channel) = self.channel {
            let mut bind = Self::binding(channel);
            self.device
                .client
                .control(self.handle, 0xda0006, &mut bind)?;
            self.channel = None;
        }
        Ok(())
    }
}
impl Drop for NativeFenceSurface {
    fn drop(&mut self) {
        // Stop DRM notifications while the original surface/channel still exist.
        // The kernel waits for imported external callbacks here; it has no
        // cancellation ABI for a foreign fence that never signals.
        drop(self.context.get_mut().take());
        if let Err(e) = self.unbind() {
            diag!("semaphore channel unbind: {e}");
        }
        if let Err(e) = self.device.client.free(self.device.subdevice, self.handle) {
            diag!("semaphore surface cleanup: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        ffi::c_int,
        fs::{File, OpenOptions},
        io::Read,
        process::Command,
        thread,
        time::{Duration, Instant},
    };
    #[repr(C)]
    struct PollFd {
        fd: c_int,
        events: i16,
        revents: i16,
    }
    #[cfg(target_os = "android")]
    type Nfds = u32;
    #[cfg(not(target_os = "android"))]
    type Nfds = std::ffi::c_ulong;
    unsafe extern "C" {
        fn poll(fds: *mut PollFd, count: Nfds, timeout: c_int) -> c_int;
        fn pipe2(fds: *mut c_int, flags: c_int) -> c_int;
        fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    }
    fn poll_readable(fd: RawFd, timeout: i32) -> bool {
        let mut p = PollFd {
            fd,
            events: 1,
            revents: 0,
        };
        // SAFETY: one writable pollfd; fd remains owned by the caller.
        let ret = unsafe { poll(&mut p, 1, timeout) };
        assert!(ret >= 0, "poll: {}", io::Error::last_os_error());
        ret != 0 && p.revents & 1 != 0
    }
    #[test]
    fn nested_and_linux_uapi_fields_match_supplemental_c_oracle() {
        // build/gpu-coverage/cp2-nested-contracts.c: independent public C
        // offsets and widths, LP64 output plus ILP32 static assertions.
        assert_eq!(
            (std::mem::offset_of!(GemClose, handle), size_of::<u32>()),
            (0, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(GemClose, padding), size_of::<u32>()),
            (4, 4)
        );
        assert_eq!(
            (
                std::mem::offset_of!(SyncFileInfo, name),
                size_of::<[u8; 32]>()
            ),
            (0, 32)
        );
        assert_eq!(
            (std::mem::offset_of!(SyncFileInfo, status), size_of::<i32>()),
            (32, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(SyncFileInfo, flags), size_of::<u32>()),
            (36, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(SyncFileInfo, count), size_of::<u32>()),
            (40, 4)
        );
        assert_eq!(
            (
                std::mem::offset_of!(SyncFileInfo, padding),
                size_of::<u32>()
            ),
            (44, 4)
        );
        assert_eq!(
            (
                std::mem::offset_of!(SyncFileInfo, entries),
                size_of::<u64>()
            ),
            (48, 8)
        );
    }
    #[test]
    fn actual_private_fields_match_public_census() {
        crate::compat::assert_logical_fields(
            "NV2080_CTRL_FB_GET_SEMAPHORE_SURFACE_LAYOUT_PARAMS",
            size_of::<Layout>(),
            &[
                (
                    "maxSubmittedSemaphoreValueOffset",
                    std::mem::offset_of!(Layout, max_submitted),
                    size_of::<u64>(),
                ),
                (
                    "monitoredFenceThresholdOffset",
                    std::mem::offset_of!(Layout, monitored),
                    size_of::<u64>(),
                ),
                (
                    "size",
                    std::mem::offset_of!(Layout, stride),
                    size_of::<u64>(),
                ),
                ("caps", std::mem::offset_of!(Layout, caps), size_of::<u32>()),
            ],
        );
        crate::compat::assert_logical_fields(
            "NV_SEMAPHORE_SURFACE_ALLOC_PARAMETERS",
            size_of::<Allocate>(),
            &[
                (
                    "hSemaphoreMem",
                    std::mem::offset_of!(Allocate, memory),
                    size_of::<u32>(),
                ),
                (
                    "hMaxSubmittedMem",
                    std::mem::offset_of!(Allocate, max_submitted_memory),
                    size_of::<u32>(),
                ),
                (
                    "flags",
                    std::mem::offset_of!(Allocate, flags),
                    size_of::<u64>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "NV_SEMAPHORE_SURFACE_CTRL_BIND_CHANNEL_PARAMS",
            size_of::<Bind>(),
            &[
                (
                    "hChannel",
                    std::mem::offset_of!(Bind, channel),
                    size_of::<u32>(),
                ),
                (
                    "numNotifyIndices",
                    std::mem::offset_of!(Bind, count),
                    size_of::<u32>(),
                ),
                (
                    "notifyIndices",
                    std::mem::offset_of!(Bind, indices),
                    size_of::<[u32; 8]>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "NvKmsKapiPrivImportSemaphoreSurfaceParams",
            size_of::<Import>(),
            &[
                (
                    "hClient",
                    std::mem::offset_of!(Import, client),
                    size_of::<u32>(),
                ),
                (
                    "hSemaphoreSurface",
                    std::mem::offset_of!(Import, surface),
                    size_of::<u32>(),
                ),
                (
                    "semaphoreSurfaceSize",
                    std::mem::offset_of!(Import, size),
                    size_of::<u64>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "drm_nvidia_semsurf_fence_ctx_create_params",
            size_of::<ContextCreate>(),
            &[
                (
                    "index",
                    std::mem::offset_of!(ContextCreate, index),
                    size_of::<u64>(),
                ),
                (
                    "nvkms_params_ptr",
                    std::mem::offset_of!(ContextCreate, parameters),
                    size_of::<u64>(),
                ),
                (
                    "nvkms_params_size",
                    std::mem::offset_of!(ContextCreate, parameters_size),
                    size_of::<u64>(),
                ),
                (
                    "handle",
                    std::mem::offset_of!(ContextCreate, handle),
                    size_of::<u32>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "drm_nvidia_semsurf_fence_create_params",
            size_of::<FenceCreate>(),
            &[
                (
                    "fence_context_handle",
                    std::mem::offset_of!(FenceCreate, context),
                    size_of::<u32>(),
                ),
                (
                    "timeout_value_ms",
                    std::mem::offset_of!(FenceCreate, timeout_ms),
                    size_of::<u32>(),
                ),
                (
                    "wait_value",
                    std::mem::offset_of!(FenceCreate, value),
                    size_of::<u64>(),
                ),
                (
                    "fd",
                    std::mem::offset_of!(FenceCreate, fd),
                    size_of::<i32>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "drm_nvidia_semsurf_fence_wait_params",
            size_of::<FenceWait>(),
            &[
                (
                    "fence_context_handle",
                    std::mem::offset_of!(FenceWait, context),
                    size_of::<u32>(),
                ),
                ("fd", std::mem::offset_of!(FenceWait, fd), size_of::<i32>()),
                (
                    "pre_wait_value",
                    std::mem::offset_of!(FenceWait, pre),
                    size_of::<u64>(),
                ),
                (
                    "post_wait_value",
                    std::mem::offset_of!(FenceWait, post),
                    size_of::<u64>(),
                ),
            ],
        );
    }
    #[test]
    fn live_fence_contracts_pack_offline() {
        for c in crate::compat::measured_versions() {
            if validate_compat(c).is_err() {
                continue;
            }
            validate_compat(c).unwrap();
            for (name, value) in [
                ("LAYOUT_CAPS_MONITORED_FENCE_SUPPORTED", 1),
                ("LAYOUT_CAPS_64BIT_SEMAPHORES_SUPPORTED", 2),
                ("DRM_IOCTL_NVIDIA_SEMSURF_FENCE_CTX_CREATE", 0xc0206454),
                ("DRM_IOCTL_NVIDIA_SEMSURF_FENCE_CREATE", 0xc0186455),
                ("DRM_IOCTL_NVIDIA_SEMSURF_FENCE_WAIT", 0x40186456),
            ] {
                c.require_constant(name, value).unwrap();
            }
            if c.supports_semsurf_bind() {
                for (name, value) in [
                    ("NV2080_NOTIFIERS_GR0", 12),
                    ("NV2080_NOTIFIERS_FIFO_EVENT_MTHD", 35),
                    ("NV_SEMAPHORE_SURFACE_CTRL_CMD_BIND_CHANNEL_MAX_INDICES", 8),
                ] {
                    c.require_constant(name, value).unwrap();
                }
            }
            c.pack(c.control_layout(0x20801352).unwrap(), &Layout::default())
                .unwrap();
            c.pack(
                c.allocation_layout(0xda, size_of::<Allocate>())
                    .unwrap()
                    .unwrap(),
                &Allocate {
                    memory: 1,
                    max_submitted_memory: 1,
                    flags: 0,
                },
            )
            .unwrap();
            if c.supports_semsurf_bind() {
                for command in [0xda0002, 0xda0006] {
                    c.pack(
                        c.control_layout(command).unwrap(),
                        &Bind {
                            channel: 1,
                            count: 1,
                            indices: [0; 8],
                        },
                    )
                    .unwrap();
                }
            } else {
                assert!(c.control_layout(0xda0002).is_err());
                assert!(c.control_layout(0xda0006).is_err());
            }
            c.pack(
                "NvKmsKapiPrivImportSemaphoreSurfaceParams",
                &Import {
                    client: 1,
                    surface: 2,
                    size: 4096,
                },
            )
            .unwrap();
            c.pack(
                "drm_nvidia_semsurf_fence_ctx_create_params",
                &ContextCreate {
                    index: 0,
                    parameters: 0,
                    parameters_size: 16,
                    handle: 0,
                    padding: 0,
                },
            )
            .unwrap();
            c.pack(
                "drm_nvidia_semsurf_fence_create_params",
                &FenceCreate {
                    context: 1,
                    timeout_ms: 5000,
                    value: 1,
                    fd: -1,
                    padding: 0,
                },
            )
            .unwrap();
            c.pack(
                "drm_nvidia_semsurf_fence_wait_params",
                &FenceWait {
                    context: 1,
                    fd: 5,
                    pre: 1,
                    post: 2,
                },
            )
            .unwrap();
        }
    }
    fn set_value(surface: &NativeFenceSurface, value: u64) {
        #[repr(C, align(8))]
        struct Set {
            index: u64,
            value: u64,
        }
        surface
            .device
            .client
            .control(surface.handle, 0xda0004, &mut Set { index: 0, value })
            .unwrap();
    }
    fn device_and_drm() -> (Rc<Device>, File) {
        let client = crate::rm::Client::open().unwrap();
        // Multi-GPU container hosts list every GPU but grant one
        // /dev/nvidiaN and its render node: use the first card we can open.
        let card = client
            .cards()
            .unwrap()
            .into_iter()
            .find(|c| {
                c.valid != 0
                    && c.pci.vendor == 0x10de
                    && OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(format!("/dev/nvidia{}", c.minor))
                        .is_ok()
            })
            .unwrap();
        let pci = format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            card.pci.domain, card.pci.bus, card.pci.slot, card.pci.function
        );
        for e in std::fs::read_dir("/sys/class/drm").unwrap() {
            let e = e.unwrap();
            if !e.file_name().to_string_lossy().starts_with("renderD") {
                continue;
            }
            let device = std::fs::canonicalize(e.path().join("device")).unwrap();
            if device.file_name().unwrap().to_string_lossy() != pci {
                continue;
            }
            let fd = OpenOptions::new()
                .read(true)
                .write(true)
                .open(std::path::Path::new("/dev/dri").join(e.file_name()))
                .unwrap();
            return (client.device(card).unwrap(), fd);
        }
        panic!("no NVIDIA DRM render device at {pci}");
    }
    fn fd_count() -> usize {
        std::fs::read_dir("/proc/self/fd").unwrap().count()
    }

    #[test]
    fn sem32_rollover_and_corrupt_distance() {
        assert_eq!(
            extend_payload(0xffff_fffe, 0x1_0000_0002).unwrap(),
            0xffff_fffe
        );
        assert_eq!(extend_payload(2, 0x1_0000_0002).unwrap(), 0x1_0000_0002);
        assert!(extend_payload(1, 0x8000_0001).is_err());
        assert!(extend_payload(7, 6).is_err());
        assert_eq!(extend_payload(0xffff_ffff, u64::MAX).unwrap(), u64::MAX);
    }
    #[test]
    fn independent_caps_and_nonoverlapping_layout() {
        for caps in 0..4 {
            let mut l = Layout {
                stride: 32,
                monitored: 16,
                max_submitted: 24,
                caps,
                padding: 0,
            };
            assert_eq!(
                payload_mode(&l).unwrap() == PayloadMode::Native64,
                caps & 2 != 0
            );
            l.monitored = u64::MAX - 7;
            assert_eq!(payload_mode(&l).is_err(), caps & 1 != 0);
        }
        let l = Layout {
            stride: 32,
            monitored: 16,
            max_submitted: 16,
            caps: 1,
            padding: 0,
        };
        assert!(payload_mode(&l).is_err());
    }

    #[test]
    fn sync_status_pending_and_error_need_only_summary() {
        for status in [0, -5, -110] {
            let mut calls = 0;
            let result = sync_status_with(|info, entries| {
                calls += 1;
                assert_eq!(info.count, 0);
                assert_eq!(info.entries, 0);
                assert!(entries.is_empty());
                info.status = status;
                info.count = u32::MAX; // Not an allocation request on this path.
                Ok(())
            });
            assert_eq!(result.unwrap(), status);
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn sync_status_legacy_positive_summary_cannot_hide_child_error() {
        for (statuses, expected) in [
            (vec![1], 1),
            (vec![-110], -110),
            (vec![1, -5], -5),
            (vec![0, -5, 1], -5),
            (vec![1, 0], 0),
            (vec![1, 1, 1], 1),
        ] {
            let mut calls = 0;
            let result = sync_status_with(|info, entries| {
                calls += 1;
                if calls == 1 {
                    assert!(entries.is_empty());
                    info.status = 1; // Legacy boolean summary, including failures.
                    info.count = statuses.len() as u32;
                } else {
                    assert_eq!(info.count as usize, statuses.len());
                    assert_eq!(info.entries, entries.as_mut_ptr() as u64);
                    assert_eq!((info.flags, info.padding), (0, 0));
                    // Exact old kernel aggregation: first nonpositive wins.
                    info.status = statuses.iter().copied().find(|s| *s <= 0).unwrap_or(1);
                    for (entry, status) in entries.iter_mut().zip(&statuses) {
                        entry.status = *status;
                    }
                }
                Ok(())
            });
            assert_eq!(result.unwrap(), expected, "children={statuses:?}");
            assert_eq!(calls, 2);
        }
    }

    #[test]
    fn sync_status_rejects_invalid_counts_before_allocating() {
        for count in [
            0,
            u32::MAX / size_of::<SyncFenceInfo>() as u32 + 1,
            u32::MAX,
        ] {
            let mut calls = 0;
            let result = sync_status_with(|info, _| {
                calls += 1;
                info.status = 1;
                info.count = count;
                Ok(())
            });
            assert!(matches!(result, Err(Error::Contract(_))));
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn sync_status_count_shrink_uses_only_returned_initialized_entries() {
        let mut calls = 0;
        assert_eq!(
            sync_status_with(|info, entries| {
                calls += 1;
                info.status = 1;
                if calls == 1 {
                    info.count = 3;
                } else {
                    assert_eq!(entries.len(), 3);
                    info.count = 1;
                    entries[0].status = 1;
                    // The unused initialized tail remains pending; it must not
                    // suppress success after chain predecessors are removed.
                    assert_eq!(entries[1].status, 0);
                }
                Ok(())
            })
            .unwrap(),
            1
        );
        assert_eq!(calls, 2);
    }

    #[test]
    fn sync_status_count_growth_empty_or_second_ioctl_error_never_succeeds() {
        for returned_count in [0, 2] {
            let mut calls = 0;
            let result = sync_status_with(|info, _| {
                calls += 1;
                info.status = 1;
                info.count = if calls == 1 { 1 } else { returned_count };
                Ok(())
            });
            assert!(matches!(result, Err(Error::Contract(_))));
        }
        let mut calls = 0;
        let result = sync_status_with(|info, _| {
            calls += 1;
            if calls == 2 {
                // Linux rejects a supplied array smaller than its count.
                return Err(Error::Io(
                    "query sync_file status",
                    io::Error::from_raw_os_error(22),
                ));
            }
            info.status = 1;
            info.count = 1;
            Ok(())
        });
        assert!(matches!(result, Err(Error::Io(_, _))));
        assert_eq!(calls, 2);
    }

    #[test]
    fn sync_status_summary_pending_stays_pending_despite_later_child_completion() {
        let mut calls = 0;
        assert_eq!(
            sync_status_with(|info, entries| {
                calls += 1;
                info.count = 1;
                info.status = i32::from(calls == 1);
                if calls == 2 {
                    entries[0].status = 1;
                }
                Ok(())
            })
            .unwrap(),
            0
        );
    }

    #[test]
    fn public_fence_wire_layouts() {
        assert_eq!(size_of::<Layout>(), 32);
        assert_eq!(size_of::<Allocate>(), 16);
        assert_eq!(size_of::<Bind>(), 40);
        assert_eq!(size_of::<Import>(), 16);
        assert_eq!(size_of::<ContextCreate>(), 32);
        assert_eq!(size_of::<FenceCreate>(), 24);
        assert_eq!(size_of::<FenceWait>(), 24);
        assert_eq!(size_of::<SyncFileInfo>(), 56);
        // Independent linux/sync_file.h C oracle, GNU32/64 + five NDK CPUs.
        // i686 C alignment is4; explicit Rust overalignment preserves the wire.
        assert_eq!(size_of::<SyncFenceInfo>(), 80);
        assert_eq!(std::mem::offset_of!(SyncFenceInfo, object_name), 0);
        assert_eq!(std::mem::offset_of!(SyncFenceInfo, driver_name), 32);
        assert_eq!(std::mem::offset_of!(SyncFenceInfo, status), 64);
        assert_eq!(std::mem::offset_of!(SyncFenceInfo, flags), 68);
        assert_eq!(std::mem::offset_of!(SyncFenceInfo, timestamp_ns), 72);
        assert_eq!(request(b'd', 0x54, 32, 3), 0xc0206454);
        assert_eq!(request(b'd', 0x55, 24, 3), 0xc0186455);
        assert_eq!(request(b'd', 0x56, 24, 1), 0x40186456);
        assert_eq!(request(b'>', 4, 56, 3), 0xc0383e04);
    }

    #[test]
    #[ignore = "requires selected NVIDIA RM/DRM; no GPU commands submitted"]
    fn pending_fence_and_kernel_wait() {
        let before = fd_count();
        {
            let (device, drm) = device_and_drm();
            let space = AddressSpace::new(&device).unwrap();
            let a = NativeFenceSurface::allocate(&device, &space).unwrap();
            let b = NativeFenceSurface::allocate(&device, &space).unwrap();
            let first = a.context_fence(drm.as_raw_fd(), 1).unwrap();
            let second = b.context_fence(drm.as_raw_fd(), 1).unwrap();
            assert_eq!(sync_status(first.as_raw_fd()).unwrap(), 0);
            assert_eq!(sync_status(second.as_raw_fd()).unwrap(), 0);
            assert!(!poll_readable(first.as_raw_fd(), 0));
            b.install_wait(drm.as_raw_fd(), first.as_raw_fd(), 0, 1)
                .unwrap();
            assert_eq!(sync_status(second.as_raw_fd()).unwrap(), 0);
            set_value(&a, 1);
            assert!(poll_readable(first.as_raw_fd(), 1000));
            assert!(poll_readable(second.as_raw_fd(), 1000));
            assert_eq!(sync_status(first.as_raw_fd()).unwrap(), 1);
            assert_eq!(sync_status(second.as_raw_fd()).unwrap(), 1);
            assert_eq!(b.read_completed().unwrap(), 1);
            // Native 64-bit values, not an accidentally working low-word path.
            let wide = 0x1_0000_0001;
            let third = a.context_fence(drm.as_raw_fd(), wide).unwrap();
            assert_eq!(sync_status(third.as_raw_fd()).unwrap(), 0);
            set_value(&a, wide);
            assert!(poll_readable(third.as_raw_fd(), 1000));
            assert_eq!(a.read_completed().unwrap(), wide);
            assert_eq!(sync_status(third.as_raw_fd()).unwrap(), 1);
            let cancelled = b.context_fence(drm.as_raw_fd(), 2).unwrap();
            drop(b);
            assert!(poll_readable(cancelled.as_raw_fd(), 1000));
            assert!(sync_status(cancelled.as_raw_fd()).unwrap() < 0);
        }
        assert_eq!(fd_count(), before);
        println!(
            "PASS: pending sync_files, kernel external wait, native64 value, teardown errors, fd{before}→{before}; no GPU commands"
        );
    }

    #[test]
    #[ignore = "coordinated external CPU producer fixture for public Vulkan async probe"]
    fn cpu_producer_fixture() {
        let executable =
            std::env::var("NV_FENCE_PROBE").expect("absolute NV_FENCE_PROBE executable");
        let driver = std::env::var("NV_FENCE_DRIVER").expect("absolute NV_FENCE_DRIVER DSO");
        assert!(executable.starts_with('/') && driver.starts_with('/'));
        let error_gate = std::env::var_os("NV_FENCE_ERROR_GATE").is_some();
        let conversion = std::env::var_os("NV_FENCE_CONVERT").is_some_and(|v| v != "0");
        let conversion_skipped;
        let before = fd_count();
        {
            let (device, drm) = device_and_drm();
            let space = AddressSpace::new(&device).unwrap();
            let surface = NativeFenceSurface::allocate(&device, &space).unwrap();
            let gate = surface.context_fence(drm.as_raw_fd(), 1).unwrap();
            assert_eq!(sync_status(gate.as_raw_fd()).unwrap(), 0);
            // Always release the external dependency if the child fails or the
            // parent assertion unwinds. This fixture never spins the GPU.
            struct Release<'a>(&'a NativeFenceSurface, bool);
            impl Drop for Release<'_> {
                fn drop(&mut self) {
                    if self.1 {
                        // Destroying an uncompleted DRM timeline errors its
                        // exported dma_fence; no fabricated fd or GPU fault.
                        drop(self.0.context.borrow_mut().take());
                    } else {
                        set_value(self.0, 1);
                    }
                }
            }
            let release = Release(&surface, error_gate);
            let mut raw = [-1; 2];
            // SAFETY: pipe2 writes exactly two owned descriptors on success.
            assert_eq!(unsafe { pipe2(raw.as_mut_ptr(), 0x80000) }, 0);
            let mut read_end = unsafe { File::from_raw_fd(raw[0]) };
            let write_end = unsafe { OwnedFd::from_raw_fd(raw[1]) };
            let inherited_gate = gate.try_clone().unwrap();
            for fd in [write_end.as_raw_fd(), inherited_gate.as_raw_fd()] {
                // SAFETY: F_SETFD(2) only changes this owned descriptor's flags.
                assert_eq!(unsafe { fcntl(fd, 2, 0) }, 0);
            }
            let linker = std::env::var("NV_FENCE_LINKER").ok();
            let mut command = if std::env::var_os("NV_FENCE_PID_NS").is_some() {
                let mut c = Command::new("unshare");
                c.args(["--user", "--map-root-user", "--pid", "--fork"]);
                if let Some(ref l) = linker {
                    c.arg(l);
                }
                c.arg(&executable);
                c
            } else if let Some(l) = linker {
                let mut c = Command::new(l);
                c.arg(&executable);
                c
            } else {
                Command::new(&executable)
            };
            if let Ok(path) = std::env::var("NV_FENCE_LIBRARY_PATH") {
                command.env("LD_LIBRARY_PATH", path);
            }
            command.args([
                "--driver",
                &driver,
                "--gate-fd",
                &inherited_gate.as_raw_fd().to_string(),
                "--release-fd",
                &write_end.as_raw_fd().to_string(),
            ]);
            if error_gate {
                command.arg("--error-gate");
            }
            // Every channel must be idle when RM frees it (an abort otherwise;
            // on bionic the backend's diagnostics only reach logcat).
            command.env("NV_RM_STRICT_TEARDOWN", "1");
            if std::env::var_os("NV_FENCE_ALLOW_STOCK").is_some() {
                command.arg("--allow-stock-driver");
            }
            let mut child = command.spawn().unwrap();
            drop(write_end);
            drop(inherited_gate);
            let ready = poll_readable(read_end.as_raw_fd(), 1900);
            let mut byte = [0];
            let released_by_probe = ready && read_end.read_exact(&mut byte).is_ok();
            drop(release);
            // Lost queues are disabled before their exported DRM fences reach
            // the kernel's fixed five-second error deadline. Allow teardown
            // to observe it; producer release remains bounded at 1.9 seconds.
            let deadline = Instant::now() + Duration::from_secs(if error_gate { 8 } else { 2 });
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                if Instant::now() >= deadline {
                    child.kill().unwrap();
                    let status = child.wait().unwrap();
                    panic!("async probe exceeded watchdog after releasing dependency: {status}");
                }
                thread::sleep(Duration::from_millis(5));
            };
            assert!(
                released_by_probe,
                "probe never requested dependency release before watchdog; child status: {status}"
            );
            conversion_skipped = conversion && !error_gate && status.code() == Some(77);
            assert!(
                status.success() || conversion_skipped,
                "async probe failed: {status}"
            );
            let gate_status = sync_status(gate.as_raw_fd()).unwrap();
            if error_gate {
                assert!(gate_status < 0);
            } else {
                assert_eq!(gate_status, 1);
            }
        }
        assert_eq!(fd_count(), before);
        if conversion_skipped {
            println!(
                "SKIP: cross-type fence conversion unsupported; producer release, state and fd cleanup verified, fd{before}→{before}"
            );
        } else {
            println!("PASS: independent CPU producer fixture, fd{before}→{before}");
        }
    }
}
