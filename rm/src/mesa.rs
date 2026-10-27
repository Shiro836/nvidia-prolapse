// SPDX-License-Identifier: MIT
//! Mesa nvkmd -> Rust RM bridge, executed entirely on the calling threads.
//! The C ABI is documented in mesa/nv_rm.h.
//!
//! There is no ordinary-path worker. Submission, status, reset, host signal,
//! CPU waits, export/import, retirement and loss checks run on the API caller
//! (stock 615 does every per-frame synchronization/submission operation on
//! the caller, as traced in the stock binary). The single
//! optional thread is the lazily started foreign-fence watcher (watcher.rs).
//!
//! Ownership model:
//! * `Core` (Arc, Send+Sync): loss flag, slot pool addresses, the waiter group,
//!   the watcher, the shared-payload broadcast port and the device RM event
//!   descriptor numbers. Plain data and atomics only.
//! * `Inner` (behind one Mutex): every Rc/RefCell RM owner (client, device,
//!   address space, channels, buffers, mappings, event sources, pool pages).
//!   See `Confined` for the Send argument. Lock order: Inner -> SyncObj state
//!   -> {Shared local, slot free list, watcher state, waiter condvar}. Nothing
//!   blocks on GPU progress or a foreign fence while holding Inner.
//! * Sync payloads (`SyncObj`, Arc): the C id is the raw Arc pointer; their
//!   mapped slots are read/written without Inner.
use crate::diag::diag;
use crate::{
    abi, capabilities,
    events::{HostListener, RmEvents, TriggerPort},
    gpu::{AddressSpace, GpuMapping, Push, Queue, QueueShared, SemaphoreOp, SubmitStatus},
    reclaim::Reclaim,
    rm::{Client, Device, Error, Mapping, Memory, Result, WaitError},
    slots::{PAGE_BYTES, Slot, SlotPool, TimelineView},
    syncobj::{Foreign, Shared, SignalPlan, SyncObj, WaitPlan},
    waiters::Waiters,
    watcher::Watcher,
};
use nv_rm_core::conv::{high_u32, low_u32, u64_from_usize, usize_from_u32};
use std::{
    collections::{HashMap, VecDeque},
    ffi::c_void,
    os::fd::{AsRawFd, BorrowedFd, IntoRawFd, OwnedFd, RawFd},
    rc::Rc,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Info {
    vendor_id: u32,
    device_id: u32,
    chipset: u32,
    sm: u32,
    pci_domain: u32,
    pci_bus: u32,
    pci_device: u32,
    pci_function: u32,
    gpc_count: u32,
    tpc_count: u32,
    mp_per_tpc: u32,
    max_warps_per_mp: u32,
    cls_copy: u32,
    cls_eng2d: u32,
    cls_eng3d: u32,
    cls_m2mf: u32,
    cls_compute: u32,
    generic_kind: u32,
    vram_size: u64,
    bar_size: u64,
    name: [u8; 64],
    /// GR_GET_ZCULL_INFO followed by the GRAPHICS_ZCULL context-switch buffer
    /// size and alignment: the layout of Mesa's nv_zcull_device_info
    /// (nv_device_info.h:17-30). Meaningful only when `has_zcull_info` is 1.
    zcull: abi::ZcullInfo,
    zcull_ctxsw_size: u32,
    zcull_ctxsw_align: u32,
    has_zcull_info: u32,
    /// RM's SM ISA code (NV2080_CTRL_GR_INFO_INDEX_SM_VERSION), from which
    /// `sm` is derived.
    sm_isa: u32,
    /// NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT and
    /// NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT (the TPC count, as
    /// OGKM nv_gpu_ops.c reads it), the inputs of the compute contexts'
    /// SET_SCG_CONTROL limits.
    shader_pipe_count: u32,
    shader_pipe_sub_count: u32,
    /// Contexts that can join another context's channel group
    /// (`Queue::new_in_group`): the group's subcontext count less the
    /// creator's VEID 0 from Volta on, its channel count less one before
    /// (query_info).
    max_shared_ctx: u32,
    _reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Allocation {
    id: u64,
    size: u64,
    cpu_address: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SyncPoint {
    sync: u64,
    value: u64,
}

/// Thread-shareable device state (see module doc).
pub(crate) struct Core {
    pub(crate) pool: Arc<SlotPool>,
    pub(crate) waiters: Waiters,
    pub(crate) watcher: Watcher,
    pub(crate) trigger: TriggerPort,
    lost: AtomicBool,
    /// CPU waiters using VK_SYNC_WAIT_PENDING: only they observe submission
    /// (not completion) state, so only they need a wake from submit.
    pending_waiters: std::sync::atomic::AtomicU32,
    /// Nanoseconds since `epoch` of the last heavy channel-error scan.
    last_scan: AtomicU64,
    /// Deferred-reclamation groups waiting in Inner (lock-free hint so
    /// status/wait paths only try the lock when there is something to do).
    reclaim_pending: AtomicUsize,
    epoch: Instant,
    /// (HOST completion, channel RC) RM OS-event descriptors, owned by Inner.
    devices: (RawFd, RawFd),
}
impl Core {
    /// Device-free core over a CPU slot pool for watcher and waiter tests:
    /// no RM events (descriptors -1), a trigger port that is never used.
    #[cfg(test)]
    pub(crate) fn detached(pool: Arc<SlotPool>) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            pool,
            waiters: Waiters::new()?,
            watcher: Watcher::new()?,
            trigger: TriggerPort::detached(),
            lost: AtomicBool::new(false),
            pending_waiters: std::sync::atomic::AtomicU32::new(0),
            last_scan: AtomicU64::new(0),
            reclaim_pending: AtomicUsize::new(0),
            epoch: Instant::now(),
            devices: (-1, -1),
        }))
    }
    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }
    /// Publish loss and wake everyone. CPU-only: atomics, a condvar notify and
    /// eventfd writes; safe from Mesa's loss callback under queue locks. The
    /// first publication logs where it came from (logcat on Android).
    #[track_caller]
    pub(crate) fn mark_lost(&self) {
        if !self.lost.swap(true, Ordering::SeqCst) {
            diag!(
                "nvk Rust RM: device lost, published at {}",
                std::panic::Location::caller()
            );
            self.waiters.broadcast();
            self.watcher.wake();
        }
    }
    fn live(&self) -> Result<()> {
        if self.is_lost() {
            Err(Error::DeviceLost("Mesa RM device lost".into()))
        } else {
            Ok(())
        }
    }
    /// Claim the 1 s heavy-scan slot (stock throttles its notifier scan the
    /// same way: one vDSO clock read + cached words per check).
    fn scan_due(&self, force: bool) -> bool {
        // Saturates after 584 years of device lifetime.
        let now = u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let last = self.last_scan.load(Ordering::Relaxed);
        (force || now.saturating_sub(last) >= 1_000_000_000)
            && self
                .last_scan
                .compare_exchange(last, now, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
    }
}

struct Buffer {
    cpu: Rc<crate::cpu_map_cache::Slot<Mapping>>,
    memory: Rc<Memory>,
}
/// Deferred groups reclaimed per backend call at most (except at close).
const RECLAIM_BUDGET: usize = 64;
/// C callback returning a VA range to the caller's heap (nv_rm.h
/// NvRmVaRelease). Called at most once, after RM unmapped the range.
pub(crate) type VaRelease =
    Option<unsafe extern "C" fn(data: *mut c_void, address: u64, size: u64)>;
/// One deferred free (reclaim.rs). Items of a group run in order: the RM
/// unmaps first, then memory objects (a mapping holds its memory alive until
/// it is unmapped anyway), then the VA range is handed back to C.
enum ReclaimItem {
    Mapping(GpuMapping),
    Memory(Buffer),
    Va {
        address: u64,
        size: u64,
        release: unsafe extern "C" fn(*mut c_void, u64, u64),
        data: *mut c_void,
    },
}
/// RM owners of one slot-pool page. Field order: GPU VA, CPU VMA, memory.
struct PoolPage {
    _gpu: GpuMapping,
    _cpu: Mapping,
    _memory: Rc<Memory>,
}
/// Per-submission scratch kept in Inner (capacity retained across frames).
#[derive(Default)]
struct SubmitScratch {
    waits: Vec<SemaphoreOp>,
    signals: Vec<SemaphoreOp>,
    held: Vec<Slot>,
    /// Slots allocated for this batch (fresh payload slots, gates, points):
    /// released if the batch is not accepted.
    allocated: Vec<Slot>,
    actions: Vec<Action>,
    shadows: Vec<(Slot, u64)>,
    /// What each GPU acquire of this batch waits for (parallel to `waits`).
    acquires: Vec<Acquire>,
}
/// A GPU semaphore acquire of an accepted batch, as the CPU value that
/// satisfies it. Kept per context until the batch retires; the slot stays
/// alive meanwhile because the batch holds it (`held`, released only through
/// retired tokens).
#[derive(Clone, Copy)]
enum Acquire {
    /// Binary payload or gate slot: passes on 1.
    Binary(Slot),
    /// Timeline slot: passes at >= value (full-width value).
    Timeline(Slot, u64),
}
/// Post-acceptance state changes, applied only after the batch is accepted.
enum Action {
    /// (payload, consumed slot as planned, fresh replacement)
    Consume(*const SyncObj, Slot, Slot),
    /// (payload, value, slot the GPU release targets)
    Signal(*const SyncObj, u64, Slot),
    Rebind(*const SyncObj, Slot),
    ForeignGate(Arc<Foreign>, Slot),
    SharedGate(Arc<Shared>, u64, Slot),
    SharedPoint(Arc<Shared>, u64, Slot),
}

/// Rc-graph RM state. Field order is teardown order: channels first (their
/// held slot references last within each queue), then event listeners, pool
/// pages, the address space and finally the device/client.
struct Inner {
    queues: Vec<Option<Queue>>,
    /// Per context (same index as `queues`): acquires of accepted batches
    /// that may not have retired yet, oldest first (`unblock_locked`).
    acquires: Vec<VecDeque<(u64, Acquire)>>,
    export_queue: Option<u64>,
    maps: HashMap<u64, GpuMapping>,
    buffers: HashMap<u64, Buffer>,
    /// Frees waiting for the GPU work outstanding at free time (reclaim.rs).
    reclaim: Reclaim<ReclaimItem>,
    /// Reused output buffer of `Reclaim::pop_retired` (capacity retained).
    reclaimed: Vec<ReclaimItem>,
    /// Reused input buffer of nv_rm_va_free (capacity retained).
    freeing: Vec<ReclaimItem>,
    /// Mappings whose deferred RM unmap failed: their VA stays reserved (the
    /// C release is never called) and they are retried at teardown.
    unmap_failed: Vec<GpuMapping>,
    watcher_host: Option<HostListener>,
    events: RmEvents,
    pages: Vec<PoolPage>,
    space: AddressSpace,
    device: Rc<Device>,
    scratch: SubmitScratch,
    next: u64,
    loss_handled: bool,
    /// Perf capture (capture.rs); None unless configured for this process.
    capture: Option<crate::capture::Capture>,
    /// Late arming (capture.rs `ARM_POLL`): when the configuration was absent
    /// at open, the last time submit looked for it again.
    capture_probe: Option<Instant>,
    /// A late arm was refused (a context without timestamps): no more probes.
    capture_refused: bool,
    /// Capture device tag (PCI bus/slot), as at open.
    capture_tag: usize,
    /// Per context (same index as `queues`): the engine mask it was created
    /// with, for a capture armed after the context exists.
    context_engines: Vec<u32>,
    /// Per GPU mapping id in `maps`: (address, size, memory, offset) of the
    /// bind, so a capture armed late can index pushbuffer memory that was
    /// bound before it.
    bind_info: HashMap<u64, (u64, u64, u64, u64)>,
}
/// Inner, made movable between threads under the Bridge mutex.
struct Confined(Inner);
// SAFETY: the raw C `data` pointers of deferred VA releases are the
// caller's device objects, which nv_rm.h requires to outlive nv_rm_close and
// to be usable from any thread. Every Rc/Cell/RefCell reachable from Inner
// was created inside it and never escapes: all entry points run with the Bridge mutex
// held and return only plain data (ids, addresses, owned fds, Arcs of
// Send+Sync types). Rc reference counts are therefore only modified by the
// mutex owner, and the mutex provides the happens-before between successive
// owners on different threads. RM objects are per-client descriptors without
// thread affinity. Moving the whole graph between threads (never sharing it
// unlocked) is exactly what `Send` permits.
unsafe impl Send for Confined {}

pub(crate) struct Bridge {
    core: Arc<Core>,
    inner: Mutex<Confined>,
    /// Perf capture armed (frame markers take the lock only then); set at
    /// open or by a late arm from submit.
    capture: AtomicBool,
    /// Set by the C-boundary guard when a panic interrupted a call on this
    /// bridge (crate::ffi); never cleared.
    failed: crate::ffi::FailFlag,
    /// The process that opened the device: a forked child inherits the
    /// handle but not its RM objects' owner or threads, so it may neither use
    /// nor tear it down (RUST.md §13).
    process: crate::process::ProcessBound,
}

/// The payload behind a C sync id.
///
/// # Safety
/// `id` is 0 (rejected) or an nv_rm_sync_create id whose payload is not
/// destroyed during `'a`.
unsafe fn sync_ref<'a>(id: u64) -> Result<&'a SyncObj> {
    let pointer = sync_pointer(id)?;
    if pointer.is_null() {
        return Err(Error::Contract("null synchronization id".into()));
    }
    // SAFETY: ids are Arc::into_raw pointers returned by nv_rm_sync_create;
    // the C caller keeps the object alive for the duration of the call
    // (Vulkan external synchronization of vk_sync lifetime).
    let sync = unsafe { &*pointer };
    if !sync.valid() {
        return Err(Error::Contract("invalid synchronization id".into()));
    }
    Ok(sync)
}

impl Bridge {
    /// The device state. A poisoned lock (a thread panicked inside it; the
    /// FFI guard has marked the device lost) is fatal for the device
    /// (RUST.md §8): every later call fails as lost instead of using state a
    /// panic may have left half-updated.
    fn lock(&self) -> Result<MutexGuard<'_, Confined>> {
        self.inner.lock().map_err(|error| {
            Error::DeviceLost(format!(
                "device state lock: {error} (a thread panicked while holding it)"
            ))
        })
    }
    fn new(pci: [u32; 4]) -> Result<(Self, Info)> {
        crate::diag::install_panic_hook();
        crate::diag::install_logger();
        let client = Client::open()?;
        let card = client
            .cards()?
            .into_iter()
            .find(|c| {
                [
                    c.pci.domain,
                    u32::from(c.pci.bus),
                    u32::from(c.pci.slot),
                    u32::from(c.pci.function),
                ] == pci
            })
            .ok_or_else(|| Error::Contract("requested PCI NVIDIA GPU absent".into()))?;
        let device = client.device(card)?;
        let info = query_info(&device)?;
        let space = AddressSpace::new(&device)?;
        let events = RmEvents::new(&device)?;
        let caps = capabilities::query(&device)?;
        let view = if crate::gpu::modern_host(caps.channel)? {
            TimelineView::Native64
        } else {
            TimelineView::Split32
        };
        let core = Arc::new(Core {
            pool: Arc::new(SlotPool::new(view)),
            waiters: Waiters::new()?,
            watcher: Watcher::new()?,
            trigger: events.trigger_port(),
            lost: AtomicBool::new(false),
            pending_waiters: std::sync::atomic::AtomicU32::new(0),
            last_scan: AtomicU64::new(0),
            reclaim_pending: AtomicUsize::new(0),
            epoch: Instant::now(),
            devices: (events.as_raw_fd(), events.error_fd()),
        });
        let capture = crate::capture::config()
            .map(|c| crate::capture::Capture::new(c, usize_from_u32(pci[1] << 8 | pci[2])));
        let bridge = Self {
            core,
            capture: AtomicBool::new(capture.is_some()),
            failed: crate::ffi::FailFlag::default(),
            process: crate::process::ProcessBound::new()
                .map_err(|e| Error::Io("register the fork handler", e))?,
            inner: Mutex::new(Confined(Inner {
                queues: Vec::new(),
                acquires: Vec::new(),
                export_queue: None,
                maps: HashMap::new(),
                buffers: HashMap::new(),
                reclaim: Reclaim::default(),
                reclaimed: Vec::new(),
                freeing: Vec::new(),
                unmap_failed: Vec::new(),
                watcher_host: None,
                events,
                pages: Vec::new(),
                space,
                device,
                scratch: SubmitScratch::default(),
                next: 1,
                loss_handled: false,
                capture,
                capture_probe: None,
                capture_refused: false,
                capture_tag: usize_from_u32(pci[1] << 8 | pci[2]),
                context_engines: Vec::new(),
                bind_info: HashMap::new(),
            })),
        };
        bridge.grow(&mut bridge.lock()?.0)?;
        Ok((bridge, info))
    }

    // ---- slot pool --------------------------------------------------------
    /// Map one more 64 KiB SYS page (CPU-cached, GPU-uncached; never unmapped
    /// before device teardown). Requires Inner.
    fn grow(&self, inner: &mut Inner) -> Result<()> {
        if self.core.pool.next_page()?.is_none() {
            return Err(Error::Contract(
                "semaphore slot pool exhausted (1M live slots)".into(),
            ));
        }
        let memory = inner
            .device
            .allocate_aligned(PAGE_BYTES, PAGE_BYTES, false, true)?;
        let cpu = memory.map()?;
        let gpu = inner.space.map(&memory)?;
        if gpu
            .address()
            .checked_add(u64_from_usize(memory.size))
            .is_none_or(|end| end >= 1 << 34)
        {
            return Err(Error::Contract(
                "GPU semaphore pool collides with Mesa VA heap".into(),
            ));
        }
        // SAFETY: `cpu` maps the whole PAGE_BYTES allocation and is kept in
        // Inner.pages, which is dropped only after Bridge::drop closed the
        // pool (no page access after close); only the pool's atomics and the
        // GPU touch the page.
        unsafe {
            self.core
                .pool
                .install_page(cpu.cpu_address(), gpu.address())
        }?;
        inner.pages.push(PoolPage {
            _gpu: gpu,
            _cpu: cpu,
            _memory: memory,
        });
        Ok(())
    }
    fn slot_locked(&self, inner: &mut Inner) -> Result<Slot> {
        loop {
            if let Some(slot) = self.core.pool.try_allocate()? {
                return Ok(slot);
            }
            self.grow(inner)?;
        }
    }
    /// Allocate without holding any lock; takes Inner only to grow the pool.
    fn slot(&self) -> Result<Slot> {
        loop {
            if let Some(slot) = self.core.pool.try_allocate()? {
                return Ok(slot);
            }
            let mut guard = self.lock()?;
            if !self.core.pool.has_free()? {
                self.grow(&mut guard.0)?;
            }
        }
    }

    // ---- loss ------------------------------------------------------------
    /// Heavy scan: read every channel's VRAM error notifier. Requires Inner.
    fn scan_locked(&self, inner: &mut Inner) -> Result<()> {
        if self.core.is_lost() {
            self.lose_locked(inner);
            return Err(Error::DeviceLost("Mesa RM device lost".into()));
        }
        for queue in inner.queues.iter_mut().flatten() {
            if let Err(error) = queue.check_error() {
                diag!("nvk Rust RM channel notification: {error}");
                self.lose_locked(inner);
                return Err(Error::DeviceLost(error.to_string()));
            }
        }
        if inner.device.cpu_mapping_failed() {
            self.lose_locked(inner);
            return Err(Error::DeviceLost(
                "CPU mapping cleanup previously failed".into(),
            ));
        }
        Ok(())
    }
    /// Cheap per-call loss check: flag load + throttle compare; at most one
    /// heavy scan per second (skipped if another thread holds Inner — that
    /// thread scans on its own path).
    fn check(&self) -> Result<()> {
        self.core.live()?;
        if self.core.scan_due(false)
            && let Ok(mut guard) = self.inner.try_lock()
        {
            return self.scan_locked(&mut guard.0);
        }
        Ok(())
    }
    #[track_caller]
    fn lose_locked(&self, inner: &mut Inner) {
        self.core.mark_lost();
        if inner.loss_handled {
            return;
        }
        inner.loss_handled = true;
        for queue in inner.queues.iter_mut().flatten() {
            queue.stop_submissions();
        }
        self.core.watcher.fail_all(&self.core);
    }

    // ---- CPU waits -------------------------------------------------------
    fn wait_until(
        &self,
        deadline: Option<Instant>,
        mut predicate: impl FnMut() -> Result<bool>,
    ) -> Result<bool> {
        let core = &self.core;
        let result = core.waiters.wait(
            deadline,
            Some(core.devices),
            || {
                core.live()?;
                predicate()
            },
            |_| {
                // RC readiness / EINTR / 1 s period: authoritative error state.
                core.scan_due(true);
                let mut guard = self.lock()?;
                self.scan_locked(&mut guard.0)
            },
        );
        if let Err(Error::DeviceLost(_)) = &result {
            core.mark_lost();
        }
        result
    }
    fn queue_shared(&self, context: u64) -> Result<Arc<QueueShared>> {
        let guard = self.lock()?;
        Ok(Arc::clone(queue_ref(&guard.0, context)?.shared()))
    }
    fn wait_token(&self, context: u64, token: u64, timeout_ns: u64) -> Result<bool> {
        let start = Instant::now();
        let shared = self.queue_shared(context)?;
        self.check()?;
        if shared.retired(token)? {
            return Ok(true);
        }
        if timeout_ns == 0 {
            return Ok(false);
        }
        self.wait_until(deadline(start, timeout_ns), || shared.retired(token))
    }
    fn drain_timeout(&self, timeout_ns: u64) -> Result<()> {
        let start = Instant::now();
        let pending: Vec<(u64, u64)> = {
            let guard = self.lock()?;
            self.core.live()?;
            guard
                .0
                .queues
                .iter()
                .enumerate()
                .filter_map(|(i, q)| q.as_ref().map(|q| (u64_from_usize(i) + 1, q.last_token())))
                .collect()
        };
        for (context, token) in pending {
            let remaining = if timeout_ns == u64::MAX {
                u64::MAX
            } else {
                timeout_ns
                    .saturating_sub(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
            };
            if !self.wait_token(context, token, remaining)? {
                return Err(WaitError::Timeout {
                    operation: "RM teardown drain",
                    object: format!("context {context} token {token}"),
                    limit: Duration::from_nanos(timeout_ns),
                    elapsed: start.elapsed(),
                }
                .into());
            }
        }
        Ok(())
    }

    // ---- loss teardown: never free a channel blocked on our semaphore ----
    /// Device loss or an expired bounded drain, before RM-freeing channels.
    /// Satisfy every GPU acquire of the context's unretired batches (gates of
    /// failed/pending foreign or shared producers, native binary/timeline
    /// payloads nobody will signal any more) so the channel runs to its
    /// completion release instead of being freed while blocked in a semaphore
    /// ACQUIRE: pre-Volta RM then preempts the channel and logs Xid 45. The
    /// device is already lost: results of the released work are undefined, and
    /// every CPU wait/status path already reports loss. Returns the context's
    /// last token and retirement view for `quiesce`.
    fn unblock_locked(&self, inner: &mut Inner, context: u64) -> Option<(u64, Arc<QueueShared>)> {
        let index = context_index(context)?;
        let queue = inner.queues.get(index)?.as_ref()?;
        let shared = Arc::clone(queue.shared());
        let token = queue.last_token();
        // Entries newer than `completed` are still held by their batch (held
        // slots are released only through retired tokens <= completed).
        // Unreadable completion (the device is lost) releases every entry:
        // releasing an acquire the GPU already passed is harmless, keeping
        // one it is blocked on frees a blocked channel.
        let completed = shared.completed().unwrap_or(0);
        let pool = &self.core.pool;
        for &(batch, acquire) in &inner.acquires[index] {
            if batch <= completed {
                continue;
            }
            let result = match acquire {
                Acquire::Binary(slot) => pool.store32(slot, 1),
                Acquire::Timeline(slot, value) => pool.host_signal(slot, value).map(|_| ()),
            };
            if let Err(error) = result {
                diag!("nvk Rust RM teardown acquire release (context {context}): {error}");
            }
        }
        inner.acquires[index].clear();
        Some((token, shared))
    }
    /// Bounded wait, ignoring the device loss flag, until every listed channel
    /// retired its last token or reported a channel error (an RC'd channel
    /// never retires and is no longer running). Woken by the completion
    /// interrupt; channel error notifiers are rechecked on RC readiness.
    fn quiesce(&self, pending: &[(u64, Arc<QueueShared>)], timeout: Duration) -> bool {
        let start = Instant::now();
        let core = &self.core;
        let result = core.waiters.wait(
            Some(start + timeout),
            Some(core.devices),
            || {
                for (token, shared) in pending {
                    match shared.retired(*token) {
                        Ok(true) | Err(Error::DeviceLost(_)) => {}
                        Ok(false) => return Ok(false),
                        Err(error) => return Err(error),
                    }
                }
                Ok(true)
            },
            |_| {
                let mut guard = self.lock()?;
                // Refresh every channel's error notifier: an RC'd channel no
                // longer runs, which the predicate above accepts. Its error is
                // logged here; the device is already lost.
                for queue in guard.0.queues.iter_mut().flatten() {
                    if let Err(error) = queue.check_error() {
                        diag!("nvk Rust RM teardown: channel error: {error}");
                    }
                }
                Ok(())
            },
        );
        match result {
            Ok(true) => true,
            Ok(false) => {
                diag!(
                    "nvk Rust RM teardown: channels did not retire within {timeout:?} after releasing their acquires"
                );
                false
            }
            Err(error) => {
                diag!("nvk Rust RM teardown quiesce: {error}");
                false
            }
        }
    }
    /// Invariant at every channel free: no unretired work (a running or
    /// blocked channel). A channel that reported its own error is exempt.
    fn check_idle_before_free(context: u64, queue: &Queue) {
        let token = queue.last_token();
        let blocked = match queue.shared().retired(token) {
            Ok(retired) => !retired,
            Err(Error::DeviceLost(_)) => false,
            Err(_) => true,
        };
        if blocked {
            diag!(
                "nvk Rust RM: freeing channel of context {context} with unretired work (last token {token})"
            );
            if strict_teardown() {
                std::process::abort();
            }
        }
    }

    // ---- deferred reclamation (reclaim.rs) -------------------------------
    /// Open a reclaim group: snapshot every queue's unretired last token.
    /// Zero means no submitted work can still reach the object: reclaim now.
    /// A lost queue (retired() errors) counts as unretired: its objects wait
    /// for close, where channels are freed before anything is reclaimed.
    fn reclaim_begin(inner: &mut Inner) -> u32 {
        let Inner {
            queues, reclaim, ..
        } = inner;
        reclaim.begin(queues.iter().enumerate().filter_map(|(index, queue)| {
            let queue = queue.as_ref()?;
            let token = queue.last_token();
            (token != 0 && !matches!(queue.shared().retired(token), Ok(true))).then_some((
                u32::try_from(index).expect("context ids stay far below u32::MAX"),
                token,
            ))
        }))
    }
    /// Defer `items` behind the current GPU work, or reclaim them now when
    /// nothing is outstanding. Returns Err only for an immediate unmap
    /// failure (the device is then lost and the VA stays reserved).
    fn reclaim_or_defer(
        &self,
        inner: &mut Inner,
        items: impl IntoIterator<Item = ReclaimItem>,
    ) -> Result<()> {
        // Older groups first (within the per-call budget). If the new
        // snapshot is empty, every older group has retired as well, so doing
        // this one now (possibly before an older retired one still in the
        // backlog) is safe: each group is self-contained (its mappings hold
        // their memory alive) and VA ranges of different groups are disjoint.
        self.reclaim_locked(inner, false);
        if Self::reclaim_begin(inner) != 0 {
            inner.reclaim.commit(items);
            self.core.reclaim_pending.fetch_add(1, Ordering::Release);
            return Ok(());
        }
        let mut failed = None;
        for item in items {
            if let Err(error) = self.reclaim_item(inner, item, failed.is_some()) {
                failed.get_or_insert(error);
            }
        }
        failed.map_or(Ok(()), Err)
    }
    /// Reclaim every group whose snapshot retired (all of them with `force`,
    /// used only at close after every channel was freed).
    fn reclaim_locked(&self, inner: &mut Inner, force: bool) {
        if inner.reclaim.is_empty() {
            return;
        }
        let mut out = std::mem::take(&mut inner.reclaimed);
        // Amortised: at most RECLAIM_BUDGET groups per call, so a backlog
        // that built up during a long GPU stall (tens of thousands of frees)
        // never holds Inner for one long stretch of RM unmap/free ioctls on a
        // caller (measured: 494 ms with an unbounded drain). Every call that
        // frees also reclaims up to the budget, so the backlog drains faster
        // than it grows once the work retired. Close is unbounded (force).
        let mut budget = RECLAIM_BUDGET;
        loop {
            if !force {
                if budget == 0 {
                    break;
                }
                budget -= 1;
            }
            let Inner {
                queues, reclaim, ..
            } = &mut *inner;
            let retired = |queue: u32, token: u64| match queues
                .get(usize_from_u32(queue))
                .and_then(Option::as_ref)
            {
                // Destroyed context: its teardown waited (bounded) or lost,
                // then RM freed its channel (RM preempts/unbinds on free).
                None => true,
                Some(queue) => matches!(queue.shared().retired(token), Ok(true)),
            };
            if !reclaim.pop_retired(force, retired, &mut out) {
                break;
            }
            self.core.reclaim_pending.fetch_sub(1, Ordering::Release);
            let mut failed = false;
            for item in out.drain(..) {
                failed |= self.reclaim_item(inner, item, failed).is_err();
            }
        }
        inner.reclaimed = out;
    }
    /// Lock-free hint check, then reclaim if Inner is free right now (status
    /// and wait paths never block on another thread's allocation).
    fn reclaim_try(&self) {
        if self.core.reclaim_pending.load(Ordering::Acquire) != 0
            && let Ok(mut guard) = self.inner.try_lock()
        {
            self.reclaim_locked(&mut guard.0, false);
        }
    }
    /// Perform one free. `group_failed`: an earlier unmap of the same group
    /// failed, so its VA range must stay reserved.
    fn reclaim_item(&self, inner: &mut Inner, item: ReclaimItem, group_failed: bool) -> Result<()> {
        match item {
            ReclaimItem::Mapping(mut mapping) => {
                if let Err(error) = mapping.try_unmap() {
                    diag!(
                        "nvk Rust RM deferred GPU unmap at {:#x} failed; VA reserved, device lost: {error}",
                        mapping.address()
                    );
                    inner.unmap_failed.push(mapping);
                    self.lose_locked(inner);
                    return Err(Error::DeviceLost(format!(
                        "GPU unmap failed; reservation retained: {error}"
                    )));
                }
            }
            // Memory::drop frees the RM object (its own diagnostic on error);
            // the CPU mapping was already evicted at free time.
            ReclaimItem::Memory(buffer) => drop(buffer),
            ReclaimItem::Va {
                address,
                size,
                release,
                data,
            } => {
                if group_failed {
                    diag!(
                        "nvk Rust RM VA {address:#x}+{size:#x} stays reserved after a failed unmap"
                    );
                } else {
                    // SAFETY: nv_rm_va_free's contract: `release(data, ..)` is
                    // valid until nv_rm_close returns and takes only the
                    // caller's VA-heap lock (never held across backend calls).
                    unsafe { release(data, address, size) };
                }
            }
        }
        Ok(())
    }

    fn broadcast(&self, changed: bool) {
        if changed && let Err(error) = self.core.trigger.trigger() {
            diag!("RM shared payload broadcast: {error}");
        }
    }

    /// Evaluate ANY/ALL over C points (no allocation).
    ///
    /// # Safety
    /// Every `sync` id in `points` is 0 or a live nv_rm_sync_create payload
    /// for the whole call.
    unsafe fn points_ready(&self, points: &[SyncPoint], flags: u32) -> Result<bool> {
        let any = flags & 2 != 0;
        let pending = flags & 1 != 0;
        let mut ready = !any;
        let mut broadcast = false;
        for point in points {
            // SAFETY: the ids come from this function's arguments, live for the call (# Safety).
            let (one, changed) = unsafe { sync_ref(point.sync) }?.ready(point.value, pending)?;
            broadcast |= changed;
            if any {
                ready |= one;
                if one {
                    break;
                }
            } else {
                ready &= one;
                if !one {
                    break;
                }
            }
        }
        self.broadcast(broadcast);
        Ok(ready || points.is_empty())
    }
    /// # Safety
    /// Every `sync` id in `points` is 0 or a live nv_rm_sync_create payload
    /// for the whole call.
    unsafe fn wait_points(
        &self,
        points: &[SyncPoint],
        flags: u32,
        timeout_ns: u64,
    ) -> Result<bool> {
        let start = Instant::now();
        self.check()?;
        // SAFETY: same ids as this function's, under its contract.
        if unsafe { self.points_ready(points, flags) }? {
            return Ok(true);
        }
        if timeout_ns == 0 {
            return Ok(false);
        }
        // WAIT_PENDING observes submission state: register (Dekker with the
        // submitter's fence + load) before the predicate is re-evaluated.
        struct Pending<'a>(&'a std::sync::atomic::AtomicU32);
        impl Drop for Pending<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let _pending = (flags & 1 != 0).then(|| {
            self.core.pending_waiters.fetch_add(1, Ordering::SeqCst);
            Pending(&self.core.pending_waiters)
        });
        // Pending foreign sync_files wake us through the watcher; everything
        // else is a mapped word (HOST interrupt or CPU notify). The watches
        // end before this returns, so the watcher holds no reference to the
        // caller's sync_files once the wait is over (watcher.rs). No
        // allocation unless a foreign payload is pending.
        let mut watches = Vec::new();
        for point in points {
            // SAFETY: the ids come from this function's arguments, live for the call (# Safety).
            if let Some(foreign) = unsafe { sync_ref(point.sync) }?.foreign() {
                watches.push(self.core.watcher.watch_cpu(&self.core, foreign)?);
            }
        }
        let result = self.wait_until(deadline(start, timeout_ns), || {
            // SAFETY: same ids as this function's; the wait ends within the call.
            unsafe { self.points_ready(points, flags) }
        });
        drop(watches);
        result
    }

    // ---- submission ------------------------------------------------------
    /// Waits + work + signals in one GPFIFO segment and one doorbell, on the
    /// calling thread. Phase 1 plans every payload without mutating it; Full
    /// (ring pressure) or an error releases everything allocated and leaves
    /// all payloads untouched; phase 3 applies payload changes only after the
    /// batch was accepted.
    ///
    /// # Safety
    /// Every `sync` id in `waits` and `signals` is 0 or a live
    /// nv_rm_sync_create payload for the whole call.
    unsafe fn submit(
        &self,
        context: u64,
        pushes: &[Push],
        waits: &[SyncPoint],
        foreign: &[Arc<Foreign>],
        signals: &[SyncPoint],
    ) -> Result<SubmitStatus> {
        self.core.live()?;
        let mut guard = self.lock()?;
        let inner = &mut guard.0;
        if self.core.scan_due(false) {
            self.scan_locked(inner)?;
        }
        // Amortised deferred reclamation (one emptiness check when idle).
        self.reclaim_locked(inner, false);
        if inner.capture.is_none() {
            self.capture_late_arm(inner);
        }
        let mut scratch = std::mem::take(&mut inner.scratch);
        scratch.waits.clear();
        scratch.signals.clear();
        scratch.held.clear();
        scratch.allocated.clear();
        scratch.actions.clear();
        scratch.shadows.clear();
        scratch.acquires.clear();
        // SAFETY: same ids as this function's, under its contract.
        let result = unsafe {
            self.submit_locked(
                inner,
                context,
                pushes,
                waits,
                foreign,
                signals,
                &mut scratch,
            )
        };
        match &result {
            Ok(SubmitStatus::Submitted(token)) => {
                let token = *token;
                if inner.capture.is_some() {
                    capture_submit(
                        inner,
                        context,
                        pushes,
                        [
                            c_count(waits.len()),
                            c_count(foreign.len()),
                            c_count(signals.len()),
                        ],
                        token,
                    );
                }
                record_acquires(inner, context, token, scratch.acquires.drain(..));
                let mut notify = !scratch.signals.is_empty();
                for action in scratch.actions.drain(..) {
                    match action {
                        Action::Consume(sync, planned, fresh) => {
                            // SAFETY: every Action pointer came from sync_ref on
                            // an id of this call (`submit`'s # Safety): C keeps
                            // the payload alive until the call returns.
                            let sync = unsafe { &*sync };
                            let mut state = sync.lock();
                            check_planned(&state, planned, "wait");
                            // The batch now holds the consumed slot (held
                            // above); drop the payload's own reference.
                            sync.rebind_native(&mut state, fresh);
                            notify = true;
                        }
                        Action::Rebind(sync, fresh) => {
                            // SAFETY: as for Consume: a payload of this call's ids.
                            let sync = unsafe { &*sync };
                            sync.rebind_native(&mut sync.lock(), fresh);
                        }
                        Action::Signal(sync, value, planned) => {
                            // SAFETY: as for Consume: a payload of this call's ids.
                            let sync = unsafe { &*sync };
                            let mut state = sync.lock();
                            check_planned(&state, planned, "signal");
                            sync.commit_signal(&mut state, context, token, value);
                        }
                        Action::ForeignGate(foreign, gate) => {
                            if let Err(error) = self.core.watcher.gate(&self.core, foreign, gate) {
                                // The batch is accepted and waits on a gate that
                                // can no longer be released: never as success.
                                diag!("nvk Rust RM foreign gate registration: {error}");
                                self.lose_locked(inner);
                            }
                        }
                        Action::SharedGate(shared, target, gate) => {
                            if let Err(error) = self.ensure_watcher_host(inner).and_then(|()| {
                                self.core
                                    .watcher
                                    .shared_gate(&self.core, shared, target, gate)
                            }) {
                                diag!("nvk Rust RM shared gate registration: {error}");
                                self.lose_locked(inner);
                            }
                        }
                        Action::SharedPoint(shared, target, point) => {
                            shared.add_point(target, point);
                            if let Err(error) = self
                                .ensure_watcher_host(inner)
                                .and_then(|()| self.core.watcher.publish(&self.core, shared))
                            {
                                diag!("nvk Rust RM shared publication: {error}");
                                self.lose_locked(inner);
                            }
                        }
                    }
                }
                // WAIT_PENDING waiters observe the pending state committed
                // above (under each payload lock); completion waiters are woken
                // by the GPU interrupt instead.
                std::sync::atomic::fence(Ordering::SeqCst);
                if notify && self.core.pending_waiters.load(Ordering::SeqCst) != 0 {
                    self.core.waiters.notify();
                }
            }
            _ => {
                // Not accepted: payloads untouched, batch-local slots freed.
                for slot in scratch.allocated.drain(..) {
                    self.core.pool.release(slot);
                }
                scratch.actions.clear();
            }
        }
        inner.scratch = scratch;
        result
    }
    /// # Safety
    /// As for `submit`: every `sync` id in `waits` and `signals` is 0 or a
    /// live nv_rm_sync_create payload for the whole call.
    #[allow(clippy::too_many_arguments)]
    unsafe fn submit_locked(
        &self,
        inner: &mut Inner,
        context: u64,
        pushes: &[Push],
        waits: &[SyncPoint],
        foreign: &[Arc<Foreign>],
        signals: &[SyncPoint],
        s: &mut SubmitScratch,
    ) -> Result<SubmitStatus> {
        let pool = &self.core.pool;
        queue_ref(inner, context)?;
        for f in foreign {
            if !f.ready(false)? {
                let gate = self.slot_locked(inner)?;
                s.allocated.push(gate);
                s.held.push(gate);
                s.waits.push(SemaphoreOp::AcquireBinary(pool.gpu(gate)));
                s.acquires.push(Acquire::Binary(gate));
                s.actions.push(Action::ForeignGate(Arc::clone(f), gate));
            }
        }
        for point in waits {
            // SAFETY: the ids come from this function's arguments, live for the call (# Safety).
            let sync = unsafe { sync_ref(point.sync) }?;
            let (plan, consumes) = {
                let state = sync.lock();
                (
                    sync.plan_wait(&state, point.value)?,
                    sync.wait_consumes(&state),
                )
            };
            match plan {
                WaitPlan::Complete => {
                    if consumes {
                        // A satisfied binary/foreign wait still unsignals it.
                        let fresh = self.slot_locked(inner)?;
                        s.allocated.push(fresh);
                        s.actions.push(Action::Rebind(sync, fresh));
                    }
                }
                WaitPlan::Binary(slot) => {
                    let fresh = self.slot_locked(inner)?;
                    s.allocated.push(fresh);
                    s.held.push(slot);
                    s.waits.push(SemaphoreOp::AcquireBinary(pool.gpu(slot)));
                    s.acquires.push(Acquire::Binary(slot));
                    s.actions.push(Action::Consume(sync, slot, fresh));
                }
                WaitPlan::Timeline(slot, value) => {
                    s.held.push(slot);
                    s.acquires.push(Acquire::Timeline(slot, value));
                    let value = if pool.view == TimelineView::Native64 {
                        value
                    } else {
                        u64::from(low_u32(value))
                    };
                    s.waits
                        .push(SemaphoreOp::AcquireTimeline(pool.gpu(slot), value));
                }
                WaitPlan::Foreign(f) => {
                    let gate = self.slot_locked(inner)?;
                    let fresh = self.slot_locked(inner)?;
                    s.allocated.push(gate);
                    s.allocated.push(fresh);
                    s.held.push(gate);
                    s.waits.push(SemaphoreOp::AcquireBinary(pool.gpu(gate)));
                    s.acquires.push(Acquire::Binary(gate));
                    s.actions.push(Action::ForeignGate(f, gate));
                    s.actions.push(Action::Rebind(sync, fresh));
                }
                WaitPlan::Shared(shared, target) => {
                    let gate = self.slot_locked(inner)?;
                    s.allocated.push(gate);
                    s.held.push(gate);
                    s.waits.push(SemaphoreOp::AcquireBinary(pool.gpu(gate)));
                    s.acquires.push(Acquire::Binary(gate));
                    s.actions.push(Action::SharedGate(shared, target, gate));
                }
            }
        }
        for point in signals {
            // SAFETY: the ids come from this function's arguments, live for the call (# Safety).
            let sync = unsafe { sync_ref(point.sync) }?;
            // A payload consumed by a wait in this same batch is signalled on
            // its replacement slot (Vulkan: wait then signal of one binary).
            let rebound = s.actions.iter().rev().find_map(|a| match a {
                Action::Consume(p, _, fresh) | Action::Rebind(p, fresh)
                    if std::ptr::eq(*p, sync) =>
                {
                    Some(*fresh)
                }
                _ => None,
            });
            let plan = match rebound {
                Some(fresh) if !sync.timeline => SignalPlan::Binary(fresh),
                _ => sync.plan_signal(&sync.lock(), point.value)?,
            };
            match plan {
                SignalPlan::Binary(slot) => {
                    s.held.push(slot);
                    s.signals.push(SemaphoreOp::ReleaseBinary(pool.gpu(slot)));
                    s.actions.push(Action::Signal(sync, point.value, slot));
                }
                SignalPlan::Timeline(slot, value) => {
                    pool.check_split_distance(slot, value)?;
                    s.held.push(slot);
                    if pool.view == TimelineView::Split32 {
                        s.shadows.push((slot, value));
                    }
                    let wire = if pool.view == TimelineView::Native64 {
                        value
                    } else {
                        u64::from(low_u32(value))
                    };
                    s.signals
                        .push(SemaphoreOp::ReleaseTimeline(pool.gpu(slot), wire));
                    s.actions.push(Action::Signal(sync, value, slot));
                }
                SignalPlan::ReplaceForeign => {
                    let fresh = self.slot_locked(inner)?;
                    s.allocated.push(fresh);
                    s.held.push(fresh);
                    s.signals.push(SemaphoreOp::ReleaseBinary(pool.gpu(fresh)));
                    s.actions.push(Action::Rebind(sync, fresh));
                    s.actions.push(Action::Signal(sync, point.value, fresh));
                }
                SignalPlan::Shared(shared, target) => {
                    let point_slot = self.slot_locked(inner)?;
                    s.allocated.push(point_slot);
                    s.held.push(point_slot);
                    s.signals
                        .push(SemaphoreOp::ReleaseBinary(pool.gpu(point_slot)));
                    s.actions
                        .push(Action::SharedPoint(shared, target, point_slot));
                }
            }
        }
        let space = &inner.space;
        let queue = queue_slot(&mut inner.queues, context)?;
        let shadows = &s.shadows;
        let result = queue.submit(pushes, space, &s.waits, &s.signals, &s.held, |_| {
            for &(slot, value) in shadows {
                pool.raise_shadow(slot, value)?;
            }
            Ok(())
        });
        let status = match result {
            Ok(status) => status,
            Err(error) => {
                // The queue marked its channel lost (partially written ring or
                // failed overflow mapping): the device cannot continue.
                self.lose_locked(inner);
                return Err(Error::DeviceLost(error.to_string()));
            }
        };
        if let SubmitStatus::Submitted(_) = status {
            // Allocated references now belong to payloads/gates/points via
            // the actions; nothing to release.
            s.allocated.clear();
        }
        Ok(status)
    }

    fn ensure_watcher_host(&self, inner: &mut Inner) -> Result<()> {
        let fd = match &inner.watcher_host {
            Some(listener) => listener.as_raw_fd(),
            None => {
                let listener = inner.events.host_listener()?;
                let fd = listener.as_raw_fd();
                inner.watcher_host = Some(listener);
                fd
            }
        };
        if !self.core.watcher.has_host()? {
            self.core.watcher.set_host(fd)?;
        }
        Ok(())
    }

    /// SYNC_FD export of the binary payload `id`.
    ///
    /// # Safety
    /// `id` is 0 or a live nv_rm_sync_create payload, and `drm_fd` is
    /// negative (rejected where needed) or open, for the whole call.
    unsafe fn export_sync_file(&self, id: u64, drm_fd: i32) -> Result<Option<OwnedFd>> {
        // SAFETY: `id` is this function's, under its contract.
        let sync = unsafe { sync_ref(id) }?;
        if sync.timeline {
            return Err(Error::Contract("timeline SYNC_FD export is invalid".into()));
        }
        let mut guard = self.lock()?;
        let inner = &mut guard.0;
        self.core.live()?;
        enum Next {
            Done(Option<OwnedFd>),
            Fence(u64, u64),
            Gate(Arc<Shared>, u64),
            /// Reserved native slot; this export holds one reference.
            Slot(Slot),
        }
        let next = {
            let state = sync.lock();
            match &state.payload {
                crate::syncobj::Payload::Foreign(f) => Next::Done(Some(f.duplicate()?)),
                crate::syncobj::Payload::Native(slot) => {
                    if self.core.pool.read32(*slot)? == 1 {
                        Next::Done(None)
                    } else if state.pending == Some(crate::syncobj::RESERVED) {
                        // Signal reserved for a submit-thread submission that
                        // has not reached a ring yet: the export queue's GPU
                        // acquire waits on the reserved slot itself.
                        self.core.pool.retain(*slot);
                        Next::Slot(*slot)
                    } else if let Some((context, token)) = state.pending {
                        Next::Fence(context, token)
                    } else {
                        // No submitted signal. Vulkan forbids this export
                        // (VUID-VkSemaphoreGetFdInfoKHR-handleType-03254,
                        // VUID-VkFenceGetFdInfoKHR-handleType-01454). A GPU
                        // wait on this slot would be orphaned: Mesa resets the
                        // payload right after every SYNC_FD export
                        // (vk_semaphore.c:700, vk_fence.c:468), which rebinds a
                        // captured slot, so any later signal lands elsewhere
                        // and the exported fence would never signal. Fail
                        // loudly instead of handing out a dead fence.
                        diag!(
                            "nvk Rust RM: SYNC_FD export of an unsignaled binary payload with no submitted signal"
                        );
                        return Err(Error::Contract(
                            "SYNC_FD export without a submitted signal operation".into(),
                        ));
                    }
                }
                crate::syncobj::Payload::Shared(shared) => {
                    let shared = Arc::clone(shared);
                    let target = state.snapshot.unwrap_or_else(|| shared.target());
                    drop(state);
                    let changed = shared.refresh()?;
                    self.broadcast(changed);
                    if shared.completed() >= target {
                        Next::Done(None)
                    } else if shared.failed() {
                        return Err(Error::DeviceLost(
                            "shared synchronization producer failed".into(),
                        ));
                    } else {
                        Next::Gate(shared, target)
                    }
                }
            }
        };
        match next {
            Next::Done(fd) => Ok(fd),
            Next::Fence(context, token) => {
                let queue = queue_ref(inner, context)?;
                // SAFETY: `drm_fd` is this function's, under its contract.
                unsafe { queue.context_fence(drm_fd, token) }.map(Some)
            }
            Next::Slot(slot) => {
                // The export batch acquires the reserved slot directly: the
                // submit thread's later GPU release completes it. No CPU wait
                // happens here; ring pressure is a named capacity error.
                let pool = &self.core.pool;
                let context = match inner.export_queue {
                    Some(context) => context,
                    None => match self.create_context(inner, 1, None) {
                        Ok(context) => {
                            inner.export_queue = Some(context);
                            context
                        }
                        Err(error) => {
                            pool.release(slot);
                            return Err(error);
                        }
                    },
                };
                let waits = [SemaphoreOp::AcquireBinary(pool.gpu(slot))];
                let space = &inner.space;
                let queue = queue_slot(&mut inner.queues, context)?;
                let status = queue.submit(&[], space, &waits, &[], &[slot], |_| Ok(()));
                pool.release(slot); // the accepted batch holds its own reference
                let token = match status {
                    Ok(SubmitStatus::Submitted(token)) => token,
                    Ok(_) => {
                        return Err(Error::Io(
                            "external fence export capacity exhausted",
                            std::io::Error::from_raw_os_error(12),
                        ));
                    }
                    Err(error) => return Err(error),
                };
                record_acquires(inner, context, token, [Acquire::Binary(slot)]);
                let queue = queue_ref(inner, context)?;
                // SAFETY: `drm_fd` is this function's, under its contract.
                unsafe { queue.context_fence(drm_fd, token) }.map(Some)
            }
            Next::Gate(shared, target) => {
                // Imported shared payloads have no process-local producer
                // queue: an internal export queue waits on a watcher-owned
                // gate (released when the shared generation completes).
                let context = match inner.export_queue {
                    Some(context) => context,
                    None => {
                        let context = self.create_context(inner, 1, None)?;
                        inner.export_queue = Some(context);
                        context
                    }
                };
                let pool = &self.core.pool;
                let gate = self.slot_locked(inner)?;
                let waits = [SemaphoreOp::AcquireBinary(pool.gpu(gate))];
                let space = &inner.space;
                let queue = queue_slot(&mut inner.queues, context)?;
                let status = queue.submit(&[], space, &waits, &[], &[gate], |_| Ok(()));
                let token = match status {
                    Ok(SubmitStatus::Submitted(token)) => {
                        record_acquires(inner, context, token, [Acquire::Binary(gate)]);
                        token
                    }
                    other => {
                        pool.release(gate);
                        return match other {
                            Ok(_) => Err(Error::Io(
                                "external fence export capacity exhausted",
                                std::io::Error::from_raw_os_error(12),
                            )),
                            Err(error) => Err(error),
                        };
                    }
                };
                // The watcher takes the gate's CPU reference (the export
                // batch holds its own). If it cannot, the accepted export
                // batch can never complete: publish loss, never a dead fence.
                if let Err(error) = self.ensure_watcher_host(inner) {
                    pool.release(gate);
                    diag!("nvk Rust RM shared export gate registration: {error}");
                    self.lose_locked(inner);
                    return Err(error);
                }
                if let Err(error) = self
                    .core
                    .watcher
                    .shared_gate(&self.core, shared, target, gate)
                {
                    diag!("nvk Rust RM shared export gate registration: {error}");
                    self.lose_locked(inner);
                    return Err(error);
                }
                let queue = queue_ref(inner, context)?;
                // SAFETY: `drm_fd` is this function's, under its contract.
                unsafe { queue.context_fence(drm_fd, token) }.map(Some)
            }
        }
    }

    /// Arm the perf capture in a process that is already running (capture.rs
    /// "Late arming"): at most every `ARM_POLL`, look for the configuration
    /// again; once present, every live context gets its timestamp ring (as
    /// create_context does for an armed process) and the capture's
    /// VA → allocation index is seeded from the live binds. A context whose
    /// timestamps cannot be enabled refuses the arm loudly (no partial
    /// capture) and the probe stops.
    fn capture_late_arm(&self, inner: &mut Inner) {
        if inner.capture_refused {
            return;
        }
        let now = Instant::now();
        if inner
            .capture_probe
            .is_some_and(|last| now.duration_since(last) < crate::capture::ARM_POLL)
        {
            return;
        }
        inner.capture_probe = Some(now);
        let Some(config) = crate::capture::config() else {
            return;
        };
        let mut cap = crate::capture::Capture::new(config, inner.capture_tag);
        for (i, queue) in inner.queues.iter_mut().enumerate() {
            let Some(queue) = queue.as_mut() else {
                continue;
            };
            if let Err(error) = queue.enable_timing(&inner.space) {
                crate::diag::diag!(
                    "nvk Rust RM capture: late arm refused, context {} has no timestamps: {error}",
                    i + 1
                );
                // Never retried: the probe interval would repeat the refusal.
                inner.capture_refused = true;
                return;
            }
            cap.add_context(
                u64_from_usize(i) + 1,
                inner.context_engines.get(i).copied().unwrap_or(0),
                queue.channel_class(),
            );
        }
        for &(address, size, memory, offset) in inner.bind_info.values() {
            cap.binds.insert(address, (size, memory, offset));
        }
        crate::diag::diag!(
            "nvk Rust RM capture armed late: {} contexts, {} binds indexed",
            inner.queues.iter().flatten().count(),
            cap.binds.len()
        );
        inner.capture = Some(cap);
        self.capture.store(true, Ordering::Relaxed);
    }
    /// `share`: the context whose TSG the new context's channel joins on
    /// its own subcontext (VEID); `None` gives the context a TSG of its own.
    fn create_context(&self, inner: &mut Inner, engines: u32, share: Option<u64>) -> Result<u64> {
        let retire = self.slot_locked(inner)?;
        let created = match share {
            None => Queue::new_with_events(
                &inner.device,
                &inner.space,
                engines,
                &inner.events,
                &self.core.pool,
                retire,
            ),
            Some(share) => match queue_ref(inner, share) {
                Ok(share) => Queue::new_in_group(
                    &inner.device,
                    &inner.space,
                    engines,
                    &inner.events,
                    &self.core.pool,
                    retire,
                    share,
                ),
                Err(error) => Err(error),
            },
        };
        match created {
            Ok(mut queue) => {
                if let Some(cap) = inner.capture.as_mut() {
                    // Loud: an armed capture without timestamps is refused.
                    queue.enable_timing(&inner.space)?;
                    cap.add_context(
                        u64_from_usize(inner.queues.len()) + 1,
                        engines,
                        queue.channel_class(),
                    );
                }
                inner.queues.push(Some(queue));
                inner.acquires.push(VecDeque::new());
                inner.context_engines.push(engines);
                Ok(u64_from_usize(inner.queues.len()))
            }
            Err(error) => {
                self.core.pool.release(retire);
                Err(error)
            }
        }
    }
}
/// Perf capture of one accepted batch (capture.rs): drains completed
/// timestamps of every queue and, while recording, writes the batch with the
/// pushbuffer words of each push that lies in a currently CPU-mapped
/// allocation (others are recorded without contents).
fn capture_submit(inner: &mut Inner, context: u64, pushes: &[Push], counts: [u32; 3], token: u64) {
    let Some(mut cap) = inner.capture.take() else {
        return;
    };
    let device = &inner.device;
    let active = cap.active(|| {
        let mut time = 0u64;
        // The GPU time only labels the capture start; a failed read leaves
        // it 0 and the capture runs on.
        if let Err(error) = device
            .client
            .control(device.subdevice, 0x20800403, &mut time)
        {
            diag!("nvk Rust RM capture: GPU time read failed: {error}");
        }
        time
    });
    let mut times = Vec::new();
    for (i, q) in inner.queues.iter_mut().enumerate() {
        if let Some(q) = q.as_mut() {
            if let Err(error) = q.drain_timing(&mut times) {
                diag!("nvk Rust RM capture: timing ring read failed: {error}");
            }
            if active {
                for (tok, start, end) in times.drain(..) {
                    cap.time(u64_from_usize(i) + 1, tok, start, end);
                }
            }
            times.clear();
        }
    }
    if active {
        let data: Vec<_> = pushes
            .iter()
            .map(|p| {
                let words = if cap.config.push {
                    capture_push(inner, &cap.binds, p)
                } else {
                    None
                };
                (p.address, p.size, p.flags, words)
            })
            .collect();
        // Texture header / sampler pools this batch binds: their contents
        // as the CPU wrote them, read at submission (capture.rs REC_POOL).
        let pools = crate::capture::pool_binds(data.iter().filter_map(|d| d.3.as_deref()));
        cap.submit(context, token, counts, &data);
        for pool in pools {
            let bytes = capture_read(inner, &cap.binds, pool.address, pool.size);
            cap.pool(token, &pool, bytes.as_deref());
        }
    }
    inner.capture = Some(cap);
}
fn capture_push(
    inner: &Inner,
    binds: &std::collections::BTreeMap<u64, (u64, u64, u64)>,
    p: &Push,
) -> Option<Vec<u8>> {
    capture_read(inner, binds, p.address, p.size)
}
/// Bytes [address, address + size) of a CPU-mapped allocation (None when the
/// range is not inside one bind of a currently CPU-mapped allocation).
fn capture_read(
    inner: &Inner,
    binds: &std::collections::BTreeMap<u64, (u64, u64, u64)>,
    address: u64,
    size: u32,
) -> Option<Vec<u8>> {
    let (&start, &(len, memory, offset)) = binds.range(..=address).next_back()?;
    if address + u64::from(size) > start + len {
        return None;
    }
    let buffer = inner.buffers.get(&memory)?;
    if buffer.cpu.refs() == 0 {
        return None;
    }
    // A bind offset beyond the address space cannot be read: not captured.
    let at = usize::try_from(offset + address - start).ok()?;
    let mut words = vec![0u8; usize_from_u32(size)];
    let read = buffer.cpu.acquire(
        || Err(Error::Contract("capture: mapping vanished".into())),
        // Sound only after the GPU handoff: UNVERIFIED: capture copies pushbuffers NVK finished
        // writing before nv_rm_submit; another app thread could still write a bind (capture mode only).
        |m: &Mapping| m.read_bytes(at, &mut words),
    );
    match read {
        Ok(r) => {
            if let Err(error) = buffer.cpu.release() {
                diag!("nvk Rust RM capture: push mapping release: {error}");
            }
            // A failed read leaves this push uncaptured (capture mode only).
            match r {
                Ok(()) => Some(words),
                Err(error) => {
                    diag!("nvk Rust RM capture: push read: {error}");
                    None
                }
            }
        }
        Err(error) => {
            diag!("nvk Rust RM capture: push mapping: {error}");
            None
        }
    }
}
/// Plan/commit consistency: submission plans read each payload under its own
/// lock and commit after acceptance (under Inner). Only a host operation on
/// the same payload racing the submit call (a Vulkan external-synchronization
/// violation) can change the slot in between; the GPU then waits on or
/// releases a slot the payload no longer names. Make that visible.
fn check_planned(state: &crate::syncobj::SyncState, planned: Slot, what: &str) {
    if !matches!(state.payload, crate::syncobj::Payload::Native(slot) if slot == planned) {
        diag!(
            "nvk Rust RM: payload changed between {what} planning and commit              (concurrent host operation on a submitted semaphore/fence)"
        );
    }
}
/// Append an accepted batch's acquires; drop entries of batches the GPU has
/// already completed (one mapped-word read, only when something can go).
fn record_acquires(
    inner: &mut Inner,
    context: u64,
    token: u64,
    acquires: impl IntoIterator<Item = Acquire>,
) {
    let index = context_index(context)
        .expect("the context was resolved by the submit that accepted the batch");
    let list = &mut inner.acquires[index];
    if list.front().is_some()
        && let Some(Ok(completed)) = inner.queues[index].as_ref().map(|q| q.shared().completed())
    {
        while list.front().is_some_and(|(t, _)| *t <= completed) {
            list.pop_front();
        }
    }
    list.extend(acquires.into_iter().map(|a| (token, a)));
}
/// Test hook (fixtures set it in the probe child): a channel freed while it
/// still has unretired work aborts the process instead of only logging, so
/// the violation is observable on bionic too (diagnostics go to logcat).
fn strict_teardown() -> bool {
    std::env::var_os("NV_RM_STRICT_TEARDOWN").is_some_and(|v| v != "0")
}
/// Index of a 1-based context id (`nv_rm_context_create` returns
/// `queues.len()` after the push) into `Inner.queues`; None for 0 and for ids
/// beyond the address space.
fn context_index(context: u64) -> Option<usize> {
    context.checked_sub(1).and_then(|i| usize::try_from(i).ok())
}
/// The live queue of `context`, borrowed from the queue table alone so other
/// fields of Inner stay usable.
fn queue_slot(queues: &mut [Option<Queue>], context: u64) -> Result<&mut Queue> {
    context_index(context)
        .and_then(|i| queues.get_mut(i))
        .and_then(Option::as_mut)
        .ok_or_else(|| Error::Contract("unknown RM context".into()))
}
/// A count of an array C passed with a u32 count (`nv_rm.h`): it fits u32.
fn c_count(len: usize) -> u32 {
    u32::try_from(len).expect("nv_rm.h arrays arrive with u32 counts")
}
/// The payload pointer of a sync id (`nv_rm_sync_create`'s `Arc::into_raw`
/// address, exposed as an integer for C).
fn sync_pointer(id: u64) -> Result<*const SyncObj> {
    usize::try_from(id)
        .map(std::ptr::with_exposed_provenance)
        .map_err(|error| Error::Contract(format!("synchronization id {id:#x}: {error}")))
}
fn queue_ref(inner: &Inner, context: u64) -> Result<&Queue> {
    context
        .checked_sub(1)
        .and_then(|i| inner.queues.get(usize::try_from(i).ok()?))
        .and_then(Option::as_ref)
        .ok_or_else(|| Error::Contract("unknown RM context".into()))
}
/// The deadline of a caller's wait. `u64::MAX` is the caller asking for no
/// bound (RUST.md §7's exemption): a Vulkan `UINT64_MAX` timeout, or `nvkmd_rm.c`'s
/// ring-pressure and context-sync waits, which pass `UINT64_MAX` as NVK's nouveau
/// backend does for the same operations (Mesa `nvkmd_nouveau_ctx.c`,
/// `nvkmd_nouveau_exec_ctx_sync`: `drmSyncobjWait(..., INT64_MAX, ...)`). Such a
/// wait still ends: the leader reads every channel's error notifier at least
/// once per second (waiters.rs `SCAN_PERIOD`) and a dead channel ends it as
/// device loss.
fn deadline(start: Instant, timeout_ns: u64) -> Option<Instant> {
    (timeout_ns != u64::MAX).then(|| start + Duration::from_nanos(timeout_ns))
}

impl Drop for Bridge {
    fn drop(&mut self) {
        // Bounded drain (5 s) on the closing thread, then stop the only other
        // thread, then RM teardown. Nothing continues after this returns:
        // callers may dlclose the DSO immediately.
        let drained = self.drain_timeout(5_000_000_000);
        if let Err(e) = &drained {
            diag!("RM close drain failed; tearing down lost channels: {e}");
            match self.lock() {
                Ok(mut guard) => self.lose_locked(&mut guard.0),
                Err(error) => diag!("RM close: {error}"),
            }
        }
        self.core.watcher.stop(&self.core);
        if drained.is_err() {
            // Release what the lost channels wait on, then let them run out
            // (bounded): never free a channel blocked in an acquire.
            let pending: Vec<_> = match self.lock() {
                Ok(mut guard) => {
                    let inner = &mut guard.0;
                    (1..=u64_from_usize(inner.queues.len()))
                        .filter_map(|context| self.unblock_locked(inner, context))
                        .collect()
                }
                Err(error) => {
                    diag!("RM close: {error}");
                    Vec::new()
                }
            };
            self.quiesce(&pending, Duration::from_secs(5));
        }
        let mut guard = match self.lock() {
            Ok(guard) => guard,
            Err(error) => {
                // State a panic may have left half-updated is not walked:
                // its channels and objects drop with the Bridge, in field
                // order, and leaked payloads stop touching the pool's pages.
                diag!("RM close: {error}; explicit teardown skipped");
                self.core.pool.close();
                return;
            }
        };
        let inner = &mut guard.0;
        for (index, queue) in inner.queues.iter().enumerate() {
            if let Some(queue) = queue {
                Self::check_idle_before_free(u64_from_usize(index) + 1, queue);
            }
        }
        // Leaked payloads (API misuse) must not touch unmapped pages.
        self.core.pool.close();
        // Free every channel directly, with no schedule-off first (stock
        // order: drain, then RM free; libnvidia-glcore 580 ELF b78280).
        // Channels go first (Inner's first
        // field, so the same order as the implicit drop), then nothing can
        // reach deferred objects any more:
        // unmap/free them and return their VA ranges to C before this
        // returns (nv_rm.h: releases never run after nv_rm_close).
        inner.queues.clear();
        inner.export_queue = None;
        self.reclaim_locked(inner, true);
        // The rest of Inner (mappings, listeners, pool pages, space, device)
        // drops with the Bridge after this function, in field order.
    }
}

fn query_info(d: &Device) -> Result<Info> {
    crate::external::validate(d)?;
    let mut arch = [0u32; 4];
    d.client.control(d.subdevice, 0x20801701, &mut arch)?;
    let mut name = [0u32; 17];
    d.client.control(d.subdevice, 0x20800110, &mut name)?;
    let mut gr = [[0xcu32, 0], [0xd, 0], [0x20, 0], [0x7, 0], [0x9, 0]];
    let mut query = abi::GrInfo {
        count: u32::try_from(gr.len()).expect("a five-entry GR info query"),
        entries: u64_from_usize(gr.as_mut_ptr().expose_provenance()),
        ..Default::default()
    };
    d.client.control(d.subdevice, 0x20801201, &mut query)?;
    let caps = capabilities::select(&d.classes()?, gr[0][1])?;
    // Contexts that can join a channel group. From Volta on each takes a
    // subcontext of its own: RM's per-chip count is GR info
    // NV2080_CTRL_GR_INFO_INDEX_MAX_SUBCONTEXT_COUNT (ctrl0080gr.h 0x2c; RM's
    // own FIFO bound reads the same entry, OGKM kernel_fifo.c:2791 /
    // kernel_fifo_gv100.c:181-207), within the 128 channels a group holds
    // (gv100 dev_ram.h NV_RAMRL_ENTRY_TSG_LENGTH_MAX). Before Volta there are
    // no subcontexts: joining channels share the group's GR context and the
    // group holds 32 channels (gm107 dev_ram.h NV_RAMRL_ENTRY_TSG_LENGTH_MAX,
    // Maxwell and Pascal). The group's first context keeps one of them.
    let joinable = if caps.usermode.is_some() {
        let mut sub = [[0x2cu32, 0]];
        let mut query = abi::GrInfo {
            count: u32::try_from(sub.len()).expect("a one-entry GR info query"),
            entries: u64_from_usize(sub.as_mut_ptr().expose_provenance()),
            ..Default::default()
        };
        d.client.control(d.subdevice, 0x20801201, &mut query)?;
        sub[0][1].min(0x80)
    } else {
        0x20
    };
    let max_shared_ctx = joinable.saturating_sub(1);
    // Route info is {u32 flags,pad,u64 data}; final mask follows at offset16.
    let mut gpc = [0u64; 3];
    d.client.control(d.subdevice, 0x2080122a, &mut gpc)?;
    let mut tpc_count = 0;
    for i in 0..32u32 {
        if low_u32(gpc[2]) & (1 << i) != 0 {
            let mut tpc = [0u64, 0, u64::from(i)];
            d.client.control(d.subdevice, 0x2080122b, &mut tpc)?;
            tpc_count += high_u32(tpc[2]).count_ones();
        }
    }
    let mut fb = [0x7u32, 0];
    let mut fbq = [1u64, u64_from_usize(fb.as_mut_ptr().expose_provenance())];
    d.client.control(d.subdevice, 0x20801301, &mut fbq)?;
    let mut text = [0; 64];
    for (i, w) in name[1..].iter().enumerate() {
        text[i * 4..i * 4 + 4].copy_from_slice(&w.to_ne_bytes());
    }
    let (zcull, zcull_ctxsw_size, zcull_ctxsw_align, has_zcull_info) = match query_zcull(d) {
        Ok((info, ctxsw)) => (info, ctxsw.size, ctxsw.alignment, 1),
        Err(e) => {
            diag!("nvk Rust RM: ZCULL disabled on this device: {e}");
            (abi::ZcullInfo::default(), 0, 0, 0)
        }
    };
    let c = d.card;
    Ok(Info {
        vendor_id: u32::from(c.pci.vendor),
        device_id: u32::from(c.pci.device),
        chipset: arch[0]
            .checked_add(arch[1])
            .ok_or_else(|| Error::Contract("chipset identity overflow".into()))?,
        sm: caps.sm,
        pci_domain: c.pci.domain,
        pci_bus: u32::from(c.pci.bus),
        pci_device: u32::from(c.pci.slot),
        pci_function: u32::from(c.pci.function),
        gpc_count: low_u32(gpc[2]).count_ones(),
        tpc_count,
        mp_per_tpc: gr[2][1],
        max_warps_per_mp: gr[1][1],
        cls_copy: caps.copy,
        cls_eng2d: caps.eng2d,
        cls_eng3d: caps.eng3d,
        cls_m2mf: caps.m2mf,
        cls_compute: caps.compute,
        generic_kind: caps.generic_kind,
        vram_size: u64::from(fb[1]) * 1024,
        bar_size: c.framebuffer_size,
        name: text,
        zcull,
        zcull_ctxsw_size,
        zcull_ctxsw_align,
        has_zcull_info,
        sm_isa: gr[0][1],
        shader_pipe_count: gr[3][1],
        shader_pipe_sub_count: gr[4][1],
        max_shared_ctx,
        _reserved: 0,
    })
}

/// Most subregions Mesa's ZCULL layout code accepts (nil zcull.rs
/// MAX_SUBREGIONS; it asserts beyond that).
const ZCULL_MAX_SUBREGIONS: u32 = 16;

/// ZCULL region geometry for NVK's ZCULL path, the RM counterpart of
/// DRM_NOUVEAU_GET_ZCULL_INFO: the ten NV2080 GR_GET_ZCULL_INFO fields as the
/// RM reports them (the stock 615.78.08 driver queries the same control,
/// 0x28 bytes, once per subdevice at device open: libnvidia-eglcore loop
/// E 0x4e3a70..0x4e3aa2, `mov $0x20801206,%ecx` at E 0x4e3a86; and the
/// nouveau GSP path copies them field by field, linux
/// nvkm/subdev/gsp/rm/r570/gr.c r570_gr_get_ctxbufs_and_zcull_info) plus the
/// GRAPHICS_ZCULL context-switch buffer size and alignment that every
/// graphics channel binds (gpu.rs bind_zcull). Values Mesa's layout code
/// cannot handle (zero divisors or alignments, more than 16 subregions) are
/// refused with the reason, so ZCULL stays off instead of aborting in nil.
fn query_zcull(d: &Device) -> Result<(abi::ZcullInfo, abi::EngineContextProperties)> {
    let mut info = abi::ZcullInfo::default();
    d.client.control(d.subdevice, 0x20801206, &mut info)?;
    let mut ctxsw = abi::EngineContextProperties {
        engine: crate::gpu::ENGINE_GRAPHICS_ZCULL,
        ..Default::default()
    };
    d.client.control(d.handle, 0x801707, &mut ctxsw)?;
    let divisors = [
        info.subregion_width_align_pixels,
        info.subregion_height_align_pixels,
        info.pixel_squares_by_aliquots,
        info.subregion_count,
    ];
    if divisors.contains(&0) || info.subregion_count > ZCULL_MAX_SUBREGIONS {
        return Err(Error::Contract(format!(
            "GR_GET_ZCULL_INFO outside the supported range (subregion align {}x{}, \
             pixels per aliquot {}, subregions {} of at most {ZCULL_MAX_SUBREGIONS})",
            info.subregion_width_align_pixels,
            info.subregion_height_align_pixels,
            info.pixel_squares_by_aliquots,
            info.subregion_count,
        )));
    }
    if ctxsw.size == 0 {
        return Err(Error::Contract(
            "RM reports no GRAPHICS_ZCULL context size".into(),
        ));
    }
    Ok((info, ctxsw))
}
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
fn guarded(entry: &'static str, bridge: Option<&Bridge>, f: impl FnOnce() -> i32) -> i32 {
    crate::ffi::guard(
        entry,
        bridge.map(|b| -> &dyn crate::ffi::Failable { b }),
        || -4,
        f,
    )
}
/// The bridge argument of an entry point, or the named error for NULL.
fn present(bridge: Option<&Bridge>) -> Result<&Bridge> {
    bridge.ok_or_else(|| Error::Contract("null RM bridge".into()))
}
/// The `VkResult` an `nv_rm_*` entry point returns for `error` (`nv_rm.h`; `nvkmd_rm.c`
/// passes it through as the `VkResult`): `VK_ERROR_DEVICE_LOST` (-4) for a lost
/// device, `VK_ERROR_OUT_OF_HOST_MEMORY` (-1) for an `ENOMEM` syscall, and
/// `VK_ERROR_INITIALIZATION_FAILED` (-3) for every other failure.
fn vk_status(error: &Error) -> i32 {
    match error {
        Error::DeviceLost(_) => -4,
        Error::Io(_, error) if error.raw_os_error() == Some(12) => -1,
        Error::Io(..)
        | Error::Rm { .. }
        | Error::Contract(_)
        | Error::Wait(_)
        | Error::BarPressure { .. } => -3,
    }
}
fn report(result: Result<()>) -> i32 {
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
unsafe fn output<T>(p: *mut T, value: T) -> Result<()> {
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
unsafe fn input<'a, T>(ptr: *const T, count: u32, maximum: u32) -> Result<&'a [T]> {
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
fn status_of(result: Result<SubmitStatus>, out: *mut u64) -> i32 {
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
        unsafe { output(out, token) }
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
    guarded("nv_rm_open", None, move || {
        report((|| {
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
        })())
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
enum Placement {
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
    guarded("nv_rm_alloc", object, move || {
        report((|| {
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
            let size = usize::try_from(size)
                .map_err(|error| Error::Contract(format!("memory size {size}: {error}")))?;
            let alignment = usize::try_from(alignment.max(4096)).map_err(|error| {
                Error::Contract(format!("memory alignment {alignment}: {error}"))
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
                Placement::Vram | Placement::Gart => s.device.allocate_aligned(
                    size,
                    alignment,
                    false,
                    placement == Placement::Gart,
                )?,
            };
            let id = next_id(s)?;
            let a = Allocation {
                id,
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
        })())
    })
}
fn next_id(s: &mut Inner) -> Result<u64> {
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
pub(crate) unsafe extern "C" fn nv_rm_map(p: *mut Bridge, memory: u64, out: *mut usize) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_map", object, move || {
        report((|| {
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
pub(crate) unsafe extern "C" fn nv_rm_unmap(p: *mut Bridge, memory: u64) -> i32 {
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
pub(crate) unsafe extern "C" fn nv_rm_free(p: *mut Bridge, memory: u64) -> i32 {
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
    memory: u64,
    offset: u64,
    size: u64,
    address: u64,
    kind: u32,
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_bind", object, move || {
        report((|| {
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
            let map = s
                .space
                .map_range(&buffer.memory, offset, size, Some(address), kind)?;
            if map.address() != address {
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
            unsafe { output(out, id) }
        })())
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
pub(crate) unsafe extern "C" fn nv_rm_unbind(p: *mut Bridge, mapping: u64) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_unbind", object, move || {
        report((|| {
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
        })())
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
    mappings: *const u64,
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
                    .get(id)
                    .ok_or_else(|| Error::Contract("VA free of unknown GPU mapping".into()))?;
                if map.address() < address
                    || map.address().saturating_add(map.length()) > end
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
                s.bind_info.remove(id);
            }
            freeing.extend(ids.iter().map(|id| {
                ReclaimItem::Mapping(
                    s.maps
                        .remove(id)
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
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_create", object, move || {
        report((|| {
            if out.is_null() {
                return Err(Error::Contract("null context output".into()));
            }
            let b = present(object)?;
            b.core.live()?;
            let mut guard = b.lock()?;
            let id = b.create_context(&mut guard.0, engines, None)?;
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, id) }
        })())
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
    share: u64,
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_create_shared", object, move || {
        report((|| {
            if out.is_null() {
                return Err(Error::Contract("null context output".into()));
            }
            let b = present(object)?;
            b.core.live()?;
            let mut guard = b.lock()?;
            let id = b.create_context(&mut guard.0, engines, Some(share))?;
            // SAFETY: nv_rm.h: an output pointer is NULL or writable for the call; output() rejects NULL.
            unsafe { output(out, id) }
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
pub(crate) unsafe extern "C" fn nv_rm_context_destroy(p: *mut Bridge, id: u64) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_destroy", object, move || {
        report((|| {
            let b = present(object)?;
            let token = queue_ref(&b.lock()?.0, id)?.last_token();
            // Bounded quiesce: an unsignaled (e.g. foreign-gated) batch is lost
            // after 5 s; its acquires are then released so the channel runs
            // out, and only the idle channel is RM-freed (no schedule-off
            // first, as stock) before memory is released.
            // A wait that fails (device lost) is handled like one that ran out.
            let drained = b
                .wait_token(id, token, 5_000_000_000)
                .unwrap_or_else(|error| {
                    diag!("nvk Rust RM context {id} destroy drain: {error}");
                    false
                });
            if !drained {
                let pending = {
                    let mut guard = b.lock()?;
                    b.lose_locked(&mut guard.0);
                    b.unblock_locked(&mut guard.0, id)
                };
                // Released acquires let the channel run out before its free.
                b.quiesce(pending.as_slice(), Duration::from_secs(5));
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
            Bridge::check_idle_before_free(id, &queue);
            s.acquires[context_index(id).expect("the queue of this id was just taken")].clear();
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
    id: u64,
    pushes: *const Push,
    count: u32,
    drm_fd: i32,
    wait_fds: *const i32,
    wait_count: u32,
    out: *mut u64,
) -> i32 {
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
    id: u64,
    pushes: *const Push,
    count: u32,
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_submit", object, move || {
        // SAFETY: forwards this entry point's own arguments, under its contract.
        unsafe { nv_rm_submit_waits(p, id, pushes, count, -1, std::ptr::null(), 0, out) }
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
    id: u64,
    token: u64,
    complete: *mut u32,
) -> i32 {
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
    id: u64,
    token: u64,
    timeout_ns: u64,
) -> i32 {
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
    id: u64,
    token: u64,
    drm_fd: i32,
    out: *mut i32,
) -> i32 {
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
    id: u64,
    pushes: *const Push,
    count: u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_exec", object, move || {
        loop {
            let mut token = 0;
            // SAFETY: this entry point's own arguments, under the same
            // contract; `token` is a live local.
            let result = unsafe { nv_rm_submit(p, id, pushes, count, &raw mut token) };
            if result != 0 && result != 1 {
                return result;
            }
            // Synchronous by contract (nv_rm.h nv_rm_exec), so unbounded like
            // nvkmd_rm.c's context sync; see `deadline` for how loss ends it.
            // SAFETY: this entry point's own device argument.
            let wait = unsafe { nv_rm_wait(p, id, token, u64::MAX) };
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
    memory: u64,
    drm_fd: i32,
    out: *mut i32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_export", object, move || {
        report((|| {
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
pub(crate) unsafe extern "C" fn nv_rm_import(
    p: *mut Bridge,
    drm_fd: i32,
    dma_fd: i32,
    out: *mut Allocation,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_import", object, move || {
        report((|| {
            if drm_fd < 0 || dma_fd < 0 || out.is_null() {
                return Err(Error::Contract("invalid dma-buf import arguments".into()));
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
                id,
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
        })())
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
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_create", object, move || {
        report((|| {
            if out.is_null() || flags & !7 != 0 {
                return Err(Error::Contract("invalid sync create arguments".into()));
            }
            let b = present(object)?;
            b.core.live()?;
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
            unsafe { output(out, id) }
        })())
    })
}
/// # Safety
/// `id` is a live nv_rm_sync_create payload, destroyed exactly once, with no
/// other call using it in flight or after; the device pointer is unused.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_destroy(_p: *mut Bridge, id: u64) -> i32 {
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
pub(crate) unsafe extern "C" fn nv_rm_sync_destroy_deferred(p: *mut Bridge, id: u64) -> i32 {
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
fn needs_slot(error: &Error) -> bool {
    matches!(error, Error::Contract(m) if m.ends_with("needs a replacement slot"))
}
/// # Safety
/// nv_rm.h's contract for every entry point: the device pointer is NULL or a
/// live nv_rm_open device that stays alive for the whole call; every output
/// pointer is NULL or writable, and every input array readable for its count;
/// every descriptor argument is open for the call; every sync id is 0 or a
/// live nv_rm_sync_create payload.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_reset(p: *mut Bridge, id: u64) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_reset", object, move || {
        report((|| {
            let b = present(object)?;
            b.core.live()?;
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            let sync = unsafe { sync_ref(id) }?;
            let mut fresh = if sync.needs_fresh_slot() {
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
pub(crate) unsafe extern "C" fn nv_rm_sync_move(
    p: *mut Bridge,
    dst: u64,
    src: u64,
    _shared_source: u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_move", object, move || {
        report((|| {
            let b = present(object)?;
            b.core.live()?;
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
            target.install_moved(moved);
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
pub(crate) unsafe extern "C" fn nv_rm_sync_signal(p: *mut Bridge, id: u64, value: u64) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_signal", object, move || {
        report((|| {
            let b = present(object)?;
            b.core.live()?;
            // SAFETY: nv_rm.h: a sync id argument is 0 or a live nv_rm_sync_create payload for the whole call.
            let sync = unsafe { sync_ref(id) }?;
            let mut fresh = if !sync.timeline && sync.needs_fresh_slot() {
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
pub(crate) unsafe extern "C" fn nv_rm_sync_get_value(
    p: *mut Bridge,
    id: u64,
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
    id: u64,
    fd: i32,
    opaque: u32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_import_fd", object, move || {
        report((|| {
            let b = present(object)?;
            b.core.live()?;
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
pub(crate) unsafe extern "C" fn nv_rm_sync_export_fd(
    p: *mut Bridge,
    id: u64,
    opaque: u32,
    drm_fd: i32,
    out: *mut i32,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_sync_export_fd", object, move || {
        report((|| {
            if out.is_null() {
                return Err(Error::Contract("null synchronization fd output".into()));
            }
            let b = present(object)?;
            b.core.live()?;
            if b.capture.load(Ordering::Relaxed)
                && let Some(cap) = b.lock()?.0.capture.as_mut()
            {
                cap.mark(0, id, 2 + opaque);
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
        })())
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
    dst: u64,
    src: u64,
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
    id: u64,
    pushes: *const Push,
    count: u32,
    waits: *const SyncPoint,
    wait_count: u32,
    signals: *const SyncPoint,
    signal_count: u32,
    out: *mut u64,
) -> i32 {
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
    id: u64,
    pushes: *const Push,
    count: u32,
    waits: *const SyncPoint,
    wait_count: u32,
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_submit_sync", object, move || {
        // SAFETY: forwards this entry point's own arguments, under its contract.
        unsafe {
            nv_rm_submit_batch(
                p,
                id,
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
    id: u64,
    signals: *const SyncPoint,
    count: u32,
    out: *mut u64,
) -> i32 {
    // SAFETY: nv_rm.h: the device argument is NULL or a live nv_rm_open device, kept alive by the caller for the whole call.
    let object = unsafe { p.as_ref() };
    guarded("nv_rm_context_signal", object, move || {
        // SAFETY: forwards this entry point's own arguments, under its contract.
        unsafe {
            nv_rm_submit_batch(
                p,
                id,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};
    #[test]
    fn status_codes_match_the_c_abi() {
        // nvkmd_rm.c rm_result() returns the status as the VkResult.
        let io = |errno| Error::Io("op", std::io::Error::from_raw_os_error(errno));
        assert_eq!(vk_status(&Error::DeviceLost("x".into())), -4);
        assert_eq!(vk_status(&io(12)), -1);
        assert_eq!(vk_status(&io(22)), -3);
        assert_eq!(vk_status(&Error::Contract("x".into())), -3);
        assert_eq!(
            vk_status(&Error::Rm {
                operation: "op",
                status: 0x1f
            }),
            -3
        );
        let wait = WaitError::Timeout {
            operation: "op",
            object: "x".into(),
            limit: Duration::ZERO,
            elapsed: Duration::ZERO,
        };
        assert_eq!(vk_status(&wait.into()), -3);
        let bar = Error::BarPressure {
            used: 1,
            requested: 2,
            limit: 3,
        };
        assert_eq!(vk_status(&bar), -3);
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
            unsafe { nv_rm_sync_destroy_deferred(std::ptr::null_mut(), 7) },
            0
        );
        // SAFETY: as above.
        assert_ne!(unsafe { nv_rm_sync_reset(std::ptr::null_mut(), 7) }, 0);
        // SAFETY: id 0 is rejected before any dereference.
        assert!(unsafe { sync_ref(0) }.is_err());
    }
}

#[cfg(test)]
#[path = "mesa_hardware_tests.rs"]
mod hardware_tests;

#[cfg(test)]
#[path = "mesa_reclaim_tests.rs"]
mod reclaim_tests;

#[cfg(test)]
#[path = "mesa_multiqueue_tests.rs"]
mod multiqueue_tests;
