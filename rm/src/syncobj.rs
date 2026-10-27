// SPDX-License-Identifier: MIT
//! Caller-side synchronization payloads (Mesa vk_sync backing).
//!
//! Ordinary (process-local) payloads are one 16-byte slot of the device slot
//! pool (slots.rs). The mapped word is the only truth: status reads it, CPU
//! reset/signal write it, the GPU acquires/releases it. No CPU shadow counter,
//! publication job or point tree exists for these payloads.
//!
//! Binary GPU wait = detach. Vulkan defines a binary semaphore wait as setting
//! it unsignaled. Stock 615 encodes acquire==1 followed by a GPU release of 0
//! into the same slot. We instead encode the acquire on the current slot,
//! hand that slot's reference to the submission, and bind a fresh zeroed slot
//! to the semaphore (logically unsignaled, no pending signal). The old slot is
//! never rewritten by the CPU and is recycled only after the waiting batch has
//! retired (slots.rs lifetime rule), so no CPU reset can ever clear a slot a
//! pending GPU acquire still needs, and a later signal (possibly on another
//! queue) never targets a slot an earlier wait has yet to consume. The device
//! runs Mesa's THREADED_ON_DEMAND submit mode (rm-integration.patch,
//! nvk_device.c), so Mesa may additionally reset a permanent binary payload
//! after a caller-side submission that waited on it (pinned vk_queue.c
//! IMMEDIATE branch) or move it into a temporary before a submit-thread
//! submission (THREADED branch). After detach-on-wait the payload already
//! holds a fresh zeroed slot it owns alone, so that reset is an in-place store
//! of 0 to an unsignaled slot; a move transfers the payload (take_for_move).
//!
//! CPU reset (vkResetFences, copy-export reset) writes 0 in place when the CPU
//! owner holds the only reference (no unretired submission references the
//! slot); otherwise the payload gets a fresh slot and the captured slot stays
//! untouched until its submissions retire. CPU binary signal follows the same
//! rule. Timeline values are monotonic full-width words (slots.rs).
//!
//! Imported SYNC_FD payloads keep their owned descriptor and cache only a
//! terminal status. OPAQUE_FD keeps this driver's private sealed-memfd format
//! (sync_memory.rs, unchanged ABI): CPU atomics shared across processes, with
//! local GPU signal points published into it by caller paths and the watcher.
use crate::{
    rm::{Error, Result},
    slots::{Slot, SlotPool},
    sync_memory::SyncMemory,
};
use std::{
    collections::BTreeMap,
    os::fd::{AsRawFd, BorrowedFd, OwnedFd, RawFd},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicI32, AtomicU64, Ordering},
    },
};

const MAGIC: u64 = 0x4e56_524d_5359_4e32; // "NVRMSYN2"

/// An imported sync_file generation. Linux sync_file.h status 0 = active;
/// dma-fence publishes errors before signalling, and fences::sync_status
/// checks every child, so a terminal status (success or error) is permanent.
pub(crate) struct Foreign {
    fd: OwnedFd,
    terminal: AtomicI32,
}
impl Foreign {
    pub(crate) fn new(fd: OwnedFd, status: i32) -> Self {
        Self {
            fd,
            terminal: AtomicI32::new(status),
        }
    }
    /// Terminal status, querying the kernel only while still pending. A failed
    /// ioctl leaves the cache untouched (never success).
    pub(crate) fn status(&self) -> Result<i32> {
        let cached = self.terminal.load(Ordering::Acquire);
        if cached != 0 {
            return Ok(cached);
        }
        let status = crate::fences::sync_status(self.fd.as_raw_fd())?;
        if status != 0 {
            self.terminal.store(status, Ordering::Release);
        }
        Ok(status)
    }
    pub(crate) fn cached(&self) -> i32 {
        self.terminal.load(Ordering::Acquire)
    }
    /// Test stand-in for the kernel reporting a terminal status: later
    /// `status` calls return it without the `sync_file` ioctl.
    #[cfg(test)]
    pub(crate) fn settle(&self, status: i32) {
        self.terminal.store(status, Ordering::Release);
    }
    pub(crate) fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
    pub(crate) fn duplicate(&self) -> Result<OwnedFd> {
        self.fd
            .try_clone()
            .map_err(|e| Error::Io("duplicate sync_file", e))
    }
    /// Readiness predicate: error never becomes success.
    pub(crate) fn ready(&self, pending: bool) -> Result<bool> {
        let status = self.status()?;
        if status < 0 {
            return Err(Error::DeviceLost(format!(
                "imported sync_file producer error {status}"
            )));
        }
        Ok(pending || status > 0)
    }
}

// Private opaque memfd protocol (format unchanged from the event design).
const SHARED_MAGIC: u64 = 0x4e56524d53594e01;
const COMPLETED: usize = 0;
const PENDING: usize = 8;
const TARGET: usize = 16;
const HEADER: usize = 24;
const KIND: usize = 32;
const ERROR: usize = 40;

/// Local GPU signals still to be published into a shared payload.
#[derive(Default)]
struct SharedLocal {
    /// Binary/timeline points: a fresh binary slot released to 1 by the GPU
    /// stands for completion of `target` (one reference owned here).
    points: BTreeMap<u64, Slot>,
    /// A converted local timeline slot still receiving submitted GPU values
    /// up to `.1`; its 64-bit value is published as completion.
    sources: Vec<(Slot, u64)>,
}

/// OPAQUE_FD payload: fixed-size sealed memfd of AtomicU64 words shared across
/// processes (no GPU mapping, RM handle or process pointer inside).
pub(crate) struct Shared {
    memory: SyncMemory,
    pub(crate) timeline: bool,
    pool: Arc<SlotPool>,
    local: Mutex<SharedLocal>,
}
impl Shared {
    fn word(&self, offset: usize) -> &AtomicU64 {
        self.memory.word(offset)
    }
    fn load(&self, offset: usize) -> u64 {
        self.word(offset).load(Ordering::Acquire)
    }
    /// True only for an actual change of the shared word.
    fn maximum(&self, offset: usize, value: u64) -> bool {
        self.word(offset).fetch_max(value, Ordering::AcqRel) < value
    }
    fn store(&self, offset: usize, value: u64) -> bool {
        self.word(offset).swap(value, Ordering::AcqRel) != value
    }
    fn create(pool: &Arc<SlotPool>, timeline: bool, words: [u64; 4]) -> Result<Self> {
        let memory = SyncMemory::new_cpu()?;
        let shared = Self {
            memory,
            timeline,
            pool: Arc::clone(pool),
            local: Mutex::new(SharedLocal::default()),
        };
        for (offset, value) in [COMPLETED, PENDING, TARGET, ERROR].into_iter().zip(words) {
            shared.word(offset).store(value, Ordering::Relaxed);
        }
        shared
            .word(KIND)
            .store(u64::from(timeline), Ordering::Relaxed);
        shared.word(HEADER).store(SHARED_MAGIC, Ordering::Release);
        Ok(shared)
    }
    fn import(pool: &Arc<SlotPool>, timeline: bool, fd: BorrowedFd<'_>) -> Result<Self> {
        let memory = SyncMemory::import_cpu(fd)?;
        let shared = Self {
            memory,
            timeline,
            pool: Arc::clone(pool),
            local: Mutex::new(SharedLocal::default()),
        };
        if shared.load(HEADER) != SHARED_MAGIC || shared.load(KIND) != u64::from(timeline) {
            return Err(Error::Contract(
                "opaque synchronization payload format/type mismatch".into(),
            ));
        }
        Ok(shared)
    }
    /// Local GPU points. A poisoned lock (a thread panicked inside it; the FFI
    /// guard has marked the device lost) is fatal for the payload (RUST.md §8).
    fn local(&self) -> Result<MutexGuard<'_, SharedLocal>> {
        self.local.lock().map_err(|error| {
            Error::DeviceLost(format!(
                "shared synchronization payload lock: {error} (a thread panicked while holding it)"
            ))
        })
    }
    /// Publish completed local GPU points. Returns whether the shared word
    /// changed (the caller then broadcasts HOST once).
    pub(crate) fn refresh(&self) -> Result<bool> {
        let mut local = self.local()?;
        let mut completed = self.load(COMPLETED);
        for (&target, &slot) in &local.points {
            if self.pool.read32(slot)? == 1 {
                completed = completed.max(target);
            }
        }
        for &(slot, _) in &local.sources {
            completed = completed.max(self.pool.timeline_value(slot)?);
        }
        let changed = self.maximum(COMPLETED, completed);
        let completed = self.load(COMPLETED);
        let pool = Arc::clone(&self.pool);
        local.points.retain(|target, slot| {
            let keep = *target > completed;
            if !keep {
                pool.release(*slot);
            }
            keep
        });
        local.sources.retain(|&(slot, last)| {
            let keep = completed < last;
            if !keep {
                pool.release(slot);
            }
            keep
        });
        Ok(changed)
    }
    /// Whether local GPU points are still unpublished. A poisoned payload
    /// answers yes: work in an unknown state is never reported as done, so
    /// callers keep refreshing it (which fails) or fail it.
    pub(crate) fn has_local_work(&self) -> bool {
        match self.local() {
            Ok(local) => !local.points.is_empty() || !local.sources.is_empty(),
            Err(error) => {
                crate::diag::diag!("RM sync payload: {error}");
                true
            }
        }
    }
    pub(crate) fn completed(&self) -> u64 {
        self.load(COMPLETED)
    }
    /// Current binary generation.
    pub(crate) fn target(&self) -> u64 {
        self.load(TARGET)
    }
    pub(crate) fn export(&self) -> Result<OwnedFd> {
        self.memory.export()
    }
    pub(crate) fn failed(&self) -> bool {
        self.load(ERROR) != 0
    }
    /// Completed values stay completed; only unfinished points become errors.
    pub(crate) fn fail(&self) -> bool {
        // Record what did complete before failing the rest. A slot that cannot
        // be read any more counts as unfinished: it becomes an error below.
        if let Err(error) = self.refresh() {
            crate::diag::diag!("RM sync payload: completion read before failing: {error}");
        }
        self.store(ERROR, 1)
    }
    fn ready(&self, target: u64, pending: bool) -> Result<bool> {
        if self.load(COMPLETED) >= target {
            return Ok(true);
        }
        if self.failed() {
            return Err(Error::DeviceLost(
                "shared synchronization producer failed".into(),
            ));
        }
        Ok(pending && self.load(PENDING) >= target)
    }
}
impl Drop for Shared {
    fn drop(&mut self) {
        let Ok(local) = self.local.get_mut() else {
            // A poisoned payload's slots stay out of circulation.
            crate::diag::diag!("RM sync payload dropped with a poisoned lock; its slots are kept");
            return;
        };
        let local = std::mem::take(local);
        for (_, slot) in local.points {
            self.pool.release(slot);
        }
        for (slot, _) in local.sources {
            self.pool.release(slot);
        }
    }
}

pub(crate) enum Payload {
    /// One reference to a pool slot (binary 0/1, or timeline value).
    Native(Slot),
    Foreign(Arc<Foreign>),
    Shared(Arc<Shared>),
}

/// `SyncState::pending` of a binary payload whose signal operation was
/// reserved for a submit-thread submission ([`SyncObj::reserve_into`]).
/// Queue contexts are numbered from 1, so context 0 names no queue.
pub(crate) const RESERVED: (u64, u64) = (0, 0);

pub(crate) struct SyncState {
    pub(crate) payload: Payload,
    /// Native binary: the accepted submission (context id, token) whose GPU
    /// release targets the current slot. Used by SYNC_FD export/WAIT_PENDING.
    pub(crate) pending: Option<(u64, u64)>,
    /// Native timeline: maximum submitted GPU signal / host value (WAIT_PENDING).
    pub(crate) pending_max: u64,
    /// Shared binary: generation retained by a moved/captured handle.
    pub(crate) snapshot: Option<u64>,
}

/// Mesa vk_sync payload. The C side holds an Arc reference as its 64-bit id.
pub(crate) struct SyncObj {
    magic: u64,
    pub(crate) timeline: bool,
    pool: Arc<SlotPool>,
    state: Mutex<SyncState>,
}

/// What a GPU wait must encode for one payload, decided under its lock.
pub(crate) enum WaitPlan {
    /// Already satisfied: nothing to encode.
    Complete,
    /// Acquire == 1 on a binary slot whose reference moves to the submission
    /// (the payload already has its fresh replacement slot).
    Binary(Slot),
    /// Acquire >= value on a retained timeline slot (+1 reference for the batch).
    Timeline(Slot, u64),
    /// Pending foreign sync_file: needs a CPU-released gate.
    Foreign(Arc<Foreign>),
    /// Pending shared payload generation/value: needs a CPU-released gate.
    Shared(Arc<Shared>, u64),
}

impl SyncObj {
    /// `slot` is a freshly allocated pool slot (one reference, zeroed).
    pub(crate) fn new(
        pool: &Arc<SlotPool>,
        timeline: bool,
        initial: u64,
        slot: Slot,
    ) -> Result<Self> {
        if timeline {
            pool.init_timeline(slot, initial)?;
        } else if initial != 0 {
            pool.store32(slot, 1)?;
        }
        Ok(Self {
            magic: MAGIC,
            timeline,
            pool: Arc::clone(pool),
            state: Mutex::new(SyncState {
                payload: Payload::Native(slot),
                pending: None,
                pending_max: if timeline { initial } else { 0 },
                snapshot: None,
            }),
        })
    }
    pub(crate) fn valid(&self) -> bool {
        self.magic == MAGIC
    }
    /// The object's state. A poisoned lock (a thread panicked inside it; the
    /// FFI guard has marked the device lost) is fatal for the object
    /// (RUST.md §8).
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, SyncState>> {
        self.state.lock().map_err(|error| {
            Error::DeviceLost(format!(
                "synchronization object lock: {error} (a thread panicked while holding it)"
            ))
        })
    }

    /// Replace the payload, releasing the old native slot reference.
    fn replace(&self, state: &mut SyncState, payload: Payload) {
        let old = std::mem::replace(&mut state.payload, payload);
        if let Payload::Native(slot) = old {
            self.pool.release(slot);
        }
        state.pending = None;
        state.snapshot = None;
    }

    fn shared_target(&self, state: &SyncState, shared: &Shared, value: u64) -> u64 {
        if self.timeline {
            value
        } else {
            state.snapshot.unwrap_or_else(|| shared.load(TARGET))
        }
    }

    /// Nonblocking predicate. `pending` = VK_SYNC_WAIT_PENDING. For shared
    /// payloads the local points are published first; the returned flag says
    /// whether a cross-process broadcast is owed.
    pub(crate) fn ready(&self, value: u64, pending: bool) -> Result<(bool, bool)> {
        let state = self.lock()?;
        match &state.payload {
            Payload::Native(slot) => {
                let ready = if self.timeline {
                    self.pool.timeline_value(*slot)? >= value
                        || (pending && state.pending_max >= value)
                } else {
                    self.pool.read32(*slot)? == 1 || (pending && state.pending.is_some())
                };
                if ready {
                    crate::platform::device_read_barrier();
                }
                Ok((ready, false))
            }
            Payload::Foreign(foreign) => Ok((foreign.ready(pending)?, false)),
            Payload::Shared(shared) => {
                let target = self.shared_target(&state, shared, value);
                let shared = Arc::clone(shared);
                drop(state);
                let changed = shared.refresh()?;
                Ok((shared.ready(target, pending)?, changed))
            }
        }
    }

    /// Pending foreign descriptor a blocked CPU waiter must watch, if any.
    pub(crate) fn foreign(&self) -> Result<Option<Arc<Foreign>>> {
        Ok(match &self.lock()?.payload {
            Payload::Foreign(f) if f.cached() == 0 => Some(Arc::clone(f)),
            _ => None,
        })
    }

    /// Timeline value (vkGetSemaphoreCounterValue).
    pub(crate) fn value(&self) -> Result<(u64, bool)> {
        let state = self.lock()?;
        match &state.payload {
            Payload::Native(slot) => Ok((self.pool.timeline_value(*slot)?, false)),
            Payload::Shared(shared) => {
                let shared = Arc::clone(shared);
                drop(state);
                // Completed values remain readable after a producer failure;
                // the failure surfaces on waits for unfinished values.
                let changed = shared.refresh()?;
                Ok((shared.completed(), changed))
            }
            Payload::Foreign(_) => Err(Error::Contract("sync_file is not a timeline".into())),
        }
    }

    /// CPU reset of a binary payload. `fresh` is a pre-allocated slot used if
    /// the current slot is still referenced by unretired submissions; returns
    /// it unused otherwise (the caller releases it). Returns (unused fresh,
    /// shared broadcast owed).
    pub(crate) fn reset(&self, fresh: Option<Slot>) -> Result<(Option<Slot>, bool)> {
        if self.timeline {
            return Err(Error::Contract("reset of timeline payload".into()));
        }
        let mut state = self.lock()?;
        match &state.payload {
            Payload::Native(slot) if self.pool.refs(*slot) == 1 => {
                // Sole owner: no GPU acquire/release of this slot is pending.
                self.pool.store32(*slot, 0)?;
                state.pending = None;
                Ok((fresh, false))
            }
            Payload::Shared(shared) => {
                let target = shared
                    .load(TARGET)
                    .checked_add(1)
                    .ok_or_else(|| Error::Contract("binary payload generation exhausted".into()))?;
                let changed = shared.store(TARGET, target);
                state.snapshot = None;
                Ok((fresh, changed))
            }
            _ => {
                let Some(fresh) = fresh else {
                    return Err(Error::Contract(
                        "binary reset needs a replacement slot".into(),
                    ));
                };
                self.replace(&mut state, Payload::Native(fresh));
                Ok((None, false))
            }
        }
    }
    /// Whether reset()/signal() may need a replacement slot (racy hint used to
    /// pre-allocate outside the payload lock; reset() re-checks under it).
    pub(crate) fn needs_fresh_slot(&self) -> Result<bool> {
        Ok(match &self.lock()?.payload {
            Payload::Native(slot) => self.pool.refs(*slot) != 1,
            Payload::Foreign(_) => true,
            Payload::Shared(_) => false,
        })
    }

    /// CPU signal. Binary: set 1 (fresh slot if captured / foreign). Timeline:
    /// monotonic host signal. Returns (unused fresh, shared broadcast owed);
    /// the caller notifies local waiters afterwards.
    pub(crate) fn signal(&self, value: u64, fresh: Option<Slot>) -> Result<(Option<Slot>, bool)> {
        let mut state = self.lock()?;
        match &state.payload {
            Payload::Native(slot) if self.timeline => {
                self.pool.host_signal(*slot, value)?;
                state.pending_max = state.pending_max.max(value);
                Ok((fresh, false))
            }
            Payload::Native(slot) if self.pool.refs(*slot) == 1 => {
                self.pool.store32(*slot, 1)?;
                Ok((fresh, false))
            }
            Payload::Shared(shared) => {
                let target = self.shared_target(&state, shared, value);
                let shared = Arc::clone(shared);
                drop(state);
                let mut changed = shared.maximum(PENDING, target);
                changed |= shared.maximum(COMPLETED, target);
                Ok((fresh, changed))
            }
            _ => {
                // vk_sync_import_sync_file(-1) signals; a captured slot or a
                // foreign payload is replaced, never written.
                let Some(fresh) = fresh else {
                    return Err(Error::Contract(
                        "binary signal needs a replacement slot".into(),
                    ));
                };
                if self.timeline {
                    return Err(Error::Contract(
                        "cannot CPU-signal imported sync_file".into(),
                    ));
                }
                self.pool.store32(fresh, 1)?;
                self.replace(&mut state, Payload::Native(fresh));
                Ok((None, false))
            }
        }
    }

    /// Plan a GPU wait (submission phase 1). Binary native waits detach: the
    /// payload is rebound to `fresh` only when `commit` is true (phase 3).
    pub(crate) fn plan_wait(&self, state: &SyncState, value: u64) -> Result<WaitPlan> {
        match &state.payload {
            Payload::Native(slot) if self.timeline => {
                if self.pool.timeline_value(*slot)? >= value {
                    Ok(WaitPlan::Complete)
                } else {
                    self.pool.check_split_distance(*slot, value)?;
                    Ok(WaitPlan::Timeline(*slot, value))
                }
            }
            Payload::Native(slot) => Ok(WaitPlan::Binary(*slot)),
            Payload::Foreign(foreign) => {
                // New API request: fresh status unless terminal (never success
                // on error; an error rejects the submission).
                if foreign.ready(false)? {
                    Ok(WaitPlan::Complete)
                } else {
                    Ok(WaitPlan::Foreign(Arc::clone(foreign)))
                }
            }
            Payload::Shared(shared) => {
                let target = self.shared_target(state, shared, value);
                shared.refresh()?;
                if shared.ready(target, false)? {
                    Ok(WaitPlan::Complete)
                } else {
                    Ok(WaitPlan::Shared(Arc::clone(shared), target))
                }
            }
        }
    }
    /// Whether a GPU wait on this payload consumes it (needs a fresh slot).
    pub(crate) fn wait_consumes(&self, state: &SyncState) -> bool {
        !self.timeline && matches!(state.payload, Payload::Native(_) | Payload::Foreign(_))
    }

    /// GPU signal target for a submission: the native slot (timeline value or
    /// binary 1), or for shared payloads the generation/value a fresh point
    /// stands for. Foreign signal targets are replaced by `fresh` first.
    pub(crate) fn plan_signal(&self, state: &SyncState, value: u64) -> Result<SignalPlan> {
        match &state.payload {
            Payload::Native(slot) => Ok(if self.timeline {
                SignalPlan::Timeline(*slot, value)
            } else {
                SignalPlan::Binary(*slot)
            }),
            Payload::Foreign(_) => Ok(SignalPlan::ReplaceForeign),
            Payload::Shared(shared) => {
                let target = if self.timeline {
                    value
                } else if let Some(reserved) = state.snapshot {
                    // A reservation temporary (reserve_into) or a moved
                    // handle: the generation was fixed when it was captured.
                    reserved
                } else {
                    let target = shared.load(TARGET);
                    if shared.load(PENDING) >= target {
                        target.checked_add(1).ok_or_else(|| {
                            Error::Contract("binary payload generation exhausted".into())
                        })?
                    } else {
                        target
                    }
                };
                Ok(SignalPlan::Shared(Arc::clone(shared), target))
            }
        }
    }
    pub(crate) fn commit_signal(
        &self,
        state: &mut SyncState,
        context: u64,
        token: u64,
        value: u64,
    ) {
        if let Payload::Native(_) = state.payload {
            if self.timeline {
                state.pending_max = state.pending_max.max(value);
            } else {
                state.pending = Some((context, token));
            }
        }
    }
    /// Bind a deferred binary signal operation to this payload's current
    /// slot (Mesa submit-thread submissions, rm-integration.patch
    /// `vk_queue_reserve_binary_signals`). Vulkan counts the signal as
    /// submitted when vkQueueSubmit returns; the GPU release happens later on
    /// the submit thread. `dst` (a private temporary that the deferred batch
    /// signals) takes a reference to the slot, and this payload records the
    /// [`RESERVED`] pending signal, so WAIT_PENDING, binary waits/moves and
    /// SYNC_FD export observe the signal before it reaches a ring, and a later
    /// reset/move/consume of this payload cannot redirect it. Returns false
    /// (nothing changed) when the payload is not an unsignaled native binary
    /// without a pending signal; the caller then keeps the semaphore itself.
    /// Shared payloads reserve the generation the signal will complete
    /// (published as PENDING now, cross-process); returns (reserved,
    /// shared broadcast owed).
    pub(crate) fn reserve_into(&self, dst: &SyncObj) -> Result<(bool, bool)> {
        if self.timeline || dst.timeline || std::ptr::eq(self, dst) {
            return Err(Error::Contract("invalid binary signal reservation".into()));
        }
        let mut state = self.lock()?;
        let slot = match &state.payload {
            Payload::Native(slot) => *slot,
            Payload::Shared(shared) => {
                let shared = Arc::clone(shared);
                drop(state);
                // Same generation choice as plan_signal, fixed now.
                let mut target = shared.load(TARGET);
                if shared.load(PENDING) >= target {
                    target = target.checked_add(1).ok_or_else(|| {
                        Error::Contract("binary payload generation exhausted".into())
                    })?;
                }
                let mut changed = shared.maximum(TARGET, target);
                changed |= shared.maximum(PENDING, target);
                let mut reserved = dst.lock()?;
                dst.replace(&mut reserved, Payload::Shared(shared));
                reserved.snapshot = Some(target);
                return Ok((true, changed));
            }
            Payload::Foreign(_) => return Ok((false, false)),
        };
        if state.pending.is_some() || self.pool.read32(slot)? == 1 {
            return Ok((false, false));
        }
        self.pool.retain(slot);
        state.pending = Some(RESERVED);
        drop(state);
        let mut target = dst.lock()?;
        dst.replace(&mut target, Payload::Native(slot));
        Ok((true, false))
    }
    pub(crate) fn rebind_native(&self, state: &mut SyncState, fresh: Slot) {
        self.replace(state, Payload::Native(fresh));
    }

    /// Opaque export: convert a local payload to the shared memfd format once,
    /// transferring the local slot (and any pending GPU release into it) to
    /// the shared payload's publication sources. Returns the payload.
    pub(crate) fn share(&self) -> Result<Arc<Shared>> {
        let mut state = self.lock()?;
        let slot = match &state.payload {
            Payload::Shared(shared) => return Ok(Arc::clone(shared)),
            Payload::Foreign(_) => {
                return Err(Error::Contract(
                    "sync_file payload cannot be exported as opaque payload".into(),
                ));
            }
            Payload::Native(slot) => *slot,
        };
        let shared = if self.timeline {
            let completed = self.pool.timeline_value(slot)?;
            let pending = state.pending_max.max(completed);
            let shared = Shared::create(&self.pool, true, [completed, pending, 1, 0])?;
            if pending > completed {
                shared.local()?.sources.push((slot, pending));
            } else {
                self.pool.release(slot);
            }
            shared
        } else {
            let signaled = self.pool.read32(slot)? == 1;
            let pending = state.pending.is_some();
            let shared = Shared::create(
                &self.pool,
                false,
                [u64::from(signaled), u64::from(signaled || pending), 1, 0],
            )?;
            if !signaled && pending {
                shared.local()?.points.insert(1, slot);
            } else {
                self.pool.release(slot);
            }
            shared
        };
        let shared = Arc::new(shared);
        // The slot reference moved into `shared`; do not release it again.
        state.payload = Payload::Shared(Arc::clone(&shared));
        state.pending = None;
        state.snapshot = None;
        Ok(shared)
    }
    pub(crate) fn import_opaque(&self, fd: BorrowedFd<'_>) -> Result<()> {
        // Transactional: validate fully before replacing the payload.
        let shared = Arc::new(Shared::import(&self.pool, self.timeline, fd)?);
        let mut state = self.lock()?;
        self.replace(&mut state, Payload::Shared(shared));
        Ok(())
    }
    pub(crate) fn import_sync_file(&self, fd: BorrowedFd<'_>) -> Result<()> {
        if self.timeline {
            return Err(Error::Contract("SYNC_FD timeline import is invalid".into()));
        }
        let status = crate::fences::sync_status(fd.as_raw_fd())?;
        let owned = fd
            .try_clone_to_owned()
            .map_err(|e| Error::Io("retain imported sync_file", e))?;
        let foreign = Arc::new(Foreign::new(owned, status));
        let mut state = self.lock()?;
        self.replace(&mut state, Payload::Foreign(foreign));
        Ok(())
    }
    /// Mesa move (binary only): dst takes src's payload, src becomes a fresh
    /// unsignaled native payload — or, for a shared source, both keep the
    /// shared memory with dst retaining the old generation (src advances).
    pub(crate) fn take_for_move(&self, fresh: Slot) -> Result<(SyncState, bool)> {
        if self.timeline {
            return Err(Error::Contract("invalid timeline sync move".into()));
        }
        let mut state = self.lock()?;
        if let Payload::Shared(shared) = &state.payload {
            let shared = Arc::clone(shared);
            let generation = state.snapshot.unwrap_or_else(|| shared.load(TARGET));
            let next = shared
                .load(TARGET)
                .checked_add(1)
                .ok_or_else(|| Error::Contract("binary payload generation exhausted".into()))?;
            let changed = shared.store(TARGET, next);
            state.snapshot = None;
            self.pool.release(fresh);
            return Ok((
                SyncState {
                    payload: Payload::Shared(shared),
                    pending: None,
                    pending_max: 0,
                    snapshot: Some(generation),
                },
                changed,
            ));
        }
        let moved = std::mem::replace(
            &mut *state,
            SyncState {
                payload: Payload::Native(fresh),
                pending: None,
                pending_max: 0,
                snapshot: None,
            },
        );
        Ok((moved, false))
    }
    pub(crate) fn install_moved(&self, moved: SyncState) -> Result<()> {
        let mut state = self.lock()?;
        let old = std::mem::replace(&mut *state, moved);
        if let Payload::Native(slot) = old.payload {
            self.pool.release(slot);
        }
        Ok(())
    }
}
impl Drop for SyncObj {
    fn drop(&mut self) {
        match self.state.get_mut() {
            Ok(state) => {
                if let Payload::Native(slot) = state.payload {
                    self.pool.release(slot);
                }
            }
            // A poisoned object's slot stays out of circulation.
            Err(_) => crate::diag::diag!(
                "RM synchronization object dropped with a poisoned lock; its slot is kept"
            ),
        }
        self.magic = 0;
    }
}

pub(crate) enum SignalPlan {
    Binary(Slot),
    Timeline(Slot, u64),
    /// Signal on an imported sync_file payload: rebind to a fresh native slot.
    ReplaceForeign,
    Shared(Arc<Shared>, u64),
}

impl Shared {
    /// Record a local GPU point for `target` (submission accepted). The point
    /// reference moves here.
    pub(crate) fn add_point(&self, target: u64, point: Slot) -> Result<()> {
        let mut local = match self.local() {
            Ok(local) => local,
            Err(error) => {
                self.pool.release(point);
                return Err(error);
            }
        };
        if !self.timeline {
            // Monotonic: a signal reserved earlier (SyncObj::reserve_into)
            // commits after later moves/resets advanced the generation.
            self.maximum(TARGET, target);
        }
        if let Some(previous) = local.points.insert(target, point) {
            self.pool.release(previous);
        }
        drop(local);
        self.maximum(PENDING, target);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slots::{TimelineView, tests::HostPages};
    use std::os::fd::AsFd;

    fn fresh(pool: &Arc<SlotPool>) -> Slot {
        pool.try_allocate().unwrap().unwrap()
    }

    /// RUST.md §8: a poisoned object lock fails its users as device lost,
    /// and dropping the object keeps its slot out of circulation.
    #[test]
    fn poisoned_object_lock_fails_as_device_lost() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let sync = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let Payload::Native(slot) = sync.lock().unwrap().payload else {
            panic!("a new binary object has a native payload");
        };
        std::thread::scope(|scope| {
            let holder = scope.spawn(|| {
                let _state = sync.state.lock().unwrap();
                panic!("poison the object lock");
            });
            assert!(holder.join().is_err());
        });
        assert!(matches!(sync.lock(), Err(Error::DeviceLost(_))));
        assert!(matches!(sync.foreign(), Err(Error::DeviceLost(_))));
        assert!(matches!(sync.needs_fresh_slot(), Err(Error::DeviceLost(_))));
        drop(sync);
        assert_eq!(pool.refs(slot), 1);
    }

    #[test]
    fn binary_status_reset_in_place_only_without_gpu_references() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let sync = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        assert!(!sync.ready(0, false).unwrap().0);
        assert!(sync.signal(0, None).unwrap().0.is_none());
        assert!(sync.ready(0, false).unwrap().0);
        let Payload::Native(first) = sync.lock().unwrap().payload else {
            panic!()
        };
        // Sole owner: reset writes the same slot (stock same-slot reset).
        let (unused, _) = sync.reset(None).unwrap();
        assert!(unused.is_none());
        assert!(!sync.ready(0, false).unwrap().0);
        let Payload::Native(again) = sync.lock().unwrap().payload else {
            panic!()
        };
        assert_eq!(first, again);
        // A pending GPU release captures the slot: reset must not write it.
        sync.signal(0, None).unwrap();
        pool.retain(first);
        assert!(sync.needs_fresh_slot().unwrap());
        let replacement = fresh(&pool);
        assert!(sync.reset(Some(replacement)).unwrap().0.is_none());
        assert_eq!(pool.read32(first).unwrap(), 1, "captured slot untouched");
        let Payload::Native(now) = sync.lock().unwrap().payload else {
            panic!()
        };
        assert_eq!(now, replacement);
        assert!(!sync.ready(0, false).unwrap().0);
        assert_eq!(pool.refs(first), 1, "only the submission reference remains");
        pool.release(first); // submission retired → slot free
        assert_eq!(pool.refs(first), 0);
    }

    #[test]
    fn binary_gpu_wait_detaches_the_slot_and_leaves_payload_unsignaled() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let sync = SyncObj::new(&pool, false, 1, fresh(&pool)).unwrap();
        let mut state = sync.lock().unwrap();
        let WaitPlan::Binary(slot) = sync.plan_wait(&state, 0).unwrap() else {
            panic!()
        };
        assert!(sync.wait_consumes(&state));
        let replacement = fresh(&pool);
        pool.retain(slot); // the accepted batch's reference
        sync.rebind_native(&mut state, replacement);
        drop(state);
        // The waiting submission owns the old slot (still 1 until retirement).
        assert_eq!(pool.refs(slot), 1);
        assert_eq!(pool.read32(slot).unwrap(), 1);
        assert!(!sync.ready(0, false).unwrap().0);
        assert!(
            !sync.ready(0, true).unwrap().0,
            "no pending signal after wait"
        );
        // A later signal targets the fresh slot, never the consumed one.
        let SignalPlan::Binary(target) = sync.plan_signal(&sync.lock().unwrap(), 0).unwrap() else {
            panic!()
        };
        assert_eq!(target, replacement);
        sync.commit_signal(&mut sync.lock().unwrap(), 7, 42, 0);
        assert!(
            sync.ready(0, true).unwrap().0,
            "WAIT_PENDING sees the submission"
        );
        assert_eq!(sync.lock().unwrap().pending, Some((7, 42)));
        pool.release(slot);
    }

    #[test]
    fn move_transfers_slot_and_gives_source_a_fresh_one() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let src = SyncObj::new(&pool, false, 1, fresh(&pool)).unwrap();
        let dst = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let Payload::Native(old) = src.lock().unwrap().payload else {
            panic!()
        };
        let (moved, _) = src.take_for_move(fresh(&pool)).unwrap();
        dst.install_moved(moved).unwrap();
        assert!(dst.ready(0, false).unwrap().0);
        assert!(!src.ready(0, false).unwrap().0);
        let Payload::Native(now) = dst.lock().unwrap().payload else {
            panic!()
        };
        assert_eq!(now, old);
        drop(src);
        drop(dst);
        assert_eq!(pool.refs(old), 0);
    }

    #[test]
    fn reserved_binary_signal_is_pending_at_once_and_survives_move() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let src = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let temp = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let Payload::Native(slot) = src.lock().unwrap().payload else {
            panic!()
        };
        assert!(!src.ready(0, true).unwrap().0);
        assert_eq!(src.reserve_into(&temp).unwrap(), (true, false));
        assert!(src.ready(0, true).unwrap().0, "reserved signal is pending");
        assert!(!src.ready(0, false).unwrap().0, "but not complete");
        assert_eq!(src.lock().unwrap().pending, Some(RESERVED));
        // A second reservation (binary already has a pending signal) is refused.
        let other = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        assert_eq!(src.reserve_into(&other).unwrap(), (false, false));
        // A consumer captures the payload before the deferred signal runs.
        let consumer = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let (moved, _) = src.take_for_move(fresh(&pool)).unwrap();
        consumer.install_moved(moved).unwrap();
        // The deferred batch signals the temporary: it still targets the
        // captured slot, never the source's replacement.
        let SignalPlan::Binary(target) = temp.plan_signal(&temp.lock().unwrap(), 0).unwrap() else {
            panic!()
        };
        assert_eq!(target, slot);
        pool.gpu_write32(slot, 1);
        assert!(consumer.ready(0, false).unwrap().0);
        assert!(
            !src.ready(0, true).unwrap().0,
            "source holds a fresh payload"
        );
        drop((src, temp, other, consumer));
        assert_eq!(pool.refs(slot), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create")]
    fn reserved_shared_binary_signal_fixes_its_generation() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let producer = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let shared = producer.share().unwrap();
        let fd = shared.memory.export().unwrap();
        let consumer = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        consumer.import_opaque(fd.as_fd()).unwrap();
        assert!(!consumer.ready(0, true).unwrap().0);
        let temp = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        assert!(producer.reserve_into(&temp).unwrap().0);
        assert!(
            consumer.ready(0, true).unwrap().0,
            "pending across handles at once"
        );
        let reserved = temp.lock().unwrap().snapshot.unwrap();
        // The consumer's capture advances the generation before the commit.
        let (moved, _) = consumer.take_for_move(fresh(&pool)).unwrap();
        assert_eq!(moved.snapshot, Some(reserved));
        let SignalPlan::Shared(_, target) = temp.plan_signal(&temp.lock().unwrap(), 0).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            target, reserved,
            "deferred signal keeps the reserved generation"
        );
        assert!(shared.target() > reserved, "later capture is not undone");
    }

    #[test]
    fn timeline_wait_plan_skips_satisfied_values_and_keeps_full_width() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let sync = SyncObj::new(&pool, true, 0xffff_fffe, fresh(&pool)).unwrap();
        assert!(matches!(
            sync.plan_wait(&sync.lock().unwrap(), 3).unwrap(),
            WaitPlan::Complete
        ));
        let WaitPlan::Timeline(_, v) = sync
            .plan_wait(&sync.lock().unwrap(), 0x1_0000_0001)
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(v, 0x1_0000_0001);
        sync.signal(0x1_0000_0001, None).unwrap();
        assert!(sync.ready(0x1_0000_0001, false).unwrap().0);
        assert_eq!(sync.value().unwrap().0, 0x1_0000_0001);
        sync.signal(5, None).unwrap();
        assert_eq!(sync.value().unwrap().0, 0x1_0000_0001, "monotonic");
        assert!(sync.reset(None).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create")]
    fn foreign_terminal_error_is_never_success() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let fd = crate::platform::create_sealable_memfd().unwrap();
        let foreign = Foreign::new(fd, -110);
        for pending in [false, true] {
            let error = foreign.ready(pending).unwrap_err();
            assert!(matches!(&error, Error::DeviceLost(_)));
            assert!(error.to_string().contains("-110"));
        }
        let ok = Foreign::new(crate::platform::create_sealable_memfd().unwrap(), 1);
        assert!(ok.ready(false).unwrap());
        assert!(ok.duplicate().is_ok());
        // A pending memfd stand-in cannot be queried as a sync_file: the
        // ioctl error must surface, never readiness.
        let pending = Foreign::new(crate::platform::create_sealable_memfd().unwrap(), 0);
        assert!(pending.ready(false).is_err());
        assert_eq!(pending.cached(), 0);
        drop(pool);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create")]
    fn shared_binary_publishes_local_points_and_failure_never_completes() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let producer = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        // Pending local GPU signal before export: it must survive conversion.
        let Payload::Native(local) = producer.lock().unwrap().payload else {
            panic!()
        };
        producer.commit_signal(&mut producer.lock().unwrap(), 1, 9, 0);
        let shared = producer.share().unwrap();
        let fd = shared.memory.export().unwrap();
        let consumer = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        consumer.import_opaque(fd.as_fd()).unwrap();
        assert!(!consumer.ready(0, false).unwrap().0);
        assert!(
            consumer.ready(0, true).unwrap().0,
            "pending visible across handles"
        );
        pool.gpu_write32(local, 1); // GPU release into the transferred slot
        let (ready, changed) = producer.ready(0, false).unwrap();
        assert!(ready && changed, "publication owes one broadcast");
        assert!(consumer.ready(0, false).unwrap().0);
        assert_eq!(pool.refs(local), 0, "published point released");
        // Next generation: GPU signal plan advances the generation.
        let SignalPlan::Shared(sh, target) =
            producer.plan_signal(&producer.lock().unwrap(), 0).unwrap()
        else {
            panic!()
        };
        assert_eq!(target, 2);
        let point = fresh(&pool);
        sh.add_point(target, point).unwrap();
        assert!(!consumer.ready(0, false).unwrap().0);
        assert!(sh.fail());
        assert!(matches!(
            consumer.ready(0, false),
            Err(Error::DeviceLost(_))
        ));
        assert!(!sh.fail(), "repeated failure is quiet");
        drop(sh);
        drop(producer);
        drop(consumer);
        drop(shared);
        assert_eq!(pool.refs(point), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create")]
    fn shared_timeline_conversion_publishes_pending_gpu_values() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let sync = SyncObj::new(&pool, true, 3, fresh(&pool)).unwrap();
        let Payload::Native(local) = sync.lock().unwrap().payload else {
            panic!()
        };
        sync.commit_signal(&mut sync.lock().unwrap(), 1, 2, 7);
        let shared = sync.share().unwrap();
        assert_eq!(sync.value().unwrap().0, 3);
        assert!(sync.ready(7, true).unwrap().0);
        pool.gpu_write64(local, 5);
        assert_eq!(sync.value().unwrap(), (5, true));
        pool.gpu_write64(local, 7);
        assert_eq!(sync.value().unwrap().0, 7);
        assert_eq!(pool.refs(local), 0, "source done at its last pending value");
        assert!(!shared.has_local_work());
        // Invalid import is transactional.
        assert!(
            sync.import_opaque(SyncMemory::new_cpu().unwrap().export().unwrap().as_fd())
                .is_err()
        );
        assert_eq!(sync.value().unwrap().0, 7);
    }
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create")]
    fn import_and_foreign_replacement_release_slots_exactly_once() {
        let (pool, _p) = HostPages::pool(TimelineView::Native64, 1);
        let source = SyncObj::new(&pool, false, 1, fresh(&pool)).unwrap();
        let shared = source.share().unwrap();
        let fd = shared.export().unwrap();
        let sync = SyncObj::new(&pool, false, 0, fresh(&pool)).unwrap();
        let Payload::Native(original) = sync.lock().unwrap().payload else {
            panic!()
        };
        sync.import_opaque(fd.as_fd()).unwrap();
        assert_eq!(pool.refs(original), 0, "import released the replaced slot");
        assert!(sync.ready(0, false).unwrap().0, "imported shared state");
        // A foreign payload is never written: CPU signal/reset/GPU-signal plans
        // rebind to a fresh native slot.
        let foreign = Arc::new(Foreign::new(
            crate::platform::create_sealable_memfd().unwrap(),
            1,
        ));
        sync.rebind_native(&mut sync.lock().unwrap(), fresh(&pool));
        {
            let mut state = sync.lock().unwrap();
            let old = std::mem::replace(&mut state.payload, Payload::Foreign(Arc::clone(&foreign)));
            let Payload::Native(slot) = old else { panic!() };
            pool.release(slot);
        }
        assert!(sync.needs_fresh_slot().unwrap());
        assert!(matches!(
            sync.plan_signal(&sync.lock().unwrap(), 0).unwrap(),
            SignalPlan::ReplaceForeign
        ));
        assert!(matches!(
            sync.plan_wait(&sync.lock().unwrap(), 0).unwrap(),
            WaitPlan::Complete
        ));
        assert!(sync.wait_consumes(&sync.lock().unwrap()));
        let replacement = fresh(&pool);
        assert!(sync.signal(0, Some(replacement)).unwrap().0.is_none());
        assert!(sync.ready(0, false).unwrap().0);
        assert_eq!(
            Arc::strong_count(&foreign),
            1,
            "payload dropped its foreign owner"
        );
        drop(sync);
        assert_eq!(pool.refs(replacement), 0);
        drop((source, shared));
    }
}
