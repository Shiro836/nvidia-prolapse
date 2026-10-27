// SPDX-License-Identifier: MIT
//! The one optional driver thread: `nvk-rm-foreign`.
//!
//! Started lazily on the first *pending* foreign dependency (an imported
//! sync_file still active at submission or CPU wait) or shared opaque payload
//! that needs publication. Idle otherwise: blocked in poll with no timeout.
//!
//! Divergence from NVIDIA's driver 615: it hands an imported sync_file to the kernel
//! with DRM_IOCTL_NVIDIA_SEMSURF_FENCE_WAIT, whose public implementation
//! (nvidia-drm-fence.c:1639-1750) returns success even when callback setup
//! fails, advances the semaphore on a producer *error*, and (774-784) makes
//! context destruction spin on uncancellable foreign callbacks. Here the
//! watcher owns a duplicate of the descriptor, decodes the authoritative
//! sync_file status on readiness, releases the GPU gate slot only on success,
//! publishes device loss on error (dependent work is never released as
//! successful), and is cancelled through its eventfd and joined at teardown.
//!
//! Descriptor lifetime: the watcher's references to imported `sync_file`s end
//! before the caller can observe the operation they serve. A gate's reference
//! is dropped before the gate slot is released to the GPU, so the dependent
//! work (and every wait on it) completes only after it. A blocked CPU waiter
//! holds a [`CpuWatch`]; dropping it ends the watcher's reference before the
//! wait returns. The watcher polls raw descriptor numbers outside the state
//! lock, so a CPU waiter that removes its entry while the watcher polls the
//! set containing it waits until the watcher has left that poll: closing a
//! polled descriptor would turn it into POLLNVAL (device loss) or let a new
//! descriptor reuse its number.
//!
//! Locking: `state` is the only lock; it is never held across poll, and the
//! `left_poll` condition variable waits on it. A poisoned `state` or `thread`
//! lock (a thread panicked while holding it, and the FFI guard marked the
//! device lost) is fatal for the watcher (RUST.md §8): entry points return
//! `Error::DeviceLost`, the thread marks the device lost and exits, and
//! teardown skips the state it can no longer trust.
use crate::diag::diag;
use crate::{
    mesa::Core,
    reactor::{Readiness, WakeFd},
    rm::{Error, Result, SyncError},
    slots::Slot,
    syncobj::{Foreign, Shared},
};
use std::{
    os::fd::{AsRawFd, RawFd},
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError},
    thread::JoinHandle,
};

/// The error a poisoned watcher lock turns into (module doc: locking).
fn poisoned<T>(object: &'static str, _poisoned: &PoisonError<T>) -> Error {
    SyncError::Poisoned { object }.into()
}

#[derive(Default)]
struct WatchState {
    stop: bool,
    /// Pending foreign dependency of accepted GPU work; the slot reference is
    /// the watcher's (the batch holds its own).
    gates: Vec<(Arc<Foreign>, Slot)>,
    /// Foreign payloads blocked CPU waiters depend on, one entry per payload.
    waits: Vec<CpuWait>,
    /// Shared opaque payloads with local GPU points to publish (needs HOST).
    publish: Vec<Arc<Shared>>,
    /// Shared payload generations GPU work waits for (needs HOST broadcasts).
    shared_gates: Vec<(Arc<Shared>, u64, Slot)>,
    host_fd: Option<RawFd>,
    /// Number of the descriptor set the watcher built last, bumped under this
    /// lock each time it builds one.
    snapshot: u64,
    /// The watcher is in poll on the descriptors of `snapshot`.
    polling: bool,
}

/// One foreign payload watched for CPU waiters.
struct CpuWait {
    foreign: Arc<Foreign>,
    /// Live [`CpuWatch`] guards for it.
    waiters: usize,
}

impl WatchState {
    fn needs_host(&self) -> bool {
        !self.publish.is_empty() || !self.shared_gates.is_empty()
    }
}

pub(crate) struct Watcher {
    state: Mutex<WatchState>,
    /// Signalled when the watcher leaves a poll (`polling` turns false).
    left_poll: Condvar,
    wake: WakeFd,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Watcher {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            state: Mutex::new(WatchState::default()),
            left_poll: Condvar::new(),
            wake: WakeFd::new()?,
            thread: Mutex::new(None),
        })
    }
    fn lock(&self) -> Result<MutexGuard<'_, WatchState>> {
        self.state
            .lock()
            .map_err(|error| poisoned("RM foreign watcher state", &error))
    }
    pub(crate) fn wake(&self) {
        if let Err(error) = self.wake.notify() {
            diag!("RM foreign watcher wake failed: {error}");
        }
    }
    fn ensure_started(&self, core: &Arc<Core>) -> Result<()> {
        let mut thread = self
            .thread
            .lock()
            .map_err(|error| poisoned("RM foreign watcher thread handle", &error))?;
        if thread.is_none() {
            if self.lock()?.stop {
                return Err(SyncError::Closed("RM foreign watcher").into());
            }
            let owner = Arc::clone(core);
            *thread = Some(
                std::thread::Builder::new()
                    .name("nvk-rm-foreign".into())
                    .spawn(move || {
                        // A panic here must still publish loss: gated GPU work
                        // would otherwise never be released or failed.
                        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&owner)))
                            .is_err()
                        {
                            diag!("nvk Rust RM foreign watcher panicked; device lost");
                            owner.mark_lost();
                        }
                    })
                    .map_err(|e| Error::Io("start RM foreign watcher", e))?,
            );
        }
        Ok(())
    }
    /// The watcher's own HOST listener descriptor (owned by the Bridge).
    pub(crate) fn has_host(&self) -> Result<bool> {
        Ok(self.lock()?.host_fd.is_some())
    }
    pub(crate) fn set_host(&self, fd: RawFd) -> Result<()> {
        self.lock()?.host_fd = Some(fd);
        Ok(())
    }
    /// Takes ownership of one reference to `slot`.
    pub(crate) fn gate(&self, core: &Arc<Core>, foreign: Arc<Foreign>, slot: Slot) -> Result<()> {
        let mut state = match self.ensure_started(core).and_then(|()| self.lock()) {
            Ok(state) => state,
            Err(error) => {
                core.pool.release(slot);
                return Err(error);
            }
        };
        state.gates.push((foreign, slot));
        drop(state);
        self.wake();
        Ok(())
    }
    /// Wake CPU waiters when `foreign` completes, until the returned guard
    /// is dropped.
    pub(crate) fn watch_cpu(
        &self,
        core: &Arc<Core>,
        foreign: Arc<Foreign>,
    ) -> Result<CpuWatch<'_>> {
        self.ensure_started(core)?;
        let mut state = self.lock()?;
        if let Some(wait) = state
            .waits
            .iter_mut()
            .find(|wait| Arc::ptr_eq(&wait.foreign, &foreign))
        {
            wait.waiters += 1;
        } else {
            state.waits.push(CpuWait {
                foreign: Arc::clone(&foreign),
                waiters: 1,
            });
            drop(state);
            self.wake();
        }
        Ok(CpuWatch {
            watcher: self,
            foreign,
        })
    }
    /// The last guard of a payload removes its entry; the watcher's reference
    /// ends before this returns.
    fn unwatch_cpu(&self, foreign: &Arc<Foreign>) {
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => {
                diag!("RM CPU watch removal: {error}");
                return;
            }
        };
        // Absent: the watcher already retired it (terminal status or loss).
        let Some(index) = state
            .waits
            .iter()
            .position(|wait| Arc::ptr_eq(&wait.foreign, foreign))
        else {
            return;
        };
        let wait = &mut state.waits[index];
        wait.waiters -= 1;
        if wait.waiters != 0 {
            return;
        }
        let retired = state.waits.remove(index);
        if state.polling {
            // The poll in progress may hold this descriptor number. Any set
            // the watcher builds from now on excludes it, so waiting for the
            // current one to end is enough. Not bounded by a timeout: the
            // watcher leaves poll on our eventfd write, and on every exit
            // path, unwinding included (`Polling`).
            let snapshot = state.snapshot;
            self.wake();
            match self
                .left_poll
                .wait_while(state, |s| s.polling && s.snapshot == snapshot)
            {
                Ok(left) => state = left,
                Err(error) => {
                    // Closing a descriptor the watcher may still poll is safe
                    // here: poll never consumes readiness, and the watcher
                    // ends at its next lock of the poisoned state.
                    diag!(
                        "RM CPU watch removal: {}",
                        poisoned("RM foreign watcher state", &error)
                    );
                    drop(retired);
                    return;
                }
            }
        }
        drop(state);
        drop(retired);
    }
    pub(crate) fn publish(&self, core: &Arc<Core>, shared: Arc<Shared>) -> Result<()> {
        self.ensure_started(core)?;
        let mut state = self.lock()?;
        if !state.publish.iter().any(|s| Arc::ptr_eq(s, &shared)) {
            state.publish.push(shared);
            drop(state);
            self.wake();
        }
        Ok(())
    }
    /// Takes ownership of one reference to `slot`.
    pub(crate) fn shared_gate(
        &self,
        core: &Arc<Core>,
        shared: Arc<Shared>,
        target: u64,
        slot: Slot,
    ) -> Result<()> {
        let mut state = match self.ensure_started(core).and_then(|()| self.lock()) {
            Ok(state) => state,
            Err(error) => {
                core.pool.release(slot);
                return Err(error);
            }
        };
        state.shared_gates.push((shared, target, slot));
        drop(state);
        self.wake();
        Ok(())
    }
    /// Teardown: stop and join (bounded: the thread blocks only in poll on
    /// descriptors that include our eventfd, or in a single status ioctl).
    /// Remaining gates are dropped without release (never success); shared
    /// payloads with unpublished local points are failed and broadcast.
    pub(crate) fn stop(&self, core: &Core) {
        match self.lock() {
            Ok(mut state) => state.stop = true,
            Err(error) => diag!("RM foreign watcher stop: {error}"),
        }
        self.wake();
        match self.thread.lock() {
            Ok(mut thread) => {
                if let Some(thread) = thread.take()
                    && thread.join().is_err()
                {
                    diag!("RM foreign watcher thread panicked");
                }
            }
            Err(error) => diag!(
                "RM foreign watcher stop: {}",
                poisoned("RM foreign watcher thread handle", &error)
            ),
        }
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => {
                diag!("RM foreign watcher teardown skipped: {error}");
                return;
            }
        };
        for (foreign, slot) in state.gates.drain(..) {
            drop(foreign);
            core.pool.release(slot);
        }
        for (_, _, slot) in state.shared_gates.drain(..) {
            core.pool.release(slot);
        }
        state.waits.clear();
        let mut changed = false;
        for shared in state.publish.drain(..) {
            // Successful drain published everything already; unfinished
            // points (lost/undrained device) become errors, never success.
            match shared.refresh() {
                Ok(published) => changed |= published,
                Err(error) => diag!("RM shared payload refresh at loss: {error}"),
            }
            if shared.has_local_work() {
                changed |= shared.fail();
            }
        }
        if changed && let Err(error) = core.trigger.trigger() {
            diag!("RM final shared publication: {error}");
        }
    }
    /// Device loss: fail shared payloads still awaiting local completion and
    /// drop gates (their dependent work must never run as successful).
    pub(crate) fn fail_all(&self, core: &Core) {
        let mut state = match self.lock() {
            Ok(state) => state,
            Err(error) => {
                diag!("RM foreign watcher loss publication: {error}");
                return;
            }
        };
        let mut changed = false;
        for shared in &state.publish {
            if shared.has_local_work() {
                changed |= shared.fail();
            }
        }
        state
            .publish
            .retain(|shared| shared.has_local_work() && !shared.failed());
        // Waiters observe the loss flag; keeping readable descriptors watched
        // after loss would only spin this thread.
        state.waits.clear();
        for (foreign, slot) in state.gates.drain(..) {
            drop(foreign);
            core.pool.release(slot);
        }
        for (_, _, slot) in state.shared_gates.drain(..) {
            core.pool.release(slot);
        }
        drop(state);
        if changed && let Err(error) = core.trigger.trigger() {
            diag!("RM publish device loss: {error}");
        }
    }
}

/// Marks the watcher as polling the current descriptor set; dropping it
/// (after poll, or while unwinding) clears the mark and wakes CPU waiters that
/// wait for the poll to end.
struct Polling<'a>(&'a Watcher);
impl Drop for Polling<'_> {
    fn drop(&mut self) {
        match self.0.lock() {
            Ok(mut state) => state.polling = false,
            // Waiters woken below re-lock and fail with the same error.
            Err(error) => diag!("RM foreign watcher poll end: {error}"),
        }
        self.0.left_poll.notify_all();
    }
}

/// A CPU waiter's interest in one foreign payload (`Watcher::watch_cpu`).
pub(crate) struct CpuWatch<'a> {
    watcher: &'a Watcher,
    foreign: Arc<Foreign>,
}
impl Drop for CpuWatch<'_> {
    fn drop(&mut self) {
        self.watcher.unwatch_cpu(&self.foreign);
    }
}

fn run(core: &Arc<Core>) {
    let watcher = &core.watcher;
    let mut readiness = Readiness::default();
    let mut fds: Vec<RawFd> = Vec::new();
    loop {
        let host = {
            let mut state = match watcher.lock() {
                Ok(state) => state,
                Err(error) => {
                    diag!("RM foreign watcher: {error}");
                    core.mark_lost();
                    return;
                }
            };
            if state.stop {
                return;
            }
            fds.clear();
            fds.push(watcher.wake.as_raw_fd());
            fds.extend(state.gates.iter().map(|(f, _)| f.raw_fd()));
            fds.extend(state.waits.iter().map(|wait| wait.foreign.raw_fd()));
            let host = state.host_fd.filter(|_| state.needs_host());
            fds.extend(host);
            state.snapshot = state.snapshot.wrapping_add(1);
            state.polling = true;
            host
        };
        let polling = Polling(watcher);
        fds.sort_unstable();
        fds.dedup();
        // No timer: every source is a real descriptor (sync_files, RM HOST,
        // our eventfd). HUP/ERR on the RM HOST listener is device loss.
        // Unbounded on purpose (RUST.md §7): this is the watcher thread's idle
        // wait, not a caller's. Close ends it through `wake`; every caller
        // waiting on a watched sync_file carries its own deadline (waiters.rs).
        let devices = host.map(|fd| (fd, fd));
        let polled = readiness.dispatch(&fds, devices, true);
        drop(polling);
        if let Err(error) = polled {
            diag!("RM foreign watcher: {error}");
            core.mark_lost();
            match watcher.lock() {
                Ok(mut state) => state.stop = true,
                Err(error) => diag!("RM foreign watcher: {error}"),
            }
            return;
        }
        if readiness.observed(watcher.wake.as_raw_fd())
            && let Err(error) = watcher.wake.drain()
        {
            diag!("RM foreign watcher wake drain: {error}");
        }
        service(core, &readiness, host);
    }
}

/// What a foreign gate's status means for its slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GateOutcome {
    /// Still active: keep watching.
    Pending,
    /// Signalled successfully: release the GPU acquire.
    Released,
    /// Failed, or the device is lost: never released as success.
    Dropped,
}

fn service(core: &Arc<Core>, readiness: &Readiness, host: Option<RawFd>) {
    let watcher = &core.watcher;
    if core.is_lost() {
        // Loss may be published from a Mesa callback that must not issue RM
        // ioctls; the cross-process failure broadcast happens here instead.
        watcher.fail_all(core);
        return;
    }
    let mut state = match watcher.lock() {
        Ok(state) => state,
        Err(error) => {
            diag!("RM foreign watcher: {error}");
            core.mark_lost();
            return;
        }
    };
    let mut changed = false;
    let mut lost = core.is_lost();
    // Foreign gates: decode authoritative status only for readable fds. A
    // settled gate drops our sync_file reference before its slot is released
    // to the GPU (module doc: descriptor lifetime).
    let mut index = 0;
    while let Some((foreign, _)) = state.gates.get(index) {
        let outcome = if lost {
            GateOutcome::Dropped
        } else if !readiness.observed(foreign.raw_fd()) && foreign.cached() == 0 {
            GateOutcome::Pending
        } else {
            match foreign.status() {
                Ok(0) => GateOutcome::Pending,
                Ok(status) if status > 0 => GateOutcome::Released,
                result => {
                    diag!(
                        "nvk Rust RM imported dependency failed ({}); dependent work stays blocked, device lost",
                        match result {
                            Ok(status) => format!("producer status {status}"),
                            Err(error) => error.to_string(),
                        }
                    );
                    lost = true;
                    GateOutcome::Dropped
                }
            }
        };
        if outcome == GateOutcome::Pending {
            index += 1;
            continue;
        }
        let (foreign, slot) = state.gates.remove(index);
        drop(foreign);
        if outcome == GateOutcome::Released {
            // Success only: release the GPU acquire, then drop our slot ref.
            // A release that does not reach the slot leaves the dependent
            // work blocked: device lost, never a logged hang.
            if let Err(error) = core.pool.store32(slot, 1) {
                diag!("RM foreign gate release failed; device lost: {error}");
                lost = true;
            }
        }
        core.pool.release(slot);
    }
    // CPU-wait interest: cache terminal status, wake waiters.
    state.waits.retain(|CpuWait { foreign, .. }| {
        if foreign.cached() != 0 {
            changed = true;
            return false;
        }
        if !readiness.observed(foreign.raw_fd()) {
            return true;
        }
        match foreign.status() {
            Ok(0) => true,
            _ => {
                changed = true;
                false
            }
        }
    });
    let host_event = host.is_some_and(|fd| readiness.observed(fd));
    let mut broadcast = false;
    if host_event || readiness.observed(watcher.wake.as_raw_fd()) {
        state.publish.retain(|shared| {
            match shared.refresh() {
                Ok(published) => {
                    broadcast |= published;
                    changed |= published;
                }
                Err(error) => {
                    diag!("RM shared publication: {error}");
                    lost = true;
                }
            }
            shared.has_local_work()
        });
        state.shared_gates.retain(|(shared, target, slot)| {
            match shared.refresh() {
                Ok(published) => broadcast |= published,
                Err(error) => {
                    // The gate could only complete through a later refresh.
                    diag!("RM shared gate refresh failed; device lost: {error}");
                    lost = true;
                }
            }
            if shared.completed() >= *target {
                if let Err(error) = core.pool.store32(*slot, 1) {
                    diag!("RM shared gate release failed; device lost: {error}");
                    lost = true;
                }
                core.pool.release(*slot);
                false
            } else if shared.failed() || lost {
                diag!(
                    "nvk Rust RM shared producer failed; dependent work stays blocked, device lost"
                );
                lost = true;
                core.pool.release(*slot);
                false
            } else {
                true
            }
        });
    }
    drop(state);
    if broadcast && let Err(error) = core.trigger.trigger() {
        diag!("RM shared broadcast: {error}");
    }
    if lost && !core.is_lost() {
        core.mark_lost();
    } else if changed {
        core.waiters.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slots::{TimelineView, tests::HostPages};
    use std::{
        io::Write,
        os::{fd::OwnedFd, unix::net::UnixStream},
        time::{Duration, Instant},
    };

    /// Pollable stand-in for an imported `sync_file`: readable once the peer
    /// writes; `Foreign::settle` supplies the status the kernel would report.
    fn stand_in() -> (Arc<Foreign>, UnixStream) {
        let (ours, peer) = UnixStream::pair().unwrap();
        (Arc::new(Foreign::new(OwnedFd::from(ours), 0)), peer)
    }

    /// Spin until the watcher polls a descriptor set built after `after`.
    fn polled_after(watcher: &Watcher, after: u64) -> u64 {
        let start = Instant::now();
        loop {
            {
                let state = watcher.lock().unwrap();
                if state.polling && state.snapshot != after {
                    return state.snapshot;
                }
            }
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "watcher never polled"
            );
            std::thread::yield_now();
        }
    }

    /// Started watcher idle in poll; returns its descriptor-set number.
    fn idle(core: &Arc<Core>) -> u64 {
        core.watcher.ensure_started(core).unwrap();
        polled_after(&core.watcher, 0)
    }

    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn cpu_watch_ends_the_watchers_reference_before_the_wait_returns() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let core = Core::detached(pool).unwrap();
        let before = idle(&core);
        // Never readable: on its own the watcher would keep the entry forever.
        let (foreign, _peer) = stand_in();
        let first = core.watcher.watch_cpu(&core, Arc::clone(&foreign)).unwrap();
        let second = core.watcher.watch_cpu(&core, Arc::clone(&foreign)).unwrap();
        // Test, two guards, one watcher entry.
        assert_eq!(Arc::strong_count(&foreign), 4);
        // The watcher now polls a set holding the descriptor: removal has to
        // wait for that poll to end.
        polled_after(&core.watcher, before);
        drop(first);
        assert_eq!(
            Arc::strong_count(&foreign),
            3,
            "entry kept for the other waiter"
        );
        drop(second);
        assert_eq!(
            Arc::strong_count(&foreign),
            1,
            "watcher still references the sync_file after the last waiter left"
        );
        drop(foreign);
        // The descriptor it polled is gone; it must keep serving, not see
        // POLLNVAL and publish loss.
        let slot = core.pool.try_allocate().unwrap().unwrap();
        core.pool.retain(slot);
        let (gated, mut peer) = stand_in();
        core.watcher.gate(&core, Arc::clone(&gated), slot).unwrap();
        gated.settle(1);
        peer.write_all(&[1]).unwrap();
        let start = Instant::now();
        while core.pool.read32(slot).unwrap() != 1 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "gate never released"
            );
            std::thread::yield_now();
        }
        assert!(!core.is_lost());
        core.pool.release(slot);
        core.watcher.stop(&core);
    }

    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn gate_drops_its_sync_file_before_releasing_the_slot() {
        // The release is what lets dependent work, and every wait on it,
        // complete; the watcher's reference must already be gone then.
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let core = Core::detached(pool).unwrap();
        idle(&core);
        for round in 0..200 {
            let slot = core.pool.try_allocate().unwrap().unwrap();
            core.pool.retain(slot);
            let (foreign, mut peer) = stand_in();
            core.watcher
                .gate(&core, Arc::clone(&foreign), slot)
                .unwrap();
            foreign.settle(1);
            peer.write_all(&[1]).unwrap();
            let start = Instant::now();
            while core.pool.read32(slot).unwrap() != 1 {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "gate never released"
                );
                std::hint::spin_loop();
            }
            assert_eq!(
                Arc::strong_count(&foreign),
                1,
                "round {round}: slot released while the watcher held the sync_file"
            );
            core.pool.release(slot);
        }
        assert!(!core.is_lost());
        core.watcher.stop(&core);
    }

    /// RUST.md §8: a poisoned watcher state fails new dependencies as device
    /// lost, and teardown does not panic on it.
    #[test]
    #[cfg_attr(miri, ignore = "eventfd on a real descriptor")]
    fn poisoned_state_fails_new_watches_as_device_lost() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let core = Core::detached(pool).unwrap();
        std::thread::scope(|scope| {
            let holder = scope.spawn(|| {
                let _state = core.watcher.state.lock().unwrap();
                panic!("poison the watcher state");
            });
            assert!(holder.join().is_err());
        });
        let (foreign, _peer) = stand_in();
        let result = core.watcher.watch_cpu(&core, foreign).map(|_| ());
        assert!(
            result.as_ref().is_err_and(Error::is_device_lost),
            "{result:?}"
        );
        assert!(
            core.watcher
                .has_host()
                .is_err_and(|error| error.is_device_lost())
        );
        core.watcher.stop(&core);
    }
}
