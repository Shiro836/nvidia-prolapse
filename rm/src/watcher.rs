// SPDX-License-Identifier: MIT
//! The one optional driver thread: `nvk-rm-foreign`.
//!
//! Started lazily on the first *pending* foreign dependency (an imported
//! sync_file still active at submission or CPU wait) or shared opaque payload
//! that needs publication. Idle otherwise: blocked in poll with no timeout.
//!
//! Divergence from stock 615 (documented, see state timeout-policy and
//! fence-notification-trace): stock hands an imported sync_file to the kernel
//! with DRM_IOCTL_NVIDIA_SEMSURF_FENCE_WAIT, whose public implementation
//! (nvidia-drm-fence.c:1639-1750) returns success even when callback setup
//! fails, advances the semaphore on a producer *error*, and (774-784) makes
//! context destruction spin on uncancellable foreign callbacks. Here the
//! watcher owns a duplicate of the descriptor, decodes the authoritative
//! sync_file status on readiness, releases the GPU gate slot only on success,
//! publishes device loss on error (dependent work is never released as
//! successful), and is cancelled through its eventfd and joined at teardown.
use crate::diag::diag;
use crate::{
    mesa::Core,
    reactor::{Readiness, WakeFd},
    rm::{Error, Result},
    slots::Slot,
    syncobj::{Foreign, Shared},
};
use std::{
    os::fd::{AsRawFd, RawFd},
    sync::{Arc, Mutex, MutexGuard},
    thread::JoinHandle,
};

#[derive(Default)]
struct WatchState {
    stop: bool,
    /// Pending foreign dependency of accepted GPU work; the slot reference is
    /// the watcher's (the batch holds its own).
    gates: Vec<(Arc<Foreign>, Slot)>,
    /// Foreign payloads a blocked CPU waiter depends on.
    waits: Vec<Arc<Foreign>>,
    /// Shared opaque payloads with local GPU points to publish (needs HOST).
    publish: Vec<Arc<Shared>>,
    /// Shared payload generations GPU work waits for (needs HOST broadcasts).
    shared_gates: Vec<(Arc<Shared>, u64, Slot)>,
    host_fd: Option<RawFd>,
}
impl WatchState {
    fn needs_host(&self) -> bool {
        !self.publish.is_empty() || !self.shared_gates.is_empty()
    }
}

pub(crate) struct Watcher {
    state: Mutex<WatchState>,
    wake: WakeFd,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Watcher {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            state: Mutex::new(WatchState::default()),
            wake: WakeFd::new()?,
            thread: Mutex::new(None),
        })
    }
    fn lock(&self) -> MutexGuard<'_, WatchState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
    pub(crate) fn wake(&self) {
        if let Err(error) = self.wake.notify() {
            diag!("RM foreign watcher wake failed: {error}");
        }
    }
    fn ensure_started(&self, core: &Arc<Core>) -> Result<()> {
        let mut thread = self.thread.lock().unwrap_or_else(|p| p.into_inner());
        if thread.is_none() {
            if self.lock().stop {
                return Err(Error::DeviceLost("RM foreign watcher stopped".into()));
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
    pub(crate) fn has_host(&self) -> bool {
        self.lock().host_fd.is_some()
    }
    pub(crate) fn set_host(&self, fd: RawFd) {
        self.lock().host_fd = Some(fd);
    }
    /// Takes ownership of one reference to `slot`.
    pub(crate) fn gate(&self, core: &Arc<Core>, foreign: Arc<Foreign>, slot: Slot) -> Result<()> {
        if let Err(error) = self.ensure_started(core) {
            core.pool.release(slot);
            return Err(error);
        }
        self.lock().gates.push((foreign, slot));
        self.wake();
        Ok(())
    }
    pub(crate) fn watch_cpu(&self, core: &Arc<Core>, foreign: Arc<Foreign>) -> Result<()> {
        self.ensure_started(core)?;
        let mut state = self.lock();
        if !state.waits.iter().any(|f| Arc::ptr_eq(f, &foreign)) {
            state.waits.push(foreign);
            drop(state);
            self.wake();
        }
        Ok(())
    }
    pub(crate) fn publish(&self, core: &Arc<Core>, shared: Arc<Shared>) -> Result<()> {
        self.ensure_started(core)?;
        let mut state = self.lock();
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
        if let Err(error) = self.ensure_started(core) {
            core.pool.release(slot);
            return Err(error);
        }
        self.lock().shared_gates.push((shared, target, slot));
        self.wake();
        Ok(())
    }
    /// Teardown: stop and join (bounded: the thread blocks only in poll on
    /// descriptors that include our eventfd, or in a single status ioctl).
    /// Remaining gates are dropped without release (never success); shared
    /// payloads with unpublished local points are failed and broadcast.
    pub(crate) fn stop(&self, core: &Core) {
        self.lock().stop = true;
        self.wake();
        if let Some(thread) = self.thread.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = thread.join();
        }
        let mut state = self.lock();
        for (_, slot) in state.gates.drain(..) {
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
            if let Ok(published) = shared.refresh() {
                changed |= published;
            }
            if shared.has_local_work() {
                changed |= shared.fail();
            }
        }
        if changed {
            if let Err(error) = core.trigger.trigger() {
                diag!("RM final shared publication: {error}");
            }
        }
    }
    /// Device loss: fail shared payloads still awaiting local completion and
    /// drop gates (their dependent work must never run as successful).
    pub(crate) fn fail_all(&self, core: &Core) {
        let mut state = self.lock();
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
        for (_, slot) in state.gates.drain(..) {
            core.pool.release(slot);
        }
        for (_, _, slot) in state.shared_gates.drain(..) {
            core.pool.release(slot);
        }
        drop(state);
        if changed {
            if let Err(error) = core.trigger.trigger() {
                diag!("RM publish device loss: {error}");
            }
        }
    }
}

fn run(core: &Arc<Core>) {
    let watcher = &core.watcher;
    let mut readiness = Readiness::default();
    let mut fds: Vec<RawFd> = Vec::new();
    loop {
        let host = {
            let state = watcher.lock();
            if state.stop {
                return;
            }
            fds.clear();
            fds.push(watcher.wake.as_raw_fd());
            fds.extend(state.gates.iter().map(|(f, _)| f.raw_fd()));
            fds.extend(state.waits.iter().map(|f| f.raw_fd()));
            let host = state.host_fd.filter(|_| state.needs_host());
            fds.extend(host);
            host
        };
        fds.sort_unstable();
        fds.dedup();
        // No timer: every source is a real descriptor (sync_files, RM HOST,
        // our eventfd). HUP/ERR on the RM HOST listener is device loss.
        let devices = host.map(|fd| (fd, fd));
        if let Err(error) = readiness.dispatch(&fds, devices, true) {
            diag!("RM foreign watcher: {error}");
            core.mark_lost();
            watcher.lock().stop = true;
            return;
        }
        if readiness.observed(watcher.wake.as_raw_fd()) {
            if let Err(error) = watcher.wake.drain() {
                diag!("RM foreign watcher wake drain: {error}");
            }
        }
        service(core, &readiness, host);
    }
}

fn service(core: &Arc<Core>, readiness: &Readiness, host: Option<RawFd>) {
    let watcher = &core.watcher;
    if core.is_lost() {
        // Loss may be published from a Mesa callback that must not issue RM
        // ioctls; the cross-process failure broadcast happens here instead.
        watcher.fail_all(core);
        return;
    }
    let mut state = watcher.lock();
    let mut changed = false;
    let mut lost = core.is_lost();
    // Foreign gates: decode authoritative status only for readable fds.
    state.gates.retain(|(foreign, slot)| {
        if lost {
            core.pool.release(*slot);
            return false;
        }
        if !readiness.observed(foreign.raw_fd()) && foreign.cached() == 0 {
            return true;
        }
        match foreign.status() {
            Ok(0) => true,
            Ok(status) if status > 0 => {
                // Success only: release the GPU acquire, then drop our ref.
                if let Err(error) = core.pool.store32(*slot, 1) {
                    diag!("RM foreign gate release: {error}");
                }
                core.pool.release(*slot);
                false
            }
            result => {
                diag!(
                    "nvk Rust RM imported dependency failed ({}); dependent work stays blocked, device lost",
                    match result {
                        Ok(status) => format!("producer status {status}"),
                        Err(error) => error.to_string(),
                    }
                );
                lost = true;
                core.pool.release(*slot);
                false
            }
        }
    });
    // CPU-wait interest: cache terminal status, wake waiters.
    state.waits.retain(|foreign| {
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
            if let Ok(published) = shared.refresh() {
                broadcast |= published;
            }
            if shared.completed() >= *target {
                if let Err(error) = core.pool.store32(*slot, 1) {
                    diag!("RM shared gate release: {error}");
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
    if broadcast {
        if let Err(error) = core.trigger.trigger() {
            diag!("RM shared broadcast: {error}");
        }
    }
    if lost && !core.is_lost() {
        core.mark_lost();
    } else if changed {
        core.waiters.notify();
    }
}
