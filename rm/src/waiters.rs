// SPDX-License-Identifier: MIT
//! Caller-side CPU waits: one device HOST RM event descriptor, leader/follower.
//!
//! Public nvidia_poll (OGKM kernel-open/nvidia/nv.c) registers the waitqueue,
//! then reads and CLEARS a dataless per-file flag. Two threads polling the same
//! file therefore steal each other's readiness (stock 615 shows exactly this:
//! its second waiter on a shared fence fd woke up to ~1 ms late). Here exactly
//! one waiting thread at a time (the leader) polls the device HOST and RC
//! descriptors plus a private eventfd. Whenever the leader leaves poll it bumps
//! a generation and wakes every follower, which re-evaluate their predicates
//! (the GPU wrote the mapped word before raising the non-stall interrupt), and
//! leadership passes to whichever waiter claims it next. The kernel flag keeps
//! an interrupt that arrives while no leader is polling, so it cannot be lost.
//!
//! CPU-side state changes (host timeline signal, CPU binary signal, foreign or
//! shared completion observed by the watcher, loss) use notify(): a Dekker pair
//! of SeqCst fences around (value store | waiter registration) guarantees that
//! either the waiter's predicate sees the new value or the signaller sees the
//! registration and wakes it. No RM trigger ioctl is needed for local waiters.
//!
//! The only deadline is the caller's. The leader additionally returns from poll
//! at least once per second so the device-level loss scan runs (stock splits
//! fence waits into <= 1 s chunks with a forced fault check for the same reason).
use crate::diag::diag;
use crate::{
    reactor::{Readiness, WakeFd},
    rm::Result,
};
use std::{
    os::fd::{AsRawFd, RawFd},
    sync::{
        Condvar, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence},
    },
    time::{Duration, Instant},
};

/// Longest uninterrupted leader poll: the forced loss-scan period.
pub(crate) const SCAN_PERIOD: Duration = Duration::from_secs(1);

pub(crate) struct Waiters {
    registered: AtomicU32,
    followers: AtomicU32,
    leader: AtomicBool,
    generation: AtomicU64,
    lock: Mutex<()>,
    changed: Condvar,
    wake: WakeFd,
}

/// Why the leader's poll returned, for the caller's loss handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceEvent {
    /// RC readiness, EINTR, or the periodic scan deadline: read error state.
    Scan,
}

struct Registration<'a>(&'a Waiters);
impl Drop for Registration<'_> {
    fn drop(&mut self) {
        self.0.registered.fetch_sub(1, Ordering::SeqCst);
    }
}
struct Leadership<'a>(&'a Waiters);
impl Drop for Leadership<'_> {
    fn drop(&mut self) {
        // Release first, then wake followers: one of them takes over polling.
        self.0.leader.store(false, Ordering::SeqCst);
        self.0.bump();
    }
}

impl Waiters {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            registered: AtomicU32::new(0),
            followers: AtomicU32::new(0),
            leader: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            lock: Mutex::new(()),
            changed: Condvar::new(),
            wake: WakeFd::new()?,
        })
    }

    #[cfg(test)]
    pub(crate) fn registered(&self) -> u32 {
        self.registered.load(Ordering::SeqCst)
    }

    /// Generation bump + follower wake + leader interrupt.
    fn bump(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        if self.followers.load(Ordering::SeqCst) != 0 {
            // A follower checks the generation while holding this lock and
            // only then sleeps; taking it orders our notify after that check.
            let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
            self.changed.notify_all();
        }
    }

    /// Unconditional wake of every waiter (loss, cancellation, teardown).
    pub(crate) fn broadcast(&self) {
        self.bump();
        if let Err(error) = self.wake.notify() {
            diag!("RM waiter wake failed: {error}");
        }
    }

    /// Signaller side of the Dekker protocol. Call after the predicate-visible
    /// store (slot word, status cache, loss flag). Costs one fence and one load
    /// when nobody waits; never a syscall then.
    pub(crate) fn notify(&self) {
        fence(Ordering::SeqCst);
        if self.registered.load(Ordering::SeqCst) == 0 {
            return;
        }
        self.bump();
        if self.leader.load(Ordering::SeqCst) {
            if let Err(error) = self.wake.notify() {
                diag!("RM waiter wake failed: {error}");
            }
        }
    }

    /// Block until `predicate` returns true (Ok(true)), the deadline passes
    /// (Ok(false)), or an error (including loss) is returned. `devices` are the
    /// (HOST, RC) RM event descriptors; `scan` runs on RC readiness, EINTR, or
    /// at least once per SCAN_PERIOD while leading. No sleep-poll: followers
    /// sleep on a condition variable, the leader in poll on real events.
    pub(crate) fn wait(
        &self,
        deadline: Option<Instant>,
        devices: Option<(RawFd, RawFd)>,
        mut predicate: impl FnMut() -> Result<bool>,
        mut scan: impl FnMut(DeviceEvent) -> Result<()>,
    ) -> Result<bool> {
        self.registered.fetch_add(1, Ordering::SeqCst);
        let _registration = Registration(self);
        let mut readiness = Readiness::default();
        let mut next_scan = Instant::now() + SCAN_PERIOD;
        loop {
            // Dekker: registration (above) is ordered before reading state.
            fence(Ordering::SeqCst);
            let generation = self.generation.load(Ordering::SeqCst);
            if predicate()? {
                return Ok(true);
            }
            let now = Instant::now();
            if deadline.is_some_and(|d| now >= d) {
                return Ok(false);
            }
            if self
                .leader
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                let leadership = Leadership(self);
                // A CPU-side notify between our predicate and the claim saw no
                // leader; it bumped the generation instead. Re-evaluate.
                if self.generation.load(Ordering::SeqCst) != generation {
                    drop(leadership);
                    continue;
                }
                if now >= next_scan {
                    next_scan = now + SCAN_PERIOD;
                    scan(DeviceEvent::Scan)?;
                    drop(leadership);
                    continue;
                }
                let limit = deadline.map_or(next_scan, |d| d.min(next_scan));
                let timeout = limit.saturating_duration_since(now);
                let mut fds = [self.wake.as_raw_fd(), -1, -1];
                let mut count = 1;
                if let Some((host, rc)) = devices {
                    fds[1] = host;
                    fds[2] = rc;
                    count = 3;
                }
                let outcome = readiness.dispatch_timeout(&fds[..count], devices, Some(timeout))?;
                if readiness.observed(self.wake.as_raw_fd()) {
                    self.wake.drain()?;
                }
                if outcome.rc_ready || (outcome.timed_out && Instant::now() >= next_scan) {
                    next_scan = Instant::now() + SCAN_PERIOD;
                    scan(DeviceEvent::Scan)?;
                }
                drop(leadership);
            } else {
                self.followers.fetch_add(1, Ordering::SeqCst);
                let mut guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
                loop {
                    if self.generation.load(Ordering::SeqCst) != generation
                        || !self.leader.load(Ordering::SeqCst)
                    {
                        break;
                    }
                    match deadline {
                        None => {
                            guard = self.changed.wait(guard).unwrap_or_else(|p| p.into_inner());
                        }
                        Some(d) => {
                            let remaining = d.saturating_duration_since(Instant::now());
                            if remaining.is_zero() {
                                break;
                            }
                            guard = self
                                .changed
                                .wait_timeout(guard, remaining)
                                .unwrap_or_else(|p| p.into_inner())
                                .0;
                        }
                    }
                }
                drop(guard);
                self.followers.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rm::Error;
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };

    /// CPU-only "slot" and a host-side signaller using the Dekker notify.
    struct Fixture {
        waiters: Waiters,
        value: AtomicU64,
        lost: AtomicBool,
    }
    impl Fixture {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                waiters: Waiters::new().unwrap(),
                value: AtomicU64::new(0),
                lost: AtomicBool::new(false),
            })
        }
        fn signal(&self, value: u64) {
            self.value.fetch_max(value, Ordering::SeqCst);
            self.waiters.notify();
        }
        fn wait(&self, target: u64, deadline: Option<Instant>) -> Result<bool> {
            self.waiters.wait(
                deadline,
                None,
                || {
                    if self.lost.load(Ordering::SeqCst) {
                        return Err(Error::DeviceLost("test loss".into()));
                    }
                    Ok(self.value.load(Ordering::SeqCst) >= target)
                },
                |_| Ok(()),
            )
        }
    }

    #[test]
    fn dekker_skip_never_loses_a_wakeup_under_stress() {
        // Rounds of: N waiters wait for value r while a signaller races their
        // registration with its store. Every other round the signaller first
        // waits until all are registered (forcing the parked paths: leader in
        // poll, followers on the condvar). A lost wakeup would only be
        // recovered by the 1 s leader scan period or the deadline, so each
        // round's latency is bounded far below that.
        let fixture = Fixture::new();
        let waiters = 3;
        let rounds = 2000u64;
        let barrier = Arc::new(Barrier::new(waiters + 1));
        let threads: Vec<_> = (0..waiters)
            .map(|_| {
                let f = Arc::clone(&fixture);
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut worst = Duration::ZERO;
                    for round in 1..=rounds {
                        b.wait();
                        let start = Instant::now();
                        let deadline = start + Duration::from_secs(10);
                        assert!(f.wait(round, Some(deadline)).unwrap(), "timeout at {round}");
                        worst = worst.max(start.elapsed());
                    }
                    worst
                })
            })
            .collect();
        for round in 1..=rounds {
            barrier.wait();
            if round % 2 == 0 {
                while fixture.waiters.registered() < waiters as u32 {
                    std::thread::yield_now();
                }
            } else if round % 3 == 0 {
                std::thread::yield_now();
            }
            fixture.signal(round);
        }
        for t in threads {
            let worst = t.join().unwrap();
            assert!(worst < Duration::from_millis(300), "lost wakeup: {worst:?}");
        }
        assert_eq!(fixture.waiters.registered(), 0);
    }

    #[test]
    fn device_event_wakes_leader_and_every_follower_without_cpu_notify() {
        // Stand-ins for the RM HOST/RC descriptors (eventfds). The "GPU"
        // writes the value and then raises HOST readiness only — the path a
        // real non-stall interrupt takes. Followers depend on the leader's
        // generation bump; RC readiness must run the scan, whose error ends
        // every waiter.
        let fixture = Fixture::new();
        let host = Arc::new(WakeFd::new().unwrap());
        let rc = Arc::new(WakeFd::new().unwrap());
        let scans = Arc::new(AtomicU64::new(0));
        let fail_scan = Arc::new(AtomicBool::new(false));
        let waiters = 3;
        for round in 1..=300u64 {
            let ready = Arc::new(Barrier::new(waiters + 1));
            let threads: Vec<_> = (0..waiters)
                .map(|_| {
                    let (f, h, r, scans, fail, ready) = (
                        Arc::clone(&fixture),
                        Arc::clone(&host),
                        Arc::clone(&rc),
                        Arc::clone(&scans),
                        Arc::clone(&fail_scan),
                        Arc::clone(&ready),
                    );
                    std::thread::spawn(move || {
                        ready.wait();
                        let start = Instant::now();
                        let result = f.waiters.wait(
                            Some(start + Duration::from_secs(10)),
                            Some((h.as_raw_fd(), r.as_raw_fd())),
                            || {
                                if f.lost.load(Ordering::SeqCst) {
                                    return Err(Error::DeviceLost("published loss".into()));
                                }
                                Ok(f.value.load(Ordering::SeqCst) >= round)
                            },
                            |_| {
                                scans.fetch_add(1, Ordering::SeqCst);
                                r.drain()?;
                                if fail.load(Ordering::SeqCst) {
                                    // As Bridge: publish loss, wake everyone.
                                    f.lost.store(true, Ordering::SeqCst);
                                    f.waiters.broadcast();
                                    Err(Error::DeviceLost("injected channel error".into()))
                                } else {
                                    Ok(())
                                }
                            },
                        );
                        (result, start.elapsed())
                    })
                })
                .collect();
            ready.wait();
            while fixture.waiters.registered() < waiters as u32 {
                std::thread::yield_now();
            }
            if round == 300 {
                // Final round: a channel error (RC) instead of completion.
                fail_scan.store(true, Ordering::SeqCst);
                rc.notify().unwrap();
                for t in threads {
                    let (result, elapsed) = t.join().unwrap();
                    assert!(matches!(result, Err(Error::DeviceLost(_))));
                    assert!(
                        elapsed < Duration::from_millis(300),
                        "loss wake {elapsed:?}"
                    );
                }
                break;
            }
            fixture.value.store(round, Ordering::SeqCst);
            host.notify().unwrap(); // dataless readiness, no notify()
            for t in threads {
                let (result, elapsed) = t.join().unwrap();
                assert!(result.unwrap());
                assert!(
                    elapsed < Duration::from_millis(300),
                    "follower missed bump: {elapsed:?}"
                );
            }
            // The leader consumed the readiness (as nvidia_poll does).
            host.drain().unwrap();
        }
        assert!(scans.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn signal_with_no_waiter_costs_no_wake() {
        let fixture = Fixture::new();
        fixture.signal(1);
        // No eventfd counter was written: a leader poll with zero timeout
        // reports nothing ready.
        let mut readiness = Readiness::default();
        let fd = fixture.waiters.wake.as_raw_fd();
        let out = readiness
            .dispatch_timeout(&[fd], None, Some(Duration::ZERO))
            .unwrap();
        assert!(out.timed_out && !readiness.observed(fd));
        assert!(fixture.wait(1, Some(Instant::now())).unwrap());
    }

    #[test]
    fn deadline_and_loss_end_every_waiter() {
        let fixture = Fixture::new();
        let start = Instant::now();
        assert!(
            !fixture
                .wait(5, Some(start + Duration::from_millis(30)))
                .unwrap()
        );
        assert!(start.elapsed() >= Duration::from_millis(30));
        let threads: Vec<_> = (0..3)
            .map(|_| {
                let f = Arc::clone(&fixture);
                std::thread::spawn(move || f.wait(9, None))
            })
            .collect();
        while fixture.waiters.registered() < 3 {
            std::thread::yield_now();
        }
        fixture.lost.store(true, Ordering::SeqCst);
        fixture.waiters.broadcast();
        for t in threads {
            assert!(matches!(t.join().unwrap(), Err(Error::DeviceLost(_))));
        }
    }

    #[test]
    fn two_waiters_on_one_value_both_wake_promptly() {
        // CPU path of the two-waiter latency gate: both waiters (one leader,
        // one follower) must observe the signal well under 1 ms (p99).
        let fixture = Fixture::new();
        let mut lags = Vec::new();
        for round in 1..=200u64 {
            let ready = Arc::new(Barrier::new(3));
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    let f = Arc::clone(&fixture);
                    let r = Arc::clone(&ready);
                    std::thread::spawn(move || {
                        r.wait();
                        assert!(f.wait(round, None).unwrap());
                        Instant::now()
                    })
                })
                .collect();
            ready.wait();
            while fixture.waiters.registered() < 2 {
                std::thread::yield_now();
            }
            // Let both reach their blocking point (leader poll / follower cv).
            std::thread::sleep(Duration::from_micros(200));
            let start = Instant::now();
            fixture.signal(round);
            for t in threads {
                lags.push(t.join().unwrap().saturating_duration_since(start));
            }
        }
        lags.sort();
        let p99 = lags[lags.len() * 99 / 100];
        assert!(p99 < Duration::from_millis(1), "two-waiter p99 {p99:?}");
    }
}
