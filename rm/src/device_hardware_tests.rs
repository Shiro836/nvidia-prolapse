// SPDX-License-Identifier: MIT
//! Local-GPU fixtures for the caller-side bridge (ignored by default).
//! Run alone: cargo test --offline --no-default-features --lib
//!   device::hardware_tests -- --ignored --test-threads=1 --nocapture
use super::*;
use std::{
    os::fd::{AsFd, FromRawFd, OwnedFd},
    sync::Barrier,
    thread,
    time::Duration,
};
use test_ffi::{self as c, Point, SyncHandle};

/// Safe calls into the C ABI, shared by the sibling fixture modules.
#[path = "device_test_ffi.rs"]
pub(super) mod test_ffi;

fn selected_pci() -> [u32; 4] {
    let client = Client::open().unwrap();
    let card = client
        .cards()
        .unwrap()
        .into_iter()
        .next()
        .expect("NVIDIA RM GPU");
    [
        card.pci.domain,
        u32::from(card.pci.bus),
        u32::from(card.pci.slot),
        u32::from(card.pci.function),
    ]
}
fn open() -> Arc<Bridge> {
    Arc::new(Bridge::new(selected_pci()).unwrap().0)
}
fn context(b: &Bridge) -> u64 {
    let mut id = 0;
    assert_eq!(c::context_create(b, 1, &mut id), 0);
    id
}
fn create(b: &Bridge, timeline: bool, initial: u64) -> SyncHandle {
    c::sync_create(b, u32::from(timeline), initial).expect("nv_rm_sync_create status 0")
}
fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

#[test]
#[ignore = "local NVIDIA GPU: ordinary signal then exporter close; independent imported counter must complete"]
fn clean_exporter_close_publishes_gpu_completion() {
    let producer = open();
    let consumer = open();
    let ctx = context(&producer);
    let signal = create(&producer, true, 0);
    let mut exported = -1;
    assert_eq!(
        c::sync_export_fd(&producer, &signal, 1, None, &mut exported),
        0
    );
    assert!(exported >= 0, "opaque export returns a descriptor");
    // SAFETY: a successful opaque export returns a new open descriptor that
    // only this test owns (nv_rm_sync_export_fd hands out `into_raw_fd`), and
    // it is not -1 (asserted above).
    let exported = unsafe { OwnedFd::from_raw_fd(exported) };
    let imported = create(&consumer, true, 0);
    assert_eq!(
        c::sync_import_fd(&consumer, &imported, exported.as_fd(), 1),
        0
    );
    let mut token = 0;
    let point = Point {
        sync: &signal,
        value: 1,
    };
    assert_eq!(c::context_signal(&producer, ctx, &[point], &mut token), 0);
    // No CPU status query, fence wait, or grace period before close.
    assert_eq!(c::sync_destroy(&producer, signal), 0);
    drop(producer);
    drop(exported);
    let mut value = 0;
    assert_eq!(c::sync_get_value(&consumer, &imported, &mut value), 0);
    assert_eq!(value, 1);
    assert_eq!(c::sync_destroy(&consumer, imported), 0);
}

#[test]
#[ignore = "local NVIDIA GPU: bounded teardown of owned unsignaled timeline gate, no foreign kernel callbacks"]
fn unsignaled_owned_gate_has_bounded_context_teardown() {
    let bridge = open();
    let ctx = context(&bridge);
    let wait = create(&bridge, true, 0);
    let mut token = 0;
    let point = Point {
        sync: &wait,
        value: 1,
    };
    assert_eq!(c::submit_sync(&bridge, ctx, &[], &[point], &mut token), 0);
    assert!(!bridge.wait_token(ctx, token, 0).unwrap());
    let start = Instant::now();
    assert_eq!(c::context_destroy(&bridge, ctx), 0);
    assert!(start.elapsed() >= Duration::from_secs(5));
    assert!(start.elapsed() < Duration::from_secs(10));
    assert!(bridge.core.is_lost());
    assert_eq!(c::sync_destroy(&bridge, wait), 0);
    drop(bridge);
    println!(
        "owned unsignaled GPU gate context+device cleanup {:?}",
        start.elapsed()
    );
}

/// Two-waiter wake latency: two threads wait the same object; both must wake
/// < 1 ms p99 after the signal. `gpu`: a fence signalled by GPU work gated on
/// a host timeline value (interrupt -> leader -> follower). Otherwise the two
/// threads wait the timeline itself (host signal -> Dekker notify).
fn two_waiters(gpu: bool) -> (Duration, Duration, Duration) {
    let before = fd_count();
    let bridge = open();
    let ctx = context(&bridge);
    let gate = Arc::new(create(&bridge, true, 0));
    let fence = Arc::new(create(&bridge, false, 0));
    let rounds = 300u64;
    let mut first = Vec::new();
    let mut second = Vec::new();
    for round in 1..=rounds {
        let target = if gpu {
            let mut token = 0;
            let w = Point {
                sync: &gate,
                value: round,
            };
            let s = Point {
                sync: &fence,
                value: 0,
            };
            assert_eq!(
                c::submit_batch(&bridge, ctx, &[], &[w], &[s], &mut token),
                0
            );
            (Arc::clone(&fence), 0)
        } else {
            (Arc::clone(&gate), round)
        };
        let ready = Arc::new(Barrier::new(3));
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let ready = Arc::clone(&ready);
                let bridge = Arc::clone(&bridge);
                let (sync, value) = (Arc::clone(&target.0), target.1);
                thread::spawn(move || {
                    ready.wait();
                    let target = Point { sync: &sync, value };
                    let r = c::sync_wait_many(&bridge, &[target], 0, u64::MAX);
                    assert_eq!(r, 0);
                    Instant::now()
                })
            })
            .collect();
        ready.wait();
        // Let both reach their blocking point before signalling.
        while bridge.core.waiters.registered() < 2 {
            thread::yield_now();
        }
        thread::sleep(Duration::from_micros(300));
        let start = Instant::now();
        assert_eq!(c::sync_signal(&bridge, &gate, round), 0);
        let mut times: Vec<_> = waiters
            .into_iter()
            .map(|t| t.join().unwrap().saturating_duration_since(start))
            .collect();
        drop(target);
        times.sort();
        first.push(times[0]);
        second.push(times[1]);
        if gpu {
            assert_eq!(c::sync_reset(&bridge, &fence), 0);
        }
    }
    for id in [gate, fence] {
        let id = Arc::into_inner(id).expect("every waiter thread was joined");
        assert_eq!(c::sync_destroy(&bridge, id), 0);
    }
    assert_eq!(c::context_destroy(&bridge, ctx), 0);
    drop(bridge);
    assert_eq!(fd_count(), before, "fd leak");
    // Follower lag: how long after the first waiter the second one woke.
    let mut lag: Vec<_> = first
        .iter()
        .zip(&second)
        .map(|(a, b)| b.saturating_sub(*a))
        .collect();
    first.sort();
    second.sort();
    lag.sort();
    let p = |v: &Vec<Duration>, q: usize| v[(v.len() * q / 100).min(v.len() - 1)];
    println!(
        "{} two-waiter wake after signal: first p50 {:?} p99 {:?}; second p50 {:?} p99 {:?} max {:?}; second-after-first lag p50 {:?} p99 {:?}",
        if gpu { "GPU fence" } else { "host timeline" },
        p(&first, 50),
        p(&first, 99),
        p(&second, 50),
        p(&second, 99),
        second.last().unwrap(),
        p(&lag, 50),
        p(&lag, 99)
    );
    assert!(p(&lag, 99) < Duration::from_millis(1), "follower lag");
    (p(&first, 99), p(&second, 99), p(&second, 50))
}
#[test]
#[ignore = "local NVIDIA GPU: two threads wait one GPU-signalled fence; both wake < 1 ms p99"]
fn two_waiters_gpu_fence_wake_latency() {
    let (first, second, _) = two_waiters(true);
    assert!(first < Duration::from_millis(1), "first p99 {first:?}");
    assert!(second < Duration::from_millis(1), "second p99 {second:?}");
}
#[test]
#[ignore = "local NVIDIA GPU: two threads wait one host-signalled timeline; both wake < 1 ms p99"]
fn two_waiters_host_timeline_wake_latency() {
    let (first, second, _) = two_waiters(false);
    assert!(first < Duration::from_millis(1), "first p99 {first:?}");
    assert!(second < Duration::from_millis(1), "second p99 {second:?}");
}

/// Slot lifetime under retire / detach-on-wait / reset-while-captured / move /
/// import: thousands of frames must recycle slots (no pool growth beyond the
/// first page) and every GPU-observed predicate must hold.
#[test]
#[ignore = "local NVIDIA GPU: slot recycling across retire, binary wait detach, reset, move, import"]
fn slot_lifetime_recycles_under_gpu_retirement() {
    let before = fd_count();
    let bridge = open();
    let ctx = context(&bridge);
    let semaphore = create(&bridge, false, 0);
    let fence = create(&bridge, false, 0);
    let timeline = create(&bridge, true, 0);
    for frame in 1..=3000u64 {
        // Frame: wait previous semaphore signal (detach), signal semaphore +
        // fence + timeline in one batch.
        let waits = if frame > 1 {
            vec![Point {
                sync: &semaphore,
                value: 0,
            }]
        } else {
            vec![]
        };
        let signals = [
            Point {
                sync: &semaphore,
                value: 0,
            },
            Point {
                sync: &fence,
                value: 0,
            },
            Point {
                sync: &timeline,
                value: frame,
            },
        ];
        let mut token = 0;
        loop {
            let r = c::submit_batch(&bridge, ctx, &[], &waits, &signals, &mut token);
            if r == 1 {
                assert_eq!(c::wait(&bridge, ctx, token, u64::MAX), 0);
                continue;
            }
            assert_eq!(r, 0);
            break;
        }
        // Copy-export semantics: reset while the GPU release may be pending.
        if frame % 2 == 0 {
            let p = Point {
                sync: &fence,
                value: 0,
            };
            assert_eq!(c::sync_wait_many(&bridge, &[p], 0, u64::MAX), 0);
        }
        assert_eq!(c::sync_reset(&bridge, &fence), 0);
        if frame % 100 == 0 {
            // Move: dst takes the pending payload, src gets a fresh slot.
            let dst = create(&bridge, false, 0);
            assert_eq!(c::sync_move(&bridge, &dst, &semaphore, 0), 0);
            let p = Point {
                sync: &dst,
                value: 0,
            };
            assert_eq!(c::sync_wait_many(&bridge, &[p], 0, u64::MAX), 0);
            assert_eq!(c::sync_destroy(&bridge, dst), 0);
            // The semaphore lost its signal to dst: re-signal on the CPU so
            // the next frame's wait has a signal operation.
            assert_eq!(c::sync_signal(&bridge, &semaphore, 0), 0);
        }
    }
    let p = Point {
        sync: &timeline,
        value: 3000,
    };
    assert_eq!(c::sync_wait_many(&bridge, &[p], 0, u64::MAX), 0);
    let mut value = 0;
    assert_eq!(c::sync_get_value(&bridge, &timeline, &mut value), 0);
    assert_eq!(value, 3000);
    // Retire everything, then confirm a single pool page sufficed.
    let token = queue_ref(&bridge.lock().unwrap().0, ctx)
        .unwrap()
        .last_token();
    assert_eq!(c::wait(&bridge, ctx, token.get(), u64::MAX), 0);
    let mut t = 0;
    assert_eq!(c::submit(&bridge, ctx, &[], &mut t), 0);
    assert_eq!(
        bridge.core.pool.next_page().unwrap(),
        Some(1),
        "slots were not recycled"
    );
    for id in [semaphore, fence, timeline] {
        assert_eq!(c::sync_destroy(&bridge, id), 0);
    }
    assert_eq!(c::context_destroy(&bridge, ctx), 0);
    assert!(!bridge.core.is_lost());
    drop(bridge);
    assert_eq!(fd_count(), before, "fd leak");
    println!("PASS: 3000 frames, one 4096-slot page, fd {before}->{before}");
}

/// Ring pressure must be transactional: NOT_READY accepts no wait, push or
/// signal and leaves every payload (including a consuming binary wait)
/// untouched; after the gate opens, every accepted signal completes.
#[test]
#[ignore = "local NVIDIA GPU: GPFIFO ring full behind a host gate, payloads untouched on NOT_READY"]
fn ring_full_leaves_payloads_untouched() {
    let before = fd_count();
    let bridge = open();
    let ctx = context(&bridge);
    let gate = create(&bridge, true, 0);
    let consumed = create(&bridge, false, 1);
    let mut accepted = Vec::new();
    let full_token = loop {
        let target = create(&bridge, false, 0);
        let waits = [
            Point {
                sync: &gate,
                value: 1,
            },
            Point {
                sync: &consumed,
                value: 0,
            },
        ];
        let signal = Point {
            sync: &target,
            value: 0,
        };
        // Only the retry after the ring is full includes the consuming wait.
        let wait_count = 1;
        let mut token = 0;
        let r = c::submit_batch(
            &bridge,
            ctx,
            &[],
            &waits[..wait_count],
            &[signal],
            &mut token,
        );
        if r == 1 {
            // Retry the same batch with the consuming wait: still full.
            let slot_of = |h: &SyncHandle| match h.obj().unwrap().lock().unwrap().payload {
                crate::syncobj::Payload::Native(slot) => slot,
                _ => panic!("native"),
            };
            let (c_slot, t_slot) = (slot_of(&consumed), slot_of(&target));
            let r2 = c::submit_batch(&bridge, ctx, &[], &waits[..2], &[signal], &mut token);
            assert_eq!(r2, 1, "still full");
            assert_eq!(slot_of(&consumed), c_slot, "consuming wait not committed");
            assert_eq!(slot_of(&target), t_slot);
            assert_eq!(bridge.core.pool.refs(c_slot), 1);
            assert_eq!(bridge.core.pool.refs(t_slot), 1);
            assert!(consumed.obj().unwrap().ready(0, false).unwrap().0);
            assert!(target.obj().unwrap().lock().unwrap().pending.is_none());
            assert_eq!(c::sync_destroy(&bridge, target), 0);
            break token;
        }
        assert_eq!(r, 0);
        accepted.push(target);
    };
    assert!(accepted.len() > 300, "ring filled after {}", accepted.len());
    assert_eq!(c::sync_signal(&bridge, &gate, 1), 0);
    assert_eq!(c::wait(&bridge, ctx, full_token, u64::MAX), 0);
    let last = queue_ref(&bridge.lock().unwrap().0, ctx)
        .unwrap()
        .last_token();
    assert_eq!(c::wait(&bridge, ctx, last.get(), u64::MAX), 0);
    let accepted_count = accepted.len();
    for id in accepted {
        let p = Point {
            sync: &id,
            value: 0,
        };
        assert_eq!(c::sync_wait_many(&bridge, &[p], 0, 0), 0);
        assert_eq!(c::sync_destroy(&bridge, id), 0);
    }
    for id in [gate, consumed] {
        assert_eq!(c::sync_destroy(&bridge, id), 0);
    }
    assert_eq!(c::context_destroy(&bridge, ctx), 0);
    assert!(!bridge.core.is_lost());
    drop(bridge);
    assert_eq!(fd_count(), before);
    println!(
        "PASS: ring full after {accepted_count} gated batches, NOT_READY transactional, all signals completed"
    );
}

unsafe extern "C" {
    fn fork() -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn kill(pid: i32, signal: i32) -> i32;
    fn _exit(status: i32) -> !;
}
/// Runs `child` in a forked child, which exits with its return value, and
/// returns that exit code; `None` if the child did not exit normally within
/// `timeout` (it is then killed).
fn in_forked_child(child: impl FnOnce() -> i32, timeout: Duration) -> Option<i32> {
    // SAFETY: fork has no memory-safety precondition. The child runs only
    // `child` and then _exit: it never unwinds or returns into the test
    // harness and runs no destructors of the parent's state.
    let pid = unsafe { fork() };
    assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
    if pid == 0 {
        let code = child();
        // SAFETY: async-signal-safe process exit; nothing runs after it.
        unsafe { _exit(code) };
    }
    let start = Instant::now();
    let mut status = 0;
    loop {
        // SAFETY: waits for our own child; `status` is a live local.
        // WNOHANG (1): the wait is bounded by `timeout`.
        let got = unsafe { waitpid(pid, &raw mut status, 1) };
        if got == pid {
            break;
        }
        assert!(got == 0, "waitpid: {}", std::io::Error::last_os_error());
        if start.elapsed() > timeout {
            // SAFETY: signals our own child (SIGKILL), then reaps it.
            unsafe { kill(pid, 9) };
            // SAFETY: as above; blocking reap of the killed child.
            unsafe { waitpid(pid, &raw mut status, 0) };
            return None;
        }
        thread::sleep(Duration::from_millis(10));
    }
    // WIFEXITED: the low 7 bits (terminating signal) are zero; WEXITSTATUS:
    // bits 8..15 (sys/wait.h).
    (status.trailing_zeros() >= 7).then_some((status >> 8) & 0xff)
}

/// A device inherited across fork fails in the child instead of acting on
/// the parent's RM objects (rm/src/process.rs): a submit returns
/// VK_ERROR_DEVICE_LOST (-4), nv_rm_is_lost reads 1, and close keeps every
/// object and descriptor. The parent's device is untouched.
#[test]
#[ignore = "local NVIDIA GPU: device inherited across fork fails in the child, parent keeps working"]
fn forked_child_cannot_use_or_tear_down_the_parents_device() {
    let before = fd_count();
    let (status, opened) = c::OpenedBridge::open(selected_pci());
    assert_eq!(status, 0);
    let opened = opened.expect("nv_rm_open returned a device");
    let ctx = context(opened.bridge());
    let mut token = 0;
    assert_eq!(c::submit(opened.bridge(), ctx, &[], &mut token), 0);
    assert_eq!(c::wait(opened.bridge(), ctx, token, 10_000_000_000), 0);
    let code = in_forked_child(
        || {
            // SAFETY: this is the forked child, which ends in _exit and never
            // returns into the code holding `opened`.
            let child = unsafe { opened.alias_in_forked_child() };
            let fds = fd_count();
            let mut t = 0;
            let mut failed = 0;
            if c::submit(child.bridge(), ctx, &[], &mut t) != -4 {
                failed |= 1;
            }
            if c::is_lost(child.bridge()) != 1 {
                failed |= 2;
            }
            child.close();
            if fd_count() != fds {
                failed |= 4;
            }
            failed
        },
        Duration::from_secs(10),
    );
    assert_eq!(
        code,
        Some(0),
        "child: 1 = submit not -4, 2 = is_lost not 1, 4 = close released descriptors"
    );
    assert_eq!(c::is_lost(opened.bridge()), 0);
    let mut token = 0;
    assert_eq!(c::submit(opened.bridge(), ctx, &[], &mut token), 0);
    assert_eq!(c::wait(opened.bridge(), ctx, token, 10_000_000_000), 0);
    assert_eq!(c::context_destroy(opened.bridge(), ctx), 0);
    opened.close();
    assert_eq!(fd_count(), before, "fd leak");
    println!(
        "PASS: forked child got -4, read lost, close kept its descriptors; parent submits after the child exited"
    );
}
