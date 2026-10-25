//! Local-GPU fixtures for the caller-side bridge (ignored by default).
//! Run alone: cargo test --offline --no-default-features --lib
//!   mesa::hardware_tests -- --ignored --test-threads=1 --nocapture
use super::*;
use std::{sync::Barrier, thread, time::Duration};

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
        card.pci.bus as u32,
        card.pci.slot as u32,
        card.pci.function as u32,
    ]
}
fn open() -> Box<Bridge> {
    Box::new(Bridge::new(selected_pci()).unwrap().0)
}
fn context(b: &mut Bridge) -> u64 {
    let mut id = 0;
    assert_eq!(unsafe { nv_rm_context_create(b, 1, &mut id) }, 0);
    id
}
fn create(b: &mut Bridge, timeline: bool, initial: u64) -> u64 {
    let mut id = 0;
    assert_eq!(
        unsafe { nv_rm_sync_create(b, u32::from(timeline), initial, &mut id) },
        0
    );
    id
}
fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}
struct SendPtr(*mut Bridge);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[test]
#[ignore = "local NVIDIA GPU: ordinary signal then exporter close; independent imported counter must complete"]
fn clean_exporter_close_publishes_gpu_completion() {
    let mut producer = open();
    let mut consumer = open();
    let ctx = context(&mut producer);
    let signal = create(&mut producer, true, 0);
    let mut exported = -1;
    assert_eq!(
        unsafe { nv_rm_sync_export_fd(&mut *producer, signal, 1, -1, &mut exported) },
        0
    );
    let imported = create(&mut consumer, true, 0);
    assert_eq!(
        unsafe { nv_rm_sync_import_fd(&mut *consumer, imported, exported, 1) },
        0
    );
    let mut token = 0;
    let point = SyncPoint {
        sync: signal,
        value: 1,
    };
    assert_eq!(
        unsafe { nv_rm_context_signal(&mut *producer, ctx, &point, 1, &mut token) },
        0
    );
    // No CPU status query, fence wait, or grace period before close.
    assert_eq!(unsafe { nv_rm_sync_destroy(&mut *producer, signal) }, 0);
    drop(producer);
    unsafe { libc_close(exported) };
    let mut value = 0;
    assert_eq!(
        unsafe { nv_rm_sync_get_value(&mut *consumer, imported, &mut value) },
        0
    );
    assert_eq!(value, 1);
    assert_eq!(unsafe { nv_rm_sync_destroy(&mut *consumer, imported) }, 0);
}
unsafe extern "C" {
    #[link_name = "close"]
    fn libc_close(fd: i32) -> i32;
}

#[test]
#[ignore = "local NVIDIA GPU: bounded teardown of owned unsignaled timeline gate, no foreign kernel callbacks"]
fn unsignaled_owned_gate_has_bounded_context_teardown() {
    let mut bridge = open();
    let ctx = context(&mut bridge);
    let wait = create(&mut bridge, true, 0);
    let mut token = 0;
    let point = SyncPoint {
        sync: wait,
        value: 1,
    };
    assert_eq!(
        unsafe {
            nv_rm_submit_sync(
                &mut *bridge,
                ctx,
                std::ptr::null(),
                0,
                &point,
                1,
                &mut token,
            )
        },
        0
    );
    assert!(!bridge.wait_token(ctx, token, 0).unwrap());
    let start = Instant::now();
    assert_eq!(unsafe { nv_rm_context_destroy(&mut *bridge, ctx) }, 0);
    assert!(start.elapsed() >= Duration::from_secs(5));
    assert!(start.elapsed() < Duration::from_secs(10));
    assert!(bridge.core.is_lost());
    assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, wait) }, 0);
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
    let mut bridge = open();
    let ctx = context(&mut bridge);
    let gate = create(&mut bridge, true, 0);
    let fence = create(&mut bridge, false, 0);
    let rounds = 300u64;
    let mut first = Vec::new();
    let mut second = Vec::new();
    for round in 1..=rounds {
        let target = if gpu {
            let mut token = 0;
            let w = SyncPoint {
                sync: gate,
                value: round,
            };
            let s = SyncPoint {
                sync: fence,
                value: 0,
            };
            assert_eq!(
                unsafe {
                    nv_rm_submit_batch(
                        &mut *bridge,
                        ctx,
                        std::ptr::null(),
                        0,
                        &w,
                        1,
                        &s,
                        1,
                        &mut token,
                    )
                },
                0
            );
            SyncPoint {
                sync: fence,
                value: 0,
            }
        } else {
            SyncPoint {
                sync: gate,
                value: round,
            }
        };
        let ready = Arc::new(Barrier::new(3));
        let pointer = Arc::new(SendPtr(&mut *bridge));
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let ready = Arc::clone(&ready);
                let pointer = Arc::clone(&pointer);
                thread::spawn(move || {
                    ready.wait();
                    let p = pointer.0;
                    let r = unsafe { nv_rm_sync_wait_many(p, &target, 1, 0, u64::MAX) };
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
        assert_eq!(unsafe { nv_rm_sync_signal(&mut *bridge, gate, round) }, 0);
        let mut times: Vec<_> = waiters
            .into_iter()
            .map(|t| t.join().unwrap().saturating_duration_since(start))
            .collect();
        times.sort();
        first.push(times[0]);
        second.push(times[1]);
        if gpu {
            assert_eq!(unsafe { nv_rm_sync_reset(&mut *bridge, fence) }, 0);
        }
    }
    for id in [gate, fence] {
        assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, id) }, 0);
    }
    assert_eq!(unsafe { nv_rm_context_destroy(&mut *bridge, ctx) }, 0);
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
    let mut bridge = open();
    let ctx = context(&mut bridge);
    let semaphore = create(&mut bridge, false, 0);
    let fence = create(&mut bridge, false, 0);
    let timeline = create(&mut bridge, true, 0);
    for frame in 1..=3000u64 {
        // Frame: wait previous semaphore signal (detach), signal semaphore +
        // fence + timeline in one batch.
        let waits = if frame > 1 {
            vec![SyncPoint {
                sync: semaphore,
                value: 0,
            }]
        } else {
            vec![]
        };
        let signals = [
            SyncPoint {
                sync: semaphore,
                value: 0,
            },
            SyncPoint {
                sync: fence,
                value: 0,
            },
            SyncPoint {
                sync: timeline,
                value: frame,
            },
        ];
        let mut token = 0;
        loop {
            let r = unsafe {
                nv_rm_submit_batch(
                    &mut *bridge,
                    ctx,
                    std::ptr::null(),
                    0,
                    waits.as_ptr(),
                    waits.len() as u32,
                    signals.as_ptr(),
                    3,
                    &mut token,
                )
            };
            if r == 1 {
                assert_eq!(unsafe { nv_rm_wait(&mut *bridge, ctx, token, u64::MAX) }, 0);
                continue;
            }
            assert_eq!(r, 0);
            break;
        }
        // Copy-export semantics: reset while the GPU release may be pending.
        if frame % 2 == 0 {
            let p = SyncPoint {
                sync: fence,
                value: 0,
            };
            assert_eq!(
                unsafe { nv_rm_sync_wait_many(&mut *bridge, &p, 1, 0, u64::MAX) },
                0
            );
        }
        assert_eq!(unsafe { nv_rm_sync_reset(&mut *bridge, fence) }, 0);
        if frame % 100 == 0 {
            // Move: dst takes the pending payload, src gets a fresh slot.
            let dst = create(&mut bridge, false, 0);
            assert_eq!(
                unsafe { nv_rm_sync_move(&mut *bridge, dst, semaphore, 0) },
                0
            );
            let p = SyncPoint {
                sync: dst,
                value: 0,
            };
            assert_eq!(
                unsafe { nv_rm_sync_wait_many(&mut *bridge, &p, 1, 0, u64::MAX) },
                0
            );
            assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, dst) }, 0);
            // The semaphore lost its signal to dst: re-signal on the CPU so
            // the next frame's wait has a signal operation.
            assert_eq!(unsafe { nv_rm_sync_signal(&mut *bridge, semaphore, 0) }, 0);
        }
    }
    let p = SyncPoint {
        sync: timeline,
        value: 3000,
    };
    assert_eq!(
        unsafe { nv_rm_sync_wait_many(&mut *bridge, &p, 1, 0, u64::MAX) },
        0
    );
    let mut value = 0;
    assert_eq!(
        unsafe { nv_rm_sync_get_value(&mut *bridge, timeline, &mut value) },
        0
    );
    assert_eq!(value, 3000);
    // Retire everything, then confirm a single pool page sufficed.
    let token = queue_ref(&bridge.lock().0, ctx).unwrap().last_token();
    assert_eq!(unsafe { nv_rm_wait(&mut *bridge, ctx, token, u64::MAX) }, 0);
    let mut t = 0;
    assert_eq!(
        unsafe { nv_rm_submit(&mut *bridge, ctx, std::ptr::null(), 0, &mut t) },
        0
    );
    assert_eq!(
        bridge.core.pool.next_page(),
        Some(1),
        "slots were not recycled"
    );
    for id in [semaphore, fence, timeline] {
        assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, id) }, 0);
    }
    assert_eq!(unsafe { nv_rm_context_destroy(&mut *bridge, ctx) }, 0);
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
    let mut bridge = open();
    let ctx = context(&mut bridge);
    let gate = create(&mut bridge, true, 0);
    let consumed = create(&mut bridge, false, 1);
    let mut accepted = Vec::new();
    let full_token = loop {
        let target = create(&mut bridge, false, 0);
        let waits = [
            SyncPoint {
                sync: gate,
                value: 1,
            },
            SyncPoint {
                sync: consumed,
                value: 0,
            },
        ];
        let signal = SyncPoint {
            sync: target,
            value: 0,
        };
        // Only the retry after the ring is full includes the consuming wait.
        let wait_count = 1u32;
        let mut token = 0;
        let r = unsafe {
            nv_rm_submit_batch(
                &mut *bridge,
                ctx,
                std::ptr::null(),
                0,
                waits.as_ptr(),
                wait_count,
                &signal,
                1,
                &mut token,
            )
        };
        if r == 1 {
            // Retry the same batch with the consuming wait: still full.
            let slot_of = |id: u64| match sync_ref(id).unwrap().lock().payload {
                crate::syncobj::Payload::Native(slot) => slot,
                _ => panic!("native"),
            };
            let (c_slot, t_slot) = (slot_of(consumed), slot_of(target));
            let r2 = unsafe {
                nv_rm_submit_batch(
                    &mut *bridge,
                    ctx,
                    std::ptr::null(),
                    0,
                    waits.as_ptr(),
                    2,
                    &signal,
                    1,
                    &mut token,
                )
            };
            assert_eq!(r2, 1, "still full");
            assert_eq!(slot_of(consumed), c_slot, "consuming wait not committed");
            assert_eq!(slot_of(target), t_slot);
            assert_eq!(bridge.core.pool.refs(c_slot), 1);
            assert_eq!(bridge.core.pool.refs(t_slot), 1);
            assert!(sync_ref(consumed).unwrap().ready(0, false).unwrap().0);
            assert!(sync_ref(target).unwrap().lock().pending.is_none());
            assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, target) }, 0);
            break token;
        }
        assert_eq!(r, 0);
        accepted.push(target);
    };
    assert!(accepted.len() > 300, "ring filled after {}", accepted.len());
    assert_eq!(unsafe { nv_rm_sync_signal(&mut *bridge, gate, 1) }, 0);
    assert_eq!(
        unsafe { nv_rm_wait(&mut *bridge, ctx, full_token, u64::MAX) },
        0
    );
    let last = queue_ref(&bridge.lock().0, ctx).unwrap().last_token();
    assert_eq!(unsafe { nv_rm_wait(&mut *bridge, ctx, last, u64::MAX) }, 0);
    for id in &accepted {
        let p = SyncPoint {
            sync: *id,
            value: 0,
        };
        assert_eq!(
            unsafe { nv_rm_sync_wait_many(&mut *bridge, &p, 1, 0, 0) },
            0
        );
        assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, *id) }, 0);
    }
    for id in [gate, consumed] {
        assert_eq!(unsafe { nv_rm_sync_destroy(&mut *bridge, id) }, 0);
    }
    assert_eq!(unsafe { nv_rm_context_destroy(&mut *bridge, ctx) }, 0);
    assert!(!bridge.core.is_lost());
    drop(bridge);
    assert_eq!(fd_count(), before);
    println!(
        "PASS: ring full after {} gated batches, NOT_READY transactional, all signals completed",
        accepted.len()
    );
}
