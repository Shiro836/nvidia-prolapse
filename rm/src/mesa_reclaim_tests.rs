// SPDX-License-Identifier: MIT
//! Liveness fixtures for deferred reclamation and foreign-gate wakeups
//! (local NVIDIA GPU, ignored by default). Run alone:
//!   cargo test --offline --no-default-features --lib mesa::reclaim_tests \
//!     -- --ignored --test-threads=1 --nocapture
//! NV_RM_STRESS_SECONDS sets the concurrency stress duration (default 20),
//! NV_RM_WAKE_ROUNDS the lost-wakeup rounds (default 300).
//!
//! The GPU is selected by RM card identity (vendor 0x10de, an openable
//! /dev/nvidiaN); its render node is found through sysfs by PCI address.
use super::hardware_tests::test_ffi::{self as c, CpuMap, Point, ReleaseSink, SyncHandle};
use super::*;
use nv_rm_core::conv::{high_u32, low_u32, u64_from_usize};
use std::{
    fs::{File, OpenOptions},
    os::fd::{AsFd, AsRawFd, FromRawFd},
    sync::{Barrier, atomic::AtomicU32, mpsc},
    thread,
    time::Duration,
};

/// Mesa's VA heap starts at 16 GiB; fixtures use fixed, disjoint addresses.
const VA_BASE: u64 = 1 << 36;

fn selected_pci() -> [u32; 4] {
    let client = Client::open().unwrap();
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
        .expect("openable NVIDIA RM GPU");
    [
        card.pci.domain,
        u32::from(card.pci.bus),
        u32::from(card.pci.slot),
        u32::from(card.pci.function),
    ]
}
fn render_node(pci: [u32; 4]) -> File {
    let name = format!("{:04x}:{:02x}:{:02x}.{:x}", pci[0], pci[1], pci[2], pci[3]);
    for e in std::fs::read_dir("/sys/class/drm").unwrap() {
        let e = e.unwrap();
        if !e.file_name().to_string_lossy().starts_with("renderD") {
            continue;
        }
        let device = std::fs::canonicalize(e.path().join("device")).unwrap();
        if device.file_name().unwrap().to_string_lossy() == name {
            return OpenOptions::new()
                .read(true)
                .write(true)
                .open(std::path::Path::new("/dev/dri").join(e.file_name()))
                .unwrap();
        }
    }
    panic!("no NVIDIA DRM render node for {name}");
}
fn open(pci: [u32; 4]) -> Arc<Bridge> {
    Arc::new(Bridge::new(pci).unwrap().0)
}
fn context(b: &Bridge) -> u64 {
    let mut id = 0;
    assert_eq!(c::context_create(b, 1, &mut id), 0);
    id
}
fn create(b: &Bridge, timeline: bool, initial: u64) -> SyncHandle {
    c::sync_create(b, u32::from(timeline), initial).expect("nv_rm_sync_create status 0")
}
fn alloc(b: &Bridge, size: u64, gart: bool) -> u64 {
    let mut a = Allocation {
        id: 0,
        size: 0,
        cpu_address: 0,
    };
    assert_eq!(c::alloc(b, size, 4096, u32::from(gart), 0, 0, &mut a), 0);
    a.id
}
fn bind(b: &Bridge, memory: u64, size: u64, address: u64) -> u64 {
    let mut id = 0;
    assert_eq!(c::bind(b, memory, 0, size, address, 0, &mut id), 0);
    id
}
fn map(b: &Bridge, memory: u64) -> CpuMap<'_> {
    CpuMap::new(b, memory)
}
fn submit(
    b: &Bridge,
    ctx: u64,
    pushes: &[Push],
    waits: &[Point<'_>],
    signals: &[Point<'_>],
) -> u64 {
    let mut token = 0;
    let r = c::submit_batch(b, ctx, pushes, waits, signals, &mut token);
    assert_eq!(r, 0, "submit");
    token
}
fn point(sync: &SyncHandle, value: u64) -> Point<'_> {
    Point { sync, value }
}
fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}
fn groups(b: &Bridge) -> usize {
    b.lock().unwrap().0.reclaim.groups()
}
/// Status calls until every retired deferred group is reclaimed (each call
/// reclaims at most RECLAIM_BUDGET groups).
fn drain_backlog(b: &Bridge) {
    for _ in 0..1_000_000 {
        assert_eq!(c::check_status(b), 0);
        if groups(b) == 0 {
            return;
        }
    }
    panic!("deferred backlog did not drain");
}

/// Records deferred VA releases (stands in for Mesa's util_vma_heap_free).
#[derive(Default)]
struct Released {
    calls: AtomicU32,
    ranges: Mutex<Vec<(u64, u64)>>,
}
impl ReleaseSink for Released {
    fn release(&self, address: u64, size: u64) {
        self.ranges.lock().unwrap().push((address, size));
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}
/// The bridge may run a deferred release at any later call or at close, so
/// the sink lives for the rest of the process (a few bytes per fixture).
fn released() -> &'static Released {
    Box::leak(Box::default())
}
fn va_free(
    b: &Bridge,
    mappings: &[u64],
    address: u64,
    size: u64,
    released: &'static Released,
) -> i32 {
    c::va_free(b, mappings, address, size, released)
}

/// Producer side of a pending foreign sync_file: a batch on `producer` gated
/// on host timeline `gate` >= `value` signals binary `sem`; the exported
/// SYNC_FD is the producer queue's kernel fence (pending until the host
/// signal). Returns the owned sync_file descriptor.
fn pending_sync_file(
    producer: &Bridge,
    ctx: u64,
    gate: &SyncHandle,
    value: u64,
    sem: &SyncHandle,
    drm: &File,
) -> OwnedFd {
    // The previous round's signal completed and was never waited: unsignal
    // it first (a binary may not be signalled twice without a wait).
    assert_eq!(c::sync_reset(producer, sem), 0);
    submit(producer, ctx, &[], &[point(gate, value)], &[point(sem, 0)]);
    let mut fd = -1;
    assert_eq!(
        c::sync_export_fd(producer, sem, 0, Some(drm.as_fd()), &mut fd),
        0
    );
    assert!(fd >= 0);
    // SAFETY: a successful SYNC_FD export returns a new open descriptor that
    // only this caller owns (nv_rm_sync_export_fd hands out `into_raw_fd`),
    // and it is not -1 (asserted above).
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    assert_eq!(
        crate::fences::sync_status(fd.as_raw_fd()).unwrap(),
        0,
        "sync_file pending"
    );
    fd
}

/// 5a. The Android cycle: GPU work gated on a foreign sync_file whose producer
/// signal comes from the thread that frees unrelated memory afterwards. The
/// former synchronous all-queue drain blocked that thread forever (until a
/// kernel timeout); now every free returns promptly and the real RM teardown
/// and VA release run once the gate opened and the batch retired.
#[test]
#[ignore = "local NVIDIA GPU: free/unbind while a batch waits on a pending foreign sync_file"]
fn free_while_foreign_gated_returns_promptly_and_reclaims_after_retirement() {
    let before = fd_count();
    let pci = selected_pci();
    let drm = render_node(pci);
    let producer = open(pci);
    let consumer = open(pci);
    let (pp, cp): (&Bridge, &Bridge) = (&producer, &consumer);
    let pctx = context(&producer);
    let cctx = context(&consumer);
    let gate = Arc::new(create(&producer, true, 0));
    let sem = create(&producer, false, 0);
    let sync_file = pending_sync_file(pp, pctx, &gate, 1, &sem, &drm);
    // Consumer batch waits on the foreign fence through a watcher-owned gate.
    let mut ctoken = 0;
    assert_eq!(
        c::submit_waits(
            cp,
            cctx,
            &[],
            drm.as_fd(),
            &[sync_file.as_fd()],
            &mut ctoken
        ),
        0
    );
    assert!(!consumer.wait_token(cctx, ctoken, 0).unwrap());
    // Unrelated memory with GPU mappings on the consumer.
    let size = 1 << 20;
    let (va0, va1) = (VA_BASE, VA_BASE + (4 << 20));
    let unrelated = alloc(cp, size, false);
    let mapping = bind(cp, unrelated, size, va0);
    let other = alloc(cp, 65536, true);
    let other_map = bind(cp, other, 65536, va1);
    let released = released();
    // If any free blocked on the gated work, this watchdog provides the
    // producer signal so the test fails instead of hanging.
    let (tx, rx) = mpsc::channel::<()>();
    let watchdog = {
        let (pp, gate) = (Arc::clone(&producer), Arc::clone(&gate));
        thread::spawn(move || {
            if rx.recv_timeout(Duration::from_secs(5)).is_err() {
                c::sync_signal(&pp, &gate, 1);
                return true;
            }
            false
        })
    };
    let start = Instant::now();
    assert_eq!(c::unbind(cp, other_map), 0);
    let unbind = start.elapsed();
    assert_eq!(va_free(cp, &[mapping], va0, size, released), 0);
    let va = start.elapsed();
    assert_eq!(c::free(cp, unrelated), 0);
    assert_eq!(c::free(cp, other), 0);
    let all = start.elapsed();
    tx.send(()).unwrap();
    assert!(
        !watchdog.join().unwrap(),
        "a free blocked on foreign-gated work"
    );
    assert!(all < Duration::from_millis(50), "frees took {all:?}");
    // Deferred, not performed: the batch is still gated.
    assert!(!consumer.wait_token(cctx, ctoken, 0).unwrap());
    assert_eq!(released.calls.load(Ordering::SeqCst), 0);
    assert_eq!(groups(&consumer), 3, "VA group + two memory groups");
    // The producer signal (the app's next frame), then the gate opens.
    assert_eq!(c::sync_signal(pp, &gate, 1), 0);
    assert_eq!(c::wait(cp, cctx, ctoken, 10_000_000_000), 0);
    let mut done = 0;
    assert_eq!(c::poll(cp, cctx, ctoken, &mut done), 0);
    assert_eq!(done, 1);
    assert_eq!(released.calls.load(Ordering::SeqCst), 1);
    assert_eq!(*released.ranges.lock().unwrap(), [(va0, size)]);
    assert_eq!(groups(&consumer), 0);
    {
        let inner = &consumer.lock().unwrap().0;
        assert!(!inner.maps.contains_key(&mapping) && !inner.maps.contains_key(&other_map));
        assert!(!inner.buffers.contains_key(&unrelated) && !inner.buffers.contains_key(&other));
        assert!(inner.unmap_failed.is_empty());
    }
    assert!(!consumer.core.is_lost() && !producer.core.is_lost());
    drop(sync_file);
    let gate = Arc::into_inner(gate).expect("the watchdog was joined");
    for id in [gate, sem] {
        assert_eq!(c::sync_destroy(pp, id), 0);
    }
    assert_eq!(c::context_destroy(pp, pctx), 0);
    assert_eq!(c::context_destroy(cp, cctx), 0);
    drop(producer);
    drop(consumer);
    drop(drm);
    assert_eq!(fd_count(), before, "fd leak");
    println!(
        "PASS: foreign-gated batch pending; unbind {unbind:?}, VA free {va:?}, all frees {all:?}; \
         1 deferred VA release after retirement; fd {before}->{before}"
    );
}

/// Public host SEM_* methods (clc96f.h: SEM_ADDR_LO 0x5c .. SEM_EXECUTE 0x6c;
/// EXECUTE OPERATION 2:0 ACQUIRE=0/RELEASE=1, ACQUIRE_SWITCH_TSG 12,
/// RELEASE_WFI 20, PAYLOAD_SIZE 24 = 64-bit).
fn semaphore(words: &mut Vec<u32>, address: u64, value: u64, release: bool) {
    let execute = if release { 1 | (1 << 20) } else { 1 << 12 } | (1 << 24);
    crate::gpu::method(
        words,
        0,
        0x5c,
        &[
            low_u32(address),
            high_u32(address),
            low_u32(value),
            high_u32(value),
            execute,
        ],
    );
}

/// 5b. Driver-internal free of memory that a gated pending batch still uses
/// (nvk_queue.c:65 drops the old SLM area while earlier work may run): after
/// the free calls returned, the GPU writes the freed memory, reads it back
/// with an exact-equality acquire and only then writes a witness. An early
/// unmap would MMU-fault the channel (Xid, device loss) or hang the acquire.
#[test]
#[ignore = "local NVIDIA GPU: GPU writes/reads freed memory after the free call; unmapped only after retirement"]
fn freed_memory_stays_mapped_until_the_gated_batch_retires() {
    let before = fd_count();
    let pci = selected_pci();
    let bridge = open(pci);
    let b: &Bridge = &bridge;
    let ctx = context(&bridge);
    let channel = queue_ref(&bridge.lock().unwrap().0, ctx)
        .unwrap()
        .channel_class();
    assert!(
        crate::gpu::modern_host(channel).unwrap(),
        "fixture encodes Volta+ host methods"
    );
    let gate = create(&bridge, true, 0);
    let size = 65536;
    let (va_m, va_p, va_r) = (
        VA_BASE + (16 << 20),
        VA_BASE + (20 << 20),
        VA_BASE + (24 << 20),
    );
    let freed = alloc(b, size, false);
    let freed_map = bind(b, freed, size, va_m);
    let push = alloc(b, size, true);
    let push_cpu = map(b, push);
    let push_map = bind(b, push, size, va_p);
    let witness = alloc(b, size, true);
    let witness_cpu = map(b, witness);
    let witness_map = bind(b, witness, size, va_r);
    let value: u64 = 0x5eed_0000_0000_0000
        | u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                & 0xffff_ffff,
        )
        .unwrap();
    witness_cpu.write_u64(0, 0);
    let mut words = Vec::new();
    semaphore(&mut words, va_m, value, true);
    semaphore(&mut words, va_m, value, false);
    semaphore(&mut words, va_r, value, true);
    for (i, word) in words.iter().enumerate() {
        push_cpu.write_u32(i, *word);
    }
    std::sync::atomic::fence(Ordering::SeqCst);
    let token = submit(
        b,
        ctx,
        &[Push {
            address: va_p,
            size: u32::try_from(words.len() * 4).unwrap(),
            flags: 0,
        }],
        &[point(&gate, 1)],
        &[],
    );
    let released = released();
    let start = Instant::now();
    assert_eq!(va_free(b, &[freed_map], va_m, size, released), 0);
    assert_eq!(c::free(b, freed), 0);
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(50), "free took {elapsed:?}");
    assert_eq!(released.calls.load(Ordering::SeqCst), 0);
    assert_eq!(groups(&bridge), 2);
    assert!(!bridge.wait_token(ctx, token, 0).unwrap(), "still gated");
    // Now the GPU runs the writes/reads of the freed range.
    assert_eq!(c::sync_signal(b, &gate, 1), 0);
    assert_eq!(c::wait(b, ctx, token, 10_000_000_000), 0);
    let observed = witness_cpu.read_u64(0);
    assert_eq!(observed, value, "GPU read back the freed memory");
    assert_eq!(c::check_status(b), 0);
    assert!(!bridge.core.is_lost());
    assert_eq!(
        released.calls.load(Ordering::SeqCst),
        1,
        "reclaimed by the status call"
    );
    assert_eq!(groups(&bridge), 0);
    for (cpu, memory, mapping, address) in [
        (push_cpu, push, push_map, va_p),
        (witness_cpu, witness, witness_map, va_r),
    ] {
        assert_eq!(cpu.unmap(), 0);
        assert_eq!(va_free(b, &[mapping], address, size, released), 0);
        assert_eq!(c::free(b, memory), 0);
    }
    assert_eq!(
        released.calls.load(Ordering::SeqCst),
        3,
        "idle frees are immediate"
    );
    assert_eq!(c::sync_destroy(b, gate), 0);
    assert_eq!(c::context_destroy(b, ctx), 0);
    drop(bridge);
    assert_eq!(fd_count(), before, "fd leak");
    println!(
        "PASS: freed range written+read by GPU after free ({elapsed:?}), witness {observed:#x}, \
         unmapped after retirement; no loss"
    );
}

/// Shared VA slot pool for the stress allocators; deferred releases put the
/// slot back (as Mesa's heap would) — a slot must never be released twice or
/// while still mapped (RM rejects an overlapping fixed bind).
struct VaPool {
    free: Mutex<Vec<u64>>,
    released: AtomicU64,
}
impl ReleaseSink for VaPool {
    fn release(&self, address: u64, _size: u64) {
        let mut free = self.free.lock().unwrap();
        assert!(!free.contains(&address), "VA slot released twice");
        free.push(address);
        self.released.fetch_add(1, Ordering::SeqCst);
    }
}
/// Like [`released`]: deferred releases may run until bridge close.
fn va_pool(slots: impl Iterator<Item = u64>) -> &'static VaPool {
    Box::leak(Box::new(VaPool {
        free: Mutex::new(slots.collect()),
        released: AtomicU64::new(0),
    }))
}
const SLOT: u64 = 2 << 20;

/// 5c. Concurrency stress: allocator threads (alloc/bind/map/free), a
/// submitter freeing memory behind host-gated batches, a foreign-gated
/// consumer path, and CPU waiters. Every wait has a 10 s deadline; any
/// timeout, loss or slow free fails the run.
#[test]
#[ignore = "local NVIDIA GPU: alloc/bind/free + gated submits + waits under contention (NV_RM_STRESS_SECONDS)"]
fn concurrent_alloc_free_gated_submit_stress() {
    let seconds: u64 = std::env::var("NV_RM_STRESS_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let before = fd_count();
    let pci = selected_pci();
    let drm = Arc::new(render_node(pci));
    let bridge = open(pci);
    let producer = open(pci);
    let pool = va_pool((0..512).map(|i| VA_BASE + (64 << 20) + i * SLOT));
    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let slow_free = Arc::new(AtomicU64::new(0)); // worst free, ns
    let va_frees = Arc::new(AtomicU64::new(0));
    let timeline = Arc::new(create(&bridge, true, 0));
    let deadline = 10_000_000_000u64;
    let timed = |slow: &AtomicU64, f: &mut dyn FnMut()| {
        let t = Instant::now();
        f();
        slow.fetch_max(
            u64::try_from(t.elapsed().as_nanos()).unwrap(),
            Ordering::Relaxed,
        );
    };
    let take_slot = |pool: &VaPool| -> Option<u64> { pool.free.lock().unwrap().pop() };
    let mut threads = Vec::new();
    // Allocators.
    for seed in 0..3u64 {
        let (b, stop, slow, frees) = (
            Arc::clone(&bridge),
            Arc::clone(&stop),
            Arc::clone(&slow_free),
            Arc::clone(&va_frees),
        );
        threads.push(thread::spawn(move || {
            let mut ops = 0u64;
            let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ seed;
            while !stop.load(Ordering::Relaxed) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let size = 65536 << (x % 5); // 64 KiB .. 1 MiB
                let gart = x & 32 != 0;
                let Some(va) = take_slot(pool) else {
                    thread::yield_now();
                    continue;
                };
                let memory = alloc(&b, size, gart);
                let mapping = bind(&b, memory, size, va);
                if gart && x & 64 != 0 {
                    let cpu = map(&b, memory);
                    cpu.write_u32(0, low_u32(ops));
                    assert_eq!(cpu.unmap(), 0);
                }
                timed(&slow, &mut || {
                    let r = c::va_free(&b, &[mapping], va, SLOT, pool);
                    assert_eq!(r, 0);
                    assert_eq!(c::free(&b, memory), 0);
                });
                frees.fetch_add(1, Ordering::Relaxed);
                ops += 1;
            }
            ops
        }));
    }
    // Submitter: host-gated batches with a driver-internal free while gated.
    {
        let (b, stop, slow, frees, failures, timeline) = (
            Arc::clone(&bridge),
            Arc::clone(&stop),
            Arc::clone(&slow_free),
            Arc::clone(&va_frees),
            Arc::clone(&failures),
            Arc::clone(&timeline),
        );
        threads.push(thread::spawn(move || {
            let ctx = context(&b);
            let gate = create(&b, true, 0);
            let fence = create(&b, false, 0);
            let mut k = 0u64;
            while !stop.load(Ordering::Relaxed) {
                k += 1;
                let Some(va) = take_slot(pool) else {
                    thread::yield_now();
                    k -= 1;
                    continue;
                };
                let memory = alloc(&b, 65536, false);
                let mapping = bind(&b, memory, 65536, va);
                submit(
                    &b,
                    ctx,
                    &[],
                    &[point(&gate, k)],
                    &[point(&fence, 0), point(&timeline, k)],
                );
                timed(&slow, &mut || {
                    assert_eq!(c::va_free(&b, &[mapping], va, SLOT, pool), 0);
                    assert_eq!(c::free(&b, memory), 0);
                });
                frees.fetch_add(1, Ordering::Relaxed);
                assert_eq!(c::sync_signal(&b, &gate, k), 0);
                let f = point(&fence, 0);
                match c::sync_wait_many(&b, &[f], 0, deadline) {
                    0 => {}
                    r => failures
                        .lock()
                        .unwrap()
                        .push(format!("submitter fence wait {r} at {k}")),
                }
                assert_eq!(c::sync_reset(&b, &fence), 0);
            }
            let last = queue_ref(&b.lock().unwrap().0, ctx).unwrap().last_token();
            assert_eq!(c::wait(&b, ctx, last.get(), deadline), 0);
            for id in [gate, fence] {
                assert_eq!(c::sync_destroy(&b, id), 0);
            }
            assert_eq!(c::context_destroy(&b, ctx), 0);
            k
        }));
    }
    // Foreign-gated consumer: producer device signals a sync_file only after
    // the consumer thread freed memory behind the gated batch.
    {
        let (b, p, stop, slow, frees, failures, drm) = (
            Arc::clone(&bridge),
            Arc::clone(&producer),
            Arc::clone(&stop),
            Arc::clone(&slow_free),
            Arc::clone(&va_frees),
            Arc::clone(&failures),
            Arc::clone(&drm),
        );
        threads.push(thread::spawn(move || {
            let ctx = context(&b);
            let pctx = context(&p);
            let gate = create(&p, true, 0);
            let sem = create(&p, false, 0);
            let mut k = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let Some(va) = take_slot(pool) else {
                    thread::yield_now();
                    continue;
                };
                k += 1;
                let fd = pending_sync_file(&p, pctx, &gate, k, &sem, &drm);
                let memory = alloc(&b, 65536, true);
                let mapping = bind(&b, memory, 65536, va);
                let mut token = 0;
                assert_eq!(
                    c::submit_waits(&b, ctx, &[], drm.as_fd(), &[fd.as_fd()], &mut token),
                    0
                );
                timed(&slow, &mut || {
                    assert_eq!(c::va_free(&b, &[mapping], va, SLOT, pool), 0);
                    assert_eq!(c::free(&b, memory), 0);
                });
                frees.fetch_add(1, Ordering::Relaxed);
                assert_eq!(c::sync_signal(&p, &gate, k), 0);
                match c::wait(&b, ctx, token, deadline) {
                    0 => {}
                    r => failures
                        .lock()
                        .unwrap()
                        .push(format!("foreign-gated wait {r} at {k}")),
                }
                drop(fd);
            }
            let last = queue_ref(&p.lock().unwrap().0, pctx).unwrap().last_token();
            assert_eq!(c::wait(&p, pctx, last.get(), deadline), 0);
            for id in [gate, sem] {
                assert_eq!(c::sync_destroy(&p, id), 0);
            }
            assert_eq!(c::context_destroy(&p, pctx), 0);
            assert_eq!(c::context_destroy(&b, ctx), 0);
            k
        }));
    }
    // CPU waiters on the submitter's GPU-signalled timeline.
    for _ in 0..2 {
        let (b, stop, failures, timeline) = (
            Arc::clone(&bridge),
            Arc::clone(&stop),
            Arc::clone(&failures),
            Arc::clone(&timeline),
        );
        threads.push(thread::spawn(move || {
            let mut wakes = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let mut value = 0;
                assert_eq!(c::sync_get_value(&b, &timeline, &mut value), 0);
                let w = point(&timeline, value + 1);
                match c::sync_wait_many(&b, &[w], 0, deadline) {
                    0 => wakes += 1,
                    r => failures
                        .lock()
                        .unwrap()
                        .push(format!("timeline waiter {r} at {value}")),
                }
            }
            wakes
        }));
    }
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) && failures.lock().unwrap().is_empty() {
        thread::sleep(Duration::from_millis(200));
    }
    stop.store(true, Ordering::SeqCst);
    // Release waiters parked on a value the stopped submitter never reaches.
    let submitter_join_guard = Instant::now();
    let counts: Vec<u64> = threads
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            if i == 5 {
                assert_eq!(c::sync_signal(&bridge, &timeline, u64::MAX / 2), 0);
            }
            t.join().unwrap()
        })
        .collect();
    assert!(submitter_join_guard.elapsed() < Duration::from_secs(30));
    let failures = failures.lock().unwrap().clone();
    assert!(failures.is_empty(), "{failures:?}");
    assert!(!bridge.core.is_lost() && !producer.core.is_lost());
    // Everything retired: status calls reclaim the backlog (bounded per call).
    drain_backlog(&bridge);
    assert_eq!(groups(&bridge), 0);
    let frees = va_frees.load(Ordering::SeqCst);
    assert_eq!(
        pool.released.load(Ordering::SeqCst),
        frees,
        "one release per VA free"
    );
    assert_eq!(pool.free.lock().unwrap().len(), 512);
    let timeline = Arc::into_inner(timeline).expect("every thread was joined");
    assert_eq!(c::sync_destroy(&bridge, timeline), 0);
    drop(bridge);
    drop(producer);
    drop(drm);
    assert_eq!(fd_count(), before, "fd leak");
    let worst = Duration::from_nanos(slow_free.load(Ordering::SeqCst));
    println!(
        "PASS: {seconds}s stress: alloc/free ops {:?}, host-gated batches {}, foreign-gated batches {}, \
         waiter wakes {:?}; VA frees {frees} = releases; worst free {worst:?}; 0 timeouts; fd {before}->{before}",
        &counts[..3],
        counts[3],
        counts[4],
        &counts[5..]
    );
    assert!(worst < Duration::from_millis(250), "slow free {worst:?}");
}

/// 5d. Lost-wakeup stress for the watcher's gate release and the
/// leader/follower waiters under contention. Each round: a pending foreign
/// sync_file gates a consumer batch (GPU acquire on a watcher gate) that
/// signals a timeline; two threads CPU-wait that timeline (HOST interrupt ->
/// leader -> follower) and two wait an imported copy of the same sync_file
/// (watcher -> Dekker notify). One host signal on the producer releases all.
/// Every waiter must wake well below the 1 s leader scan period: a lost
/// wakeup would only be rescued by that scan (or the 2 s deadline).
#[test]
#[ignore = "local NVIDIA GPU: watcher gate release + contended CPU waiters, no lost wakeups (NV_RM_WAKE_ROUNDS)"]
fn watcher_gate_release_and_contended_waiters_never_lose_a_wakeup() {
    let rounds: u64 = std::env::var("NV_RM_WAKE_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let before = fd_count();
    let pci = selected_pci();
    let drm = render_node(pci);
    let producer = open(pci);
    let consumer = open(pci);
    let (pp, cp): (&Bridge, &Bridge) = (&producer, &consumer);
    let pctx = context(&producer);
    let cctx = context(&consumer);
    let gate = create(&producer, true, 0);
    let sem = create(&producer, false, 0);
    let gpu_wait = create(&consumer, false, 0);
    let cpu_wait = Arc::new(create(&consumer, false, 0));
    let timeline = Arc::new(create(&consumer, true, 0));
    let mut lat = Vec::new();
    for round in 1..=rounds {
        let fd = pending_sync_file(pp, pctx, &gate, round, &sem, &drm);
        assert_eq!(c::sync_import_fd(cp, &gpu_wait, fd.as_fd(), 0), 0);
        assert_eq!(c::sync_import_fd(cp, &cpu_wait, fd.as_fd(), 0), 0);
        drop(fd);
        submit(
            cp,
            cctx,
            &[],
            &[point(&gpu_wait, 0)],
            &[point(&timeline, round)],
        );
        let ready = Arc::new(Barrier::new(5));
        let waiters: Vec<_> = (0..4)
            .map(|i| {
                let ready = Arc::clone(&ready);
                let cp = Arc::clone(&consumer);
                let (timeline, cpu_wait) = (Arc::clone(&timeline), Arc::clone(&cpu_wait));
                thread::spawn(move || {
                    let target = if i < 2 {
                        point(&timeline, round)
                    } else {
                        point(&cpu_wait, 0)
                    };
                    ready.wait();
                    let r = c::sync_wait_many(&cp, &[target], 0, 2_000_000_000);
                    (r, Instant::now())
                })
            })
            .collect();
        ready.wait();
        while consumer.core.waiters.registered() < 4 {
            thread::yield_now();
        }
        // Let them park (leader in ppoll, followers on the condvar).
        thread::sleep(Duration::from_micros(300));
        let signal = Instant::now();
        assert_eq!(c::sync_signal(pp, &gate, round), 0);
        for w in waiters {
            let (r, at) = w.join().unwrap();
            assert_eq!(
                r, 0,
                "round {round}: wait result {r} (timeout = lost wakeup)"
            );
            let l = at.saturating_duration_since(signal);
            assert!(
                l < Duration::from_millis(100),
                "round {round}: wake after {l:?}"
            );
            lat.push(l);
        }
        assert_eq!(c::sync_reset(cp, &cpu_wait), 0);
    }
    assert!(!consumer.core.is_lost() && !producer.core.is_lost());
    let cpu_wait = Arc::into_inner(cpu_wait).expect("every waiter was joined");
    let timeline = Arc::into_inner(timeline).expect("every waiter was joined");
    for id in [gpu_wait, cpu_wait, timeline] {
        assert_eq!(c::sync_destroy(cp, id), 0);
    }
    for id in [gate, sem] {
        assert_eq!(c::sync_destroy(pp, id), 0);
    }
    assert_eq!(c::context_destroy(pp, pctx), 0);
    assert_eq!(c::context_destroy(cp, cctx), 0);
    drop(producer);
    drop(consumer);
    drop(drm);
    assert_eq!(fd_count(), before, "fd leak");
    lat.sort();
    let q = |p: usize| lat[(lat.len() * p / 100).min(lat.len() - 1)];
    println!(
        "PASS: {rounds} rounds x 4 contended waiters (2 GPU-gated timeline, 2 foreign CPU): \
         p50 {:?} p99 {:?} max {:?}; 0 timeouts",
        q(50),
        q(99),
        lat.last().unwrap()
    );
}

/// poll(2) one descriptor for readability (test helper, libc ABI).
fn poll_readable(fd: i32, timeout_ms: i32) -> bool {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    unsafe extern "C" {
        #[cfg(target_os = "android")]
        fn poll(fds: *mut PollFd, count: u32, timeout: i32) -> i32;
        #[cfg(not(target_os = "android"))]
        fn poll(fds: *mut PollFd, count: std::ffi::c_ulong, timeout: i32) -> i32;
    }
    let mut p = PollFd {
        fd,
        events: 1,
        revents: 0,
    };
    let start = Instant::now();
    loop {
        let left = timeout_ms - i32::try_from(start.elapsed().as_millis()).unwrap();
        // SAFETY: `p` is one exclusive, initialised `struct pollfd` (poll(2):
        // int fd, short events, short revents, the layout PollFd declares) and
        // the count is 1; poll only writes `revents`. Any fd value is allowed.
        let n = unsafe { poll(&raw mut p, 1, left.max(0)) };
        if n > 0 {
            return p.revents & 1 != 0;
        }
        if n == 0 || left <= 0 {
            return false;
        }
    }
}

/// App-startup churn, the pattern of a new Android process presenting its
/// first frames (HAL anb.c + Mesa vk_queue/vk_fence): per frame, a temporary
/// acquire payload signalled by import(-1) (Mesa vk_sync_import_sync_file(-1)
/// -> signal), a render batch waiting on it (and, every other frame, on the
/// peer device's last exported release fence) and signalling a fresh
/// render-done semaphore, a release batch waiting on render-done and
/// signalling a pooled export fence, SYNC_FD export followed by Mesa's reset
/// of that fence, temporary destruction, a temporary AHB-style memory
/// import/free, and context churn. Two devices in one process, two threads
/// each. Every exported sync_file must signal within 100 ms.
#[test]
#[ignore = "local NVIDIA GPU: startup-style sync churn on 2 devices x 2 threads; every export signals < 100 ms (NV_RM_STARTUP_SECONDS)"]
fn startup_sync_churn_every_exported_fence_signals() {
    let seconds: u64 = std::env::var("NV_RM_STARTUP_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let before = fd_count();
    let pci = selected_pci();
    let drm = Arc::new(render_node(pci));
    let devices = [open(pci), open(pci)];
    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    // Last exported release fence of each device, consumed by the other one.
    let latest: Arc<[Mutex<Option<OwnedFd>>; 2]> = Arc::new([Mutex::new(None), Mutex::new(None)]);
    let worst = Arc::new(AtomicU64::new(0));
    let released = Arc::new(Released::default());
    let mut threads = Vec::new();
    for t in 0..4usize {
        let device = t % 2;
        let b = Arc::clone(&devices[device]);
        let (stop, failures, latest, drm, worst) = (
            Arc::clone(&stop),
            Arc::clone(&failures),
            Arc::clone(&latest),
            Arc::clone(&drm),
            Arc::clone(&worst),
        );
        threads.push(thread::spawn(move || {
            let mut ctx = context(&b);
            let fences: Vec<SyncHandle> = (0..3).map(|_| create(&b, false, 0)).collect();
            // Mesa's heap: a range is reusable only after its release ran.
            let pool = va_pool((0..64).map(|i| VA_BASE + (1 << 32) + (u64_from_usize(t) * 64 + i) * SLOT));
            let mut frames = 0u64;
            while !stop.load(Ordering::Relaxed) {
                frames += 1;
                // vkAcquireImageANDROID(-1): temporary payload, CPU-signalled.
                let acquire = create(&b, false, 0);
                assert_eq!(c::sync_signal(&b, &acquire, 0), 0);
                // Foreign wait on the peer device's last release fence.
                let peer = if frames.is_multiple_of(2) {
                    latest[1 - device]
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|fd| fd.try_clone().unwrap())
                } else {
                    None
                };
                let foreign = peer.map(|fd| {
                    let id = create(&b, false, 0);
                    assert_eq!(c::sync_import_fd(&b, &id, fd.as_fd(), 0), 0);
                    id
                });
                let mut waits = vec![point(&acquire, 0)];
                if let Some(id) = &foreign {
                    waits.push(point(id, 0));
                }
                let render_done = create(&b, false, 0);
                submit(&b, ctx, &[], &waits, &[point(&render_done, 0)]);
                // Mesa destroys the submitted temporaries right away.
                assert_eq!(c::sync_destroy(&b, acquire), 0);
                if let Some(id) = foreign {
                    assert_eq!(c::sync_destroy(&b, id), 0);
                }
                // dequeue: temporary AHB import + free (GetMemoryFdProperties).
                let va = pool.free.lock().unwrap().pop().expect("deferred VA ranges are reclaimed");
                let memory = alloc(&b, 65536, true);
                let mapping = bind(&b, memory, 65536, va);
                let r = c::va_free(&b, &[mapping], va, SLOT, pool);
                assert_eq!(r, 0);
                assert_eq!(c::free(&b, memory), 0);
                // vkQueueSignalReleaseImageANDROID: pooled export fence.
                let fence = &fences[usize::try_from(frames % 3).unwrap()];
                let mut ready = 0;
                let p = point(fence, 0);
                assert_eq!(c::sync_status_many(&b, &[p], 0, &mut ready), 0);
                if ready == 1 {
                    assert_eq!(c::sync_reset(&b, fence), 0);
                }
                submit(&b, ctx, &[], &[point(&render_done, 0)], &[point(fence, 0)]);
                let mut fd = -1;
                assert_eq!(
                    c::sync_export_fd(&b, fence, 0, Some(drm.as_fd()), &mut fd),
                    0
                );
                // Copy transference: Mesa resets the fence after export.
                assert_eq!(c::sync_reset(&b, fence), 0);
                assert_eq!(c::sync_destroy(&b, render_done), 0);
                if fd >= 0 {
                    let start = Instant::now();
                    let signalled = poll_readable(fd, 100);
                    let waited = start.elapsed();
                    worst.fetch_max(u64::try_from(waited.as_nanos()).unwrap(), Ordering::Relaxed);
                    let status = crate::fences::sync_status(fd).unwrap_or(-1);
                    if !signalled || status != 1 {
                        failures.lock().unwrap().push(format!(
                            "device {device} thread {t} frame {frames}: export fd not signalled within 100 ms (status {status})"
                        ));
                        stop.store(true, Ordering::SeqCst);
                    }
                    // SAFETY: a successful SYNC_FD export returns a new open
                    // descriptor that only this thread owns
                    // (nv_rm_sync_export_fd hands out `into_raw_fd`); fd >= 0.
                    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
                    *latest[device].lock().unwrap() = Some(owned);
                }
                if frames.is_multiple_of(200) {
                    // Context churn (activity restarts recreate queues).
                    let last = queue_ref(&b.lock().unwrap().0, ctx).unwrap().last_token();
                    assert_eq!(c::wait(&b, ctx, last.get(), 10_000_000_000), 0);
                    assert_eq!(c::context_destroy(&b, ctx), 0);
                    ctx = context(&b);
                }
            }
            let last = queue_ref(&b.lock().unwrap().0, ctx).unwrap().last_token();
            assert_eq!(c::wait(&b, ctx, last.get(), 10_000_000_000), 0);
            for f in fences {
                assert_eq!(c::sync_destroy(&b, f), 0);
            }
            assert_eq!(c::context_destroy(&b, ctx), 0);
            (frames, pool)
        }));
    }
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) && !stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
    stop.store(true, Ordering::SeqCst);
    let joined: Vec<(u64, &'static VaPool)> =
        threads.into_iter().map(|t| t.join().unwrap()).collect();
    let frames: Vec<u64> = joined.iter().map(|(f, _)| *f).collect();
    // Everything retired: one uncontended status call per device reclaims
    // every group; each deferred range came back exactly once.
    for d in &devices {
        drain_backlog(d);
    }
    for (f, pool) in &joined {
        assert_eq!(
            pool.released.load(Ordering::SeqCst),
            *f,
            "one release per VA free"
        );
        released.calls.fetch_add(low_u32(*f), Ordering::SeqCst);
    }
    let failures = failures.lock().unwrap().clone();
    assert!(failures.is_empty(), "{failures:?}");
    for d in &devices {
        assert!(!d.core.is_lost());
    }
    for slot in latest.iter() {
        slot.lock().unwrap().take();
    }
    let releases = released.calls.load(Ordering::SeqCst);
    drop(devices);
    drop(joined); // VA pools outlive every possible release (close runs them)
    drop(drm);
    assert_eq!(fd_count(), before, "fd leak");
    println!(
        "PASS: {seconds}s startup churn, frames per thread {frames:?} (2 devices x 2 threads), every export \
         signalled; worst export wait {:?}; VA releases {releases}; fd {before}->{before}",
        Duration::from_nanos(worst.load(Ordering::SeqCst))
    );
}

/// An exported SYNC_FD must never be a fence nothing can signal: exporting an
/// unsignaled binary payload with no submitted signal (invalid Vulkan usage)
/// fails loudly instead of creating a GPU wait that Mesa's post-export reset
/// would orphan. A signalled payload exports as -1, a submitted one as a
/// real pending fence that signals.
#[test]
#[ignore = "local NVIDIA GPU: SYNC_FD export without submitted signal is an error, never a dead fence"]
fn sync_fd_export_never_returns_an_unsignalable_fence() {
    let before = fd_count();
    let pci = selected_pci();
    let drm = render_node(pci);
    let bridge = open(pci);
    let b: &Bridge = &bridge;
    let ctx = context(b);
    let idle = create(b, false, 0);
    let mut fd = -1;
    assert_ne!(
        c::sync_export_fd(b, &idle, 0, Some(drm.as_fd()), &mut fd),
        0
    );
    assert_eq!(fd, -1);
    assert_eq!(c::sync_signal(b, &idle, 0), 0);
    assert_eq!(
        c::sync_export_fd(b, &idle, 0, Some(drm.as_fd()), &mut fd),
        0
    );
    assert_eq!(fd, -1, "signalled payload exports as already-signalled");
    let submitted = create(b, false, 0);
    submit(b, ctx, &[], &[], &[point(&submitted, 0)]);
    assert_eq!(
        c::sync_export_fd(b, &submitted, 0, Some(drm.as_fd()), &mut fd),
        0
    );
    if fd >= 0 {
        assert!(poll_readable(fd, 100));
        // SAFETY: a successful SYNC_FD export returns a new open descriptor
        // that only this test owns (nv_rm_sync_export_fd hands out
        // `into_raw_fd`); fd >= 0. Dropping it closes it.
        drop(unsafe { OwnedFd::from_raw_fd(fd) });
    }
    assert!(!bridge.core.is_lost());
    for id in [idle, submitted] {
        assert_eq!(c::sync_destroy(b, id), 0);
    }
    assert_eq!(c::context_destroy(b, ctx), 0);
    drop(bridge);
    drop(drm);
    assert_eq!(fd_count(), before);
    println!(
        "PASS: export without submitted signal rejected; signalled -> -1; submitted -> signalling fence"
    );
}

/// Every accepted batch must execute without a later doorbell. Each worker
/// submits one small batch and then waits for exactly that token (100 ms)
/// without submitting anything else on its queue: a doorbell that the GPU
/// host processed before the GP_PUT update became visible would strand the
/// batch until the next submit, which here never comes (the Launcher hang:
/// RenderThread drained a queue whose last batch never ran). Two devices,
/// two queues each, plus BAR1 write traffic on another thread to congest the
/// posted-write path. NV_RM_KICK_SECONDS sets the duration (default 20).
#[test]
#[ignore = "local NVIDIA GPU: every submitted token retires < 100 ms with no later kick (NV_RM_KICK_SECONDS)"]
fn every_submitted_batch_runs_without_a_later_doorbell() {
    let seconds: u64 = std::env::var("NV_RM_KICK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let before = fd_count();
    let pci = selected_pci();
    let devices = [open(pci), open(pci)];
    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let worst = Arc::new(AtomicU64::new(0));
    let slow = Arc::new(Mutex::new(Vec::<String>::new()));
    let mut threads = Vec::new();
    for t in 0..4usize {
        let b = Arc::clone(&devices[t % 2]);
        let (stop, failures, worst, slow) = (
            Arc::clone(&stop),
            Arc::clone(&failures),
            Arc::clone(&worst),
            Arc::clone(&slow),
        );
        threads.push(thread::spawn(move || {
            let ctx = context(&b);
            let fence = create(&b, false, 0);
            let mut n = 0u64;
            // NV_RM_KICK_IDLE_MS: random idle gaps (0..N ms) between batches
            // so the GPU and its memory enter idle/low-power states, as an
            // idle desktop app's sporadic frames do.
            let idle: u64 = std::env::var("NV_RM_KICK_IDLE_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let mut x = 0x2545_f491_4f6c_dd1du64 ^ (u64_from_usize(t) + 1);
            while !stop.load(Ordering::Relaxed) {
                n += 1;
                if idle != 0 {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    thread::sleep(Duration::from_micros(x % (idle * 1000)));
                }
                let token = submit(&b, ctx, &[], &[], &[point(&fence, 0)]);
                let start = Instant::now();
                let r = c::wait(&b, ctx, token, 100_000_000);
                worst.fetch_max(
                    u64::try_from(start.elapsed().as_nanos()).unwrap(),
                    Ordering::Relaxed,
                );
                if r != 0 {
                    // Distinguish slow from stranded: 2 s more without any
                    // doorbell, then one more batch (a doorbell) on this queue.
                    let late = c::wait(&b, ctx, token, 2_000_000_000) == 0;
                    let waited = start.elapsed();
                    if late {
                        // Slow (GPU shared with other clients), not stranded:
                        // it ran without any further doorbell. Reported.
                        slow.lock()
                            .unwrap()
                            .push(format!("thread {t} batch {n}: {waited:?}"));
                    } else {
                        let kick = Instant::now();
                        let next = submit(&b, ctx, &[], &[], &[]);
                        let after_kick = c::wait(&b, ctx, token, 1_000_000_000) == 0;
                        let next_done = c::wait(&b, ctx, next, 1_000_000_000) == 0;
                        failures.lock().unwrap().push(format!(
                            "thread {t} batch {n} token {token}: STRANDED, not retired within \
                             {waited:?} without a doorbell; retired after the next doorbell: \
                             {after_kick} ({:?}); next batch done: {next_done}",
                            kick.elapsed()
                        ));
                        stop.store(true, Ordering::SeqCst);
                    }
                }
                assert_eq!(c::sync_reset(&b, &fence), 0);
            }
            let last = queue_ref(&b.lock().unwrap().0, ctx).unwrap().last_token();
            assert_eq!(c::wait(&b, ctx, last.get(), 10_000_000_000), 0);
            assert_eq!(c::sync_destroy(&b, fence), 0);
            assert_eq!(c::context_destroy(&b, ctx), 0);
            n
        }));
    }
    // BAR1 traffic: CPU writes into mapped VRAM on both devices (off when
    // NV_RM_KICK_IDLE_MS asks for an idle GPU).
    let bar1 = std::env::var("NV_RM_KICK_IDLE_MS").is_err();
    for d in (0..2usize).filter(|_| bar1) {
        let b = Arc::clone(&devices[d]);
        let stop = Arc::clone(&stop);
        threads.push(thread::spawn(move || {
            let size = 4u64 << 20;
            let memory = alloc(&b, size, false);
            let cpu = map(&b, memory);
            assert_eq!(
                cpu.len(),
                usize::try_from(size).unwrap(),
                "the fill covers exactly the allocation"
            );
            let mut words = 0u64;
            while !stop.load(Ordering::Relaxed) {
                words = cpu.fill_u64_counting(words);
            }
            assert_eq!(cpu.unmap(), 0);
            assert_eq!(c::free(&b, memory), 0);
            words
        }));
    }
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) && !stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
    stop.store(true, Ordering::SeqCst);
    let counts: Vec<u64> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    let failures = failures.lock().unwrap().clone();
    for d in &devices {
        assert!(!d.core.is_lost());
    }
    drop(devices);
    assert_eq!(fd_count(), before, "fd leak");
    assert!(
        failures.is_empty(),
        "{} stranded batches: {failures:?}",
        failures.len()
    );
    let slow = slow.lock().unwrap().clone();
    println!(
        "PASS: {seconds}s, batches per queue {:?}, BAR1 words {:?}, 0 stranded (every token retired \
         without a later doorbell); worst wait {:?}; retired after > 100 ms (no doorbell needed): {} {slow:?}",
        &counts[..4],
        &counts[4..],
        Duration::from_nanos(worst.load(Ordering::SeqCst)),
        slow.len()
    );
}

/// A GPU acquire released only by the CPU (host timeline signal, the path of
/// watcher gate releases and vkSignalSemaphore) must be re-polled by the GPU
/// without any doorbell, also after idle periods. Each worker submits a batch
/// gated on a host timeline value, idles a random 0..N ms (the channel parks
/// on the failed acquire, the GPU may enter idle states), signals on the CPU
/// and requires retirement within 100 ms with no further submit.
/// NV_RM_KICK_SECONDS / NV_RM_KICK_IDLE_MS as above (idle default 30).
#[test]
#[ignore = "local NVIDIA GPU: CPU-released GPU acquires retire < 100 ms after idle gaps, no doorbell"]
fn cpu_released_acquire_runs_without_a_doorbell_after_idle() {
    let seconds: u64 = std::env::var("NV_RM_KICK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let idle: u64 = std::env::var("NV_RM_KICK_IDLE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30)
        .max(1);
    let before = fd_count();
    let pci = selected_pci();
    let devices = [open(pci), open(pci)];
    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let worst = Arc::new(AtomicU64::new(0));
    let slow = Arc::new(AtomicU64::new(0));
    let threads: Vec<_> = (0..4usize)
        .map(|t| {
            let b = Arc::clone(&devices[t % 2]);
            let (stop, failures, worst, slow) = (Arc::clone(&stop), Arc::clone(&failures), Arc::clone(&worst), Arc::clone(&slow));
            thread::spawn(move || {
                let ctx = context(&b);
                let gate = create(&b, true, 0);
                let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ (u64_from_usize(t) + 7);
                let mut n = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    n += 1;
                    let token = submit(&b, ctx, &[], &[point(&gate, n)], &[]);
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    thread::sleep(Duration::from_micros(x % (idle * 1000)));
                    assert_eq!(c::sync_signal(&b, &gate, n), 0);
                    let start = Instant::now();
                    let r = c::wait(&b, ctx, token, 100_000_000);
                    worst.fetch_max(u64::try_from(start.elapsed().as_nanos()).unwrap(), Ordering::Relaxed);
                    if r != 0 {
                        let late = c::wait(&b, ctx, token, 2_000_000_000) == 0;
                        if late {
                            // Slow under GPU contention, not stranded.
                            slow.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let next = submit(&b, ctx, &[], &[], &[]);
                        let after = c::wait(&b, ctx, token, 1_000_000_000) == 0;
                        let drained = c::wait(&b, ctx, next, 1_000_000_000);
                        failures.lock().unwrap().push(format!(
                            "thread {t} batch {n}: acquire not re-polled within 100 ms of the CPU release; \
                             STRANDED: not retired within 2.1 s without doorbell (late={late}); after a doorbell: {after} \
                             (next batch wait status {drained})"
                        ));
                        stop.store(true, Ordering::SeqCst);
                    }
                }
                let last = queue_ref(&b.lock().unwrap().0, ctx).unwrap().last_token();
                assert_eq!(c::wait(&b, ctx, last.get(), 10_000_000_000), 0);
                assert_eq!(c::sync_destroy(&b, gate), 0);
                assert_eq!(c::context_destroy(&b, ctx), 0);
                n
            })
        })
        .collect();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) && !stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
    stop.store(true, Ordering::SeqCst);
    let counts: Vec<u64> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    let failures = failures.lock().unwrap().clone();
    for d in &devices {
        assert!(!d.core.is_lost());
    }
    drop(devices);
    assert_eq!(fd_count(), before, "fd leak");
    assert!(failures.is_empty(), "{failures:?}");
    println!(
        "PASS: {seconds}s, gated batches per queue {counts:?}, 0 stranded (every CPU-released acquire \
         retired without a doorbell); worst wait after release {:?}; slow > 100 ms (no doorbell needed): {}",
        Duration::from_nanos(worst.load(Ordering::SeqCst)),
        slow.load(Ordering::SeqCst)
    );
}
