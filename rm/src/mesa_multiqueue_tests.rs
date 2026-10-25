//! Local-GPU fixtures for contexts that share a TSG (ignored by default).
//! Two compute contexts in one TSG on different subcontexts (VEIDs) run on
//! the GR engine at the same time; the same pair in two TSGs is time-sliced.
//! Run alone (host NAK bridge built by native/compiler):
//!   NV_NATIVE_COMPILER_PATH=.../libnv_native_nak.so cargo test --offline
//!   --lib mesa::multiqueue_tests -- --ignored --test-threads=1 --nocapture
use super::*;
use crate::{compute::ComputeJob, gpu::GpuMapping, rm::Mapping};

/// 16 workgroups of 64 invocations: a small fraction of the SMs, so two such
/// dispatches fit side by side when the hardware runs them concurrently.
const GROUPS: u32 = 16;
const INVOCATIONS: usize = GROUPS as usize * 64;
/// Target duration of one dispatch. Every workgroup runs for about this long,
/// far below RM's default context-switch timeout (RmEngineContextSwitchTimeoutUs
/// 0x3fffff, nvrm_registry.h), which a time-sliced control has to drain.
const TARGET: Duration = Duration::from_millis(120);

fn open() -> Box<Bridge> {
    let client = Client::open().unwrap();
    let card = client
        .cards()
        .unwrap()
        .into_iter()
        .next()
        .expect("NVIDIA RM GPU");
    let pci = [
        card.pci.domain,
        card.pci.bus as u32,
        card.pci.slot as u32,
        card.pci.function as u32,
    ];
    Box::new(Bridge::new(pci).unwrap().0)
}
fn context(b: &mut Bridge, engines: u32, share: Option<u64>) -> u64 {
    let mut id = 0;
    let status = match share {
        None => unsafe { nv_rm_context_create(b, engines, &mut id) },
        Some(share) => unsafe { nv_rm_context_create_shared(b, 16, share, &mut id) },
    };
    assert_eq!(status, 0);
    id
}
fn subcontext(b: &Bridge, ctx: u64) -> (u32, u32) {
    queue_ref(&b.lock().0, ctx).unwrap().subcontext()
}

/// `iters` steps of x = x * 1664525 + 1013904223 (mod 2^32) as one affine map.
fn lcg_power(mut iters: u32) -> (u32, u32) {
    let (mut a, mut c) = (1u32, 0u32);
    let (mut sa, mut sc) = (1664525u32, 1013904223u32);
    while iters != 0 {
        if iters & 1 != 0 {
            // (a, c) after (sa, sc): x -> sa * (a x + c) + sc.
            c = sa.wrapping_mul(c).wrapping_add(sc);
            a = sa.wrapping_mul(a);
        }
        sc = sa.wrapping_mul(sc).wrapping_add(sc);
        sa = sa.wrapping_mul(sa);
        iters >>= 1;
    }
    (a, c)
}

struct Work {
    seed: u32,
    iters: u32,
    push: Push,
    _job: ComputeJob,
    _push_cpu: Mapping,
    _push_va: GpuMapping,
    out: Mapping,
    _out_va: GpuMapping,
    _done_va: GpuMapping,
}
impl Work {
    fn new(b: &Bridge, program: &crate::compiler::Shader, iters: u32, seed: u32) -> Self {
        let guard = b.lock();
        let inner = &guard.0;
        let (device, space) = (&inner.device, &inner.space);
        let compute = crate::capabilities::query(device).unwrap().compute;
        let out_mem = device.allocate_vram(INVOCATIONS * 4).unwrap();
        let out_va = space.map(&out_mem).unwrap();
        let mut out = out_mem.map().unwrap();
        for i in 0..INVOCATIONS {
            out.write_u32(i * 4, 0).unwrap();
        }
        let done_mem = device.allocate_vram(4096).unwrap();
        let done_va = space.map(&done_mem).unwrap();
        // Root table of the NAK bridge: binding 0 address + size at 0,
        // push constants at 1024.
        let mut constants = vec![0u8; 1152];
        constants[..8].copy_from_slice(&out_va.address().to_le_bytes());
        constants[8..12].copy_from_slice(&((INVOCATIONS * 4) as u32).to_le_bytes());
        constants[1024..1028].copy_from_slice(&iters.to_le_bytes());
        constants[1028..1032].copy_from_slice(&seed.to_le_bytes());
        let job = ComputeJob::compiled(
            device,
            space,
            done_va.address(),
            program,
            &constants,
            [GROUPS, 1, 1],
        )
        .unwrap();
        let mut words = Vec::new();
        crate::gpu::method(&mut words, 1, 0, &[compute]); // SET_OBJECT
        crate::gpu::method(
            &mut words,
            1,
            0x7b0,
            &crate::gpu::hi_lo(crate::compute::LOCAL_WINDOW),
        );
        crate::gpu::method(
            &mut words,
            1,
            0x2a0,
            &crate::gpu::hi_lo(crate::compute::SHARED_WINDOW),
        );
        crate::gpu::method(&mut words, 1, 0x2b4, &[(job.address >> 8) as u32]); // SEND_PCAS_A
        crate::gpu::method(&mut words, 1, 0x2c0, &[9]); // SEND_SIGNALING_PCAS2_B: invalidate + schedule
        let push_mem = device.allocate_aligned(4096, 4096, false, true).unwrap();
        let push_va = space.map(&push_mem).unwrap();
        let mut push_cpu = push_mem.map().unwrap();
        for (i, word) in words.iter().enumerate() {
            push_cpu.write_u32(i * 4, *word).unwrap();
        }
        let push = Push {
            address: push_va.address(),
            size: (words.len() * 4) as u32,
            flags: 0,
        };
        Self {
            seed,
            iters,
            push,
            _job: job,
            _push_cpu: push_cpu,
            _push_va: push_va,
            out,
            _out_va: out_va,
            _done_va: done_va,
        }
    }
    fn clear(&mut self) {
        for i in 0..INVOCATIONS {
            self.out.write_u32(i * 4, 0).unwrap();
        }
    }
    fn check(&self) {
        let (a, c) = lcg_power(self.iters);
        for i in 0..INVOCATIONS {
            let x = self.seed.wrapping_add(i as u32);
            let want = a.wrapping_mul(x).wrapping_add(c);
            let got = self.out.read_u32(i * 4).unwrap();
            assert_eq!(got, want, "seed {:#x} invocation {i}", self.seed);
        }
    }
}

/// Submit every (context, work) at once, wait for all; wall time from the
/// first submission to the last completion. Results are checked afterwards.
fn run(b: &mut Bridge, set: &mut [(u64, &mut Work)]) -> Duration {
    for (_, w) in set.iter_mut() {
        w.clear();
    }
    let start = Instant::now();
    let mut tokens = Vec::new();
    for (ctx, w) in set.iter() {
        let mut token = 0;
        let status = unsafe {
            nv_rm_submit_batch(
                b,
                *ctx,
                &w.push,
                1,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut token,
            )
        };
        assert_eq!(status, 0);
        tokens.push((*ctx, token));
    }
    for (ctx, token) in tokens {
        assert!(
            b.wait_token(ctx, token, 10_000_000_000).unwrap(),
            "context {ctx} did not complete in 10 s"
        );
    }
    let elapsed = start.elapsed();
    for (_, w) in set.iter() {
        w.check();
    }
    elapsed
}
fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
#[ignore = "local NVIDIA GPU: two contexts in one TSG run concurrently, two TSGs time-slice"]
fn shared_tsg_contexts_run_concurrently() {
    let words: Vec<u32> = include_bytes!("../shaders/lcg-spin.spv")
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let program = crate::compiler::compile(&words, c"main").unwrap();
    let mut bridge = open();
    let b = &mut *bridge;
    // The owner is a graphics queue's context (copy + 3D + compute, ZCULL
    // bound), as for the stock graphics queue whose TSG the compute queue
    // joins; the other two are compute-only.
    let owner = context(b, 1 | 4 | 16, None);
    let shared = context(b, 16, Some(owner));
    let separate = context(b, 16, None);
    let (veid_owner, tsg_owner) = subcontext(b, owner);
    let (veid_shared, tsg_shared) = subcontext(b, shared);
    let (veid_separate, tsg_separate) = subcontext(b, separate);
    println!(
        "owner VEID {veid_owner} TSG {tsg_owner:#x}; shared VEID {veid_shared} TSG {tsg_shared:#x}; separate VEID {veid_separate} TSG {tsg_separate:#x}"
    );
    // One ZCULL buffer per TSG, bound by its graphics channel.
    let zcull = |b: &Bridge, ctx| queue_ref(&b.lock().0, ctx).unwrap().zcull_range();
    assert!(zcull(b, owner).is_some());
    assert_eq!(zcull(b, shared), zcull(b, owner));
    assert_eq!(zcull(b, separate), None);
    assert_eq!((veid_owner, veid_shared, veid_separate), (0, 1, 0));
    assert_eq!(tsg_owner, tsg_shared);
    assert_ne!(tsg_owner, tsg_separate);
    // Non-compute engines on a shared context are refused with a reason,
    // and the refusal leaves the device usable.
    let mut refused = 0;
    assert_ne!(
        unsafe { nv_rm_context_create_shared(b, 4 | 16, owner, &mut refused) },
        0
    );
    assert!(!b.core.is_lost());

    // Calibrate the iteration count for TARGET on the owner context alone.
    let probe_iters = 1 << 20;
    let mut probe = Work::new(b, &program, probe_iters, 0x1234);
    run(b, &mut [(owner, &mut probe)]);
    let t = run(b, &mut [(owner, &mut probe)]);
    let iters = ((f64::from(probe_iters) * TARGET.as_secs_f64() / t.as_secs_f64()) as u32)
        .clamp(1 << 16, 1 << 28);
    println!(
        "calibration: {probe_iters} iterations {:.2} ms -> {iters} iterations",
        ms(t)
    );
    drop(probe);

    let mut wa = Work::new(b, &program, iters, 0x1000_0000);
    let mut wb = Work::new(b, &program, iters, 0x2000_0000);
    let mut wc = Work::new(b, &program, iters, 0x3000_0000);
    // Warm-up of every context.
    run(b, &mut [(owner, &mut wa)]);
    run(b, &mut [(shared, &mut wb)]);
    run(b, &mut [(separate, &mut wc)]);
    let (mut ta, mut tb, mut tc, mut tab, mut tac) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..5 {
        ta.push(run(b, &mut [(owner, &mut wa)]));
        tb.push(run(b, &mut [(shared, &mut wb)]));
        tc.push(run(b, &mut [(separate, &mut wc)]));
        tab.push(run(b, &mut [(owner, &mut wa), (shared, &mut wb)]));
        tac.push(run(b, &mut [(owner, &mut wa), (separate, &mut wc)]));
    }
    let (ta, tb, tc, tab, tac) = (median(ta), median(tb), median(tc), median(tab), median(tac));
    let shared_factor = (ta + tb).as_secs_f64() / tab.as_secs_f64();
    let separate_factor = (ta + tc).as_secs_f64() / tac.as_secs_f64();
    println!(
        "alone: A {:.2} ms, B (shared TSG) {:.2} ms, C (own TSG) {:.2} ms; A+B one TSG {:.2} ms (overlap {:.2}x); A+C two TSGs {:.2} ms (overlap {:.2}x)",
        ms(ta),
        ms(tb),
        ms(tc),
        ms(tab),
        shared_factor,
        ms(tac),
        separate_factor
    );
    assert!(
        shared_factor > 1.6,
        "one TSG did not run the two contexts concurrently"
    );
    assert!(
        separate_factor < 1.25,
        "two TSGs overlapped: the control does not time-slice"
    );

    // The owner goes first; its TSG, VEID 0 share and ZCULL buffer stay alive
    // for the shared context, which keeps working; VEID 0 is not handed out
    // again, the next context gets VEID 2.
    assert_eq!(unsafe { nv_rm_context_destroy(b, owner) }, 0);
    run(b, &mut [(shared, &mut wb)]);
    let again = context(b, 16, Some(shared));
    assert_eq!(subcontext(b, again), (2, tsg_shared));
    run(b, &mut [(again, &mut wa), (shared, &mut wb)]);
    for ctx in [again, shared, separate] {
        assert_eq!(unsafe { nv_rm_context_destroy(b, ctx) }, 0);
    }
    assert!(!b.core.is_lost());
    drop((wa, wb, wc));
    drop(bridge);
}
