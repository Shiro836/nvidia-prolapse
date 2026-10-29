// SPDX-License-Identifier: MIT
//! The context scenarios (`submit-fusion`): creation and destruction of
//! channel, bind-only and shared contexts and their failures; how pending
//! pushes, GPU waits and signals fuse into `nv_rm_submit_batch` calls (empty,
//! ordinary, exactly-full and split lists, continuations across the cut);
//! ring-pressure retries; failures that keep exactly the unaccepted work;
//! refusals checked before anything is submitted; `sync`; and the CPU path of
//! bind-only contexts.
use nv_rm_abi::NV_RM_EXEC_INCOMPLETE;
use nvkmd_sys::{
    NVKMD_BIND_OP_BIND, NVKMD_BIND_OP_UNBIND, NVKMD_ENGINE_3D, NVKMD_ENGINE_BIND,
    NVKMD_ENGINE_COMPUTE, NVKMD_ENGINE_COPY, NVKMD_MEM_GART, VK_ERROR_DEVICE_LOST,
    VK_ERROR_FEATURE_NOT_PRESENT, VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VK_ERROR_OUT_OF_HOST_MEMORY, VK_NOT_READY, VK_SUCCESS, VK_SYNC_IS_TIMELINE, VkResult,
    nvkmd_ctx_bind, nvkmd_ctx_exec, vk_sync,
};

use crate::fake::{self, live_allocations, note, state};
use crate::fake_ctx::{FIRST_TOKEN, RING_TOKEN};

use crate::fake_sync::FIRST_SYNC;
use crate::mesa::{self, Ctx, Dev, ForeignSync, Pdev, Sync, VkDevice};

/// First push address of the suites' lists; pushes are 4 bytes apart.
const PUSH_BASE: u64 = 0x1000;
const KIB: u64 = 1024;

struct World {
    pdev: Pdev,
    vk: VkDevice,
    dev: Dev,
}

impl World {
    fn new() -> Self {
        fake::reset();
        let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");
        let vk = VkDevice::new();
        let dev = pdev.create_dev(Some(&vk)).expect("the KMD device opens");
        vk.attach(&dev);
        state().loss_device = vk.as_ptr().expose_provenance();
        Self { pdev, vk, dev }
    }

    fn case(&self, name: &str) {
        fake::reset_case();
        self.vk.clear_lost();
        note(format!("-- {name}"));
    }

    fn channel(&self, engines: u32) -> Ctx {
        self.dev
            .create_ctx(Some(&self.vk), engines)
            .expect("the fake RM creates contexts")
    }

    fn destroy(self) {
        self.vk.clear_lost();
        self.dev.destroy();
        state().loss_device = 0;
        drop(self.vk);
        self.pdev.destroy();
        assert_eq!(live_allocations(), 0, "devices free what they allocated");
        let state = state();
        assert_eq!(state.rm_opens, state.rm_closes, "every RM device is closed");
    }
}

/// `count` complete 4-byte pushes from [`PUSH_BASE`] on.
fn pushes(count: u64) -> Vec<nvkmd_ctx_exec> {
    (0..count)
        .map(|i| nvkmd_ctx_exec {
            addr: PUSH_BASE + 4 * i,
            size_B: 4,
            incomplete: false,
            no_prefetch: false,
        })
        .collect()
}

/// What a case's submissions looked like: (pushes, first address, waits,
/// signals, status) per batch.
fn batches() -> Vec<(usize, u64, usize, usize, VkResult)> {
    state()
        .ctx
        .batches
        .iter()
        .map(|batch| {
            assert!(
                batch
                    .pushes
                    .windows(2)
                    .all(|pair| pair[1].address == pair[0].address + 4),
                "a batch carries consecutive pushes in exec order"
            );
            (
                batch.pushes.len(),
                batch.pushes.first().map_or(0, |push| push.address),
                batch.waits.len(),
                batch.signals.len(),
                batch.status,
            )
        })
        .collect()
}

fn ring_waits() -> usize {
    state()
        .ctx
        .waits
        .iter()
        .filter(|&&(_, token, timeout)| token == RING_TOKEN && timeout == u64::MAX)
        .count()
}

fn outcome(case: &str, result: VkResult) {
    note(format!("{case}: {result}"));
}

/// The two payloads every fusion case uses: a timeline (signalled to 9,
/// waited for at 8) and a binary one.
struct Payloads {
    timeline: Sync,
    binary: Sync,
}

impl Payloads {
    fn new(world: &World) -> Self {
        let opaque = world.pdev.sync_types()[0];
        let timeline =
            Sync::init(&world.vk, opaque, VK_SYNC_IS_TIMELINE, 0).expect("a timeline payload");
        let binary = Sync::init(&world.vk, opaque, 0, 0).expect("a binary payload");
        Self { timeline, binary }
    }

    fn wait(&self) -> [(*mut vk_sync, u64); 1] {
        [(self.timeline.as_ptr(), 8)]
    }

    fn signals(&self) -> [(*mut vk_sync, u64); 2] {
        [(self.timeline.as_ptr(), 9), (self.binary.as_ptr(), 0)]
    }

    fn finish(self, vk: &VkDevice) {
        self.timeline.finish(vk);
        self.binary.finish(vk);
    }
}

pub(crate) fn submit_fusion() {
    let world = World::new();
    creation(&world);
    let payloads = Payloads::new(&world);
    assert_eq!(state().sync.creates, 2, "two payloads from {FIRST_SYNC} on");
    let ctx = world.channel(NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE);
    fusion(&world, &ctx, &payloads);
    ring_pressure(&world, &ctx, &payloads);
    failures(&world, &ctx, &payloads);
    refusals(&world, &ctx, &payloads);
    ordering(&world, &ctx, &payloads);
    syncs(&world, &ctx, &payloads);
    world.case("destroy");
    ctx.destroy();
    assert_eq!(state().ctx.destroys, 1);
    bind_context(&world, &payloads);
    world.case("finish");
    payloads.finish(&world.vk);
    world.destroy();
}

/// Channel, bind-only and shared contexts; creation failures; destruction.
fn creation(world: &World) {
    world.case("create");
    let ctx = world.channel(NVKMD_ENGINE_COPY);
    let bind = world
        .dev
        .create_ctx(Some(&world.vk), NVKMD_ENGINE_BIND)
        .expect("a bind context");
    assert_eq!(state().ctx.creates, 1, "a bind context has no RM context");
    let shared = world
        .dev
        .create_shared_ctx(Some(&world.vk), NVKMD_ENGINE_COMPUTE, Some(&ctx));
    outcome(
        "shared context",
        shared.as_ref().map_or_else(|e| *e, |_| VK_SUCCESS),
    );
    let shared = shared.expect("a shared context in the channel's group");

    let refusals: [(&str, Result<Ctx, VkResult>); 3] = [
        (
            "shared context without a share",
            world
                .dev
                .create_shared_ctx(Some(&world.vk), NVKMD_ENGINE_COMPUTE, None),
        ),
        (
            "shared context of a bind context",
            world
                .dev
                .create_shared_ctx(Some(&world.vk), NVKMD_ENGINE_COMPUTE, Some(&bind)),
        ),
        (
            "bind-only shared context",
            world
                .dev
                .create_shared_ctx(Some(&world.vk), NVKMD_ENGINE_BIND, Some(&ctx)),
        ),
    ];
    for (case, result) in refusals {
        let result = result.map(Ctx::destroy);
        note(format!("{case}: {result:?}"));
        assert_eq!(result, Err(VK_ERROR_INITIALIZATION_FAILED), "{case}");
    }
    assert!(
        state().diagnostic.contains("share"),
        "the refusal names the share"
    );
    assert_eq!(state().ctx.creates, 2, "refused shares reach no RM entry");

    state().ctx.create_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    let result = world
        .dev
        .create_ctx(Some(&world.vk), NVKMD_ENGINE_3D)
        .map(Ctx::destroy);
    note(format!("RM context creation fails: {result:?}"));
    assert_eq!(result, Err(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    let result = world
        .dev
        .create_shared_ctx(Some(&world.vk), NVKMD_ENGINE_COMPUTE, Some(&ctx))
        .map(Ctx::destroy);
    note(format!("RM shared context creation fails: {result:?}"));
    assert_eq!(result, Err(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    state().ctx.create_status = VK_SUCCESS;
    assert!(
        state().diagnostic.contains("create"),
        "the failure names the creation"
    );

    let creates = state().ctx.creates;
    fake::fail_next_calloc();
    let result = world
        .dev
        .create_ctx(Some(&world.vk), NVKMD_ENGINE_3D)
        .map(Ctx::destroy);
    fake::stop_failing_calloc();
    note(format!("context allocation fails: {result:?}"));
    assert_eq!(result, Err(VK_ERROR_OUT_OF_HOST_MEMORY));
    assert_eq!(
        state().ctx.creates,
        creates,
        "no RM context without storage"
    );

    shared.destroy();
    bind.destroy();
    state().ctx.destroy_status = VK_ERROR_DEVICE_LOST;
    ctx.destroy();
    state().ctx.destroy_status = VK_SUCCESS;
    assert_eq!(state().ctx.destroys, 2, "one RM destroy per channel");
    assert!(
        !world.vk.is_lost(),
        "a failed bounded destroy is reported, not the device's loss"
    );
}

/// Empty, ordinary, exactly-full and split lists: only the first batch
/// carries the waits and only the last the signals; a continuation across
/// entry 1000 moves the cut back.
fn fusion(world: &World, ctx: &Ctx, payloads: &Payloads) {
    for total in [0_u64, 1, 1000, 1001, 2000] {
        world.case(&format!("{total} pushes"));
        assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
        assert_eq!(ctx.exec(&world.vk, &pushes(total)), VK_SUCCESS);
        let result = ctx.signal(&world.vk, &payloads.signals());
        outcome("signal", result);
        assert_eq!(result, VK_SUCCESS);
        let expected = if total <= 1000 {
            vec![(
                usize::try_from(total).expect("small"),
                if total == 0 { 0 } else { PUSH_BASE },
                1,
                2,
                VK_SUCCESS,
            )]
        } else {
            vec![
                (1000, PUSH_BASE, 1, 0, VK_SUCCESS),
                (
                    usize::try_from(total - 1000).expect("small"),
                    PUSH_BASE + 4000,
                    0,
                    2,
                    VK_SUCCESS,
                ),
            ]
        };
        assert_eq!(batches(), expected, "{total} pushes");
        assert_eq!(ring_waits(), 0);
    }

    world.case("continuation across the cut");
    let mut list = pushes(1001);
    list[999].incomplete = true;
    assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(ctx.exec(&world.vk, &list), VK_SUCCESS);
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    assert_eq!(
        batches(),
        vec![
            (999, PUSH_BASE, 1, 0, VK_SUCCESS),
            (2, PUSH_BASE + 4 * 999, 0, 2, VK_SUCCESS)
        ]
    );
    let flags: Vec<u32> = state().ctx.batches[1]
        .pushes
        .iter()
        .map(|push| push.flags)
        .collect();
    assert_eq!(flags, vec![NV_RM_EXEC_INCOMPLETE, 0]);

    world.case("signals only");
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    assert_eq!(batches(), vec![(0, 0, 0, 2, VK_SUCCESS)]);
    world.case("waits only");
    assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(batches(), vec![(0, 0, 1, 0, VK_SUCCESS)]);
    world.case("empty flush");
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(ctx.signal(&world.vk, &[]), VK_SUCCESS);
    assert_eq!(batches(), vec![], "nothing to submit, no batch");
    world.case("zero-signal call");
    assert_eq!(ctx.exec(&world.vk, &pushes(1)), VK_SUCCESS);
    assert_eq!(ctx.signal(&world.vk, &[]), VK_SUCCESS);
    assert_eq!(batches(), vec![(1, PUSH_BASE, 0, 0, VK_SUCCESS)]);
    world.case("timeline wait for 0");
    let skipped = [(payloads.timeline.as_ptr(), 0)];
    assert_eq!(ctx.wait(&world.vk, &skipped), VK_SUCCESS);
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(batches(), vec![], "a wait for value 0 is satisfied");
}

/// `VK_NOT_READY` accepts nothing: the flushing thread waits for the named
/// batch and resubmits the same work, waits and signals included.
fn ring_pressure(world: &World, ctx: &Ctx, payloads: &Payloads) {
    world.case("ring pressure");
    state().ctx.submit_script.extend([VK_NOT_READY]);
    assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(ctx.exec(&world.vk, &pushes(1)), VK_SUCCESS);
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    assert_eq!(
        batches(),
        vec![
            (1, PUSH_BASE, 1, 2, VK_NOT_READY),
            (1, PUSH_BASE, 1, 2, VK_SUCCESS)
        ]
    );
    assert_eq!(ring_waits(), 1);

    world.case("ring pressure on both batches of a split list");
    state()
        .ctx
        .submit_script
        .extend([VK_NOT_READY, VK_SUCCESS, VK_NOT_READY, VK_SUCCESS]);
    assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(ctx.exec(&world.vk, &pushes(1001)), VK_SUCCESS);
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    assert_eq!(
        batches(),
        vec![
            (1000, PUSH_BASE, 1, 0, VK_NOT_READY),
            (1000, PUSH_BASE, 1, 0, VK_SUCCESS),
            (1, PUSH_BASE + 4000, 0, 2, VK_NOT_READY),
            (1, PUSH_BASE + 4000, 0, 2, VK_SUCCESS)
        ]
    );
    assert_eq!(ring_waits(), 2);
    assert_eq!(ctx.sync(&world.vk), VK_SUCCESS);
    let last = state().ctx.waits.last().copied();
    assert_eq!(
        last.map(|(_, token, _)| token),
        Some(FIRST_TOKEN + 3),
        "the last accepted batch is the context's completion point"
    );
}

/// Failed ring waits and refused batches keep every unaccepted push, wait
/// and signal; a failure after an accepted batch never replays it.
fn failures(world: &World, ctx: &Ctx, payloads: &Payloads) {
    for ring in [false, true] {
        world.case(if ring {
            "ring wait fails"
        } else {
            "batch refused"
        });
        {
            let mut state = state();
            state.ctx.wait_status = VK_ERROR_DEVICE_LOST;
            state.ctx.submit_script.extend([if ring {
                VK_NOT_READY
            } else {
                VK_ERROR_DEVICE_LOST
            }]);
        }
        assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
        assert_eq!(ctx.exec(&world.vk, &pushes(1)), VK_SUCCESS);
        let result = ctx.signal(&world.vk, &payloads.signals());
        outcome("signal", result);
        assert_eq!(result, VK_ERROR_DEVICE_LOST);
        assert_eq!(ring_waits(), usize::from(ring));
        assert!(
            state().diagnostic.contains("submit"),
            "the failure names the submission"
        );
        state().ctx.wait_status = VK_SUCCESS;
        // The retry carries the same push and wait (the signals belong to
        // the failed call).
        assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
        let retried = batches();
        assert_eq!(
            retried.last(),
            Some(&(1, PUSH_BASE, 1, 0, VK_SUCCESS)),
            "unaccepted work is kept"
        );
    }

    world.case("failure after an accepted batch");
    state()
        .ctx
        .submit_script
        .extend([VK_SUCCESS, VK_ERROR_DEVICE_LOST]);
    assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(ctx.exec(&world.vk, &pushes(1001)), VK_SUCCESS);
    assert_eq!(
        ctx.signal(&world.vk, &payloads.signals()),
        VK_ERROR_DEVICE_LOST
    );
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    assert_eq!(
        batches(),
        vec![
            (1000, PUSH_BASE, 1, 0, VK_SUCCESS),
            (1, PUSH_BASE + 4000, 0, 2, VK_ERROR_DEVICE_LOST),
            (1, PUSH_BASE + 4000, 0, 2, VK_SUCCESS)
        ],
        "the accepted prefix and its waits are not replayed"
    );
}

/// Requests refused before anything is submitted.
fn refusals(world: &World, ctx: &Ctx, payloads: &Payloads) {
    world.case("continuation chain longer than a batch");
    // The list can never be cut, so it stays pending until its context is
    // destroyed: a context of its own.
    let stuck = world.channel(NVKMD_ENGINE_3D);
    let mut list = pushes(2001);
    for push in &mut list[1000..2000] {
        push.incomplete = true;
    }
    assert_eq!(stuck.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(stuck.exec(&world.vk, &list), VK_SUCCESS);
    let result = stuck.signal(&world.vk, &payloads.signals());
    outcome("signal", result);
    assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    assert_eq!(batches(), vec![], "not even the valid first batch");
    let end = nvkmd_ctx_exec {
        addr: PUSH_BASE + 4 * 2001,
        size_B: 4,
        incomplete: false,
        no_prefetch: false,
    };
    assert_eq!(stuck.exec(&world.vk, &[end]), VK_SUCCESS);
    let result = stuck.flush(&world.vk);
    outcome("flush after the chain ends", result);
    assert_eq!(
        result, VK_ERROR_FEATURE_NOT_PRESENT,
        "the list is kept, and 1000 incomplete pushes still cannot be cut"
    );
    stuck.destroy();
    assert_eq!(batches(), vec![], "destruction discards unsubmitted work");

    world.case("foreign signal");
    let foreign = ForeignSync::new(0);
    assert_eq!(ctx.exec(&world.vk, &pushes(1)), VK_SUCCESS);
    let signals = [(payloads.timeline.as_ptr(), 9), (foreign.as_ptr(), 0)];
    let result = ctx.signal(&world.vk, &signals);
    outcome("signal", result);
    assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    assert_eq!(batches(), vec![], "every signal is checked first");
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(batches(), vec![(1, PUSH_BASE, 0, 0, VK_SUCCESS)]);

    world.case("foreign wait");
    let result = ctx.wait(&world.vk, &[(foreign.as_ptr(), 0)]);
    outcome("binary foreign wait", result);
    assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    let timeline = ForeignSync::new(VK_SYNC_IS_TIMELINE);
    let result = ctx.wait(&world.vk, &[(timeline.as_ptr(), 0)]);
    outcome("foreign timeline wait for 0", result);
    assert_eq!(result, VK_SUCCESS, "satisfied before the type check");
    let result = ctx.wait(
        &world.vk,
        &[(payloads.timeline.as_ptr(), 8), (timeline.as_ptr(), 3)],
    );
    outcome("RM wait then foreign wait", result);
    assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    // The RM wait before the refused one stays pending.
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(batches(), vec![(0, 0, 1, 0, VK_SUCCESS)]);

    world.case("invalid push ranges");
    for (addr, size) in [
        (PUSH_BASE + 2, 4),
        (PUSH_BASE, 6),
        (PUSH_BASE, 0),
        (PUSH_BASE, 1 << 23),
    ] {
        let push = nvkmd_ctx_exec {
            addr,
            size_B: size,
            incomplete: false,
            no_prefetch: false,
        };
        let result = ctx.exec(&world.vk, &[push]);
        outcome(&format!("push {addr:#x}+{size:#x}"), result);
        assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    }
    world.case("exec refused after an incomplete push");
    // The call is all or nothing: had its valid prefix been appended, that
    // prefix's trailing incomplete push could never end a batch, and every
    // later flush would be refused.
    let mut list = pushes(3);
    list[0].no_prefetch = true;
    list[1].incomplete = true;
    list[2].size_B = 2;
    let result = ctx.exec(&world.vk, &list);
    outcome("exec", result);
    assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    let result = ctx.flush(&world.vk);
    outcome("flush", result);
    assert_eq!(result, VK_SUCCESS);
    assert_eq!(batches(), vec![], "nothing of the refused call is pending");
    list[2].size_B = 4;
    assert_eq!(ctx.exec(&world.vk, &list), VK_SUCCESS);
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(batches(), vec![(3, PUSH_BASE, 0, 0, VK_SUCCESS)]);
    let flags: Vec<u32> = state().ctx.batches[0]
        .pushes
        .iter()
        .map(|push| push.flags)
        .collect();
    assert_eq!(
        flags,
        [nv_rm_abi::NV_RM_EXEC_NO_PREFETCH, NV_RM_EXEC_INCOMPLETE, 0]
    );
}

/// Work before a wait is flushed first; consecutive waits share one batch.
fn ordering(world: &World, ctx: &Ctx, payloads: &Payloads) {
    world.case("wait after pushes");
    assert_eq!(ctx.exec(&world.vk, &pushes(3)), VK_SUCCESS);
    assert_eq!(ctx.wait(&world.vk, &payloads.wait()), VK_SUCCESS);
    assert_eq!(
        ctx.wait(&world.vk, &[(payloads.binary.as_ptr(), 0)]),
        VK_SUCCESS
    );
    assert_eq!(ctx.exec(&world.vk, &pushes(2)), VK_SUCCESS);
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    assert_eq!(
        batches(),
        vec![
            (3, PUSH_BASE, 0, 0, VK_SUCCESS),
            (2, PUSH_BASE, 2, 2, VK_SUCCESS)
        ]
    );
    let waits = state().ctx.batches[1].waits.clone();
    note(format!(
        "fused waits: {:?}",
        waits
            .iter()
            .map(|w| (w.sync.get(), w.value))
            .collect::<Vec<_>>()
    ));
}

/// `sync` flushes and waits for the last accepted batch on the caller.
fn syncs(world: &World, ctx: &Ctx, payloads: &Payloads) {
    world.case("sync");
    assert_eq!(ctx.exec(&world.vk, &pushes(1)), VK_SUCCESS);
    assert_eq!(ctx.sync(&world.vk), VK_SUCCESS);
    let context = state().ctx.batches[0].context;
    let waits = state().ctx.waits.clone();
    assert_eq!(waits, vec![(context, FIRST_TOKEN, u64::MAX)]);
    world.case("sync fails");
    state().ctx.wait_status = VK_ERROR_DEVICE_LOST;
    assert_eq!(ctx.signal(&world.vk, &payloads.signals()), VK_SUCCESS);
    let result = ctx.sync(&world.vk);
    outcome("sync", result);
    assert_eq!(result, VK_ERROR_DEVICE_LOST);
    assert!(state().diagnostic.contains("wait for context completion"));
    state().ctx.wait_status = VK_SUCCESS;
}

/// A bind-only context: CPU waits and signals through Mesa's `vk_sync`
/// wrappers, VA binds through Mesa's `nvkmd_va` wrappers, no RM context.
fn bind_context(world: &World, payloads: &Payloads) {
    world.case("bind context");
    let ctx = world
        .dev
        .create_ctx(Some(&world.vk), NVKMD_ENGINE_BIND)
        .expect("a bind context");
    let result = ctx.wait(&world.vk, &payloads.wait());
    outcome("CPU wait", result);
    assert_eq!(result, VK_SUCCESS);
    assert_eq!(state().sync.waits.len(), 1, "one CPU wait on the payload");
    let result = ctx.signal(&world.vk, &payloads.signals()[..1]);
    outcome("CPU signal", result);
    assert_eq!(result, VK_SUCCESS);
    assert_eq!(state().sync.signals, 1);
    assert_eq!(state().sync.last_signal, (FIRST_SYNC, 9));
    state().sync.signal_status = VK_ERROR_DEVICE_LOST;
    let result = ctx.signal(&world.vk, &payloads.signals());
    outcome("CPU signal fails", result);
    assert_eq!(result, VK_ERROR_DEVICE_LOST);
    assert_eq!(state().sync.signals, 2, "a failed signal stops the list");
    state().sync.signal_status = VK_SUCCESS;

    let log = Some(&world.vk);
    let mem = world
        .dev
        .alloc_mem(log, 64 * KIB, 4 * KIB, NVKMD_MEM_GART)
        .expect("the allocation is served");
    let va = world
        .dev
        .alloc_va(log, 0, 0x06, 64 * KIB, 0, 0)
        .expect("a reservation");
    let va_ptr = core::ptr::from_ref(va.get()).cast_mut();
    let mem_ptr = core::ptr::from_ref(mem.get()).cast_mut();
    let bind = |op, offset| nvkmd_ctx_bind {
        op,
        va: va_ptr,
        va_offset_B: offset,
        mem: mem_ptr,
        mem_offset_B: offset,
        range_B: 16 * KIB,
    };
    let binds = [
        bind(NVKMD_BIND_OP_BIND, 0),
        bind(NVKMD_BIND_OP_BIND, 16 * KIB),
    ];
    // The allocation's own VA was bound at creation.
    let before = state().mem.binds;
    let result = ctx.bind(&world.vk, &binds);
    outcome("binds", result);
    assert_eq!(result, VK_SUCCESS);
    assert_eq!(state().mem.binds, before + 2);
    state().mem.bind_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    let more = [
        bind(NVKMD_BIND_OP_BIND, 32 * KIB),
        bind(NVKMD_BIND_OP_BIND, 48 * KIB),
    ];
    let result = ctx.bind(&world.vk, &more);
    outcome("bind fails", result);
    assert_eq!(result, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    assert_eq!(
        state().mem.binds,
        before + 3,
        "a failed bind stops the list"
    );
    state().mem.bind_status = VK_SUCCESS;
    let result = ctx.bind(&world.vk, &[bind(NVKMD_BIND_OP_UNBIND, 0)]);
    outcome("unbind", result);
    assert_eq!(result, VK_SUCCESS);
    assert_eq!(state().mem.unbinds, 1);
    assert_eq!(ctx.flush(&world.vk), VK_SUCCESS);
    assert_eq!(ctx.sync(&world.vk), VK_SUCCESS);
    ctx.destroy();
    assert_eq!(state().ctx.creates, 0, "no RM context for a bind context");
    assert_eq!(state().ctx.destroys, 0);
    let state = state();
    assert!(state.ctx.batches.is_empty() && state.ctx.waits.is_empty());
    drop(state);
    va.free();
    mem.unref();
}
