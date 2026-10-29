// SPDX-License-Identifier: MIT
//! The memory and VA scenarios: allocation, import, export, CPU maps, VA
//! reservations from the window's two heaps, binds and unbinds (`memory-api`),
//! and the cleanup paths whose failures cannot be returned to Vulkan and
//! publish the device's loss instead (`memory-cleanup`).
use nvkmd_sys::{
    NVKMD_MEM_CAN_MAP, NVKMD_MEM_COHERENT, NVKMD_MEM_GART, NVKMD_MEM_LOCAL, NVKMD_MEM_MAP_FIXED,
    NVKMD_MEM_POOL_BLOCK, NVKMD_MEM_SHARED, NVKMD_MEM_VRAM, NVKMD_VA_ALLOC_FIXED, NVKMD_VA_REPLAY,
    NVKMD_VA_SPARSE, VK_ERROR_DEVICE_LOST, VK_ERROR_FEATURE_NOT_PRESENT,
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_INVALID_EXTERNAL_HANDLE,
    VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS, VK_ERROR_MEMORY_MAP_FAILED,
    VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_TOO_MANY_OBJECTS,
    VK_ERROR_UNKNOWN, VK_SUCCESS, VkResult,
};

use crate::fake::{self, EXPORTED_FD, IMPORT_SIZE, MAP_ADDRESS, live_allocations, note, state};
use crate::mesa::{self, Dev, Mem, Pdev, Va, VkDevice};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
/// Start of the caller's VA window and of its capture/replay half
/// (`nvkmd_nouveau_dev.c:51-58` splits its window the same way).
const WINDOW_START: u64 = 1 << 34;
const REPLAY_START: u64 = 1 << 39;
const WINDOW_END: u64 = 1 << 40;

/// A physical device, a logical stand-in device and the KMD device made for
/// it, whose loss hooks point at the fake RM.
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
        state().loss_device = vk.as_ptr().expose_provenance();
        Self { pdev, vk, dev }
    }

    /// The logical device, the log object of every call.
    fn log(&self) -> &VkDevice {
        &self.vk
    }

    /// Starts a case: the fake RM's knobs and counters and the loss state of
    /// both sides back to their defaults.
    fn case(&self, name: &str) {
        fake::reset_case();
        self.vk.clear_lost();
        note(format!("-- {name}"));
    }

    /// The device was lost once through `operation`: Mesa counted it and the
    /// loss reached the RM.
    fn expect_lost(&self, operation: &str) {
        assert!(self.vk.is_lost(), "{operation}: Mesa counts the loss");
        let state = state();
        assert!(
            state.rm_lost && state.loss_calls == 1,
            "{operation}: the loss reaches the RM once"
        );
        assert!(
            state.diagnostic.contains(operation),
            "{operation}: the loss names the step ({})",
            state.diagnostic
        );
    }

    fn expect_not_lost(&self, case: &str) {
        assert!(!self.vk.is_lost(), "{case}: the device is not lost");
        assert_eq!(state().loss_calls, 0, "{case}: no loss is published");
    }

    fn destroy(self) {
        self.dev.destroy();
        state().loss_device = 0;
        drop(self.vk);
        self.pdev.destroy();
        assert_eq!(live_allocations(), 0, "devices free what they allocated");
        let state = state();
        assert_eq!(
            state.opens, state.closes,
            "every render node open is closed"
        );
        assert_eq!(state.rm_opens, state.rm_closes, "every RM device is closed");
    }
}

fn describe_mem(mem: &Mem) -> String {
    let base = mem.get();
    let va = mem.va();
    let va = va.get();
    format!(
        "mem flags={:#x} size={:#x} bind_align={} va=[{:#x}, +{:#x}) kind={:#x} va_flags={:#x}",
        base.flags, base.size_B, base.bind_align_B, va.addr, va.size_B, va.pte_kind, va.flags
    )
}

fn describe_va(va: &Va) -> String {
    let va = va.get();
    format!(
        "va [{:#x}, +{:#x}) kind={:#x} flags={:#x}",
        va.addr, va.size_B, va.pte_kind, va.flags
    )
}

fn outcome<T>(case: &str, result: &Result<T, VkResult>, describe: impl Fn(&T) -> String) {
    match result {
        Ok(value) => note(format!("{case}: {}", describe(value))),
        Err(result) => note(format!("{case}: {result}")),
    }
}

/// Allocations, imports, exports, maps, VA reservations and binds through
/// Mesa's interface, with the RM calls they make and the objects Mesa sees.
pub(crate) fn api() {
    let world = World::new();
    allocations(&world);
    tiled(&world);
    imports_and_exports(&world);
    maps(&world);
    reservations(&world);
    binds(&world);
    world.destroy();
}

fn allocations(world: &World) {
    let log = Some(world.log());
    let cases: [(&str, u64, u64, u32); 5] = [
        (
            "system memory",
            4 * KIB,
            4 * KIB,
            NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP,
        ),
        (
            "mappable device memory",
            4 * KIB,
            4 * KIB,
            NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM | NVKMD_MEM_CAN_MAP,
        ),
        (
            "private device memory",
            100 * KIB,
            4 * KIB,
            NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM,
        ),
        (
            "shared device memory",
            64 * KIB,
            2 * MIB,
            NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM | NVKMD_MEM_SHARED,
        ),
        ("pool block", 2 * MIB, 4 * KIB, NVKMD_MEM_POOL_BLOCK),
    ];
    for (case, size, alignment, flags) in cases {
        world.case(case);
        let mem = world.dev.alloc_mem(log, size, alignment, flags);
        outcome(case, &mem, describe_mem);
        let mem = mem.expect("the allocation is served");
        assert_ne!(
            mem.get().flags & NVKMD_MEM_COHERENT,
            0,
            "{case}: every allocation is coherent"
        );
        assert!(
            mem.va().get().addr >= WINDOW_START && mem.va().get().addr < REPLAY_START,
            "{case}: an allocation's VA comes from the ordinary heap"
        );
        mem.unref();
        world.expect_not_lost(case);
    }

    world.case("pool block outside its contract");
    let refused = world
        .dev
        .alloc_mem(log, MIB, 4 * KIB, NVKMD_MEM_POOL_BLOCK)
        .map(Mem::unref);
    note(format!("pool block of 1 MiB: {refused:?}"));
    assert_eq!(refused, Err(VK_ERROR_UNKNOWN));
    assert!(
        state().diagnostic.contains("pool block"),
        "the refusal is named"
    );
    assert_eq!(state().mem.allocs, 0, "nothing is asked of the RM");

    world.case("RM allocation fails");
    state().mem.alloc_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    let failed = world
        .dev
        .alloc_mem(log, 64 * KIB, 4 * KIB, NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM)
        .map(Mem::unref);
    note(format!("RM allocation fails: {failed:?}"));
    assert_eq!(failed, Err(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    assert_eq!(state().mem.binds, 0, "no VA for a failed allocation");
    world.expect_not_lost("RM allocation fails");
}

fn tiled(world: &World) {
    let log = Some(world.log());
    let flags = NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM;
    let cases: [(&str, u8, u16, Option<VkResult>); 4] = [
        ("no modifier", 0, 0, None),
        ("generic kind, block height 2", 0x06, 0x10, None),
        (
            "block height past the 2D modifier",
            0x06,
            0x60,
            Some(VK_ERROR_FEATURE_NOT_PRESENT),
        ),
        (
            "kind other than the generic one",
            0xdb,
            0x10,
            Some(VK_ERROR_FEATURE_NOT_PRESENT),
        ),
    ];
    for (case, kind, tile, refused) in cases {
        world.case(case);
        let mem = world
            .dev
            .alloc_tiled_mem(log, 64 * KIB, 4 * KIB, kind, tile, flags);
        outcome(case, &mem, describe_mem);
        match (mem, refused) {
            (Ok(mem), None) => {
                assert_eq!(
                    mem.va().get().pte_kind,
                    kind,
                    "{case}: the VA carries the kind"
                );
                mem.unref();
            }
            (Err(result), Some(expected)) => {
                assert_eq!(result, expected, "{case}");
                assert_eq!(state().mem.allocs, 0, "{case}: nothing is asked of the RM");
            }
            (mem, expected) => panic!(
                "{case}: {:?}, expected {expected:?}",
                mem.map(|mem| describe_mem(&mem))
            ),
        }
    }
}

fn imports_and_exports(world: &World) {
    let log = Some(world.log());
    world.case("import");
    let mem = world.dev.import_dma_buf(log, 22);
    outcome("import", &mem, describe_mem);
    let mem = mem.expect("the import is served");
    assert_eq!(mem.get().size_B, IMPORT_SIZE, "the RM's size");
    let exported = mem.export(log);
    note(format!("export: {exported:?}"));
    assert_eq!(exported, Ok(EXPORTED_FD));
    state().mem.export_status = VK_ERROR_TOO_MANY_OBJECTS;
    let exported = mem.export(log);
    note(format!("export fails: {exported:?}"));
    assert_eq!(exported, Err(VK_ERROR_TOO_MANY_OBJECTS));
    mem.unref();

    world.case("import fails");
    state().mem.import_status = VK_ERROR_INVALID_EXTERNAL_HANDLE;
    let failed = world.dev.import_dma_buf(log, 22).map(Mem::unref);
    note(format!("import fails: {failed:?}"));
    assert_eq!(failed, Err(VK_ERROR_INVALID_EXTERNAL_HANDLE));
    world.expect_not_lost("import fails");
}

fn maps(world: &World) {
    let log = Some(world.log());
    world.case("maps");
    let mem = world
        .dev
        .alloc_mem(log, 4 * KIB, 4 * KIB, NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP)
        .expect("the allocation is served");
    let first = mem.map(log).expect("the memory maps");
    let second = mem.map(log).expect("the memory maps again");
    note(format!(
        "map, map: {:#x}, {:#x}",
        first.addr(),
        second.addr()
    ));
    assert_eq!(first.addr(), MAP_ADDRESS, "the RM's address");
    assert_eq!(state().mem.maps, 1, "Mesa counts internal maps; one RM map");
    mem.unmap();
    mem.unmap();
    assert_eq!(state().mem.unmaps, 1, "the last unmap reaches the RM");
    let overmap = mem.overmap(log);
    note(format!("overmap: {overmap}"));
    assert_eq!(overmap, VK_ERROR_FEATURE_NOT_PRESENT);
    let handle = mem.log_handle();
    note(format!("log handle: {handle}"));
    assert_eq!(
        u64::from(handle),
        state().mem.last_allocation,
        "the RM handle"
    );
    mem.unref();
    world.expect_not_lost("maps");
}

fn reservations(world: &World) {
    let log = Some(world.log());
    world.case("reservations");
    let refusals: [(&str, u32, u64, u64, VkResult); 4] = [
        (
            "sparse",
            NVKMD_VA_SPARSE,
            4 * KIB,
            0,
            VK_ERROR_FEATURE_NOT_PRESENT,
        ),
        ("empty", 0, 0, 0, VK_ERROR_OUT_OF_DEVICE_MEMORY),
        (
            "larger than the window",
            0,
            WINDOW_END - WINDOW_START - 4094,
            0,
            VK_ERROR_OUT_OF_DEVICE_MEMORY,
        ),
        (
            "alignment not a power of two",
            0,
            4 * KIB,
            12 * KIB,
            VK_ERROR_FEATURE_NOT_PRESENT,
        ),
    ];
    for (case, flags, size, alignment, expected) in refusals {
        let result = world
            .dev
            .alloc_va(log, flags, 0, size, alignment, 0)
            .map(Va::free);
        note(format!("{case}: {result:?}"));
        assert_eq!(result, Err(expected), "{case}");
    }

    let plain = world
        .dev
        .alloc_va(log, 0, 0, 5 * KIB, 64 * KIB, 0)
        .expect("an ordinary reservation");
    let capture = world
        .dev
        .alloc_va(log, NVKMD_VA_REPLAY, 0x06, 4 * KIB, 0, 0)
        .expect("a capture reservation");
    note(format!("ordinary: {}", describe_va(&plain)));
    note(format!("capture: {}", describe_va(&capture)));
    assert!(
        plain.get().addr >= WINDOW_START && plain.get().addr < REPLAY_START,
        "ordinary reservations stay below the replay range"
    );
    assert_eq!(plain.get().size_B, 8 * KIB, "sizes are page granular");
    assert!(
        capture.get().addr >= REPLAY_START && capture.get().addr < WINDOW_END,
        "capture reservations come from the replay range"
    );

    fixed_reservations(world, &plain, &capture);
    plain.free();
    capture.free();
    world.expect_not_lost("reservations");
}

/// Fixed addresses: replayed capture addresses only, inside the replay range,
/// aligned and free.
fn fixed_reservations(world: &World, plain: &Va, capture: &Va) {
    let log = Some(world.log());
    let replay = NVKMD_VA_REPLAY | NVKMD_VA_ALLOC_FIXED;
    let fixed_cases: [(&str, u32, u64, Option<VkResult>); 5] = [
        (
            "fixed address without replay",
            NVKMD_VA_ALLOC_FIXED,
            capture.get().addr + 64 * KIB,
            Some(VK_ERROR_FEATURE_NOT_PRESENT),
        ),
        (
            "replay of a taken address",
            replay,
            capture.get().addr,
            Some(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS),
        ),
        (
            "replay of an ordinary address",
            replay,
            plain.get().addr,
            Some(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS),
        ),
        (
            "replay of an unaligned address",
            replay,
            REPLAY_START + 4 * KIB,
            Some(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS),
        ),
        ("replay of a free address", replay, REPLAY_START, None),
    ];
    for (case, flags, address, refused) in fixed_cases {
        let result = world
            .dev
            .alloc_va(log, flags, 0, 64 * KIB, 64 * KIB, address);
        outcome(case, &result, describe_va);
        match (result, refused) {
            (Ok(va), None) => {
                assert_eq!(va.get().addr, address, "{case}: at the replayed address");
                va.free();
            }
            (Err(result), Some(expected)) => assert_eq!(result, expected, "{case}"),
            (result, expected) => panic!(
                "{case}: {:?}, expected {expected:?}",
                result.map(|va| describe_va(&va))
            ),
        }
    }
}

fn binds(world: &World) {
    let log = Some(world.log());
    world.case("binds");
    let mem = world
        .dev
        .alloc_mem(log, 64 * KIB, 4 * KIB, NVKMD_MEM_GART)
        .expect("the allocation is served");
    let va = world
        .dev
        .alloc_va(log, 0, 0x06, 64 * KIB, 0, 0)
        .expect("a reservation");
    for offset in [32 * KIB, 0, 16 * KIB] {
        assert_eq!(va.bind(log, offset, &mem, offset, 16 * KIB), VK_SUCCESS);
    }
    let refusals: [(&str, VkResult); 4] = [
        (
            "bind past the memory",
            va.bind(log, 48 * KIB, &mem, 56 * KIB, 16 * KIB),
        ),
        ("partial unbind", va.unbind(log, 8 * KIB, 16 * KIB)),
        ("empty unbind", va.unbind(log, 0, 0)),
        (
            "unbind past the reservation",
            va.unbind(log, 48 * KIB, 32 * KIB),
        ),
    ];
    for (case, result) in refusals {
        note(format!("{case}: {result}"));
        assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT, "{case}");
    }
    let result = va.unbind(log, 0, 48 * KIB);
    note(format!("unbind of three bindings: {result}"));
    assert_eq!(result, VK_SUCCESS);
    assert_eq!(state().mem.unbinds, 3, "each covered binding is unmapped");

    state().mem.bind_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    let result = va.bind(log, 0, &mem, 0, 16 * KIB);
    note(format!("RM bind fails: {result}"));
    assert_eq!(result, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    state().mem.bind_status = VK_SUCCESS;
    for offset in [0, 16 * KIB] {
        assert_eq!(va.bind(log, offset, &mem, 0, 16 * KIB), VK_SUCCESS);
    }
    state().mem.unbind_status = VK_ERROR_INITIALIZATION_FAILED;
    let result = va.unbind(log, 0, 32 * KIB);
    note(format!("RM unbind fails: {result}"));
    assert_eq!(result, VK_ERROR_INITIALIZATION_FAILED);
    state().mem.unbind_status = VK_SUCCESS;
    va.free();
    let freed = state().mem.last_va_free.0.clone();
    note(format!("VA free hands over {freed:?}"));
    assert_eq!(freed.len(), 2, "a failed unbind keeps its binding");
    mem.unref();
    world.expect_not_lost("binds");
}

/// The cleanup paths: failures of void callbacks and of rollbacks publish the
/// device's loss at once (through Mesa for a logical device, to the RM for
/// the probe's device); map failures stay recoverable unless the device is
/// lost; a failed VA free keeps its range reserved; a deferred release keeps
/// the range until the RM returns it.
pub(crate) fn cleanup() {
    let world = World::new();
    map_failures(&world);
    free_failures(&world);
    rollbacks(&world);
    va_frees(&world);
    world.destroy();
    unmap_failure_without_logical_device();
}

fn mappable(world: &World) -> Mem {
    world
        .dev
        .alloc_mem(
            Some(world.log()),
            4 * KIB,
            4 * KIB,
            NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP,
        )
        .expect("the allocation is served")
}

fn map_failures(world: &World) {
    let log = Some(world.log());
    world.case("map failures");
    let mem = mappable(world);
    for status in [
        VK_ERROR_OUT_OF_HOST_MEMORY,
        VK_ERROR_OUT_OF_DEVICE_MEMORY,
        VK_ERROR_INITIALIZATION_FAILED,
    ] {
        fake::reset_case();
        state().mem.map_status = status;
        let (result, map) = mem.ops_map(log, 0, core::ptr::null_mut());
        note(format!("RM map fails {status}: {result}"));
        let expected = if status == VK_ERROR_INITIALIZATION_FAILED {
            VK_ERROR_MEMORY_MAP_FAILED
        } else {
            status
        };
        assert_eq!(result, expected, "out-of-memory passes through");
        assert!(map.is_null(), "a failed map returns no address");
        assert_eq!(state().mem.unmaps, 0);
        assert!(state().diagnostic.contains("CPU map failed"));
        world.expect_not_lost("recoverable map failure");
    }

    fake::reset_case();
    let (result, map) = mem.ops_map(log, NVKMD_MEM_MAP_FIXED, core::ptr::null_mut());
    note(format!("fixed map: {result}"));
    assert_eq!(result, VK_ERROR_FEATURE_NOT_PRESENT);
    assert!(map.is_null());
    assert_eq!(state().mem.maps, 0, "nothing is asked of the RM");

    // vkMapMemory has no VK_ERROR_DEVICE_LOST: the loss is published and the
    // map fails.
    for (case, status, publishes) in [
        ("RM map reports the loss", VK_ERROR_DEVICE_LOST, false),
        (
            "RM map lost the device",
            VK_ERROR_INITIALIZATION_FAILED,
            true,
        ),
    ] {
        world.case(case);
        state().mem.map_status = status;
        state().mem.map_publishes_loss = publishes;
        let (result, map) = mem.ops_map(log, 0, core::ptr::null_mut());
        note(format!("{case}: {result}"));
        assert_eq!(result, VK_ERROR_MEMORY_MAP_FAILED);
        assert!(map.is_null());
        world.expect_lost("CPU map");
    }

    world.case("RM map returns no address");
    state().mem.map_address = 0;
    let (result, map) = mem.ops_map(log, 0, core::ptr::null_mut());
    note(format!("null map: {result}"));
    assert_eq!(result, VK_ERROR_MEMORY_MAP_FAILED);
    assert!(map.is_null());
    assert_eq!(state().mem.unmaps, 1, "the map reference is dropped");
    assert!(state().diagnostic.contains("null address"));
    world.expect_not_lost("null map");

    world.case("null map rollback fails");
    state().mem.map_address = 0;
    state().mem.unmap_status = VK_ERROR_INITIALIZATION_FAILED;
    let (result, _) = mem.ops_map(log, 0, core::ptr::null_mut());
    note(format!("null map, failed rollback: {result}"));
    assert_eq!(result, VK_ERROR_MEMORY_MAP_FAILED);
    world.expect_lost("null CPU map rollback");

    // A void callback has already dropped its caller's reference: even an
    // out-of-memory status of the RM is irreversible here.
    world.case("unmap fails");
    state().mem.unmap_status = VK_ERROR_OUT_OF_HOST_MEMORY;
    mem.ops_unmap(core::ptr::without_provenance_mut(MAP_ADDRESS));
    world.expect_lost("CPU unmap");

    world.case("map failures done");
    mem.unref();
}

fn free_failures(world: &World) {
    for (case, status) in [
        ("free", VK_SUCCESS),
        ("free fails", VK_ERROR_INITIALIZATION_FAILED),
    ] {
        world.case(case);
        let mem = mappable(world);
        state().mem.free_status = status;
        mem.unref();
        assert_eq!(state().mem.frees, 1);
        if status == VK_SUCCESS {
            world.expect_not_lost(case);
        } else {
            world.expect_lost("memory free");
        }
    }
}

fn rollbacks(world: &World) {
    let log = Some(world.log());
    for imported in [false, true] {
        for (cleanup, status) in [
            ("rollback", VK_SUCCESS),
            ("rollback fails", VK_ERROR_INITIALIZATION_FAILED),
        ] {
            let case = format!(
                "{} whose VA bookkeeping fails, {cleanup}",
                if imported { "import" } else { "allocation" }
            );
            world.case(&case);
            state().mem.free_status = status;
            // The memory object is the first allocation, its VA the second.
            fake::fail_calloc_after(1);
            let result = if imported {
                world.dev.import_dma_buf(log, 22)
            } else {
                world
                    .dev
                    .alloc_mem(log, 4 * KIB, 4 * KIB, NVKMD_MEM_CAN_MAP)
            }
            .map(Mem::unref);
            fake::stop_failing_calloc();
            note(format!("{case}: {result:?}"));
            // vkAllocateMemory has no VK_ERROR_DEVICE_LOST: a failed rollback
            // publishes the loss and the allocation keeps its own failure.
            assert_eq!(result, Err(VK_ERROR_OUT_OF_HOST_MEMORY));
            assert_eq!(state().mem.frees, 1, "{case}: the RM allocation is freed");
            if status == VK_SUCCESS {
                world.expect_not_lost(&case);
            } else {
                world.expect_lost(if imported {
                    "import rollback free"
                } else {
                    "allocation rollback free"
                });
            }
        }
    }
}

fn va_frees(world: &World) {
    let log = Some(world.log());
    let replay = NVKMD_VA_REPLAY | NVKMD_VA_ALLOC_FIXED;
    world.case("VA frees");
    let mem = mappable(world);

    // VA teardown hands every mapping to one RM VA free and never unbinds
    // itself. A failed free keeps the range reserved (a replay of it fails)
    // and publishes the loss.
    for (case, status) in [
        ("VA free", VK_SUCCESS),
        ("VA free fails", VK_ERROR_INITIALIZATION_FAILED),
    ] {
        world.case(case);
        state().mem.va_free_status = status;
        let va = world
            .dev
            .alloc_va(log, NVKMD_VA_REPLAY, 0, 4 * KIB, 4 * KIB, 0)
            .expect("a capture reservation");
        let reserved = va.get().addr;
        assert_eq!(va.bind(log, 0, &mem, 0, 4 * KIB), VK_SUCCESS);
        va.free();
        {
            let state = state();
            assert_eq!(
                (state.mem.binds, state.mem.unbinds, state.mem.va_frees),
                (1, 0, 1)
            );
            assert_eq!(state.mem.last_va_free.0.len(), 1);
            assert_eq!(
                (state.mem.last_va_free.1, state.mem.last_va_free.2),
                (reserved, 4 * KIB)
            );
        }
        if status == VK_SUCCESS {
            world.expect_not_lost(case);
        } else {
            world.expect_lost("VA free; reservation retained");
        }
        state().mem.va_free_status = VK_SUCCESS;
        let again = world
            .dev
            .alloc_va(log, replay, 0, 4 * KIB, 4 * KIB, reserved);
        outcome(&format!("{case}: replay of the range"), &again, describe_va);
        let expected = if status == VK_SUCCESS {
            Ok(())
        } else {
            Err(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
        };
        assert_eq!(again.map(Va::free), expected, "{case}");
    }

    // A deferred release keeps the range until the RM returns it.
    world.case("deferred release");
    state().mem.defer_release = true;
    let va = world
        .dev
        .alloc_va(log, NVKMD_VA_REPLAY, 0, 4 * KIB, 4 * KIB, 0)
        .expect("a capture reservation");
    let reserved = va.get().addr;
    assert_eq!(va.bind(log, 0, &mem, 0, 4 * KIB), VK_SUCCESS);
    va.free();
    assert!(fake::release_pending(), "the release waits for retirement");
    let early = world
        .dev
        .alloc_va(log, replay, 0, 4 * KIB, 4 * KIB, reserved)
        .map(Va::free);
    note(format!("replay before the release: {early:?}"));
    assert_eq!(early, Err(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS));
    fake::run_deferred_release();
    let again = world
        .dev
        .alloc_va(log, replay, 0, 4 * KIB, 4 * KIB, reserved)
        .expect("the released range can be replayed");
    // Without bindings the release runs at once.
    again.free();
    {
        let state = state();
        assert_eq!(state.mem.va_frees, 2);
        assert_eq!(state.mem.last_va_free.0, [0_u64; 0]);
    }
    assert!(!fake::release_pending());
    world.expect_not_lost("deferred release");

    // An application-ordered rebind unbinds what the range held at once
    // before binding again.
    world.case("rebind");
    let va = world
        .dev
        .alloc_va(log, 0, 0, 4 * KIB, 4 * KIB, 0)
        .expect("a reservation");
    assert_eq!(va.bind(log, 0, &mem, 0, 4 * KIB), VK_SUCCESS);
    assert_eq!(va.bind(log, 0, &mem, 0, 4 * KIB), VK_SUCCESS);
    let counts = {
        let state = state();
        (state.mem.binds, state.mem.unbinds)
    };
    assert_eq!(counts, (2, 1), "the rebind unmapped the old binding first");
    va.free();
    assert_eq!(state().mem.last_va_free.0.len(), 1);
    world.expect_not_lost("rebind");

    world.case("VA frees done");
    mem.unref();
}

/// The physical device's probe device has no logical device: a failed void
/// cleanup still publishes the loss, to the RM.
fn unmap_failure_without_logical_device() {
    fake::reset();
    let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");
    let dev = pdev.create_dev(None).expect("the KMD device opens");
    note("-- unmap fails on the probe's device");
    let mem = dev
        .alloc_mem(None, 4 * KIB, 4 * KIB, NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP)
        .expect("the allocation is served");
    state().allow_no_vk_loss = true;
    state().mem.unmap_status = VK_ERROR_INITIALIZATION_FAILED;
    mem.ops_unmap(core::ptr::without_provenance_mut(MAP_ADDRESS));
    {
        let state = state();
        assert!(
            state.rm_lost && state.loss_calls == 1,
            "the loss reaches the RM"
        );
    }
    state().mem.unmap_status = VK_SUCCESS;
    mem.unref();
    dev.destroy();
    pdev.destroy();
    assert_eq!(live_allocations(), 0, "devices free what they allocated");
}
