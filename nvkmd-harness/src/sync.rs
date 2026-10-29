// SPDX-License-Identifier: MIT
//! The sync payload scenarios: creation, host operations, handle import and
//! export and their failures through Mesa's `vk_sync_*` wrappers
//! (`sync-api`), and CPU waits racing the device's loss, deadlines, host
//! operations on a lost device and waits that outlast any fixed age
//! (`wait-loss`).
use nvkmd_sys::{
    VK_ERROR_DEVICE_LOST, VK_ERROR_INVALID_EXTERNAL_HANDLE, VK_ERROR_OUT_OF_DEVICE_MEMORY,
    VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_TOO_MANY_OBJECTS, VK_ERROR_UNKNOWN, VK_SUCCESS,
    VK_SYNC_IS_SHAREABLE, VK_SYNC_IS_SHARED, VK_SYNC_IS_TIMELINE, VK_SYNC_WAIT_ANY,
    VK_SYNC_WAIT_COMPLETE, VK_SYNC_WAIT_PENDING, VK_TIMEOUT, VkResult, vk_sync_flags, vk_sync_type,
};

use crate::fake::{self, live_allocations, note, state};
use crate::fake_sync::{EXPORTED_SYNC_FD, FIRST_SYNC, TIMELINE_VALUE, WaitScript};
use crate::mesa::{self, Dev, Pdev, Sync, VkDevice};

/// A descriptor the suites import (never opened: only the fake RM sees it).
const IMPORTED_FD: i32 = 66;
/// Mesa's clock at the start of a scripted wait.
const NOW: u64 = 1_000_000_000;
const MS: u64 = 1_000_000;

/// A physical device, a stand-in NVK device and its KMD device, attached as
/// NVK attaches them, with the loss hooks pointing at the fake RM.
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

    /// The binary/timeline type with opaque fds.
    fn opaque_type(&self) -> &vk_sync_type {
        self.pdev.sync_types()[0]
    }

    /// The binary type with sync files.
    fn file_type(&self) -> &vk_sync_type {
        self.pdev.sync_types()[1]
    }

    fn sync(&self, sync_type: &vk_sync_type, flags: vk_sync_flags, initial: u64) -> Sync {
        Sync::init(&self.vk, sync_type, flags, initial).expect("the fake RM creates payloads")
    }

    /// Starts a case: the fake RM's knobs and counters and the loss state of
    /// both sides back to their defaults.
    fn case(&self, name: &str) {
        fake::reset_case();
        self.vk.clear_lost();
        note(format!("-- {name}"));
    }

    /// The device was lost once and the loss reached the RM; the diagnostic
    /// names `what`.
    fn expect_lost(&self, what: &str) {
        assert!(self.vk.is_lost(), "{what}: Mesa counts the loss");
        let state = state();
        assert!(
            state.rm_lost && state.loss_calls == 1,
            "{what}: the loss reaches the RM once"
        );
        assert!(
            state.diagnostic.contains(what),
            "{what}: the loss is named ({})",
            state.diagnostic
        );
    }

    fn expect_not_lost(&self, case: &str) {
        assert!(!self.vk.is_lost(), "{case}: the device is not lost");
        assert_eq!(state().loss_calls, 0, "{case}: no loss is published");
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

fn result<T: std::fmt::Debug>(case: &str, value: &T) {
    note(format!("{case}: {value:?}"));
}

/// The diagnostic of the case names `operation`.
fn named(operation: &str) {
    let diagnostic = state().diagnostic.clone();
    assert!(
        diagnostic.contains(operation),
        "the failure names {operation} ({diagnostic})"
    );
}

/// Payload creation, host operations, handles and their failures.
pub(crate) fn api() {
    let world = World::new();
    let syncs = creation(&world);
    host_operations(&world, &syncs);
    handles(&world, &syncs);
    failures(&world, &syncs);
    finish(&world, &syncs.binary);
    let Syncs {
        binary,
        other,
        timeline,
        file,
    } = syncs;
    world.case("finish");
    for sync in [binary, other, timeline, file] {
        sync.finish(&world.vk);
    }
    assert_eq!(state().sync.destroys, 4, "every payload is dropped once");
    world.expect_not_lost("finish");
    world.destroy();
}

struct Syncs {
    binary: Sync,
    other: Sync,
    timeline: Sync,
    file: Sync,
}

fn creation(world: &World) -> Syncs {
    world.case("create");
    let binary = world.sync(world.opaque_type(), VK_SYNC_IS_SHAREABLE, 0);
    let other = world.sync(world.opaque_type(), 0, 0);
    let timeline = world.sync(world.opaque_type(), VK_SYNC_IS_TIMELINE, 5);
    let file = world.sync(world.file_type(), 0, 0);
    assert_eq!(state().sync.creates, 4, "one payload per sync");

    world.case("create fails");
    state().sync.create_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    let failed = Sync::init(&world.vk, world.opaque_type(), 0, 0).map(|_| ());
    result("vk_sync_init", &failed);
    assert_eq!(failed, Err(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    named("create sync payload");
    world.expect_not_lost("create fails");
    Syncs {
        binary,
        other,
        timeline,
        file,
    }
}

fn host_operations(world: &World, syncs: &Syncs) {
    let vk = &world.vk;
    world.case("host operations");
    assert_eq!(syncs.binary.reset(vk), VK_SUCCESS);
    assert_eq!(syncs.binary.signal(vk, 0), VK_SUCCESS);
    assert_eq!(syncs.timeline.signal(vk, 9), VK_SUCCESS);
    assert_eq!(
        state().sync.last_signal,
        (FIRST_SYNC + 2, 9),
        "a timeline signal carries its value"
    );
    let value = syncs.timeline.get_value(vk);
    result("vk_sync_get_value", &value);
    assert_eq!(value, Ok(TIMELINE_VALUE));
    assert_eq!(syncs.binary.move_from(vk, &syncs.other), VK_SUCCESS);
    assert_eq!(state().sync.last_shared, 0, "a private source");
    syncs
        .other
        .set_flags(VK_SYNC_IS_SHAREABLE | VK_SYNC_IS_SHARED);
    assert_eq!(syncs.binary.move_from(vk, &syncs.other), VK_SUCCESS);
    assert_eq!(state().sync.last_shared, 1, "a shared source");
    syncs.other.set_flags(0);
    for done in [1, 0] {
        state().sync.reserve_done = done;
        let reserved = syncs.binary.reserve_from(vk, &syncs.other);
        result("reserve_signal", &reserved);
        assert_eq!(reserved, (VK_SUCCESS, done != 0));
    }
    world.expect_not_lost("host operations");
}

fn handles(world: &World, syncs: &Syncs) {
    let vk = &world.vk;
    world.case("sync file of -1");
    assert_eq!(syncs.file.import_sync_file(vk, -1), VK_SUCCESS);
    {
        let state = state();
        assert_eq!(
            (state.sync.signals, state.sync.imports),
            (1, 0),
            "Mesa signals instead of importing -1"
        );
    }
    world.case("sync file");
    assert_eq!(syncs.file.import_sync_file(vk, IMPORTED_FD), VK_SUCCESS);
    assert_eq!(state().sync.last_opaque, 0, "a sync file import");
    let fd = syncs.file.export_sync_file(vk);
    result("vk_sync_export_sync_file", &fd);
    assert_eq!(fd, Ok(EXPORTED_SYNC_FD));
    assert_eq!(state().sync.last_opaque, 0, "a sync file export");

    world.case("opaque fd");
    assert_eq!(syncs.binary.import_opaque_fd(vk, IMPORTED_FD), VK_SUCCESS);
    assert_eq!(state().sync.last_opaque, 1, "an opaque import");
    assert_eq!(
        syncs.binary.flags() & VK_SYNC_IS_SHARED,
        VK_SYNC_IS_SHARED,
        "Mesa marks an imported payload shared"
    );
    let fd = syncs.binary.export_opaque_fd(vk);
    result("vk_sync_export_opaque_fd", &fd);
    assert_eq!(fd, Ok(EXPORTED_SYNC_FD));
    assert_eq!(state().sync.last_opaque, 1, "an opaque export");
    assert_eq!(state().sync.exports, 1, "only the explicit export exports");
    syncs.binary.set_flags(VK_SYNC_IS_SHAREABLE);
    world.expect_not_lost("handles");
}

/// A failure case: the operation's name, the knob that fails it, the call
/// and the status it must return.
type Case = (
    &'static str,
    fn(&mut fake::State),
    fn(&Syncs, &VkDevice) -> VkResult,
    VkResult,
);

/// Every RM failure is returned as the RM's status, named, without loss.
fn failures(world: &World, syncs: &Syncs) {
    let vk = &world.vk;
    let cases: [Case; 9] = [
        (
            "signal sync payload",
            |state| state.sync.signal_status = VK_ERROR_OUT_OF_HOST_MEMORY,
            |syncs, vk| syncs.timeline.signal(vk, 10),
            VK_ERROR_OUT_OF_HOST_MEMORY,
        ),
        (
            "query sync value",
            |state| state.sync.get_value_status = VK_ERROR_DEVICE_LOST,
            |syncs, vk| syncs.timeline.get_value(vk).err().unwrap_or(VK_SUCCESS),
            VK_ERROR_DEVICE_LOST,
        ),
        (
            "reset sync payload",
            |state| state.sync.reset_status = VK_ERROR_OUT_OF_HOST_MEMORY,
            |syncs, vk| syncs.binary.reset(vk),
            VK_ERROR_OUT_OF_HOST_MEMORY,
        ),
        (
            "move sync payload",
            |state| state.sync.move_status = VK_ERROR_OUT_OF_HOST_MEMORY,
            |syncs, vk| syncs.binary.move_from(vk, &syncs.other),
            VK_ERROR_OUT_OF_HOST_MEMORY,
        ),
        (
            "reserve binary signal",
            |state| state.sync.reserve_status = VK_ERROR_OUT_OF_HOST_MEMORY,
            |syncs, vk| {
                let (result, reserved) = syncs.binary.reserve_from(vk, &syncs.other);
                assert!(!reserved, "a failed reservation is not reserved");
                result
            },
            VK_ERROR_OUT_OF_HOST_MEMORY,
        ),
        (
            "import sync file",
            |state| state.sync.import_status = VK_ERROR_INVALID_EXTERNAL_HANDLE,
            |syncs, vk| syncs.file.import_sync_file(vk, IMPORTED_FD),
            VK_ERROR_INVALID_EXTERNAL_HANDLE,
        ),
        (
            "import opaque sync fd",
            |state| state.sync.import_status = VK_ERROR_INVALID_EXTERNAL_HANDLE,
            |syncs, vk| {
                let result = syncs.binary.import_opaque_fd(vk, IMPORTED_FD);
                assert_eq!(
                    syncs.binary.flags() & VK_SYNC_IS_SHARED,
                    0,
                    "a failed import leaves the payload private"
                );
                result
            },
            VK_ERROR_INVALID_EXTERNAL_HANDLE,
        ),
        (
            "export sync file",
            |state| state.sync.export_status = VK_ERROR_TOO_MANY_OBJECTS,
            |syncs, vk| syncs.file.export_sync_file(vk).err().unwrap_or(VK_SUCCESS),
            VK_ERROR_TOO_MANY_OBJECTS,
        ),
        (
            "export opaque sync fd",
            |state| state.sync.export_status = VK_ERROR_TOO_MANY_OBJECTS,
            |syncs, vk| {
                syncs
                    .binary
                    .export_opaque_fd(vk)
                    .err()
                    .unwrap_or(VK_SUCCESS)
            },
            VK_ERROR_TOO_MANY_OBJECTS,
        ),
    ];
    for (operation, fail, run, expected) in cases {
        world.case(&format!("{operation} fails"));
        fail(&mut state());
        let got = run(syncs, vk);
        result(operation, &got);
        assert_eq!(got, expected, "{operation}: the RM's status");
        named(operation);
        world.expect_not_lost(operation);
    }
}

/// `finish` cannot return a failure: it is the device's loss, once.
fn finish(world: &World, sync: &Sync) {
    world.case("finish fails");
    state().sync.destroy_status = VK_ERROR_UNKNOWN;
    sync.finish_again(&world.vk);
    world.expect_lost("sync disposal failed");
    state().sync.destroy_status = VK_SUCCESS;
    sync.finish_again(&world.vk);
    assert_eq!(
        state().loss_calls,
        1,
        "a later successful finish publishes nothing"
    );
}

/// The four wait modes of the C backend's tests: binary/timeline, complete or
/// pending.
#[derive(Clone, Copy, Debug)]
enum Mode {
    Binary,
    Timeline,
    TimelinePending,
    BinaryPending,
}

/// What happens during the RM's wait.
#[derive(Clone, Copy, Debug)]
enum Outcome {
    /// Mesa's loss is published while waiting; the wait fails.
    MesaLoss,
    /// The caller's finite deadline passes.
    Deadline,
    /// The points complete.
    Complete,
    /// The RM fails the wait with its own status.
    RmError,
    /// The points complete at the same instant as Mesa's loss.
    CompleteWithMesaLoss,
    /// The RM's maintenance publishes loss while waiting; the wait fails.
    RmLoss,
    /// The points complete at the same instant as the RM's loss.
    CompleteWithRmLoss,
}

impl Outcome {
    const ALL: [Self; 7] = [
        Self::MesaLoss,
        Self::Deadline,
        Self::Complete,
        Self::RmError,
        Self::CompleteWithMesaLoss,
        Self::RmLoss,
        Self::CompleteWithRmLoss,
    ];

    fn script(self) -> WaitScript {
        let completes = |status| WaitScript {
            status,
            elapse: Some(MS),
            ..WaitScript::COMPLETE
        };
        match self {
            Self::MesaLoss => WaitScript {
                status: VK_ERROR_DEVICE_LOST,
                lose_mesa: true,
                ..WaitScript::COMPLETE
            },
            Self::Deadline => WaitScript {
                status: 2,
                elapse: None,
                ..WaitScript::COMPLETE
            },
            Self::Complete => completes(VK_SUCCESS),
            Self::RmError => completes(VK_ERROR_UNKNOWN),
            Self::CompleteWithMesaLoss => WaitScript {
                lose_mesa: true,
                ..completes(VK_SUCCESS)
            },
            Self::RmLoss => WaitScript {
                status: VK_ERROR_DEVICE_LOST,
                lose_rm: true,
                ..WaitScript::COMPLETE
            },
            Self::CompleteWithRmLoss => WaitScript {
                lose_rm: true,
                ..completes(VK_SUCCESS)
            },
        }
    }

    const fn expected(self) -> VkResult {
        match self {
            Self::Deadline => VK_TIMEOUT,
            Self::Complete => VK_SUCCESS,
            Self::RmError => VK_ERROR_UNKNOWN,
            Self::MesaLoss
            | Self::CompleteWithMesaLoss
            | Self::RmLoss
            | Self::CompleteWithRmLoss => VK_ERROR_DEVICE_LOST,
        }
    }
}

/// CPU waits racing loss, deadlines and host operations on a lost device.
pub(crate) fn wait_loss() {
    let world = World::new();
    world.case("syncs");
    let binary = [
        world.sync(world.opaque_type(), 0, 0),
        world.sync(world.opaque_type(), 0, 0),
    ];
    let timeline = [
        world.sync(world.opaque_type(), VK_SYNC_IS_TIMELINE, 0),
        world.sync(world.opaque_type(), VK_SYNC_IS_TIMELINE, 0),
    ];
    matrix(&world, &binary, &timeline);
    single_waits(&world, &binary[0]);
    wait_sizes(&world, &binary[0]);
    host_operations_after_loss(&world, &binary[0], &binary[1]);
    world.case("finish");
    for sync in binary.into_iter().chain(timeline) {
        sync.finish(&world.vk);
    }
    world.destroy();
}

/// Every mode, ALL and ANY, every outcome: one RM wait with the caller's
/// points, flags and remaining time, and loss from either side returned.
fn matrix(world: &World, binary: &[Sync; 2], timeline: &[Sync; 2]) {
    for mode in [
        Mode::Binary,
        Mode::Timeline,
        Mode::TimelinePending,
        Mode::BinaryPending,
    ] {
        for any in [false, true] {
            for outcome in Outcome::ALL {
                world.case(&format!("{mode:?} any={any} {outcome:?}"));
                let (syncs, value) = match mode {
                    Mode::Binary | Mode::BinaryPending => (binary, 0),
                    Mode::Timeline | Mode::TimelinePending => (timeline, 3),
                };
                let pending = matches!(mode, Mode::TimelinePending | Mode::BinaryPending);
                let flags = (if any { VK_SYNC_WAIT_ANY } else { 0 })
                    | (if pending {
                        VK_SYNC_WAIT_PENDING
                    } else {
                        VK_SYNC_WAIT_COMPLETE
                    });
                let deadline = if matches!(outcome, Outcome::Deadline) {
                    NOW + 2 * MS
                } else {
                    u64::MAX
                };
                {
                    let mut state = state();
                    state.sync.clock = Some(NOW);
                    state.sync.wait = outcome.script();
                }
                let got = mesa::wait_many(
                    &world.vk,
                    &[(&syncs[0], value), (&syncs[1], value)],
                    flags,
                    deadline,
                );
                result("wait_many", &got);
                assert_eq!(got, outcome.expected(), "{mode:?} any={any} {outcome:?}");
                let state = state();
                let ids: Vec<u64> = state.sync.waits[0]
                    .points
                    .iter()
                    .map(|point| point.sync.get())
                    .collect();
                assert_eq!(state.sync.waits.len(), 1, "one RM wait");
                let call = &state.sync.waits[0];
                assert_eq!(call.flags, flags, "the caller's flags");
                assert_eq!(
                    call.timeout,
                    if deadline == u64::MAX {
                        u64::MAX
                    } else {
                        2 * MS
                    },
                    "the caller's remaining time"
                );
                assert!(
                    call.points.iter().all(|point| point.value == value),
                    "the caller's values"
                );
                assert_eq!(ids.len(), 2, "both points");
                match outcome {
                    Outcome::RmLoss | Outcome::CompleteWithRmLoss => {
                        assert!(world.vk.is_lost(), "the RM's loss becomes Mesa's");
                        assert!(
                            state.diagnostic.contains("RM maintenance reported"),
                            "the RM's loss is named"
                        );
                    }
                    Outcome::MesaLoss | Outcome::CompleteWithMesaLoss => {
                        assert!(world.vk.is_lost(), "Mesa's loss stays");
                    }
                    Outcome::Deadline | Outcome::Complete | Outcome::RmError => {
                        assert!(!world.vk.is_lost(), "no loss");
                    }
                }
            }
        }
    }
}

/// Single waits through Mesa's `vk_sync_wait`: expiry, loss already known on
/// either side (no RM wait; Mesa's count is read before the RM's flag), and a
/// wait that completes after more than six seconds without being taken for
/// loss.
fn single_waits(world: &World, sync: &Sync) {
    let vk = &world.vk;
    world.case("expired deadline");
    {
        let mut state = state();
        state.sync.clock = Some(NOW);
        state.sync.wait = WaitScript {
            status: 2,
            elapse: None,
            ..WaitScript::COMPLETE
        };
    }
    let got = sync.wait(vk, 0, VK_SYNC_WAIT_COMPLETE, 0);
    result("vk_sync_wait", &got);
    assert_eq!(got, VK_TIMEOUT);
    assert_eq!(state().sync.waits[0].timeout, 0, "a past deadline polls");
    world.expect_not_lost("expired deadline");

    world.case("Mesa already lost");
    vk.count_loss();
    let got = sync.wait(vk, 0, VK_SYNC_WAIT_COMPLETE, u64::MAX);
    result("vk_sync_wait", &got);
    assert_eq!(got, VK_ERROR_DEVICE_LOST);
    {
        let state = state();
        assert!(state.sync.waits.is_empty(), "no RM wait");
        assert_eq!(
            state.is_lost_calls, 0,
            "Mesa's loss count is read before the RM's flag"
        );
    }

    world.case("RM already lost");
    state().rm_lost = true;
    let got = sync.wait(vk, 0, VK_SYNC_WAIT_COMPLETE, u64::MAX);
    result("vk_sync_wait", &got);
    assert_eq!(got, VK_ERROR_DEVICE_LOST);
    assert!(vk.is_lost(), "the RM's loss becomes Mesa's");
    {
        let state = state();
        assert!(state.sync.waits.is_empty(), "no RM wait");
        assert!(
            state.diagnostic.contains("RM maintenance reported"),
            "the RM's loss is named"
        );
    }

    world.case("completion after six seconds");
    {
        let mut state = state();
        state.sync.clock = Some(NOW);
        state.sync.wait = WaitScript {
            elapse: Some(6_251 * MS),
            ..WaitScript::COMPLETE
        };
    }
    let got = sync.wait(vk, 0, VK_SYNC_WAIT_COMPLETE, u64::MAX);
    result("vk_sync_wait", &got);
    assert_eq!(got, VK_SUCCESS);
    assert_eq!(state().sync.clock, Some(NOW + 6_251 * MS));
    assert_eq!(
        state().sync.waits[0].timeout,
        u64::MAX,
        "an infinite wait stays infinite"
    );
    world.expect_not_lost("completion after six seconds");
}

/// An empty wait; waits that fit the stack array (no allocation) and one
/// that does not, whose allocation may fail.
fn wait_sizes(world: &World, sync: &Sync) {
    let vk = &world.vk;
    world.case("empty wait");
    let got = mesa::wait_none(vk, world.opaque_type());
    result("wait_many", &got);
    assert_eq!(got, VK_SUCCESS);
    assert!(state().sync.waits.is_empty(), "no RM wait");
    world.case("empty wait on a lost device");
    vk.count_loss();
    let got = mesa::wait_none(vk, world.opaque_type());
    result("wait_many", &got);
    assert_eq!(got, VK_ERROR_DEVICE_LOST);

    let points = |count: usize| vec![(sync, 0); count];
    world.case("16 points");
    fake::fail_next_calloc();
    let got = mesa::wait_many(vk, &points(16), VK_SYNC_WAIT_COMPLETE, u64::MAX);
    fake::stop_failing_calloc();
    result("wait_many", &got);
    assert_eq!(got, VK_SUCCESS, "16 points need no allocation");
    world.case("17 points");
    let got = mesa::wait_many(vk, &points(17), VK_SYNC_WAIT_COMPLETE, u64::MAX);
    result("wait_many", &got);
    assert_eq!(got, VK_SUCCESS);
    assert_eq!(state().sync.waits[0].points.len(), 17);
    world.case("17 points, allocation fails");
    fake::fail_next_calloc();
    let got = mesa::wait_many(vk, &points(17), VK_SYNC_WAIT_COMPLETE, u64::MAX);
    fake::stop_failing_calloc();
    result("wait_many", &got);
    assert_eq!(got, VK_ERROR_OUT_OF_HOST_MEMORY);
    assert!(state().sync.waits.is_empty(), "no RM wait");
    world.expect_not_lost("17 points, allocation fails");
}

/// On a lost device, host operations still reach the RM, which decides what
/// succeeds (`vkSignalSemaphore` and `vkResetFences` list no
/// `VK_ERROR_DEVICE_LOST`), while waits return the loss without waiting.
fn host_operations_after_loss(world: &World, sync: &Sync, other: &Sync) {
    let vk = &world.vk;
    world.case("host operations after loss");
    assert_eq!(vk.set_lost(c"offline device failure"), VK_ERROR_DEVICE_LOST);
    assert!(state().rm_lost, "the loss reached the RM");
    let created =
        Sync::init(vk, world.opaque_type(), VK_SYNC_IS_SHAREABLE, 0).expect("create after loss");
    let results = [
        ("signal", sync.signal(vk, 0)),
        ("reset", sync.reset(vk)),
        ("move", sync.move_from(vk, other)),
        ("import", created.import_opaque_fd(vk, IMPORTED_FD)),
        (
            "export",
            created.export_opaque_fd(vk).err().unwrap_or(VK_SUCCESS),
        ),
    ];
    for (operation, got) in results {
        result(operation, &got);
        assert_eq!(got, VK_SUCCESS, "{operation} after loss");
    }
    let got = sync.wait(vk, 0, VK_SYNC_WAIT_COMPLETE, u64::MAX);
    result("vk_sync_wait", &got);
    assert_eq!(got, VK_ERROR_DEVICE_LOST);
    {
        let state = state();
        assert!(state.sync.waits.is_empty(), "no RM wait after loss");
        assert_eq!(
            (state.sync.creates, state.sync.signals, state.sync.resets),
            (1, 1, 1),
            "host operations reach the RM"
        );
        assert_eq!(state.loss_calls, 1, "the loss is published once");
    }
    created.finish(vk);
    assert_eq!(state().sync.destroys, 1, "finish after loss reaches the RM");
    assert_eq!(state().loss_calls, 1, "and publishes nothing new");
}
