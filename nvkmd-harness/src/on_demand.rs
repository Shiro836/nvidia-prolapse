// SPDX-License-Identifier: MIT
//! Mesa's `VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND` (our `vk_queue.c`
//! patch) run on a stand-in logical device and a queue made by
//! `vk_queue_init`, submitting to a harness driver hook through Mesa's own
//! `vkQueueSubmit2`, semaphore and wait entrypoints.
//!
//! The invariant: the driver hook never receives a wait whose signal has not
//! been submitted or host-signalled, so a driver blocking on its ring can
//! never depend on work only the submitting thread could still submit. Once
//! the submit thread is drained, ordinary submissions run on the caller
//! again, in order; a deferred submission's binary signal is reserved at
//! submit time, so a binary wait queued behind it does not block.
//!
//! A timeline sync type of the harness stands in for the RM payload: per
//! sync object, `pending` is the largest submitted or host-signalled value
//! (what `VK_SYNC_WAIT_PENDING` observes) and `completed` the largest
//! completed one; the driver hook completes each signal at submission.
//!
//! Threading: Mesa's submit thread calls the hook and the sync callbacks
//! concurrently with the suite; all their state sits behind one mutex, and
//! the condition variable wakes CPU waits on every publication. No lock is
//! held while calling back into Mesa.
use std::collections::HashMap;
use std::ptr;
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard};
use std::thread::ThreadId;
use std::time::Duration;

use nvkmd_sys::{
    NVKMD_SYS_VK_QUEUE_SUBMIT_SIGNAL_COUNT, NVKMD_SYS_VK_QUEUE_SUBMIT_SIGNALS,
    NVKMD_SYS_VK_QUEUE_SUBMIT_WAIT_COUNT, NVKMD_SYS_VK_QUEUE_SUBMIT_WAITS,
    VK_DEVICE_TIMELINE_MODE_NATIVE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    VK_QUEUE_SUBMIT_MODE_IMMEDIATE, VK_QUEUE_SUBMIT_MODE_THREADED,
    VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND, VK_SEMAPHORE_TYPE_TIMELINE,
    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, VK_STRUCTURE_TYPE_SUBMIT_INFO_2, VK_SUCCESS,
    VK_SYNC_FEATURE_BINARY, VK_SYNC_FEATURE_CPU_RESET, VK_SYNC_FEATURE_CPU_SIGNAL,
    VK_SYNC_FEATURE_CPU_WAIT, VK_SYNC_FEATURE_GPU_WAIT, VK_SYNC_FEATURE_TIMELINE,
    VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL, VK_SYNC_FEATURE_WAIT_PENDING, VK_SYNC_IS_TIMELINE,
    VK_SYNC_WAIT_ANY, VK_SYNC_WAIT_PENDING, VK_TIMEOUT, VkResult, VkSemaphore,
    VkSemaphoreCreateInfo, VkSemaphoreSignalInfo, VkSemaphoreSubmitInfo, VkSemaphoreTypeCreateInfo,
    VkSemaphoreWaitInfo, VkSubmitInfo2, vk_device, vk_queue, vk_queue_submit, vk_sync,
    vk_sync_signal, vk_sync_type, vk_sync_wait, vk_sync_wait_flags,
};

use crate::fake::{live_allocations, note, reset};
use crate::mesa::{PhysicalDevice, RuntimeQueue, VkDevice};
use nv_rm_core::conv::usize_from_u32;

/// How long a CPU wait of the stand-in type may block before the suite
/// fails: every wait in the suite is satisfied by work it submits itself.
const WAIT_LIMIT: Duration = Duration::from_secs(10);

/// The values of one sync object of the stand-in type.
#[derive(Clone, Copy, Default)]
struct Payload {
    pending: u64,
    completed: u64,
    /// A reserved temporary: signals land on this sync object (address).
    target: Option<usize>,
    /// Created with `VK_SYNC_IS_TIMELINE` (a binary one is signalled at 1).
    timeline: bool,
}

#[derive(Default)]
struct Driver {
    /// Live sync objects of the stand-in type, by address.
    syncs: HashMap<usize, Payload>,
    calls: u32,
    caller_calls: u32,
    caller: Option<ThreadId>,
    /// Per driver call: the first signal value, 0 without signals.
    order: Vec<u64>,
    reserves: u32,
}

static DRIVER: LazyLock<Mutex<Driver>> = LazyLock::new(|| Mutex::new(Driver::default()));
/// Notified on every publication; waits on [`DRIVER`]'s mutex.
static PUBLISHED: Condvar = Condvar::new();

fn driver() -> MutexGuard<'static, Driver> {
    DRIVER
        .lock()
        .expect("a panic in a callback aborts the harness, so the lock is never poisoned")
}

fn payload(driver: &mut Driver, sync: *const vk_sync) -> &mut Payload {
    driver
        .syncs
        .get_mut(&sync.addr())
        .expect("Mesa uses only sync objects it initialised")
}

/// Publishes a signal of `value` on `sync` (binary: signalled).
fn publish(sync: *const vk_sync, value: u64) {
    let mut driver = driver();
    let own = *payload(&mut driver, sync);
    let value = if own.timeline { value } else { 1 };
    let target = own.target;
    let payload = match target {
        Some(address) => driver
            .syncs
            .get_mut(&address)
            .expect("the reserved source lives"),
        None => payload(&mut driver, sync),
    };
    payload.pending = payload.pending.max(value);
    payload.completed = payload.completed.max(value);
    PUBLISHED.notify_all();
}

fn ready(driver: &mut Driver, wait: &vk_sync_wait, flags: vk_sync_wait_flags) -> bool {
    let payload = *payload(driver, wait.sync);
    let target = if payload.timeline { wait.wait_value } else { 1 };
    payload.completed >= target || (flags & VK_SYNC_WAIT_PENDING != 0 && payload.pending >= target)
}

unsafe extern "C" fn init(_: *mut vk_device, sync: *mut vk_sync, value: u64) -> VkResult {
    // SAFETY: vk_sync_init (vk_sync.c) sets the header's type and flags
    // before calling init; the object is live for the call.
    let flags = unsafe { (*sync).flags };
    let payload = Payload {
        pending: value,
        completed: value,
        target: None,
        timeline: flags & VK_SYNC_IS_TIMELINE != 0,
    };
    driver().syncs.insert(sync.addr(), payload);
    VK_SUCCESS
}

unsafe extern "C" fn finish(_: *mut vk_device, sync: *mut vk_sync) {
    driver().syncs.remove(&sync.addr());
}

unsafe extern "C" fn signal(_: *mut vk_device, sync: *mut vk_sync, value: u64) -> VkResult {
    publish(sync, value);
    VK_SUCCESS
}

unsafe extern "C" fn get_value(_: *mut vk_device, sync: *mut vk_sync, value: *mut u64) -> VkResult {
    let completed = payload(&mut driver(), sync).completed;
    // SAFETY: Mesa passes a writable output.
    unsafe { value.write(completed) };
    VK_SUCCESS
}

unsafe extern "C" fn reset_sync(_: *mut vk_device, sync: *mut vk_sync) -> VkResult {
    let mut driver = driver();
    let payload = payload(&mut driver, sync);
    payload.pending = 0;
    payload.completed = 0;
    VK_SUCCESS
}

unsafe extern "C" fn move_sync(
    _: *mut vk_device,
    dst: *mut vk_sync,
    src: *mut vk_sync,
) -> VkResult {
    let mut driver = driver();
    let moved = *payload(&mut driver, src);
    let dst = payload(&mut driver, dst);
    dst.pending = moved.pending;
    dst.completed = moved.completed;
    let src = payload(&mut driver, src);
    src.pending = 0;
    src.completed = 0;
    VK_SUCCESS
}

/// The reserved signal is pending at once; the temporary forwards to it.
unsafe extern "C" fn reserve_signal(
    _: *mut vk_device,
    dst: *mut vk_sync,
    src: *mut vk_sync,
    reserved: *mut bool,
) -> VkResult {
    let mut driver = driver();
    payload(&mut driver, dst).target = Some(src.addr());
    payload(&mut driver, src).pending = 1;
    driver.reserves += 1;
    PUBLISHED.notify_all();
    drop(driver);
    // SAFETY: Mesa passes a writable output.
    unsafe { reserved.write(true) };
    VK_SUCCESS
}

unsafe extern "C" fn wait_many(
    _: *mut vk_device,
    count: u32,
    waits: *const vk_sync_wait,
    flags: vk_sync_wait_flags,
    deadline: u64,
) -> VkResult {
    assert_eq!(flags & VK_SYNC_WAIT_ANY, 0, "the suite waits for all");
    let waits = if count == 0 {
        &[][..]
    } else {
        // SAFETY: Mesa passes `count` live waits.
        unsafe { std::slice::from_raw_parts(waits, usize_from_u32(count)) }
    };
    let mut driver = driver();
    loop {
        if waits.iter().all(|wait| ready(&mut driver, wait, flags)) {
            return VK_SUCCESS;
        }
        if deadline != u64::MAX {
            // Only zero-timeout checks reach here in this suite.
            return VK_TIMEOUT;
        }
        let (next, timeout) = PUBLISHED
            .wait_timeout(driver, WAIT_LIMIT)
            .expect("a panic in a callback aborts the harness");
        assert!(
            !timeout.timed_out(),
            "a CPU wait of the suite was never satisfied"
        );
        driver = next;
    }
}

static TYPE: vk_sync_type = vk_sync_type {
    size: size_of::<vk_sync>(),
    features: VK_SYNC_FEATURE_BINARY
        | VK_SYNC_FEATURE_CPU_RESET
        | VK_SYNC_FEATURE_TIMELINE
        | VK_SYNC_FEATURE_GPU_WAIT
        | VK_SYNC_FEATURE_CPU_WAIT
        | VK_SYNC_FEATURE_CPU_SIGNAL
        | VK_SYNC_FEATURE_WAIT_PENDING
        | VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL,
    init: Some(init),
    finish: Some(finish),
    signal: Some(signal),
    signal_many: None,
    get_value: Some(get_value),
    reset: Some(reset_sync),
    reset_many: None,
    move_: Some(move_sync),
    reserve_signal: Some(reserve_signal),
    wait: None,
    wait_many: Some(wait_many),
    import_opaque_fd: None,
    export_opaque_fd: None,
    import_sync_file: None,
    import_win32_handle: None,
    export_win32_handle: None,
    set_win32_export_params: None,
    export_sync_file: None,
};

/// Reads the member at `at` of the live submission `submit`.
///
/// # Safety
///
/// `at` is the offset of a member of type `T` in `vk_queue_submit`.
unsafe fn member<T: Copy>(submit: *const vk_queue_submit, at: u32) -> T {
    // SAFETY: in bounds (caller).
    let slot = unsafe { submit.byte_add(usize_from_u32(at)) };
    // SAFETY: aligned (Mesa's layout); Mesa does not write the submission
    // while the driver hook runs.
    unsafe { slot.cast::<T>().read() }
}

/// The driver hook: checks every wait is already pending, records the
/// calling thread and the first signal value, completes the signals.
unsafe extern "C" fn driver_submit(_: *mut vk_queue, submit: *mut vk_queue_submit) -> VkResult {
    // SAFETY: the offsets of these members (wrapper.h).
    let wait_count: u32 = unsafe { member(submit, NVKMD_SYS_VK_QUEUE_SUBMIT_WAIT_COUNT) };
    // SAFETY: as above.
    let waits: *const vk_sync_wait = unsafe { member(submit, NVKMD_SYS_VK_QUEUE_SUBMIT_WAITS) };
    // SAFETY: as above.
    let signal_count: u32 = unsafe { member(submit, NVKMD_SYS_VK_QUEUE_SUBMIT_SIGNAL_COUNT) };
    // SAFETY: as above.
    let signals: *const vk_sync_signal =
        unsafe { member(submit, NVKMD_SYS_VK_QUEUE_SUBMIT_SIGNALS) };
    let waits = if wait_count == 0 {
        &[][..]
    } else {
        // SAFETY: the submission holds `wait_count` waits.
        unsafe { std::slice::from_raw_parts(waits, usize_from_u32(wait_count)) }
    };
    let signals = if signal_count == 0 {
        &[][..]
    } else {
        // SAFETY: the submission holds `signal_count` signals.
        unsafe { std::slice::from_raw_parts(signals, usize_from_u32(signal_count)) }
    };
    let mut driver = driver();
    for wait in waits {
        assert!(
            ready(&mut driver, wait, VK_SYNC_WAIT_PENDING),
            "the driver received a wait whose signal was not submitted"
        );
    }
    if driver.caller == Some(std::thread::current().id()) {
        driver.caller_calls += 1;
    }
    driver.calls += 1;
    let first = signals.first().map_or(0, |signal| signal.signal_value);
    driver.order.push(first);
    drop(driver);
    for signal in signals {
        publish(signal.sync, signal.signal_value);
    }
    VK_SUCCESS
}

/// (driver calls, of them on the suite's thread).
fn calls() -> (u32, u32) {
    let driver = driver();
    (driver.calls, driver.caller_calls)
}

fn semaphore_info(semaphore: VkSemaphore, value: u64) -> VkSemaphoreSubmitInfo {
    VkSemaphoreSubmitInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        pNext: ptr::null(),
        semaphore,
        value,
        stageMask: VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        deviceIndex: 0,
    }
}

fn submit_info(
    waits: &[VkSemaphoreSubmitInfo],
    signals: &[VkSemaphoreSubmitInfo],
) -> VkSubmitInfo2 {
    VkSubmitInfo2 {
        sType: VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        pNext: ptr::null(),
        flags: 0,
        waitSemaphoreInfoCount: u32::try_from(waits.len()).expect("at most one wait"),
        pWaitSemaphoreInfos: waits.as_ptr(),
        commandBufferInfoCount: 0,
        pCommandBufferInfos: ptr::null(),
        signalSemaphoreInfoCount: u32::try_from(signals.len()).expect("at most one signal"),
        pSignalSemaphoreInfos: signals.as_ptr(),
    }
}

/// One submission: an optional wait and an optional signal.
fn submit(
    queue: &RuntimeQueue,
    wait: Option<(VkSemaphore, u64)>,
    signal: Option<(VkSemaphore, u64)>,
) {
    let waits: Vec<_> = wait
        .map(|(s, v)| semaphore_info(s, v))
        .into_iter()
        .collect();
    let signals: Vec<_> = signal
        .map(|(s, v)| semaphore_info(s, v))
        .into_iter()
        .collect();
    let result = queue.submit(&[submit_info(&waits, &signals)]);
    assert_eq!(result, VK_SUCCESS, "vkQueueSubmit2");
}

fn timeline(vk: &VkDevice) -> VkSemaphore {
    let kind = VkSemaphoreTypeCreateInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        pNext: ptr::null(),
        semaphoreType: VK_SEMAPHORE_TYPE_TIMELINE,
        initialValue: 0,
    };
    let info = VkSemaphoreCreateInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        pNext: ptr::from_ref(&kind).cast(),
        flags: 0,
    };
    vk.create_semaphore(&info)
        .expect("vkCreateSemaphore (timeline)")
}

fn host_signal(vk: &VkDevice, semaphore: VkSemaphore, value: u64) {
    let info = VkSemaphoreSignalInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        pNext: ptr::null(),
        semaphore,
        value,
    };
    assert_eq!(vk.signal_semaphore(&info), VK_SUCCESS, "vkSignalSemaphore");
}

/// Waits without a timeout, as an application may; the stand-in type bounds
/// it ([`WAIT_LIMIT`]).
fn host_wait(vk: &VkDevice, semaphore: VkSemaphore, value: u64) {
    let info = VkSemaphoreWaitInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        pNext: ptr::null(),
        flags: 0,
        semaphoreCount: 1,
        pSemaphores: &raw const semaphore,
        pValues: &raw const value,
    };
    assert_eq!(
        vk.wait_semaphores(&info, u64::MAX),
        VK_SUCCESS,
        "vkWaitSemaphores"
    );
}

fn checkpoint(what: &str, expected: (u32, u32)) {
    let got = calls();
    note(format!(
        "{what}: driver calls {}, on the submitting thread {}",
        got.0, got.1
    ));
    assert_eq!(got, expected, "{what}");
}

/// `THREADED_ON_DEMAND` routing (see the module doc).
pub(crate) fn submit_on_demand() {
    reset();
    *driver() = Driver {
        caller: Some(std::thread::current().id()),
        ..Driver::default()
    };
    let types = [ptr::from_ref(&TYPE), ptr::null()];
    let physical = PhysicalDevice::with_sync_types(types.as_ptr());
    let vk = VkDevice::new();
    vk.set_queue_runtime(
        &physical,
        VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND,
        VK_DEVICE_TIMELINE_MODE_NATIVE,
    );
    let queue = RuntimeQueue::new(&vk, driver_submit);
    assert_eq!(
        queue.submit_mode(),
        VK_QUEUE_SUBMIT_MODE_IMMEDIATE,
        "a fresh queue submits on the caller"
    );
    let gate = timeline(&vk);
    let done = timeline(&vk);

    submit(&queue, None, Some((done, 1)));
    submit(&queue, Some((done, 1)), Some((done, 2)));
    checkpoint("pending waits on a fresh queue", (2, 2));
    assert_eq!(queue.submit_mode(), VK_QUEUE_SUBMIT_MODE_IMMEDIATE);

    // A wait before its signal on the submitting thread, then work behind it:
    // none of it reaches the driver until the signal exists.
    submit(&queue, Some((gate, 1)), None);
    note(format!(
        "wait before signal: queue threaded {}",
        queue.submit_mode() == VK_QUEUE_SUBMIT_MODE_THREADED
    ));
    assert_eq!(queue.submit_mode(), VK_QUEUE_SUBMIT_MODE_THREADED);
    for value in 3..40 {
        submit(&queue, None, Some((done, value)));
    }
    submit(&queue, Some((done, 2)), Some((done, 40)));
    checkpoint("behind the unsignalled wait", (2, 2));
    host_signal(&vk, gate, 1);
    host_wait(&vk, done, 40);
    checkpoint("after the host signal", (41, 2));
    {
        let driver = driver();
        assert_eq!(driver.order[..3], [1, 2, 0]);
        for (index, value) in driver.order.iter().enumerate().skip(3) {
            assert_eq!(
                *value,
                u64::try_from(index).expect("small"),
                "submission order through the thread"
            );
        }
    }

    // Drained thread: ordinary submissions run on the caller again.
    submit(&queue, Some((gate, 1)), Some((done, 41)));
    submit(&queue, None, Some((done, 42)));
    checkpoint("drained queue", (43, 4));
    assert_eq!(queue.wait_idle(), VK_SUCCESS, "vkQueueWaitIdle");
    checkpoint("idle-queue signal", (44, 5));

    // A second wait-before-signal episode on the same queue.
    submit(&queue, Some((gate, 2)), Some((done, 43)));
    submit(&queue, None, Some((done, 44)));
    checkpoint("second episode, before the signal", (44, 5));
    host_signal(&vk, gate, 2);
    host_wait(&vk, done, 44);
    checkpoint("second episode, after the signal", (46, 5));
    {
        let driver = driver();
        assert!(
            driver
                .order
                .windows(2)
                .all(|pair| pair[1] == 0 || pair[1] > pair[0]),
            "signal values rise in submission order"
        );
    }

    deferred_binary_signal(&vk, &queue, gate, done);

    vk.destroy_semaphore(gate);
    vk.destroy_semaphore(done);
    drop(queue);
    drop(vk);
    drop(physical);
    assert!(driver().syncs.is_empty(), "every sync object was finished");
    assert_eq!(live_allocations(), 0, "the runtime frees what it allocated");
}

/// A deferred submission's binary signal is reserved at submit time, so a
/// binary wait submitted behind it does not wait for the deferred work
/// (which only this thread could release).
fn deferred_binary_signal(
    vk: &VkDevice,
    queue: &RuntimeQueue,
    gate: VkSemaphore,
    done: VkSemaphore,
) {
    let binary_info = VkSemaphoreCreateInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
    };
    let binary = vk
        .create_semaphore(&binary_info)
        .expect("vkCreateSemaphore (binary)");
    let gate_wait = [semaphore_info(gate, 3)];
    let binary_op = [semaphore_info(binary, 0)];
    assert_eq!(
        queue.submit(&[submit_info(&gate_wait, &binary_op)]),
        VK_SUCCESS
    );
    let reserves = driver().reserves;
    note(format!("deferred binary signal: reservations {reserves}"));
    assert_eq!(reserves, 1);
    submit(queue, None, None);
    assert_eq!(queue.submit(&[submit_info(&binary_op, &[])]), VK_SUCCESS);
    submit(queue, None, Some((done, 45)));
    checkpoint("binary wait behind a deferred signal", (46, 5));
    host_signal(vk, gate, 3);
    host_wait(vk, done, 45);
    checkpoint("after the third host signal", (50, 5));

    vk.destroy_semaphore(binary);
}
