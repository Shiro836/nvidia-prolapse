// SPDX-License-Identifier: MIT
//! The fake RM's sync payload entries (`nv_rm_sync_*`) and Mesa's clock
//! (`os_time_get_nano`, replaced with `--wrap` so waits see scripted time).
//! Every entry checks the device, records its arguments and result in the call
//! log and returns the status the suite chose. A wait follows a script: it may
//! publish Mesa's or the RM's loss while "sleeping", advance the clock, and
//! returns the scripted status.
use std::ffi::{c_char, c_int};

use nv_rm_abi::{NvRmDevice, NvRmSyncId, NvRmSyncPoint};
use nvkmd_sys::{_vk_device_set_lost, VK_SUCCESS, vk_device};

use crate::fake::{check_device, render_fd, state};

/// First payload id the fake hands out.
pub(crate) const FIRST_SYNC: u64 = 77;
/// The descriptor `nv_rm_sync_export_fd` hands out.
pub(crate) const EXPORTED_SYNC_FD: c_int = 88;
/// The value `nv_rm_sync_get_value` reports unless a suite says otherwise.
pub(crate) const TIMELINE_VALUE: u64 = 8;

/// What the fake `nv_rm_sync_wait_many` does.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WaitScript {
    /// The status it returns (0 complete, 2 timeout, negative failure).
    pub(crate) status: i32,
    /// Publishes Mesa's loss of the device (`vk_device_set_lost`) while
    /// waiting.
    pub(crate) lose_mesa: bool,
    /// Sets the RM's loss flag while waiting.
    pub(crate) lose_rm: bool,
    /// Time the wait takes: `None` = the whole timeout it was given.
    pub(crate) elapse: Option<u64>,
}

impl WaitScript {
    /// Completes at once.
    pub(crate) const COMPLETE: Self = Self {
        status: VK_SUCCESS,
        lose_mesa: false,
        lose_rm: false,
        elapse: Some(0),
    };
}

/// A wait the fake received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WaitCall {
    pub(crate) points: Vec<NvRmSyncPoint>,
    pub(crate) flags: u32,
    pub(crate) timeout: u64,
}

/// Knobs and counters of the sync entries.
#[derive(Debug)]
pub(crate) struct Syncs {
    next_id: u64,
    pub(crate) create_status: i32,
    pub(crate) destroy_status: i32,
    pub(crate) signal_status: i32,
    pub(crate) get_value_status: i32,
    pub(crate) reset_status: i32,
    pub(crate) move_status: i32,
    pub(crate) reserve_status: i32,
    /// What `nv_rm_sync_reserve` answers in `*reserved`.
    pub(crate) reserve_done: u32,
    pub(crate) import_status: i32,
    pub(crate) export_status: i32,
    pub(crate) wait: WaitScript,
    /// Mesa's clock while a suite scripts it; `None` = the real clock.
    pub(crate) clock: Option<u64>,
    pub(crate) creates: u32,
    pub(crate) destroys: u32,
    pub(crate) signals: u32,
    pub(crate) resets: u32,
    pub(crate) moves: u32,
    pub(crate) reserves: u32,
    pub(crate) imports: u32,
    pub(crate) exports: u32,
    pub(crate) waits: Vec<WaitCall>,
    /// The last `(id, value)` of a signal; `shared_source` of a move;
    /// `opaque` of an import or export.
    pub(crate) last_signal: (u64, u64),
    pub(crate) last_shared: u32,
    pub(crate) last_opaque: u32,
}

impl Syncs {
    pub(crate) const fn new() -> Self {
        Self {
            next_id: FIRST_SYNC,
            create_status: VK_SUCCESS,
            destroy_status: VK_SUCCESS,
            signal_status: VK_SUCCESS,
            get_value_status: VK_SUCCESS,
            reset_status: VK_SUCCESS,
            move_status: VK_SUCCESS,
            reserve_status: VK_SUCCESS,
            reserve_done: 1,
            import_status: VK_SUCCESS,
            export_status: VK_SUCCESS,
            wait: WaitScript::COMPLETE,
            clock: None,
            creates: 0,
            destroys: 0,
            signals: 0,
            resets: 0,
            moves: 0,
            reserves: 0,
            imports: 0,
            exports: 0,
            waits: Vec::new(),
            last_signal: (0, 0),
            last_shared: 0,
            last_opaque: 0,
        }
    }

    /// Defaults for the next case; ids keep counting.
    pub(crate) const fn next_case(&self) -> Self {
        let mut next = Self::new();
        next.next_id = self.next_id;
        next
    }
}

fn check_sync(device: *const NvRmDevice, sync: NvRmSyncId) {
    check_device(device);
    let next = state().sync.next_id;
    assert!(
        (FIRST_SYNC..next).contains(&sync.get()),
        "only payloads the fake made are passed back"
    );
}

/// Creates a payload: the next id.
///
/// # Safety
/// `out` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_create(
    device: *mut NvRmDevice,
    flags: u32,
    initial: u64,
    out: *mut NvRmSyncId,
) -> i32 {
    check_device(device);
    let mut state = state();
    state.sync.creates += 1;
    let status = state.sync.create_status;
    let id = state.sync.next_id;
    if status == VK_SUCCESS {
        state.sync.next_id += 1;
        // SAFETY: nv_rm_sync_create's contract: `out` is writable.
        unsafe { out.write(NvRmSyncId::new(id)) };
    }
    state.record(format!(
        "nv_rm_sync_create(flags={flags:#x}, initial={initial}) = {status}{}",
        if status == VK_SUCCESS {
            format!(" (sync {id})")
        } else {
            String::new()
        }
    ));
    status
}
const _: nv_rm_abi::NvRmSyncCreate<NvRmDevice> = nv_rm_sync_create;

/// Drops a payload reference.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_sync_destroy_deferred(
    device: *mut NvRmDevice,
    sync: NvRmSyncId,
) -> i32 {
    check_sync(device, sync);
    let mut state = state();
    state.sync.destroys += 1;
    let status = state.sync.destroy_status;
    state.record(format!(
        "nv_rm_sync_destroy_deferred({}) = {status}",
        sync.get()
    ));
    status
}
const _: nv_rm_abi::NvRmSyncDestroyDeferred<NvRmDevice> = nv_rm_sync_destroy_deferred;

/// A host signal.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_sync_signal(
    device: *mut NvRmDevice,
    sync: NvRmSyncId,
    value: u64,
) -> i32 {
    check_sync(device, sync);
    let mut state = state();
    state.sync.signals += 1;
    state.sync.last_signal = (sync.get(), value);
    let status = state.sync.signal_status;
    state.record(format!(
        "nv_rm_sync_signal({}, {value}) = {status}",
        sync.get()
    ));
    status
}
const _: nv_rm_abi::NvRmSyncSignal<NvRmDevice> = nv_rm_sync_signal;

/// A timeline payload's value: [`TIMELINE_VALUE`], written on success only.
///
/// # Safety
/// `value` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_get_value(
    device: *mut NvRmDevice,
    sync: NvRmSyncId,
    value: *mut u64,
) -> i32 {
    check_sync(device, sync);
    let mut state = state();
    let status = state.sync.get_value_status;
    if status == VK_SUCCESS {
        // SAFETY: nv_rm_sync_get_value's contract: `value` is writable.
        unsafe { value.write(TIMELINE_VALUE) };
    }
    state.record(format!("nv_rm_sync_get_value({}) = {status}", sync.get()));
    status
}
const _: nv_rm_abi::NvRmSyncGetValue<NvRmDevice> = nv_rm_sync_get_value;

/// Resets a binary payload.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_sync_reset(device: *mut NvRmDevice, sync: NvRmSyncId) -> i32 {
    check_sync(device, sync);
    let mut state = state();
    state.sync.resets += 1;
    let status = state.sync.reset_status;
    state.record(format!("nv_rm_sync_reset({}) = {status}", sync.get()));
    status
}
const _: nv_rm_abi::NvRmSyncReset<NvRmDevice> = nv_rm_sync_reset;

/// Moves `src`'s payload into `dst`.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_sync_move(
    device: *mut NvRmDevice,
    dst: NvRmSyncId,
    src: NvRmSyncId,
    shared: u32,
) -> i32 {
    check_sync(device, dst);
    check_sync(device, src);
    let mut state = state();
    state.sync.moves += 1;
    state.sync.last_shared = shared;
    let status = state.sync.move_status;
    state.record(format!(
        "nv_rm_sync_move({}, {}, shared={shared}) = {status}",
        dst.get(),
        src.get()
    ));
    status
}
const _: nv_rm_abi::NvRmSyncMove<NvRmDevice> = nv_rm_sync_move;

/// Reserves `src`'s deferred signal for `dst`: `*reserved` is the scripted
/// answer, written whatever the status (the backend must not trust it on
/// failure).
///
/// # Safety
/// `reserved` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_reserve(
    device: *mut NvRmDevice,
    dst: NvRmSyncId,
    src: NvRmSyncId,
    reserved: *mut u32,
) -> i32 {
    check_sync(device, dst);
    check_sync(device, src);
    let mut state = state();
    state.sync.reserves += 1;
    let status = state.sync.reserve_status;
    let done = state.sync.reserve_done;
    // SAFETY: nv_rm_sync_reserve's contract: `reserved` is writable.
    unsafe { reserved.write(done) };
    state.record(format!(
        "nv_rm_sync_reserve({}, {}) = {status} (reserved {done})",
        dst.get(),
        src.get()
    ));
    status
}
const _: nv_rm_abi::NvRmSyncReserve<NvRmDevice> = nv_rm_sync_reserve;

/// Imports a handle's fence state.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_sync_import_fd(
    device: *mut NvRmDevice,
    sync: NvRmSyncId,
    fd: c_int,
    opaque: u32,
) -> i32 {
    check_sync(device, sync);
    let mut state = state();
    state.sync.imports += 1;
    state.sync.last_opaque = opaque;
    let status = state.sync.import_status;
    state.record(format!(
        "nv_rm_sync_import_fd({}, fd={fd}, opaque={opaque}) = {status}",
        sync.get()
    ));
    status
}
const _: nv_rm_abi::NvRmSyncImportFd<NvRmDevice> = nv_rm_sync_import_fd;

/// Exports [`EXPORTED_SYNC_FD`] (written on success only) through the
/// device's render node.
///
/// # Safety
/// `fd` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_export_fd(
    device: *mut NvRmDevice,
    sync: NvRmSyncId,
    opaque: u32,
    drm_fd: c_int,
    fd: *mut c_int,
) -> i32 {
    check_sync(device, sync);
    assert_eq!(
        drm_fd,
        render_fd(),
        "exports go through the device's render node"
    );
    let mut state = state();
    state.sync.exports += 1;
    state.sync.last_opaque = opaque;
    let status = state.sync.export_status;
    if status == VK_SUCCESS {
        // SAFETY: nv_rm_sync_export_fd's contract: `fd` is writable.
        unsafe { fd.write(EXPORTED_SYNC_FD) };
    }
    state.record(format!(
        "nv_rm_sync_export_fd({}, opaque={opaque}) = {status}",
        sync.get()
    ));
    status
}
const _: nv_rm_abi::NvRmSyncExportFd<NvRmDevice> = nv_rm_sync_export_fd;

/// Waits as the suite's [`WaitScript`] says.
///
/// # Safety
/// `points` is readable for `count` entries.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_sync_wait_many(
    device: *mut NvRmDevice,
    points: *const NvRmSyncPoint,
    count: u32,
    flags: u32,
    timeout: u64,
) -> i32 {
    check_device(device);
    assert!(count > 0, "an empty wait never reaches the RM");
    let len = usize::try_from(count).expect("a u32 count fits usize");
    // SAFETY: nv_rm_sync_wait_many's contract: readable for `count` entries.
    let points = unsafe { core::slice::from_raw_parts(points, len) }.to_vec();
    for point in &points {
        check_sync(device, point.sync);
    }
    let (script, loss_device) = {
        let mut state = state();
        let script = state.sync.wait;
        let list: Vec<String> = points
            .iter()
            .map(|point| format!("{}:{}", point.sync.get(), point.value))
            .collect();
        state.record(format!(
            "nv_rm_sync_wait_many([{}], flags={flags:#x}, timeout={timeout}) = {}",
            list.join(", "),
            script.status
        ));
        state.sync.waits.push(WaitCall {
            points,
            flags,
            timeout,
        });
        let elapsed = script.elapse.unwrap_or(timeout);
        if let Some(now) = state.sync.clock.as_mut() {
            *now = now.saturating_add(elapsed);
        }
        if script.lose_rm {
            state.rm_lost = true;
        }
        (script, state.loss_device)
    };
    if script.lose_mesa {
        assert_ne!(loss_device, 0, "a scripted Mesa loss names its device");
        let vk = core::ptr::with_exposed_provenance_mut::<vk_device>(loss_device);
        let message: &[u8] = b"scripted loss during wait\0";
        // SAFETY: the suite's live logical device, whose hooks reach the fake
        // with the state unlocked; "%s" consumes the one string argument.
        unsafe {
            _vk_device_set_lost(
                vk,
                c"harness".as_ptr(),
                0,
                c"%s".as_ptr(),
                message.as_ptr().cast::<c_char>(),
            );
        }
    }
    script.status
}
const _: nv_rm_abi::NvRmSyncWaitMany<NvRmDevice> = nv_rm_sync_wait_many;

unsafe extern "C" {
    fn __real_os_time_get_nano() -> i64;
}

/// Mesa's `os_time_get_nano`: the suite's scripted clock, else the real one.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn __wrap_os_time_get_nano() -> i64 {
    let scripted = state().sync.clock;
    match scripted {
        Some(now) => i64::try_from(now).expect("scripted times fit Mesa's clock"),
        // SAFETY: Mesa's clock, no preconditions.
        None => unsafe { __real_os_time_get_nano() },
    }
}
