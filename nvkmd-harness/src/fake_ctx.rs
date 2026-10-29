// SPDX-License-Identifier: MIT
//! The fake RM's context entries: `nv_rm_context_create`,
//! `nv_rm_context_create_shared`, `nv_rm_context_destroy`,
//! `nv_rm_submit_batch` and `nv_rm_wait`. Every entry checks the device and
//! the handles it receives, records its arguments and result in the call log
//! and returns the status the suite chose; a submission follows a script of
//! statuses, one per call.
//!
//! A submission's push list is logged as its count, first and last address
//! and a checksum of every (address, size, flags) triple, so a list cut or
//! kept differently shows in the log without printing thousands of entries.
use std::collections::VecDeque;

use nv_rm_abi::{NvRmContextId, NvRmDevice, NvRmExec, NvRmSyncPoint, NvRmToken};
use nvkmd_sys::{VK_NOT_READY, VK_SUCCESS};

use crate::fake::{check_device, state};

/// First RM context handle the fake hands out.
pub(crate) const FIRST_CONTEXT: u64 = 33;
/// The token a submission refused for ring pressure names: the oldest batch.
pub(crate) const RING_TOKEN: u64 = 9;
/// Tokens of accepted submissions are this plus the submission's index.
pub(crate) const FIRST_TOKEN: u64 = 100;

/// A submission the fake received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Batch {
    pub(crate) context: u64,
    pub(crate) pushes: Vec<NvRmExec>,
    pub(crate) waits: Vec<NvRmSyncPoint>,
    pub(crate) signals: Vec<NvRmSyncPoint>,
    pub(crate) status: i32,
}

/// Knobs and counters of the context entries.
#[derive(Debug)]
pub(crate) struct Contexts {
    next_id: u64,
    pub(crate) create_status: i32,
    pub(crate) destroy_status: i32,
    /// Statuses of the next submissions, in order; success once empty.
    pub(crate) submit_script: VecDeque<i32>,
    /// What `nv_rm_wait` returns.
    pub(crate) wait_status: i32,
    pub(crate) creates: u32,
    pub(crate) destroys: u32,
    /// Every submission since the case started.
    pub(crate) batches: Vec<Batch>,
    /// Every `nv_rm_wait` since the case started: (context, token, timeout).
    pub(crate) waits: Vec<(u64, u64, u64)>,
}

impl Contexts {
    pub(crate) const fn new() -> Self {
        Self {
            next_id: FIRST_CONTEXT,
            create_status: VK_SUCCESS,
            destroy_status: VK_SUCCESS,
            submit_script: VecDeque::new(),
            wait_status: VK_SUCCESS,
            creates: 0,
            destroys: 0,
            batches: Vec::new(),
            waits: Vec::new(),
        }
    }

    /// Defaults for the next case; handles keep counting.
    pub(crate) const fn next_case(&self) -> Self {
        let mut next = Self::new();
        next.next_id = self.next_id;
        next
    }
}

fn check_context(device: *const NvRmDevice, context: NvRmContextId) {
    check_device(device);
    let next = state().ctx.next_id;
    assert!(
        (FIRST_CONTEXT..next).contains(&context.get()),
        "only contexts the fake made are passed back"
    );
}

/// A C list as a vector (empty for count 0, whatever the pointer).
///
/// # Safety
/// `items` is readable for `count` elements when `count` is not 0.
unsafe fn list<T: Copy>(items: *const T, count: u32) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    assert!(!items.is_null(), "a non-empty list has its elements");
    let len = usize::try_from(count).expect("a u32 count fits usize");
    // SAFETY: readable for `count` elements (caller).
    unsafe { core::slice::from_raw_parts(items, len) }.to_vec()
}

fn points(points: &[NvRmSyncPoint]) -> String {
    let list: Vec<String> = points
        .iter()
        .map(|point| format!("{}:{}", point.sync.get(), point.value))
        .collect();
    format!("[{}]", list.join(", "))
}

/// FNV-1a over every push's address, size and flags.
fn checksum(pushes: &[NvRmExec]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for push in pushes {
        for word in [push.address, u64::from(push.size_B), u64::from(push.flags)] {
            for byte in word.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    hash
}

fn describe_pushes(pushes: &[NvRmExec]) -> String {
    match (pushes.first(), pushes.last()) {
        (Some(first), Some(last)) => format!(
            "{} [{:#x}..{:#x}] sum={:#018x}",
            pushes.len(),
            first.address,
            last.address,
            checksum(pushes)
        ),
        _ => "0".to_owned(),
    }
}

/// Creates a context: the next handle.
///
/// # Safety
/// `out` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_create(
    device: *mut NvRmDevice,
    engines: u32,
    out: *mut NvRmContextId,
) -> i32 {
    check_device(device);
    let mut state = state();
    state.ctx.creates += 1;
    let status = state.ctx.create_status;
    let id = state.ctx.next_id;
    if status == VK_SUCCESS {
        state.ctx.next_id += 1;
        // SAFETY: nv_rm_context_create's contract: `out` is writable.
        unsafe { out.write(NvRmContextId::new(id)) };
    }
    state.record(format!(
        "nv_rm_context_create(engines={engines:#x}) = {status}{}",
        if status == VK_SUCCESS {
            format!(" (context {id})")
        } else {
            String::new()
        }
    ));
    status
}
const _: nv_rm_abi::NvRmContextCreate<NvRmDevice> = nv_rm_context_create;

/// Creates a context in `share`'s channel group: the next handle.
///
/// # Safety
/// `out` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_context_create_shared(
    device: *mut NvRmDevice,
    engines: u32,
    share: NvRmContextId,
    out: *mut NvRmContextId,
) -> i32 {
    check_context(device, share);
    let mut state = state();
    state.ctx.creates += 1;
    let status = state.ctx.create_status;
    let id = state.ctx.next_id;
    if status == VK_SUCCESS {
        state.ctx.next_id += 1;
        // SAFETY: nv_rm_context_create_shared's contract: `out` is writable.
        unsafe { out.write(NvRmContextId::new(id)) };
    }
    state.record(format!(
        "nv_rm_context_create_shared(engines={engines:#x}, share={}) = {status}{}",
        share.get(),
        if status == VK_SUCCESS {
            format!(" (context {id})")
        } else {
            String::new()
        }
    ));
    status
}
const _: nv_rm_abi::NvRmContextCreateShared<NvRmDevice> = nv_rm_context_create_shared;

/// Destroys a context.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_context_destroy(
    device: *mut NvRmDevice,
    context: NvRmContextId,
) -> i32 {
    check_context(device, context);
    let mut state = state();
    state.ctx.destroys += 1;
    let status = state.ctx.destroy_status;
    state.record(format!(
        "nv_rm_context_destroy({}) = {status}",
        context.get()
    ));
    status
}
const _: nv_rm_abi::NvRmContextDestroy<NvRmDevice> = nv_rm_context_destroy;

/// Submits a batch: the next scripted status; the token is [`RING_TOKEN`] on
/// `VK_NOT_READY`, else [`FIRST_TOKEN`] plus the batch's index in the case.
///
/// # Safety
/// Each list is readable for its count; `token` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_submit_batch(
    device: *mut NvRmDevice,
    context: NvRmContextId,
    pushes: *const NvRmExec,
    count: u32,
    waits: *const NvRmSyncPoint,
    wait_count: u32,
    signals: *const NvRmSyncPoint,
    signal_count: u32,
    token: *mut NvRmToken,
) -> i32 {
    check_context(device, context);
    // SAFETY: nv_rm_submit_batch's contract: each list is readable for its count.
    let pushes = unsafe { list(pushes, count) };
    // SAFETY: as above.
    let waits = unsafe { list(waits, wait_count) };
    // SAFETY: as above.
    let signals = unsafe { list(signals, signal_count) };
    let mut state = state();
    let status = state.ctx.submit_script.pop_front().unwrap_or(VK_SUCCESS);
    let index = u64::try_from(state.ctx.batches.len()).expect("few batches per case");
    let issued = if status == VK_NOT_READY {
        RING_TOKEN
    } else {
        FIRST_TOKEN + index
    };
    // SAFETY: nv_rm_submit_batch's contract: `token` is writable.
    unsafe { token.write(NvRmToken::new(issued)) };
    state.record(format!(
        "nv_rm_submit_batch({}, pushes={}, waits={}, signals={}) = {status} (token {issued})",
        context.get(),
        describe_pushes(&pushes),
        points(&waits),
        points(&signals)
    ));
    state.ctx.batches.push(Batch {
        context: context.get(),
        pushes,
        waits,
        signals,
        status,
    });
    status
}
const _: nv_rm_abi::NvRmSubmitBatch<NvRmDevice> = nv_rm_submit_batch;

/// Waits for a token: the suite's status.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_wait(
    device: *mut NvRmDevice,
    context: NvRmContextId,
    token: NvRmToken,
    timeout: u64,
) -> i32 {
    check_context(device, context);
    let mut state = state();
    let status = state.ctx.wait_status;
    state.ctx.waits.push((context.get(), token.get(), timeout));
    state.record(format!(
        "nv_rm_wait({}, token {}, timeout={timeout:#x}) = {status}",
        context.get(),
        token.get()
    ));
    status
}
const _: nv_rm_abi::NvRmWait<NvRmDevice> = nv_rm_wait;
