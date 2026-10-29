// SPDX-License-Identifier: MIT
//! Contexts (`nvkmd_ctx`): the device's `create_ctx` and `create_shared_ctx`
//! entries and the context ops table (`nvkmd_rm.c` `rm_ctx_submit` …
//! `nvkmd_rm_create_shared_ctx`).
//!
//! A context with engines owns one RM context (a channel); a bind-only
//! context (`NVKMD_ENGINE_BIND`) owns none and carries out its work on the
//! CPU: waits through Mesa's `vk_sync_wait_many`, signals through
//! `vk_sync_signal`, VA binds through Mesa's `nvkmd_va_*` wrappers.
//!
//! A channel context collects work between flushes: pushes (`exec`) and GPU
//! acquires (`wait`) accumulate, and one `nv_rm_submit_batch` carries them
//! with the signals of the call that flushes. Waits ride on the first
//! submission of a flush, signals on its last; a list longer than one
//! submission is cut by `nvkmd_rm_logic::push`. On ring pressure
//! (`VK_NOT_READY`) the flushing thread waits for the oldest batch and
//! retries; a submission the RM accepted is never replayed, so a failure
//! keeps only the unaccepted suffix.
//!
//! This is the per-submit path: the lists keep their capacity between
//! flushes, the signal list is a kept scratch buffer, and nothing is logged
//! unless a call fails. Mesa serialises calls on one context (the queue's
//! submit lock); the entries take no lock of their own, and none is held
//! across an RM call.
use std::ptr::NonNull;

use nv_rm_abi::{
    NV_RM_ENGINE_2D, NV_RM_ENGINE_3D, NV_RM_ENGINE_COMPUTE, NV_RM_ENGINE_COPY, NV_RM_ENGINE_M2MF,
    NvRmContextId, NvRmExec, NvRmSyncPoint, NvRmToken, nv_rm_context_create,
    nv_rm_context_create_shared, nv_rm_context_destroy, nv_rm_submit_batch, nv_rm_wait,
};
use nv_rm_core::conv::usize_from_u32;
use nvkmd_rm_logic::push::{self, Exec, ExecError, PushError};
use nvkmd_rm_logic::wait::{self, Wait, WaitError, WaitSync};
use nvkmd_sys::{
    NVKMD_BIND_OP_BIND, NVKMD_BIND_OP_UNBIND, NVKMD_ENGINE_2D, NVKMD_ENGINE_3D, NVKMD_ENGINE_BIND,
    NVKMD_ENGINE_COMPUTE, NVKMD_ENGINE_COPY, NVKMD_ENGINE_M2MF, VK_ERROR_DEVICE_LOST,
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_OUT_OF_HOST_MEMORY, VK_NOT_READY, VK_SUCCESS,
    VK_SYNC_WAIT_COMPLETE, VkResult, nvkmd_ctx, nvkmd_ctx_bind, nvkmd_ctx_exec, nvkmd_ctx_ops,
    nvkmd_dev, nvkmd_engines, nvkmd_va_bind_mem, nvkmd_va_unbind, vk_device, vk_object_base,
    vk_sync_signal, vk_sync_wait, vk_sync_wait_many,
};

use super::dev::{RmDev, logical_device};
use super::sync::{is_timeline, payload};
use super::{Diag, diag, rm_result, try_alloc_zeroed, unsupported, vk_error, vk_errorf};

// Mesa's engine bits are passed to the RM unchanged (`nv_rm.h`).
const _: () = assert!(NVKMD_ENGINE_COPY == NV_RM_ENGINE_COPY);
const _: () = assert!(NVKMD_ENGINE_2D == NV_RM_ENGINE_2D);
const _: () = assert!(NVKMD_ENGINE_3D == NV_RM_ENGINE_3D);
const _: () = assert!(NVKMD_ENGINE_M2MF == NV_RM_ENGINE_M2MF);
const _: () = assert!(NVKMD_ENGINE_COMPUTE == NV_RM_ENGINE_COMPUTE);

/// A context. Mesa holds `&base`; [`ctx_destroy`] frees it.
#[repr(C)]
pub(super) struct RmCtx {
    base: nvkmd_ctx,
    /// The logical device of the log object the context was made with; a
    /// bind context waits and signals through it.
    vk: NonNull<vk_device>,
    /// The RM context; `None` for a bind context.
    channel: Option<NvRmContextId>,
    /// Pushes not yet submitted, in `exec` order.
    pending: Vec<NvRmExec>,
    /// GPU acquires for the next submission. They only name payloads: Mesa
    /// keeps the syncs alive until the submission that carries them was
    /// accepted, and the RM then holds its own references.
    waits: Vec<NvRmSyncPoint>,
    /// Scratch for one `signal` call's points, kept for its capacity.
    signals: Vec<NvRmSyncPoint>,
    /// Completion point of the last accepted submission.
    last_token: Option<NvRmToken>,
}

static CTX_OPS: nvkmd_ctx_ops = nvkmd_ctx_ops {
    destroy: Some(ctx_destroy),
    wait: Some(ctx_wait),
    exec: Some(ctx_exec),
    bind: Some(ctx_bind),
    signal: Some(ctx_signal),
    flush: Some(ctx_flush),
    sync: Some(ctx_sync),
};

/// A borrowed list from C as a slice: `count` elements at `items`, which
/// may be NULL only when `count` is 0.
///
/// # Safety
/// `items` is readable for `count` elements, not written during `'a`.
unsafe fn list<'a, T>(items: *const T, count: u32) -> &'a [T] {
    if count == 0 {
        return &[];
    }
    let items = NonNull::new(items.cast_mut()).expect("Mesa passes a list with its count");
    // SAFETY: readable for `count` elements and not written during `'a`
    // (caller); `NonNull` and aligned as Mesa's array of `T`.
    unsafe { core::slice::from_raw_parts(items.as_ptr(), usize_from_u32(count)) }
}

/// A list's pointer for the RM: NULL when empty, as the RM's contract
/// allows only then.
fn list_ptr<T>(items: &[T]) -> *const T {
    if items.is_empty() {
        core::ptr::null()
    } else {
        items.as_ptr()
    }
}

/// A list's length for the RM. Pushes are cut to at most
/// [`push::MAX_PUSHES`] per submission; waits and signals come from Mesa's
/// `u32` counts.
fn list_len<T>(items: &[T]) -> u32 {
    u32::try_from(items.len()).expect("submission lists hold at most u32::MAX entries")
}

impl RmCtx {
    /// Our context behind Mesa's pointer, or `None` for NULL or another
    /// backend's.
    ///
    /// # Safety
    /// `base` is NULL or points to a live `nvkmd_ctx` that no other call uses
    /// while the reference lives (Mesa serialises calls per context).
    unsafe fn from_base<'a>(base: *mut nvkmd_ctx) -> Option<&'a mut Self> {
        // SAFETY: NULL or a live nvkmd_ctx (caller).
        let ctx = unsafe { base.as_ref() }?;
        if !core::ptr::eq(ctx.ops, &raw const CTX_OPS) {
            return None;
        }
        // SAFETY: a ctx with our ops is the first field of a live `RmCtx`,
        // not used by another call meanwhile (caller).
        Some(unsafe { &mut *base.cast::<Self>() })
    }

    /// The context's device.
    fn dev(&self) -> &RmDev {
        dev_of(&self.base)
    }

    /// Submits `pushes` with the pending waits and `signals` as one batch,
    /// waiting out ring pressure; on success the waits are consumed and the
    /// batch's token is the context's last (`nvkmd_rm.c` `rm_ctx_submit`).
    fn submit(
        &mut self,
        log: *mut vk_object_base,
        channel: NvRmContextId,
        pushes: core::ops::Range<usize>,
        signals: &[NvRmSyncPoint],
    ) -> VkResult {
        let rm = self.dev().rm();
        let pushes = &self.pending[pushes];
        let mut token = NvRmToken::default();
        let status = loop {
            // SAFETY: the device's live RM device; every list is readable for
            // its count during the call; `token` is a writable local.
            let status = unsafe {
                nv_rm_submit_batch(
                    rm,
                    channel,
                    list_ptr(pushes),
                    list_len(pushes),
                    list_ptr(&self.waits),
                    list_len(&self.waits),
                    list_ptr(signals),
                    list_len(signals),
                    &raw mut token,
                )
            };
            if status != VK_NOT_READY {
                break status;
            }
            // Ring pressure: the RM accepted nothing and names its oldest
            // batch. The wait terminates: the device runs Mesa's
            // THREADED_ON_DEMAND submit mode, so every wait this backend
            // receives was VK_SYNC_WAIT_PENDING (its signal submitted or
            // host-signalled) when submitted, and the batches of all rings
            // form a DAG in submission order (vk_queue.c
            // vk_queue_submit_final). Only imported sync_files depend on
            // producers outside the device. Age alone is not loss.
            // SAFETY: the device's live RM device.
            let status = unsafe { nv_rm_wait(rm, channel, token, u64::MAX) };
            if status != VK_SUCCESS {
                break status;
            }
        };
        if status != VK_SUCCESS {
            return rm_result(log, status, "submit asynchronous work");
        }
        self.waits.clear();
        // Token 0 names no batch; `sync` then has nothing to wait for.
        self.last_token = Some(token).filter(|token| token.get() != 0);
        VK_SUCCESS
    }

    /// Submits the pending pushes and waits, `signals` on the last
    /// submission; a call with no pushes still submits when it has waits or
    /// signals (`nvkmd_rm.c` `rm_ctx_flush_signals`).
    fn flush_signals(&mut self, log: *mut vk_object_base, signals: &[NvRmSyncPoint]) -> VkResult {
        let Some(channel) = self.channel else {
            // A bind context never holds pushes or GPU waits (`exec` refuses
            // them, `wait` waits on the CPU): nothing to submit.
            return VK_SUCCESS;
        };
        let total = self.pending.len();
        // Every cut is checked before any work is accepted.
        if let Err(PushError::ChainTooLong { .. }) =
            push::check_chunks(total, |i| push::is_incomplete(&self.pending[i]))
        {
            return unsupported(log, "push continuation chain exceeding 1000 entries");
        }
        let mut result = VK_SUCCESS;
        if total == 0 && (!signals.is_empty() || !self.waits.is_empty()) {
            result = self.submit(log, channel, 0..0, signals);
        }
        let mut offset = 0;
        while offset < total {
            let count = push::first_chunk(total - offset, |i| {
                push::is_incomplete(&self.pending[offset + i])
            })
            .expect("every cut was checked above");
            let last = offset + count == total;
            result = self.submit(
                log,
                channel,
                offset..offset + count,
                if last { signals } else { &[] },
            );
            if result != VK_SUCCESS {
                break;
            }
            offset += count;
        }
        // A later submission may fail after earlier ones were accepted: keep
        // only the unaccepted suffix, so a retry never replays accepted work.
        self.pending.drain(..offset);
        result
    }

    fn flush(&mut self, log: *mut vk_object_base) -> VkResult {
        self.flush_signals(log, &[])
    }
}

/// The device of a context's base (a borrow of the base alone, so the
/// context's lists stay usable alongside it).
fn dev_of(base: &nvkmd_ctx) -> &RmDev {
    // SAFETY: Mesa destroys a context before its device; the pointer is the
    // one `create` stored.
    unsafe { RmDev::from_base(base.dev) }.expect("a context's device is ours")
}

/// Runs a context entry inside the panic guard; a panic is the device's loss
/// to the caller.
fn guarded(entry: &'static str, body: impl FnOnce() -> VkResult) -> VkResult {
    nv_ffi::guard::<Diag, VkResult>(entry, None, || VK_ERROR_DEVICE_LOST, body)
}

/// The context of an entry.
///
/// # Safety
/// As [`RmCtx::from_base`]; Mesa calls a context's ops with that context.
unsafe fn ctx_of<'a>(base: *mut nvkmd_ctx) -> &'a mut RmCtx {
    // SAFETY: the caller's contract.
    unsafe { RmCtx::from_base(base) }.expect("Mesa calls its own backend's contexts")
}

/// `create_ctx` and `create_shared_ctx` after their own checks
/// (`nvkmd_rm.c` `rm_ctx_create_in`).
///
/// # Safety
/// `base` is a live device, `log` a live Vulkan object, `share` NULL or a
/// live context, `out` writable for one pointer.
unsafe fn create(
    base: *mut nvkmd_dev,
    log: *mut vk_object_base,
    engines: nvkmd_engines,
    share: *mut nvkmd_ctx,
    out: *mut *mut nvkmd_ctx,
) -> VkResult {
    let out = NonNull::new(out).expect("Mesa passes a writable context output");
    // SAFETY: a live device (contract).
    let dev = unsafe { RmDev::from_base(base) }.expect("Mesa calls its own backend's devices");
    let share = if share.is_null() {
        None
    } else {
        // SAFETY: a live context (contract), not used by another call while
        // this one reads it.
        let channel = unsafe { RmCtx::from_base(share) }
            .filter(|share| core::ptr::eq(share.base.dev, base))
            .and_then(|share| share.channel);
        match channel {
            Some(channel) if engines != NVKMD_ENGINE_BIND => Some(channel),
            // A shared context needs a channel of its own in the other
            // context's TSG; a bind-only context or share has none.
            Some(_) | None => {
                return vk_errorf(
                    log,
                    VK_ERROR_INITIALIZATION_FAILED,
                    format_args!(
                        "Native Rust RM shared context: the share must be a channel context of the same device"
                    ),
                );
            }
        }
    };
    // SAFETY: `log` is a live Vulkan object (contract).
    let Some(vk) = (unsafe { logical_device(log) }) else {
        diag(format_args!(
            "nvkmd_dev.create_ctx: the log object has no logical device"
        ));
        return VK_ERROR_INITIALIZATION_FAILED;
    };
    let Some(storage) = try_alloc_zeroed::<RmCtx>() else {
        return vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY);
    };
    let channel = if engines == NVKMD_ENGINE_BIND {
        None
    } else {
        let mut channel = NvRmContextId::default();
        let (status, operation) = match share {
            // SAFETY: the device's live RM device; `channel` is a writable local.
            Some(share) => (
                unsafe { nv_rm_context_create_shared(dev.rm(), engines, share, &raw mut channel) },
                "create shared-TSG context",
            ),
            // SAFETY: as above.
            None => (
                unsafe { nv_rm_context_create(dev.rm(), engines, &raw mut channel) },
                "create context",
            ),
        };
        let result = rm_result(log, status, operation);
        if result != VK_SUCCESS {
            drop(storage);
            return result;
        }
        Some(channel)
    };
    let ctx = Box::write(
        storage,
        RmCtx {
            base: nvkmd_ctx {
                ops: &raw const CTX_OPS,
                dev: base,
            },
            vk,
            channel,
            pending: Vec::new(),
            waits: Vec::new(),
            signals: Vec::new(),
            last_token: None,
        },
    );
    // SAFETY: writable (contract).
    unsafe { out.write(Box::into_raw(ctx).cast()) };
    VK_SUCCESS
}

/// The device's `create_ctx`: a context for `engines`, with its own RM
/// context unless it is bind-only.
///
/// # Safety
/// Mesa's `nvkmd_dev_ops.create_ctx` contract: a live device, a live log
/// object, `out` writable.
pub(super) unsafe extern "C" fn dev_create_ctx(
    base: *mut nvkmd_dev,
    log: *mut vk_object_base,
    engines: nvkmd_engines,
    out: *mut *mut nvkmd_ctx,
) -> VkResult {
    guarded("nvkmd_dev.create_ctx", || {
        // SAFETY: the contract; no share.
        unsafe { create(base, log, engines, core::ptr::null_mut(), out) }
    })
}

/// The device's `create_shared_ctx`: a context whose channel joins the TSG
/// of `share`'s.
///
/// # Safety
/// Mesa's `nvkmd_dev_ops.create_shared_ctx` contract: as
/// [`dev_create_ctx`], `share` NULL or a live context.
pub(super) unsafe extern "C" fn dev_create_shared_ctx(
    base: *mut nvkmd_dev,
    log: *mut vk_object_base,
    engines: nvkmd_engines,
    share: *mut nvkmd_ctx,
    out: *mut *mut nvkmd_ctx,
) -> VkResult {
    guarded("nvkmd_dev.create_shared_ctx", || {
        if share.is_null() {
            return vk_errorf(
                log,
                VK_ERROR_INITIALIZATION_FAILED,
                format_args!("Native Rust RM shared context without a share"),
            );
        }
        // SAFETY: the contract.
        unsafe { create(base, log, engines, share, out) }
    })
}

/// The context's `destroy`: unsubmitted work is discarded (Vulkan makes
/// idling explicit), and the RM context is destroyed by the RM's bounded
/// path, never an unbounded dependency wait here. A failure cannot be
/// returned and is reported.
///
/// # Safety
/// A context `create_ctx` returned, not used again.
unsafe extern "C" fn ctx_destroy(base: *mut nvkmd_ctx) {
    nv_ffi::guard_or_abort::<Diag, ()>("nvkmd_ctx.destroy", || {
        // SAFETY: a live context (contract).
        unsafe { ctx_of(base) };
        // SAFETY: a ctx with our ops is the first field of an `RmCtx` that
        // `create` leaked with Box::into_raw; this is its one destroy.
        let ctx = unsafe { Box::from_raw(base.cast::<RmCtx>()) };
        if let Some(channel) = ctx.channel {
            // SAFETY: the device's live RM device.
            let status = unsafe { nv_rm_context_destroy(ctx.dev().rm(), channel) };
            if status != VK_SUCCESS {
                diag(format_args!(
                    "Native Rust RM bounded context destroy failed: {status}"
                ));
            }
        }
        drop(ctx);
    });
}

/// The context's `wait`: work already pending is flushed first, so the
/// waits precede only later work; consecutive waits share one acquire
/// prefix. A bind context waits on the CPU instead, since its VA changes run
/// on the CPU (`nvkmd_rm.c` `rm_ctx_wait`).
///
/// # Safety
/// Mesa's `nvkmd_ctx_ops.wait` contract: `waits` readable for `count`
/// entries, each a live sync.
unsafe extern "C" fn ctx_wait(
    base: *mut nvkmd_ctx,
    log: *mut vk_object_base,
    count: u32,
    waits: *const vk_sync_wait,
) -> VkResult {
    guarded("nvkmd_ctx.wait", || {
        // SAFETY: the contract.
        let ctx = unsafe { ctx_of(base) };
        if !ctx.pending.is_empty() {
            let result = ctx.flush(log);
            if result != VK_SUCCESS {
                return result;
            }
        }
        if ctx.channel.is_none() {
            // SAFETY: the context's live logical device; `waits` as the
            // contract says.
            return unsafe {
                vk_sync_wait_many(
                    ctx.vk.as_ptr(),
                    count,
                    waits,
                    VK_SYNC_WAIT_COMPLETE,
                    u64::MAX,
                )
            };
        }
        // SAFETY: readable for `count` entries (contract).
        let waits = unsafe { list(waits, count) };
        if ctx.waits.try_reserve(waits.len()).is_err() {
            return vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY);
        }
        let dev = dev_of(&ctx.base);
        for item in waits {
            let sync = NonNull::new(item.sync).expect("Mesa passes a sync per wait");
            // SAFETY: a live sync (contract).
            let timeline = unsafe { is_timeline(sync) };
            // SAFETY: as above.
            let id = unsafe { payload(dev, sync) };
            let wait = Wait {
                sync: id.map_or(WaitSync::Foreign, WaitSync::Rm),
                timeline,
                value: item.wait_value,
            };
            match wait::acquire(wait) {
                Ok(Some(point)) => ctx.waits.push(point),
                Ok(None) => {}
                Err(WaitError::Foreign) => {
                    return unsupported(log, "non-RM synchronization payload");
                }
            }
        }
        VK_SUCCESS
    })
}

/// The context's `exec`: appends pushes to the pending list, all or none: a
/// refused push leaves the list as it was, so a valid prefix ending on an
/// incomplete push cannot leave a chain no flush can cut (`nvkmd_rm.c`
/// `rm_ctx_exec`).
///
/// # Safety
/// Mesa's `nvkmd_ctx_ops.exec` contract: `execs` readable for `count`
/// entries.
unsafe extern "C" fn ctx_exec(
    base: *mut nvkmd_ctx,
    log: *mut vk_object_base,
    count: u32,
    execs: *const nvkmd_ctx_exec,
) -> VkResult {
    guarded("nvkmd_ctx.exec", || {
        // SAFETY: the contract.
        let ctx = unsafe { ctx_of(base) };
        if ctx.channel.is_none() {
            // NVK never pushes to a bind context; it has no channel to run
            // them on.
            return unsupported(log, "pushes on a bind-only context");
        }
        // SAFETY: readable for `count` entries (contract).
        let execs = unsafe { list(execs, count) };
        if ctx.pending.try_reserve(execs.len()).is_err() {
            return vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY);
        }
        let before = ctx.pending.len();
        for item in execs {
            let exec = push::exec(Exec {
                addr: item.addr,
                size: item.size_B,
                incomplete: item.incomplete,
                no_prefetch: item.no_prefetch,
            });
            match exec {
                Ok(push) => ctx.pending.push(push),
                Err(ExecError::InvalidRange { .. }) => {
                    ctx.pending.truncate(before);
                    return unsupported(log, "invalid push range");
                }
            }
        }
        VK_SUCCESS
    })
}

/// The context's `bind`: pending work is flushed, then each VA change runs
/// on the CPU through Mesa's wrappers, in order, stopping at the first
/// failure (`nvkmd_rm.c` `rm_ctx_bind`).
///
/// # Safety
/// Mesa's `nvkmd_ctx_ops.bind` contract: `binds` readable for `count`
/// entries, each naming live reservations and memory.
unsafe extern "C" fn ctx_bind(
    base: *mut nvkmd_ctx,
    log: *mut vk_object_base,
    count: u32,
    binds: *const nvkmd_ctx_bind,
) -> VkResult {
    guarded("nvkmd_ctx.bind", || {
        // SAFETY: the contract.
        let ctx = unsafe { ctx_of(base) };
        let result = ctx.flush(log);
        if result != VK_SUCCESS {
            return result;
        }
        // SAFETY: readable for `count` entries (contract).
        let binds = unsafe { list(binds, count) };
        for bind in binds {
            let result = match bind.op {
                // SAFETY: live reservation and memory (contract).
                NVKMD_BIND_OP_BIND => unsafe {
                    nvkmd_va_bind_mem(
                        bind.va,
                        log,
                        bind.va_offset_B,
                        bind.mem,
                        bind.mem_offset_B,
                        bind.range_B,
                    )
                },
                // SAFETY: a live reservation (contract).
                NVKMD_BIND_OP_UNBIND => unsafe {
                    nvkmd_va_unbind(bind.va, log, bind.va_offset_B, bind.range_B)
                },
                other => {
                    return unsupported(log, &format!("bind operation {other}"));
                }
            };
            if result != VK_SUCCESS {
                return result;
            }
        }
        VK_SUCCESS
    })
}

/// The context's `signal`: a channel context submits the pending work with
/// these signals on its last batch (one batch even without work); every
/// signal is checked before anything is submitted. A bind context flushes
/// and signals on the CPU (`nvkmd_rm.c` `rm_ctx_signal`).
///
/// # Safety
/// Mesa's `nvkmd_ctx_ops.signal` contract: `signals` readable for `count`
/// entries, each a live sync.
unsafe extern "C" fn ctx_signal(
    base: *mut nvkmd_ctx,
    log: *mut vk_object_base,
    count: u32,
    signals: *const vk_sync_signal,
) -> VkResult {
    guarded("nvkmd_ctx.signal", || {
        // SAFETY: the contract.
        let ctx = unsafe { ctx_of(base) };
        // SAFETY: readable for `count` entries (contract).
        let signals = unsafe { list(signals, count) };
        if ctx.channel.is_none() {
            let result = ctx.flush(log);
            if result != VK_SUCCESS {
                return result;
            }
            for signal in signals {
                // SAFETY: the context's live logical device and a live sync
                // (contract).
                let result =
                    unsafe { vk_sync_signal(ctx.vk.as_ptr(), signal.sync, signal.signal_value) };
                if result != VK_SUCCESS {
                    return result;
                }
            }
            return VK_SUCCESS;
        }
        if signals.is_empty() {
            return ctx.flush(log);
        }
        let mut points = core::mem::take(&mut ctx.signals);
        points.clear();
        let result = if points.try_reserve(signals.len()).is_err() {
            vk_error(log, VK_ERROR_OUT_OF_HOST_MEMORY)
        } else {
            let dev = dev_of(&ctx.base);
            let mut foreign = false;
            for signal in signals {
                let sync = NonNull::new(signal.sync).expect("Mesa passes a sync per signal");
                // SAFETY: a live sync (contract).
                let Some(sync) = (unsafe { payload(dev, sync) }) else {
                    foreign = true;
                    break;
                };
                points.push(NvRmSyncPoint {
                    sync,
                    value: signal.signal_value,
                });
            }
            if foreign {
                unsupported(log, "non-RM synchronization signal")
            } else {
                ctx.flush_signals(log, &points)
            }
        };
        ctx.signals = points;
        result
    })
}

/// The context's `flush`: submits pending work and waits.
///
/// # Safety
/// Mesa's `nvkmd_ctx_ops.flush` contract.
unsafe extern "C" fn ctx_flush(base: *mut nvkmd_ctx, log: *mut vk_object_base) -> VkResult {
    guarded("nvkmd_ctx.flush", || {
        // SAFETY: the contract.
        unsafe { ctx_of(base) }.flush(log)
    })
}

/// The context's `sync`: flushes, then waits on the caller's thread for the
/// last accepted submission. Unbounded as Mesa asks (`nvkmd.c`
/// `NVK_DEBUG_PUSH_SYNC`, the only caller): the RM fails the wait when the
/// channel faults.
///
/// # Safety
/// Mesa's `nvkmd_ctx_ops.sync` contract.
unsafe extern "C" fn ctx_sync(base: *mut nvkmd_ctx, log: *mut vk_object_base) -> VkResult {
    guarded("nvkmd_ctx.sync", || {
        // SAFETY: the contract.
        let ctx = unsafe { ctx_of(base) };
        let result = ctx.flush(log);
        let (VK_SUCCESS, Some(channel), Some(token)) = (result, ctx.channel, ctx.last_token) else {
            return result;
        };
        // SAFETY: the device's live RM device.
        let status = unsafe { nv_rm_wait(ctx.dev().rm(), channel, token, u64::MAX) };
        rm_result(log, status, "wait for context completion")
    })
}
