// SPDX-License-Identifier: MIT
//! Safe test-side calls into the `nv_rm_*` C ABI of this module's parent.
//!
//! The hardware fixtures drive the bridge exactly as Mesa does, through the
//! exported C functions. This module is the one place that turns their raw
//! pointer arguments into Rust types, so the fixtures themselves hold no
//! `unsafe`. Each wrapper passes its arguments through unchanged (same ids,
//! counts and values) and returns the C status, so a fixture's assertions are
//! the ones it would make against the C ABI.
//!
//! What a reference cannot prove is checked here instead:
//! - Synchronization ids are `Arc::into_raw` pointers (`nv_rm_sync_create`),
//!   dereferenced by every call that takes one. A [`SyncHandle`] owns one id:
//!   only `sync_create` makes one, only `sync_destroy` (by value) ends it, and
//!   every other call and every [`Point`] borrows it, so the borrow checker
//!   proves no call uses an id after or while it is destroyed. Threads share a
//!   handle as `Arc<SyncHandle>` and destroy it after `Arc::into_inner`.
//! - VA release callbacks run whenever the bridge reclaims (any later call,
//!   or close), so their sink is `&'static`.
//! - CPU mappings are only reachable through [`CpuMap`], which borrows the
//!   bridge, holds the map reference `nv_rm_free` refuses to free under, and
//!   bounds-checks every access.
//!
//! Threading: every function may be called from any thread (the C ABI is
//! thread-safe per bridge). No wrapper takes a lock of its own.
use super::*;
use std::os::fd::{AsRawFd, BorrowedFd, RawFd};
use std::ptr;

/// One synchronization object created by [`sync_create`]: the id is live
/// while this value exists (no Clone/Copy; [`sync_destroy`] consumes it).
/// The object itself holds its slot pool (`Arc<SlotPool>`, syncobj.rs), so a
/// handle stays valid memory even after the bridge that made it is closed.
#[must_use = "a sync object is destroyed only by sync_destroy"]
#[derive(Debug)]
pub(in crate::mesa) struct SyncHandle {
    id: u64,
}
impl SyncHandle {
    /// The payload behind the id, for fixtures that inspect bridge state.
    pub(in crate::mesa) fn obj(&self) -> Result<&crate::syncobj::SyncObj> {
        // SAFETY: the id came from nv_rm_sync_create and is destroyed only by
        // sync_destroy, which consumes this handle; the returned borrow of
        // `self` ends before that can happen, so the payload is not destroyed
        // while the reference lives.
        unsafe { sync_ref(self.id) }
    }
}
/// A (sync, value) pair of a wait or signal list; borrows the handle, so the
/// id stays live for as long as the list exists.
#[derive(Clone, Copy)]
pub(in crate::mesa) struct Point<'a> {
    pub(in crate::mesa) sync: &'a SyncHandle,
    pub(in crate::mesa) value: u64,
}
/// Longest wait or signal list a fixture passes; lists are lowered into a
/// stack array of the C `SyncPoint` (no allocation per call).
const MAX_POINTS: usize = 8;
struct Lowered {
    points: [SyncPoint; MAX_POINTS],
    count: usize,
}
impl Lowered {
    fn new(points: &[Point<'_>]) -> Self {
        assert!(points.len() <= MAX_POINTS, "{} sync points", points.len());
        let mut lowered = Self {
            points: [SyncPoint { sync: 0, value: 0 }; MAX_POINTS],
            count: points.len(),
        };
        for (out, point) in lowered.points.iter_mut().zip(points) {
            *out = SyncPoint {
                sync: point.sync.id,
                value: point.value,
            };
        }
        lowered
    }
    fn get(&self) -> &[SyncPoint] {
        &self.points[..self.count]
    }
}

/// The bridge pointer the C ABI takes. Every `nv_rm_*` entry used here only
/// forms `&Bridge` from it (mesa.rs `bridge()`), so a pointer derived from a
/// shared borrow is enough and no `&mut Bridge` is ever created while other
/// threads use the same bridge.
fn raw(b: &Bridge) -> *mut Bridge {
    ptr::from_ref(b).cast_mut()
}
/// (pointer, count) of an input array. An empty array passes null, the value
/// nv_rm.h documents for a zero count.
fn array<T>(items: &[T]) -> (*const T, u32) {
    let count = u32::try_from(items.len()).expect("fixture arrays are small");
    if items.is_empty() {
        (ptr::null(), 0)
    } else {
        (items.as_ptr(), count)
    }
}
fn fd_or_none(fd: Option<BorrowedFd<'_>>) -> RawFd {
    fd.map_or(-1, |fd| fd.as_raw_fd())
}

pub(in crate::mesa) fn context_create(b: &Bridge, engines: u32, out: &mut u64) -> i32 {
    // SAFETY: `raw(b)` points to a live bridge for the call; `out` is an
    // exclusive u64 the call writes once.
    unsafe { nv_rm_context_create(raw(b), engines, out) }
}
pub(in crate::mesa) fn context_create_shared(
    b: &Bridge,
    engines: u32,
    share: u64,
    out: &mut u64,
) -> i32 {
    // SAFETY: as context_create; `share` is a context id, looked up in the
    // bridge's queue map (never dereferenced).
    unsafe { nv_rm_context_create_shared(raw(b), engines, share, out) }
}
pub(in crate::mesa) fn context_destroy(b: &Bridge, ctx: u64) -> i32 {
    // SAFETY: live bridge; context ids are map keys.
    unsafe { nv_rm_context_destroy(raw(b), ctx) }
}
pub(in crate::mesa) fn check_status(b: &Bridge) -> i32 {
    // SAFETY: live bridge; no other argument.
    unsafe { nv_rm_check_status(raw(b)) }
}
pub(in crate::mesa) fn wait(b: &Bridge, ctx: u64, token: u64, timeout_ns: u64) -> i32 {
    // SAFETY: live bridge; ids and token are plain values.
    unsafe { nv_rm_wait(raw(b), ctx, token, timeout_ns) }
}
pub(in crate::mesa) fn poll(b: &Bridge, ctx: u64, token: u64, complete: &mut u32) -> i32 {
    // SAFETY: live bridge; `complete` is an exclusive u32 the call writes.
    unsafe { nv_rm_poll(raw(b), ctx, token, complete) }
}

pub(in crate::mesa) fn alloc(
    b: &Bridge,
    size: u64,
    alignment: u64,
    placement: u32,
    kind: u32,
    tile: u32,
    out: &mut Allocation,
) -> i32 {
    // SAFETY: live bridge; `out` is an exclusive Allocation the call writes.
    unsafe { nv_rm_alloc(raw(b), size, alignment, placement, kind, tile, out) }
}
pub(in crate::mesa) fn bind(
    b: &Bridge,
    memory: u64,
    offset: u64,
    size: u64,
    address: u64,
    kind: u32,
    out: &mut u64,
) -> i32 {
    // SAFETY: live bridge; memory ids are map keys; `out` is an exclusive
    // u64 the call writes.
    unsafe { nv_rm_bind(raw(b), memory, offset, size, address, kind, out) }
}
pub(in crate::mesa) fn unbind(b: &Bridge, mapping: u64) -> i32 {
    // SAFETY: live bridge; mapping ids are map keys.
    unsafe { nv_rm_unbind(raw(b), mapping) }
}
pub(in crate::mesa) fn free(b: &Bridge, memory: u64) -> i32 {
    // SAFETY: live bridge; memory ids are map keys. The call refuses memory
    // that is still CPU-mapped, so no live CpuMap is invalidated by it.
    unsafe { nv_rm_free(raw(b), memory) }
}

/// Receiver of deferred VA releases (Mesa's util_vma_heap_free stands here).
pub(in crate::mesa) trait ReleaseSink: Sync + 'static {
    fn release(&self, address: u64, size: u64);
}
/// # Safety
/// `data` is a `&'static T` cast to a pointer by [`va_free`].
unsafe extern "C" fn release_to<T: ReleaseSink>(data: *mut c_void, address: u64, size: u64) {
    // SAFETY: va_free passes `ptr::from_ref::<T>(sink)` with this
    // instantiation, and the sink is 'static, so it outlives every deferred
    // release, including the ones bridge close runs.
    let sink = unsafe { &*data.cast_const().cast::<T>() };
    sink.release(address, size);
}
pub(in crate::mesa) fn va_free<T: ReleaseSink>(
    b: &Bridge,
    mappings: &[u64],
    address: u64,
    size: u64,
    sink: &'static T,
) -> i32 {
    let (ids, count) = array(mappings);
    // SAFETY: live bridge; `ids` is valid for `count` u64s (or null for 0);
    // the callback/data pair is release_to::<T> with a 'static T, sound
    // whenever and however often the bridge runs it.
    unsafe {
        nv_rm_va_free(
            raw(b),
            ids,
            count,
            address,
            size,
            Some(release_to::<T>),
            ptr::from_ref(sink).cast_mut().cast(),
        )
    }
}

/// A CPU mapping of one RM allocation, valid while this value lives: it holds
/// the map reference taken by `nv_rm_map` (the mapping cache never evicts a
/// referenced mapping, cpu_map_cache.rs `evict`), `nv_rm_free` refuses the
/// memory while it is referenced, and the borrow keeps the bridge open.
/// GPU-visible memory: every access is volatile, nothing hands out a reference.
pub(in crate::mesa) struct CpuMap<'b> {
    bridge: &'b Bridge,
    memory: u64,
    base: *mut u8,
    len: usize,
}
impl<'b> CpuMap<'b> {
    /// Maps `memory`; panics if `nv_rm_map` fails.
    pub(in crate::mesa) fn new(b: &'b Bridge, memory: u64) -> Self {
        let mut address = 0usize;
        // SAFETY: live bridge; `address` is an exclusive usize the call writes.
        let status = unsafe { nv_rm_map(raw(b), memory, &mut address) };
        assert_eq!(status, 0, "nv_rm_map");
        // The mapping covers the whole allocation, rounded up to CPU pages
        // (rm.rs Memory::map_once), so the allocation size bounds it.
        let len = b
            .lock()
            .0
            .buffers
            .get(&memory)
            .expect("mapped memory is known")
            .memory
            .size;
        let base = ptr::with_exposed_provenance_mut::<u8>(address);
        assert!(
            !base.is_null() && base.align_offset(8) == 0,
            "mapping base {address:#x}"
        );
        Self {
            bridge: b,
            memory,
            base,
            len,
        }
    }
    fn element<T>(&self, index: usize) -> *mut T {
        let offset = index
            .checked_mul(size_of::<T>())
            .filter(|offset| {
                offset
                    .checked_add(size_of::<T>())
                    .is_some_and(|end| end <= self.len)
            })
            .unwrap_or_else(|| panic!("index {index} outside the {}-byte mapping", self.len));
        self.base.wrapping_add(offset).cast()
    }
    pub(in crate::mesa) fn write_u32(&self, index: usize, value: u32) {
        let p = self.element::<u32>(index);
        // SAFETY: `p` is in bounds (element checks offset + 4 <= len) and 4-aligned (8-aligned base,
        // index * 4); the mapping is live while `self` is (type doc). The GPU
        // may access it, hence volatile and no reference.
        unsafe { p.write_volatile(value) };
    }
    pub(in crate::mesa) fn write_u64(&self, index: usize, value: u64) {
        let p = self.element::<u64>(index);
        // SAFETY: as write_u32, 8-aligned.
        unsafe { p.write_volatile(value) };
    }
    /// Mapped length in bytes (the allocation size).
    pub(in crate::mesa) fn len(&self) -> usize {
        self.len
    }
    /// Writes `first`, `first + 1`, ... to every u64 of the mapping in order
    /// (one bounds check for the run; BAR1 write-traffic generator).
    pub(in crate::mesa) fn fill_u64_counting(&self, first: u64) -> u64 {
        let count = self.len / size_of::<u64>();
        let words = self.base.cast::<u64>();
        let mut value = first;
        for i in 0..count {
            let p = words.wrapping_add(i);
            // SAFETY: i < len / 8, so the 8 bytes at i * 8 are inside the
            // mapping; 8-aligned base; live while `self` is; volatile because
            // the GPU may access it.
            unsafe { p.write_volatile(value) };
            value = value.wrapping_add(1);
        }
        value
    }
    pub(in crate::mesa) fn read_u64(&self, index: usize) -> u64 {
        let p = self.element::<u64>(index);
        // SAFETY: as write_u64; the GPU may have written it, hence volatile.
        unsafe { p.read_volatile() }
    }
    /// Drops the map reference; the C status of `nv_rm_unmap`.
    pub(in crate::mesa) fn unmap(self) -> i32 {
        // SAFETY: live bridge; this value is consumed, so no access through
        // `base` follows the unmap.
        unsafe { nv_rm_unmap(raw(self.bridge), self.memory) }
    }
}

/// `Ok(handle)` on status 0, else `Err(status)`.
pub(in crate::mesa) fn sync_create(
    b: &Bridge,
    flags: u32,
    initial: u64,
) -> std::result::Result<SyncHandle, i32> {
    let mut id = 0;
    // SAFETY: live bridge; `id` is an exclusive u64 the call writes.
    let status = unsafe { nv_rm_sync_create(raw(b), flags, initial, &mut id) };
    if status == 0 {
        Ok(SyncHandle { id })
    } else {
        Err(status)
    }
}
pub(in crate::mesa) fn sync_destroy(b: &Bridge, sync: SyncHandle) -> i32 {
    // SAFETY: the handle owns a live id (SyncHandle doc); it is consumed
    // here, so nothing can pass the id again.
    unsafe { nv_rm_sync_destroy(raw(b), sync.id) }
}
pub(in crate::mesa) fn sync_reset(b: &Bridge, sync: &SyncHandle) -> i32 {
    // SAFETY: live bridge; the borrowed handle keeps the id live.
    unsafe { nv_rm_sync_reset(raw(b), sync.id) }
}
pub(in crate::mesa) fn sync_signal(b: &Bridge, sync: &SyncHandle, value: u64) -> i32 {
    // SAFETY: live bridge; the borrowed handle keeps the id live.
    unsafe { nv_rm_sync_signal(raw(b), sync.id, value) }
}
pub(in crate::mesa) fn sync_move(
    b: &Bridge,
    dst: &SyncHandle,
    src: &SyncHandle,
    shared_source: u32,
) -> i32 {
    // SAFETY: live bridge; both borrowed handles keep their ids live.
    unsafe { nv_rm_sync_move(raw(b), dst.id, src.id, shared_source) }
}
pub(in crate::mesa) fn sync_get_value(b: &Bridge, sync: &SyncHandle, out: &mut u64) -> i32 {
    // SAFETY: live bridge; borrowed handle; `out` is an exclusive u64.
    unsafe { nv_rm_sync_get_value(raw(b), sync.id, out) }
}
pub(in crate::mesa) fn sync_status_many(
    b: &Bridge,
    points: &[Point<'_>],
    flags: u32,
    out: &mut u32,
) -> i32 {
    let lowered = Lowered::new(points);
    let (points, count) = array(lowered.get());
    // SAFETY: live bridge; `points` valid for `count` (or null for 0), every
    // id in it borrowed from a live handle; `out` is an exclusive u32.
    unsafe { nv_rm_sync_status_many(raw(b), points, count, flags, out) }
}
pub(in crate::mesa) fn sync_wait_many(
    b: &Bridge,
    points: &[Point<'_>],
    flags: u32,
    timeout_ns: u64,
) -> i32 {
    let lowered = Lowered::new(points);
    let (points, count) = array(lowered.get());
    // SAFETY: live bridge; `points` valid for `count` (or null for 0), every
    // id in it borrowed from a live handle for the whole (blocking) call.
    unsafe { nv_rm_sync_wait_many(raw(b), points, count, flags, timeout_ns) }
}
pub(in crate::mesa) fn sync_import_fd(
    b: &Bridge,
    sync: &SyncHandle,
    fd: BorrowedFd<'_>,
    opaque: u32,
) -> i32 {
    // SAFETY: live bridge; borrowed handle; `fd` is open for the call.
    unsafe { nv_rm_sync_import_fd(raw(b), sync.id, fd.as_raw_fd(), opaque) }
}
/// `drm` None passes -1 (opaque exports take no DRM fd). The returned
/// descriptor in `out` (or -1) is the caller's to own.
pub(in crate::mesa) fn sync_export_fd(
    b: &Bridge,
    sync: &SyncHandle,
    opaque: u32,
    drm: Option<BorrowedFd<'_>>,
    out: &mut i32,
) -> i32 {
    // SAFETY: live bridge; borrowed handle; `drm` open for the call or -1;
    // `out` is an exclusive i32.
    unsafe { nv_rm_sync_export_fd(raw(b), sync.id, opaque, fd_or_none(drm), out) }
}

pub(in crate::mesa) fn submit_batch(
    b: &Bridge,
    ctx: u64,
    pushes: &[Push],
    waits: &[Point<'_>],
    signals: &[Point<'_>],
    out: &mut u64,
) -> i32 {
    let (waits, signals) = (Lowered::new(waits), Lowered::new(signals));
    let (pushes, count) = array(pushes);
    let (waits, wait_count) = array(waits.get());
    let (signals, signal_count) = array(signals.get());
    // SAFETY: live bridge; each array valid for its count (or null for 0),
    // every sync id in them borrowed from a live handle; `out` is an
    // exclusive u64.
    unsafe {
        nv_rm_submit_batch(
            raw(b),
            ctx,
            pushes,
            count,
            waits,
            wait_count,
            signals,
            signal_count,
            out,
        )
    }
}
pub(in crate::mesa) fn submit_sync(
    b: &Bridge,
    ctx: u64,
    pushes: &[Push],
    waits: &[Point<'_>],
    out: &mut u64,
) -> i32 {
    let waits = Lowered::new(waits);
    let (pushes, count) = array(pushes);
    let (waits, wait_count) = array(waits.get());
    // SAFETY: as submit_batch.
    unsafe { nv_rm_submit_sync(raw(b), ctx, pushes, count, waits, wait_count, out) }
}
pub(in crate::mesa) fn submit(b: &Bridge, ctx: u64, pushes: &[Push], out: &mut u64) -> i32 {
    let (pushes, count) = array(pushes);
    // SAFETY: live bridge; `pushes` valid for `count` (or null for 0); `out`
    // is an exclusive u64.
    unsafe { nv_rm_submit(raw(b), ctx, pushes, count, out) }
}
pub(in crate::mesa) fn context_signal(
    b: &Bridge,
    ctx: u64,
    signals: &[Point<'_>],
    out: &mut u64,
) -> i32 {
    let signals = Lowered::new(signals);
    let (signals, count) = array(signals.get());
    // SAFETY: as submit_batch.
    unsafe { nv_rm_context_signal(raw(b), ctx, signals, count, out) }
}
/// `waits` are sync_file descriptors, borrowed for the call (the bridge dups
/// the ones it keeps).
pub(in crate::mesa) fn submit_waits(
    b: &Bridge,
    ctx: u64,
    pushes: &[Push],
    drm: BorrowedFd<'_>,
    waits: &[BorrowedFd<'_>],
    out: &mut u64,
) -> i32 {
    let (pushes, count) = array(pushes);
    let (fds, wait_count) = array(waits);
    // SAFETY: live bridge; `pushes` valid for `count`; `fds` points to
    // `wait_count` BorrowedFd values, which are repr(transparent) over RawFd
    // (std::os::fd::BorrowedFd docs), so it reads as that many open i32
    // descriptors; `out` is an exclusive u64.
    unsafe {
        nv_rm_submit_waits(
            raw(b),
            ctx,
            pushes,
            count,
            drm.as_raw_fd(),
            fds.cast::<RawFd>(),
            wait_count,
            out,
        )
    }
}
