// SPDX-License-Identifier: MIT
//! The world the backend under test sees: a fake RM (`nv_rm_*`), a fake render
//! node behind libc's `open`/`fstat`/`close` and libdrm's version query, a
//! counting `calloc`/`free` that can fail one chosen call, and Mesa's
//! diagnostic sinks. The libc, libdrm and Mesa functions are replaced with the
//! linker's `--wrap` (the backend and Mesa reach `__wrap_<name>`; the real
//! function stays reachable as `__real_<name>`), so the same objects run here
//! as in `libvulkan_nouveau.so`.
//!
//! Every RM, render-node and libdrm call is appended to the call log, which the
//! runner compares byte for byte between the C and the Rust backend. RM entries
//! the suites do not expect abort the harness.
//!
//! Threading: the scenario state is one global behind [`state`]; the
//! allocation table has its own lock because Rust's own deallocations reach
//! [`__wrap_free`] while the scenario state is locked. Neither lock is held
//! while calling into the backend or Mesa.
use std::ffi::{CStr, VaList, c_char, c_int, c_void};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use nv_rm_abi::{
    NV_RM_VA_END, NV_RM_VA_START, NvRmAllocation, NvRmDevice, NvRmInfo, NvRmMappingId,
    NvRmMemoryId, NvRmVaRelease, NvRmZcullInfo,
};
use nvkmd_sys::{
    _drmVersion, DRM_CAP_SYNCOBJ_TIMELINE, NVKMD_SYS_VK_DEVICE_CHECK_STATUS,
    NVKMD_SYS_VK_DEVICE_NOTIFY_LOST, NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA,
    VK_ERROR_INITIALIZATION_FAILED, VK_SUCCESS, VkResult, drmVersionPtr, stat, vk_device,
};

use crate::mesa;

/// The RM device handle the fake hands out. Never dereferenced: the backend
/// only passes it back.
pub(crate) const RM_DEVICE: usize = 42;
/// The descriptor of the fake render node: a real descriptor (of `/dev/null`,
/// held for the whole run) so that owning-descriptor types may check it is
/// open, but every call the backend makes on it is answered here and the fake
/// `close` leaves it open. Opened first thing, so its number is the same in
/// every run.
static RENDER: OnceLock<c_int> = OnceLock::new();

/// Opens the fake render node's descriptor; call before any suite.
pub(crate) fn init() {
    let file = std::fs::File::open("/dev/null").expect("/dev/null opens");
    let fd = std::os::fd::IntoRawFd::into_raw_fd(file);
    assert!(
        RENDER.set(fd).is_ok(),
        "the render node descriptor is opened once"
    );
}

pub(crate) fn render_fd() -> c_int {
    *RENDER.get().expect("fake::init ran before the suites")
}
/// The path the suites put in `drmDevice.nodes[DRM_NODE_RENDER]`.
pub(crate) const RENDER_PATH: &CStr = c"/fake/render";
/// `st_rdev` of the fake render node.
pub(crate) const RENDER_RDEV: u64 = 0x1234;
/// glibc `O_RDWR | O_CLOEXEC`, which both opens of the render node use.
const RENDER_OPEN_FLAGS: c_int = 0o2_000_002;

/// What the fake `drmGetVersion` returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Version {
    /// The NVIDIA DRM driver.
    NvidiaDrm,
    /// `drmGetVersion` fails.
    Missing,
    /// Another DRM driver.
    Nouveau,
    /// A version record without a name.
    Unnamed,
}

/// Scenario knobs and counters. The suites set the knobs, call the backend
/// and check the counters.
#[derive(Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent scenario knobs, each set by the suites on its own"
)]
pub(crate) struct State {
    log: Vec<String>,
    pub(crate) open_error: bool,
    pub(crate) version: Version,
    pub(crate) stat_error: bool,
    pub(crate) rm_error: bool,
    pub(crate) rm_partial: bool,
    /// Applied to what `nv_rm_open` reports.
    pub(crate) report: fn(&mut NvRmInfo),
    pub(crate) opens: u32,
    pub(crate) failed_opens: u32,
    pub(crate) closes: u32,
    pub(crate) rm_opens: u32,
    pub(crate) rm_closes: u32,
    pub(crate) loss_calls: u32,
    pub(crate) rm_lost: bool,
    /// `nv_rm_is_lost` calls since the scenario or case started.
    pub(crate) is_lost_calls: u32,
    /// The messages passed to Mesa's `vk_errorf`/`vk_errorv` since the
    /// scenario or case started, one per line.
    pub(crate) diagnostic: String,
    /// The logical device whose loss hooks `nv_rm_close` checks and whose
    /// loss `nv_rm_mark_lost` must follow (address; 0 = none).
    pub(crate) loss_device: usize,
    /// `nv_rm_mark_lost` without a logical device is expected (physical
    /// device probing).
    pub(crate) allow_no_vk_loss: bool,
    /// The fake RM's memory and VA entries.
    pub(crate) mem: Memory,
    /// The fake RM's sync payload entries and Mesa's clock.
    pub(crate) sync: crate::fake_sync::Syncs,
    /// The fake RM's context entries.
    pub(crate) ctx: crate::fake_ctx::Contexts,
    /// What `drmGetCap(DRM_CAP_SYNCOBJ_TIMELINE)` answers on
    /// [`SYNC_PROVIDER_FD`]; `None` = the query is unexpected.
    pub(crate) timeline_cap: Option<u64>,
    sysfs: Vec<c_int>,
}

/// Knobs and counters of the fake RM's memory and VA entries: the status each
/// returns, what a map returns, and whether a VA free defers its release.
#[derive(Debug)]
pub(crate) struct Memory {
    pub(crate) alloc_status: i32,
    pub(crate) import_status: i32,
    pub(crate) free_status: i32,
    pub(crate) map_status: i32,
    pub(crate) unmap_status: i32,
    pub(crate) export_status: i32,
    pub(crate) bind_status: i32,
    pub(crate) unbind_status: i32,
    pub(crate) va_free_status: i32,
    /// What `nv_rm_map` returns as the CPU address.
    pub(crate) map_address: usize,
    /// `nv_rm_map` publishes the device's loss before returning (an RM whose
    /// rollback failed while it reports the original status).
    pub(crate) map_publishes_loss: bool,
    /// `nv_rm_va_free` with mappings keeps the release for
    /// [`run_deferred_release`] instead of running it.
    pub(crate) defer_release: bool,
    deferred: Option<Release>,
    next_allocation: u64,
    next_mapping: u64,
    /// The handle of the last allocation or import.
    pub(crate) last_allocation: u64,
    pub(crate) allocs: u32,
    pub(crate) frees: u32,
    pub(crate) maps: u32,
    pub(crate) unmaps: u32,
    pub(crate) binds: u32,
    pub(crate) unbinds: u32,
    pub(crate) va_frees: u32,
    /// The mappings, address and size of the last `nv_rm_va_free`.
    pub(crate) last_va_free: (Vec<u64>, u64, u64),
}

/// A release `nv_rm_va_free` deferred.
#[derive(Debug, Clone, Copy)]
struct Release {
    callback: unsafe extern "C" fn(*mut c_void, u64, u64),
    data: usize,
    address: u64,
    size: u64,
}

/// First handle the fake RM gives an allocation; mapping handles start at
/// [`FIRST_MAPPING`].
pub(crate) const FIRST_MEMORY: u64 = 55;
/// First handle the fake RM gives a mapping; allocation handles stay below it.
pub(crate) const FIRST_MAPPING: u64 = 1000;
/// What `nv_rm_map` returns unless a suite says otherwise.
pub(crate) const MAP_ADDRESS: usize = 0x1234_5000;
/// The size of every allocation `nv_rm_import` makes.
pub(crate) const IMPORT_SIZE: u64 = 0x2_0000;
/// The dma-buf `nv_rm_export` hands out.
pub(crate) const EXPORTED_FD: c_int = 123;

impl Memory {
    const fn new() -> Self {
        Self {
            alloc_status: VK_SUCCESS,
            import_status: VK_SUCCESS,
            free_status: VK_SUCCESS,
            map_status: VK_SUCCESS,
            unmap_status: VK_SUCCESS,
            export_status: VK_SUCCESS,
            bind_status: VK_SUCCESS,
            unbind_status: VK_SUCCESS,
            va_free_status: VK_SUCCESS,
            map_address: MAP_ADDRESS,
            map_publishes_loss: false,
            defer_release: false,
            deferred: None,
            next_allocation: FIRST_MEMORY,
            next_mapping: FIRST_MAPPING,
            last_allocation: 0,
            allocs: 0,
            frees: 0,
            maps: 0,
            unmaps: 0,
            binds: 0,
            unbinds: 0,
            va_frees: 0,
            last_va_free: (Vec::new(), 0, 0),
        }
    }
}

impl State {
    const fn new() -> Self {
        Self {
            log: Vec::new(),
            open_error: false,
            version: Version::NvidiaDrm,
            stat_error: false,
            rm_error: false,
            rm_partial: false,
            report: |_| {},
            opens: 0,
            failed_opens: 0,
            closes: 0,
            rm_opens: 0,
            rm_closes: 0,
            loss_calls: 0,
            rm_lost: false,
            is_lost_calls: 0,
            diagnostic: String::new(),
            loss_device: 0,
            allow_no_vk_loss: false,
            mem: Memory::new(),
            sync: crate::fake_sync::Syncs::new(),
            ctx: crate::fake_ctx::Contexts::new(),
            timeline_cap: None,
            sysfs: Vec::new(),
        }
    }

    pub(crate) fn record(&mut self, line: String) {
        self.log.push(line);
    }

    fn report(&mut self, message: &str) {
        self.diagnostic.push_str(message);
        self.diagnostic.push('\n');
    }
}

static STATE: Mutex<State> = Mutex::new(State::new());

/// The scenario state.
pub(crate) fn state() -> MutexGuard<'static, State> {
    STATE
        .lock()
        .expect("no harness code panics while it holds the fake state")
}

/// Starts a scenario: every knob and counter back to its default, no
/// allocation left from the last one. The call log is kept.
pub(crate) fn reset() {
    assert_eq!(
        live_allocations(),
        0,
        "the previous scenario freed everything it allocated"
    );
    let mut state = state();
    let log = std::mem::take(&mut state.log);
    *state = State::new();
    state.log = log;
    CALLOC_FAIL.store(0, Ordering::Relaxed);
    CALLOC_CALLS.store(0, Ordering::Relaxed);
}

/// Starts a case inside a scenario whose devices stay open: the fake RM's
/// memory knobs and counters, the loss counters and the last diagnostic back
/// to their defaults. Handles keep counting up, so every object of a
/// scenario has its own.
pub(crate) fn reset_case() {
    let mut state = state();
    let (memory, mapping) = (state.mem.next_allocation, state.mem.next_mapping);
    state.mem = Memory::new();
    state.sync = state.sync.next_case();
    state.ctx = state.ctx.next_case();
    state.mem.next_allocation = memory;
    state.mem.next_mapping = mapping;
    state.loss_calls = 0;
    state.rm_lost = false;
    state.is_lost_calls = 0;
    state.allow_no_vk_loss = false;
    state.diagnostic.clear();
}

/// Appends a marker line (scenario name, a checked result) to the call log.
pub(crate) fn note(line: impl Into<String>) {
    state().record(line.into());
}

/// Takes the call log recorded so far.
pub(crate) fn take_log() -> Vec<String> {
    std::mem::take(&mut state().log)
}

// --- calloc/free -----------------------------------------------------------

/// Ordinal of the `calloc` to fail (1-based, counted since [`reset`]); 0 =
/// none. Single-threaded scenarios: `Relaxed` orders nothing else.
static CALLOC_FAIL: AtomicU64 = AtomicU64::new(0);
/// `calloc` calls since [`reset`].
static CALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
/// Live `calloc` results, as addresses (0 = free slot). A fixed table: the
/// allocator itself must not allocate.
static LIVE: Mutex<[usize; 128]> = Mutex::new([0; 128]);

fn live() -> MutexGuard<'static, [usize; 128]> {
    LIVE.lock()
        .expect("the allocation table is never locked across a panic")
}

/// Makes the next `calloc` fail with NULL, as an exhausted heap would.
pub(crate) fn fail_next_calloc() {
    CALLOC_FAIL.store(CALLOC_CALLS.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
}

/// Makes the `calloc` after the next `skipped` ones fail.
pub(crate) fn fail_calloc_after(skipped: u64) {
    CALLOC_FAIL.store(
        CALLOC_CALLS.load(Ordering::Relaxed) + skipped + 1,
        Ordering::Relaxed,
    );
}

/// Stops failing `calloc`.
pub(crate) fn stop_failing_calloc() {
    CALLOC_FAIL.store(0, Ordering::Relaxed);
}

/// `calloc` results not yet freed.
pub(crate) fn live_allocations() -> usize {
    live().iter().filter(|slot| **slot != 0).count()
}

unsafe extern "C" {
    fn __real_calloc(count: usize, size: usize) -> *mut c_void;
    fn __real_free(pointer: *mut c_void);
    fn __real_open64(path: *const c_char, flags: c_int, ...) -> c_int;
    fn __real_close(fd: c_int) -> c_int;
    fn __real_fstat64(fd: c_int, buf: *mut stat) -> c_int;
    fn vsnprintf(
        buffer: *mut c_char,
        size: usize,
        format: *const c_char,
        args: VaList<'_>,
    ) -> c_int;
}

/// libc `calloc`, failing the call [`fail_next_calloc`] chose.
///
/// # Safety
/// As libc `calloc`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap_calloc(count: usize, size: usize) -> *mut c_void {
    let call = CALLOC_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if call == CALLOC_FAIL.load(Ordering::Relaxed) {
        return core::ptr::null_mut();
    }
    // SAFETY: forwarded unchanged to libc's calloc.
    let pointer = unsafe { __real_calloc(count, size) };
    if !pointer.is_null() {
        let mut live = live();
        let slot = live
            .iter_mut()
            .find(|slot| **slot == 0)
            .expect("the scenarios keep fewer than 128 allocations alive");
        *slot = pointer.addr();
    }
    pointer
}

/// libc `free`, retiring tracked `calloc` results.
///
/// # Safety
/// As libc `free`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap_free(pointer: *mut c_void) {
    if !pointer.is_null() {
        let mut live = live();
        if let Some(slot) = live.iter_mut().find(|slot| **slot == pointer.addr()) {
            *slot = 0;
        }
    }
    // SAFETY: forwarded unchanged to libc's free.
    unsafe { __real_free(pointer) };
}

// --- render node -----------------------------------------------------------

/// libc `open`: the fake render node, or a real read-only open under `/sys/`
/// (Mesa's CPU detection behind `util_cache_granularity` reads sysfs). Any
/// other path aborts the harness.
///
/// # Safety
/// `path` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap_open64(
    path: *const c_char,
    flags: c_int,
    _mode: ...
) -> c_int {
    // SAFETY: open's contract: `path` is a NUL-terminated string.
    let name = unsafe { CStr::from_ptr(path) };
    if name.to_bytes().starts_with(b"/sys/") {
        assert_eq!(flags & 3, 0, "sysfs is opened read-only");
        // SAFETY: forwarded to libc; read-only opens take no mode.
        let fd = unsafe { __real_open64(path, flags) };
        if fd >= 0 {
            state().sysfs.push(fd);
        }
        return fd;
    }
    if name == c"/dev/null" {
        // The harness's own descriptor for the fake render node (`init`).
        // SAFETY: forwarded to libc; read-only opens take no mode.
        return unsafe { __real_open64(path, flags) };
    }
    assert_eq!(name, RENDER_PATH, "only the fake render node is opened");
    let mut state = state();
    state.opens += 1;
    if state.open_error {
        state.failed_opens += 1;
        state.record(format!(
            "open({}, {flags:#o}) = -1 EACCES",
            name.to_string_lossy()
        ));
        set_errno(EACCES);
        return -1;
    }
    assert_eq!(
        flags & RENDER_OPEN_FLAGS,
        RENDER_OPEN_FLAGS,
        "the render node is opened O_RDWR | O_CLOEXEC"
    );
    state.record(format!(
        "open({}, {flags:#o}) = {}",
        name.to_string_lossy(),
        render_fd()
    ));
    render_fd()
}

/// libc `close`: the fake render node, sysfs descriptors and descriptors the
/// harness's own std closes go to libc.
///
/// # Safety
/// As libc `close`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap_close(fd: c_int) -> c_int {
    let mut state = state();
    if fd == render_fd() {
        state.closes += 1;
        state.record(format!("close({fd}) = 0"));
        return 0;
    }
    if let Some(index) = state.sysfs.iter().position(|open| *open == fd) {
        state.sysfs.swap_remove(index);
    }
    drop(state);
    // SAFETY: forwarded unchanged to libc.
    unsafe { __real_close(fd) }
}

/// libc `fstat` (glibc `fstat64`): the fake render node is a character device
/// with [`RENDER_RDEV`].
///
/// # Safety
/// `buf` is valid for one `struct stat` write.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap_fstat64(fd: c_int, buf: *mut stat) -> c_int {
    let mut state = state();
    if fd != render_fd() {
        assert!(
            state.sysfs.contains(&fd),
            "fstat only on the render node and sysfs"
        );
        drop(state);
        // SAFETY: forwarded unchanged to libc.
        return unsafe { __real_fstat64(fd, buf) };
    }
    if state.stat_error {
        state.record(format!("fstat({fd}) = -1 EIO"));
        set_errno(EIO);
        return -1;
    }
    state.record(format!("fstat({fd}) = 0"));
    // SAFETY: `stat` is a C struct of integers, for which all-zero is valid.
    let mut value: stat = unsafe { core::mem::zeroed() };
    value.st_rdev = RENDER_RDEV;
    // SAFETY: fstat's contract: `buf` is writable for one `struct stat`.
    unsafe { buf.write(value) };
    0
}

const EACCES: c_int = 13;
const EIO: c_int = 5;

unsafe extern "C" {
    fn __errno_location() -> *mut c_int;
}

fn set_errno(value: c_int) {
    // SAFETY: glibc's __errno_location returns this thread's errno, valid for
    // the thread's lifetime.
    let errno = unsafe { __errno_location() };
    // SAFETY: `errno` points to this thread's errno (above).
    unsafe { errno.write(value) };
}

/// The version record `drmGetVersion` hands out; the backend only reads
/// `name` and frees it with `drmFreeVersion`.
struct VersionRecord(core::cell::UnsafeCell<_drmVersion>);
// SAFETY: written only by `__wrap_drmGetVersion` while it holds the fake
// state lock, and read by the single scenario thread between that call and
// `drmFreeVersion`.
unsafe impl Sync for VersionRecord {}
static VERSION: VersionRecord = VersionRecord(core::cell::UnsafeCell::new(_drmVersion {
    version_major: 0,
    version_minor: 0,
    version_patchlevel: 0,
    name_len: 0,
    name: core::ptr::null_mut(),
    date_len: 0,
    date: core::ptr::null_mut(),
    desc_len: 0,
    desc: core::ptr::null_mut(),
}));

/// libdrm `drmGetVersion` of the fake render node.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn __wrap_drmGetVersion(fd: c_int) -> drmVersionPtr {
    let mut state = state();
    assert_eq!(fd, render_fd(), "the version query is on the render node");
    let name = match state.version {
        Version::Missing => {
            state.record(format!("drmGetVersion({fd}) = NULL"));
            return core::ptr::null_mut();
        }
        Version::NvidiaDrm => c"nvidia-drm".as_ptr(),
        Version::Nouveau => c"nouveau".as_ptr(),
        Version::Unnamed => core::ptr::null(),
    };
    let line = format!("drmGetVersion({fd}) = {:?}", state.version);
    state.record(line);
    let record = VERSION.0.get();
    // SAFETY: `record` points to the static record; nothing else accesses it
    // while the state lock is held (see `VersionRecord`).
    unsafe { (*record).name = name.cast_mut() };
    record
}

/// libdrm `drmFreeVersion`: the record is static.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn __wrap_drmFreeVersion(version: drmVersionPtr) {
    assert_eq!(version, VERSION.0.get(), "only the fake record is freed");
    state().record("drmFreeVersion".to_owned());
}

/// The descriptor a logical device's DRM sync provider is made for
/// (`vk_device_set_drm_fd`); never opened.
pub(crate) const SYNC_PROVIDER_FD: c_int = 99;

/// libdrm `drmGetCap`, which only Mesa's DRM sync provider asks
/// (`util/u_sync_provider.c` `util_sync_provider_drm`), and only while a
/// suite sets [`State::timeline_cap`]; the RM backend never does.
///
/// # Safety
///
/// As libdrm `drmGetCap`: `value` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap_drmGetCap(
    fd: c_int,
    capability: u64,
    value: *mut u64,
) -> c_int {
    let mut state = state();
    let answer = state
        .timeline_cap
        .expect("drmGetCap is asked only for a sync provider a suite creates");
    assert_eq!(
        fd, SYNC_PROVIDER_FD,
        "the capability query is on the provider's descriptor"
    );
    assert_eq!(
        capability,
        u64::from(DRM_CAP_SYNCOBJ_TIMELINE),
        "the provider asks for timeline syncobjs only"
    );
    state.record(format!("drmGetCap({fd}, SYNCOBJ_TIMELINE) = 0, {answer}"));
    // SAFETY: writable (caller).
    unsafe { value.write(answer) };
    0
}

/// The syncobj queries of libdrm are never used by the RM backend: its sync
/// types own their payloads.
macro_rules! drm_tripwires {
    ($($name:ident),*) => {$(
        /// Aborts: the backend never calls this.
        #[unsafe(no_mangle)]
        pub(crate) extern "C" fn $name() -> c_int {
            panic!(concat!("unexpected ", stringify!($name)));
        }
    )*};
}
drm_tripwires!(
    __wrap_drmSyncobjCreate,
    __wrap_drmSyncobjWait,
    __wrap_drmSyncobjDestroy,
    __wrap_drmSyncobjTimelineWait
);

// --- Mesa diagnostics ------------------------------------------------------

fn format_message(format: *const c_char, args: VaList<'_>) -> String {
    let mut buffer: [c_char; 512] = [0; 512];
    // SAFETY: `buffer` is writable for its length; `format` and `args` are the
    // caller's printf format and matching arguments (Mesa's contract).
    unsafe { vsnprintf(buffer.as_mut_ptr(), buffer.len(), format, args) };
    // SAFETY: vsnprintf NUL-terminates within the buffer.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Mesa's `__vk_errorf`: records the message, returns `result`. The real one
/// would walk the log object to its instance, which the suites do not build.
///
/// # Safety
/// `format` is NULL or a printf format matching the arguments.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap___vk_errorf(
    _object: *const c_void,
    result: VkResult,
    _file: *const c_char,
    _line: c_int,
    format: *const c_char,
    args: ...
) -> VkResult {
    if !format.is_null() {
        let message = format_message(format, args);
        state().report(&message);
    }
    result
}

/// Mesa's `__vk_errorv` (reached from `vk_device_set_lost`), as
/// [`__wrap___vk_errorf`].
///
/// # Safety
/// `format` is NULL or a printf format matching `args`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn __wrap___vk_errorv(
    _object: *const c_void,
    result: VkResult,
    _file: *const c_char,
    _line: c_int,
    format: *const c_char,
    args: VaList<'_>,
) -> VkResult {
    if !format.is_null() {
        let message = format_message(format, args);
        state().report(&message);
    }
    result
}

/// Mesa's `__vk_log_impl`: dropped (its objects are not real Vulkan objects).
#[unsafe(no_mangle)]
///
/// # Safety
/// None: nothing is read.
pub(crate) unsafe extern "C" fn __wrap___vk_log_impl(
    _severity: c_int,
    _types: c_int,
    _object_count: c_int,
    _objects: *const *const c_void,
    _file: *const c_char,
    _line: c_int,
    _format: *const c_char,
    _args: ...
) {
}

// --- the RM ----------------------------------------------------------------

/// What the fake RM reports: every field distinct and non-zero, so a field
/// copied to the wrong place or narrowed shows in the logged device info.
fn info() -> NvRmInfo {
    let mut name: [c_char; 64] = [0; 64];
    for (slot, byte) in name.iter_mut().zip(b"NVIDIA Fake GPU") {
        *slot = c_char::from_ne_bytes([*byte]);
    }
    NvRmInfo {
        vendor_id: 0x10de,
        device_id: 0x2204,
        chipset: 0x172,
        sm: 86,
        pci_domain: 1,
        pci_bus: 2,
        pci_device: 3,
        pci_function: 4,
        gpc_count: 7,
        tpc_count: 41,
        mp_per_tpc: 2,
        max_warps_per_mp: 48,
        cls_copy: 0xc7b5,
        cls_eng2d: 0x902d,
        cls_eng3d: 0xc797,
        cls_m2mf: 0xa140,
        cls_compute: 0xc7c0,
        generic_kind: 6,
        vram_size: 0x6_0000_0000,
        bar_size: 0x1000_0000,
        name,
        zcull: NvRmZcullInfo {
            width_align_pixels: 32,
            height_align_pixels: 16,
            pixel_squares_by_aliquots: 0x0010_0010,
            aliquot_total: 0x1200,
            region_byte_multiplier: 0x20,
            region_header_size: 0x780,
            subregion_header_size: 0xc0,
            subregion_count: 0x10,
            subregion_width_align_pixels: 0x200,
            subregion_height_align_pixels: 0x40,
        },
        zcull_ctxsw_size: 0x5d000,
        zcull_ctxsw_align: 0x1000,
        has_zcull_info: 1,
        sm_isa: 0x806,
        shader_pipe_count: 82,
        shader_pipe_sub_count: 41,
        max_shared_ctx: 63,
        reserved: 0,
    }
}

fn rm_device() -> *mut NvRmDevice {
    core::ptr::without_provenance_mut(RM_DEVICE)
}

pub(crate) fn check_device(device: *const NvRmDevice) {
    assert_eq!(
        device.addr(),
        RM_DEVICE,
        "only the fake RM device is passed back"
    );
}

/// Opens the fake GPU at 1:2:3.4. With `rm_error` it fails
/// `VK_ERROR_INITIALIZATION_FAILED`, writing the device only with
/// `rm_partial` (an RM that failed after creating its device).
///
/// # Safety
/// `out` and `info_out` are writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_open(
    domain: u32,
    bus: u32,
    device: u32,
    function: u32,
    out: *mut *mut NvRmDevice,
    info_out: *mut NvRmInfo,
) -> i32 {
    let mut state = state();
    state.rm_opens += 1;
    assert_eq!(
        (domain, bus, device, function),
        (1, 2, 3, 4),
        "the GPU's PCI address"
    );
    if !state.rm_error || state.rm_partial {
        // SAFETY: nv_rm_open's contract: `out` is writable.
        unsafe { out.write(rm_device()) };
    }
    let mut report = info();
    (state.report)(&mut report);
    // SAFETY: nv_rm_open's contract: `info_out` is writable.
    unsafe { info_out.write(report) };
    let status = if state.rm_error {
        VK_ERROR_INITIALIZATION_FAILED
    } else {
        VK_SUCCESS
    };
    let written = if !state.rm_error || state.rm_partial {
        "device"
    } else {
        "no device"
    };
    state.record(format!(
        "nv_rm_open({domain}, {bus}, {device}, {function}) = {status} ({written})"
    ));
    status
}
const _: nv_rm_abi::NvRmOpen<NvRmDevice> = nv_rm_open;

/// Closes the fake device. When a logical device is registered, its loss
/// hooks must already be cleared: nothing may call back into a backend that
/// is closing.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_close(device: *mut NvRmDevice) {
    check_device(device);
    let mut state = state();
    if state.loss_device != 0 {
        let vk = core::ptr::with_exposed_provenance_mut::<vk_device>(state.loss_device);
        for (offset, what) in [
            (NVKMD_SYS_VK_DEVICE_NOTIFY_LOST, "notify_lost"),
            (NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA, "notify_lost_data"),
            (NVKMD_SYS_VK_DEVICE_CHECK_STATUS, "check_status"),
        ] {
            assert!(
                mesa::pointer_member_is_null(vk, offset),
                "{what} cleared before nv_rm_close"
            );
        }
    }
    state.rm_closes += 1;
    state.record(format!("nv_rm_close({})", device.addr()));
}

/// Publishes the fake device's loss. With a logical device registered, Mesa
/// must have counted the loss first (`vk_device_set_lost` notifies after
/// incrementing).
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_mark_lost(device: *const NvRmDevice) {
    check_device(device);
    let mut state = state();
    if state.loss_device == 0 {
        assert!(
            state.allow_no_vk_loss,
            "loss without a logical device is expected here"
        );
    } else {
        let vk = core::ptr::with_exposed_provenance_mut::<vk_device>(state.loss_device);
        assert!(mesa::is_lost(vk), "Mesa counts the loss before notifying");
    }
    state.loss_calls += 1;
    state.rm_lost = true;
    state.record(format!("nv_rm_mark_lost({})", device.addr()));
}

/// The fake device's loss flag.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_is_lost(device: *const NvRmDevice) -> i32 {
    check_device(device);
    let mut state = state();
    state.is_lost_calls += 1;
    let lost = i32::from(state.rm_lost);
    state.record(format!("nv_rm_is_lost({}) = {lost}", device.addr()));
    lost
}

/// The device and allocation handles a memory entry receives are the fake's.
fn check_memory(device: *const NvRmDevice, memory: NvRmMemoryId) {
    check_device(device);
    assert!(
        (FIRST_MEMORY..FIRST_MAPPING).contains(&memory.get()),
        "only allocations the fake made are passed back"
    );
}

/// Allocates: the next handle, the requested size.
///
/// # Safety
/// `out` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_alloc(
    device: *mut NvRmDevice,
    size: u64,
    alignment: u64,
    placement: u32,
    pte_kind: u32,
    tile_mode: u32,
    out: *mut NvRmAllocation,
) -> i32 {
    check_device(device);
    let mut state = state();
    state.mem.allocs += 1;
    let status = state.mem.alloc_status;
    let id = state.mem.next_allocation;
    if status == VK_SUCCESS {
        state.mem.next_allocation += 1;
        state.mem.last_allocation = id;
        // SAFETY: nv_rm_alloc's contract: `out` is writable.
        unsafe {
            out.write(NvRmAllocation {
                id: NvRmMemoryId::new(id),
                size,
                cpu_address: 0,
            });
        }
    }
    state.record(format!(
        "nv_rm_alloc(size={size:#x}, align={alignment:#x}, placement={placement}, kind={pte_kind:#x}, \
         tile={tile_mode}) = {status}{}",
        if status == VK_SUCCESS { format!(" (memory {id})") } else { String::new() }
    ));
    status
}
const _: nv_rm_abi::NvRmAlloc<NvRmDevice> = nv_rm_alloc;

/// Imports a dma-buf as an allocation of [`IMPORT_SIZE`] bytes.
///
/// # Safety
/// `out` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_import(
    device: *mut NvRmDevice,
    drm_fd: c_int,
    fd: c_int,
    out: *mut NvRmAllocation,
) -> i32 {
    check_device(device);
    assert_eq!(
        drm_fd,
        render_fd(),
        "imports go through the device's render node"
    );
    let mut state = state();
    let status = state.mem.import_status;
    let id = state.mem.next_allocation;
    if status == VK_SUCCESS {
        state.mem.next_allocation += 1;
        state.mem.last_allocation = id;
        // SAFETY: nv_rm_import's contract: `out` is writable.
        unsafe {
            out.write(NvRmAllocation {
                id: NvRmMemoryId::new(id),
                size: IMPORT_SIZE,
                cpu_address: 0,
            });
        }
    }
    state.record(format!(
        "nv_rm_import(fd={fd}) = {status}{}",
        if status == VK_SUCCESS {
            format!(" (memory {id}, size {IMPORT_SIZE:#x})")
        } else {
            String::new()
        }
    ));
    status
}

/// Frees an allocation.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_free(device: *mut NvRmDevice, memory: NvRmMemoryId) -> i32 {
    check_memory(device, memory);
    let mut state = state();
    state.mem.frees += 1;
    let status = state.mem.free_status;
    state.record(format!("nv_rm_free({}) = {status}", memory.get()));
    status
}
const _: nv_rm_abi::NvRmFree<NvRmDevice> = nv_rm_free;

/// Maps an allocation at [`Memory::map_address`].
///
/// # Safety
/// `address` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_map(
    device: *mut NvRmDevice,
    memory: NvRmMemoryId,
    address: *mut usize,
) -> i32 {
    check_memory(device, memory);
    let mut state = state();
    state.mem.maps += 1;
    if state.mem.map_publishes_loss {
        state.rm_lost = true;
    }
    let status = state.mem.map_status;
    let mapped = state.mem.map_address;
    // SAFETY: nv_rm_map's contract: `address` is writable.
    unsafe { address.write(mapped) };
    state.record(format!(
        "nv_rm_map({}) = {status} ({mapped:#x})",
        memory.get()
    ));
    status
}
const _: nv_rm_abi::NvRmMap<NvRmDevice> = nv_rm_map;

/// Drops a map reference.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_unmap(device: *mut NvRmDevice, memory: NvRmMemoryId) -> i32 {
    check_memory(device, memory);
    let mut state = state();
    state.mem.unmaps += 1;
    let status = state.mem.unmap_status;
    state.record(format!("nv_rm_unmap({}) = {status}", memory.get()));
    status
}
const _: nv_rm_abi::NvRmUnmap<NvRmDevice> = nv_rm_unmap;

/// Exports an allocation as [`EXPORTED_FD`].
///
/// # Safety
/// `fd` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_export(
    device: *mut NvRmDevice,
    memory: NvRmMemoryId,
    drm_fd: c_int,
    fd: *mut c_int,
) -> i32 {
    check_memory(device, memory);
    assert_eq!(
        drm_fd,
        render_fd(),
        "exports go through the device's render node"
    );
    let mut state = state();
    let status = state.mem.export_status;
    if status == VK_SUCCESS {
        // SAFETY: nv_rm_export's contract: `fd` is writable.
        unsafe { fd.write(EXPORTED_FD) };
    }
    state.record(format!("nv_rm_export({}) = {status}", memory.get()));
    status
}

/// Binds a range of an allocation at a VA of the caller's window: the next
/// mapping handle.
///
/// # Safety
/// `mapping` is writable.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_bind(
    device: *mut NvRmDevice,
    memory: NvRmMemoryId,
    memory_offset: u64,
    size: u64,
    address: u64,
    pte_kind: u32,
    mapping: *mut NvRmMappingId,
) -> i32 {
    check_memory(device, memory);
    assert!(
        (NV_RM_VA_START..NV_RM_VA_END).contains(&address),
        "binds are inside the caller's window"
    );
    let mut state = state();
    state.mem.binds += 1;
    let status = state.mem.bind_status;
    let id = state.mem.next_mapping;
    if status == VK_SUCCESS {
        state.mem.next_mapping += 1;
        // SAFETY: nv_rm_bind's contract: `mapping` is writable.
        unsafe { mapping.write(NvRmMappingId::new(id)) };
    }
    state.record(format!(
        "nv_rm_bind(memory {}, offset={memory_offset:#x}, size={size:#x}, va={address:#x}, \
         kind={pte_kind:#x}) = {status}{}",
        memory.get(),
        if status == VK_SUCCESS {
            format!(" (mapping {id})")
        } else {
            String::new()
        }
    ));
    status
}
const _: nv_rm_abi::NvRmBind<NvRmDevice> = nv_rm_bind;

/// Unmaps one mapping now.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn nv_rm_unbind(device: *mut NvRmDevice, mapping: NvRmMappingId) -> i32 {
    check_device(device);
    assert!(
        mapping.get() >= FIRST_MAPPING,
        "only mappings the fake made are passed back"
    );
    let mut state = state();
    state.mem.unbinds += 1;
    let status = state.mem.unbind_status;
    state.record(format!(
        "nv_rm_unbind(mapping {}) = {status}",
        mapping.get()
    ));
    status
}
const _: nv_rm_abi::NvRmUnbind<NvRmDevice> = nv_rm_unbind;

/// Frees a VA reservation with its mappings. On success the range goes back
/// through `release`: at once, or (with [`Memory::defer_release`] and
/// mappings to retire) when the suite runs [`run_deferred_release`].
///
/// # Safety
/// `mappings` holds `count` ids; `release` and `data` as `NvRmVaRelease`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_rm_va_free(
    device: *mut NvRmDevice,
    mappings: *const NvRmMappingId,
    count: u32,
    address: u64,
    size: u64,
    release: NvRmVaRelease,
    data: *mut c_void,
) -> i32 {
    check_device(device);
    let release = release.expect("a VA free names its release");
    assert!(!data.is_null(), "a VA free passes its release data");
    assert!(
        address >= NV_RM_VA_START && size != 0 && size <= NV_RM_VA_END - address,
        "the range is inside the caller's window"
    );
    assert!(!mappings.is_null(), "the mapping list is never NULL");
    let count = usize::try_from(count).expect("u32 fits usize on the host");
    // SAFETY: `count` ids at `mappings` (contract).
    let ids: Vec<u64> = unsafe { core::slice::from_raw_parts(mappings, count) }
        .iter()
        .map(|mapping| mapping.get())
        .collect();
    let mut state = state();
    state.mem.va_frees += 1;
    let status = state.mem.va_free_status;
    state.record(format!(
        "nv_rm_va_free(mappings {ids:?}, va={address:#x}, size={size:#x}) = {status}"
    ));
    state.mem.last_va_free = (ids, address, size);
    if status != VK_SUCCESS {
        return status;
    }
    let deferred = Release {
        callback: release,
        data: data.expose_provenance(),
        address,
        size,
    };
    if state.mem.defer_release && count != 0 {
        state.mem.deferred = Some(deferred);
        return status;
    }
    drop(state);
    run_release(deferred);
    status
}
const _: nv_rm_abi::NvRmVaFree<NvRmDevice> = nv_rm_va_free;

fn run_release(release: Release) {
    note(format!(
        "release(va={:#x}, size={:#x})",
        release.address, release.size
    ));
    // SAFETY: the callback and data the backend passed to nv_rm_va_free, run
    // once, with the state lock released (the backend may call the fakes).
    unsafe {
        (release.callback)(
            core::ptr::with_exposed_provenance_mut(release.data),
            release.address,
            release.size,
        );
    }
}

/// Runs the release the last `nv_rm_va_free` deferred, as the RM does once the
/// work using the range retired.
pub(crate) fn run_deferred_release() {
    let release = state()
        .mem
        .deferred
        .take()
        .expect("a deferred release is pending");
    run_release(release);
}

/// Whether a deferred release is pending.
pub(crate) fn release_pending() -> bool {
    state().mem.deferred.is_some()
}

/// RM entries no suite expects. Each aborts the harness before reading an
/// argument, so the declared parameter list does not matter.
macro_rules! rm_tripwires {
    ($($name:ident),*) => {$(
        /// Aborts: no suite expects this entry.
        #[unsafe(no_mangle)]
        pub(crate) extern "C" fn $name() -> i32 {
            panic!(concat!("unexpected ", stringify!($name)));
        }
    )*};
}
rm_tripwires!(
    nv_rm_check_status,
    nv_rm_context_fence,
    nv_rm_context_signal,
    nv_rm_exec,
    nv_rm_poll,
    nv_rm_submit,
    nv_rm_submit_sync,
    nv_rm_submit_waits,
    nv_rm_sync_destroy,
    nv_rm_sync_status_many,
    nv_rm_timestamp
);

/// Formats the fields of a created physical device the backend filled, for
/// the call log: the C and the Rust backend must fill them identically.
pub(crate) fn describe_pdev(pdev: &nvkmd_sys::nvkmd_pdev) -> String {
    let info = &pdev.dev_info;
    let kmd = &pdev.kmd_info;
    let name = |bytes: &[c_char]| -> String {
        bytes
            .iter()
            .map(|byte| byte.to_ne_bytes()[0])
            .take_while(|byte| *byte != 0)
            .map(char::from)
            .collect()
    };
    let mut line = String::new();
    let zcull = &info.zcull_info;
    write!(
        line,
        "pdev type={} device_id={:#x} chipset={:#x} name={:?} chipset_name={:?} \
         pci={}:{}:{}.{} rev={} sm={} gpc={} tpc={} mp_per_tpc={} warps={} transfer={} \
         nc_atom={} cls={:#x}/{:#x}/{:#x}/{:#x}/{:#x}/{:#x} vram={:#x} bar={:#x} \
         smem_max={} smem={:?}/{} zcull={} [{} {} {:#x} {:#x} {:#x} {:#x} {:#x} {} {} {} {:#x} {:#x}] \
         kmd=[dma_buf={} sparse={} vram_used={} tiled={} map_fixed={} overmap={} compression={} \
         rm_compression={} shared_ctx={} isa={:#x} pipes={}/{}] bind_align={} render_dev={:#x} \
         primary_dev={:#x} debug={}",
        info.type_,
        info.device_id,
        info.chipset,
        name(&info.device_name),
        name(&info.chipset_name),
        info.pci.domain,
        info.pci.bus,
        info.pci.dev,
        info.pci.func,
        info.pci.revision_id,
        info.sm,
        info.gpc_count,
        info.tpc_count,
        info.mp_per_tpc,
        info.max_warps_per_mp,
        info.has_transfer_queue,
        info.nc_atom_size_B,
        info.cls_copy,
        info.cls_eng2d,
        info.cls_eng3d,
        info.cls_m2mf,
        info.cls_compute,
        info.cls_gpfifo,
        info.vram_size_B,
        info.bar_size_B,
        info.max_smem_per_wg_kB,
        info.sm_smem_sizes_kB,
        info.sm_smem_size_count,
        info.has_zcull_info,
        zcull.width_align_pixels,
        zcull.height_align_pixels,
        zcull.pixel_squares_by_aliquots,
        zcull.aliquot_total,
        zcull.zcull_region_byte_multiplier,
        zcull.zcull_region_header_size,
        zcull.zcull_subregion_header_size,
        zcull.subregion_count,
        zcull.subregion_width_align_pixels,
        zcull.subregion_height_align_pixels,
        zcull.ctxsw_size,
        zcull.ctxsw_align,
        kmd.has_dma_buf,
        kmd.has_sparse,
        kmd.has_get_vram_used,
        kmd.has_alloc_tiled,
        kmd.has_map_fixed,
        kmd.has_overmap,
        kmd.has_compression,
        kmd.has_rm_compression,
        kmd.max_shared_ctx,
        kmd.sm_isa_version,
        kmd.shader_pipe_count,
        kmd.shader_pipe_sub_count,
        pdev.bind_align_B,
        pdev.drm.render_dev,
        pdev.drm.primary_dev,
        pdev.debug_flags,
    )
    .expect("formatting into a String cannot fail");
    line
}
