//! Opt-in command-stream capture and per-submission GPU timestamps for
//! performance measurement (perf work, 2026-10-15). Off by default; when the
//! configuration below is absent the backend's cost is one `Option` check and
//! one clock read per submit, a configuration read (property lookup) once per
//! second, a bind-table entry per bind, and one atomic load per fence export.
//!
//! Configuration is read at device open and, while absent, again by submit at
//! most every [`ARM_POLL`] ("late arming", 2026-10-16): setting the properties
//! (or, on the host, nothing: the environment cannot change) on a process that
//! is already running arms it within a second, so an intermittent bug can be
//! captured in the process that shows it without a restart. A late arm enables
//! the timestamp releases on every live context and indexes the binds that
//! exist at that moment for pushbuffer copies. Properties:
//! * host: environment `NV_RM_CAPTURE=<dir>`, `NV_RM_CAPTURE_PROC=<substr>`,
//!   `NV_RM_CAPTURE_FRAMES=<n>`, `NV_RM_CAPTURE_MODE=push|ts`;
//! * Android: properties `debug.nvrm.capture`, `debug.nvrm.capture.proc`,
//!   `debug.nvrm.capture.frames`, `debug.nvrm.capture.mode` (same meaning;
//!   the environment wins when both are set). For a late arm set `.proc`,
//!   `.mode` and `.frames` first and `debug.nvrm.capture` last: every polling
//!   process arms as soon as `debug.nvrm.capture` is set, with whatever
//!   filter is set at that moment.
//!
//! A process whose /proc/self/cmdline contains `proc` (any process when empty)
//! is *armed*: it keeps a VA→allocation index and, on Volta+ hosts,
//! adds two host semaphore releases with timestamps to every submission
//! (start: after the batch's acquires, no WFI; end: after the existing
//! SET_REFERENCE/WFI + membar of the completion tail). Capture starts when the
//! file `<dir>/go` exists (checked at most every 200 ms from submit) and stops
//! after `frames` fence exports (sync-file exports = presents for both the
//! Android EGL and Vulkan WSI paths) or 30 s. Output:
//! `<dir>/cap-<pid>-<device>.bin`, records described in `tools/nvcap.py`.
//!
//! The extra releases change the stream slightly (one extra GPFIFO entry per
//! submission for the start release); runs that measure FPS use mode `ts`,
//! which never copies pushbuffer contents.

use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub(crate) const MAGIC: &[u8; 8] = b"NVCAP01\0";
/// Record types (u32 tag, u32 payload length, payload).
pub(crate) const REC_START: u32 = 1; // cpu_ns u64, gpu_ns u64, pid u32, pad u32
pub(crate) const REC_SUBMIT: u32 = 2; // see Capture::submit
pub(crate) const REC_TIME: u32 = 3; // context u64, token u64, start u64, end u64
pub(crate) const REC_MARK: u32 = 4; // cpu_ns u64, context u64, token u64, kind u32, pad u32
pub(crate) const REC_END: u32 = 5; // cpu_ns u64, reason u32, pad u32
pub(crate) const REC_CONTEXT: u32 = 6; // context u64, engines u32, channel class u32
/// Pool contents (push mode): token u64, address u64, subchannel u32, kind u32
/// (1 texture header pool, 2 sampler pool), entries u32, captured bytes u32,
/// then the bytes (0 when the pool is not CPU-mapped).
pub(crate) const REC_POOL: u32 = 7;

/// Largest pool copied per binding (32 B entries).
const POOL_MAX_BYTES: u64 = 1 << 20;

/// A SET_TEX_HEADER_POOL / SET_TEX_SAMPLER_POOL binding found in a batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PoolBind {
    pub subchannel: u32,
    pub kind: u32,
    pub address: u64,
    pub entries: u32,
    pub size: u32,
}

/// Method offsets shared by every 3D class from FERMI_A (NV9097) and every
/// compute class from KEPLER_COMPUTE_A (NVA0C0): SET_TEX_SAMPLER_POOL_A/B/C at
/// 0x155c/0x1560/0x1564, SET_TEX_HEADER_POOL_A/B/C at 0x1574/0x1578/0x157c
/// (cl9097.h:2268-2290, cla0c0.h:507-520, clce97.h, clcec0.h). Subchannels 0
/// (3D) and 1 (compute) as NVK and nvrm bind them.
const SAMPLER_POOL_A: u32 = 0x155c;
const HEADER_POOL_A: u32 = 0x1574;

/// Pool bindings of one batch, from its captured pushbuffer words (method
/// headers decoded as nv_push.h writes them: INC 1, NON_INC 3, IMMD 4,
/// ONE_INC 5). The last value of each pool per subchannel wins.
pub(crate) fn pool_binds<'a>(pushes: impl Iterator<Item = &'a [u8]>) -> Vec<PoolBind> {
    // [subchannel][sampler, header][A, B, C], Some once written.
    let mut regs = [[[None::<u32>; 3]; 2]; 2];
    let mut touched = [[false; 2]; 2];
    for bytes in pushes {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut i = 0;
        while i < words.len() {
            let h = words[i];
            let op = h >> 29;
            let count = ((h >> 16) & 0x1fff) as usize;
            let sc = ((h >> 13) & 7) as usize;
            let method = (h & 0xfff) << 2;
            i += 1;
            let values: &[u32] = match op {
                1 | 3 | 5 => {
                    let end = (i + count).min(words.len());
                    let v = &words[i..end];
                    i = end;
                    v
                }
                _ => continue, // IMMD (no pool method is sent immediate) / other
            };
            if sc > 1 {
                continue;
            }
            for (n, &v) in values.iter().enumerate() {
                let m = match op {
                    1 => method + 4 * n as u32,
                    3 => method,
                    _ => method + if n == 0 { 0 } else { 4 },
                };
                for (k, base) in [SAMPLER_POOL_A, HEADER_POOL_A].into_iter().enumerate() {
                    if (base..base + 12).contains(&m) {
                        regs[sc][k][((m - base) / 4) as usize] = Some(v);
                        touched[sc][k] = true;
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for sc in 0..2 {
        for k in 0..2 {
            if !touched[sc][k] {
                continue;
            }
            let [Some(a), Some(b), Some(c)] = regs[sc][k] else {
                continue;
            };
            let entries = c.saturating_add(1);
            let bytes = (u64::from(entries) * 32).min(POOL_MAX_BYTES);
            out.push(PoolBind {
                subchannel: sc as u32,
                kind: if k == 0 { 2 } else { 1 },
                // OFFSET_UPPER is 7:0 on NV9097 and 24:0 on newer classes
                // (clce97.h:3213); unused bits are written as 0.
                address: u64::from(a & 0x1ff_ffff) << 32 | u64::from(b),
                entries,
                size: bytes as u32,
            });
        }
    }
    out
}

/// Late arming: how often a process without a capture configuration looks for
/// one again (from submit, Bridge mutex held; one property read per poll).
pub(crate) const ARM_POLL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug)]
pub(crate) struct Config {
    pub dir: PathBuf,
    pub frames: u32,
    pub push: bool,
}

#[cfg(target_os = "android")]
fn property(name: &str) -> Option<String> {
    unsafe extern "C" {
        fn __system_property_get(
            name: *const std::os::raw::c_char,
            value: *mut std::os::raw::c_char,
        ) -> i32;
    }
    let name = std::ffi::CString::new(name).ok()?;
    // PROP_VALUE_MAX = 92 (sys/system_properties.h).
    let mut value = [0 as std::os::raw::c_char; 92];
    let n = unsafe { __system_property_get(name.as_ptr(), value.as_mut_ptr()) };
    if n <= 0 {
        return None;
    }
    let bytes: Vec<u8> = value[..n as usize].iter().map(|&c| c as u8).collect();
    String::from_utf8(bytes).ok()
}
#[cfg(not(target_os = "android"))]
fn property(_name: &str) -> Option<String> {
    None
}

fn setting(env: &str, prop: &str) -> Option<String> {
    std::env::var(env)
        .ok()
        .or_else(|| property(prop))
        .filter(|v| !v.is_empty())
}

/// None = capture off for this process (the default).
pub(crate) fn config() -> Option<Config> {
    let dir = setting("NV_RM_CAPTURE", "debug.nvrm.capture")?;
    let filter = setting("NV_RM_CAPTURE_PROC", "debug.nvrm.capture.proc").unwrap_or_default();
    if !filter.is_empty() {
        let cmdline = std::fs::read("/proc/self/cmdline").unwrap_or_default();
        let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
        if !cmdline.contains(&filter) {
            return None;
        }
    }
    let frames = setting("NV_RM_CAPTURE_FRAMES", "debug.nvrm.capture.frames")
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let push = setting("NV_RM_CAPTURE_MODE", "debug.nvrm.capture.mode").as_deref() != Some("ts");
    // Note: a `go` file containing "push" or "ts" overrides this per capture.
    Some(Config {
        dir: PathBuf::from(dir),
        frames,
        push,
    })
}

enum State {
    /// Last check of the go file (None: check at the next submit).
    Armed(Option<Instant>),
    Active { since: Instant, frames: u32 },
    Done,
}

/// Per-device capture state; lives in the bridge's `Inner` (mutex held).
pub(crate) struct Capture {
    pub config: Config,
    state: State,
    file: Option<BufWriter<File>>,
    /// VA start → (length, allocation id, allocation offset); mode push only.
    pub binds: BTreeMap<u64, (u64, u64, u64)>,
    epoch: Instant,
    device_tag: usize,
    /// (context, engines, channel class) of every context created.
    contexts: Vec<(u64, u32, u32)>,
    scratch: Vec<u8>,
    sequence: u32,
}

impl Capture {
    pub(crate) fn new(config: Config, device_tag: usize) -> Self {
        crate::diag::diag!(
            "nvk Rust RM capture armed: dir {} frames {} mode {} (start: create {}/go)",
            config.dir.display(),
            config.frames,
            if config.push { "push" } else { "ts" },
            config.dir.display()
        );
        Self {
            config,
            state: State::Armed(None),
            file: None,
            binds: BTreeMap::new(),
            epoch: Instant::now(),
            device_tag,
            contexts: Vec::new(),
            scratch: Vec::new(),
            sequence: 0,
        }
    }
    pub(crate) fn cpu_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }
    /// Advance the arm/active/done state; true while recording.
    pub(crate) fn active(&mut self, gpu_now: impl FnOnce() -> u64) -> bool {
        match self.state {
            State::Active { since, .. } => {
                if since.elapsed() > Duration::from_secs(30) {
                    self.finish(1);
                    return false;
                }
                true
            }
            State::Done => false,
            State::Armed(last) => {
                if last.is_some_and(|l| l.elapsed() < Duration::from_millis(200)) {
                    return false;
                }
                let go = self.config.dir.join("go");
                let Ok(content) = std::fs::read_to_string(&go) else {
                    self.state = State::Armed(Some(Instant::now()));
                    return false;
                };
                if content.contains("push") {
                    self.config.push = true;
                } else if content.contains("ts") {
                    self.config.push = false;
                }
                if let Some(n) = content
                    .split_whitespace()
                    .find_map(|w| w.strip_prefix("frames=").and_then(|v| v.parse().ok()))
                {
                    self.config.frames = n;
                }
                self.sequence += 1;
                let path = self.config.dir.join(format!(
                    "cap-{}-{:x}-{}.bin",
                    std::process::id(),
                    self.device_tag,
                    self.sequence
                ));
                match File::create(&path) {
                    Ok(f) => {
                        let mut w = BufWriter::with_capacity(1 << 20, f);
                        let _ = w.write_all(MAGIC);
                        self.file = Some(w);
                        let mut p = Vec::new();
                        p.extend_from_slice(&self.cpu_ns().to_le_bytes());
                        p.extend_from_slice(&gpu_now().to_le_bytes());
                        p.extend_from_slice(&std::process::id().to_le_bytes());
                        p.extend_from_slice(&u32::from(self.config.push).to_le_bytes());
                        self.record(REC_START, &p);
                        for (c, e, k) in self.contexts.clone() {
                            self.write_context(c, e, k);
                        }
                        self.state = State::Active {
                            since: Instant::now(),
                            frames: 0,
                        };
                        crate::diag::diag!("nvk Rust RM capture started: {}", path.display());
                        true
                    }
                    Err(e) => {
                        crate::diag::diag!("nvk Rust RM capture: cannot create {}: {e}", path.display());
                        self.state = State::Done;
                        false
                    }
                }
            }
        }
    }
    pub(crate) fn record(&mut self, tag: u32, payload: &[u8]) {
        if let Some(w) = self.file.as_mut() {
            let _ = w.write_all(&tag.to_le_bytes());
            let _ = w.write_all(&(payload.len() as u32).to_le_bytes());
            let _ = w.write_all(payload);
        }
    }
    pub(crate) fn add_context(&mut self, context: u64, engines: u32, class: u32) {
        self.contexts.push((context, engines, class));
        if self.is_active() {
            self.write_context(context, engines, class);
        }
    }
    fn write_context(&mut self, context: u64, engines: u32, class: u32) {
        let mut p = Vec::new();
        p.extend_from_slice(&context.to_le_bytes());
        p.extend_from_slice(&engines.to_le_bytes());
        p.extend_from_slice(&class.to_le_bytes());
        self.record(REC_CONTEXT, &p);
    }
    /// Submit header: cpu_ns u64, context u64, token u64, waits u32, foreign
    /// u32, signals u32, pushes u32; then per push: address u64, size u32,
    /// flags u32, captured bytes u32, then the captured words.
    pub(crate) fn submit(
        &mut self,
        context: u64,
        token: u64,
        counts: [u32; 3],
        pushes: &[(u64, u32, u32, Option<Vec<u8>>)],
    ) {
        let mut p = std::mem::take(&mut self.scratch);
        p.clear();
        p.extend_from_slice(&self.cpu_ns().to_le_bytes());
        p.extend_from_slice(&context.to_le_bytes());
        p.extend_from_slice(&token.to_le_bytes());
        for c in counts {
            p.extend_from_slice(&c.to_le_bytes());
        }
        p.extend_from_slice(&(pushes.len() as u32).to_le_bytes());
        for (address, size, flags, data) in pushes {
            p.extend_from_slice(&address.to_le_bytes());
            p.extend_from_slice(&size.to_le_bytes());
            p.extend_from_slice(&flags.to_le_bytes());
            let data = data.as_deref().unwrap_or(&[]);
            p.extend_from_slice(&(data.len() as u32).to_le_bytes());
            p.extend_from_slice(data);
        }
        self.record(REC_SUBMIT, &p);
        self.scratch = p;
    }
    /// Pool contents bound by the batch `token` (REC_POOL).
    pub(crate) fn pool(&mut self, token: u64, pool: &PoolBind, bytes: Option<&[u8]>) {
        let bytes = bytes.unwrap_or(&[]);
        let mut p = Vec::with_capacity(32 + bytes.len());
        p.extend_from_slice(&token.to_le_bytes());
        p.extend_from_slice(&pool.address.to_le_bytes());
        p.extend_from_slice(&pool.subchannel.to_le_bytes());
        p.extend_from_slice(&pool.kind.to_le_bytes());
        p.extend_from_slice(&pool.entries.to_le_bytes());
        p.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        p.extend_from_slice(bytes);
        self.record(REC_POOL, &p);
    }
    pub(crate) fn time(&mut self, context: u64, token: u64, start: u64, end: u64) {
        let mut p = [0u8; 32];
        p[..8].copy_from_slice(&context.to_le_bytes());
        p[8..16].copy_from_slice(&token.to_le_bytes());
        p[16..24].copy_from_slice(&start.to_le_bytes());
        p[24..].copy_from_slice(&end.to_le_bytes());
        self.record(REC_TIME, &p);
    }
    /// Fence export (frame marker). Counts frames while active.
    pub(crate) fn mark(&mut self, context: u64, token: u64, kind: u32) {
        let State::Active { frames, .. } = &mut self.state else {
            return;
        };
        *frames += 1;
        let done = *frames > self.config.frames;
        let mut p = Vec::with_capacity(32);
        p.extend_from_slice(&self.cpu_ns().to_le_bytes());
        p.extend_from_slice(&context.to_le_bytes());
        p.extend_from_slice(&token.to_le_bytes());
        p.extend_from_slice(&kind.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        self.record(REC_MARK, &p);
        if done {
            self.finish(0);
        }
    }
    pub(crate) fn finish(&mut self, reason: u32) {
        let mut p = Vec::new();
        p.extend_from_slice(&self.cpu_ns().to_le_bytes());
        p.extend_from_slice(&reason.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        self.record(REC_END, &p);
        if let Some(mut w) = self.file.take() {
            let _ = w.flush();
        }
        // Re-arm: the next capture starts when `go` is created again.
        let _ = std::fs::remove_file(self.config.dir.join("go"));
        self.state = if reason == 2 { State::Done } else { State::Armed(None) };
        crate::diag::diag!("nvk Rust RM capture finished (reason {reason})");
    }
    pub(crate) fn is_active(&self) -> bool {
        matches!(self.state, State::Active { .. })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.file.is_some() {
            self.finish(2);
        }
    }
}

/// Per-queue timestamp ring: entry i (32 B) = start release record (16 B:
/// payload u32, pad, timestamp u64) + end release record (16 B).
pub(crate) const TS_ENTRIES: usize = 1024;
pub(crate) struct TimingRing {
    pub cpu: crate::rm::Mapping,
    pub gpu: crate::gpu::GpuMapping,
    pub next: u32,
    /// (sequence, token) of submissions whose end record is not read yet.
    pub pending: VecDeque<(u32, u64)>,
}
impl TimingRing {
    pub(crate) fn entry_address(&self, seq: u32) -> u64 {
        self.gpu.address() + (seq as usize % TS_ENTRIES * 32) as u64
    }
    /// Pops completed entries: (token, start_ns, end_ns).
    pub(crate) fn drain(&mut self, out: &mut Vec<(u64, u64, u64)>) {
        crate::platform::device_read_barrier();
        while let Some(&(seq, token)) = self.pending.front() {
            let base = seq as usize % TS_ENTRIES * 32;
            let end_payload = self.cpu.read_u32(base + 16).unwrap_or(0);
            if end_payload != seq {
                // Stale entries can only be newer work still in flight.
                break;
            }
            let rd = |o: usize| -> u64 {
                let lo = self.cpu.read_u32(o).unwrap_or(0) as u64;
                let hi = self.cpu.read_u32(o + 4).unwrap_or(0) as u64;
                lo | hi << 32
            };
            out.push((token, rd(base + 8), rd(base + 24)));
            self.pending.pop_front();
        }
    }
}

#[cfg(test)]
mod pool_bind_tests {
    use super::*;
    fn header(op: u32, count: u32, sc: u32, method: u32) -> u32 {
        op << 29 | count << 16 | sc << 13 | method >> 2
    }
    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    #[test]
    fn finds_3d_and_compute_pools() {
        // nvrm's nvrm_emit_tex_pools: INC x3 header pool, IMMD invalidate,
        // INC x3 sampler pool; compute: the same on subchannel 1.
        let w = [
            header(1, 3, 0, HEADER_POOL_A),
            0xfe,
            0xffeb_f000,
            0xfff,
            header(4, 0, 0, 0x1330),
            header(1, 3, 0, SAMPLER_POOL_A),
            0xfe,
            0xffed_f000,
            0xfff,
            header(1, 3, 1, HEADER_POOL_A),
            0x1,
            0x2000,
            0x0,
        ];
        let b = bytes(&w);
        let mut got = pool_binds([b.as_slice()].into_iter());
        got.sort_by_key(|p| (p.subchannel, p.kind));
        assert_eq!(
            got,
            vec![
                PoolBind {
                    subchannel: 0,
                    kind: 1,
                    address: 0xfe_ffeb_f000,
                    entries: 4096,
                    size: 131072
                },
                PoolBind {
                    subchannel: 0,
                    kind: 2,
                    address: 0xfe_ffed_f000,
                    entries: 4096,
                    size: 131072
                },
                PoolBind {
                    subchannel: 1,
                    kind: 1,
                    address: 0x1_0000_2000,
                    entries: 1,
                    size: 32
                },
            ]
        );
    }
    #[test]
    fn ignores_other_methods_and_partial_binds() {
        // Only B written: not a complete binding.
        let w = [
            header(1, 1, 0, HEADER_POOL_A + 4),
            0x1234,
            header(3, 2, 0, 0x2000),
            1,
            2,
        ];
        let b = bytes(&w);
        assert!(pool_binds([b.as_slice()].into_iter()).is_empty());
    }
}
