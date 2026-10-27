// SPDX-License-Identifier: MIT
//! C ABI of video/nv_video.h (v1 entry points, v2 additions: HDR static metadata and the
//! per-unit displayable-picture count).
use super::{
    NvVideoCaps, NvVideoDeviceInfo, VideoDevice, VideoError,
    hdr::HdrStaticMetadata,
    session::{Session, SessionInfo, Target},
};
use crate::ffi::{FailFlag, Failable};
use crate::process::ProcessBound;
use std::{cell::RefCell, ffi::CString};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(e: VideoError) -> i32 {
    let msg = e.message().replace('\0', " ");
    crate::diag::diag!("nv-video: {msg}");
    LAST_ERROR.with(|l| {
        *l.borrow_mut() = CString::new(msg).expect("interior NULs were replaced above");
    });
    e.code()
}

/// The shared C-boundary guard (crate::ffi::guard) for an entry point on
/// `object`: a panic, or a call on an object an earlier panic marked failed,
/// fails with NV_VIDEO_E_DEVICE_LOST and sets the thread's last error.
fn guard(
    entry: &'static str,
    object: Option<&dyn Failable>,
    f: impl FnOnce() -> Result<i32, VideoError>,
) -> i32 {
    crate::ffi::guard(
        entry,
        object,
        || {
            Err(VideoError::DeviceLost(
                "internal panic; the object is marked failed".into(),
            ))
        },
        f,
    )
    .unwrap_or_else(fail)
}

/// Device handle given to C (Box::into_raw; released by nv_video_device_close).
pub struct NvVideoDevice {
    device: VideoDevice,
    failed: FailFlag,
    /// A forked child may neither use nor tear down the parent's device.
    process: ProcessBound,
}
/// Session handle given to C (Box::into_raw; released by
/// nv_video_session_destroy). Only shared references to the handle exist; the
/// session itself is borrowed mutably through the cell for one call at a time
/// (nv_video.h: callers serialise each session).
pub struct NvVideoSession {
    session: RefCell<Session>,
    failed: FailFlag,
    /// A forked child may neither use nor tear down the parent's session.
    process: ProcessBound,
}
// SAFETY: a session's RM objects are only reachable through the session; the
// header requires callers to serialise each session, so moving it between
// threads never shares its Rc graph (or the RefCell flag) concurrently.
unsafe impl Send for NvVideoSession {}
/// The object was marked failed, or a forked child is using its parent's
/// object (logged: the child must open its own).
fn unusable(failed: &FailFlag, process: &ProcessBound) -> bool {
    if !process.is_current() {
        crate::diag::diag!(
            "nv-video: object created by pid {} used in a forked child; open a new device in this process",
            process.creator()
        );
        return true;
    }
    failed.is_set()
}
impl Failable for NvVideoDevice {
    fn failed(&self) -> bool {
        unusable(&self.failed, &self.process)
    }
    fn mark_failed(&self) {
        self.failed.set();
    }
}
impl Failable for NvVideoSession {
    fn failed(&self) -> bool {
        unusable(&self.failed, &self.process)
    }
    fn mark_failed(&self) {
        self.failed.set();
    }
}
impl NvVideoSession {
    /// The session for one call; a second borrow means the caller broke the
    /// header's one-call-at-a-time rule.
    fn session(&self) -> Result<std::cell::RefMut<'_, Session>, VideoError> {
        self.session.try_borrow_mut().map_err(|error| {
            VideoError::Invalid(format!("session used by two calls at once: {error}"))
        })
    }
}

#[repr(C)]
pub struct NvVideoSessionInfo {
    pub codec: u32,
    pub max_width: u32,
    pub max_height: u32,
    pub bit_depth: u32,
    pub extra_output_slots: u32,
    pub codec_config: *const u8,
    pub codec_config_len: usize,
    pub reserved: [u32; 8],
}

#[repr(C)]
pub struct NvVideoPicture {
    pub id: u64,
    pub timestamp: u64,
    pub width: u32,
    pub height: u32,
    pub crop_left: u32,
    pub crop_top: u32,
    pub crop_width: u32,
    pub crop_height: u32,
    pub bit_depth: u32,
    pub color_primaries: u32,
    pub transfer: u32,
    pub matrix: u32,
    pub full_range: u32,
    pub keyframe: u32,
    pub reserved: [u32; 5],
}

/// `mastering_display_colour_volume` as coded in the bitstream (ST 2086 units).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvVideoMasteringDisplay {
    pub display_primaries_x: [u16; 3],
    pub display_primaries_y: [u16; 3],
    pub white_point_x: u16,
    pub white_point_y: u16,
    pub max_luminance: u32,
    pub min_luminance: u32,
}

/// HDR static metadata of a picture (ABI v2, `nv_video_picture_hdr`). A field group with
/// its `has_*` flag clear carries no values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvVideoHdrMetadata {
    pub has_mastering_display: u32,
    pub mastering_display: NvVideoMasteringDisplay,
    pub has_content_light_level: u32,
    pub max_content_light_level: u32,
    pub max_pic_average_light_level: u32,
    pub reserved: [u32; 6],
}

impl From<HdrStaticMetadata> for NvVideoHdrMetadata {
    fn from(m: HdrStaticMetadata) -> Self {
        let mut out = Self::default();
        if let Some(d) = m.mastering {
            out.has_mastering_display = 1;
            out.mastering_display = NvVideoMasteringDisplay {
                display_primaries_x: d.primaries_x,
                display_primaries_y: d.primaries_y,
                white_point_x: d.white_x,
                white_point_y: d.white_y,
                max_luminance: d.max_luminance,
                min_luminance: d.min_luminance,
            };
        }
        if let Some((cll, fall)) = m.content_light {
            out.has_content_light_level = 1;
            out.max_content_light_level = u32::from(cll);
            out.max_pic_average_light_level = u32::from(fall);
        }
        out
    }
}

#[repr(C)]
pub struct NvVideoTarget {
    pub dmabuf_fd: [i32; 2],
    pub offset: [u64; 2],
    pub pitch: [u32; 2],
    pub modifier: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub acquire_fence_fd: i32,
    pub cr_dmabuf_fd: i32,
    pub cr_pitch: u32,
    pub cr_offset: u64,
    pub reserved: [u32; 2],
}

fn null<T>(p: *const T, what: &str) -> Result<(), VideoError> {
    if p.is_null() {
        Err(VideoError::Invalid(format!("{what} is NULL")))
    } else {
        Ok(())
    }
}

/// # Safety
/// `out` must be writable; `info` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_device_open(
    out: *mut *mut NvVideoDevice,
    info: *mut NvVideoDeviceInfo,
) -> i32 {
    guard("nv_video_device_open", None, || {
        crate::diag::install_logger();
        null(out, "out")?;
        let dev = VideoDevice::open()?;
        if !info.is_null() {
            // SAFETY: caller-provided writable struct (non-NULL checked).
            unsafe { info.write(dev.info) };
        }
        let handle = NvVideoDevice {
            device: dev,
            failed: FailFlag::default(),
            process: ProcessBound::new()
                .map_err(|e| VideoError::NoMemory(format!("register the fork handler: {e}")))?,
        };
        // SAFETY: checked non-NULL, writable (# Safety). The Box is released
        // only by nv_video_device_close.
        unsafe { out.write(Box::into_raw(Box::new(handle))) };
        Ok(0)
    })
}

/// # Safety
/// `dev` from nv_video_device_open (or NULL), not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_device_close(dev: *mut NvVideoDevice) {
    crate::ffi::guard_or_abort("nv_video_device_close", || {
        if dev.is_null() {
            return;
        }
        // SAFETY: ownership returns from the caller (# Safety), exactly once.
        let owned = unsafe { Box::from_raw(dev) };
        release(owned, "nv_video_device_close");
    });
}

/// Drop a handle C gave back, unless it is failed: a panic may have left its
/// state inconsistent, or a forked child holds the parent's objects; it is
/// then kept (with its RM objects) until process exit instead of torn down.
fn release<T: Failable>(handle: Box<T>, entry: &str) {
    if handle.failed() {
        crate::diag::diag!(
            "{entry}: object is failed or from another process; its resources are kept"
        );
        std::mem::forget(handle);
    } else {
        drop(handle);
    }
}

/// # Safety
/// `dev` NULL or a live device, `caps` NULL or writable, for the call.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_video_query_caps(
    dev: *mut NvVideoDevice,
    codec: u32,
    caps: *mut NvVideoCaps,
) -> i32 {
    // SAFETY: `dev` is NULL or a live device for the call (# Safety).
    let device = unsafe { dev.as_ref() };
    guard(
        "nv_video_query_caps",
        device.map(|d| -> &dyn Failable { d }),
        || {
            null(dev, "dev")?;
            null(caps, "caps")?;
            let d = device.ok_or_else(|| VideoError::Invalid("dev is NULL".into()))?;
            let c = d.device.caps(codec)?;
            // SAFETY: checked non-NULL, writable for the call (# Safety).
            unsafe { caps.write(c) };
            Ok(0)
        },
    )
}

/// # Safety
/// `dev` NULL or a live device, `info` NULL or readable, `out` NULL or
/// writable, for the call; `codec_config` readable for `codec_config_len`
/// bytes when non-NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_session_create(
    dev: *mut NvVideoDevice,
    info: *const NvVideoSessionInfo,
    out: *mut *mut NvVideoSession,
) -> i32 {
    // SAFETY: `dev` is NULL or a live device for the call (# Safety).
    let device = unsafe { dev.as_ref() };
    guard(
        "nv_video_session_create",
        device.map(|d| -> &dyn Failable { d }),
        || {
            null(dev, "dev")?;
            null(info, "info")?;
            null(out, "out")?;
            let d = device.ok_or_else(|| VideoError::Invalid("dev is NULL".into()))?;
            // SAFETY: checked non-NULL, readable for the call (# Safety).
            let i = unsafe { &*info };
            let cfg = if i.codec_config.is_null() || i.codec_config_len == 0 {
                &[][..]
            } else {
                // SAFETY: non-NULL, readable for its length (# Safety).
                unsafe { std::slice::from_raw_parts(i.codec_config, i.codec_config_len) }
            };
            let s = Session::new(
                &d.device,
                SessionInfo {
                    codec: i.codec,
                    max_width: i.max_width,
                    max_height: i.max_height,
                    bit_depth: i.bit_depth,
                    extra_output_slots: i.extra_output_slots,
                },
                cfg,
            )?;
            let handle = NvVideoSession {
                session: RefCell::new(s),
                failed: FailFlag::default(),
                process: ProcessBound::new()
                    .map_err(|e| VideoError::NoMemory(format!("register the fork handler: {e}")))?,
            };
            // SAFETY: checked non-NULL, writable (# Safety). The Box is
            // released only by nv_video_session_destroy.
            unsafe { out.write(Box::into_raw(Box::new(handle))) };
            Ok(0)
        },
    )
}

/// # Safety
/// `s` from nv_video_session_create (or NULL), not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_session_destroy(s: *mut NvVideoSession) {
    crate::ffi::guard_or_abort("nv_video_session_destroy", || {
        if s.is_null() {
            return;
        }
        // SAFETY: ownership returns from the caller (# Safety), exactly once.
        let owned = unsafe { Box::from_raw(s) };
        release(owned, "nv_video_session_destroy");
    });
}

/// The session argument of a session entry point.
///
/// # Safety
/// `s` is NULL or a live session from nv_video_session_create, not destroyed
/// during `'a`.
unsafe fn session_ref<'a>(s: *mut NvVideoSession) -> Option<&'a NvVideoSession> {
    // SAFETY: NULL or live for 'a (# Safety); only shared references to the
    // handle are ever made.
    unsafe { s.as_ref() }
}

/// The handle behind a checked `null(s, ..)`.
fn present(handle: Option<&NvVideoSession>) -> Result<&NvVideoSession, VideoError> {
    handle.ok_or_else(|| VideoError::Invalid("session is NULL".into()))
}

/// # Safety
/// `s` NULL or a live session; `data` readable for `len` bytes, for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_decode(
    s: *mut NvVideoSession,
    data: *const u8,
    len: usize,
    timestamp: u64,
    flags: u32,
) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_decode",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            if data.is_null() && len != 0 {
                return Err(VideoError::Invalid("data NULL with nonzero length".into()));
            }
            let d = if len == 0 {
                &[][..]
            } else {
                // SAFETY: non-NULL, readable for `len` bytes (# Safety).
                unsafe { std::slice::from_raw_parts(data, len) }
            };
            let shown = present(handle)?.session()?.decode(d, timestamp, flags)?;
            // ABI v2: the displayable pictures this unit produced (callers test rc < 0).
            i32::try_from(shown).map_err(|error| {
                VideoError::Invalid(format!("unit produced more than 2^31 pictures: {error}"))
            })
        },
    )
}

/// # Safety
/// `s` NULL or a live session, `pic` NULL or writable, for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_next_picture(
    s: *mut NvVideoSession,
    pic: *mut NvVideoPicture,
) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_next_picture",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            null(pic, "pic")?;
            let Some((id, timestamp, i)) = present(handle)?.session()?.next_picture() else {
                return Ok(0);
            };
            let p = NvVideoPicture {
                id,
                timestamp,
                width: i.width,
                height: i.height,
                crop_left: i.crop[0],
                crop_top: i.crop[1],
                crop_width: i.crop[2],
                crop_height: i.crop[3],
                bit_depth: i.bit_depth,
                color_primaries: i.color_primaries,
                transfer: i.transfer,
                matrix: i.matrix,
                full_range: u32::from(i.full_range),
                keyframe: u32::from(i.keyframe),
                reserved: [0; 5],
            };
            // SAFETY: checked non-NULL, writable for the call (# Safety).
            unsafe { pic.write(p) };
            Ok(1)
        },
    )
}

/// # Safety
/// `s` NULL or a live session, `metadata` NULL or writable, for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_picture_hdr(
    s: *mut NvVideoSession,
    id: u64,
    metadata: *mut NvVideoHdrMetadata,
) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_picture_hdr",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            null(metadata, "metadata")?;
            let m = present(handle)?.session()?.picture_hdr(id)?;
            // SAFETY: checked non-NULL, writable for the call (# Safety).
            unsafe { metadata.write(NvVideoHdrMetadata::from(m)) };
            Ok(0)
        },
    )
}

/// # Safety
/// `s` NULL or a live session, `width`/`height` NULL or writable, for the call.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_video_required_size(
    s: *mut NvVideoSession,
    width: *mut u32,
    height: *mut u32,
) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_required_size",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            null(width, "width")?;
            null(height, "height")?;
            let (w, h) = present(handle)?.session()?.required_size().ok_or_else(|| {
                VideoError::Invalid("no unit of this session returned NV_VIDEO_E_RESIZE".into())
            })?;
            // SAFETY: checked non-NULL, writable for the call (# Safety).
            unsafe { width.write(w) };
            // SAFETY: as `width`.
            unsafe { height.write(h) };
            Ok(0)
        },
    )
}

/// # Safety
/// `s` NULL or a live session for the call.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn nv_video_flush(s: *mut NvVideoSession) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_flush",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            present(handle)?.session()?.flush();
            Ok(0)
        },
    )
}

/// # Safety
/// `s` NULL or a live session, `target` NULL or readable, `release_fence_fd`
/// NULL or writable, for the call; every non-negative fd in `target` is open
/// for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_copy_out(
    s: *mut NvVideoSession,
    id: u64,
    target: *const NvVideoTarget,
    release_fence_fd: *mut i32,
) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_copy_out",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            null(target, "target")?;
            null(release_fence_fd, "release_fence_fd")?;
            // SAFETY: checked non-NULL, readable for the call (# Safety).
            let t = unsafe { &*target };
            let target = Target {
                dmabuf_fd: [t.dmabuf_fd[0], t.dmabuf_fd[1], t.cr_dmabuf_fd],
                offset: [t.offset[0], t.offset[1], t.cr_offset],
                pitch: [t.pitch[0], t.pitch[1], t.cr_pitch],
                modifier: t.modifier,
                format: t.format,
                width: t.width,
                height: t.height,
                acquire_fence_fd: t.acquire_fence_fd,
            };
            let mut session = present(handle)?.session()?;
            // SAFETY: the target's non-negative fds are open for the call (#
            // Safety).
            let fd = unsafe { session.copy_out(id, &target) }?;
            // SAFETY: checked non-NULL, writable for the call (# Safety).
            unsafe { release_fence_fd.write(fd.map_or(-1, std::os::fd::IntoRawFd::into_raw_fd)) };
            Ok(0)
        },
    )
}

/// # Safety
/// `s` NULL or a live session for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nv_video_picture_release(s: *mut NvVideoSession, id: u64) -> i32 {
    // SAFETY: `s` is NULL or a live session for the call (# Safety).
    let handle = unsafe { session_ref(s) };
    guard(
        "nv_video_picture_release",
        handle.map(|h| -> &dyn Failable { h }),
        || {
            null(s, "session")?;
            present(handle)?.session()?.release(id)?;
            Ok(0)
        },
    )
}

/// This thread's last error message (NUL-terminated, owned by the library,
/// valid until this thread's next failing call), or NULL if the guard caught
/// a panic here.
#[unsafe(no_mangle)]
pub extern "C" fn nv_video_last_error() -> *const std::ffi::c_char {
    crate::ffi::guard("nv_video_last_error", None, std::ptr::null, || {
        LAST_ERROR.with(|l| l.borrow().as_ptr())
    })
}

// The C struct keeps its v1 size (the Cr-plane fields took over reserved words).
const _: () = assert!(size_of::<NvVideoTarget>() == 80);
// ABI v2 structs: fixed C layout, no implicit padding.
const _: () = assert!(size_of::<NvVideoMasteringDisplay>() == 24);
const _: () = assert!(size_of::<NvVideoHdrMetadata>() == 64);

#[cfg(test)]
mod tests {
    use super::super::hdr::{HdrStaticMetadata, MasteringDisplay};
    use super::NvVideoHdrMetadata;

    #[test]
    fn hdr_metadata_conversion() {
        let m = HdrStaticMetadata {
            mastering: Some(MasteringDisplay {
                primaries_x: [34000, 13250, 7500],
                primaries_y: [16000, 34500, 3000],
                white_x: 15635,
                white_y: 16450,
                max_luminance: 10_000_000,
                min_luminance: 1,
            }),
            content_light: Some((1000, 400)),
        };
        let c = NvVideoHdrMetadata::from(m);
        assert_eq!((c.has_mastering_display, c.has_content_light_level), (1, 1));
        assert_eq!(
            c.mastering_display.display_primaries_x,
            [34000, 13250, 7500]
        );
        assert_eq!(c.mastering_display.max_luminance, 10_000_000);
        assert_eq!(
            (c.max_content_light_level, c.max_pic_average_light_level),
            (1000, 400)
        );
        let e = NvVideoHdrMetadata::from(HdrStaticMetadata::default());
        assert_eq!((e.has_mastering_display, e.has_content_light_level), (0, 0));
        assert_eq!(e, NvVideoHdrMetadata::default());
    }
}
