// SPDX-License-Identifier: MIT
//! Safe calls into the nv_video.h C ABI for the host decode tests.
//!
//! The tests drive the decoder through its exported C functions, as the Codec2 component does.
//! This module owns the raw handles and turns every pointer argument into a Rust type, so the
//! test programs hold no `unsafe`:
//! - [`Device`] and [`Session`] own the handles `nv_video_device_open` / `nv_video_session_create`
//!   return and release them in `Drop`; a session borrows its device, so it cannot outlive it,
//!   and its calls take `&mut self` (the header requires each session to be serialised).
//! - Descriptors in an `NvVideoTarget` must be open for the call: `copy_out` checks that each
//!   one is -1 or one of the descriptors the caller lends it.
//! - The release fence `copy_out` returns is the caller's ([`OwnedFd`]).
//!
//! Single-threaded: the handles hold raw pointers and are neither `Send` nor `Sync`.
use nv_bionic_native::video::NvVideoDeviceInfo;
use nv_bionic_native::video::ffi::{
    NvVideoDevice, NvVideoHdrMetadata, NvVideoPicture, NvVideoSession, NvVideoSessionInfo,
    NvVideoTarget, nv_video_copy_out, nv_video_decode, nv_video_device_close, nv_video_device_open,
    nv_video_last_error, nv_video_next_picture, nv_video_picture_hdr, nv_video_picture_release,
    nv_video_session_create, nv_video_session_destroy,
};
use std::ffi::CStr;
use std::marker::PhantomData;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr;

/// The calling thread's last nv_video error message, copied out.
pub(crate) fn last_error() -> String {
    let message = nv_video_last_error();
    if message.is_null() {
        return "(no error message)".into();
    }
    // SAFETY: non-null; nv_video_last_error returns the calling thread's last-error CString
    // (NUL-terminated, video/ffi.rs LAST_ERROR), valid until the next nv_video call on this
    // thread; it is copied into a String before any other call.
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

/// An open video device; closed on drop.
pub(crate) struct Device {
    raw: ptr::NonNull<NvVideoDevice>,
}
impl Device {
    /// The C status and, on success, the device and its info.
    pub(crate) fn open() -> (i32, Option<(Self, NvVideoDeviceInfo)>) {
        let mut raw = ptr::null_mut();
        let mut info = NvVideoDeviceInfo {
            abi_version: 0,
            pci_vendor: 0,
            pci_device: 0,
            chipset: 0,
            nvdec_class: 0,
            nvdec_engines: 0,
            name: [0; 64],
        };
        // SAFETY: both outputs are exclusive, writable values of the types the call writes.
        let status = unsafe { nv_video_device_open(&mut raw, &mut info) };
        let device = (status >= 0)
            .then(|| ptr::NonNull::new(raw))
            .flatten()
            .map(|raw| (Self { raw }, info));
        (status, device)
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: `raw` came from nv_video_device_open and is closed exactly once, here; every
        // Session borrowing this device has been dropped (lifetime).
        unsafe { nv_video_device_close(self.raw.as_ptr()) };
    }
}

/// One decode session on a [`Device`]; destroyed on drop.
pub(crate) struct Session<'d> {
    raw: ptr::NonNull<NvVideoSession>,
    _device: PhantomData<&'d Device>,
}
impl<'d> Session<'d> {
    /// `info.codec_config`/`codec_config_len` are replaced by `config` (null and 0 when empty).
    pub(crate) fn create(
        device: &'d Device,
        mut info: NvVideoSessionInfo,
        config: &[u8],
    ) -> (i32, Option<Self>) {
        (info.codec_config, info.codec_config_len) = if config.is_empty() {
            (ptr::null(), 0)
        } else {
            (config.as_ptr(), config.len())
        };
        let mut raw = ptr::null_mut();
        // SAFETY: the device is live (borrowed); `info` points to a full session info whose
        // codec_config is null or valid for codec_config_len bytes (set above from `config`,
        // which outlives the call); `raw` is an exclusive output.
        let status = unsafe { nv_video_session_create(device.raw.as_ptr(), &info, &mut raw) };
        let session = (status >= 0)
            .then(|| ptr::NonNull::new(raw))
            .flatten()
            .map(|raw| Self {
                raw,
                _device: PhantomData,
            });
        (status, session)
    }
    /// One access unit, or `None` to drain (null, 0).
    pub(crate) fn decode(&mut self, unit: Option<&[u8]>, timestamp: u64, flags: u32) -> i32 {
        let (data, len) = unit.map_or((ptr::null(), 0), |u| (u.as_ptr(), u.len()));
        // SAFETY: live session, exclusively borrowed; `data` is null with len 0 or valid for
        // `len` bytes for the call.
        unsafe { nv_video_decode(self.raw.as_ptr(), data, len, timestamp, flags) }
    }
    /// The C status (1 = a picture, 0 = none) and the picture it wrote (all zero otherwise).
    pub(crate) fn next_picture(&mut self) -> (i32, NvVideoPicture) {
        let mut picture = NvVideoPicture {
            id: 0,
            timestamp: 0,
            width: 0,
            height: 0,
            crop_left: 0,
            crop_top: 0,
            crop_width: 0,
            crop_height: 0,
            bit_depth: 0,
            color_primaries: 0,
            transfer: 0,
            matrix: 0,
            full_range: 0,
            keyframe: 0,
            reserved: [0; 5],
        };
        // SAFETY: live session, exclusively borrowed; `picture` is an exclusive, initialised
        // NvVideoPicture the call may overwrite.
        let status = unsafe { nv_video_next_picture(self.raw.as_ptr(), &mut picture) };
        (status, picture)
    }
    pub(crate) fn picture_hdr(&mut self, id: u64) -> (i32, NvVideoHdrMetadata) {
        let mut metadata = NvVideoHdrMetadata::default();
        // SAFETY: live session, exclusively borrowed; `metadata` is an exclusive output.
        let status = unsafe { nv_video_picture_hdr(self.raw.as_ptr(), id, &mut metadata) };
        (status, metadata)
    }
    /// Copies picture `id` into `target`. Every descriptor in `target` must be -1 or one of
    /// `fds` (panics otherwise), which keeps them open for the call. Returns the C status and
    /// the release fence, if the call produced one.
    pub(crate) fn copy_out(
        &mut self,
        id: u64,
        target: &NvVideoTarget,
        fds: &[BorrowedFd<'_>],
    ) -> (i32, Option<OwnedFd>) {
        let lent = |fd: i32| fd == -1 || fds.iter().any(|b| b.as_raw_fd() == fd);
        let used = [
            target.dmabuf_fd[0],
            target.dmabuf_fd[1],
            target.cr_dmabuf_fd,
            target.acquire_fence_fd,
        ];
        assert!(
            used.into_iter().all(lent),
            "target descriptors {used:?} not lent"
        );
        let mut fence = -1;
        // SAFETY: live session, exclusively borrowed; `target` is a full NvVideoTarget whose
        // descriptors are -1 or borrowed open for the call (checked above); `fence` is an
        // exclusive i32 output.
        let status = unsafe { nv_video_copy_out(self.raw.as_ptr(), id, target, &mut fence) };
        let fence = (fence >= 0).then(|| {
            // SAFETY: a non-negative release fence is a new descriptor the call hands to the
            // caller (`into_raw_fd`, video/ffi.rs nv_video_copy_out); nothing else owns it.
            unsafe { OwnedFd::from_raw_fd(fence) }
        });
        (status, fence)
    }
    pub(crate) fn picture_release(&mut self, id: u64) -> i32 {
        // SAFETY: live session, exclusively borrowed; ids are looked up, never dereferenced.
        unsafe { nv_video_picture_release(self.raw.as_ptr(), id) }
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: `raw` came from nv_video_session_create and is destroyed exactly once, here.
        unsafe { nv_video_session_destroy(self.raw.as_ptr()) };
    }
}

/// poll(2) for POLLIN with a 2 s timeout, leaving `fd` open; exits 3 if it does not signal.
pub(crate) fn wait_fd_keep(fd: BorrowedFd<'_>) {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    unsafe extern "C" {
        fn poll(fds: *mut PollFd, count: std::ffi::c_ulong, timeout: i32) -> i32;
    }
    let mut p = PollFd {
        fd: fd.as_raw_fd(),
        events: 1,
        revents: 0,
    };
    // SAFETY: `p` is one exclusive, initialised `struct pollfd` (poll(2): int fd, short events,
    // short revents) and the count is 1; poll only writes `revents`.
    if unsafe { poll(&mut p, 1, 2000) } != 1 {
        eprintln!("release fence did not signal");
        std::process::exit(3);
    }
}
/// [`wait_fd_keep`], then closes the fence; nothing for `None`.
pub(crate) fn wait_fd(fd: Option<OwnedFd>) {
    if let Some(fd) = fd {
        wait_fd_keep(std::os::fd::AsFd::as_fd(&fd));
        drop(fd);
    }
}
