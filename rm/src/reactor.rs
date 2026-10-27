// SPDX-License-Identifier: MIT
//! Linux readiness primitives for caller-side waits and the foreign watcher.
//!
//! RM dataless event readiness is consumed by poll itself. Exactly one thread
//! polls a given RM descriptor at a time (waiters.rs leader, or the watcher's
//! own listener); never epoll it or share it between concurrent pollers.
//! eventfd is only our wake/cancellation signal, not a GPU completion signal.
use crate::rm::{Error, Result};
use std::{
    ffi::{c_int, c_short, c_void},
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    time::Duration,
};

// glibc sys/poll.h uses unsigned long; bionic bits/poll.h uses unsigned int.
#[cfg(target_os = "android")]
type Nfds = u32;
#[cfg(not(target_os = "android"))]
type Nfds = std::ffi::c_ulong;
#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}
unsafe extern "C" {
    fn eventfd(init: u32, flags: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn ppoll(
        fds: *mut PollFd,
        count: Nfds,
        timeout: *const TimeSpec,
        sigmask: *const c_void,
    ) -> c_int;
}
/// struct timespec of the selected libc ABI: time_t and long are both the
/// pointer width on bionic and on non-time64 glibc (the plain ppoll symbol).
#[repr(C)]
struct TimeSpec {
    seconds: std::ffi::c_long,
    nanoseconds: std::ffi::c_long,
}
/// Outcome of one dispatch: RC/EINTR asks for an error-state scan; timed_out
/// means the bounded wait expired with nothing ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dispatch {
    pub rc_ready: bool,
    pub timed_out: bool,
}
pub(crate) struct WakeFd(OwnedFd);
impl WakeFd {
    pub fn new() -> Result<Self> {
        // Linux UAPI EFD_CLOEXEC | EFD_NONBLOCK. The RM event FD stays blocking.
        // SAFETY: no pointer arguments.
        let fd = unsafe { eventfd(0, 0o2000000 | 0o4000) };
        if fd < 0 {
            return Err(Error::Io(
                "create RM wakeup eventfd",
                io::Error::last_os_error(),
            ));
        }
        // SAFETY: a successful eventfd returns a new descriptor nobody else owns.
        Ok(Self(unsafe { OwnedFd::from_raw_fd(fd) }))
    }
    pub fn notify(&self) -> Result<()> {
        let value = 1u64;
        loop {
            // SAFETY: writes 8 bytes from a live local u64 to an owned fd.
            let n = unsafe { write(self.as_raw_fd(), (&raw const value).cast(), 8) };
            if n == 8 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            match e.kind() {
                io::ErrorKind::Interrupted => continue,
                // An already saturated eventfd is already readable.
                io::ErrorKind::WouldBlock => return Ok(()),
                _ => return Err(Error::Io("wake RM waiter", e)),
            }
        }
    }
    pub fn drain(&self) -> Result<()> {
        let mut value = 0u64;
        loop {
            // SAFETY: reads at most 8 bytes into a live local u64 from an owned fd.
            let n = unsafe { read(self.as_raw_fd(), (&raw mut value).cast(), 8) };
            if n == 8 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            match e.kind() {
                io::ErrorKind::Interrupted => continue,
                io::ErrorKind::WouldBlock => return Ok(()),
                _ => return Err(Error::Io("consume RM wakeup", e)),
            }
        }
    }
}
impl AsRawFd for WakeFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_raw_fd()
    }
}

/// One blocking wait, no timer. Owners retain all descriptors across this call.
/// POLLERR/HUP on a sync_file is readiness: its authoritative status is read by
/// the payload owner. Invalid descriptors are an ownership error, not readiness.
#[cfg(test)]
pub(crate) fn wait(fds: &[RawFd]) -> Result<()> {
    wait_with_devices(fds, None).map(|_| ())
}

#[cfg(test)]
pub(crate) fn wait_with_devices(fds: &[RawFd], devices: Option<(RawFd, RawFd)>) -> Result<bool> {
    Readiness::default().dispatch(fds, devices, true)
}

/// The owner reuses syscall storage; descriptor ownership remains in State.
#[derive(Default)]
pub(crate) struct Readiness {
    fds: Vec<PollFd>,
    interrupted: bool,
}
impl Readiness {
    /// Only valid for the descriptor set retained through the last dispatch.
    pub(crate) fn observed(&self, fd: RawFd) -> bool {
        self.interrupted
            || self
                .fds
                .iter()
                .any(|p| p.fd == fd && p.revents & (1 | 2 | 0x08 | 0x10) != 0)
    }

    /// Block until readiness (None = no timeout) or return immediately when
    /// may_block is false. Returns whether RC/EINTR requests an error scan.
    pub(crate) fn dispatch(
        &mut self,
        fds: &[RawFd],
        devices: Option<(RawFd, RawFd)>,
        may_block: bool,
    ) -> Result<bool> {
        let timeout = if may_block {
            None
        } else {
            Some(Duration::ZERO)
        };
        self.dispatch_timeout(fds, devices, timeout)
            .map(|outcome| outcome.rc_ready)
    }

    /// One ppoll with an optional relative timeout (nanosecond resolution, so
    /// a caller's finite Vulkan deadline is not rounded to milliseconds).
    pub(crate) fn dispatch_timeout(
        &mut self,
        fds: &[RawFd],
        devices: Option<(RawFd, RawFd)>,
        timeout: Option<Duration>,
    ) -> Result<Dispatch> {
        // Keep syscall scratch storage across dispatches, but rebuild each watch
        // from the owner's current retained descriptors and clear old revents.
        self.interrupted = false;
        self.fds.clear();
        self.fds.extend(fds.iter().map(|&fd| PollFd {
            fd,
            events: 1 | 2,
            revents: 0,
        }));
        let fds = &mut self.fds;
        let count = Nfds::try_from(fds.len())
            .map_err(|error| Error::Contract(format!("too many RM wait descriptors: {error}")))?;
        let spec = timeout.map(|t| TimeSpec {
            seconds: std::ffi::c_long::try_from(t.as_secs()).unwrap_or(std::ffi::c_long::MAX),
            nanoseconds: std::ffi::c_long::from(
                i32::try_from(t.subsec_nanos()).expect("subsec_nanos is below 10^9"),
            ),
        });
        let pointer = spec.as_ref().map_or(std::ptr::null(), |spec| {
            std::ptr::from_ref::<TimeSpec>(spec)
        });
        // SAFETY: `count` pollfds in the live Vec, a NULL or live local
        // timespec, and no signal mask (the app's mask stays untouched).
        let n = unsafe { ppoll(fds.as_mut_ptr(), count, pointer, std::ptr::null()) };
        if n >= 0 {
            if fds.iter().any(|p| p.revents & 0x20 != 0) {
                return Err(Error::Contract(
                    "invalid descriptor in RM event reactor".into(),
                ));
            }
            // nvidia_poll returns HUP for a removed GPU and ERR for an
            // invalid/unavailable event device (public nv.c). This is an
            // authoritative failure. Foreign sync_file errors are instead
            // decoded by their payload owner during the next condition check.
            if fds.iter().any(|p| {
                devices.is_some_and(|(host, rc)| p.fd == host || p.fd == rc)
                    && p.revents & (0x08 | 0x10) != 0
            }) {
                return Err(Error::DeviceLost(
                    "RM notification device failed or disconnected".into(),
                ));
            }
            return Ok(Dispatch {
                rc_ready: fds
                    .iter()
                    .any(|p| devices.is_some_and(|(_, rc)| p.fd == rc) && p.revents & (1 | 2) != 0),
                timed_out: n == 0,
            });
        }
        let e = io::Error::last_os_error();
        // RM poll consumes readiness while evaluating descriptors. Even on
        // EINTR return to predicate checking before attempting another poll.
        if e.kind() == io::ErrorKind::Interrupted {
            self.interrupted = true;
            return Ok(Dispatch {
                rc_ready: true,
                timed_out: false,
            });
        }
        Err(Error::Io("wait RM notifications", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn reused_poll_storage_drops_old_watches_and_readiness() {
        let host = WakeFd::new().unwrap();
        let rc = WakeFd::new().unwrap();
        let job = WakeFd::new().unwrap();
        let devices = Some((host.as_raw_fd(), rc.as_raw_fd()));
        let all = [host.as_raw_fd(), rc.as_raw_fd(), job.as_raw_fd()];
        let mut readiness = Readiness::default();
        rc.notify().unwrap();
        assert!(readiness.dispatch(&all, devices, false).unwrap());
        let capacity = readiness.fds.capacity();
        // RC remains readable, but removing its watch must remove its old
        // revents too. Reusing scratch is not retaining a descriptor owner.
        assert!(
            !readiness
                .dispatch(&[job.as_raw_fd()], devices, false)
                .unwrap()
        );
        assert!(!readiness.dispatch(&[], devices, false).unwrap());
        rc.drain().unwrap();
        host.notify().unwrap();
        assert!(!readiness.dispatch(&all, devices, false).unwrap());
        host.drain().unwrap();
        assert!(!readiness.dispatch(&all, devices, false).unwrap());
        rc.notify().unwrap();
        assert!(readiness.dispatch(&all, devices, false).unwrap());
        assert_eq!(readiness.fds.capacity(), capacity);
    }
    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn interrupted_dispatch_invalidates_all_retained_foreign_watches() {
        // Exercise the exact state left by the EINTR syscall branch without
        // process-wide signal handlers or timing-sensitive signal injection.
        let mut readiness = Readiness {
            interrupted: true,
            ..Default::default()
        };
        assert!(readiness.observed(123));
        assert!(readiness.observed(456));
        readiness.dispatch(&[], None, false).unwrap();
        assert!(
            !readiness.observed(123),
            "next dispatch cannot inherit EINTR"
        );
        readiness.fds.push(PollFd {
            fd: 123,
            events: 1,
            revents: 0x08 | 0x10,
        });
        assert!(
            readiness.observed(123),
            "foreign errors require status query"
        );
        assert!(!readiness.observed(456));
    }
    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn only_rc_readiness_requests_error_dispatch() {
        let host = WakeFd::new().unwrap();
        let rc = WakeFd::new().unwrap();
        let jobs = WakeFd::new().unwrap();
        let foreign = WakeFd::new().unwrap();
        let devices = Some((host.as_raw_fd(), rc.as_raw_fd()));
        let fds = [
            host.as_raw_fd(),
            rc.as_raw_fd(),
            jobs.as_raw_fd(),
            foreign.as_raw_fd(),
        ];
        // Each poll has an already-readable FD, so a classification failure
        // cannot strand the test in the production infinite poll.
        for source in [&host, &jobs, &foreign] {
            source.notify().unwrap();
            assert!(!wait_with_devices(&fds, devices).unwrap());
            source.drain().unwrap();
        }
        rc.notify().unwrap();
        assert!(wait_with_devices(&fds, devices).unwrap());
        host.notify().unwrap();
        jobs.notify().unwrap();
        assert!(wait_with_devices(&fds, devices).unwrap());
        rc.drain().unwrap();
        assert!(!wait_with_devices(&fds, devices).unwrap());
        host.drain().unwrap();
        jobs.drain().unwrap();
        // Idle workers omit HOST but keep RC and jobs; retained HOST readiness
        // must not masquerade as an RC error when a new job wakes the worker.
        host.notify().unwrap();
        jobs.notify().unwrap();
        assert!(!wait_with_devices(&[rc.as_raw_fd(), jobs.as_raw_fd()], devices).unwrap());
        jobs.drain().unwrap();
        rc.notify().unwrap();
        assert!(wait_with_devices(&[rc.as_raw_fd(), jobs.as_raw_fd()], devices).unwrap());
    }
    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn wake_before_registration_is_retained_and_drained() {
        let wake = WakeFd::new().unwrap();
        for _ in 0..100 {
            wake.notify().unwrap();
        }
        wait(&[wake.as_raw_fd()]).unwrap();
        wake.drain().unwrap();
        // A second wake after consuming a coalesced set must still work.
        wake.notify().unwrap();
        wait(&[wake.as_raw_fd()]).unwrap();
        wake.drain().unwrap();
    }
    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn invalid_fd_is_reported_instead_of_a_busy_loop() {
        // Large positive non-open descriptor (negative means ignored to poll).
        assert!(wait(&[c_int::MAX]).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "ppoll on real descriptors")]
    fn device_disconnect_is_loss_but_foreign_hup_rechecks_its_owner() {
        let (reader, writer) = std::os::unix::net::UnixStream::pair().unwrap();
        drop(writer);
        let fd = reader.as_raw_fd();
        assert!(wait_with_devices(&[fd], None).is_ok());
        assert!(matches!(
            wait_with_devices(&[fd], Some((fd, fd))),
            Err(Error::DeviceLost(_))
        ));
    }
}
