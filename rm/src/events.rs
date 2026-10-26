// SPDX-License-Identifier: MIT
//! Public RM dataless OS-event notifications (NV0005 + Unix event ioctls).
//!
//! A cached pair of blocking GPU FDs per Device: HOST completion notifications
//! and channel RC errors have separate waitqueues. Only the worker may poll
//! these FDs: NVIDIA poll consumes/coalesces dataless readiness. HOST readiness
//! means recheck completion; RC readiness means check channel errors. Neither
//! event is itself evidence that a submitted job completed.
//! Authority: OGKM event_api.c, event_notification.c, osapi.c, nv.c and the
//! exact-release C probes. No SEMSURF or DRM dependency.
use crate::diag::diag;
use crate::{
    abi,
    compat::Compat,
    rm::{self, Device, Error, Result},
};
use std::{
    any::Any,
    fs::File,
    os::fd::{AsRawFd, RawFd},
    rc::Rc,
};

const HOST: u32 = 35;
const DATALESS: u32 = 0x1000_0000;
const SUBDEVICE: u32 = 0x2000_0000;
const NONSTALL: u32 = 0x0800_0000;

pub(crate) struct RmEvents {
    source: Rc<EventSource>,
}

pub(crate) struct EventSource {
    resources: Option<SourceResources>,
}
struct SourceResources {
    device: Rc<Device>,
    fd: File,
    error_fd: File,
    listener: u32,
    state: Lifecycle,
}

/// An RC listener retaining both its channel and the shared event source.
/// Drop before releasing the queue's channel resources. The retained owner must
/// actually own the channel (not merely carry its numeric handle).
pub(crate) struct ChannelEvents {
    resources: Option<ChannelResources>,
}
struct ChannelResources {
    source: Rc<EventSource>,
    channel: u32,
    listener: u32,
    _owner: Rc<dyn Any>,
}

#[derive(Default, Debug)]
struct Lifecycle {
    os_event: bool,
    listener: bool,
    notification: bool,
    error_os_event: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cleanup {
    Disable,
    Listener,
    OsEvent,
    ErrorOsEvent,
}
impl Lifecycle {
    /// Stop at the first failure. In particular, never close/free the OS event
    /// while a listener may still post through it (also on old kernel drivers).
    fn cleanup(&mut self, mut run: impl FnMut(Cleanup) -> Result<()>) -> Result<()> {
        for (active, operation) in [
            (&mut self.notification, Cleanup::Disable),
            (&mut self.listener, Cleanup::Listener),
            (&mut self.os_event, Cleanup::OsEvent),
            (&mut self.error_os_event, Cleanup::ErrorOsEvent),
        ] {
            if *active {
                run(operation)?;
                *active = false;
            }
        }
        Ok(())
    }
}

fn validate_compat(compat: Compat) -> Result<()> {
    for name in ["nv_ioctl_alloc_os_event_t", "nv_ioctl_free_os_event_t"] {
        compat.require_same_layout(name)?;
    }
    compat.pack("NV0005_ALLOC_PARAMETERS", &abi::EventAlloc::default())?;
    compat.pack(
        "NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS",
        &abi::EventNotification::default(),
    )?;
    compat.pack(
        "NV2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS",
        &abi::EventTrigger::default(),
    )?;
    for (name, value) in [
        ("NV_ESC_ALLOC_OS_EVENT", 206),
        ("NV_ESC_FREE_OS_EVENT", 207),
        ("NV01_EVENT", 5),
        ("NV01_EVENT_OS_EVENT", 0x79),
        ("NV01_EVENT_WITHOUT_EVENT_DATA", u64::from(DATALESS)),
        ("NV01_EVENT_SUBDEVICE_SPECIFIC", u64::from(SUBDEVICE)),
        ("NV01_EVENT_NONSTALL_INTR", u64::from(NONSTALL)),
        ("NV2080_NOTIFIERS_FIFO_EVENT_MTHD", u64::from(HOST)),
        ("NV2080_CTRL_CMD_EVENT_SET_NOTIFICATION", 0x20800301),
        ("NV2080_CTRL_CMD_EVENT_SET_TRIGGER_FIFO", 0x20800308),
        ("NV2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE", 0),
        ("NV2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT", 2),
    ] {
        compat.require_constant(name, value)?;
    }
    Ok(())
}

fn notify_index(instance: u32, index: u32, nonstall: bool) -> Result<u32> {
    if instance > 255 || index > 65535 {
        return Err(Error::Contract(
            "RM event subdevice/notifier index exceeds public NV0005 fields".into(),
        ));
    }
    Ok(DATALESS | SUBDEVICE | (instance << 16) | index | if nonstall { NONSTALL } else { 0 })
}

fn rc_index(compat: Compat, class: u32) -> Result<u32> {
    let constant = match class {
        0xa06f => "NVA06F_NOTIFIERS_RC",
        0xa16f => "NVA16F_NOTIFIERS_RC",
        0xb06f => "NVB06F_NOTIFIERS_RC",
        0xc06f => "NVC06F_NOTIFIERS_RC",
        0xc36f => "NVC36F_NOTIFIERS_RC",
        0xc46f => "NVC46F_NOTIFIERS_RC",
        0xc56f => "NVC56F_NOTIFIERS_RC",
        0xc86f => "NVC86F_NOTIFIERS_RC",
        0xc96f => "NVC96F_NOTIFIERS_RC",
        0xca6f => "NVCA6F_NOTIFIERS_RC",
        _ => {
            return Err(Error::Contract(format!(
                "no public RC event contract for channel class {class:#x}"
            )));
        }
    };
    compat.require_constant(constant, 0)?;
    Ok(0)
}

fn os_event(device: &Device, fd: RawFd, allocate: bool) -> Result<()> {
    let mut params = abi::OsEvent {
        client: device.client.root,
        device: device.handle,
        fd: u32::try_from(fd)
            .map_err(|error| Error::Contract(format!("negative RM event FD: {error}")))?,
        status: 0,
    };
    let operation = if allocate {
        "allocate RM OS event"
    } else {
        "free RM OS event"
    };
    // ALLOC associates the current ioctl file with the numeric FD key. Merely
    // sending fd in a control-FD ioctl would attach the wrong waitqueue.
    rm::exchange(fd, if allocate { 206 } else { 207 }, &mut params, operation)?;
    rm::status(operation, params.status)
}
fn notification(device: &Device, action: u32) -> Result<()> {
    device.client.control(
        device.subdevice,
        0x20800301,
        &mut abi::EventNotification {
            event: HOST,
            action,
            ..Default::default()
        },
    )
}
fn listener(device: &Device, parent: u32, fd: RawFd, index: u32) -> Result<u32> {
    device.client.allocate(
        parent,
        5,
        &mut abi::EventAlloc {
            parent_client: device.client.root,
            source: 0,
            class: 0x79,
            notify_index: index,
            // NvP64 holding the descriptor, sign-extended as the C ABI does
            // for an int stored in a 64-bit field (nv-unix-nvos-params-wrappers.h).
            data: i64::from(fd).cast_unsigned(),
        },
    )
}

impl RmEvents {
    pub(crate) fn new(device: &Rc<Device>) -> Result<Self> {
        if let Some(source) = device.events.borrow().upgrade() {
            return Ok(Self { source });
        }
        validate_compat(device.client.compat)?;
        let index = notify_index(device.subdevice_instance, HOST, true)?;
        // std::fs opens CLOEXEC without O_NONBLOCK. Public nvidia_poll only
        // installs its waitqueue when O_NONBLOCK is clear. Register a fresh GPU
        // FD to this RM client; never duplicate the device's general-purpose FD.
        let fd = device.client.gpu_fd(device.card.minor)?;
        // A distinct file description, not dup(): kernel dataless readiness is
        // held on the ioctl file's queue. HOST wakes must not consume RC wakes.
        let error_fd = device.client.gpu_fd(device.card.minor)?;
        let mut source = EventSource {
            resources: Some(SourceResources {
                device: Rc::clone(device),
                fd,
                error_fd,
                listener: 0,
                state: Lifecycle::default(),
            }),
        };
        let r = source
            .resources
            .as_mut()
            .expect("constructed with resources above");
        os_event(device, r.fd.as_raw_fd(), true)?;
        r.state.os_event = true;
        r.listener = listener(device, device.subdevice, r.fd.as_raw_fd(), index)?;
        r.state.listener = true;
        // A failed control may have reached firmware before failing locally.
        // Attempt DISABLE during rollback too, then remove the actual listener.
        r.state.notification = true;
        notification(device, 2)?;
        os_event(device, r.error_fd.as_raw_fd(), true)?;
        r.state.error_os_event = true;
        let source = Rc::new(source);
        *device.events.borrow_mut() = Rc::downgrade(&source);
        Ok(Self { source })
    }

    /// Only channel RC listeners post to this FD. It shares the cached source
    /// lifetime with HOST; the worker is the sole consumer of both descriptors.
    pub(crate) fn error_fd(&self) -> RawFd {
        self.source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped")
            .error_fd
            .as_raw_fd()
    }

    pub(crate) fn attach_channel<T: 'static>(
        &self,
        channel: u32,
        channel_class: u32,
        owner: Rc<T>,
    ) -> Result<ChannelEvents> {
        let r = self
            .source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped");
        let index = rc_index(r.device.client.compat, channel_class)?;
        let index = notify_index(r.device.subdevice_instance, index, false)?;
        let handle = listener(&r.device, channel, r.error_fd.as_raw_fd(), index)?;
        Ok(ChannelEvents {
            resources: Some(ChannelResources {
                source: Rc::clone(&self.source),
                channel,
                listener: handle,
                _owner: owner,
            }),
        })
    }
}
impl AsRawFd for RmEvents {
    fn as_raw_fd(&self) -> RawFd {
        self.source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped")
            .fd
            .as_raw_fd()
    }
}

/// A second, independent HOST completion descriptor on the same source. RM
/// posts each FIFO-event notification to every registered OS event, each with
/// its own dataless flag (public event_notification.c / nv.c), so its poller
/// never consumes the readiness of the waiters' device descriptor. The source's
/// shared REPEAT enable covers it; this listener never sends ENABLE/DISABLE.
pub(crate) struct HostListener {
    source: Rc<EventSource>,
    fd: Option<File>,
    listener: u32,
    os_event: bool,
}
impl RmEvents {
    pub(crate) fn host_listener(&self) -> Result<HostListener> {
        let r = self
            .source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped");
        let index = notify_index(r.device.subdevice_instance, HOST, true)?;
        let fd = r.device.client.gpu_fd(r.device.card.minor)?;
        let raw = fd.as_raw_fd();
        let mut owned = HostListener {
            source: Rc::clone(&self.source),
            fd: Some(fd),
            listener: 0,
            os_event: false,
        };
        os_event(&r.device, raw, true)?;
        owned.os_event = true;
        owned.listener = listener(&r.device, r.device.subdevice, raw, index)?;
        Ok(owned)
    }
    /// Thread-independent broadcast handle (see trigger()).
    pub(crate) fn trigger_port(&self) -> TriggerPort {
        let device = &self
            .source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped")
            .device;
        TriggerPort {
            port: device.client.port(),
            subdevice: device.subdevice,
        }
    }
}
impl AsRawFd for HostListener {
    fn as_raw_fd(&self) -> RawFd {
        self.fd.as_ref().expect("live HOST listener").as_raw_fd()
    }
}
impl Drop for HostListener {
    fn drop(&mut self) {
        let device = &self
            .source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped")
            .device;
        let raw = self.as_raw_fd();
        if self.listener != 0 {
            if let Err(error) = device.client.free(device.subdevice, self.listener) {
                // Keep the descriptor/OS event alive: RM may still post to it.
                diag!("RM extra HOST listener teardown failed; retaining it: {error}");
                std::mem::forget(self.fd.take());
                return;
            }
        }
        if self.os_event {
            if let Err(error) = os_event(device, raw, false) {
                diag!("RM extra HOST OS event free failed; retaining fd: {error}");
                std::mem::forget(self.fd.take());
            }
        }
    }
}

/// Copy/Send broadcast of the public HOST FIFO event (hEvent 0) used after a
/// shared opaque payload changes. Valid while the owning Bridge is open.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TriggerPort {
    port: crate::rm::ControlPort,
    subdevice: u32,
}
impl TriggerPort {
    pub(crate) fn trigger(&self) -> Result<()> {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        self.port.control(
            self.subdevice,
            0x20800308,
            &mut abi::EventTrigger { event: 0 },
        )
    }
}
impl Drop for EventSource {
    fn drop(&mut self) {
        let Some(mut r) = self.resources.take() else {
            return;
        };
        let result = r.state.cleanup(|operation| match operation {
            Cleanup::Disable => notification(&r.device, 0),
            Cleanup::Listener => r.device.client.free(r.device.subdevice, r.listener),
            Cleanup::OsEvent => os_event(&r.device, r.fd.as_raw_fd(), false),
            Cleanup::ErrorOsEvent => os_event(&r.device, r.error_fd.as_raw_fd(), false),
        });
        if let Err(error) = result {
            // Keep FD, device/client and listener backing alive together. A
            // cleanup failure is not permission to race old kernel posting.
            diag!("RM event teardown failed; retaining owned resources: {error}");
            std::mem::forget(r);
        }
    }
}
impl Drop for ChannelEvents {
    fn drop(&mut self) {
        let Some(r) = self.resources.take() else {
            return;
        };
        let device = &r
            .source
            .resources
            .as_ref()
            .expect("an event source holds its resources until it is dropped")
            .device;
        if let Err(error) = device.client.free(r.channel, r.listener) {
            diag!("RM channel event teardown failed; retaining channel and event source: {error}");
            std::mem::forget(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cleanup_is_transactional_and_retries_only_remaining_owners() {
        for failure in 0..4 {
            let mut state = Lifecycle {
                notification: true,
                listener: true,
                os_event: true,
                error_os_event: true,
            };
            let mut seen = Vec::new();
            assert!(
                state
                    .cleanup(|step| {
                        seen.push(step);
                        if seen.len() == failure + 1 {
                            Err(Error::Contract("injected teardown fault".into()))
                        } else {
                            Ok(())
                        }
                    })
                    .is_err()
            );
            assert_eq!(seen.len(), failure + 1);
            let mut retry = Vec::new();
            state
                .cleanup(|step| {
                    retry.push(step);
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                retry,
                [
                    Cleanup::Disable,
                    Cleanup::Listener,
                    Cleanup::OsEvent,
                    Cleanup::ErrorOsEvent
                ][failure..]
            );
            state.cleanup(|_| panic!("owner freed twice")).unwrap();
        }
    }
    #[test]
    fn partial_initialization_unwinds_only_acquired_resources() {
        for stage in 0..5 {
            let mut state = Lifecycle {
                os_event: stage >= 1,
                listener: stage >= 2,
                notification: stage >= 3,
                error_os_event: stage >= 4,
            };
            let mut seen = Vec::new();
            state
                .cleanup(|step| {
                    seen.push(step);
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                seen,
                [
                    (stage >= 3, Cleanup::Disable),
                    (stage >= 2, Cleanup::Listener),
                    (stage >= 1, Cleanup::OsEvent),
                    (stage >= 4, Cleanup::ErrorOsEvent),
                ]
                .into_iter()
                .filter_map(|(acquired, operation)| acquired.then_some(operation))
                .collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn event_indices_do_not_confuse_flags_channel_and_host() {
        assert_eq!(notify_index(0, 35, true).unwrap(), 0x38000023);
        assert_eq!(notify_index(2, 0, false).unwrap(), 0x30020000);
        assert!(notify_index(256, 0, true).is_err());
        assert!(notify_index(0, 65536, false).is_err());
    }
    #[test]
    fn exact_event_abi_does_not_require_semaphore_surfaces() {
        for version in ["535.309.01", "550.163.01", "580.178.04", "615.71.09"] {
            let compat = Compat::for_version(version).unwrap();
            validate_compat(compat).unwrap();
            assert_eq!(rc_index(compat, 0xa06f).unwrap(), 0);
            assert!(rc_index(compat, 0xdead006f).is_err());
            assert!(rc_index(compat, 0xc76f).is_err());
        }
    }

    #[test]
    #[ignore = "requires NVIDIA RM; bounded event setup/trigger only, no GPU submission; run alone"]
    fn host_trigger_cached_ownership_and_fd_cleanup() {
        use std::{
            sync::mpsc,
            time::{Duration, Instant},
        };
        let fd_count = || std::fs::read_dir("/proc/self/fd").unwrap().count();
        let before = fd_count();
        {
            let client = rm::Client::open().unwrap();
            let mut failures = Vec::new();
            let device = client
                .cards()
                .unwrap()
                .into_iter()
                .find_map(|card| match client.device(card) {
                    Ok(device) => Some(device),
                    Err(error) => {
                        failures.push(format!(
                            "{:04x}:{:02x}:{:02x}.{}: {error}",
                            card.pci.domain, card.pci.bus, card.pci.slot, card.pci.function
                        ));
                        None
                    }
                })
                .unwrap_or_else(|| panic!("no openable NVIDIA device: {}", failures.join("; ")));
            for _ in 0..4 {
                let cycle_before = fd_count();
                let events = RmEvents::new(&device).unwrap();
                assert_ne!(events.as_raw_fd(), events.error_fd());
                assert_eq!(fd_count(), cycle_before + 2, "source must own one FD pair");
                let alias = RmEvents::new(&device).unwrap();
                assert_eq!(events.as_raw_fd(), alias.as_raw_fd());
                assert_eq!(events.error_fd(), alias.error_fd());
                assert_eq!(fd_count(), cycle_before + 2, "cached pair was reopened");
                for fd in [events.as_raw_fd(), events.error_fd()] {
                    let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{fd}")).unwrap();
                    let flags = info
                        .lines()
                        .find_map(|l| {
                            l.strip_prefix("flags:")
                                .map(|v| u32::from_str_radix(v.trim(), 8).unwrap())
                        })
                        .unwrap();
                    assert_eq!(flags & 0o4000, 0, "RM poll FD must be blocking");
                    assert_ne!(flags & 0o2000000, 0, "RM event FD must be CLOEXEC");
                }
                drop(events);
                // Verify a remaining owner keeps the listener enabled, and a
                // post before poll remains pending. Watchdog is a CPU eventfd,
                // so a missing NVIDIA wake cannot leave this fixture blocked.
                let watchdog = crate::reactor::WakeFd::new().unwrap();
                let (done_tx, done_rx) = mpsc::channel();
                alias.trigger_port().trigger().unwrap();
                let elapsed = std::thread::scope(|scope| {
                    let watchdog = &watchdog;
                    scope.spawn(move || {
                        if done_rx.recv_timeout(Duration::from_millis(500)).is_err() {
                            watchdog.notify().unwrap();
                        }
                    });
                    let start = Instant::now();
                    let rc_ready = crate::reactor::wait_with_devices(
                        &[alias.as_raw_fd(), alias.error_fd(), watchdog.as_raw_fd()],
                        Some((alias.as_raw_fd(), alias.error_fd())),
                    )
                    .unwrap();
                    assert!(
                        !rc_ready,
                        "HOST broadcast reached the channel RC descriptor"
                    );
                    let elapsed = start.elapsed();
                    done_tx.send(()).ok();
                    elapsed
                });
                assert!(
                    elapsed < Duration::from_millis(250),
                    "NVIDIA event did not wake poll: {elapsed:?}"
                );
                drop(watchdog);
                drop(alias);
                assert_eq!(fd_count(), cycle_before, "source pair FD leak");
            }
        }
        assert_eq!(fd_count(), before, "event/client FD leak");
    }
}
