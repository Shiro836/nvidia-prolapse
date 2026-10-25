//! One NVDEC channel: TSG on the NVDEC0 runlist, SYNC context share, host
//! GPFIFO channel, decoder-class object (setup facts as in the V0 spike,
//! nvdec.rs), plus a 64-bit completion timeline on an RM semaphore surface so
//! that completion is observable without CPU polling: the CPU waits on a
//! sync_file of that surface, other channels acquire its payload.
//!
//! Stock reference: per-picture submission order and status handling,
//! stock libnvcuvid 615.71.09 at ELF 0x141d50; stock waits on its own shared
//! semaphore between the two NVDEC engines, we use one engine.
use super::{VResult, VideoError};
use crate::{
    fences::NativeFenceSurface,
    gpu::{method, AddressSpace, Object},
    rm::{Device, Mapping, Memory},
};
use std::{os::fd::AsRawFd, rc::Rc, time::Duration};

const NV2080_ENGINE_TYPE_NVDEC0: u32 = 0x13;
const ENTRIES: usize = 1024;
const COMMAND_BYTES: usize = 65536;

pub struct NvdecChannel {
    device: Rc<Device>,
    pub decoder_class: u32,
    /// Host GPFIFO class (relocation encoding, `super::reloc`).
    pub host_class: u32,
    ring: Mapping,
    command: Mapping,
    command_va: u64,
    doorbell: Mapping,
    /// Doorbell page behind BAR1 (HOPPER_USERMODE_A) rather than BAR0.
    bar1_doorbell: bool,
    token: u32,
    put: usize,
    cmd_off: usize,
    /// Last submitted timeline value.
    pub submitted: u64,
    pub(crate) timeline: NativeFenceSurface,
    objects: Vec<Rc<Object>>,
    _keep: Vec<Rc<Memory>>,
    _maps: Vec<crate::gpu::GpuMapping>,
}

impl NvdecChannel {
    pub fn new(device: &Rc<Device>, space: &AddressSpace, decoder_class: u32) -> VResult<Self> {
        let caps = crate::capabilities::query(device)?;
        let host_class = caps.channel;
        if !crate::gpu::modern_host(host_class)? {
            return Err(VideoError::Unsupported(format!(
                "NVDEC sessions need a Volta+ host class (have {host_class:#x})"
            )));
        }
        let queue_memory = device.allocate_memory(65536, true)?;
        let command_memory = device.allocate_vram(COMMAND_BYTES)?;
        let queue_va = space.map(&queue_memory)?;
        let command_va = space.map(&command_memory)?;
        let mut ring = queue_memory.map()?;
        let command = command_memory.map()?;
        let usermode = nvdec_usermode_class(&device.classes()?).ok_or_else(|| {
            VideoError::Unsupported(
                "NVDEC doorbell: RM advertises neither HOPPER_USERMODE_A (0xc661) nor \
                 VOLTA_USERMODE_A (0xc361)"
                    .into(),
            )
        })?;
        // rm.rs map_usermode_class: 0xc661 with {bBar1Mapping = 1, bPriv = 0},
        // 0xc361 with NULL parameters, as stock (nvdec_usermode_class).
        let doorbell = device.map_usermode_class(usermode)?;
        for i in 0..65536 / 4 {
            ring.write_u32(i * 4, 0)?;
        }
        // kernel_channel.c:741-785 (TSG engine == channel engine);
        // kernel_fifo_gp102.c:144-160 (NVDEC needs a SYNC subcontext).
        let mut group_params = crate::abi::ChannelGroup {
            vaspace: space.vaspace.handle,
            engine: NV2080_ENGINE_TYPE_NVDEC0,
            ..Default::default()
        };
        let group = Object::new(
            device,
            device.handle,
            0xa06c,
            &mut group_params,
            vec![Rc::clone(&space.vaspace)],
        )?;
        let mut context_params = crate::abi::ContextShare {
            vaspace: space.vaspace.handle,
            flags: 0,
            subcontext: 0,
        };
        let context = Object::new(
            device,
            group.handle,
            0x9067,
            &mut context_params,
            vec![Rc::clone(&group), Rc::clone(&space.vaspace)],
        )?;
        let mut channel_params = crate::abi::Channel {
            fifo_address: queue_va.address(),
            entries: ENTRIES as u32,
            context_share: context.handle,
            engine: NV2080_ENGINE_TYPE_NVDEC0,
            ..Default::default()
        };
        channel_params.userd_memory[0] = queue_memory.handle;
        channel_params.userd_offsets[0] = (ENTRIES * 8) as u64;
        let channel = Object::new(
            device,
            group.handle,
            host_class,
            &mut channel_params,
            vec![Rc::clone(&group), Rc::clone(&context)],
        )?;
        device.client.compat.require_constant(
            "NV2080_ENGINE_TYPE_NVDEC0",
            u64::from(NV2080_ENGINE_TYPE_NVDEC0),
        )?;
        let mut decoder_params = crate::abi::VideoDecoder {
            size: std::mem::size_of::<crate::abi::VideoDecoder>() as u32,
            ..Default::default()
        };
        let decoder = Object::new(
            device,
            channel.handle,
            decoder_class,
            &mut decoder_params,
            vec![Rc::clone(&channel)],
        )?;
        let timeline = NativeFenceSurface::new(device, space, channel.handle)?;
        let mut token = 0u32;
        device
            .client
            .control(channel.handle, 0xc36f0108, &mut token)?;
        device
            .client
            .control(group.handle, 0xa06c0101, &mut [1u8, 0, 0])?;
        let mut ch = Self {
            device: Rc::clone(device),
            decoder_class,
            host_class,
            ring,
            command,
            command_va: command_va.address(),
            doorbell,
            bar1_doorbell: usermode >= 0xc661,
            token,
            put: 0,
            cmd_off: 0,
            submitted: 0,
            timeline,
            objects: vec![decoder, channel, context, group],
            _keep: vec![queue_memory, command_memory],
            _maps: vec![queue_va, command_va],
        };
        let mut bind = Vec::new();
        method(&mut bind, 4, 0, &[decoder_class]); // SET_OBJECT
        ch.submit(bind)?;
        Ok(ch)
    }

    /// Append the timeline release and kick. Returns the timeline value that
    /// signals completion. Never waits.
    pub fn submit(&mut self, mut words: Vec<u32>) -> VResult<u64> {
        let value = self.submitted + 1;
        crate::gpu::timeline_release(
            &mut words,
            self.host_class,
            self.timeline.native64(),
            self.timeline.address(),
            value,
        )?;
        let bytes = words.len() * 4;
        if bytes > COMMAND_BYTES / 8 {
            return Err(VideoError::Invalid("NVDEC method list too long".into()));
        }
        // At most 4 pictures are in flight (descriptor ring), each well under
        // COMMAND_BYTES/8, so wrapping never overwrites unread commands.
        if self.cmd_off + bytes > COMMAND_BYTES {
            self.cmd_off = 0;
        }
        for (k, w) in words.iter().enumerate() {
            self.command.write_u32(self.cmd_off + k * 4, *w)?;
        }
        let entry =
            (self.command_va + self.cmd_off as u64) | ((words.len() as u64) << 42) | (1 << 41);
        self.cmd_off += bytes.next_multiple_of(256);
        self.timeline.publish_maximum(value)?;
        self.ring.write_u32(self.put * 8, entry as u32)?;
        self.ring
            .write_u32(self.put * 8 + 4, (entry >> 32) as u32)?;
        self.put = (self.put + 1) % ENTRIES;
        crate::platform::device_full_barrier();
        let put = self.put as u32;
        self.ring.write_u32(ENTRIES * 8 + 0x8c, put)?; // GPPut
        crate::platform::device_full_barrier();
        // Ring and USERD are VRAM written through BAR1 (write-combined). A BAR0
        // doorbell (pre-Hopper usermode) can overtake those posted writes, so
        // GP_PUT is read back through BAR1 first, exactly as the graphics
        // queue's kick (gpu.rs Queue::kick, measured stranded batches without it).
        if !self.bar1_doorbell {
            let visible = self.ring.read_u32(ENTRIES * 8 + 0x8c)?;
            if visible != put {
                return Err(VideoError::DeviceLost(format!(
                    "NVDEC GP_PUT read back {visible} after writing {put}; channel state unknown"
                )));
            }
            crate::platform::device_full_barrier();
        }
        self.doorbell.write_u32(0x90, self.token)?;
        crate::platform::device_full_barrier();
        self.submitted = value;
        Ok(value)
    }

    pub fn completed(&self) -> VResult<u64> {
        Ok(self.timeline.read_completed()?)
    }

    /// Block until `value` completed: sync_file of the timeline + poll(2), no
    /// spinning. A 2 s bound turns a hung engine into DEVICE_LOST.
    pub fn wait(&self, drm: &std::fs::File, value: u64) -> VResult<()> {
        if self.completed()? >= value {
            return Ok(());
        }
        let fd = self.timeline.context_fence(drm.as_raw_fd(), value)?;
        wait_sync_file(fd.as_raw_fd(), Duration::from_secs(2)).map_err(|e| {
            let gpget = self.ring.read_u32(ENTRIES * 8 + 0x88).unwrap_or(u32::MAX);
            VideoError::DeviceLost(format!(
                "NVDEC picture {value} did not complete ({e}; completed {:?}, GPGet {gpget}, put {})",
                self.completed().ok(),
                self.put
            ))
        })?;
        crate::platform::device_full_barrier();
        Ok(())
    }
}

/// Doorbell class of an NVDEC channel, as stock libnvcuvid chooses it: the
/// first class RM advertises from {HOPPER_USERMODE_A 0xc661, VOLTA_USERMODE_A
/// 0xc361} (never Turing/Ampere USERMODE). Stock: libnvcuvid 615.71.09 ELF
/// 0x6c940..0x6ca11, table at .rodata 0x10696b0 = {0xc661, 0xc361}. Observed in rmtrace on
/// GB202/615 (0xc661 + params), AD102/610.57.04 and GA106/550.107.02 (0xc361,
/// NULL params). Allocating 0xc661 with bBar1Mapping on a pre-Hopper GPU fails
/// NV_ERR_INVALID_ARGUMENT (usermode_api.c: no pBar1VF there, mem.c
/// memConstructCommon(NULL)): the 20261015 A2000 session_create failure.
pub(crate) fn nvdec_usermode_class(classes: &[u32]) -> Option<u32> {
    [0xc661, 0xc361].into_iter().find(|c| classes.contains(c))
}

#[repr(C)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}
unsafe extern "C" {
    fn poll(fds: *mut PollFd, n: std::ffi::c_ulong, timeout: i32) -> i32;
}

/// poll(POLLIN) on a sync_file until it signals or `timeout` elapses.
pub(crate) fn wait_sync_file(fd: i32, timeout: Duration) -> Result<(), String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let mut p = PollFd {
            fd,
            events: 1,
            revents: 0,
        };
        // SAFETY: one valid pollfd for the duration of the call.
        let r = unsafe { poll(&mut p, 1, left.as_millis().min(i32::MAX as u128) as i32) };
        if r > 0 {
            if p.revents & (8 | 32) != 0 {
                return Err(format!("sync_file poll error revents {:#x}", p.revents));
            }
            return Ok(());
        }
        if r == 0 {
            return Err(format!("sync_file not signalled after {timeout:?}"));
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::Interrupted {
            return Err(format!("poll: {e}"));
        }
    }
}

impl Drop for NvdecChannel {
    fn drop(&mut self) {
        let _ = self.timeline.unbind();
        if let Some(group) = self.objects.last() {
            let _ = self
                .device
                .client
                .control(group.handle, 0xa06c0101, &mut [0u8, 0, 0]);
        }
        while !self.objects.is_empty() {
            drop(self.objects.remove(0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::nvdec_usermode_class;
    #[test]
    fn usermode_class_matches_stock_per_generation() {
        // Usermode subsets of the per-generation class lists (0xc361 allocated
        // by stock on GA106/550 and AD102/610, 0xc661 on GB202/615 in rmtrace).
        let blackwell = [0xc361, 0xc461, 0xc561, 0xc661, 0xcfb0];
        let ada = [0xc361, 0xc461, 0xc561, 0xc9b0];
        let ampere = [0xc361, 0xc461, 0xc561, 0xc7b0];
        let turing = [0xc361, 0xc461, 0xc4b0];
        assert_eq!(nvdec_usermode_class(&blackwell), Some(0xc661));
        assert_eq!(nvdec_usermode_class(&ada), Some(0xc361));
        assert_eq!(nvdec_usermode_class(&ampere), Some(0xc361));
        assert_eq!(nvdec_usermode_class(&turing), Some(0xc361));
        assert_eq!(nvdec_usermode_class(&[0xc461, 0xc561]), None);
    }
}
