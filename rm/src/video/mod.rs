// SPDX-License-Identifier: MIT
//! nv_video (video/nv_video.h v1 entry points, v2 additions): NVDEC decode
//! sessions on the native RM backend. No NVIDIA userspace is loaded.
//!
//! Layering (see video/README.md for the codec plug-in contract):
//! - `channel`: one NVDEC GPFIFO channel + its completion timeline.
//! - `session`: codec-agnostic resources (DPB surface pool, 4-slot descriptor /
//!   bitstream ring, display queue, picture refcounts) and `copy_out`.
//! - `CodecDecoder`: per-codec parse → descriptor/method builder (`vp9`, `hevc`;
//!   AV1 / H.264 plug in the same way), in the logic crate `nv_video`, re-exported here.
//! - `ffi`: the C ABI.
//!
//! Stock behaviour this follows was observed in libnvcuvid 615.71.09 (RM
//! traces, disassembly); ELF addresses are cited, no code is copied.
pub(crate) mod channel;
pub mod ffi;
#[cfg(test)]
pub(crate) mod pretend;
pub(crate) mod session;

pub use nv_video::{
    ABI_VERSION, Allocator, BspCapsV2, BuildContext, CODEC_AV1, CODEC_H264, CODEC_HEVC, CODEC_VP9,
    CodecDecoder, DecodeJob, FORMAT_NV12, FORMAT_P010, FORMAT_YUV420P, Job, NVDEC_CLASSES,
    NvVideoCaps, NvVideoDeviceInfo, NvdecEngineCaps, NvdecGen, PictureInfo, SurfaceLayout,
    SurfaceRef, VResult, VideoError, av1, bits, bitstream, blackwell_gob_kinds, bytes_per_sample,
    caps, class_gap, engine_gob_kinds, h264, hdr, hevc, output_layout_verified, plane_kind_bits,
    reloc, surface_format_code, too_large, unsupported, untested_classes_allowed, vp9,
};

use crate::rm::{self, Client};
use std::rc::Rc;

/// The device handle: PCI identity + NVDEC class only. Sessions open their
/// own RM client so that a session can live on any thread (RM objects are
/// thread-confined `Rc`s).
#[derive(Clone, Debug)]
pub struct VideoDevice {
    /// PCI domain, bus, device and function of the GPU.
    pub pci: [u32; 4],
    pub(crate) info: NvVideoDeviceInfo,
    /// BSP caps of NVDEC0, the engine sessions submit to.
    pub(crate) engine0: NvdecEngineCaps,
    /// The raw BSP capsTbl NVDEC0's caps were decoded from, for diagnostics.
    pub engine0_caps_tbl: [u8; 8],
}

fn pci_string(p: [u32; 4]) -> String {
    rm::bdf_string(p)
}

/// Open the RM device of `pci` (or of NV_VIDEO_PCI, or else the first NVIDIA
/// GPU whose /dev/nvidiaN this process may open), through the RM backend's
/// selection (`Client::select_card`): inaccessible GPUs of a multi-GPU host
/// are skipped; a requested BDF is never substituted.
pub(crate) fn open_rm(pci: Option<[u32; 4]>) -> VResult<(Rc<rm::Device>, [u32; 4])> {
    let client = Client::open().map_err(|e| VideoError::Unsupported(format!("RM open: {e}")))?;
    let want = match pci {
        Some(p) => Some(p),
        None => match std::env::var("NV_VIDEO_PCI") {
            Ok(s) => Some(parse_pci(&s)?),
            Err(std::env::VarError::NotPresent) => None,
            Err(error @ std::env::VarError::NotUnicode(_)) => {
                return Err(VideoError::Invalid(format!("NV_VIDEO_PCI: {error}")));
            }
        },
    };
    let card = client
        .select_card(want)
        .map_err(|e| VideoError::Unsupported(format!("select NVIDIA GPU: {e}")))?;
    Ok((client.device(card)?, rm::card_bdf(&card)))
}

fn parse_pci(s: &str) -> VResult<[u32; 4]> {
    let bad = || VideoError::Invalid(format!("NV_VIDEO_PCI={s:?}: expected dddd:bb:dd.f"));
    let (dom, rest) = s.split_once(':').ok_or_else(bad)?;
    let (bus, rest) = rest.split_once(':').ok_or_else(bad)?;
    let (dev, fun) = rest.split_once('.').ok_or_else(bad)?;
    let h = |v: &str| {
        u32::from_str_radix(v, 16)
            .map_err(|error| VideoError::Invalid(format!("{}: {error}", bad())))
    };
    Ok([h(dom)?, h(bus)?, h(dev)?, h(fun)?])
}

impl VideoDevice {
    /// Open the GPU `NV_VIDEO_PCI` names, or else the first one this process may open
    /// (`nv_video_device_open` in `nv_video.h`), and read its NVDEC class and caps.
    ///
    /// # Errors
    /// No accessible NVIDIA GPU (with the reason for each candidate), or a failed RM query.
    pub fn open() -> VResult<Self> {
        let (device, pci) = open_rm(None)?;
        let classes = device.classes()?;
        let nvdec_class = NVDEC_CLASSES
            .iter()
            .map(|c| c.0)
            .find(|c| classes.contains(c))
            .unwrap_or(0);
        let mut raw = [0u32; 17];
        // NV2080_CTRL_CMD_GPU_GET_NAME_STRING (ASCII flavour), as mesa.rs query_info.
        device
            .client
            .control(device.subdevice, 0x20800110, &mut raw)?;
        let mut name = [0u8; 64];
        for (i, w) in raw[1..].iter().enumerate() {
            name[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        name[63] = 0;
        let card_device = device.card.pci.device;
        let mut arch = [0u32; 4];
        // NV2080_CTRL_CMD_MC_GET_ARCH_INFO: architecture | implementation.
        device
            .client
            .control(device.subdevice, 0x20801701, &mut arch)?;
        // Stock reads every present engine's caps at device init and fails the device when the control
        // fails (gens.rs `engine_records_4fc88`, ELF 0x4fc88); sessions here use NVDEC0 only.
        let mut p = BspCapsV2::default();
        let engine0 = if nvdec_class != 0 {
            device
                .client
                .control(device.handle, 0x801c02, &mut p)
                .map_err(|e| {
                    VideoError::Unsupported(format!(
                        "NV0080_CTRL_CMD_BSP_GET_CAPS_V2 (NVDEC0) failed: {e}"
                    ))
                })?;
            NvdecEngineCaps::from_caps_table(p.caps_tbl, nvdec_class)
        } else {
            NvdecEngineCaps::default()
        };
        Ok(Self {
            pci,
            engine0,
            engine0_caps_tbl: p.caps_tbl,
            info: NvVideoDeviceInfo {
                abi_version: ABI_VERSION,
                pci_vendor: 0x10de,
                pci_device: u32::from(card_device),
                chipset: arch[0] | arch[1],
                nvdec_class,
                // Sessions drive NVDEC0 only (the stock 0x13/0x16 alternation is
                // optional).
                nvdec_engines: u32::from(nvdec_class != 0),
                name,
            },
        })
    }
    pub(crate) fn caps(&self, codec: u32) -> VResult<NvVideoCaps> {
        if !(CODEC_H264..=CODEC_AV1).contains(&codec) {
            return Err(VideoError::Invalid(format!("unknown codec {codec}")));
        }
        Ok(caps(codec, self.info.nvdec_class, &self.engine0))
    }
}

/// Open the nvidia-drm render node of `pci` (sync_file export, dma-buf import).
pub(crate) fn open_render_node(pci: [u32; 4]) -> VResult<std::fs::File> {
    let want = pci_string(pci);
    let entries = std::fs::read_dir("/sys/class/drm")
        .map_err(|e| VideoError::Unsupported(format!("/sys/class/drm: {e}")))?;
    for e in entries.flatten() {
        if !e.file_name().to_string_lossy().starts_with("renderD") {
            continue;
        }
        let Ok(dev) = std::fs::canonicalize(e.path().join("device")) else {
            continue;
        };
        if dev.file_name().is_some_and(|n| n.to_string_lossy() == want) {
            return std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(std::path::Path::new("/dev/dri").join(e.file_name()))
                .map_err(|err| {
                    VideoError::Unsupported(format!("open render node of {want}: {err}"))
                });
        }
    }
    Err(VideoError::Unsupported(format!(
        "no DRM render node for NVIDIA GPU {want} (nvidia-drm loaded with modeset?)"
    )))
}

/// Test support: a dma-buf exported by nvidia-drm from our own VRAM
/// allocation, readable through the CPU, standing in for a gralloc buffer.
pub struct TestDmaBuf {
    /// The exported dma-buf.
    pub fd: std::os::fd::OwnedFd,
    memory: Rc<rm::Memory>,
    _device: Rc<rm::Device>,
}

impl TestDmaBuf {
    /// A dma-buf of at least `size` bytes (rounded up to 64 KiB) on `dev`'s GPU.
    ///
    /// # Errors
    /// No DRM render node for the GPU, or a failed allocation or export.
    pub fn new(dev: &VideoDevice, size: usize) -> VResult<Self> {
        use std::os::fd::AsFd;
        let (device, _) = open_rm(Some(dev.pci))?;
        let drm = open_render_node(dev.pci)?;
        let memory = device.allocate_vram(size.next_multiple_of(1 << 16))?;
        let fd = crate::external::export(&memory, drm.as_fd())?;
        Ok(Self {
            fd,
            memory,
            _device: device,
        })
    }
    /// Copy `out.len()` bytes at `offset` out of the buffer, once every GPU write to it
    /// has completed.
    ///
    /// # Errors
    /// The CPU mapping fails or the range is outside the buffer.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> VResult<()> {
        // Sound only after the GPU handoff: test helper: the caller waited for every copy into this dma-buf.
        self.memory.map()?.read_bytes(offset, out)?;
        Ok(())
    }
    /// Set every byte of the buffer to `byte`, before any GPU work references it.
    ///
    /// # Errors
    /// The CPU mapping fails.
    pub fn fill(&self, byte: u8) -> VResult<()> {
        let v = vec![byte; self.memory.size];
        // Sound only after the GPU handoff: test helper: called before any GPU work references the buffer.
        self.memory.map()?.write_bytes(0, &v)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live: NVDEC0's BSP capsTbl on this host and what it enables (`cargo test -- --ignored engine0`).
    #[test]
    #[ignore = "needs an NVIDIA GPU"]
    fn engine0_bsp_caps_live() {
        let d = VideoDevice::open().unwrap();
        println!(
            "class {:#x} capsTbl {:02x?} -> {:?}",
            d.info.nvdec_class, d.engine0_caps_tbl, d.engine0
        );
    }
}
