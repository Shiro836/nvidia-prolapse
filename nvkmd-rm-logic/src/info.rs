// SPDX-License-Identifier: MIT
//! What NVK learns about the device from the RM: Mesa's `nv_device_info` and
//! `nvkmd_info` fields derived from [`NvRmInfo`] (`nv_rm_open`).
//!
//! Two fields are not derived here because they come from Mesa itself: the
//! shared memory configurations (`nv_device_info_init_shared_mem_sizes`, from
//! `sm`) and the non-coherent atom (`util_cache_granularity`). The boundary
//! calls those after copying these values in.
use std::num::TryFromIntError;

use nv_rm_abi::{NvRmInfo, NvRmZcullInfo};
use nv_rm_core::ByteSize;
use thiserror::Error;

use crate::memory::PteKind;

/// Granularity of VA binds (`nvkmd_pdev::bind_align_B`): the 4 KiB small page.
/// Device memory is still allocated in 64 KiB pages ([`crate::memory::BIG_PAGE`]).
pub const BIND_ALIGN: ByteSize = ByteSize::new(4096);

/// Size of `nv_device_info::device_name` (Mesa `nv_device_info.h:40`).
pub const DEVICE_NAME_LEN: usize = 64;
/// Size of `nv_device_info::chipset_name` (Mesa `nv_device_info.h:41`).
pub const CHIPSET_NAME_LEN: usize = 16;

/// The PCI location (`nv_device_info::pci`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(missing_docs, reason = "PCI domain/bus/device/function")]
pub struct PciAddress {
    pub domain: u16,
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
}

/// The engine classes the RM allocated (`nv_device_info::cls_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(missing_docs, reason = "named as the nv_device_info fields")]
pub struct Classes {
    pub copy: u16,
    pub eng2d: u16,
    pub eng3d: u16,
    pub m2mf: u16,
    pub compute: u16,
}

/// ZCULL geometry and the context-switch buffer the RM binds to every graphics
/// channel (`nv_zcull_device_info`, Mesa `nv_device_info.h:17-30`, filled field
/// for field like `nouveau_ws_device_zcull_info`, `nouveau_device.c:271-284`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zcull {
    /// Region geometry (`GR_GET_ZCULL_INFO`), in `nv_zcull_device_info` order.
    pub geometry: NvRmZcullInfo,
    /// `nv_zcull_device_info::ctxsw_size`.
    pub ctxsw_size: u32,
    /// `nv_zcull_device_info::ctxsw_align`.
    pub ctxsw_align: u32,
}

/// `nv_device_info` as the RM backend fills it, before Mesa's own shared
/// memory and cache-line fields. The type is always `NV_DEVICE_TYPE_DIS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(missing_docs, reason = "named as the nv_device_info fields")]
pub struct DeviceInfo {
    pub device_id: u16,
    pub chipset: u16,
    /// NUL-terminated, zero-padded.
    pub device_name: [u8; DEVICE_NAME_LEN],
    /// `NV%03X` of the chipset, NUL-terminated, zero-padded.
    pub chipset_name: [u8; CHIPSET_NAME_LEN],
    pub pci: PciAddress,
    pub sm: u8,
    pub gpc_count: u8,
    pub tpc_count: u16,
    pub mp_per_tpc: u8,
    pub max_warps_per_mp: u8,
    pub classes: Classes,
    pub vram_size: ByteSize,
    pub bar_size: ByteSize,
    /// `None` when the RM has no ZCULL information (`has_zcull_info` false).
    pub zcull: Option<Zcull>,
}

/// `nvkmd_info` (Mesa `nvkmd.h:125-149`). Every capability not listed is false:
/// no sparse binding, no VRAM usage query, no fixed or over-mapping, no
/// dedicated-allocation compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KmdInfo {
    /// dma-buf import and export: always.
    pub has_dma_buf: bool,
    /// Tiled (DRM modifier) allocations: on devices with a generic
    /// block-linear kind ([`PteKind::GENERIC_TURING`] or [`PteKind::GENERIC_FERMI`]).
    pub has_alloc_tiled: bool,
    /// The stock driver's compression policy (compressible generic kind for
    /// device-only memory, [`crate::memory::Placement::VramCompressible`]):
    /// from Turing on.
    pub has_rm_compression: bool,
    /// Contexts that can join another context's channel group.
    pub max_shared_ctx: u32,
    /// The SM ISA code (`NV2080_CTRL_GR_INFO_INDEX_SM_VERSION`).
    pub sm_isa_version: u32,
    /// `NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT`.
    pub shader_pipe_count: u32,
    /// `NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT`.
    pub shader_pipe_sub_count: u32,
}

/// An RM value that does not fit the Mesa field it is reported in.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("RM device info {field} = {value:#x} does not fit nv_device_info")]
pub struct InfoError {
    /// The `NvRmInfo` field.
    pub field: &'static str,
    /// Its value.
    pub value: u32,
    /// The narrowing that failed.
    #[source]
    pub source: TryFromIntError,
}

fn narrow<T: TryFrom<u32, Error = TryFromIntError>>(
    field: &'static str,
    value: u32,
) -> Result<T, InfoError> {
    T::try_from(value).map_err(|source| InfoError {
        field,
        value,
        source,
    })
}

/// The `nv_device_info` fields the RM reports.
///
/// # Errors
/// [`InfoError`] when a value does not fit its Mesa field: NVK would otherwise
/// see a different device than the RM opened.
pub fn device_info(info: &NvRmInfo) -> Result<DeviceInfo, InfoError> {
    Ok(DeviceInfo {
        device_id: narrow("device_id", info.device_id)?,
        chipset: narrow("chipset", info.chipset)?,
        device_name: device_name(info),
        chipset_name: chipset_name(info.chipset),
        pci: PciAddress {
            domain: narrow("pci_domain", info.pci_domain)?,
            bus: narrow("pci_bus", info.pci_bus)?,
            dev: narrow("pci_device", info.pci_device)?,
            func: narrow("pci_function", info.pci_function)?,
        },
        sm: narrow("sm", info.sm)?,
        gpc_count: narrow("gpc_count", info.gpc_count)?,
        tpc_count: narrow("tpc_count", info.tpc_count)?,
        mp_per_tpc: narrow("mp_per_tpc", info.mp_per_tpc)?,
        max_warps_per_mp: narrow("max_warps_per_mp", info.max_warps_per_mp)?,
        classes: Classes {
            copy: narrow("cls_copy", info.cls_copy)?,
            eng2d: narrow("cls_eng2d", info.cls_eng2d)?,
            eng3d: narrow("cls_eng3d", info.cls_eng3d)?,
            m2mf: narrow("cls_m2mf", info.cls_m2mf)?,
            compute: narrow("cls_compute", info.cls_compute)?,
        },
        vram_size: ByteSize::new(info.vram_size),
        bar_size: ByteSize::new(info.bar_size),
        zcull: (info.has_zcull_info != 0).then_some(Zcull {
            geometry: info.zcull,
            ctxsw_size: info.zcull_ctxsw_size,
            ctxsw_align: info.zcull_ctxsw_align,
        }),
    })
}

/// The `nvkmd_info` of the device.
pub fn kmd_info(info: &NvRmInfo) -> KmdInfo {
    let turing = info.generic_kind == u32::from(PteKind::GENERIC_TURING.get());
    let fermi = info.generic_kind == u32::from(PteKind::GENERIC_FERMI.get());
    KmdInfo {
        has_dma_buf: true,
        has_alloc_tiled: turing || fermi,
        has_rm_compression: turing,
        max_shared_ctx: info.max_shared_ctx,
        sm_isa_version: info.sm_isa,
        shader_pipe_count: info.shader_pipe_count,
        shader_pipe_sub_count: info.shader_pipe_sub_count,
    }
}

/// The RM's name up to its NUL, cut to fit with the terminator. A name without
/// a NUL in its 64 bytes is cut the same way instead of read past its end.
fn device_name(info: &NvRmInfo) -> [u8; DEVICE_NAME_LEN] {
    let mut out = [0; DEVICE_NAME_LEN];
    let name = info.name.iter().map(|c| c.to_ne_bytes()[0]);
    for (slot, byte) in out[..DEVICE_NAME_LEN - 1]
        .iter_mut()
        .zip(name.take_while(|&b| b != 0))
    {
        *slot = byte;
    }
    out
}

/// `NV` and the chipset in at least three upper-case hex digits (`NV192`).
/// Ten bytes at most, so it always fits.
fn chipset_name(chipset: u32) -> [u8; CHIPSET_NAME_LEN] {
    let text = format!("NV{chipset:03X}");
    let mut out = [0; CHIPSET_NAME_LEN];
    out[..text.len()].copy_from_slice(text.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An Ada-shaped report; the tests check the field mapping, not these values.
    fn ad102() -> NvRmInfo {
        let mut name = [0; 64];
        for (slot, byte) in name.iter_mut().zip(b"NVIDIA GeForce RTX 4090") {
            *slot = std::ffi::c_char::from_ne_bytes([*byte]);
        }
        NvRmInfo {
            vendor_id: 0x10de,
            device_id: 0x2684,
            chipset: 0x192,
            sm: 89,
            pci_domain: 0,
            pci_bus: 1,
            pci_device: 0,
            pci_function: 0,
            gpc_count: 11,
            tpc_count: 72,
            mp_per_tpc: 2,
            max_warps_per_mp: 48,
            cls_copy: 0xc7b5,
            cls_eng2d: 0x902d,
            cls_eng3d: 0xc997,
            cls_m2mf: 0,
            cls_compute: 0xc9c0,
            generic_kind: 0x06,
            vram_size: 24 << 30,
            bar_size: 256 << 20,
            name,
            zcull: NvRmZcullInfo {
                width_align_pixels: 64,
                ..NvRmZcullInfo::default()
            },
            zcull_ctxsw_size: 0x1000,
            zcull_ctxsw_align: 0x1000,
            has_zcull_info: 1,
            sm_isa: 0x809,
            shader_pipe_count: 144,
            shader_pipe_sub_count: 72,
            max_shared_ctx: 63,
            reserved: 0,
        }
    }

    fn c_str(bytes: &[u8]) -> &[u8] {
        let end = bytes.iter().position(|&b| b == 0).expect("NUL-terminated");
        &bytes[..end]
    }

    #[test]
    fn fields_are_copied_and_names_formatted() {
        let dev = device_info(&ad102()).expect("every value fits");
        assert_eq!(dev.device_id, 0x2684);
        assert_eq!(dev.chipset, 0x192);
        assert_eq!(c_str(&dev.device_name), b"NVIDIA GeForce RTX 4090");
        assert_eq!(c_str(&dev.chipset_name), b"NV192");
        assert_eq!(
            dev.pci,
            PciAddress {
                domain: 0,
                bus: 1,
                dev: 0,
                func: 0
            }
        );
        assert_eq!(dev.classes.compute, 0xc9c0);
        assert_eq!(dev.vram_size, ByteSize::new(24 << 30));
        assert_eq!(c_str(&chipset_name(0x50)), b"NV050");
        assert_eq!(c_str(&chipset_name(u32::MAX)), b"NVFFFFFFFF");
    }

    #[test]
    fn names_are_cut_to_fit_their_terminator() {
        // As snprintf("%s") into device_name[64].
        let mut info = ad102();
        info.name = [std::ffi::c_char::from_ne_bytes(*b"x"); 64];
        let dev = device_info(&info).expect("every value fits");
        assert_eq!(dev.device_name[..63], [b'x'; 63]);
        assert_eq!(dev.device_name[63], 0);
    }

    #[test]
    fn zcull_is_reported_only_when_the_rm_has_it() {
        let dev = device_info(&ad102()).expect("every value fits");
        let zcull = dev.zcull.expect("has_zcull_info is set");
        assert_eq!(zcull.geometry.width_align_pixels, 64);
        assert_eq!((zcull.ctxsw_size, zcull.ctxsw_align), (0x1000, 0x1000));
        let mut info = ad102();
        info.has_zcull_info = 0;
        assert_eq!(device_info(&info).expect("every value fits").zcull, None);
    }

    #[test]
    fn values_that_do_not_fit_are_refused() {
        let refused =
            |info: &NvRmInfo| device_info(info).map_err(|error| (error.field, error.value));
        let mut info = ad102();
        info.pci_bus = 0x100;
        assert_eq!(refused(&info), Err(("pci_bus", 0x100)));
        let mut info = ad102();
        info.cls_eng3d = 0x1_c997;
        assert_eq!(refused(&info), Err(("cls_eng3d", 0x1_c997)));
    }

    #[test]
    fn capabilities_follow_the_generic_kind() {
        let mut info = ad102();
        let turing = kmd_info(&info);
        assert!(turing.has_dma_buf && turing.has_alloc_tiled && turing.has_rm_compression);
        assert_eq!(
            (
                turing.max_shared_ctx,
                turing.sm_isa_version,
                turing.shader_pipe_count,
                turing.shader_pipe_sub_count
            ),
            (63, 0x809, 144, 72)
        );
        info.generic_kind = 0xfe;
        let fermi = kmd_info(&info);
        assert!(fermi.has_alloc_tiled && !fermi.has_rm_compression);
        info.generic_kind = 0;
        let none = kmd_info(&info);
        assert!(!none.has_alloc_tiled && !none.has_rm_compression);
        info.generic_kind = 0x106;
        assert!(!kmd_info(&info).has_alloc_tiled);
    }
}
