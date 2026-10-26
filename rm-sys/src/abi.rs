// SPDX-License-Identifier: MIT
//! Public NVIDIA RM wire layouts. Authority: matching open-gpu-kernel-modules
//! nvos.h, nv-ioctl.h, class/cl0080.h, ctrl0000gpu.h and ctrl0080gpu.h.
//! Explicit padding fixes logical offsets across LP64 and supported ILP32 ABIs.
//! compat.rs marshals named fields to exact-release public layouts; it never
//! copies implicit Rust padding or assumes this canonical layout on old drivers.
//! tools/abi-check.c independently verifies these against NVIDIA's C headers.
#![allow(dead_code)]

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Alloc {
    pub root: u32,
    pub parent: u32,
    pub object: u32,
    pub class: u32,
    pub parameters: u64,
    pub size: u32,
    pub status: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Control {
    pub client: u32,
    pub object: u32,
    pub command: u32,
    pub flags: u32,
    pub parameters: u64,
    pub size: u32,
    pub status: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Free {
    pub root: u32,
    pub parent: u32,
    pub object: u32,
    pub status: u32,
}
/// Public nv_ioctl_{alloc,free}_os_event_t (same wire fields).
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct OsEvent {
    pub client: u32,
    pub device: u32,
    pub fd: u32,
    pub status: u32,
}
/// NV0005_ALLOC_PARAMETERS; data is a numeric OS-event FD, never a pointer.
#[repr(C, align(8))]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct EventAlloc {
    pub parent_client: u32,
    pub source: u32,
    pub class: u32,
    pub notify_index: u32,
    pub data: u64,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct EventNotification {
    pub event: u32,
    pub action: u32,
    pub notify_state: u8,
    pub padding0: [u8; 3],
    pub info32: u32,
    pub info16: u16,
    pub padding1: [u8; 2],
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct EventTrigger {
    pub event: u32,
}
#[repr(C)]
#[derive(Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Version {
    pub command: u32,
    pub reply: u32,
    pub version: [u8; 64],
}
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Pci {
    pub domain: u32,
    pub bus: u8,
    pub slot: u8,
    pub function: u8,
    pub padding: u8,
    pub vendor: u16,
    pub device: u16,
}
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Card {
    pub valid: u8,
    pub padding0: [u8; 3],
    pub pci: Pci,
    pub gpu_id: u32,
    pub interrupt: u16,
    pub padding1: [u8; 2],
    pub registers: u64,
    pub register_size: u64,
    pub framebuffer: u64,
    pub framebuffer_size: u64,
    pub minor: u32,
    pub name: [u8; 10],
    pub padding2: [u8; 2],
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct GpuInfo {
    pub gpu_id: u32,
    pub flags: u32,
    pub device_instance: u32,
    pub subdevice_instance: u32,
    pub sli_status: u32,
    pub board_id: u32,
    pub gpu_instance: u32,
    pub numa_id: i32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct DeviceAlloc {
    pub device_id: u32,
    pub client_share: u32,
    pub target_client: u32,
    pub target_device: u32,
    pub flags: u32,
    pub padding0: u32,
    pub va_size: u64,
    pub va_start: u64,
    pub va_limit: u64,
    pub va_mode: u32,
    pub padding1: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ClassList {
    pub count: u32,
    pub padding: u32,
    pub classes: u64,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MemoryAlloc {
    pub owner: u32,
    pub memory_type: u32,
    pub flags: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: i32,
    pub attr: u32,
    pub attr2: u32,
    pub format: u32,
    pub compression: u32,
    pub zcull: u32,
    pub padding0: u32,
    pub range_low: u64,
    pub range_high: u64,
    pub size: u64,
    pub alignment: u64,
    pub offset: u64,
    pub limit: u64,
    pub address: u64,
    pub ctag_offset: u32,
    pub vaspace: u32,
    pub internal_flags: u32,
    pub tag: u32,
    pub numa_node: i32,
    pub padding1: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MapMemory {
    pub client: u32,
    pub device: u32,
    pub memory: u32,
    pub padding: u32,
    pub offset: u64,
    pub length: u64,
    pub linear_address: u64,
    pub status: u32,
    pub flags: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MapWithFd {
    pub parameters: MapMemory,
    pub fd: i32,
    pub padding: u32,
}

#[repr(C, align(8))]
#[derive(Default, Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct UnmapMemory {
    pub client: u32,
    pub device: u32,
    pub memory: u32,
    pub padding: u32,
    pub linear_address: u64,
    pub status: u32,
    pub flags: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct PerfBoost {
    pub flags: u32,
    pub duration: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct CopyEngine {
    pub version: u32,
    pub engine_type: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct EnginePartners {
    pub engine_type: u32,
    pub partnership_class: u32,
    pub runqueue: u32,
    pub count: u32,
    pub partners: [u32; 32],
}

#[repr(C, align(8))]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct EngineClassList {
    pub engine_type: u32,
    pub count: u32,
    pub classes: u64,
}

pub const fn iowr(number: u8, size: usize) -> u32 {
    assert!(size < 16384);
    (3 << 30) | ((size as u32) << 16) | ((b'F' as u32) << 8) | number as u32
}

#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct VaSpace {
    pub index: u32,
    pub flags: u32,
    pub size: u64,
    pub internal_start: u64,
    pub internal_limit: u64,
    pub big_page_size: u32,
    pub padding0: u32,
    pub base: u64,
    pub pasid: u32,
    pub padding1: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct VirtualMemory {
    pub offset: u64,
    pub limit: u64,
    pub vaspace: u32,
    pub padding: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MapDma {
    pub client: u32,
    pub device: u32,
    pub dma: u32,
    pub memory: u32,
    pub offset: u64,
    pub length: u64,
    pub flags: u32,
    pub flags2: u32,
    pub kind: u32,
    pub padding0: u32,
    pub address: u64,
    pub status: u32,
    pub padding1: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct UnmapDma {
    pub client: u32,
    pub device: u32,
    pub dma: u32,
    pub memory: u32,
    pub flags: u32,
    pub padding0: u32,
    pub address: u64,
    pub size: u64,
    pub status: u32,
    pub padding1: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ChannelGroup {
    pub error: u32,
    pub ecc_error: u32,
    pub vaspace: u32,
    pub engine: u32,
    pub vgpu: u8,
    pub padding: [u8; 3],
    pub reserved: u32,
    pub flags: u32,
}
/// NV_BSP_ALLOCATION_PARAMETERS (= NV_NVDEC_ALLOCATION_PARAMETERS, nvos.h).
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct VideoDecoder {
    pub size: u32,
    pub prohibit_multiple_instances: u32,
    pub engine_instance: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ContextShare {
    pub vaspace: u32,
    pub flags: u32,
    pub subcontext: u32,
}
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MemoryDescriptor {
    pub base: u64,
    pub size: u64,
    pub address_space: u32,
    pub cache: u32,
}
#[repr(C)]
#[derive(Default, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct Channel {
    pub error: u32,
    pub buffer: u32,
    pub fifo_address: u64,
    pub entries: u32,
    pub flags: u32,
    pub context_share: u32,
    pub vaspace: u32,
    pub handle_vaspace: u32,
    pub userd_memory: [u32; 8],
    pub padding: u32,
    pub userd_offsets: [u64; 8],
    pub engine: u32,
    pub cid: u32,
    pub subdevice: u32,
    pub ecc_error: u32,
    pub instance_memory: MemoryDescriptor,
    pub userd_memory_descriptor: MemoryDescriptor,
    pub ramfc_memory: MemoryDescriptor,
    pub method_memory: MemoryDescriptor,
    pub physical_group: u32,
    pub internal_flags: u32,
    pub error_memory: MemoryDescriptor,
    pub ecc_error_memory: MemoryDescriptor,
    pub process_id: u32,
    pub subprocess_id: u32,
    pub encrypt_iv: [u32; 3],
    pub decrypt_iv: [u32; 3],
    pub hmac_nonce: [u32; 8],
    pub tpc_config: u32,
    pub padding1: u32,
}

#[repr(C)]
#[derive(Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct GrInfo {
    pub count: u32,
    pub padding: u32,
    pub entries: u64,
    pub route_flags: u32,
    pub padding1: u32,
    pub route: u64,
}

/// Optional public query; an available ABI does not imply static BAR1 is enabled.
#[repr(C, align(8))]
#[derive(Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct StaticBar1Info {
    pub enabled: u8,
    pub write_combined: u8,
    pub padding: [u8; 6],
    pub start: u64,
    pub size: u64,
}

/// NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS (ctrl0080fifo.h);
/// identical in all 225 public tags (tools/probe-zcull.py).
#[repr(C)]
#[derive(Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct EngineContextProperties {
    pub engine: u32,
    pub alignment: u32,
    pub size: u32,
}

/// NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS (ctrl2080gr.h): ten NvU32 fields in
/// header order, identical in all 225 public tags (tools/probe-zcull.py).
/// Same order as Mesa's nv_zcull_device_info / drm_nouveau_get_zcull_info
/// (nv_device_info.h:17-30) without the trailing ctxsw size/alignment.
#[repr(C)]
#[derive(
    Default, Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct ZcullInfo {
    pub width_align_pixels: u32,
    pub height_align_pixels: u32,
    pub pixel_squares_by_aliquots: u32,
    pub aliquot_total: u32,
    pub region_byte_multiplier: u32,
    pub region_header_size: u32,
    pub subregion_header_size: u32,
    pub subregion_count: u32,
    pub subregion_width_align_pixels: u32,
    pub subregion_height_align_pixels: u32,
}

/// NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS (ctrl2080gr.h); identical in all
/// 225 public tags (tools/probe-zcull.py).
#[repr(C)]
#[derive(Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ZcullBind {
    pub client: u32,
    pub channel: u32,
    pub address: u64,
    pub mode: u32,
    pub padding: u32,
}

pub fn layouts() -> Vec<(&'static str, usize)> {
    use std::mem::{offset_of, size_of};
    vec![
        (
            "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS",
            size_of::<EngineContextProperties>(),
        ),
        (
            "NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS",
            size_of::<ZcullBind>(),
        ),
        (
            "NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS",
            size_of::<ZcullInfo>(),
        ),
        (
            "NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS.vMemPtr",
            offset_of!(ZcullBind, address),
        ),
        ("NV2080_CTRL_GR_GET_INFO_PARAMS", size_of::<GrInfo>()),
        (
            "NV2080_CTRL_GR_GET_INFO_PARAMS.grRouteInfo",
            offset_of!(GrInfo, route_flags),
        ),
        ("NV_VASPACE_ALLOCATION_PARAMETERS", size_of::<VaSpace>()),
        (
            "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS",
            size_of::<VirtualMemory>(),
        ),
        ("NVOS46_PARAMETERS", size_of::<MapDma>()),
        ("NVOS47_PARAMETERS", size_of::<UnmapDma>()),
        (
            "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS",
            size_of::<ChannelGroup>(),
        ),
        (
            "NV_CTXSHARE_ALLOCATION_PARAMETERS",
            size_of::<ContextShare>(),
        ),
        (
            "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
            size_of::<Channel>(),
        ),
        (
            "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS.userdOffset",
            offset_of!(Channel, userd_offsets),
        ),
        (
            "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS.engineType",
            offset_of!(Channel, engine),
        ),
        (
            "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS.tpcConfigID",
            offset_of!(Channel, tpc_config),
        ),
        ("NVOS21_PARAMETERS", size_of::<Alloc>()),
        ("NVOS21_PARAMETERS.status", offset_of!(Alloc, status)),
        ("NVOS54_PARAMETERS", size_of::<Control>()),
        ("NVOS00_PARAMETERS", size_of::<Free>()),
        ("nv_ioctl_rm_api_version_t", size_of::<Version>()),
        ("nv_ioctl_card_info_t", size_of::<Card>()),
        ("nv_ioctl_card_info_t.minor_number", offset_of!(Card, minor)),
        (
            "NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS",
            size_of::<GpuInfo>(),
        ),
        ("NV0080_ALLOC_PARAMETERS", size_of::<DeviceAlloc>()),
        (
            "NV0080_ALLOC_PARAMETERS.vaMode",
            offset_of!(DeviceAlloc, va_mode),
        ),
        (
            "NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS",
            size_of::<ClassList>(),
        ),
        ("NV_MEMORY_ALLOCATION_PARAMS", size_of::<MemoryAlloc>()),
        (
            "NV_MEMORY_ALLOCATION_PARAMS.size",
            offset_of!(MemoryAlloc, size),
        ),
        (
            "NV_MEMORY_ALLOCATION_PARAMS.numaNode",
            offset_of!(MemoryAlloc, numa_node),
        ),
        ("NVOS33_PARAMETERS", size_of::<MapMemory>()),
        ("NVOS34_PARAMETERS", size_of::<UnmapMemory>()),
        (
            "NVOS34_PARAMETERS.pLinearAddress",
            offset_of!(UnmapMemory, linear_address),
        ),
        ("NVOS34_PARAMETERS.status", offset_of!(UnmapMemory, status)),
        ("NVOS34_PARAMETERS.flags", offset_of!(UnmapMemory, flags)),
        ("nv_ioctl_nvos33_parameters_with_fd", size_of::<MapWithFd>()),
        ("ioctl_alloc", iowr(0x2b, size_of::<Alloc>()) as usize),
        (
            "ioctl_card_info",
            iowr(200, 64 * size_of::<Card>()) as usize,
        ),
    ]
}

/// NV0080_CTRL_CMD_BSP_GET_CAPS_V2 (0x801c02, device object) parameters: one NVDEC instance's capsTbl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct BspCapsV2 {
    pub caps_tbl: [u8; 8],
    pub instance_id: u32,
}
