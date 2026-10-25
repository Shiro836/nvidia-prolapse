//! Exact-release RM wire adapters generated from independent public C probes.
//! Membership is never interpolated across a branch or a version backport.
//! Logical caller structures use the 615.71.09 schema; only named initialized
//! fields cross the boundary. Missing evidence and unsupported fields fail before
//! an ioctl. This is ABI translation, not a hardware-support declaration.
use crate::rm::{Error, Result};
use std::{ffi::c_void, mem::size_of, ptr};

#[derive(Debug, PartialEq, Eq)]
struct Field {
    name: &'static str,
    offset: usize,
    len: usize,
}
#[derive(Debug, PartialEq, Eq)]
struct Layout {
    name: &'static str,
    size: usize,
    fields: &'static [Field],
}
impl Layout {
    fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }
}
#[derive(Debug)]
struct Profile {
    layouts: &'static [(&'static str, &'static Layout)],
    constants: &'static [(&'static str, u64)],
}
impl Profile {
    fn layout(&self, name: &str) -> Option<&'static Layout> {
        self.layouts
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, l)| *l)
    }
    fn constant(&self, name: &str) -> Option<u64> {
        self.constants
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
    }
}
#[rustfmt::skip]
#[path = "compat_data.rs"]
mod data;

#[derive(Clone, Copy, Debug)]
pub struct Compat {
    version: &'static str,
    profile: &'static Profile,
}
impl Compat {
    pub fn for_version(version: &str) -> Result<Self> {
        let (version, profile) = data::VERSIONS.iter().find(|(v, _)| *v == version)
            .ok_or_else(|| Error::Contract(format!("NVIDIA {version}: no exact public ABI profile; generate and verify that release before use")))?;
        Ok(Self { version, profile })
    }
    pub fn version(&self) -> &'static str {
        self.version
    }
    pub fn constant(&self, name: &str) -> Option<u64> {
        self.profile.constant(name)
    }
    pub fn require_constant(&self, name: &str, value: u64) -> Result<()> {
        if self.constant(name) != Some(value) {
            return Err(self.unsupported(
                name,
                "constant absent or differs from the implemented contract",
            ));
        }
        Ok(())
    }
    fn unsupported(&self, name: &str, why: &str) -> Error {
        Error::Contract(format!(
            "NVIDIA {}: unsupported ABI {name}: {why}",
            self.version
        ))
    }
    fn layout(&self, name: &str) -> Result<&'static Layout> {
        self.profile.layout(name).ok_or_else(|| {
            self.unsupported(
                name,
                "public wire layout not measured for this exact release",
            )
        })
    }
    pub fn layout_size(&self, name: &str) -> Result<usize> {
        Ok(self.layout(name)?.size)
    }
    pub fn require_same_layout(&self, name: &str) -> Result<()> {
        let current = data::CANONICAL
            .layout(name)
            .ok_or_else(|| self.unsupported(name, "no implemented canonical layout"))?;
        if self.layout(name)? != current {
            return Err(self.unsupported(name, "wire differs; this caller has no adapter"));
        }
        Ok(())
    }
    /// Called before opening any device. Old wrappers alone are not RM evidence.
    pub fn require_client_abi(&self) -> Result<()> {
        for name in [
            "nv_ioctl_rm_api_version_t",
            "nv_ioctl_card_info_t",
            "NVOS21_PARAMETERS",
            "NVOS54_PARAMETERS",
            "NVOS00_PARAMETERS",
            "NV0080_ALLOC_PARAMETERS",
        ] {
            self.layout(name)?;
        }
        for name in [
            "nv_ioctl_rm_api_version_t",
            "nv_ioctl_register_fd_t",
            "nv_ioctl_card_info_t",
        ] {
            self.require_same_layout(name)?;
        }
        for (name, value) in [
            ("NV_IOCTL_MAGIC", 0x46),
            ("NV_ESC_CHECK_VERSION_STR", 210),
            ("NV_ESC_CARD_INFO", 200),
            ("NV_ESC_REGISTER_FD", 201),
            ("NV_ESC_RM_ALLOC", 0x2b),
            ("NV_ESC_RM_CONTROL", 0x2a),
            ("NV_ESC_RM_FREE", 0x29),
            ("NV_ESC_RM_MAP_MEMORY", 0x4e),
            ("NV_ESC_RM_UNMAP_MEMORY", 0x4f),
            ("NV_RM_API_VERSION_CMD_STRICT", 0),
            ("NV_RM_API_VERSION_REPLY_RECOGNIZED", 1),
            ("NV01_ROOT_CLIENT", 0x41),
            ("NV01_DEVICE_0", 0x80),
            ("NV20_SUBDEVICE_0", 0x2080),
        ] {
            self.require_constant(name, value)?;
        }
        Ok(())
    }
    pub fn supports_semsurf_bind(&self) -> bool {
        // BIND exists as a stub in old headers; UNBIND was added with the real
        // channel registration contract. Exact profiles preserve branch hops.
        self.constant("NV_SEMAPHORE_SURFACE_CTRL_CMD_UNBIND_CHANNEL") == Some(0xda0006)
            && self
                .layout_size("NV_SEMAPHORE_SURFACE_CTRL_BIND_CHANNEL_PARAMS")
                .ok()
                == Some(40)
    }
    pub fn supports_kind_override(&self) -> bool {
        self.profile
            .layout("NVOS46_PARAMETERS")
            .is_some_and(|l| l.field("kindOverride").is_some())
            && self.constant("NVOS46_FLAGS_PAGE_KIND_OVERRIDE_YES") == Some(0x80000)
    }
    pub fn schedule_size(&self) -> Result<usize> {
        self.layout_size("NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS")
    }
    pub fn supports_external_memory(&self) -> bool {
        self.constant("NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD") == Some(0x3d05)
            && self.constant("NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD") == Some(0x3d06)
            && self.constant("DRM_NVIDIA_GEM_IMPORT_NVKMS_MEMORY") == Some(1)
            && self.constant("DRM_NVIDIA_GEM_EXPORT_NVKMS_MEMORY") == Some(9)
            && self
                .profile
                .layout("NvKmsKapiPrivImportMemoryParams")
                .is_some()
    }
    pub fn supports_identify_object(&self) -> bool {
        self.constant("DRM_NVIDIA_GEM_IDENTIFY_OBJECT") == Some(0xe)
    }
    /// Optional cache eligibility evidence only. Older size-only queries cannot
    /// prove a nonzero static BAR range's start; never infer a zero offset.
    #[cfg(test)]
    pub fn supports_static_bar1_range(&self) -> bool {
        self.constant("NV2080_CTRL_CMD_FB_GET_STATIC_BAR1_INFO") == Some(0x20801354)
            && self
                .profile
                .layout("NV2080_CTRL_FB_GET_STATIC_BAR1_INFO_PARAMS")
                .is_some_and(|layout| {
                    [
                        "bStaticBar1Enabled",
                        "staticBar1StartOffset",
                        "staticBar1Size",
                    ]
                    .iter()
                    .all(|name| layout.field(name).is_some())
                })
    }
    pub(crate) fn control_layout(&self, command: u32) -> Result<&'static str> {
        let (constant, layout) = match command {
            0x20800301 => (
                "NV2080_CTRL_CMD_EVENT_SET_NOTIFICATION",
                "NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS",
            ),
            0x20800308 => (
                "NV2080_CTRL_CMD_EVENT_SET_TRIGGER_FIFO",
                "NV2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS",
            ),
            0x205 => (
                "NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2",
                "NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS",
            ),
            0xd01 => (
                "NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE",
                "NV0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS",
            ),
            0x3d05 => (
                "NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD",
                "NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS",
            ),
            0x3d06 => (
                "NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD",
                "NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS",
            ),
            0x801c02 => (
                "NV0080_CTRL_CMD_BSP_GET_CAPS_V2",
                "NV0080_CTRL_BSP_GET_CAPS_PARAMS_V2",
            ),
            0x800201 => (
                "NV0080_CTRL_CMD_GPU_GET_CLASSLIST",
                "NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS",
            ),
            0x2080200a => (
                "NV2080_CTRL_CMD_PERF_BOOST",
                "NV2080_CTRL_PERF_BOOST_PARAMS",
            ),
            0x20800124 => (
                "NV2080_CTRL_CMD_GPU_GET_ENGINE_CLASSLIST",
                "NV2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS",
            ),
            0x20800147 => (
                "NV2080_CTRL_CMD_GPU_GET_ENGINE_PARTNERLIST",
                "NV2080_CTRL_GPU_GET_ENGINE_PARTNERLIST_PARAMS",
            ),
            0x20800110 => (
                "NV2080_CTRL_CMD_GPU_GET_NAME_STRING",
                "NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS",
            ),
            0x20800403 => (
                "NV2080_CTRL_CMD_TIMER_GET_TIME",
                "NV2080_CTRL_TIMER_GET_TIME_PARAMS",
            ),
            0x20801201 => (
                "NV2080_CTRL_CMD_GR_GET_INFO",
                "NV2080_CTRL_GR_GET_INFO_PARAMS",
            ),
            0x801707 => (
                "NV0080_CTRL_CMD_FIFO_GET_ENGINE_CONTEXT_PROPERTIES",
                "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS",
            ),
            0x20801208 => (
                "NV2080_CTRL_CMD_GR_CTXSW_ZCULL_BIND",
                "NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS",
            ),
            0x20801206 => (
                "NV2080_CTRL_CMD_GR_GET_ZCULL_INFO",
                "NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS",
            ),
            0x2080122a => (
                "NV2080_CTRL_CMD_GR_GET_GPC_MASK",
                "NV2080_CTRL_GR_GET_GPC_MASK_PARAMS",
            ),
            0x2080122b => (
                "NV2080_CTRL_CMD_GR_GET_TPC_MASK",
                "NV2080_CTRL_GR_GET_TPC_MASK_PARAMS",
            ),
            0x20801301 => (
                "NV2080_CTRL_CMD_FB_GET_INFO",
                "NV2080_CTRL_FB_GET_INFO_PARAMS",
            ),
            0x20801352 => (
                "NV2080_CTRL_CMD_FB_GET_SEMAPHORE_SURFACE_LAYOUT",
                "NV2080_CTRL_FB_GET_SEMAPHORE_SURFACE_LAYOUT_PARAMS",
            ),
            0x20801354 => (
                "NV2080_CTRL_CMD_FB_GET_STATIC_BAR1_INFO",
                "NV2080_CTRL_FB_GET_STATIC_BAR1_INFO_PARAMS",
            ),
            0x20801701 => (
                "NV2080_CTRL_CMD_MC_GET_ARCH_INFO",
                "NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS",
            ),
            0xa06c0101 => (
                "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE",
                "NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS",
            ),
            0xa06f0103 => (
                "NVA06F_CTRL_CMD_GPFIFO_SCHEDULE",
                "NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS",
            ),
            0xa06f0104 => ("NVA06F_CTRL_CMD_BIND", "NVA06F_CTRL_BIND_PARAMS"),
            0xc36f0108 => (
                "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN",
                "NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS",
            ),
            0xda0002 => (
                "NV_SEMAPHORE_SURFACE_CTRL_CMD_BIND_CHANNEL",
                "NV_SEMAPHORE_SURFACE_CTRL_BIND_CHANNEL_PARAMS",
            ),
            0xda0006 => (
                "NV_SEMAPHORE_SURFACE_CTRL_CMD_UNBIND_CHANNEL",
                "NV_SEMAPHORE_SURFACE_CTRL_UNBIND_CHANNEL_PARAMS",
            ),
            0xda0004 => (
                "NV_SEMAPHORE_SURFACE_CTRL_CMD_SET_VALUE",
                "NV_SEMAPHORE_SURFACE_CTRL_SET_VALUE_PARAMS",
            ),
            _ => {
                return Err(self.unsupported(
                    &format!("control {command:#x}"),
                    "not in implemented command table",
                ))
            }
        };
        self.require_constant(constant, command as u64)?;
        self.layout(layout)?;
        if matches!(command, 0xda0002 | 0xda0006) && !self.supports_semsurf_bind() {
            return Err(self.unsupported(layout, "BIND/UNBIND unavailable; use the explicitly selected legacy constructor notification path"));
        }
        Ok(layout)
    }
    pub(crate) fn allocation_layout(
        &self,
        class: u32,
        size: usize,
    ) -> Result<Option<&'static str>> {
        let name = match class {
            0x5 => "NV0005_ALLOC_PARAMETERS",
            0x80 => "NV0080_ALLOC_PARAMETERS",
            0x2080 => "NV2080_ALLOC_PARAMETERS",
            0x3e | 0x40 | 0x50a0 => "NV_MEMORY_ALLOCATION_PARAMS",
            0x70 => "NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS",
            0x90f1 => "NV_VASPACE_ALLOCATION_PARAMETERS",
            0xa06c => "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS",
            0x9067 => "NV_CTXSHARE_ALLOCATION_PARAMETERS",
            0xda => "NV_SEMAPHORE_SURFACE_ALLOC_PARAMETERS",
            // Explicit known GPFIFO families, not arbitrary suffix selection.
            0xa06f | 0xa16f | 0xb06f | 0xc06f | 0xc36f | 0xc46f | 0xc56f | 0xc76f | 0xc86f
            | 0xc96f | 0xca6f => "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
            0xb0b5 | 0xc0b5 | 0xc1b5 | 0xc3b5 | 0xc5b5 | 0xc6b5 | 0xc7b5 | 0xc9b5 | 0xcab5 => {
                "NVB0B5_ALLOCATION_PARAMETERS"
            }
            // NVxxB0_VIDEO_DECODER: kernel_nvdec_engdesc.c (615.71.09:66-100)
            // requires this payload and checks its size; explicit class list.
            0xa0b0 | 0xb0b0 | 0xb6b0 | 0xc1b0 | 0xc2b0 | 0xc3b0 | 0xc4b0 | 0xc6b0 | 0xc7b0
            | 0xc9b0 | 0xcdb0 | 0xceb0 | 0xcfb0 | 0xb8b0 | 0xd1b0 | 0xd2b0 => {
                "NV_BSP_ALLOCATION_PARAMETERS"
            }
            // Only implemented public engine/usermode contracts may omit a
            // payload; runtime class discovery is an additional prerequisite.
            0xc361 | 0xc461 | 0xc561 | 0xc661 | 0xc761 | 0x902d | 0xa140 | 0xcd40 | 0xb097
            | 0xb197 | 0xc097 | 0xc197 | 0xc397 | 0xc597 | 0xc697 | 0xc797 | 0xc997 | 0xce97
            | 0xb0c0 | 0xb1c0 | 0xc0c0 | 0xc1c0 | 0xc3c0 | 0xc5c0 | 0xc6c0 | 0xc7c0 | 0xc9c0
            | 0xcec0
                if size == 0 =>
            {
                return Ok(None)
            }
            _ => {
                return Err(self.unsupported(
                    &format!("class {class:#x}"),
                    "nonempty allocation payload not implemented",
                ))
            }
        };
        self.layout(name)?;
        Ok(Some(name))
    }
    /// Private callers supply one of the initialized integer-only logical wire
    /// types matched by name; a size mismatch is rejected before touching bytes.
    pub(crate) fn pack<T>(&self, name: &str, value: &T) -> Result<Wire> {
        let logical = data::CANONICAL
            .layout(name)
            .ok_or_else(|| self.unsupported(name, "no canonical schema"))?;
        if logical.size != size_of::<T>() {
            return Err(self.unsupported(
                name,
                &format!("logical size {} != {}", size_of::<T>(), logical.size),
            ));
        }
        let target = self.layout(name)?;
        let mut wire = Wire {
            logical,
            target,
            bytes: vec![0; target.size],
        };
        for field in logical.fields {
            // Read named fields only: do not read Rust's possibly uninitialized padding.
            let input = unsafe {
                std::slice::from_raw_parts(
                    (value as *const T).cast::<u8>().add(field.offset),
                    field.len,
                )
            };
            if let Some(output) = target.field(field.name) {
                let n = input.len().min(output.len);
                wire.bytes[output.offset..output.offset + n].copy_from_slice(&input[..n]);
                if input[n..].iter().any(|b| *b != 0) {
                    return Err(
                        self.unsupported(name, &format!("nonzero truncated field {}", field.name))
                    );
                }
            } else if !(name == "NVOS21_PARAMETERS" && field.name == "paramsSize")
                && input.iter().any(|b| *b != 0)
            {
                return Err(
                    self.unsupported(name, &format!("nonzero unsupported field {}", field.name))
                );
            }
        }
        Ok(wire)
    }
}

pub(crate) struct Wire {
    logical: &'static Layout,
    target: &'static Layout,
    bytes: Vec<u8>,
}
impl Wire {
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }
    pub(crate) fn pointer(&mut self) -> *mut c_void {
        self.bytes.as_mut_ptr().cast()
    }
    pub(crate) fn address(&mut self) -> u64 {
        self.bytes.as_mut_ptr() as usize as u64
    }
    pub(crate) fn unpack<T>(&self, value: &mut T) -> Result<()> {
        if size_of::<T>() != self.logical.size {
            return Err(Error::Contract(format!(
                "{}: changed logical size during decode",
                self.logical.name
            )));
        }
        for output in self.logical.fields {
            if let Some(input) = self.target.field(output.name) {
                let n = input.len.min(output.len);
                // SAFETY: private integer-only wire caller, size checked; named
                // fields are bounds checked by metadata generation/tests.
                unsafe {
                    ptr::copy_nonoverlapping(
                        self.bytes.as_ptr().add(input.offset),
                        (value as *mut T).cast::<u8>().add(output.offset),
                        n,
                    );
                }
                // Older RM ASCII names may be 128 bytes; the logical output is
                // an explicitly truncated 64-byte C string, always terminated.
                if self.logical.name == "NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS"
                    && output.name == "gpuNameString"
                {
                    unsafe {
                        (value as *mut T)
                            .cast::<u8>()
                            .add(output.offset + output.len - 1)
                            .write(0);
                    }
                }
            } else if self.logical.name == "NV2080_CTRL_FB_GET_STATIC_BAR1_INFO_PARAMS" {
                // These outputs did not exist in older public contracts. Never
                // preserve a caller's previous value as an old driver answer.
                unsafe {
                    ptr::write_bytes(
                        (value as *mut T).cast::<u8>().add(output.offset),
                        0,
                        output.len,
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn assert_logical_fields(name: &str, size: usize, fields: &[(&str, usize, usize)]) {
    let layout = data::CANONICAL.layout(name).unwrap();
    assert_eq!(layout.size, size, "{name} size");
    assert_eq!(layout.fields.len(), fields.len(), "{name} field count");
    for &(name, offset, width) in fields {
        let field = layout.field(name).unwrap();
        assert_eq!(
            (field.offset, field.len),
            (offset, width),
            "{} {name}",
            layout.name
        );
    }
}

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn measured_versions() -> impl Iterator<Item = Compat> {
    data::VERSIONS
        .iter()
        .map(|&(version, profile)| Compat { version, profile })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi;
    #[test]
    fn optional_static_bar1_exact_layouts_and_missing_outputs() {
        const NAME: &str = "NV2080_CTRL_FB_GET_STATIC_BAR1_INFO_PARAMS";
        assert_eq!(std::mem::size_of::<abi::StaticBar1Info>(), 24);
        assert_eq!(std::mem::align_of::<abi::StaticBar1Info>(), 8);
        let mut present = 0;
        let mut size_only = 0;
        let mut with_range = 0;
        let mut with_wc = 0;
        for c in measured_versions() {
            if c.constant("NV2080_CTRL_CMD_FB_GET_STATIC_BAR1_INFO")
                .is_none()
            {
                assert!(c.control_layout(0x20801354).is_err());
                assert!(!c.supports_static_bar1_range());
                continue;
            }
            assert_eq!(c.control_layout(0x20801354).unwrap(), NAME);
            let mut value = abi::StaticBar1Info {
                padding: [0xff; 6],
                ..Default::default()
            };
            let mut wire = c.pack(NAME, &value).unwrap();
            assert!(
                wire.bytes.iter().all(|byte| *byte == 0),
                "{} padding",
                c.version()
            );
            let has_start = wire.target.field("staticBar1StartOffset").is_some();
            let has_wc = wire.target.field("bStaticBar1WriteCombined").is_some();
            assert_eq!(c.supports_static_bar1_range(), has_start);
            for (name, answer) in [
                ("bStaticBar1Enabled", 1u64),
                ("bStaticBar1WriteCombined", 1),
                ("staticBar1StartOffset", 0x1234_5678_0000_0000),
                ("staticBar1Size", 0x8765_4321_0000_0000),
            ] {
                if let Some(field) = wire.target.field(name) {
                    wire.bytes[field.offset..field.offset + field.len]
                        .copy_from_slice(&answer.to_ne_bytes()[..field.len]);
                }
            }
            // Missing old outputs must not preserve these caller sentinels.
            value.write_combined = 0xff;
            value.start = u64::MAX;
            wire.unpack(&mut value).unwrap();
            assert_eq!(value.enabled, 1);
            assert_eq!(value.write_combined, u8::from(has_wc));
            assert_eq!(
                value.start,
                if has_start { 0x1234_5678_0000_0000 } else { 0 }
            );
            assert_eq!(value.size, 0x8765_4321_0000_0000);
            assert_eq!(value.padding, [0xff; 6], "logical padding not touched");
            c.require_constant("NVOS33_FLAGS_MAPPING_REFLECTED", 2)
                .unwrap();
            c.require_constant("NVOS33_FLAGS_MAPPING_REFLECTED_ENCODED", 0x10000)
                .unwrap();
            c.require_constant("NVOS33_FLAGS_MAPPING_MASK", 0x18000)
                .unwrap();
            present += 1;
            size_only += usize::from(!has_start);
            with_range += usize::from(has_start);
            with_wc += usize::from(has_wc);
        }
        assert_eq!(
            (present, size_only, with_range, with_wc),
            (90, 2, 88, 57),
            "update only after rerunning the exact-tag static BAR1 C oracle"
        );
    }
    #[test]
    fn logical_fields_match_independent_c_offsets_without_padding() {
        macro_rules! fields {
            ($name:literal, $ty:ty, $($wire:literal => $($member:ident).+),+ $(,)?) => {{
                let layout = data::CANONICAL.layout($name).unwrap();
                let value = <$ty>::default();
                let mut count = 0;
                $(let field = layout.field($wire).unwrap();
                assert_eq!(field.offset, std::mem::offset_of!($ty, $($member).+), "{} {}", $name, $wire);
                assert_eq!(field.len, std::mem::size_of_val(&value.$($member).+), "{} {}", $name, $wire);
                count += 1;)+
                assert_eq!(count, layout.fields.len(), "unverified field in {}", $name);
            }};
        }
        fields!("NV2080_CTRL_FB_GET_STATIC_BAR1_INFO_PARAMS", abi::StaticBar1Info,
            "bStaticBar1Enabled" => enabled,
            "bStaticBar1WriteCombined" => write_combined,
            "staticBar1StartOffset" => start,
            "staticBar1Size" => size,
        );
        fields!("NVOS21_PARAMETERS", abi::Alloc,
            "hRoot" => root,
            "hObjectParent" => parent,
            "hObjectNew" => object,
            "hClass" => class,
            "pAllocParms" => parameters,
            "paramsSize" => size,
            "status" => status,
        );
        fields!("NVOS54_PARAMETERS", abi::Control,
            "hClient" => client,
            "hObject" => object,
            "cmd" => command,
            "flags" => flags,
            "params" => parameters,
            "paramsSize" => size,
            "status" => status,
        );
        fields!("NVOS00_PARAMETERS", abi::Free,
            "hRoot" => root,
            "hObjectParent" => parent,
            "hObjectOld" => object,
            "status" => status,
        );
        fields!("nv_pci_info_t", abi::Pci,
            "domain" => domain,
            "bus" => bus,
            "slot" => slot,
            "function" => function,
            "vendor_id" => vendor,
            "device_id" => device,
        );
        fields!("nv_ioctl_card_info_t", abi::Card,
            "valid" => valid,
            "pci_info.domain" => pci.domain,
            "pci_info.bus" => pci.bus,
            "pci_info.slot" => pci.slot,
            "pci_info.function" => pci.function,
            "pci_info.vendor_id" => pci.vendor,
            "pci_info.device_id" => pci.device,
            "gpu_id" => gpu_id,
            "interrupt_line" => interrupt,
            "reg_address" => registers,
            "reg_size" => register_size,
            "fb_address" => framebuffer,
            "fb_size" => framebuffer_size,
            "minor_number" => minor,
            "dev_name" => name,
        );
        fields!("NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS", abi::GpuInfo,
            "gpuId" => gpu_id,
            "gpuFlags" => flags,
            "deviceInstance" => device_instance,
            "subDeviceInstance" => subdevice_instance,
            "sliStatus" => sli_status,
            "boardId" => board_id,
            "gpuInstance" => gpu_instance,
            "numaId" => numa_id,
        );
        fields!("NV0080_ALLOC_PARAMETERS", abi::DeviceAlloc,
            "deviceId" => device_id,
            "hClientShare" => client_share,
            "hTargetClient" => target_client,
            "hTargetDevice" => target_device,
            "flags" => flags,
            "vaSpaceSize" => va_size,
            "vaStartInternal" => va_start,
            "vaLimitInternal" => va_limit,
            "vaMode" => va_mode,
        );
        fields!("NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS", abi::ClassList,
            "numClasses" => count,
            "classList" => classes,
        );
        fields!("NV_MEMORY_ALLOCATION_PARAMS", abi::MemoryAlloc,
            "owner" => owner,
            "type" => memory_type,
            "flags" => flags,
            "width" => width,
            "height" => height,
            "pitch" => pitch,
            "attr" => attr,
            "attr2" => attr2,
            "format" => format,
            "comprCovg" => compression,
            "zcullCovg" => zcull,
            "rangeLo" => range_low,
            "rangeHi" => range_high,
            "size" => size,
            "alignment" => alignment,
            "offset" => offset,
            "limit" => limit,
            "address" => address,
            "ctagOffset" => ctag_offset,
            "hVASpace" => vaspace,
            "internalflags" => internal_flags,
            "tag" => tag,
            "numaNode" => numa_node,
        );
        fields!("NV2080_CTRL_GPU_GET_ENGINE_CLASSLIST_PARAMS", abi::EngineClassList,
            "engineType" => engine_type, "numClasses" => count, "classList" => classes,
        );
        fields!("NV2080_CTRL_PERF_BOOST_PARAMS", abi::PerfBoost,
            "flags" => flags, "duration" => duration,
        );
        fields!("nv_ioctl_alloc_os_event_t", abi::OsEvent,
            "hClient" => client, "hDevice" => device, "fd" => fd, "Status" => status,
        );
        fields!("nv_ioctl_free_os_event_t", abi::OsEvent,
            "hClient" => client, "hDevice" => device, "fd" => fd, "Status" => status,
        );
        fields!("NV0005_ALLOC_PARAMETERS", abi::EventAlloc,
            "hParentClient" => parent_client, "hSrcResource" => source,
            "hClass" => class, "notifyIndex" => notify_index, "data" => data,
        );
        fields!("NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS", abi::EventNotification,
            "event" => event, "action" => action, "bNotifyState" => notify_state,
            "info32" => info32, "info16" => info16,
        );
        fields!("NV2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS", abi::EventTrigger,
            "hEvent" => event,
        );
        fields!("NVOS34_PARAMETERS", abi::UnmapMemory,
            "hClient" => client, "hDevice" => device, "hMemory" => memory,
            "pLinearAddress" => linear_address, "status" => status, "flags" => flags,
        );
        fields!("NVB0B5_ALLOCATION_PARAMETERS", abi::CopyEngine,
            "version" => version, "engineType" => engine_type,
        );
        fields!("NV_BSP_ALLOCATION_PARAMETERS", abi::VideoDecoder,
            "size" => size, "prohibitMultipleInstances" => prohibit_multiple_instances,
            "engineInstance" => engine_instance,
        );
        fields!("NV2080_CTRL_GPU_GET_ENGINE_PARTNERLIST_PARAMS", abi::EnginePartners,
            "engineType" => engine_type, "partnershipClassId" => partnership_class,
            "runqueue" => runqueue, "numPartners" => count, "partnerList" => partners,
        );
        fields!("NVOS33_PARAMETERS", abi::MapMemory,
            "hClient" => client,
            "hDevice" => device,
            "hMemory" => memory,
            "offset" => offset,
            "length" => length,
            "pLinearAddress" => linear_address,
            "status" => status,
            "flags" => flags,
        );
        fields!("nv_ioctl_nvos33_parameters_with_fd", abi::MapWithFd,
            "params.hClient" => parameters.client,
            "params.hDevice" => parameters.device,
            "params.hMemory" => parameters.memory,
            "params.offset" => parameters.offset,
            "params.length" => parameters.length,
            "params.pLinearAddress" => parameters.linear_address,
            "params.status" => parameters.status,
            "params.flags" => parameters.flags,
            "fd" => fd,
        );
        fields!("NV_VASPACE_ALLOCATION_PARAMETERS", abi::VaSpace,
            "index" => index,
            "flags" => flags,
            "vaSize" => size,
            "vaStartInternal" => internal_start,
            "vaLimitInternal" => internal_limit,
            "bigPageSize" => big_page_size,
            "vaBase" => base,
            "pasid" => pasid,
        );
        fields!("NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS", abi::VirtualMemory,
            "offset" => offset,
            "limit" => limit,
            "hVASpace" => vaspace,
        );
        fields!("NVOS46_PARAMETERS", abi::MapDma,
            "hClient" => client,
            "hDevice" => device,
            "hDma" => dma,
            "hMemory" => memory,
            "offset" => offset,
            "length" => length,
            "flags" => flags,
            "flags2" => flags2,
            "kindOverride" => kind,
            "dmaOffset" => address,
            "status" => status,
        );
        fields!("NVOS47_PARAMETERS", abi::UnmapDma,
            "hClient" => client,
            "hDevice" => device,
            "hDma" => dma,
            "hMemory" => memory,
            "flags" => flags,
            "dmaOffset" => address,
            "size" => size,
            "status" => status,
        );
        fields!("NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS", abi::ChannelGroup,
            "hObjectError" => error,
            "hObjectEccError" => ecc_error,
            "hVASpace" => vaspace,
            "engineType" => engine,
            "bIsCallingContextVgpuPlugin" => vgpu,
            "reserved" => reserved,
            "internalFlags" => flags,
        );
        fields!("NV_CTXSHARE_ALLOCATION_PARAMETERS", abi::ContextShare,
            "hVASpace" => vaspace,
            "flags" => flags,
            "subctxId" => subcontext,
        );
        fields!("NV_MEMORY_DESC_PARAMS", abi::MemoryDescriptor,
            "base" => base,
            "size" => size,
            "addressSpace" => address_space,
            "cacheAttrib" => cache,
        );
        fields!("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS", abi::Channel,
            "hObjectError" => error,
            "hObjectBuffer" => buffer,
            "gpFifoOffset" => fifo_address,
            "gpFifoEntries" => entries,
            "flags" => flags,
            "hContextShare" => context_share,
            "hVASpace" => vaspace,
            "hHandleVASpace" => handle_vaspace,
            "hUserdMemory" => userd_memory,
            "userdOffset" => userd_offsets,
            "engineType" => engine,
            "cid" => cid,
            "subDeviceId" => subdevice,
            "hObjectEccError" => ecc_error,
            "instanceMem.base" => instance_memory.base,
            "instanceMem.size" => instance_memory.size,
            "instanceMem.addressSpace" => instance_memory.address_space,
            "instanceMem.cacheAttrib" => instance_memory.cache,
            "userdMem.base" => userd_memory_descriptor.base,
            "userdMem.size" => userd_memory_descriptor.size,
            "userdMem.addressSpace" => userd_memory_descriptor.address_space,
            "userdMem.cacheAttrib" => userd_memory_descriptor.cache,
            "ramfcMem.base" => ramfc_memory.base,
            "ramfcMem.size" => ramfc_memory.size,
            "ramfcMem.addressSpace" => ramfc_memory.address_space,
            "ramfcMem.cacheAttrib" => ramfc_memory.cache,
            "mthdbufMem.base" => method_memory.base,
            "mthdbufMem.size" => method_memory.size,
            "mthdbufMem.addressSpace" => method_memory.address_space,
            "mthdbufMem.cacheAttrib" => method_memory.cache,
            "hPhysChannelGroup" => physical_group,
            "internalFlags" => internal_flags,
            "errorNotifierMem.base" => error_memory.base,
            "errorNotifierMem.size" => error_memory.size,
            "errorNotifierMem.addressSpace" => error_memory.address_space,
            "errorNotifierMem.cacheAttrib" => error_memory.cache,
            "eccErrorNotifierMem.base" => ecc_error_memory.base,
            "eccErrorNotifierMem.size" => ecc_error_memory.size,
            "eccErrorNotifierMem.addressSpace" => ecc_error_memory.address_space,
            "eccErrorNotifierMem.cacheAttrib" => ecc_error_memory.cache,
            "ProcessID" => process_id,
            "SubProcessID" => subprocess_id,
            "encryptIv" => encrypt_iv,
            "decryptIv" => decrypt_iv,
            "hmacNonce" => hmac_nonce,
            "tpcConfigID" => tpc_config,
        );
        fields!("NV2080_CTRL_GR_GET_INFO_PARAMS", abi::GrInfo,
            "grInfoListSize" => count,
            "grInfoList" => entries,
            "grRouteInfo.flags" => route_flags,
            "grRouteInfo.route" => route,
        );
        fields!("NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS", abi::EngineContextProperties,
            "engineId" => engine,
            "alignment" => alignment,
            "size" => size,
        );
        fields!("NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS", abi::ZcullBind,
            "hClient" => client,
            "hChannel" => channel,
            "vMemPtr" => address,
            "zcullMode" => mode,
        );
        fields!("NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS", abi::ZcullInfo,
            "widthAlignPixels" => width_align_pixels,
            "heightAlignPixels" => height_align_pixels,
            "pixelSquaresByAliquots" => pixel_squares_by_aliquots,
            "aliquotTotal" => aliquot_total,
            "zcullRegionByteMultiplier" => region_byte_multiplier,
            "zcullRegionHeaderSize" => region_header_size,
            "zcullSubregionHeaderSize" => subregion_header_size,
            "subregionCount" => subregion_count,
            "subregionWidthAlignPixels" => subregion_width_align_pixels,
            "subregionHeightAlignPixels" => subregion_height_align_pixels,
        );
    }
    #[test]
    fn per_client_perf_vote_fields_and_encoded_flags_match_exact_c_headers() {
        let mut checked = 0;
        for &(version, _) in data::VERSIONS {
            let c = Compat::for_version(version).unwrap();
            if c.layout("NV2080_CTRL_PERF_BOOST_PARAMS").is_err() {
                assert!(c.control_layout(0x2080200a).is_err());
                continue;
            }
            for (name, value) in [
                ("NV2080_CTRL_CMD_PERF_BOOST", 0x2080200a),
                ("NV2080_CTRL_PERF_BOOST_FLAGS_CMD_CLEAR", 0),
                ("NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX", 2),
                ("NV2080_CTRL_PERF_BOOST_FLAGS_ASYNC_YES", 1),
                ("PERF_BOOST_ASYNC_FLAG", 0x20),
                ("NV2080_CTRL_PERF_BOOST_DURATION_MAX", 3600),
            ] {
                c.require_constant(name, value).unwrap();
            }
            let wire = c
                .pack(
                    c.control_layout(0x2080200a).unwrap(),
                    &abi::PerfBoost {
                        flags: 0x22,
                        duration: 10,
                    },
                )
                .unwrap();
            assert_eq!(wire.bytes, [0x22, 0, 0, 0, 10, 0, 0, 0], "{version}");
            checked += 1;
        }
        assert_eq!(checked, 225);
    }
    #[test]
    fn cpu_unmap_cookie_preserves_full_width_and_public_status_offset() {
        assert_eq!(std::mem::size_of::<abi::UnmapMemory>(), 32);
        assert_eq!(std::mem::align_of::<abi::UnmapMemory>(), 8);
        let mut checked = 0;
        for &(version, _) in data::VERSIONS {
            let c = Compat::for_version(version).unwrap();
            if c.layout("NVOS34_PARAMETERS").is_err() {
                continue;
            }
            let mut value = abi::UnmapMemory {
                client: 0x12345678,
                device: 0x87654321,
                memory: 0x13572468,
                padding: u32::MAX,
                linear_address: 0xfedc_ba98_7654_3004,
                status: 0,
                flags: 0x0302_0000,
            };
            let mut wire = c.pack("NVOS34_PARAMETERS", &value).unwrap();
            assert_eq!(
                &wire.bytes[16..24],
                &value.linear_address.to_ne_bytes(),
                "{version}"
            );
            assert_eq!(&wire.bytes[12..16], &[0; 4], "padding must not escape");
            wire.bytes[24..28].copy_from_slice(&0x51u32.to_ne_bytes());
            wire.unpack(&mut value).unwrap();
            assert_eq!(value.status, 0x51);
            assert_eq!(value.linear_address, 0xfedc_ba98_7654_3004);
            assert_eq!(value.flags, 0x0302_0000);
            c.require_constant("NV_ESC_RM_UNMAP_MEMORY", 0x4f).unwrap();
            c.require_constant("ioctl_rm_unmap_memory", 0xc020464f)
                .unwrap();
            checked += 1;
        }
        assert_eq!(
            checked, 225,
            "update only after running the exact-tag C oracle"
        );
    }
    #[test]
    fn every_live_core_contract_packs_for_all_measured_rm_releases() {
        let mut clients = 0;
        let mut external = 0;
        let mut fences = 0;
        for &(version, _) in data::VERSIONS {
            let c = Compat::for_version(version).unwrap();
            if c.require_client_abi().is_err() {
                continue;
            }
            clients += 1;
            macro_rules! control {
                ($command:expr, $value:expr) => {{
                    let name = c.control_layout($command).unwrap();
                    c.pack(name, &$value)
                        .unwrap_or_else(|e| panic!("{version} {name}: {e}"));
                }};
            }
            macro_rules! alloc {
                ($class:expr, $value:expr) => {{
                    let value = $value;
                    let name = c
                        .allocation_layout($class, std::mem::size_of_val(&value))
                        .unwrap()
                        .unwrap();
                    c.pack(name, &value)
                        .unwrap_or_else(|e| panic!("{version} {name}: {e}"));
                }};
            }
            control!(0x205, abi::GpuInfo::default());
            control!(0xd01, [1u32, 0, 0]);
            control!(0x800201, abi::ClassList::default());
            control!(0x801c02, crate::video::BspCapsV2::default());
            control!(0x20800110, [0u32; 17]);
            control!(0x20800403, 0u64);
            control!(0x20801201, abi::GrInfo::default());
            control!(0x801707, abi::EngineContextProperties::default());
            control!(0x20801208, abi::ZcullBind::default());
            control!(0x20801206, abi::ZcullInfo::default());
            c.require_constant(
                "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_ZCULL",
                8,
            )
            .unwrap();
            c.require_constant("NV2080_CTRL_CTXSW_ZCULL_MODE_SEPARATE_BUFFER", 2)
                .unwrap();
            control!(0x2080122a, [0u64; 3]);
            control!(0x2080122b, [0u64, 0, 1]);
            control!(0x20801301, [1u64, 0]);
            control!(0x20801701, [0u32; 4]);
            control!(0xa06c0101, [1u8, 0, 0]);
            control!(0xc36f0108, 0u32);
            control!(0x20800147, abi::EnginePartners::default());
            control!(0x2080200a, abi::PerfBoost::default());
            control!(0x20800124, abi::EngineClassList::default());
            for class in [
                0xb0b5, 0xc0b5, 0xc1b5, 0xc3b5, 0xc5b5, 0xc6b5, 0xc7b5, 0xc9b5, 0xcab5,
            ] {
                alloc!(class, abi::CopyEngine::default());
                // The layout must never permit a parameterless CE allocation.
                assert!(c
                    .pack(c.allocation_layout(class, 0).unwrap().unwrap(), &())
                    .is_err());
            }
            c.pack("NVOS34_PARAMETERS", &abi::UnmapMemory::default())
                .unwrap();
            alloc!(0x80, abi::DeviceAlloc::default());
            alloc!(0x2080, 0u32);
            for class in [0x3e, 0x40, 0x50a0] {
                alloc!(class, abi::MemoryAlloc::default());
            }
            alloc!(0x70, abi::VirtualMemory::default());
            alloc!(0x90f1, abi::VaSpace::default());
            alloc!(0xa06c, abi::ChannelGroup::default());
            alloc!(0x9067, abi::ContextShare::default());
            alloc!(0xcfb0, abi::VideoDecoder::default());
            for class in [
                0xb06f, 0xc06f, 0xc36f, 0xc46f, 0xc56f, 0xc76f, 0xc86f, 0xc96f,
            ] {
                alloc!(class, abi::Channel::default());
            }
            for class in [
                0xc361, 0xc461, 0xc561, 0xc661, 0x902d, 0xa140, 0xcd40, 0xb097, 0xb197, 0xc097,
                0xc197, 0xc397, 0xc597, 0xc697, 0xc797, 0xc997, 0xce97, 0xb0c0, 0xb1c0, 0xc0c0,
                0xc1c0, 0xc3c0, 0xc5c0, 0xc6c0, 0xc7c0, 0xc9c0, 0xcec0,
            ] {
                assert!(c.allocation_layout(class, 0).unwrap().is_none());
            }
            c.pack("NVOS21_PARAMETERS", &abi::Alloc::default()).unwrap();
            c.pack("NVOS54_PARAMETERS", &abi::Control::default())
                .unwrap();
            c.pack("NVOS00_PARAMETERS", &abi::Free::default()).unwrap();
            c.pack(
                "nv_ioctl_nvos33_parameters_with_fd",
                &abi::MapWithFd::default(),
            )
            .unwrap();
            c.pack("NVOS46_PARAMETERS", &abi::MapDma::default())
                .unwrap();
            c.pack("NVOS47_PARAMETERS", &abi::UnmapDma::default())
                .unwrap();
            for (name, value) in [
                ("NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC", 0x20),
                ("NV2080_CTRL_FB_INFO_INDEX_RAM_SIZE", 7),
                ("NV2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_ASCII", 0),
                ("NV2080_ENGINE_TYPE_GRAPHICS", 1),
                ("NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM", 1),
                ("ADDR_SPACE_TYPE_SYSMEM", 1),
                ("ADDR_SPACE_TYPE_VIDMEM", 2),
                ("NV_CTXSHARE_SUBCONTEXT_SYNC", 0),
                ("NVOS32_TYPE_IMAGE", 0),
                ("NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT", 1),
                ("NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE", 0x100),
                ("NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED", 0x4000),
                ("NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED", 0x8000),
                ("NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM", 0x10000),
                ("NVOS32_ATTR_PHYSICALITY_CONTIGUOUS", 0x10000000),
                ("NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS", 0x18000000),
                ("NVOS32_ATTR_LOCATION_PCI", 0x2000000),
                ("NVOS32_ATTR_COHERENCY_CACHED", 0x20000000),
                ("NVOS32_ATTR2_ZBC_PREFER_NO_ZBC", 1),
                ("NVOS32_ATTR2_GPU_CACHEABLE_YES", 4),
                ("NVOS32_ATTR2_GPU_CACHEABLE_NO", 8),
                ("NVOS46_FLAGS_CACHE_SNOOP_ENABLE", 0x10),
                ("NVOS46_FLAGS_PAGE_SIZE_4KB", 0x100),
                ("NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE", 0x8000),
                ("NV_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR", 0),
                ("NV_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_bit", 15),
                ("DRM_IOCTL_NVIDIA_GEM_IMPORT_NVKMS_MEMORY", 0xc0206441),
                ("DRM_IOCTL_NVIDIA_GEM_EXPORT_NVKMS_MEMORY", 0xc0186449),
                ("NvKmsSurfaceMemoryLayoutBlockLinear", 0),
            ] {
                c.require_constant(name, value)
                    .unwrap_or_else(|e| panic!("{version}: {e}"));
            }
            external += usize::from(crate::external::validate_compat(c).is_ok());
            fences += usize::from(crate::fences::validate_compat(c).is_ok());
        }
        assert_eq!(clients, 225);
        assert_eq!(external, 225);
        assert_eq!(fences, 137);
    }
    #[test]
    fn exact_membership_and_unknown_version_are_distinct() {
        assert_eq!(data::VERSIONS.len(), 335);
        assert_eq!(
            data::VERSIONS
                .iter()
                .filter(|(v, _)| Compat::for_version(v).unwrap().require_client_abi().is_ok())
                .count(),
            225
        );
        assert!(Compat::for_version("580.94.13").is_ok());
        assert!(Compat::for_version("580.94.14").is_err());
        assert!(Compat::for_version("615.999").is_err());
        assert!(Compat::for_version("470.256.02")
            .unwrap()
            .require_client_abi()
            .is_err());
        assert!(Compat::for_version("615.71.09")
            .unwrap()
            .require_client_abi()
            .is_ok());
        // Published OGKM releases probed like every other tag; each must
        // resolve to its measured profile, not to a neighbour's.
        for version in ["580.196.07", "595.44.16", "595.112.07", "615.71.10"] {
            assert!(Compat::for_version(version)
                .unwrap()
                .require_client_abi()
                .is_ok());
        }
    }
    #[test]
    fn every_field_fits_and_current_rust_sizes_match_census() {
        for (_, profile) in data::VERSIONS {
            for (_, layout) in profile.layouts {
                for field in layout.fields {
                    assert!(
                        field.offset + field.len <= layout.size,
                        "{} {}",
                        layout.name,
                        field.name
                    );
                }
            }
        }
        for (name, size) in abi::layouts() {
            if let Some(l) = data::CANONICAL.layout(name) {
                assert_eq!(l.size, size, "{name}");
            }
        }
    }
    #[test]
    fn alloc_status_is_decoded_at_old_offset_not_params_size() {
        let c = Compat::for_version("515.43.04").unwrap();
        let mut p = abi::Alloc {
            size: 376,
            ..Default::default()
        };
        let mut w = c.pack("NVOS21_PARAMETERS", &p).unwrap();
        assert_eq!(w.target.field("status").unwrap().offset, 24);
        w.bytes[24..28].copy_from_slice(&0x1234u32.to_ne_bytes());
        w.unpack(&mut p).unwrap();
        assert_eq!(p.status, 0x1234);
        assert_eq!(p.size, 376);
    }
    #[test]
    fn channel_fields_relocate_and_nonzero_removed_features_fail() {
        for &(version, _) in data::VERSIONS {
            let c = Compat::for_version(version).unwrap();
            if c.layout("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS").is_err() {
                continue;
            }
            let mut p = abi::Channel {
                fifo_address: 0x1122334455667788,
                entries: 1024,
                engine: 1,
                ..Default::default()
            };
            p.userd_memory[0] = 0x1234;
            p.userd_offsets[0] = 8192;
            let w = c
                .pack("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS", &p)
                .unwrap();
            let mut out = abi::Channel::default();
            w.unpack(&mut out).unwrap();
            assert_eq!(out.fifo_address, p.fifo_address);
            assert_eq!(out.userd_memory, p.userd_memory);
            assert_eq!(out.userd_offsets, p.userd_offsets);
            assert_eq!(out.engine, p.engine);
        }
        let c = Compat::for_version("580.178.04").unwrap();
        assert!(c
            .pack(
                "NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS",
                &abi::Channel {
                    handle_vaspace: 1,
                    ..Default::default()
                }
            )
            .is_err());
    }
    #[test]
    fn schedule_and_kind_and_map_status_follow_exact_schemas() {
        let old = Compat::for_version("545.29.06").unwrap();
        assert_eq!(old.schedule_size().unwrap(), 2);
        assert!(!old.supports_semsurf_bind());
        assert!(!old.supports_kind_override());
        assert_eq!(
            old.pack("NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS", &[1u8, 0, 0])
                .unwrap()
                .len(),
            2
        );
        assert!(old
            .pack("NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS", &[1u8, 0, 1])
            .is_err());
        assert!(old
            .pack(
                "NVOS46_PARAMETERS",
                &abi::MapDma {
                    kind: 6,
                    ..Default::default()
                }
            )
            .is_err());
        assert_eq!(
            old.pack("NVOS47_PARAMETERS", &abi::UnmapDma::default())
                .unwrap()
                .len(),
            40
        );
        assert!(old
            .pack(
                "NVOS47_PARAMETERS",
                &abi::UnmapDma {
                    size: 4096,
                    ..Default::default()
                }
            )
            .is_err());
        let mut p = abi::MapDma::default();
        let mut w = old.pack("NVOS46_PARAMETERS", &p).unwrap();
        w.bytes[40..48].copy_from_slice(&0x12345678000u64.to_ne_bytes());
        w.bytes[48..52].copy_from_slice(&0x51u32.to_ne_bytes());
        w.unpack(&mut p).unwrap();
        assert_eq!(p.address, 0x12345678000);
        assert_eq!(p.status, 0x51);
        let new = Compat::for_version("580.65.06").unwrap();
        assert!(new.supports_kind_override());
        assert!(new.supports_semsurf_bind());
    }
    #[test]
    fn branch_backport_layouts_and_allocation_tails_use_c_offsets() {
        // Independent C probe rows: the 535.43 development branch gained CC
        // fields before the numerically later 535.54 release branch did.
        let dev = Compat::for_version("535.43.08").unwrap();
        let release = Compat::for_version("535.54.03").unwrap();
        assert_eq!(
            dev.layout_size("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS")
                .unwrap(),
            360
        );
        assert_eq!(
            release
                .layout_size("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS")
                .unwrap(),
            304
        );
        let mut p = abi::Channel::default();
        p.userd_memory[0] = 0x12345678;
        p.userd_offsets[0] = 0x8877665544332211;
        let wire = release
            .pack("NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS", &p)
            .unwrap();
        assert_eq!(&wire.bytes[32..36], &0x12345678u32.to_ne_bytes());
        assert_eq!(&wire.bytes[64..72], &0x8877665544332211u64.to_ne_bytes());
        assert_eq!(
            release
                .pack(
                    "NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS",
                    &abi::ChannelGroup::default()
                )
                .unwrap()
                .len(),
            20
        );
        assert_eq!(
            release
                .pack("NV_MEMORY_ALLOCATION_PARAMS", &abi::MemoryAlloc::default())
                .unwrap()
                .len(),
            120
        );
        let old = Compat::for_version("515.43.04").unwrap();
        assert_eq!(
            old.pack("NV_MEMORY_ALLOCATION_PARAMS", &abi::MemoryAlloc::default())
                .unwrap()
                .len(),
            120
        );
        assert!(old
            .pack(
                "NV_MEMORY_ALLOCATION_PARAMS",
                &abi::MemoryAlloc {
                    numa_node: 2,
                    ..Default::default()
                }
            )
            .is_err());
    }
    #[test]
    fn array_backed_controls_decode_public_named_fields() {
        fn put(wire: &mut Wire, name: &str, value: u64) {
            let field = wire.target.field(name).unwrap();
            wire.bytes[field.offset..field.offset + field.len]
                .copy_from_slice(&value.to_ne_bytes()[..field.len]);
        }
        for c in measured_versions().filter(|c| c.require_client_abi().is_ok()) {
            let mut gpc = [0u64; 3];
            let mut wire = c.pack(c.control_layout(0x2080122a).unwrap(), &gpc).unwrap();
            put(&mut wire, "grRouteInfo.flags", 3);
            put(&mut wire, "grRouteInfo.route", 7);
            put(&mut wire, "gpcMask", 0xa5);
            wire.unpack(&mut gpc).unwrap();
            assert_eq!(gpc, [3, 7, 0xa5]);
            let mut tpc = [0u64, 0, 2];
            let mut wire = c.pack(c.control_layout(0x2080122b).unwrap(), &tpc).unwrap();
            put(&mut wire, "tpcMask", 0x13);
            wire.unpack(&mut tpc).unwrap();
            assert_eq!(tpc, [0, 0, (0x13 << 32) | 2]);
            let mut arch = [0u32; 4];
            let mut wire = c
                .pack(c.control_layout(0x20801701).unwrap(), &arch)
                .unwrap();
            for (name, value) in [
                ("architecture", 1),
                ("implementation", 2),
                ("revision", 3),
                ("subRevision", 4),
            ] {
                put(&mut wire, name, value);
            }
            wire.unpack(&mut arch).unwrap();
            assert_eq!(arch, [1, 2, 3, 4]);
            let mut fb = [0u64; 2];
            let mut wire = c.pack(c.control_layout(0x20801301).unwrap(), &fb).unwrap();
            put(&mut wire, "fbInfoListSize", 3);
            put(&mut wire, "fbInfoList", 0x1234);
            wire.unpack(&mut fb).unwrap();
            assert_eq!(fb, [3, 0x1234]);
            let mut space = [0u32; 3];
            let mut wire = c.pack(c.control_layout(0xd01).unwrap(), &space).unwrap();
            put(&mut wire, "hObject", 1);
            put(&mut wire, "mapFlags", 2);
            put(&mut wire, "addrSpaceType", 3);
            wire.unpack(&mut space).unwrap();
            assert_eq!(space, [1, 2, 3]);
        }
    }
    #[test]
    fn old_name_is_bounded_and_terminated() {
        let c = Compat::for_version("580.178.04").unwrap();
        let mut name = [0u32; 17];
        let mut w = c
            .pack("NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS", &name)
            .unwrap();
        assert_eq!(w.len(), 132);
        w.bytes[4..].fill(b'x');
        w.unpack(&mut name).unwrap();
        assert_eq!(name[16].to_ne_bytes(), [b'x', b'x', b'x', 0]);
    }
}
