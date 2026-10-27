// SPDX-License-Identifier: MIT
//! Source-defined RM <-> NVIDIA DRM PRIME external memory.
//!
//! Authorities: matching open kernel module ctrl0000unix.h (0x3d05/0x3d06),
//! nv_drm_common_ioctl.h, nvkms-kapi-private.h and nvkms-kapi.c::ImportMemory.
//! The RM ioctl217 dma-buf exporter is deliberately NOT this graphics path:
//! nv-dmabuf.c:1056 rejects NVIDIA GPU attachment to its VRAM exports.
//!
//! Every operation opens a fresh DRM file, so temporary GEM handles cannot
//! alias and accidentally close handles already owned by Mesa/WSI on its fd.
//! Returned dma-buf fds own kernel references independent of the RM allocation.
use crate::diag::diag;
use crate::{
    platform,
    rm::{Device, Error, Memory, Result, RmHandle},
};
use nv_rm_core::conv::u64_from_usize;
use std::{
    fs::{File, OpenOptions},
    io,
    mem::size_of,
    os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
    rc::Rc,
};
use zerocopy::{FromBytes, Immutable, IntoBytes};

#[repr(C)]
#[derive(Default, FromBytes, IntoBytes, Immutable)]
struct RmObject {
    kind: u32,
    device: u32,
    parent: u32,
    object: u32,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct ExportObject {
    object: RmObject,
    fd: i32,
    flags: u32,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct ImportObject {
    fd: i32,
    object: RmObject,
}
#[repr(C)]
#[derive(Default, FromBytes, IntoBytes, Immutable)]
struct Surface {
    layout: u32,
    log2_gobs_x: u32,
    log2_gobs_y: u32,
    log2_gobs_z: u32,
    pitch_blocks: u32,
    generic_memory: u8,
    padding: [u8; 3],
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct ImportMemory {
    fd: i32,
    surface: Surface,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct ImportGem {
    size: u64,
    parameters: u64,
    parameters_size: u64,
    handle: u32,
    padding: u32,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct ExportGem {
    handle: u32,
    padding: u32,
    parameters: u64,
    parameters_size: u64,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct Prime {
    handle: u32,
    flags: u32,
    fd: i32,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct Identify {
    handle: u32,
    kind: u32,
}
#[repr(C)]
#[derive(FromBytes, IntoBytes, Immutable)]
struct CloseGem {
    handle: u32,
    padding: u32,
}

fn request(number: u8, size: usize, direction: u32) -> u32 {
    // _IOC_SIZEBITS = 14 (asm-generic/ioctl.h): the size field is 13:0 of bits 29:16.
    let size = u32::try_from(size)
        .ok()
        .filter(|&s| s < 16384)
        .expect("ioctl argument structs are below 16 KiB");
    (direction << 30) | (size << 16) | (u32::from(b'd') << 8) | u32::from(number)
}
fn drm_ioctl<T: FromBytes + IntoBytes + Immutable>(
    fd: BorrowedFd<'_>,
    number: u8,
    direction: u32,
    value: &mut T,
) -> Result<()> {
    // SAFETY: `value` is a live, exclusively borrowed buffer of exactly the
    // encoded size; any bytes the kernel writes are a valid T (FromBytes).
    // Nested pointer targets live until this synchronous call returns.
    let result = unsafe {
        platform::ioctl(
            fd.as_raw_fd(),
            request(number, size_of::<T>(), direction),
            std::ptr::from_mut::<T>(value).cast(),
        )
    };
    if result < 0 {
        Err(Error::Io(
            "NVIDIA DRM external-memory ioctl",
            io::Error::last_os_error(),
        ))
    } else {
        Ok(())
    }
}
fn open(path: &str) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| Error::Io("open external-memory device", e))
}
fn isolated_drm(fd: BorrowedFd<'_>) -> Result<File> {
    // Opening the proc fd path creates a new DRM open-file description, unlike
    // dup/F_DUPFD_CLOEXEC, which would share the caller's GEM handle namespace.
    open(&format!("/proc/self/fd/{}", fd.as_raw_fd()))
}
struct Gem<'a> {
    file: &'a File,
    handle: u32,
}
impl Drop for Gem<'_> {
    fn drop(&mut self) {
        let mut close = CloseGem {
            handle: self.handle,
            padding: 0,
        };
        if let Err(error) = drm_ioctl(self.file.as_fd(), 0x09, 1, &mut close) {
            diag!("external GEM cleanup: {error}");
        }
    }
}

/// Base native-memory sharing contract. Foreign GEM import has an additional
/// per-release exporter/identify gate below. Validate before advertising PRIME.
pub(crate) fn validate(device: &Device) -> Result<()> {
    validate_compat(device.client.compat)
}

pub(crate) fn validate_compat(abi: crate::compat::Compat) -> Result<()> {
    for name in [
        "drm_nvidia_gem_import_nvkms_memory_params",
        "drm_nvidia_gem_export_nvkms_memory_params",
        "NvKmsKapiPrivImportMemoryParams",
        "NvKmsKapiPrivSurfaceParams",
        "NvKmsKapiPrivExportMemoryParams",
    ] {
        abi.require_same_layout(name)?;
    }
    for (name, value) in [
        ("DRM_NVIDIA_GEM_IMPORT_NVKMS_MEMORY", 1),
        ("DRM_NVIDIA_GEM_EXPORT_NVKMS_MEMORY", 9),
        ("NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD", 0x3d05),
        ("NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD", 0x3d06),
    ] {
        abi.require_constant(name, value)?;
    }
    Ok(())
}

/// Export a generic allocation. Image layout must be provided to consumers as
/// an explicit DRM modifier and stride; no legacy implicit-tiling inference.
/// NVKMS itself uses layout=BlockLinear, genericMemory=true for generic memory
/// (nvkms-kapi.c:1666); explicit_layout overrides its geometry (2266-2329).
/// This does not alter the allocation's bytes, PTE kind or GPU VA mappings.
pub(crate) fn export(memory: &Rc<Memory>, drm_fd: BorrowedFd<'_>) -> Result<OwnedFd> {
    let device = &memory.device;
    validate(device)?;
    let client = &device.client;
    let drm = isolated_drm(drm_fd)?;
    // RM control0x3d05 requires a fresh *control* fd (nv.c:4333); GPU fd fails.
    let opaque = open("/dev/nvidiactl")?;
    let mut rm_export = ExportObject {
        object: RmObject {
            kind: 1,
            device: device.handle.get(),
            parent: device.handle.get(),
            object: memory.handle.get(),
        },
        fd: opaque.as_raw_fd(),
        flags: 0,
    };
    client.control(client.root, 0x3d05, &mut rm_export)?;
    let mut private = ImportMemory {
        fd: opaque.as_raw_fd(),
        surface: Surface {
            generic_memory: 1,
            ..Default::default()
        },
    };
    let mut imported = ImportGem {
        size: u64_from_usize(memory.size),
        parameters: u64_from_usize((&raw mut private).expose_provenance()),
        parameters_size: u64_from_usize(size_of::<ImportMemory>()),
        handle: 0,
        padding: 0,
    };
    drm_ioctl(drm.as_fd(), 0x41, 3, &mut imported)?;
    let gem = Gem {
        file: &drm,
        handle: imported.handle,
    };
    let mut prime = Prime {
        handle: gem.handle,
        flags: 0x80000 | 2, // DRM_CLOEXEC is Linux O_CLOEXEC, not bit zero.
        fd: -1,
    }; // DRM_CLOEXEC | DRM_RDWR
    drm_ioctl(drm.as_fd(), 0x2d, 3, &mut prime)?;
    if prime.fd < 0 {
        return Err(Error::Contract(
            "DRM PRIME export returned invalid fd".into(),
        ));
    }
    // SAFETY: successful PRIME_HANDLE_TO_FD returns a new owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(prime.fd) })
}

/// Import a borrowed dma-buf; ownership of its fd remains with the caller.
/// On success RM owns a separate memory reference. Vulkan's successful-import
/// fd consumption must be implemented by its caller, not this backend helper.
pub(crate) fn import(
    device: &Rc<Device>,
    drm_fd: BorrowedFd<'_>,
    dma_buf: BorrowedFd<'_>,
) -> Result<Rc<Memory>> {
    validate(device)?;
    // SAFETY: fd is borrowed and live; dma-bufs define SEEK_END as buffer size.
    let size = unsafe { platform::lseek(dma_buf.as_raw_fd(), 0, 2) };
    if size < 0 {
        return Err(Error::Io("query dma-buf size", io::Error::last_os_error()));
    }
    if size == 0 || size % 4096 != 0 {
        return Err(Error::Contract(
            "dma-buf size must be nonzero and page-aligned".into(),
        ));
    }
    let size = usize::try_from(size).map_err(|error| {
        Error::Contract(format!(
            "dma-buf does not fit process address space: {error}"
        ))
    })?;
    let drm = isolated_drm(drm_fd)?;
    let mut prime = Prime {
        handle: 0,
        flags: 0,
        fd: dma_buf.as_raw_fd(),
    };
    drm_ioctl(drm.as_fd(), 0x2e, 3, &mut prime)?;
    let gem = Gem {
        file: &drm,
        handle: prime.handle,
    };
    let abi = device.client.compat;
    let export_command = if abi.supports_identify_object() {
        abi.require_same_layout("drm_nvidia_gem_identify_object_params")?;
        let mut identity = Identify {
            handle: gem.handle,
            kind: u32::MAX,
        };
        drm_ioctl(drm.as_fd(), 0x4e, 3, &mut identity)?;
        match identity.kind {
            0 => 0x49, // NV_GEM_OBJECT_NVKMS
            1 => {
                abi.require_constant("DRM_NVIDIA_GEM_EXPORT_DMABUF_MEMORY", 0xd)?;
                abi.require_same_layout("drm_nvidia_gem_export_dmabuf_memory_params")?;
                0x4d
            }
            other => {
                return Err(Error::Contract(format!(
                    "unsupported external DRM GEM type {other}"
                )));
            }
        }
    } else {
        // Older drivers expose only the native exporter. It checks the GEM
        // type in-kernel and fails for foreign memory; never guess its type or
        // convert a foreign buffer through CPU memory on failure.
        0x49
    };
    let opaque = open("/dev/nvidiactl")?;
    let mut opaque_fd = opaque.as_raw_fd();
    let mut export_gem = ExportGem {
        handle: gem.handle,
        padding: 0,
        parameters: u64_from_usize((&raw mut opaque_fd).expose_provenance()),
        parameters_size: u64_from_usize(size_of::<i32>()),
    };
    drm_ioctl(drm.as_fd(), export_command, 3, &mut export_gem)?;
    let mut imported = ImportObject {
        fd: opaque.as_raw_fd(),
        object: RmObject {
            kind: 1,
            device: device.handle.get(),
            parent: device.handle.get(),
            object: 0,
        },
    };
    let client = &device.client;
    client.control(client.root, 0x3d06, &mut imported)?;
    let handle = imported.object.object;
    // ctrl0000client.h:60 -> hObject,mapFlags,addressSpaceType; 1=sysmem,2=vidmem.
    let mut address_space = [handle, 0, 0];
    let query = client.control(client.root, 0xd01, &mut address_space);
    let system = match query {
        Ok(()) if address_space[2] == 1 || address_space[2] == 2 => address_space[2] == 1,
        other => {
            if let Err(cleanup) = client.free(device.handle, RmHandle::new(handle)) {
                diag!("external RM import cleanup: {cleanup}");
            }
            other?;
            return Err(Error::Contract(format!(
                "unsupported imported address space {}",
                address_space[2]
            )));
        }
    };
    Ok(Memory::from_imported(
        Rc::clone(device),
        RmHandle::new(handle),
        size,
        system,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires NVIDIA RM; isolated SYS opaque-FD CPU visibility and GPU VA mapping, no GPU commands"]
    fn sysmem_opaque_fd_independent_clients() {
        use crate::{gpu::AddressSpace, rm::Client};
        let fd_count = || std::fs::read_dir("/proc/self/fd").unwrap().count();
        let before = fd_count();
        for source_dies_first in [false, true] {
            let source = Client::open().unwrap();
            let mut failures = Vec::new();
            let (card, source_device) = source
                .cards()
                .unwrap()
                .into_iter()
                .filter(|card| card.pci.vendor == 0x10de)
                .find_map(|card| match source.device(card) {
                    Ok(device) => Some((card, device)),
                    Err(error) => {
                        failures.push(format!("minor {}: {error}", card.minor));
                        None
                    }
                })
                .unwrap_or_else(|| panic!("no openable NVIDIA device: {}", failures.join("; ")));
            let target = Client::open().unwrap();
            assert!(!Rc::ptr_eq(&source, &target));
            let target_device = target.device(card).unwrap();
            let memory = source_device
                .allocate_aligned(4096, 4096, false, true)
                .unwrap();
            // This diagnostic owns the source; its extent is trusted. A production
            // importer must query object type, size and cache attributes from RM.
            let size = memory.size;
            let mut source_map = memory.map().unwrap();
            source_map.write_u32(0, 0x12345678).unwrap();
            source_map.write_u32(size - 4, 0x87654321).unwrap();
            let source_va = AddressSpace::new(&source_device).unwrap();
            let source_gpu_map = source_va.map(&memory).unwrap();
            let opaque = OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/nvidiactl")
                .unwrap();
            let mut export = ExportObject {
                object: RmObject {
                    kind: 1,
                    device: source_device.handle.get(),
                    parent: source_device.handle.get(),
                    object: memory.handle.get(),
                },
                fd: opaque.as_raw_fd(),
                flags: 0,
            };
            source.control(source.root, 0x3d05, &mut export).unwrap();
            let version = source.version.clone();
            let mut source_owners = Some((
                source_gpu_map,
                source_va,
                source_map,
                memory,
                source_device,
                source,
            ));
            if source_dies_first {
                drop(source_owners.take());
            }
            let mut import = ImportObject {
                fd: opaque.as_raw_fd(),
                object: RmObject {
                    kind: 1,
                    device: target_device.handle.get(),
                    parent: target_device.handle.get(),
                    object: 0,
                },
            };
            target.control(target.root, 0x3d06, &mut import).unwrap();
            let imported = Memory::from_imported(
                Rc::clone(&target_device),
                RmHandle::new(import.object.object),
                size,
                true,
            );
            let mut address_space = [imported.handle.get(), 0, 0];
            target
                .control(target.root, 0xd01, &mut address_space)
                .unwrap();
            assert_eq!(address_space[2], 1, "import must be SYS memory");
            drop(opaque);
            let mut target_map = imported.map().unwrap();
            let target_va = AddressSpace::new(&target_device).unwrap();
            let target_gpu_map = target_va.map(&imported).unwrap();
            assert_eq!(target_map.read_u32(0).unwrap(), 0x12345678);
            assert_eq!(target_map.read_u32(size - 4).unwrap(), 0x87654321);
            target_map.write_u32(0, 0xfeedcafe).unwrap();
            target_map.write_u32(size - 4, 0xcafefeed).unwrap();
            if let Some((_, _, original, _, _, _)) = &source_owners {
                assert_eq!(original.read_u32(0).unwrap(), 0xfeedcafe);
                assert_eq!(original.read_u32(size - 4).unwrap(), 0xcafefeed);
            }
            drop(source_owners);
            // Both the source client and export FD are now gone.
            assert_eq!(target_map.read_u32(0).unwrap(), 0xfeedcafe);
            target_map.write_u32(0, 0x10203040).unwrap();
            assert_eq!(target_map.read_u32(0).unwrap(), 0x10203040);
            eprintln!(
                "SYS opaque FD PASS: driver {version}, device {:04x}, bytes {size}, source_dies_before_import={source_dies_first}, target GPU VA {:#x}",
                card.pci.device,
                target_gpu_map.address()
            );
            drop((
                target_gpu_map,
                target_va,
                target_map,
                imported,
                target_device,
                target,
            ));
            assert_eq!(fd_count(), before, "opaque SYS import leaked an FD");
        }
    }
    #[test]
    fn nested_and_linux_uapi_fields_match_supplemental_c_oracle() {
        // build/gpu-coverage/cp2-nested-contracts.c: independent public C
        // offsets and widths, LP64 output plus ILP32 static assertions.
        assert_eq!(
            (std::mem::offset_of!(RmObject, kind), size_of::<u32>()),
            (0, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(RmObject, device), size_of::<u32>()),
            (4, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(RmObject, parent), size_of::<u32>()),
            (8, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(RmObject, object), size_of::<u32>()),
            (12, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(Prime, handle), size_of::<u32>()),
            (0, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(Prime, flags), size_of::<u32>()),
            (4, 4)
        );
        assert_eq!((std::mem::offset_of!(Prime, fd), size_of::<i32>()), (8, 4));
        assert_eq!(
            (std::mem::offset_of!(CloseGem, handle), size_of::<u32>()),
            (0, 4)
        );
        assert_eq!(
            (std::mem::offset_of!(CloseGem, padding), size_of::<u32>()),
            (4, 4)
        );
    }
    #[test]
    fn actual_private_fields_match_public_census() {
        crate::compat::assert_logical_fields(
            "NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS",
            size_of::<ExportObject>(),
            &[
                (
                    "object",
                    std::mem::offset_of!(ExportObject, object),
                    size_of::<RmObject>(),
                ),
                (
                    "fd",
                    std::mem::offset_of!(ExportObject, fd),
                    size_of::<i32>(),
                ),
                (
                    "flags",
                    std::mem::offset_of!(ExportObject, flags),
                    size_of::<u32>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS",
            size_of::<ImportObject>(),
            &[
                (
                    "fd",
                    std::mem::offset_of!(ImportObject, fd),
                    size_of::<i32>(),
                ),
                (
                    "object",
                    std::mem::offset_of!(ImportObject, object),
                    size_of::<RmObject>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "NvKmsKapiPrivSurfaceParams",
            size_of::<Surface>(),
            &[
                (
                    "layout",
                    std::mem::offset_of!(Surface, layout),
                    size_of::<u32>(),
                ),
                (
                    "blockLinear.log2GobsPerBlock.x",
                    std::mem::offset_of!(Surface, log2_gobs_x),
                    size_of::<u32>(),
                ),
                (
                    "blockLinear.log2GobsPerBlock.y",
                    std::mem::offset_of!(Surface, log2_gobs_y),
                    size_of::<u32>(),
                ),
                (
                    "blockLinear.log2GobsPerBlock.z",
                    std::mem::offset_of!(Surface, log2_gobs_z),
                    size_of::<u32>(),
                ),
                (
                    "blockLinear.pitchInBlocks",
                    std::mem::offset_of!(Surface, pitch_blocks),
                    size_of::<u32>(),
                ),
                (
                    "blockLinear.genericMemory",
                    std::mem::offset_of!(Surface, generic_memory),
                    size_of::<u8>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "NvKmsKapiPrivImportMemoryParams",
            size_of::<ImportMemory>(),
            &[
                (
                    "memFd",
                    std::mem::offset_of!(ImportMemory, fd),
                    size_of::<i32>(),
                ),
                (
                    "surfaceParams",
                    std::mem::offset_of!(ImportMemory, surface),
                    size_of::<Surface>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "drm_nvidia_gem_import_nvkms_memory_params",
            size_of::<ImportGem>(),
            &[
                (
                    "mem_size",
                    std::mem::offset_of!(ImportGem, size),
                    size_of::<u64>(),
                ),
                (
                    "nvkms_params_ptr",
                    std::mem::offset_of!(ImportGem, parameters),
                    size_of::<u64>(),
                ),
                (
                    "nvkms_params_size",
                    std::mem::offset_of!(ImportGem, parameters_size),
                    size_of::<u64>(),
                ),
                (
                    "handle",
                    std::mem::offset_of!(ImportGem, handle),
                    size_of::<u32>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "drm_nvidia_gem_export_nvkms_memory_params",
            size_of::<ExportGem>(),
            &[
                (
                    "handle",
                    std::mem::offset_of!(ExportGem, handle),
                    size_of::<u32>(),
                ),
                (
                    "nvkms_params_ptr",
                    std::mem::offset_of!(ExportGem, parameters),
                    size_of::<u64>(),
                ),
                (
                    "nvkms_params_size",
                    std::mem::offset_of!(ExportGem, parameters_size),
                    size_of::<u64>(),
                ),
            ],
        );
        crate::compat::assert_logical_fields(
            "drm_nvidia_gem_identify_object_params",
            size_of::<Identify>(),
            &[
                (
                    "handle",
                    std::mem::offset_of!(Identify, handle),
                    size_of::<u32>(),
                ),
                (
                    "object_type",
                    std::mem::offset_of!(Identify, kind),
                    size_of::<u32>(),
                ),
            ],
        );
    }
    #[test]
    fn live_external_contracts_pack_offline() {
        for c in crate::compat::measured_versions() {
            if c.require_client_abi().is_err() {
                continue;
            }
            validate_compat(c).unwrap();
            let mut export = ExportObject {
                object: RmObject {
                    kind: 1,
                    device: 2,
                    parent: 3,
                    object: 4,
                },
                fd: 5,
                flags: 0,
            };
            let w = c.pack(c.control_layout(0x3d05).unwrap(), &export).unwrap();
            export.object.object = 0;
            w.unpack(&mut export).unwrap();
            assert_eq!(export.object.object, 4);
            let imported = ImportObject {
                fd: 5,
                object: RmObject::default(),
            };
            c.pack(c.control_layout(0x3d06).unwrap(), &imported)
                .unwrap();
            c.pack(
                "NvKmsKapiPrivImportMemoryParams",
                &ImportMemory {
                    fd: 5,
                    surface: Surface {
                        generic_memory: 1,
                        ..Default::default()
                    },
                },
            )
            .unwrap();
            c.pack("NvKmsKapiPrivExportMemoryParams", &5i32).unwrap();
            c.pack(
                "drm_nvidia_gem_import_nvkms_memory_params",
                &ImportGem {
                    size: 4096,
                    parameters: 0,
                    parameters_size: 28,
                    handle: 0,
                    padding: 0,
                },
            )
            .unwrap();
            c.pack(
                "drm_nvidia_gem_export_nvkms_memory_params",
                &ExportGem {
                    handle: 1,
                    padding: 0,
                    parameters: 0,
                    parameters_size: 4,
                },
            )
            .unwrap();
            if c.supports_identify_object() {
                for (name, value) in [
                    ("NV_GEM_OBJECT_NVKMS", 0),
                    ("NV_GEM_OBJECT_DMABUF", 1),
                    ("DRM_IOCTL_NVIDIA_GEM_EXPORT_DMABUF_MEMORY", 0xc018644d),
                    ("DRM_IOCTL_NVIDIA_GEM_IDENTIFY_OBJECT", 0xc008644e),
                ] {
                    c.require_constant(name, value).unwrap();
                }
                c.require_same_layout("drm_nvidia_gem_identify_object_params")
                    .unwrap();
                c.pack(
                    "drm_nvidia_gem_identify_object_params",
                    &Identify { handle: 1, kind: 0 },
                )
                .unwrap();
                c.require_same_layout("drm_nvidia_gem_export_dmabuf_memory_params")
                    .unwrap();
            }
        }
    }
    #[test]
    fn public_external_memory_wire_layouts() {
        // Independent C probe uses matching installed-driver public headers.
        assert_eq!(size_of::<RmObject>(), 16);
        assert_eq!(size_of::<ExportObject>(), 24);
        assert_eq!(size_of::<ImportObject>(), 20);
        assert_eq!(size_of::<Surface>(), 24);
        assert_eq!(size_of::<ImportMemory>(), 28);
        assert_eq!(size_of::<ImportGem>(), 32);
        assert_eq!(size_of::<ExportGem>(), 24);
        assert_eq!(size_of::<Prime>(), 12);
        assert_eq!(size_of::<Identify>(), 8);
        assert_eq!(size_of::<CloseGem>(), 8);
        assert_eq!(request(0x41, 32, 3), 0xc0206441);
        assert_eq!(request(0x2d, 12, 3), 0xc00c642d);
        assert_eq!(request(0x09, 8, 1), 0x40086409);
    }
}
