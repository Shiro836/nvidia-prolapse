// SPDX-License-Identifier: MIT
//! Resource manager operations with explicit object ownership.
//! Public ABI authority is the matching NVIDIA open kernel module. Allocation
//! setup cross-checked against tinygrad NVKIface at 37132d7553a831fdaf900e2334d80380702c5564.
//! CPU mapping is here; gpu.rs owns GPU VA and actual channel submission.
use crate::diag::diag;
use crate::{abi, compat::Compat, platform};
use nv_rm_core::conv::{low_u32, u64_from_usize, usize_from_u32};
use std::{
    cell::{Cell, RefCell},
    fs::{File, OpenOptions},
    io,
    mem::size_of,
    os::fd::{AsRawFd, RawFd},
    ptr::NonNull,
    rc::{Rc, Weak},
    sync::Arc,
};
use zerocopy::{FromBytes, Immutable, IntoBytes};

pub use nv_rm_core::error::{ChannelError, Error, MemoryError, Result, SyncError, WaitError};
pub use nv_rm_core::units::{ByteSize, ClassId, FenceValue, GpuVa, RmHandle};
pub(crate) fn status(operation: &'static str, value: u32) -> Result<()> {
    if value == 0 {
        Ok(())
    } else {
        Err(Error::Rm {
            operation,
            status: value,
        })
    }
}
fn open(path: &str) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| Error::Io("open NVIDIA device", e))
}
// Callers pass the exact wire type of the request (integer-only, checked by
// the zerocopy bounds) and keep embedded pointer targets live for this
// synchronous ioctl.
pub(crate) fn exchange<T: FromBytes + IntoBytes + Immutable>(
    fd: RawFd,
    number: u8,
    value: &mut T,
    operation: &'static str,
) -> Result<()> {
    let request = abi::iowr(number, size_of::<T>());
    // SAFETY: `value` is a live, exclusively borrowed buffer of exactly the
    // size the request encodes, and any bytes the kernel writes are a valid T
    // (FromBytes). Embedded pointer targets (as u64 fields) are kept alive by
    // the caller for this synchronous call.
    let result = unsafe { platform::ioctl(fd, request, std::ptr::from_mut::<T>(value).cast()) };
    if result < 0 {
        Err(Error::Io(operation, io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

fn exchange_wire<T: FromBytes + IntoBytes + Immutable>(
    fd: RawFd,
    number: u8,
    compat: Compat,
    name: &str,
    value: &mut T,
    operation: &'static str,
) -> Result<()> {
    let mut wire = compat.pack(name, value)?;
    let request = abi::iowr(number, wire.len());
    // SAFETY: generated C layout and all named initialized fields were checked;
    // the owned zero-filled buffer and its embedded pointer targets live here.
    if unsafe { platform::ioctl(fd, request, wire.pointer()) } < 0 {
        return Err(Error::Io(operation, io::Error::last_os_error()));
    }
    wire.unpack(value)
}

/// The running kernel module's version (`NVRM version:` line of
/// `/proc/driver/nvidia/version`); the RM version check sends exactly this string.
///
/// # Errors
/// The file cannot be read, or has no `NVRM version:` line with a dotted version.
pub fn runtime_version() -> Result<String> {
    let text = std::fs::read_to_string("/proc/driver/nvidia/version")
        .map_err(|e| Error::Io("read running NVIDIA version", e))?;
    parse_runtime_version(&text)
}

fn parse_runtime_version(text: &str) -> Result<String> {
    text.lines()
        .find(|line| line.starts_with("NVRM version:"))
        .and_then(|line| {
            line.split_whitespace().find(|word| {
                word.contains('.') && word.bytes().all(|b| b.is_ascii_digit() || b == b'.')
            })
        })
        .map(str::to_owned)
        .ok_or_else(|| Error::Contract("unrecognized /proc/driver/nvidia/version".into()))
}

/// An RM client: the process's `/dev/nvidiactl` descriptor, the root client object
/// allocated on it, and the exact ABI profile of the running driver release. Freeing the
/// root on drop releases every object the client still owns.
pub struct Client {
    pub(crate) control: Arc<File>,
    pub(crate) root: RmHandle,
    /// The driver release the client was opened against ([`runtime_version`]).
    pub version: String,
    pub(crate) compat: Compat,
    pub(crate) cpu_page_size: usize,
    /// The client's entry in the list the exit handler frees ([`crate::exit`]).
    exit: crate::exit::Registration,
}
impl Client {
    /// Open `/dev/nvidiactl`, pass RM's strict version check and allocate the root client.
    ///
    /// # Errors
    /// The driver release has no exact ABI profile in this build, the version check fails
    /// (kernel module and userspace disagree), or opening or allocating fails.
    pub fn open() -> Result<Rc<Self>> {
        let cpu_page_size =
            platform::validate_runtime().map_err(|e| Error::Io("validate CPU mapping ABI", e))?;
        let version = runtime_version()?;
        let compat = Compat::for_version(&version)?;
        compat.require_client_abi()?;
        if version.len() >= 64 {
            return Err(Error::Contract("driver version exceeds RM buffer".into()));
        }
        let control = open("/dev/nvidiactl")?;
        let mut check = abi::Version {
            command: 0,
            reply: 0,
            version: [0; 64],
        };
        check.version[..version.len()].copy_from_slice(version.as_bytes());
        exchange(
            control.as_raw_fd(),
            210,
            &mut check,
            "strict RM version check",
        )?;
        if check.reply != 1 {
            return Err(Error::Contract(
                "RM did not recognize strict version check".into(),
            ));
        }
        let mut alloc = abi::Alloc {
            class: 0x41,
            ..Default::default()
        };
        exchange_wire(
            control.as_raw_fd(),
            0x2b,
            compat,
            "NVOS21_PARAMETERS",
            &mut alloc,
            "allocate RM client",
        )?;
        status("allocate RM client", alloc.status)?;
        if alloc.object == 0 {
            return Err(Error::Contract("RM returned a zero client handle".into()));
        }
        let root = RmHandle::new(alloc.object);
        let control = Arc::new(control);
        let exit_control = Arc::clone(&control);
        let exit = crate::exit::Registration::new(
            root,
            Box::new(move || free_object(&exit_control, compat, root, RmHandle::new(0), root)),
        );
        let exit = match exit {
            Ok(exit) => exit,
            Err(error) => {
                if let Err(free) = free_object(&control, compat, root, RmHandle::new(0), root) {
                    diag!("RM client cleanup: {free}");
                }
                return Err(Error::Io("register RM exit teardown", error));
            }
        };
        Ok(Rc::new(Self {
            control,
            root,
            version,
            compat,
            cpu_page_size,
            exit,
        }))
    }
    /// The NVIDIA GPUs RM enumerates (`NV_ESC_CARD_INFO`), valid entries only.
    ///
    /// # Errors
    /// The card-info ioctl fails or its layout differs from this build's.
    pub fn cards(&self) -> Result<Vec<abi::Card>> {
        self.compat.require_same_layout("nv_ioctl_card_info_t")?;
        let mut cards = [abi::Card::default(); 64];
        exchange(
            self.control.as_raw_fd(),
            200,
            &mut cards,
            "enumerate NVIDIA cards",
        )?;
        Ok(cards
            .into_iter()
            .filter(|c| c.valid != 0 && c.pci.vendor == 0x10de)
            .collect())
    }
    /// The GPU this process may use: `want` (a PCI BDF) or else the first RM
    /// card whose `/dev/nvidiaN` this process can open. Containers commonly
    /// see every host GPU through `/dev/nvidiactl` but get only some of the
    /// per-GPU nodes (EACCES/EPERM from the device cgroup, or ENOENT when the
    /// node is not created): those cards are skipped, never opened by
    /// accident. No fallback: an absent or inaccessible requested BDF, any
    /// other open error, or no accessible card at all fails with the reason
    /// for every candidate.
    pub(crate) fn select_card(&self, want: Option<[u32; 4]>) -> Result<abi::Card> {
        select_accessible_card(&self.cards()?, want, |minor| {
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(format!("/dev/nvidia{minor}"))
                .map(drop)
        })
    }
    pub(crate) fn control<T: FromBytes + IntoBytes + Immutable>(
        &self,
        object: RmHandle,
        command: u32,
        parameters: &mut T,
    ) -> Result<()> {
        self.port().control(object, command, parameters)
    }
    /// Plain-data handle for public RM controls from threads that never touch
    /// this Rc-owned client. Valid only while the Client (and its control fd)
    /// is alive; owners join such threads before dropping the client.
    pub(crate) fn port(&self) -> ControlPort {
        ControlPort {
            fd: self.control.as_raw_fd(),
            root: self.root,
            compat: self.compat,
        }
    }
    pub(crate) fn allocate<T: FromBytes + IntoBytes + Immutable>(
        &self,
        parent: RmHandle,
        class: ClassId,
        parameters: &mut T,
    ) -> Result<RmHandle> {
        let name = self.compat.allocation_layout(class.get(), size_of::<T>())?;
        let mut parameters_wire = name
            .map(|name| self.compat.pack(name, parameters))
            .transpose()?;
        let (address, size) = match parameters_wire.as_mut() {
            Some(wire) => (wire.address(), wire_size(wire.len())?),
            // Classes without a parameter layout allocate with no parameters.
            None => (0, 0),
        };
        let mut call = abi::Alloc {
            root: self.root.get(),
            parent: parent.get(),
            class: class.get(),
            parameters: address,
            size,
            ..Default::default()
        };
        exchange_wire(
            self.control.as_raw_fd(),
            0x2b,
            self.compat,
            "NVOS21_PARAMETERS",
            &mut call,
            "RM allocate",
        )?;
        status("RM allocate", call.status)?;
        if let Some(wire) = parameters_wire {
            wire.unpack(parameters)?;
        }
        if call.object == 0 {
            return Err(Error::Contract("RM returned zero allocation handle".into()));
        }
        Ok(RmHandle::new(call.object))
    }
    /// Allocate with a fixed-size byte payload whose public layout the caller
    /// verified against this release's measured layout (`layout_size`).
    pub(crate) fn allocate_bytes(
        &self,
        parent: RmHandle,
        class: ClassId,
        payload: &mut [u8],
    ) -> Result<RmHandle> {
        let mut call = abi::Alloc {
            root: self.root.get(),
            parent: parent.get(),
            class: class.get(),
            parameters: u64_from_usize(payload.as_mut_ptr().expose_provenance()),
            size: wire_size(payload.len())?,
            ..Default::default()
        };
        exchange_wire(
            self.control.as_raw_fd(),
            0x2b,
            self.compat,
            "NVOS21_PARAMETERS",
            &mut call,
            "RM allocate",
        )?;
        status("RM allocate", call.status)?;
        if call.object == 0 {
            return Err(Error::Contract("RM returned zero allocation handle".into()));
        }
        Ok(RmHandle::new(call.object))
    }
    pub(crate) fn map_dma(&self, parameters: &mut abi::MapDma) -> Result<()> {
        self.compat
            .require_constant("NV_ESC_RM_MAP_MEMORY_DMA", 0x57)?;
        exchange_wire(
            self.control.as_raw_fd(),
            0x57,
            self.compat,
            "NVOS46_PARAMETERS",
            parameters,
            "map GPU VA",
        )?;
        status("map GPU VA", parameters.status)
    }
    /// Full mapping teardown only; size=0 also matches the old no-size ABI.
    pub(crate) fn unmap_dma(&self, parameters: &mut abi::UnmapDma) -> Result<()> {
        self.compat
            .require_constant("NV_ESC_RM_UNMAP_MEMORY_DMA", 0x58)?;
        exchange_wire(
            self.control.as_raw_fd(),
            0x58,
            self.compat,
            "NVOS47_PARAMETERS",
            parameters,
            "unmap GPU VA",
        )?;
        status("unmap GPU VA", parameters.status)
    }
    pub(crate) fn free(&self, parent: RmHandle, object: RmHandle) -> Result<()> {
        free_object(&self.control, self.compat, self.root, parent, object)
    }
    /// Where RM placed `object` for a CPU mapping with NVOS33 `map_flags`
    /// (NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE, public ctrl0000client.h).
    fn address_space(&self, object: RmHandle, map_flags: u32) -> Result<AddressSpace> {
        // hObject, mapFlags, addrSpaceType.
        let mut query = [object.get(), map_flags, 0];
        self.control(self.root, 0xd01, &mut query)?;
        match query[2] {
            1 => Ok(AddressSpace::System),
            2 => Ok(AddressSpace::Video),
            3 => Ok(AddressSpace::Register),
            other => Err(Error::Contract(format!(
                "RM reports address space {other} for CPU mapping of object {:#x}",
                object.get()
            ))),
        }
    }
    /// Tell RM the CPU address `mmap` chose for a video-memory mapping
    /// (NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO, nv_escape.h; NVOS56, nvos.h).
    /// RM keys the mapping by this address from then on: NVOS34 must pass it
    /// (osapi.c RmUpdateDeviceMappingInfo replaces pLinearAddress, which
    /// rs_server.c serverUnmap matches).
    fn update_device_mapping(&self, device: u32, memory: u32, old: u64, new: u64) -> Result<()> {
        let mut call = abi::UpdateDeviceMapping {
            client: self.root.get(),
            device,
            memory,
            old_address: old,
            new_address: new,
            ..Default::default()
        };
        exchange(
            self.control.as_raw_fd(),
            0x5e,
            &mut call,
            "update CPU mapping address",
        )?;
        status("update CPU mapping address", call.status)
    }
    /// The long-lived `/dev/nvidiaN` of a device, registered with the control
    /// fd (NV_ESC_REGISTER_FD): the registered file holds a reference on the
    /// control file until it is closed (escape.c, osapi.c nvidia_close). Never
    /// mmapped: a mapping's VMA would keep the control file alive past exit.
    pub(crate) fn gpu_fd(&self, minor: u32) -> Result<File> {
        let gpu = open(&format!("/dev/nvidia{minor}"))?;
        let mut ctl_fd = self.control.as_raw_fd();
        exchange(
            gpu.as_raw_fd(),
            201,
            &mut ctl_fd,
            "register GPU fd with client",
        )?;
        Ok(gpu)
    }
    /// Open `card`'s `/dev/nvidiaN` and allocate its device and subdevice objects.
    ///
    /// # Errors
    /// `card` is not a valid NVIDIA entry of [`Client::cards`], its node cannot be opened,
    /// or RM refuses an allocation or control.
    pub fn device(self: &Rc<Self>, card: abi::Card) -> Result<Rc<Device>> {
        if card.valid == 0 || card.pci.vendor != 0x10de {
            return Err(Error::Contract(
                "card is not an enumerated NVIDIA GPU".into(),
            ));
        }
        let gpu = self.gpu_fd(card.minor)?;
        let mut info = abi::GpuInfo {
            gpu_id: card.gpu_id,
            ..Default::default()
        };
        self.control(self.root, 0x205, &mut info)?;
        let mut parameters = abi::DeviceAlloc {
            device_id: info.device_instance,
            client_share: self.root.get(),
            ..Default::default()
        };
        let handle = self.allocate(self.root, ClassId::new(0x80), &mut parameters)?;
        let subdevice =
            match self.allocate(handle, ClassId::new(0x2080), &mut info.subdevice_instance) {
                Ok(handle) => handle,
                Err(error) => {
                    if let Err(cleanup) = self.free(self.root, handle) {
                        diag!("cleanup: {cleanup}");
                    }
                    return Err(error);
                }
            };
        let device = Rc::new(Device {
            client: Rc::clone(self),
            handle,
            subdevice,
            subdevice_instance: info.subdevice_instance,
            events: RefCell::new(Weak::new()),
            _gpu: gpu,
            card,
            generic_kind: Cell::new(None),
            cpu_maps: Rc::new(crate::cpu_map_cache::Registry::default()),
            cpu_map_failed: Cell::new(false),
            channel_placement: Cell::new(None),
            bar_budget: Rc::new(BarBudget {
                used: Cell::new(0),
                limit: card.framebuffer_size,
            }),
        });
        // On an error the device drops here, freeing its RM objects.
        let placement = device.query_channel_placement()?;
        device.channel_placement.set(Some(placement));
        Ok(device)
    }
}
/// Borrowed control descriptor + client handle + exact ABI profile. Copy/Send:
/// it owns nothing, so its users must be joined/quiesced before the Client is
/// dropped (Bridge teardown order). Each call packs its own wire buffers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ControlPort {
    fd: RawFd,
    root: RmHandle,
    compat: Compat,
}
impl ControlPort {
    /// Test stand-in for logic that holds a port but never controls through
    /// it: no descriptor, so a control fails with EBADF.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self {
            fd: -1,
            root: RmHandle::default(),
            compat: crate::compat::measured_versions()
                .next()
                .expect("the compat table lists at least one release"),
        }
    }
    /// An RM control in the pointer-table form `{NvU32 size; NvP64 table}`
    /// (e.g. NV2080_CTRL_BUS_GET_INFO_PARAMS, NV0080_CTRL_FB_GET_CAPS_PARAMS):
    /// that parameter block has the same 16-byte layout on every release and
    /// both Android ABIs, so it is sent without the per-release layout table.
    /// RM reads and writes `table` during the call.
    pub(crate) fn control_table<E: FromBytes + IntoBytes + Immutable>(
        &self,
        object: RmHandle,
        command: u32,
        table: &mut [E],
    ) -> Result<()> {
        let mut parameters = abi::TableParams {
            size: u32::try_from(table.len())
                .map_err(|e| Error::Contract(format!("RM control table too long: {e}")))?,
            padding: 0,
            table: u64_from_usize(table.as_mut_ptr().expose_provenance()),
        };
        let mut call = abi::Control {
            client: self.root.get(),
            object: object.get(),
            command,
            parameters: u64_from_usize(std::ptr::from_mut(&mut parameters).expose_provenance()),
            size: wire_size(size_of::<abi::TableParams>())?,
            ..Default::default()
        };
        exchange_wire(
            self.fd,
            0x2a,
            self.compat,
            "NVOS54_PARAMETERS",
            &mut call,
            "RM control",
        )?;
        status("RM control", call.status)
    }
    pub(crate) fn control<T: FromBytes + IntoBytes + Immutable>(
        &self,
        object: RmHandle,
        command: u32,
        parameters: &mut T,
    ) -> Result<()> {
        let name = self.compat.control_layout(command)?;
        let mut parameters_wire = self.compat.pack(name, parameters)?;
        let mut call = abi::Control {
            client: self.root.get(),
            object: object.get(),
            command,
            parameters: parameters_wire.address(),
            size: wire_size(parameters_wire.len())?,
            ..Default::default()
        };
        exchange_wire(
            self.fd,
            0x2a,
            self.compat,
            "NVOS54_PARAMETERS",
            &mut call,
            "RM control",
        )?;
        status("RM control", call.status)?;
        parameters_wire.unpack(parameters)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // The exit handler may already have freed the root on another thread.
        if !self.exit.withdraw() {
            return;
        }
        if let Err(error) = self.free(RmHandle::new(0), self.root) {
            diag!("RM client cleanup: {error}");
        }
    }
}
/// NV_ESC_RM_FREE of `object` under `parent` in client `root` (NVOS00, nvos.h).
fn free_object(
    control: &File,
    compat: Compat,
    root: RmHandle,
    parent: RmHandle,
    object: RmHandle,
) -> Result<()> {
    let mut call = abi::Free {
        root: root.get(),
        parent: parent.get(),
        object: object.get(),
        status: 0,
    };
    exchange_wire(
        control.as_raw_fd(),
        0x29,
        compat,
        "NVOS00_PARAMETERS",
        &mut call,
        "RM free",
    )?;
    status("RM free", call.status)
}

/// One GPU opened on a [`Client`]: its device and subdevice objects, the open
/// `/dev/nvidiaN`, and the CPU mappings and BAR budget of memory allocated on it.
/// Single-threaded (`Rc`); objects allocated on the device keep it alive.
pub struct Device {
    pub(crate) client: Rc<Client>,
    pub(crate) handle: RmHandle,
    pub(crate) subdevice: RmHandle,
    pub(crate) subdevice_instance: u32,
    pub(crate) events: RefCell<Weak<crate::events::EventSource>>,
    _gpu: File,
    pub(crate) card: abi::Card,
    bar_budget: Rc<BarBudget>,
    generic_kind: Cell<Option<u32>>,
    cpu_maps: Rc<crate::cpu_map_cache::Registry<Mapping>>,
    cpu_map_failed: Cell<bool>,
    /// Where this GPU's channel buffers go, decided once at creation
    /// ([`channel_buffer::Placement`]).
    channel_placement: Cell<Option<channel_buffer::Placement>>,
}
impl Device {
    pub(crate) fn cpu_mapping_failed(&self) -> bool {
        self.cpu_map_failed.get()
    }
    pub(crate) fn cpu_mapping_slot(&self) -> Rc<crate::cpu_map_cache::Slot<Mapping>> {
        self.cpu_maps.slot()
    }
    /// RM's SM version encoding (`NV2080_CTRL_GR_INFO_INDEX_SM_VERSION`, e.g. `0x1200`).
    ///
    /// # Errors
    /// The RM control fails.
    pub fn sm_version(&self) -> Result<u32> {
        let mut entry = [0xcu32, 0]; // NV2080_CTRL_GR_INFO_INDEX_SM_VERSION
        let mut query = abi::GrInfo {
            count: 1,
            entries: u64_from_usize(entry.as_mut_ptr().expose_provenance()),
            ..Default::default()
        };
        self.client
            .control(self.subdevice, 0x20801201, &mut query)?;
        Ok(entry[1])
    }
    /// The classes the GPU supports (`NV0080_CTRL_CMD_GPU_GET_CLASSLIST`).
    ///
    /// # Errors
    /// The RM control fails, or its count is implausible or grows between the two calls.
    pub fn classes(&self) -> Result<Vec<u32>> {
        let mut query = abi::ClassList::default();
        self.client.control(self.handle, 0x800201, &mut query)?;
        if query.count > 65536 {
            return Err(Error::Contract("unreasonable RM class count".into()));
        }
        let mut classes = vec![0; usize_from_u32(query.count)];
        query.classes = u64_from_usize(classes.as_mut_ptr().expose_provenance());
        self.client.control(self.handle, 0x800201, &mut query)?;
        if usize_from_u32(query.count) > classes.len() {
            return Err(Error::Contract("RM class count grew during query".into()));
        }
        classes.truncate(usize_from_u32(query.count));
        Ok(classes)
    }
    /// Allocate `requested_size` bytes of video memory (rounded up to CPU pages).
    ///
    /// # Errors
    /// [`MemoryError`] for a zero size, or RM's refusal.
    pub fn allocate_vram(self: &Rc<Self>, requested_size: usize) -> Result<Rc<Memory>> {
        self.allocate_memory(requested_size, false)
    }
    pub(crate) fn allocate_memory(
        self: &Rc<Self>,
        requested_size: usize,
        contiguous: bool,
    ) -> Result<Rc<Memory>> {
        self.allocate_aligned(requested_size, 4096, contiguous, false)
    }
    /// NIL's uncompressed generic PTE kind (Fermi..Volta 0xfe, Turing+ 0x06).
    pub(crate) fn generic_kind(self: &Rc<Self>) -> Result<u32> {
        Ok(match self.generic_kind.get() {
            Some(kind) => kind,
            None => {
                let kind = crate::capabilities::query(self)?.generic_kind;
                self.generic_kind.set(Some(kind));
                kind
            }
        })
    }
    pub(crate) fn allocate_aligned(
        self: &Rc<Self>,
        requested_size: usize,
        alignment: usize,
        contiguous: bool,
        system: bool,
    ) -> Result<Rc<Memory>> {
        self.allocate_placed(requested_size, alignment, contiguous, system, false, None)
    }
    /// Memory that holds one block-linear (DRM modifier) image, uncompressed,
    /// with the image's PTE kind recorded on the physical allocation: NVOS32
    /// ATTR FORMAT BLOCK_LINEAR and `format` = the kind (public nvos.h
    /// NVOS32_ATTR_FORMAT 17:16). Our own GPU VAs carry the same kind
    /// explicitly; the physical attribute is what every other client of the
    /// memory gets. A dma-buf importer (nvidia-drm PRIME, then RM
    /// NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD) maps the memory with
    /// NVOS46 flags 0 and no kind override, i.e. with this kind: the stock
    /// 580/615 EGL importer does exactly that. A pitch-format request is
    /// granted kind PITCH (0x00) on Fermi..Volta, so a block-linear texture
    /// over it faults (Xid 13 TEX); Turing+ grants its generic kind 0x06,
    /// which serves both layouts, either way.
    ///
    /// NVK records the kind on the memory object the same way
    /// (nvk_device_memory.c:174-184 → nvkmd_dev_alloc_tiled_mem, nouveau
    /// winsys/nouveau_bo.c:120 `tile_flags = pte_kind << 8`, recovered by an
    /// import at :214). The stock driver also requests FORMAT BLOCK_LINEAR
    /// for dedicated / exportable / modifier images
    /// (libnvidia-eglcore.so.580.178.04 E 0xad7f18 `or $0x20000`;
    /// 615.78.08 E 0xa41b60), but together with COMPR ANY and kind 0 (RM picks
    /// a compressible kind, 0xdb on GP104, 0x08 on GB202), because its own VAs
    /// of that memory are compressible. Ours are not, and the modifier
    /// advertises no compression, so this asks for exactly the kind and no
    /// compression. RM must grant the requested kind; anything else is an
    /// error, never a different layout.
    pub(crate) fn allocate_block_linear(
        self: &Rc<Self>,
        requested_size: usize,
        alignment: usize,
        system: bool,
        kind: u32,
    ) -> Result<Rc<Memory>> {
        let generic = self.generic_kind()?;
        if kind != generic {
            return Err(MemoryError::BlockLinearKind { kind, generic }.into());
        }
        self.allocate_placed(requested_size, alignment, false, system, false, Some(kind))
    }
    /// Device-local memory that is never CPU-mapped or exported. Matches the
    /// stock 615 request (attr 0x01002000 /
    /// 0x09002000, attr2 0x4, format 0): COMPR ANY | PAGE_SIZE BIG, NONCONTIG
    /// only from 16 MiB, GPU-cacheable, ZBC default, RM picks the kind. RM
    /// grants compression on GPUs that have it (kind 0x8 on GB202).
    pub(crate) fn allocate_compressible(
        self: &Rc<Self>,
        requested_size: usize,
        alignment: usize,
    ) -> Result<Rc<Memory>> {
        if !requested_size.is_multiple_of(BIG_PAGE) {
            return Err(MemoryError::Invalid {
                operation: "allocate compressible memory",
                reason: "compressible allocation must be a 64 KiB multiple",
            }
            .into());
        }
        self.allocate_placed(
            requested_size,
            alignment.max(BIG_PAGE),
            false,
            false,
            true,
            None,
        )
    }
    /// One GPU-internal pool block (see [`pool_block`]): 2 MiB of device
    /// memory in one contiguous huge page, GPU-cached, uncompressed, never
    /// CPU-mapped or exported. Fails when RM grants other page size,
    /// location, physicality or compression than requested.
    pub(crate) fn allocate_pool_block(self: &Rc<Self>) -> Result<Rc<Memory>> {
        self.allocate_pool_block_granted().map(|(memory, _)| memory)
    }
    /// [`Self::allocate_pool_block`] plus what RM returned for it.
    pub(crate) fn allocate_pool_block_granted(
        self: &Rc<Self>,
    ) -> Result<(Rc<Memory>, PoolBlockGrant)> {
        let generic_kind = self.generic_kind()?;
        let size = u64_from_usize(POOL_BLOCK_SIZE);
        let mut parameters = abi::MemoryAlloc {
            owner: self.client.root.get(),
            memory_type: 0,
            flags: pool_block::FLAGS,
            attr: pool_block::ATTR,
            attr2: pool_block::ATTR2,
            format: generic_kind,
            size,
            alignment: 0,
            limit: size - 1,
            ..Default::default()
        };
        let handle = self
            .client
            .allocate(self.handle, ClassId::new(0x40), &mut parameters)?;
        let granted = parameters.attr & pool_block::ATTR_CHECKED;
        if granted != pool_block::ATTR & pool_block::ATTR_CHECKED || parameters.size != size {
            let status = self.client.free(self.handle, handle);
            return Err(MemoryError::PoolBlockGrant {
                attr: parameters.attr,
                size: parameters.size,
                requested_attr: pool_block::ATTR,
                requested_size: size,
                free: status.err().map(Box::new),
            }
            .into());
        }
        let grant = PoolBlockGrant {
            attr: parameters.attr,
            attr2: parameters.attr2,
            format: parameters.format,
            size: parameters.size,
        };
        let memory = Rc::new(Memory {
            device: Rc::clone(self),
            handle,
            parent: self.handle,
            size: POOL_BLOCK_SIZE,
            owned: true,
            charge_bar: false,
            _mapping_owner: None,
            nvos33_flags: 0,
            compressed: false,
            pages: PhysicalPages::Huge2M,
        });
        Ok((memory, grant))
    }
    /// The GPFIFO ring of a channel with a usermode doorbell: system memory,
    /// GPU-uncached, write-combined or write-back ([`channel_buffer`]), format
    /// 0, alignment 0, CPU-mapped through `/dev/nvidiactl` with RM's default
    /// mapping (NVOS33 flags 0).
    /// Read the inputs of the channel-buffer placement from RM and run the
    /// write-combine probe ([`channel_buffer`]).
    fn query_channel_placement(self: &Rc<Self>) -> Result<channel_buffer::Placement> {
        let port = self.client.port();
        // NV2080_CTRL_CMD_BUS_GET_INFO, entries {index, data} (ctrl2080bus.h).
        let mut bus = [
            [channel_buffer::BUS_NONCOHERENT_DMA_FLAGS, 0],
            [channel_buffer::BUS_GPU_GART_SIZE, 0],
            [channel_buffer::BUS_GPU_GART_SIZE_HI, 0],
        ];
        port.control_table(self.subdevice, 0x2080_1802, &mut bus)?;
        // NV0080_CTRL_CMD_FB_GET_CAPS, NV0080_CTRL_FB_CAPS_TBL_SIZE bytes.
        let mut caps = [0u8; 3];
        port.control_table(self.handle, 0x0080_1301, &mut caps)?;
        // NV2080_CTRL_CMD_FB_GET_INFO {index, data}, NV2080_CTRL_FB_INFO_INDEX_IS_ZERO_FB.
        let mut fb = [channel_buffer::FB_IS_ZERO_FB, 0u32];
        let mut query = [1u64, u64_from_usize(fb.as_mut_ptr().expose_provenance())];
        self.client
            .control(self.subdevice, 0x2080_1301, &mut query)?;
        let inputs = channel_buffer::Inputs {
            noncoherent_dma_flags: bus[0][1],
            gart_size_mib: (u64::from(bus[2][1]) << 32) | u64::from(bus[1][1]),
            fb_caps0: caps[0],
            zero_fb: fb[1] != 0,
        };
        channel_buffer::placement(inputs, |attr| self.probe_sysmem(attr))
    }
    /// One page of system memory with NVOS32 `attr`, freed at once: whether RM
    /// grants that coherency. An RM refusal is `false`; a failed ioctl or free
    /// is an error.
    fn probe_sysmem(self: &Rc<Self>, attr: u32) -> Result<bool> {
        let size = self.client.cpu_page_size;
        let mut parameters = abi::MemoryAlloc {
            owner: self.client.root.get(),
            flags: channel_buffer::PROBE_FLAGS,
            attr,
            size: u64_from_usize(size),
            limit: u64_from_usize(size) - 1,
            ..Default::default()
        };
        match self
            .client
            .allocate(self.handle, ClassId::new(0x3e), &mut parameters)
        {
            Ok(handle) => {
                self.client.free(self.handle, handle)?;
                Ok(true)
            }
            Err(Error::Rm { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }
    fn channel_placement(&self) -> Result<channel_buffer::Placement> {
        self.channel_placement
            .get()
            .ok_or_else(|| Error::Contract("channel placement not queried".into()))
    }
    pub(crate) fn allocate_gpfifo_ring(
        self: &Rc<Self>,
        requested_size: usize,
    ) -> Result<Rc<Memory>> {
        let placement = self.channel_placement()?;
        self.allocate_channel_buffer(requested_size, ClassId::new(0x3e), placement.ring, 0, false)
    }
    /// The USERD of a channel with a usermode doorbell ([`channel_buffer`]):
    /// video memory at RM's default page size and physicality, CPU-mapped
    /// through BAR1 with NVOS33 MAPPING DIRECT requested (flags 0x8000, as the
    /// NVIDIA driver maps its USERD block), or on a GPU without framebuffer
    /// system memory mapped like the ring.
    pub(crate) fn allocate_userd(self: &Rc<Self>) -> Result<Rc<Memory>> {
        match self.channel_placement()?.userd {
            channel_buffer::Userd::Video(request) => {
                self.allocate_channel_buffer(4096, ClassId::new(0x40), request, 0x8000, true)
            }
            channel_buffer::Userd::System(request) => {
                self.allocate_channel_buffer(4096, ClassId::new(0x3e), request, 0, false)
            }
        }
    }
    fn allocate_channel_buffer(
        self: &Rc<Self>,
        requested_size: usize,
        class: ClassId,
        request: channel_buffer::Request,
        nvos33_flags: u32,
        charge_bar: bool,
    ) -> Result<Rc<Memory>> {
        let size = platform::round_to_cpu_pages(requested_size, self.client.cpu_page_size)
            .map_err(|e| Error::Io("round RM allocation to CPU pages", e))?;
        let mut parameters = abi::MemoryAlloc {
            owner: self.client.root.get(),
            memory_type: 0,
            flags: request.flags,
            attr: request.attr,
            attr2: request.attr2,
            format: 0,
            size: u64_from_usize(size),
            alignment: 0,
            limit: u64_from_usize(size) - 1,
            ..Default::default()
        };
        let handle = self.client.allocate(self.handle, class, &mut parameters)?;
        Ok(Rc::new(Memory {
            device: Rc::clone(self),
            handle,
            parent: self.handle,
            size,
            owned: true,
            charge_bar,
            _mapping_owner: None,
            nvos33_flags,
            compressed: false,
            pages: PhysicalPages::Small4K,
        }))
    }
    fn allocate_placed(
        self: &Rc<Self>,
        requested_size: usize,
        alignment: usize,
        contiguous: bool,
        system: bool,
        compressible: bool,
        block_linear_kind: Option<u32>,
    ) -> Result<Rc<Memory>> {
        if alignment < 4096 || !alignment.is_power_of_two() {
            return Err(MemoryError::Invalid {
                operation: "allocate memory",
                reason: "invalid RM memory alignment",
            }
            .into());
        }
        if requested_size == 0 {
            return Err(MemoryError::Invalid {
                operation: "allocate memory",
                reason: "zero-sized VRAM allocation",
            }
            .into());
        }
        let size = platform::round_to_cpu_pages(requested_size, self.client.cpu_page_size)
            .map_err(|e| Error::Io("round RM allocation to CPU pages", e))?;
        let alignment = alignment.max(self.client.cpu_page_size);
        let generic_kind = self.generic_kind()?;
        let (attr, attr2, format) = if compressible {
            (
                compressible_attr(size),
                1 << 2, // GPU_CACHEABLE YES, ZBC DEFAULT
                0,      // RM chooses the (compressible) kind
            )
        } else {
            (
                ((if contiguous { 2 } else { 3 }) << 27)
                    | if system { (1 << 25) | (1 << 29) } else { 0 }
                    // FORMAT BLOCK_LINEAR (17:16 = 2); COMPR stays NONE.
                    | if block_linear_kind.is_some() { 2 << 16 } else { 0 },
                ((if system { 2 } else { 1 }) << 2) | 1, // SYS: GPU uncached; VRAM: GPU cached. ZBC_PREFER_NO_ZBC.
                block_linear_kind.unwrap_or(generic_kind),
            )
        };
        let mut parameters = abi::MemoryAlloc {
            owner: self.client.root.get(),
            memory_type: 0,
            // IGNORE_BANK_PLACEMENT | ALIGNMENT_FORCE | MEMORY_HANDLE_PROVIDED
            // | MAP_NOT_REQUIRED | PERSISTENT_VIDMEM (nvos.h).
            flags: 0x1 | 0x100 | 0x4000 | 0x8000 | if system { 0 } else { 0x10000 },
            attr,
            attr2,
            format,
            size: u64_from_usize(size),
            alignment: u64_from_usize(alignment),
            limit: u64_from_usize(size) - 1,
            ..Default::default()
        };
        let handle = self.client.allocate(
            self.handle,
            if system {
                ClassId::new(0x3e)
            } else {
                ClassId::new(0x40)
            },
            &mut parameters,
        )?;
        if let Some(kind) = block_linear_kind {
            // NVOS32 returns the granted PTE kind in `format` and the granted
            // attributes in `attr` (public nvos.h).
            if parameters.format != kind
                || (parameters.attr >> 16) & 3 != 2
                || (parameters.attr >> 12) & 3 != 0
            {
                let status = self.client.free(self.handle, handle);
                return Err(MemoryError::BlockLinearGrant {
                    size: u64_from_usize(size),
                    format: parameters.format,
                    attr: parameters.attr,
                    kind,
                    free: status.err().map(Box::new),
                }
                .into());
            }
        }
        if compressible && std::env::var_os("NV_RM_DEBUG_COMPRESSION").is_some() {
            crate::diag::diag!(
                "nvk-rm: compressible VRAM {size:#x}: RM attr {:#010x} kind {:#x} granted={}",
                parameters.attr,
                parameters.format,
                compression_granted(parameters.attr)
            );
        }
        Ok(Rc::new(Memory {
            device: Rc::clone(self),
            handle,
            parent: self.handle,
            size,
            owned: true,
            charge_bar: !system,
            _mapping_owner: None,
            nvos33_flags: 0,
            compressed: compressible && compression_granted(parameters.attr),
            pages: if compressible {
                PhysicalPages::Big64K
            } else {
                PhysicalPages::Small4K
            },
        }))
    }
}
/// RM big page size used for compressible memory and big-page mappings.
pub(crate) const BIG_PAGE: usize = 1 << 16;
/// Page size of a physical allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PhysicalPages {
    /// RM's default (4 KiB) pages.
    Small4K,
    /// 64 KiB big pages (compressible device-local memory).
    Big64K,
    /// 2 MiB huge pages, physically contiguous (a GPU-internal pool block).
    Huge2M,
}
/// The NVOS32 fields RM returns for a pool-block request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolBlockGrant {
    pub(crate) attr: u32,
    pub(crate) attr2: u32,
    pub(crate) format: u32,
    pub(crate) size: u64,
}
/// Size of one GPU-internal pool block (one 2 MiB huge page).
pub(crate) const POOL_BLOCK_SIZE: usize = 2 << 20;
/// NVOS32 request of a GPU-internal pool block, field by field from public
/// nvos.h; these are the values the proprietary driver (615.78.08) sends for
/// the device memory that holds its command buffers' QMDs, kernel parameter
/// blocks and launch counters (NV_ESC_RM_VID_HEAP_CONTROL ALLOC_SIZE: flags
/// 0xc00d, attr 0x11800000, attr2 0x100005, format = the generic kind, size
/// 2 MiB, alignment 0), which it maps once and never CPU-maps.
pub(crate) mod pool_block {
    /// IGNORE_BANK_PLACEMENT | FORCE_MEM_GROWS_DOWN | FORCE_ALIGN_HOST_PAGE |
    /// MEMORY_HANDLE_PROVIDED | MAP_NOT_REQUIRED.
    pub(crate) const FLAGS: u32 = 0x1 | 0x4 | 0x8 | 0x4000 | 0x8000;
    /// PAGE_SIZE (24:23) HUGE, PHYSICALITY (28:27) CONTIGUOUS; LOCATION VIDMEM,
    /// COHERENCY UNCACHED, COMPR NONE and FORMAT PITCH are the zero values.
    pub(crate) const ATTR: u32 = (3 << 23) | (2 << 27);
    /// ZBC (1:0) PREFER_NO_ZBC, GPU_CACHEABLE (3:2) YES, PAGE_SIZE_HUGE
    /// (21:20) 2MB.
    pub(crate) const ATTR2: u32 = 1 | (1 << 2) | (1 << 20);
    /// The granted attr must keep these fields as requested.
    pub(crate) const ATTR_CHECKED: u32 = (3 << 23) | (3 << 25) | (3 << 27) | (3 << 12);
}
/// Placement of the channel buffers of a GPU with a usermode doorbell, as the
/// NVIDIA driver decides it (libnvidia-eglcore.so.615.78.08: device init
/// E 0x968bc0, channel buffers E 0x9e0db0, allocator from E 0xa40622; no branch
/// on the GPU generation). NVOS32 fields from public nvos.h, RM inputs from
/// ctrl2080bus.h, ctrl0080fb.h and ctrl2080fb.h.
///
/// The NVIDIA driver also reads two registry keys here ("91851354", a sysmem
/// coherency override, and "ForceSysmemBuffers"); this driver has no registry,
/// which is their default (no override, video-memory USERD).
pub(crate) mod channel_buffer {
    use super::Result;

    /// One NVOS32 request: flags, attr, attr2.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Request {
        pub(crate) flags: u32,
        pub(crate) attr: u32,
        pub(crate) attr2: u32,
    }
    /// Where USERD goes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Userd {
        /// Video memory (NV01_MEMORY_LOCAL_USER), mapped through BAR1.
        Video(Request),
        /// System memory (NV01_MEMORY_SYSTEM), on a GPU without framebuffer.
        System(Request),
    }
    /// The decision for one GPU.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Placement {
        pub(crate) ring: Request,
        pub(crate) userd: Userd,
    }
    /// The RM values the decision reads.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Inputs {
        /// NV2080_CTRL_BUS_INFO_INDEX_NONCOHERENT_DMA_FLAGS.
        pub(crate) noncoherent_dma_flags: u32,
        /// NV2080_CTRL_BUS_INFO_INDEX_GPU_GART_SIZE{,_HI}, in MiB.
        pub(crate) gart_size_mib: u64,
        /// Byte 0 of the NV0080_CTRL_CMD_FB_GET_CAPS table.
        pub(crate) fb_caps0: u8,
        /// NV2080_CTRL_FB_INFO_INDEX_IS_ZERO_FB.
        pub(crate) zero_fb: bool,
    }
    pub(crate) const BUS_NONCOHERENT_DMA_FLAGS: u32 = 0xc;
    pub(crate) const BUS_GPU_GART_SIZE: u32 = 0xd;
    pub(crate) const BUS_GPU_GART_SIZE_HI: u32 = 0x32;
    /// NV2080_CTRL_BUS_INFO_NONCOHERENT_DMA_FLAGS_GPUGART (2:2).
    const GPUGART: u32 = 1 << 2;
    /// NV0080_CTRL_FB_CAPS_SUPPORT_CACHED_SYSMEM (byte 0, 0x08).
    const SUPPORT_CACHED_SYSMEM: u8 = 0x08;
    pub(crate) const FB_IS_ZERO_FB: u32 = 0x36;
    /// IGNORE_BANK_PLACEMENT | FORCE_MEM_GROWS_DOWN | FORCE_ALIGN_HOST_PAGE |
    /// MEMORY_HANDLE_PROVIDED.
    const FLAGS: u32 = 0x1 | 0x4 | 0x8 | 0x4000;
    /// | PERSISTENT_VIDMEM.
    const PERSISTENT_VIDMEM: u32 = 0x1_0000;
    /// LOCATION (26:25) PCI.
    const PCI: u32 = 1 << 25;
    /// COHERENCY (31:29) CACHED, WRITE_COMBINE, WRITE_BACK.
    const CACHED: u32 = 1 << 29;
    const WRITE_COMBINE: u32 = 2 << 29;
    const WRITE_BACK: u32 = 5 << 29;
    /// PHYSICALITY (28:27) NONCONTIGUOUS.
    const NONCONTIGUOUS: u32 = 1 << 27;
    /// ATTR2 GPU_CACHEABLE (3:2) YES / NO.
    const GPU_CACHED: u32 = 1 << 2;
    const GPU_UNCACHED: u32 = 2 << 2;
    /// NVOS32 flags of the probe allocations: MEMORY_HANDLE_PROVIDED.
    pub(crate) const PROBE_FLAGS: u32 = 0x4000;
    /// The probe's two requests: one noncontiguous page, WRITE_COMBINE first,
    /// then CACHED (the NVIDIA driver's NVOS02 flags 0x2010 / 0x1010,
    /// E 0x969d2d / E 0x969d66, as NVOS32 attributes).
    pub(crate) const PROBE_WRITE_COMBINE: u32 = PCI | NONCONTIGUOUS | WRITE_COMBINE;
    pub(crate) const PROBE_CACHED: u32 = PCI | NONCONTIGUOUS | CACHED;

    /// Decide the placement. `probe(attr)` allocates and frees one page of
    /// system memory with `attr` and says whether RM granted it.
    ///
    /// The ring is write-combined system memory when RM reports noncoherent
    /// GPUGART and, with a non-zero GART, grants a write-combined page;
    /// otherwise it is WRITE_BACK system memory (E 0x969cd2..0x969dce, then the
    /// allocator's `testb $0x1,0x2b200` at E 0xa40e10). The CACHED probe after
    /// a refused WRITE_COMBINE one does not change the outcome; it is kept so
    /// RM sees the same requests. USERD is video memory, GPU-cached only with
    /// SUPPORT_CACHED_SYSMEM (E 0xa406ee `testb $0x8,0x160`), or, on a GPU
    /// without framebuffer, system memory with the ring's request.
    pub(crate) fn placement(
        inputs: Inputs,
        mut probe: impl FnMut(u32) -> Result<bool>,
    ) -> Result<Placement> {
        let mut write_combine = inputs.noncoherent_dma_flags & GPUGART != 0;
        // GART size in bytes is MiB << 20 in 64 bits; only zero matters.
        if inputs.gart_size_mib.wrapping_shl(20) != 0 && !probe(PROBE_WRITE_COMBINE)? {
            probe(PROBE_CACHED)?;
            write_combine = false;
        }
        let ring = Request {
            flags: FLAGS,
            attr: PCI
                | if write_combine {
                    WRITE_COMBINE
                } else {
                    WRITE_BACK
                },
            attr2: GPU_UNCACHED,
        };
        let userd = if inputs.zero_fb {
            Userd::System(ring)
        } else {
            Userd::Video(Request {
                flags: FLAGS | PERSISTENT_VIDMEM,
                attr: 0,
                attr2: if inputs.fb_caps0 & SUPPORT_CACHED_SYSMEM != 0 {
                    GPU_CACHED
                } else {
                    GPU_UNCACHED
                },
            })
        };
        Ok(Placement { ring, userd })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// GB202 (615.78.08): NONCOHERENT_DMA_FLAGS 5, GART 0x20 << 32 MiB,
        /// SUPPORT_CACHED_SYSMEM, framebuffer present.
        const GB202: Inputs = Inputs {
            noncoherent_dma_flags: 5,
            gart_size_mib: 0x20 << 32,
            fb_caps0: 0x08,
            zero_fb: false,
        };
        fn run(inputs: Inputs, grants: &[bool]) -> (Placement, Vec<u32>) {
            let mut probes = Vec::new();
            let placement = placement(inputs, |attr| {
                probes.push(attr);
                Ok(grants[probes.len() - 1])
            })
            .expect("the probe does not fail");
            (placement, probes)
        }
        #[test]
        fn measured_gpu_gets_the_measured_requests() {
            let (p, probes) = run(GB202, &[true]);
            assert_eq!(probes, [PROBE_WRITE_COMBINE]);
            assert_eq!(
                (p.ring.flags, p.ring.attr, p.ring.attr2),
                (0x400d, 0x4200_0000, 0x8)
            );
            assert_eq!(
                p.userd,
                Userd::Video(Request {
                    flags: 0x1_400d,
                    attr: 0,
                    attr2: 0x4
                })
            );
        }
        #[test]
        fn userd_is_gpu_uncached_without_cached_sysmem() {
            let (p, _) = run(
                Inputs {
                    fb_caps0: 0xf7,
                    ..GB202
                },
                &[true],
            );
            assert_eq!(
                p.userd,
                Userd::Video(Request {
                    flags: 0x1_400d,
                    attr: 0,
                    attr2: 0x8
                })
            );
        }
        #[test]
        fn refused_write_combine_probe_makes_the_ring_write_back() {
            let (p, probes) = run(GB202, &[false, true]);
            assert_eq!(probes, [PROBE_WRITE_COMBINE, PROBE_CACHED]);
            assert_eq!((p.ring.attr, p.ring.attr2), (0xa200_0000, 0x8));
            let (p, _) = run(GB202, &[false, false]);
            assert_eq!(p.ring.attr, 0xa200_0000);
        }
        #[test]
        fn no_gart_skips_the_probe_and_no_gpugart_means_write_back() {
            let (p, probes) = run(
                Inputs {
                    gart_size_mib: 0,
                    ..GB202
                },
                &[],
            );
            assert!(probes.is_empty());
            assert_eq!(p.ring.attr, 0x4200_0000);
            // 2^44 MiB << 20 leaves 64 bits: the same as no GART.
            let (_, probes) = run(
                Inputs {
                    gart_size_mib: 1 << 44,
                    ..GB202
                },
                &[],
            );
            assert!(probes.is_empty());
            let (p, _) = run(
                Inputs {
                    noncoherent_dma_flags: 1,
                    ..GB202
                },
                &[true],
            );
            assert_eq!(p.ring.attr, 0xa200_0000);
        }
        #[test]
        fn zero_fb_puts_userd_in_system_memory_like_the_ring() {
            let (p, _) = run(
                Inputs {
                    zero_fb: true,
                    ..GB202
                },
                &[true],
            );
            assert_eq!(p.userd, Userd::System(p.ring));
            let (p, _) = run(
                Inputs {
                    zero_fb: true,
                    ..GB202
                },
                &[false, false],
            );
            assert_eq!(
                p.userd,
                Userd::System(Request {
                    flags: 0x400d,
                    attr: 0xa200_0000,
                    attr2: 0x8
                })
            );
        }
    }
}
const NONCONTIG_FROM: usize = 16 << 20;
/// NVOS32 attr for a compressible device-local request (nvos.h fields:
/// COMPR 13:12 = ANY, PAGE_SIZE 24:23 = BIG, PHYSICALITY 28:27 = NONCONTIGUOUS
/// from 16 MiB, else DEFAULT).
pub(crate) fn compressible_attr(size: usize) -> u32 {
    (2 << 12) | (2 << 23) | if size >= NONCONTIG_FROM { 1 << 27 } else { 0 }
}
/// RM returns the granted compression in attr COMPR (13:12).
pub(crate) fn compression_granted(attr: u32) -> bool {
    (attr >> 12) & 3 != 0
}
impl Device {
    pub(crate) fn map_usermode(self: &Rc<Self>) -> Result<Mapping> {
        self.map_usermode_class(0xc661)
    }
    /// Map the doorbell (usermode) page. Hopper+ (HOPPER_USERMODE_A 0xc661
    /// and later) is allocated with NV_HOPPER_USERMODE_A_PARAMS
    /// {bBar1Mapping = 1, bPriv = 0} (public nvos.h; usermode_api.c:62-99
    /// then backs it with the BAR1 doorbell page pBar1VF): the doorbell write
    /// then travels the same BAR1 posted-write path as the GPFIFO/GP_PUT
    /// writes that precede it, so it cannot overtake them. Public NVIDIA UVM
    /// does exactly this on Hopper+ (nv_gpu_ops.c:5646-5659), and stock 615
    /// rings its doorbell through such a BAR1 page (runtime observation:
    /// build/fence-tracing/deferred-reclaim/stock-userd/). Earlier classes
    /// only have the BAR0 page (gpu.rs kick keeps a BAR1 read-back there).
    pub(crate) fn map_usermode_class(self: &Rc<Self>, class: u32) -> Result<Mapping> {
        let handle = if class >= 0xc661 {
            if self
                .client
                .compat
                .layout_size("NV_HOPPER_USERMODE_A_PARAMS")?
                != 2
            {
                return Err(Error::Contract(
                    "NV_HOPPER_USERMODE_A_PARAMS is not {NvBool bBar1Mapping; NvBool bPriv}".into(),
                ));
            }
            let mut params = [1u8, 0u8]; // bBar1Mapping, bPriv
            self.client
                .allocate_bytes(self.subdevice, ClassId::new(class), &mut params)?
        } else {
            self.client
                .allocate(self.subdevice, ClassId::new(class), &mut ())?
        };
        Rc::new(Memory {
            device: Rc::clone(self),
            handle,
            parent: self.subdevice,
            size: 65536,
            owned: true,
            charge_bar: false,
            _mapping_owner: None,
            nvos33_flags: 0,
            compressed: false,
            pages: PhysicalPages::Small4K,
        })
        .map()
    }
}
impl Device {
    /// Borrow the channel handle: this mapping never owns/frees that handle.
    /// Caller must keep its Object alive and drop this mapping before the channel.
    /// Public580 kernel_channel.c:1340-1354 FIFO_MAPPING_ENABLE selects the full
    /// RM-owned USERD range; os.c:1103-1176 expands it to CPU pages and preserves
    /// the within-page physical offset in pLinearAddress.
    pub(crate) fn map_channel_userd(
        self: &Rc<Self>,
        channel: RmHandle,
        owner: Rc<dyn std::any::Any>,
    ) -> Result<Mapping> {
        Rc::new(Memory {
            device: Rc::clone(self),
            handle: channel,
            parent: self.handle,
            size: 512,
            owned: false,
            charge_bar: true,
            _mapping_owner: Some(owner),
            nvos33_flags: 0,
            compressed: false,
            pages: PhysicalPages::Small4K,
        })
        .map()
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        for (parent, object) in [
            (self.handle, self.subdevice),
            (self.client.root, self.handle),
        ] {
            if let Err(error) = self.client.free(parent, object) {
                diag!("RM device cleanup: {error}");
            }
        }
    }
}
/// An RM memory object (video or system memory) and the device it belongs to. Freed on
/// drop, after every CPU and GPU mapping of it is gone (they hold an `Rc`).
pub struct Memory {
    pub(crate) device: Rc<Device>,
    pub(crate) handle: RmHandle,
    parent: RmHandle,
    pub(crate) size: usize,
    owned: bool,
    charge_bar: bool,
    // USERD belongs to its channel; retain that owner through rollback/quarantine.
    _mapping_owner: Option<Rc<dyn std::any::Any>>,
    /// NVOS33 flags of a CPU mapping of owned memory (0: RM's defaults).
    nvos33_flags: u32,
    /// RM granted compression (NVOS32 attr COMPR != NONE on return) for an
    /// allocation that asked for it; only then may a VA carry a compressible kind.
    pub(crate) compressed: bool,
    /// Physical page size: with big (64 KiB) pages aligned VA ranges may map
    /// with big PTEs; a huge-page pool block maps at RM's default page size.
    pub(crate) pages: PhysicalPages,
}
impl Memory {
    /// Adopt the owned handle returned by RM object import. Import duplicates
    /// ownership; this wrapper releases it through the usual RM free path.
    pub(crate) fn from_imported(
        device: Rc<Device>,
        handle: RmHandle,
        size: usize,
        system: bool,
    ) -> Rc<Self> {
        let parent = device.handle;
        Rc::new(Self {
            device,
            handle,
            parent,
            size,
            owned: true,
            charge_bar: !system,
            _mapping_owner: None,
            nvos33_flags: 0,
            compressed: false,
            pages: PhysicalPages::Small4K,
        })
    }
    /// Map the memory into this process (`NV_ESC_RM_MAP_MEMORY` + `mmap` of the device
    /// node), charged against the device's BAR budget; idle mappings are evicted to make room.
    ///
    /// # Errors
    /// [`MemoryError`] for memory that is never CPU-mapped, an exhausted BAR budget, an
    /// earlier failed cleanup, or RM's or `mmap`'s refusal.
    pub fn map(self: &Rc<Self>) -> Result<Mapping> {
        if self.pages == PhysicalPages::Huge2M {
            return Err(MemoryError::Invalid {
                operation: "CPU-map memory",
                reason: "a GPU-internal pool block is never CPU-mapped",
            }
            .into());
        }
        loop {
            if self.device.cpu_mapping_failed() {
                return Err(MemoryError::CleanupFailed.into());
            }
            match self.map_once() {
                Ok(mapping) => return Ok(mapping),
                Err(error) if map_pressure(&error) => match self.device.cpu_maps.evict_one() {
                    Ok(true) => continue,
                    Ok(false) => return Err(error),
                    Err(cleanup) => {
                        self.device.cpu_map_failed.set(true);
                        return Err(MemoryError::EvictionFailed(Box::new(cleanup)).into());
                    }
                },
                Err(error) => return Err(error),
            }
        }
    }
    fn map_once(self: &Rc<Self>) -> Result<Mapping> {
        let device = &self.device;
        if self.size == 0 {
            return Err(MemoryError::Invalid {
                operation: "CPU-map memory",
                reason: "cannot CPU-map empty memory",
            }
            .into());
        }
        // Validate the inverse operation before acquiring the RM lease.
        device
            .client
            .compat
            .require_constant("NV_ESC_RM_UNMAP_MEMORY", 0x4f)?;
        device
            .client
            .compat
            .pack("NVOS34_PARAMETERS", &abi::UnmapMemory::default())?;
        let reservation = BarReservation::reserve(
            &device.bar_budget,
            if self.charge_bar {
                u64_from_usize(
                    platform::round_to_cpu_pages(self.size, device.client.cpu_page_size)
                        .map_err(|e| Error::Io("round BAR reservation", e))?,
                )
            } else {
                0
            },
        )?;
        let flags = if self.owned {
            self.nvos33_flags
        } else {
            1 << 17 // NVOS33 FIFO_MAPPING_ENABLE
        };
        let space = device.client.address_space(self.handle, flags)?;
        // Each mapping gets a fresh, unregistered fd (O_RDWR | O_CLOEXEC),
        // closed right after mmap: the VMA then pins only that file, never the
        // control file, so the RM client is freed in this process when its
        // control fd closes (the stock driver maps the same way,
        // libnvidia-eglcore.so.615.78.08 E 0xe75300).
        let fd = match space {
            AddressSpace::System => open("/dev/nvidiactl")?,
            AddressSpace::Video | AddressSpace::Register => {
                open(&format!("/dev/nvidia{}", device.card.minor))?
            }
        };
        let mut request = abi::MapWithFd {
            parameters: abi::MapMemory {
                client: device.client.root.get(),
                device: device.handle.get(),
                memory: self.handle.get(),
                length: u64_from_usize(self.size),
                flags,
                ..Default::default()
            },
            fd: fd.as_raw_fd(),
            padding: 0,
        };
        exchange_wire(
            device.client.control.as_raw_fd(),
            0x4e,
            device.client.compat,
            "nv_ioctl_nvos33_parameters_with_fd",
            &mut request,
            "RM map memory",
        )?;
        status("RM map memory", request.parameters.status)?;
        // RM returns an aperture/mapping cookie, NOT mmap's CPU virtual address.
        // Preserve all 64 bits (also on ILP32), and the returned mapping flags.
        // escape.c uses precisely these fields for its own mmap-context rollback.
        let mut lease = CpuMappingLease {
            resources: Some(CpuMappingResources {
                memory: Rc::clone(self),
                bar: reservation,
                unmap: abi::UnmapMemory {
                    client: request.parameters.client,
                    device: request.parameters.device,
                    memory: request.parameters.memory,
                    linear_address: request.parameters.linear_address,
                    flags: request.parameters.flags,
                    ..Default::default()
                },
            }),
            state: CpuMappingState {
                rm: true,
                vma: false,
            },
            vma: None,
        };
        let result = (|| {
            let (within_page, mapping_size) = cpu_mapping_extent(
                self.size,
                self.owned,
                request.parameters.length,
                request.parameters.linear_address,
                device.client.cpu_page_size,
            )?;
            if self.charge_bar {
                lease
                    .resources
                    .as_mut()
                    .expect("a new lease holds its resources until it is released")
                    .bar
                    .resize(u64_from_usize(mapping_size))?;
            }
            // RM associated this fd with its map. VMA ownership and RM ownership
            // are independent: both are released (NVOS34, then munmap) before the
            // BAR aperture reservation.
            // SAFETY: a fresh kernel-chosen address (NULL hint, no MAP_FIXED),
            // so no existing mapping or Rust object is replaced; fd and
            // length are the ones RM just associated with this allocation.
            // The VMA is owned by `lease` from here on, which unmaps it.
            let address = unsafe {
                platform::mmap(std::ptr::null_mut(), mapping_size, 3, 1, fd.as_raw_fd(), 0)
            };
            // MAP_FAILED is `(void *)-1` (sys/mman.h).
            if address.addr() == usize::MAX {
                return Err(Error::Io(
                    "mmap NVIDIA allocation",
                    io::Error::last_os_error(),
                ));
            }
            // Register even address zero before validating the public pointer:
            // rollback must also own that unusual but successful mmap result.
            lease.vma = Some((address.expose_provenance(), mapping_size));
            lease.state.vma = true;
            let base = NonNull::new(address.cast::<u8>()).ok_or(MemoryError::CpuMapping {
                reason: "mmap returned address zero",
            })?;
            // Lazy owned mappings survive logical map/unmap pairs. Like public
            // Mesa persistent maps, lifetime follows the allocation; idle leases
            // remain pressure-reclaimable and retain their actual BAR charge.
            // Borrowed USERD/register mappings keep their existing short lifetime.
            let retain_idle = self.owned;
            // SAFETY: `within_page` < the CPU page size <= `mapping_size`
            // (cpu_mapping_extent), so the offset stays inside this VMA.
            let start = unsafe { base.add(within_page) };
            if space == AddressSpace::Video {
                let address = u64_from_usize(start.as_ptr().addr());
                device.client.update_device_mapping(
                    request.parameters.device,
                    request.parameters.memory,
                    request.parameters.linear_address,
                    address,
                )?;
                lease
                    .resources
                    .as_mut()
                    .expect("a new lease holds its resources until it is released")
                    .unmap
                    .linear_address = address;
            }
            Ok((start.cast::<u32>(), retain_idle))
        })();
        match result {
            Ok((pointer, retain_idle)) => {
                // The VMA retains its file independently. RM keys the mapping
                // by the saved NVOS33 address (or the CPU address after the
                // update above), not by this fd.
                drop(fd);
                Ok(Mapping {
                    pointer,
                    size: self.size,
                    lease,
                    retain_idle,
                })
            }
            Err(error) => {
                if let Err(cleanup) = lease.try_unmap() {
                    // Drop will retry, then quarantine rather than free memory
                    // or refund BAR capacity while a lease may still be live.
                    device.cpu_map_failed.set(true);
                    return Err(MemoryError::RollbackFailed {
                        error: Box::new(error),
                        cleanup: Box::new(cleanup),
                    }
                    .into());
                }
                Err(error)
            }
        }
    }
}
/// RM's address space of a mappable object (ctrl0000client.h
/// NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_SYSMEM / _VIDMEM / _REGMEM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AddressSpace {
    /// Mapped through `/dev/nvidiactl`.
    System,
    /// BAR1; mapped through `/dev/nvidiaN`, then re-keyed to its CPU address.
    Video,
    /// BAR0 registers; mapped through `/dev/nvidiaN`.
    Register,
}
fn map_pressure(error: &Error) -> bool {
    match error {
        Error::Memory(MemoryError::BarPressure {
            used,
            requested,
            limit,
        }) => *used != 0 && *limit != 0 && requested <= limit,
        Error::Io("mmap NVIDIA allocation", error) => error.raw_os_error() == Some(12),
        Error::Rm {
            operation: "RM map memory",
            ..
        } => error.is_out_of_device_memory(),
        _ => false,
    }
}
impl Drop for Memory {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        if let Err(error) = self.device.client.free(self.parent, self.handle) {
            diag!("RM memory cleanup: {error}");
        }
    }
}
/// Per-device accounting includes imports and queue allocations. This is a
/// conservative process-local reservation; other clients can consume aperture,
/// so successful budgeting never suppresses an RM/mmap failure.
#[derive(Debug)]
struct BarBudget {
    used: Cell<u64>,
    limit: u64,
}
struct BarReservation {
    budget: Rc<BarBudget>,
    bytes: u64,
}
impl BarReservation {
    fn reserve(budget: &Rc<BarBudget>, bytes: u64) -> Result<Self> {
        let mut reservation = Self {
            budget: Rc::clone(budget),
            bytes: 0,
        };
        reservation.resize(bytes)?;
        Ok(reservation)
    }
    fn resize(&mut self, bytes: u64) -> Result<()> {
        let other =
            self.budget
                .used
                .get()
                .checked_sub(self.bytes)
                .ok_or(MemoryError::BarAccounting {
                    reason: "unbalanced BAR reservation",
                })?;
        let next = other.checked_add(bytes).ok_or(MemoryError::BarAccounting {
            reason: "BAR mapping budget overflow",
        })?;
        if bytes != 0 && (self.budget.limit == 0 || next > self.budget.limit) {
            return Err(MemoryError::BarPressure {
                used: other,
                requested: bytes,
                limit: self.budget.limit,
            }
            .into());
        }
        self.budget.used.set(next);
        self.bytes = bytes;
        Ok(())
    }
}
impl Drop for BarReservation {
    fn drop(&mut self) {
        self.budget.used.set(
            self.budget
                .used
                .get()
                .checked_sub(self.bytes)
                .expect("balanced BAR reservation"),
        );
    }
}

/// Production state machine shared by rollback, explicit unmap and Drop.
/// A failed step remains owned. Completed steps are never repeated on retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CpuCleanupStage {
    Vma,
    Rm,
}
#[derive(Default, Debug)]
struct CpuMappingState {
    vma: bool,
    rm: bool,
}
impl CpuMappingState {
    /// RM unmap (NVOS34) first, then munmap of the VMA: the order the NVIDIA driver releases
    /// a host-visible mapping in before the memory's RM free.
    fn cleanup(&mut self, mut release: impl FnMut(CpuCleanupStage) -> Result<()>) -> Result<()> {
        if self.rm {
            release(CpuCleanupStage::Rm)?;
            self.rm = false;
        }
        if self.vma {
            release(CpuCleanupStage::Vma)?;
            self.vma = false;
        }
        Ok(())
    }
}
impl CpuMappingState {
    fn cleanup_owned<R>(
        &mut self,
        resources: &mut Option<R>,
        mut release: impl FnMut(CpuCleanupStage, &mut R) -> Result<()>,
    ) -> Result<()> {
        let Some(owner) = resources.as_mut() else {
            return if self.vma || self.rm {
                Err(MemoryError::NotLive {
                    reason: "live CPU map lost its resource owner",
                }
                .into())
            } else {
                Ok(())
            };
        };
        self.cleanup(|step| release(step, owner))?;
        // Memory and BAR accounting are released only after BOTH operations.
        resources.take();
        Ok(())
    }
}
struct CpuMappingResources {
    memory: Rc<Memory>,
    bar: BarReservation,
    unmap: abi::UnmapMemory,
}
struct CpuMappingLease {
    resources: Option<CpuMappingResources>,
    state: CpuMappingState,
    vma: Option<(usize, usize)>,
}
impl CpuMappingLease {
    fn try_unmap(&mut self) -> Result<()> {
        let vma = self.vma;
        self.state
            .cleanup_owned(&mut self.resources, |stage, resources| match stage {
                CpuCleanupStage::Vma => {
                    let (base, size) = vma.expect("live VMA has an extent");
                    // SAFETY: (base, size) is the VMA this lease mapped and still
                    // owns (state.vma); every Mapping borrow of it has ended,
                    // since the lease is only released by its owner.
                    if unsafe {
                        platform::munmap(std::ptr::with_exposed_provenance_mut(base), size)
                    } != 0
                    {
                        Err(Error::Io(
                            "munmap NVIDIA allocation",
                            io::Error::last_os_error(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                CpuCleanupStage::Rm => {
                    let client = &resources.memory.device.client;
                    let mut request = resources.unmap;
                    exchange_wire(
                        client.control.as_raw_fd(),
                        0x4f,
                        client.compat,
                        "NVOS34_PARAMETERS",
                        &mut request,
                        "RM unmap memory",
                    )?;
                    status("RM unmap memory", request.status)
                }
            })?;
        self.vma = None;
        Ok(())
    }
}
impl Drop for CpuMappingLease {
    fn drop(&mut self) {
        if let Err(error) = self.try_unmap() {
            diag!(
                "NVIDIA CPU mapping cleanup failed; quarantining allocation and BAR reservation until process exit: {error}"
            );
            // Keep the client, device, allocation and budget alive. A destructor
            // cannot report failure or safely free/reuse these live resources.
            if let Some(resources) = self.resources.take() {
                std::mem::forget(resources);
            }
        }
    }
}
/// The CPU round-trip pattern of word `i` (a multiplicative hash of the
/// index; the index's low 32 bits are the hash input by design).
fn pattern_word(i: usize) -> u32 {
    low_u32(u64_from_usize(i)).wrapping_mul(0x9e37_79b9) ^ 0xa5a5_5a5a
}
/// An RM parameter block's byte size as the u32 `paramsSize`/`size` field of
/// NVOS21/NVOS54 (nvos.h) carries it.
fn wire_size(len: usize) -> Result<u32> {
    u32::try_from(len)
        .map_err(|error| Error::Contract(format!("RM parameter block of {len} bytes: {error}")))
}
fn cpu_mapping_extent(
    size: usize,
    owned: bool,
    returned_length: u64,
    cookie: u64,
    page: usize,
) -> Result<(usize, usize)> {
    if page == 0 || !page.is_power_of_two() {
        return Err(MemoryError::CpuMapping {
            reason: "invalid CPU page size",
        }
        .into());
    }
    // The low bits of the cookie below the page size: less than `page`, so a usize.
    let within_page = usize::try_from(cookie & (u64_from_usize(page) - 1))
        .expect("an offset below a usize page size fits usize");
    // Owned allocations must also validate RM's returned length. USERD may
    // expose a larger range than the logical register window.
    if returned_length < u64_from_usize(size) {
        return Err(MemoryError::CpuMapping {
            reason: "RM returned an undersized CPU mapping",
        }
        .into());
    }
    let exposed = if owned {
        size
    } else {
        usize::try_from(returned_length).map_err(|_unrepresentable| MemoryError::UserdLength {
            length: returned_length,
        })?
    };
    if size == 0 || !within_page.is_multiple_of(4) {
        return Err(MemoryError::CpuMapping {
            reason: "empty or unaligned RM CPU mapping",
        }
        .into());
    }
    let extent = exposed
        .checked_add(within_page)
        .ok_or(MemoryError::CpuMapping {
            reason: "CPU mapping extent overflow",
        })?;
    let rounded = platform::round_to_cpu_pages(extent, page)
        .map_err(|e| Error::Io("round mmap extent", e))?;
    if isize::try_from(rounded).is_err() {
        return Err(MemoryError::CpuMapping {
            reason: "CPU mapping exceeds Rust pointer offset range",
        }
        .into());
    }
    Ok((within_page, rounded))
}
/// A CPU mapping of a [`Memory`]. The GPU and other processes may write the memory at any
/// time, so every access is a bounds-checked volatile read or write, or atomic through an
/// `&AtomicU64` (RUST.md §3). Unmapped on drop.
pub struct Mapping {
    pointer: NonNull<u32>,
    size: usize,
    lease: CpuMappingLease,
    retain_idle: bool,
}
impl crate::cpu_map_cache::Lease for Mapping {
    fn retain_idle(&self) -> bool {
        self.retain_idle
    }
    fn release(&mut self) -> Result<()> {
        self.try_unmap()
    }
}
impl Mapping {
    pub(crate) fn try_unmap(&mut self) -> Result<()> {
        let result = self.lease.try_unmap();
        if result.is_err() {
            // Partial VMA/RM teardown makes every cached-pointer entry point
            // reject this Device, including explicit free outside map pressure.
            if let Some(resources) = &self.lease.resources {
                resources.memory.device.cpu_map_failed.set(true);
            }
        }
        result
    }
    /// The error of an unaligned or out-of-bounds access of `len` bytes at `offset`.
    fn out_of_bounds(&self, access: &'static str, offset: usize, len: usize) -> Error {
        MemoryError::OutOfBounds {
            access,
            offset,
            len,
            size: self.size,
        }
        .into()
    }
    fn live(&self) -> Result<()> {
        if self.lease.state.vma {
            Ok(())
        } else {
            Err(MemoryError::NotLive {
                reason: "CPU mapping is no longer live",
            }
            .into())
        }
    }
    /// An external Vulkan caller owns synchronization and dereference validity.
    /// This integer conveys an address only; Rust references are never created.
    pub(crate) fn cpu_address(&self) -> usize {
        assert!(self.lease.state.vma, "CPU mapping is no longer live");
        self.pointer.as_ptr().expose_provenance()
    }

    pub(crate) fn write_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        self.live()?;
        if !offset.is_multiple_of(4) || offset.checked_add(4).is_none_or(|end| end > self.size) {
            return Err(self.out_of_bounds("mapped write", offset, 4));
        }
        // SAFETY: offset + 4 <= size (checked above) keeps the word inside
        // the live mapping (live() above).
        let word = unsafe { self.pointer.as_ptr().add(offset / 4) };
        // SAFETY: in bounds and 4-byte aligned (checked above; the base is
        // 4-aligned, cpu_mapping_extent). Volatile: the GPU or another
        // process may read or write this word concurrently.
        unsafe { word.write_volatile(value) };
        Ok(())
    }
    /// Bulk copy into the mapping (bounds-checked, byte granular).
    pub(crate) fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        self.live()?;
        if offset
            .checked_add(data.len())
            .is_none_or(|end| end > self.size)
        {
            return Err(self.out_of_bounds("mapped bulk write", offset, data.len()));
        }
        // SAFETY: offset + len <= size (checked above), inside the live
        // mapping.
        let target = unsafe { self.pointer.as_ptr().cast::<u8>().add(offset) };
        // SAFETY: both ranges are valid for `len` bytes and cannot overlap
        // (`data` is a Rust slice, the target device memory). Plain copy, as
        // every caller writes ranges no GPU work in flight touches.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), target, data.len()) };
        Ok(())
    }
    /// Bulk copy out of the mapping (bounds-checked, byte granular).
    pub(crate) fn read_bytes(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        self.live()?;
        if offset
            .checked_add(out.len())
            .is_none_or(|end| end > self.size)
        {
            return Err(self.out_of_bounds("mapped bulk read", offset, out.len()));
        }
        // SAFETY: offset + len <= size (checked above), inside the live
        // mapping.
        let source = unsafe { self.pointer.as_ptr().cast_const().cast::<u8>().add(offset) };
        // SAFETY: both ranges are valid for `len` bytes and cannot overlap.
        // Plain copy, as every caller reads ranges whose GPU writers have
        // completed.
        unsafe { std::ptr::copy_nonoverlapping(source, out.as_mut_ptr(), out.len()) };
        Ok(())
    }
    pub(crate) fn read_u32(&self, offset: usize) -> Result<u32> {
        self.live()?;
        if !offset.is_multiple_of(4) || offset.checked_add(4).is_none_or(|end| end > self.size) {
            return Err(self.out_of_bounds("mapped read", offset, 4));
        }
        // SAFETY: offset + 4 <= size (checked above), inside the live mapping.
        let word = unsafe { self.pointer.as_ptr().add(offset / 4) };
        // SAFETY: in bounds and 4-byte aligned (checked above). Volatile: the
        // GPU or another process may write this word concurrently.
        Ok(unsafe { word.read_volatile() })
    }
    /// A 64-bit word of the mapping that RM, the kernel or another process
    /// may update concurrently, for atomic access only. The word must never
    /// be accessed non-atomically, nor at another width, while the
    /// reference lives (the semaphore surface's maxSubmitted word is such a
    /// word: RM and userspace both use aligned 64-bit atomics on it).
    pub(crate) fn atomic_u64(&self, offset: usize) -> Result<&std::sync::atomic::AtomicU64> {
        let address = self
            .cpu_address()
            .checked_add(offset)
            .ok_or_else(|| self.out_of_bounds("cached semaphore atomic", offset, 8))?;
        if address % align_of::<std::sync::atomic::AtomicU64>() != 0
            || offset.checked_add(8).is_none_or(|end| end > self.size)
        {
            return Err(self.out_of_bounds("cached semaphore atomic", offset, 8));
        }
        // SAFETY: offset + 8 <= size (checked above), inside the live mapping.
        let word = unsafe { self.pointer.as_ptr().cast::<u8>().add(offset) };
        // SAFETY: in bounds and 8-byte aligned (checked above); the mapping
        // stays mapped while `self` is borrowed (Drop unmaps). AtomicU64 has
        // interior mutability, so concurrent writers are allowed as long as
        // every access is atomic (this accessor's contract).
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "the offset is checked 8-byte aligned above and the mapping is page aligned"
        )]
        let word = word.cast::<std::sync::atomic::AtomicU64>();
        // SAFETY: as above: in bounds, aligned, atomic-only access.
        Ok(unsafe { &*word })
    }

    /// CPU-only allocation smoke check; deliberately does not claim GPU execution.
    /// Returns the number of words verified.
    ///
    /// # Errors
    /// The mapping is no longer live, or a word reads back differently.
    pub fn check_pattern(&mut self) -> Result<usize> {
        self.live()?;
        let count = self.size / 4;
        for i in 0..count {
            // SAFETY: i < size / 4, inside the live mapping.
            let word = unsafe { self.pointer.as_ptr().add(i) };
            // SAFETY: in bounds, 4-byte aligned; no GPU command references
            // this smoke-test allocation.
            unsafe { word.write_volatile(pattern_word(i)) };
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        for i in 0..count {
            // SAFETY: i < size / 4, inside the live mapping.
            let word = unsafe { self.pointer.as_ptr().add(i) };
            // SAFETY: in bounds, aligned; no GPU commands reference it.
            let actual = unsafe { word.read_volatile() };
            let expected = pattern_word(i);
            if actual != expected {
                return Err(MemoryError::PatternMismatch {
                    word: i,
                    actual,
                    expected,
                }
                .into());
            }
        }
        Ok(count)
    }
}

/// Normalized PCI BDF `dddd:bb:dd.f` of an RM card.
pub(crate) fn card_bdf(card: &abi::Card) -> [u32; 4] {
    [
        card.pci.domain,
        u32::from(card.pci.bus),
        u32::from(card.pci.slot),
        u32::from(card.pci.function),
    ]
}
pub(crate) fn bdf_string(p: [u32; 4]) -> String {
    format!("{:04x}:{:02x}:{:02x}.{:x}", p[0], p[1], p[2], p[3])
}

/// Selection policy of [`Client::select_card`], with the per-GPU node probe
/// injected (unit-tested without hardware).
pub(crate) fn select_accessible_card(
    cards: &[abi::Card],
    want: Option<[u32; 4]>,
    probe: impl Fn(u32) -> io::Result<()>,
) -> Result<abi::Card> {
    use std::io::ErrorKind::{NotFound, PermissionDenied};
    let candidates: Vec<&abi::Card> = cards
        .iter()
        .filter(|c| c.valid != 0 && c.pci.vendor == 0x10de)
        .collect();
    let listed = || {
        candidates
            .iter()
            .map(|c| format!("{} (minor {})", bdf_string(card_bdf(c)), c.minor))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(p) = want {
        let card = candidates
            .iter()
            .find(|c| card_bdf(c) == p)
            .ok_or_else(|| {
                Error::Contract(format!(
                    "requested NVIDIA GPU {} is not reported by RM (RM reports: [{}])",
                    bdf_string(p),
                    listed()
                ))
            })?;
        return probe(card.minor).map(|()| **card).map_err(|e| {
            Error::Contract(format!(
                "requested NVIDIA GPU {}: /dev/nvidia{} is not accessible to this process: {e}",
                bdf_string(p),
                card.minor
            ))
        });
    }
    let mut skipped = Vec::new();
    for card in &candidates {
        match probe(card.minor) {
            Ok(()) => return Ok(**card),
            Err(e) if matches!(e.kind(), PermissionDenied | NotFound) => skipped.push(format!(
                "{} /dev/nvidia{}: {e}",
                bdf_string(card_bdf(card)),
                card.minor
            )),
            Err(e) => {
                return Err(Error::Contract(format!(
                    "NVIDIA GPU {}: open /dev/nvidia{}: {e}",
                    bdf_string(card_bdf(card)),
                    card.minor
                )));
            }
        }
    }
    Err(Error::Contract(if candidates.is_empty() {
        "no NVIDIA GPU reported by RM".into()
    } else {
        format!(
            "no NVIDIA GPU accessible to this process (RM reports {}; skipped: {})",
            candidates.len(),
            skipped.join("; ")
        )
    }))
}

#[cfg(test)]
mod select_tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};
    fn card(bus: u8, minor: u32) -> abi::Card {
        abi::Card {
            valid: 1,
            pci: abi::Pci {
                bus,
                vendor: 0x10de,
                device: 0x2571,
                ..Default::default()
            },
            minor,
            ..Default::default()
        }
    }
    /// Multi-GPU container (the 20261015-nvdec-xgen Turing/Ada/Blackwell
    /// rentals): RM lists every host GPU, only one /dev/nvidiaN is granted.
    fn container(minor: u32) -> io::Result<()> {
        match minor {
            0 => Err(IoError::from(ErrorKind::PermissionDenied)),
            1 => Err(IoError::from(ErrorKind::NotFound)),
            _ => Ok(()),
        }
    }
    #[test]
    fn skips_inaccessible_nodes_to_the_granted_gpu() {
        let cards = [card(0x04, 0), card(0x2b, 1), card(0x41, 2)];
        assert_eq!(
            select_accessible_card(&cards, None, container)
                .unwrap()
                .minor,
            2
        );
    }
    #[test]
    fn requested_bdf_is_honoured_and_never_substituted() {
        let cards = [card(0x04, 0), card(0x2b, 1), card(0x41, 2)];
        let got = select_accessible_card(&cards, Some([0, 0x41, 0, 0]), container).unwrap();
        assert_eq!(got.minor, 2);
        let e = select_accessible_card(&cards, Some([0, 0x04, 0, 0]), container)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("0000:04:00.0") && e.contains("/dev/nvidia0 is not accessible"),
            "{e}"
        );
        let e = select_accessible_card(&cards, Some([0, 0x99, 0, 0]), container)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("0000:99:00.0 is not reported by RM")
                && e.contains("0000:2b:00.0 (minor 1)"),
            "{e}"
        );
    }
    #[test]
    fn nothing_accessible_or_unexpected_errors_fail_loudly() {
        let cards = [card(0x04, 0), card(0x2b, 1)];
        let e = select_accessible_card(&cards, None, container)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("no NVIDIA GPU accessible")
                && e.contains("/dev/nvidia0")
                && e.contains("/dev/nvidia1"),
            "{e}"
        );
        let busy = |_| Err(IoError::from(ErrorKind::ResourceBusy));
        let e = select_accessible_card(&cards, None, busy)
            .unwrap_err()
            .to_string();
        assert!(e.contains("open /dev/nvidia0"), "{e}");
        let e = select_accessible_card(&[], None, container)
            .unwrap_err()
            .to_string();
        assert_eq!(e, "no NVIDIA GPU reported by RM");
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn mapping_pressure_never_retries_contract_permission_or_cleanup_failures() {
        assert!(map_pressure(&Error::Memory(MemoryError::BarPressure {
            used: 4096,
            requested: 4096,
            limit: 4096
        })));
        assert!(!map_pressure(&Error::Memory(MemoryError::BarPressure {
            used: 4096,
            requested: 8192,
            limit: 4096
        })));
        assert!(!map_pressure(&Error::Memory(MemoryError::BarPressure {
            used: 4096,
            requested: 4096,
            limit: 0
        })));
        for status in [0x51, 0x1a] {
            assert!(map_pressure(&Error::Rm {
                operation: "RM map memory",
                status
            }));
            assert!(!map_pressure(&Error::Rm {
                operation: "RM unmap memory",
                status
            }));
        }
        for status in [0x0f, 0x1b, 0x1e, 0x1f, 0x56] {
            assert!(!map_pressure(&Error::Rm {
                operation: "RM map memory",
                status
            }));
        }
        assert!(map_pressure(&Error::Io(
            "mmap NVIDIA allocation",
            io::Error::from_raw_os_error(12)
        )));
        for errno in [1, 9, 13, 22, 24] {
            assert!(!map_pressure(&Error::Io(
                "mmap NVIDIA allocation",
                io::Error::from_raw_os_error(errno)
            )));
        }
        assert!(!map_pressure(&Error::Io(
            "open NVIDIA device",
            io::Error::from_raw_os_error(12)
        )));
        assert!(!map_pressure(&Error::DeviceLost("rollback failed".into())));
        assert!(!map_pressure(&MemoryError::CleanupFailed.into()));
        assert!(!map_pressure(&Error::Contract("invalid extent".into())));
    }
    #[test]
    fn cpu_cleanup_faults_preserve_stage_ownership_and_retry_once() {
        for failure in [CpuCleanupStage::Rm, CpuCleanupStage::Vma] {
            let mut state = CpuMappingState {
                vma: true,
                rm: true,
            };
            let mut calls = Vec::new();
            assert!(
                state
                    .cleanup(|step| {
                        calls.push(step);
                        if step == failure {
                            Err(Error::Contract("injected cleanup failure".into()))
                        } else {
                            Ok(())
                        }
                    })
                    .is_err()
            );
            assert!(
                state.vma,
                "the VMA remains mapped until the RM unmap and its own release succeeded"
            );
            assert_eq!(state.rm, failure == CpuCleanupStage::Rm);
            assert_eq!(
                calls,
                if failure == CpuCleanupStage::Rm {
                    vec![CpuCleanupStage::Rm]
                } else {
                    vec![CpuCleanupStage::Rm, CpuCleanupStage::Vma]
                }
            );
            calls.clear();
            state
                .cleanup(|step| {
                    calls.push(step);
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                calls,
                if failure == CpuCleanupStage::Rm {
                    vec![CpuCleanupStage::Rm, CpuCleanupStage::Vma]
                } else {
                    vec![CpuCleanupStage::Vma]
                }
            );
            assert!(!state.vma && !state.rm);
            state
                .cleanup(|_| panic!("completed cleanup must be idempotent"))
                .unwrap();
        }
    }
    #[test]
    fn post_rm_pre_mmap_rollback_only_releases_rm() {
        // Covers validation, budget expansion, and mmap failures after RM map.
        let mut state = CpuMappingState {
            vma: false,
            rm: true,
        };
        let mut calls = Vec::new();
        state
            .cleanup(|step| {
                calls.push(step);
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, [CpuCleanupStage::Rm]);
        let mut no_map = CpuMappingState::default();
        no_map
            .cleanup(|_| panic!("failed RM map owns no lease"))
            .unwrap();
    }
    #[test]
    fn cpu_mapping_extent_validates_owned_and_userd_without_truncating_cookie() {
        // The low bits come from the aperture cookie, not the host VA. High
        // physical bits are intentionally larger than ILP32's address range.
        assert_eq!(
            cpu_mapping_extent(512, false, 4096, 0xfedc_ba98_7654_0004, 4096).unwrap(),
            (4, 8192)
        );
        assert_eq!(
            cpu_mapping_extent(4096, true, 4096, u64::MAX - 4095, 4096).unwrap(),
            (0, 4096)
        );
        assert!(cpu_mapping_extent(4096, true, 4095, 0, 4096).is_err());
        assert!(cpu_mapping_extent(512, false, 511, 0, 4096).is_err());
        assert!(cpu_mapping_extent(4, true, 4, 2, 4096).is_err());
        assert!(cpu_mapping_extent(0, true, 0, 0, 4096).is_err());
        assert!(cpu_mapping_extent(4, true, 4, 0, 3).is_err());
        assert!(cpu_mapping_extent(512, false, u64::MAX, 4, 4096).is_err());
        let max = isize::MAX.unsigned_abs();
        assert!(cpu_mapping_extent(max, true, u64_from_usize(max), 4, 4096).is_err());
    }
    #[test]
    fn failed_cleanup_retains_memory_owner_and_bar_until_successful_retry() {
        struct Owner {
            _bar: BarReservation,
            drops: Rc<Cell<u32>>,
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                self.drops.set(self.drops.get() + 1);
            }
        }
        for failure in [CpuCleanupStage::Vma, CpuCleanupStage::Rm] {
            let budget = Rc::new(BarBudget {
                used: Cell::new(0),
                limit: 4096,
            });
            let drops = Rc::new(Cell::new(0));
            let mut resources = Some(Owner {
                _bar: BarReservation::reserve(&budget, 4096).unwrap(),
                drops: Rc::clone(&drops),
            });
            let mut state = CpuMappingState {
                vma: true,
                rm: true,
            };
            assert!(
                state
                    .cleanup_owned(&mut resources, |step, _| {
                        if step == failure {
                            Err(Error::Contract("injected".into()))
                        } else {
                            Ok(())
                        }
                    })
                    .is_err()
            );
            assert!(resources.is_some());
            assert_eq!(drops.get(), 0);
            assert_eq!(budget.used.get(), 4096);
            assert!(BarReservation::reserve(&budget, 1).is_err());
            state.cleanup_owned(&mut resources, |_, _| Ok(())).unwrap();
            assert!(resources.is_none());
            assert_eq!(drops.get(), 1);
            assert_eq!(budget.used.get(), 0);
            state
                .cleanup_owned(&mut resources, |_, _| panic!("double cleanup"))
                .unwrap();
            assert_eq!(drops.get(), 1);
        }
    }
    #[test]
    fn sequential_small_bar_maps_release_kernel_and_process_capacity() {
        let budget = Rc::new(BarBudget {
            used: Cell::new(0),
            limit: 256 << 20,
        });
        let kernel_used = Cell::new(0u64);
        for _ in 0..40 {
            let reservation = BarReservation::reserve(&budget, 32 << 20).unwrap();
            kernel_used.set(kernel_used.get() + (32 << 20));
            assert!(kernel_used.get() <= budget.limit);
            let mut state = CpuMappingState {
                vma: true,
                rm: true,
            };
            state
                .cleanup(|step| {
                    if step == CpuCleanupStage::Rm {
                        kernel_used.set(kernel_used.get() - (32 << 20));
                    }
                    Ok(())
                })
                .unwrap();
            drop(reservation);
            assert_eq!(kernel_used.get(), 0);
            assert_eq!(budget.used.get(), 0);
        }
    }
    #[test]
    #[ignore = "needs an accessible NVIDIA GPU"]
    fn channel_placement_is_queried_at_device_creation_on_selected_nvidia() {
        let client = Client::open().unwrap();
        let card = client
            .cards()
            .unwrap()
            .into_iter()
            .find(|card| card.pci.vendor == 0x10de)
            .expect("no NVIDIA card");
        let device = client.device(card).unwrap();
        let placement = device.channel_placement().unwrap();
        eprintln!("device {:04x}: {placement:?}", card.pci.device);
        assert!(matches!(placement.ring.attr, 0x4200_0000 | 0xa200_0000));
        let ring = device.allocate_gpfifo_ring(8192).unwrap();
        let userd = device.allocate_userd().unwrap();
        ring.map().unwrap().write_u32(0, 1).unwrap();
        let mut map = userd.map().unwrap();
        map.write_u32(0x8c, 7).unwrap();
        assert_eq!(map.read_u32(0x8c).unwrap(), 7);
    }
    #[test]
    #[ignore = "needs an accessible NVIDIA GPU"]
    fn block_linear_memory_records_the_generic_kind_on_selected_nvidia() {
        let client = Client::open().unwrap();
        let card = client
            .cards()
            .unwrap()
            .into_iter()
            .find(|card| card.pci.vendor == 0x10de)
            .expect("no NVIDIA card");
        let device = client.device(card).unwrap();
        let kind = device.generic_kind().unwrap();
        for system in [false, true] {
            // allocate_block_linear refuses anything RM does not grant exactly
            // (kind, FORMAT BLOCK_LINEAR, COMPR NONE).
            let memory = device
                .allocate_block_linear(0x5a_0000, 1 << 16, system, kind)
                .unwrap();
            assert_eq!(memory.size, 0x5a_0000);
        }
        let other = if kind == 0x06 { 0xfe } else { 0x06 };
        assert!(matches!(
            device.allocate_block_linear(1 << 16, 1 << 16, false, other),
            Err(Error::Contract(_))
        ));
    }
    #[test]
    #[ignore = "requires a selected NVIDIA RM device; bounded CPU mapping only, no GPU commands"]
    fn repeated_cpu_maps_release_rm_on_selected_nvidia() {
        let client = Client::open().unwrap();
        let mut failures = Vec::new();
        let mut selected = None;
        for card in client
            .cards()
            .unwrap()
            .into_iter()
            .filter(|card| card.pci.vendor == 0x10de)
        {
            match client.device(card) {
                Ok(device) => {
                    selected = Some((card, device));
                    break;
                }
                Err(error) => {
                    let failure = format!(
                        "NVIDIA {:04x}:{:02x}:{:02x}.{} minor {}: {error}",
                        card.pci.domain, card.pci.bus, card.pci.slot, card.pci.function, card.minor
                    );
                    eprintln!("unavailable mapping-test candidate: {failure}");
                    failures.push(failure);
                }
            }
        }
        let (card, device) = selected.unwrap_or_else(|| {
            panic!(
                "no openable NVIDIA RM device; candidate errors: {}",
                failures.join("; ")
            )
        });
        let memory = device.allocate_vram(32 << 20).unwrap();
        for index in 0..40u32 {
            let mut map = memory.map().unwrap();
            map.write_u32(0, index).unwrap();
            map.write_u32(memory.size - 4, !index).unwrap();
            assert_eq!(map.read_u32(0).unwrap(), index);
            assert_eq!(map.read_u32(memory.size - 4).unwrap(), !index);
            if index % 2 == 0 {
                map.try_unmap().unwrap();
                assert!(map.read_u32(0).is_err());
                assert!(map.write_u32(0, 0).is_err());
                map.try_unmap().unwrap();
            }
            drop(map);
            assert_eq!(device.bar_budget.used.get(), 0);
        }
        eprintln!(
            "40 x 32MiB CPU mappings released; driver {}; device {:04x}; measured BAR {}",
            client.version, card.pci.device, card.framebuffer_size
        );
    }
    #[test]
    #[ignore = "selected NVIDIA RM CPU-map pressure fixture; isolated FD/VMA accounting, no GPU commands"]
    fn idle_cpu_map_pressure_reclaims_real_vram_on_selected_nvidia() {
        fn fds() -> usize {
            std::fs::read_dir("/proc/self/fd").unwrap().count()
        }
        fn vmas() -> usize {
            std::fs::read_to_string("/proc/self/maps")
                .unwrap()
                .lines()
                .filter(|line| line.contains("/dev/nvidia"))
                .count()
        }
        let initial_fds = fds();
        let initial_vmas = vmas();
        let client = Client::open().unwrap();
        let mut device = client
            .cards()
            .unwrap()
            .into_iter()
            .filter(|card| card.pci.vendor == 0x10de)
            .find_map(|card| match client.device(card) {
                Ok(device) => Some(device),
                Err(error) => {
                    eprintln!("unavailable NVIDIA pressure-test candidate: {error}");
                    None
                }
            })
            .expect("no openable NVIDIA RM device");
        const SIZE: usize = 1 << 20;
        // Test-only local authority: no production override, no global-BAR
        // pressure claim, no modification to the physical device or RM policy.
        assert!(device.card.framebuffer_size >= u64_from_usize(2 * SIZE));
        Rc::get_mut(&mut device)
            .expect("new Device must have one owner")
            .bar_budget = Rc::new(BarBudget {
            used: Cell::new(0),
            limit: u64_from_usize(2 * SIZE),
        });
        let memories: Vec<_> = (0..4)
            .map(|_| device.allocate_vram(SIZE).unwrap())
            .collect();
        assert!(memories.iter().all(|memory| memory.size == SIZE));
        let slots: Vec<_> = (0..4).map(|_| device.cpu_mapping_slot()).collect();
        let lease_fds = fds();
        let lease_vmas = vmas();
        let maps = Cell::new(0);
        let map = |index: usize| {
            slots[index]
                .acquire(
                    || {
                        maps.set(maps.get() + 1);
                        memories[index].map()
                    },
                    Mapping::cpu_address,
                )
                .unwrap()
        };
        let write = |address: usize, value: u32| {
            // SAFETY: corresponding Slot has a live logical reference; both
            // words are within its checked 1MiB mapping, aligned and CPU-owned.
            unsafe {
                std::ptr::with_exposed_provenance_mut::<u32>(address).write_volatile(value);
                std::ptr::with_exposed_provenance_mut::<u32>(address + SIZE - 4)
                    .write_volatile(!value);
            }
            platform::device_full_barrier();
        };
        let check = |address: usize, value: u32| {
            platform::device_full_barrier();
            unsafe {
                assert_eq!(
                    std::ptr::with_exposed_provenance::<u32>(address).read_volatile(),
                    value
                );
                assert_eq!(
                    std::ptr::with_exposed_provenance::<u32>(address + SIZE - 4).read_volatile(),
                    !value
                );
            }
        };
        let active = map(0);
        write(active, 0x1357);
        let idle = map(1);
        write(idle, 0x2468);
        slots[1].release().unwrap();
        // Allocation0 stays live; successive maps must evict only idle leases.
        for index in [2, 3, 1, 2, 3, 1] {
            let address = map(index);
            if index == 1 {
                check(address, 0x2468);
            } else {
                write(address, u32::try_from(index).unwrap());
            }
            check(active, 0x1357);
            assert_eq!(device.bar_budget.used.get(), u64_from_usize(2 * SIZE));
            assert_eq!(slots[0].refs(), 1);
            slots[index].release().unwrap();
        }
        assert_eq!(maps.get(), 8, "pressure must force each idle remap");
        // The final idle lease must be retained and reused without another RM map.
        let address = map(1);
        check(address, 0x2468);
        assert_eq!(maps.get(), 8);
        slots[1].release().unwrap();
        slots[0].release().unwrap();
        for slot in &slots {
            slot.evict().unwrap();
        }
        assert_eq!(device.bar_budget.used.get(), 0);
        assert!(!device.cpu_maps.evict_one().unwrap());
        assert_eq!(fds(), lease_fds);
        assert_eq!(vmas(), lease_vmas);
        drop(slots);
        drop(memories);
        drop(device);
        drop(client);
        assert_eq!(fds(), initial_fds);
        assert_eq!(vmas(), initial_vmas);
        eprintln!(
            "local2MiB BAR budget:8 real maps across4 VRAM allocations, active lease protected, idle data/reuse and FD/VMA cleanup verified; no global-pressure claim"
        );
    }
    #[test]
    fn runtime_version_keeps_exact_two_or_three_components() {
        assert_eq!(
            parse_runtime_version(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build
GCC version: gcc 15.2.1"
            )
            .unwrap(),
            "615.71.09"
        );
        assert_eq!(
            parse_runtime_version("NVRM version: NVIDIA UNIX x86_64 Kernel Module  595.80  date")
                .unwrap(),
            "595.80"
        );
        assert!(parse_runtime_version("GCC version: gcc 15.2.1").is_err());
    }
    #[test]
    fn reservations_resize_and_release_without_a_device() {
        let budget = Rc::new(BarBudget {
            used: Cell::new(0),
            limit: 16384,
        });
        let mut first = BarReservation::reserve(&budget, 4096).unwrap();
        let second = BarReservation::reserve(&budget, 4096).unwrap();
        first.resize(8192).unwrap();
        assert_eq!(budget.used.get(), 12288);
        assert!(first.resize(16384).is_err());
        assert_eq!(budget.used.get(), 12288);
        assert!(BarReservation::reserve(&budget, 8192).is_err());
        assert_eq!(budget.used.get(), 12288);
        drop(second);
        assert_eq!(budget.used.get(), 8192);
        first.resize(4096).unwrap();
        assert_eq!(budget.used.get(), 4096);
        drop(first);
        assert_eq!(budget.used.get(), 0);
    }
    #[test]
    fn zero_unknown_aperture_and_overflow_are_explicit() {
        let unknown = Rc::new(BarBudget {
            used: Cell::new(0),
            limit: 0,
        });
        assert!(BarReservation::reserve(&unknown, 1).is_err());
        drop(BarReservation::reserve(&unknown, 0).unwrap());
        assert_eq!(unknown.used.get(), 0);
        let huge = Rc::new(BarBudget {
            used: Cell::new(u64::MAX),
            limit: u64::MAX,
        });
        assert!(BarReservation::reserve(&huge, 1).is_err());
        assert_eq!(huge.used.get(), u64::MAX);
    }
}

#[cfg(test)]
mod pool_block_tests {
    use super::*;
    fn field(v: u32, hi: u32, lo: u32) -> u32 {
        (v >> lo) & (u32::MAX >> (31 - (hi - lo)))
    }
    #[test]
    #[ignore = "local NVIDIA GPU: one GPU-internal pool block; RM grants the requested NVOS32 attributes"]
    fn pool_block_has_the_requested_attributes() {
        let client = Client::open().unwrap();
        let card = client.cards().unwrap().into_iter().next().unwrap();
        let device = client.device(card).unwrap();
        let (memory, grant) = device.allocate_pool_block_granted().unwrap();
        assert_eq!(grant.size, u64_from_usize(POOL_BLOCK_SIZE));
        assert_eq!(
            field(grant.attr, 24, 23),
            3,
            "PAGE_SIZE HUGE: {:#010x}",
            grant.attr
        );
        assert_eq!(
            field(grant.attr, 26, 25),
            0,
            "LOCATION VIDMEM: {:#010x}",
            grant.attr
        );
        assert_eq!(
            field(grant.attr, 28, 27),
            2,
            "PHYSICALITY CONTIGUOUS: {:#010x}",
            grant.attr
        );
        assert_eq!(
            field(grant.attr, 13, 12),
            0,
            "COMPR NONE: {:#010x}",
            grant.attr
        );
        assert_eq!(
            field(grant.attr, 31, 29),
            0,
            "COHERENCY UNCACHED: {:#010x}",
            grant.attr
        );
        assert_eq!(
            field(grant.attr2, 3, 2),
            1,
            "GPU_CACHEABLE YES: {:#010x}",
            grant.attr2
        );
        assert_eq!(
            field(grant.attr2, 21, 20),
            1,
            "PAGE_SIZE_HUGE 2MB: {:#010x}",
            grant.attr2
        );
        assert_eq!(
            grant.format,
            device.generic_kind().unwrap(),
            "the generic kind"
        );
        eprintln!(
            "pool block granted attr {:#010x} attr2 {:#010x} format {:#x} size {:#x}",
            grant.attr, grant.attr2, grant.format, grant.size
        );
        assert!(memory.map().is_err(), "never CPU-mapped");
        let space = crate::gpu::AddressSpace::new(&device).unwrap();
        let gpu = space.map(&memory).unwrap();
        assert_eq!(
            gpu.address().get() % u64_from_usize(POOL_BLOCK_SIZE),
            0,
            "mapped at a 2 MiB VA"
        );
    }
}
