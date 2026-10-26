// SPDX-License-Identifier: MIT
//! Native RM-managed GPU address space and Blackwell copy queue.
//! Uses public RM/GPFIFO/CE definitions, not a proprietary userspace library.
//! An explicit FERMI_VASPACE_A is passed to NV01_MEMORY_VIRTUAL and the channel.
//! This path does not require UVM-owned page tables or fixed CPU virtual addresses.
use crate::diag::diag;
use crate::{
    abi,
    rm::{self, Device, Error, Memory, Result},
    slots::{Slot, SlotPool},
};
use std::{
    collections::VecDeque,
    os::fd::OwnedFd,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

pub(crate) struct Object {
    device: Rc<Device>,
    pub(crate) handle: u32,
    parent: u32,
    _dependencies: Vec<Rc<Object>>,
}
impl Object {
    pub(crate) fn new<T>(
        device: &Rc<Device>,
        parent: u32,
        class: u32,
        parameters: &mut T,
        dependencies: Vec<Rc<Object>>,
    ) -> Result<Rc<Self>> {
        let handle = device
            .client
            .allocate(parent, class, parameters)
            .map_err(|e| Error::Contract(format!("allocate RM class {class:#x}: {e}")))?;
        Ok(Rc::new(Self {
            device: Rc::clone(device),
            handle,
            parent,
            _dependencies: dependencies,
        }))
    }
    fn engine(
        device: &Rc<Device>,
        channel: &Rc<Object>,
        channel_class: u32,
        class: u32,
    ) -> Result<Rc<Self>> {
        if !crate::capabilities::COPY_CLASSES.contains(&class) {
            return Self::new(
                device,
                channel.handle,
                class,
                &mut (),
                vec![Rc::clone(channel)],
            );
        }
        // Public channel_descendant.c selects the GR partner CE from the
        // channel on newer GPUs. Older GPUs need explicit allocation params
        // for the same selection. All our TSGs use GRAPHICS and runqueue zero.
        device
            .client
            .compat
            .require_constant("NV2080_ENGINE_TYPE_GRAPHICS", 1)?;
        let mut partners = abi::EnginePartners {
            engine_type: 1,
            partnership_class: channel_class,
            runqueue: 0,
            ..Default::default()
        };
        device
            .client
            .control(device.subdevice, 0x20800147, &mut partners)?;
        let mut params = crate::capabilities::copy_engine_parameters(
            device.client.compat,
            class,
            &partners,
            |engine| {
                let mut query = abi::EngineClassList {
                    engine_type: engine,
                    ..Default::default()
                };
                device
                    .client
                    .control(device.subdevice, 0x20800124, &mut query)?;
                if query.count == 0 || query.count > 65536 {
                    return Err(Error::Contract("invalid RM engine class count".into()));
                }
                let mut classes = vec![0u32; query.count as usize];
                query.classes = classes.as_mut_ptr() as u64;
                device
                    .client
                    .control(device.subdevice, 0x20800124, &mut query)?;
                if query.count as usize > classes.len() {
                    return Err(Error::Contract(
                        "RM engine class count grew during query".into(),
                    ));
                }
                Ok(classes[..query.count as usize].contains(&class))
            },
        )?;
        Self::new(
            device,
            channel.handle,
            class,
            &mut params,
            vec![Rc::clone(channel)],
        )
    }
}
impl Drop for Object {
    fn drop(&mut self) {
        if let Err(e) = self.device.client.free(self.parent, self.handle) {
            diag!("RM object cleanup: {e}");
        }
    }
}

const TYPED_VA_CLASS: u32 = 0x50a0;
/// NVOS32_ATTR_PAGE_SIZE_4KB (24:23 = 1), NVOS32_ATTR_FORMAT_BLOCK_LINEAR
/// (17:16 = 2), NVOS32_ATTR_DEPTH (2:0: 8 = 1, 16 = 2, 32 = 4, 64 = 5),
/// NVOS32_ATTR_Z_TYPE (18:18: FIXED 0, FLOAT 1), NVOS32_ATTR_ZS_PACKING
/// (21:19: Z24S8/S8 0, S8Z24 1, Z32 2, Z32_X24S8 5, Z16 7); nvos.h (550.54.14).
const ATTR_PAGE_4KB: u32 = 1 << 23;
const ATTR_FORMAT_BLOCK_LINEAR: u32 = 2 << 16;
const ATTR_DEPTH_8: u32 = 1;
const ATTR_DEPTH_16: u32 = 2;
const ATTR_DEPTH_32: u32 = 4;
const ATTR_DEPTH_64: u32 = 5;
const ATTR_Z_FLOAT: u32 = 1 << 18;
const fn attr_zs_packing(packing: u32) -> u32 {
    packing << 19
}
/// NVOS32_TYPE_IMAGE / DEPTH / STENCIL (nvos.h).
const TYPE_IMAGE: u32 = 0;
const TYPE_DEPTH: u32 = 1;
/// Refused by RM on Pascal for every attr (measured, sweep only).
#[cfg(test)]
const TYPE_STENCIL: u32 = 16;

/// Maxwell/Pascal depth/stencil PTE kinds (public gm107/gp100 dev_mmu.h; NIL
/// nvc0_choose_pte_kind picks exactly these for uncompressed Z/S images).
const KIND_Z16: u32 = 0x01;
const KIND_S8Z24: u32 = 0x11;
const KIND_S8: u32 = 0x2a;
const KIND_Z24S8: u32 = 0x46;
const KIND_ZF32: u32 = 0x7b;
const KIND_ZF32_X24S8: u32 = 0xc3;

/// NVOS32 type and attributes of a typed virtual allocation (drivers without
/// kindOverride). RM does not take the requested `format` as the PTE kind:
/// it chooses the kind itself from type/attr (memmgrChooseKind_HAL; the client
/// kind is honoured only with PDB_PROP_GPU_EXTERNAL_HEAP_CONTROL, which no
/// public 550 path sets, mem_mgr_tu102.c memmgrChooseKind_TU102), writes the
/// choice to the VA memdesc (heap.c memdescSetPteKind(pFbAllocInfo->format))
/// and returns it in `format` (mem_utils.c memmgrAllocResources). Turing+
/// picks GENERIC_MEMORY for every image regardless of FORMAT, but pre-Turing
/// separates pitch from block-linear kinds, so attr FORMAT PITCH (0) made RM
/// map Maxwell/Pascal block-linear images with a pitch kind and every ROP
/// write raised Xid 13 (P100 closed 550.54.14, run 20261014-p100ab; same GPU
/// on 580 kindOverride passes). Block-linear generic requests therefore ask
/// for FORMAT BLOCK_LINEAR with a 32-bit element depth, exactly as RM's own
/// external-mapping path does to obtain a block-linear kind
/// (nv_gpu_ops.c nvGpuOpsGetExternalAllocPtes: FORMAT BLOCK_LINEAR + DEPTH,
/// then memmgrChooseKind_HAL).
///
/// Depth/stencil kinds are chosen by RM only for NVOS32_TYPE_DEPTH/STENCIL,
/// from Z_TYPE, ZS_PACKING and DEPTH (memmgrChooseKind_TU102 → ChooseKindZ;
/// an IMAGE request always gets a colour/generic kind, which is what the
/// P100/P40 550 runs measured: kind 0x11 requested, refused). The request per
/// kind mirrors memmgrChooseKindZ_TU102's attribute → kind table, extended by
/// the pre-Turing ZF32 kind (Z32 FLOAT, which Turing folds into GENERIC).
/// Measured closed-KM chooser (typed_va_kind_sweep, runs 20261015-rm-r19/b:
/// P100 550.54.14 + 580.178.04, P40 550.163.01 + 550.142, GTX 1080 Ti
/// 535.309.01, GTX TITAN X 580.159.03, GTX 1080 580.173.02: all 240 rows
/// identical), TYPE_DEPTH + BLOCK_LINEAR: FIXED Z16 (any depth) → 0x01;
/// FIXED S8Z24|X8Z24 → 0x11; FIXED Z24S8 (depth 16..64) | Z24X8 → 0x46;
/// Z24S8/S8 packing + DEPTH_8 (FIXED or FLOAT) → 0x2a; FLOAT Z32 → 0x7b;
/// FLOAT Z32_X24S8 → 0xc3; every other combination and every TYPE_STENCIL
/// request → RM 0x1f; TYPE_IMAGE → 0xfe regardless of Z attributes.
/// Pitch (0) keeps the plain request.
fn typed_va_request(kind: Option<u32>, generic_kind: u32) -> (u32, u32) {
    if generic_kind == KIND_GENERIC {
        return (TYPE_IMAGE, ATTR_PAGE_4KB);
    }
    let bl = ATTR_PAGE_4KB | ATTR_FORMAT_BLOCK_LINEAR;
    match kind {
        Some(k) if k == generic_kind => (TYPE_IMAGE, bl | ATTR_DEPTH_32),
        Some(KIND_Z16) => (TYPE_DEPTH, bl | attr_zs_packing(7) | ATTR_DEPTH_16),
        Some(KIND_S8Z24) => (TYPE_DEPTH, bl | attr_zs_packing(1) | ATTR_DEPTH_32),
        Some(KIND_Z24S8) => (TYPE_DEPTH, bl | attr_zs_packing(0) | ATTR_DEPTH_32),
        Some(KIND_ZF32) => (
            TYPE_DEPTH,
            bl | ATTR_Z_FLOAT | attr_zs_packing(2) | ATTR_DEPTH_32,
        ),
        Some(KIND_ZF32_X24S8) => (
            TYPE_DEPTH,
            bl | ATTR_Z_FLOAT | attr_zs_packing(5) | ATTR_DEPTH_64,
        ),
        // NVOS32_TYPE_STENCIL is refused outright on Pascal (RM 0x1f for every
        // attr); S8 comes from TYPE_DEPTH + ZS_PACKING S8 + DEPTH_8.
        Some(KIND_S8) => (TYPE_DEPTH, bl | attr_zs_packing(0) | ATTR_DEPTH_8),
        _ => (TYPE_IMAGE, ATTR_PAGE_4KB),
    }
}

struct MappingPlan {
    typed: Option<abi::MemoryAlloc>,
    flags: u32,
    wire_kind: u32,
}
/// Keep invalid range/kind/overflow decisions before any ioctl or reservation.
/// Turing+ PTE kinds (public tu102/gb202 dev_mmu.h).
const KIND_GENERIC: u32 = 0x06;
const KIND_GENERIC_COMPRESSIBLE: u32 = 0x08;
const BIG_PAGE: u64 = crate::rm::BIG_PAGE as u64;

/// The PTE kind a VA may carry for this memory. NVK asks for the generic
/// compressible kind for images it decided to compress; only memory RM made
/// compressible at allocation keeps it (RM grants comptags there, clears them,
/// and requires big-page-aligned compressed mappings). Every other memory
/// (host-visible, sysmem, imports, exports) gets the uncompressed generic kind.
fn effective_kind(kind: Option<u32>, generic_kind: u32, compressed_memory: bool) -> Option<u32> {
    match kind {
        Some(KIND_GENERIC_COMPRESSIBLE) if generic_kind == KIND_GENERIC && !compressed_memory => {
            Some(KIND_GENERIC)
        }
        other => other,
    }
}

#[cfg(test)]
fn mapping_plan(
    compat: crate::compat::Compat,
    offset: u64,
    length: u64,
    allocation_size: u64,
    address: Option<u64>,
    kind: Option<u32>,
) -> Result<MappingPlan> {
    mapping_plan_paged(
        compat,
        offset,
        length,
        allocation_size,
        address,
        kind,
        false,
        KIND_GENERIC,
    )
}

/// `big_pages`: the physical allocation uses big pages. A mapping whose
/// offset, length and fixed VA are all 64 KiB aligned then asks for big PTEs
/// (stock maps device-local memory with NVOS46 PAGE_SIZE BIG, flags 0x208).
// One argument per independent RM input (layout, placement, kind, page and
// architecture facts); a parameter struct would only rename them.
#[allow(clippy::too_many_arguments)]
fn mapping_plan_paged(
    compat: crate::compat::Compat,
    offset: u64,
    length: u64,
    allocation_size: u64,
    address: Option<u64>,
    kind: Option<u32>,
    big_pages: bool,
    generic_kind: u32,
) -> Result<MappingPlan> {
    if length == 0
        || length > (1u64 << 40)
        || !offset.is_multiple_of(4096)
        || !length.is_multiple_of(4096)
        || offset
            .checked_add(length)
            .is_none_or(|end| end > allocation_size)
        || address.is_some_and(|a| {
            !a.is_multiple_of(4096) || a.checked_add(length).is_none_or(|end| end > (1u64 << 40))
        })
        || kind.is_some_and(|kind| kind > 255)
    {
        return Err(Error::Contract(
            "GPU mapping exceeds memory/VA/kind limits".into(),
        ));
    }
    // Pre-Turing S8 always comes from RM's chooser (typed VA), also on drivers
    // with kindOverride: closed 580.178.04 on GP100 refuses kindOverride 0x2a
    // (virtmemMapTo memmgrIsKind FB_IS_KIND_SUPPORTED, RM 0x1f) while its
    // chooser grants 0x2a for TYPE_DEPTH + S8 + DEPTH_8, and that kind renders
    // S8 correctly on GP100 (P100 550.54.14 vk-depth S8_UINT, run
    // 20261015-rm-r19b). Stock never uses kindOverride for memory objects;
    // RM picks the kind (observed in stock's RM alloc requests).
    let s8_chooser = generic_kind != KIND_GENERIC && kind == Some(KIND_S8);
    let typed = kind.is_some() && (!compat.supports_kind_override() || s8_chooser);
    let big = big_pages
        && offset.is_multiple_of(BIG_PAGE)
        && length.is_multiple_of(BIG_PAGE)
        && address.is_some_and(|a| a.is_multiple_of(BIG_PAGE));
    if kind == Some(KIND_GENERIC_COMPRESSIBLE) && big_pages && !big {
        // RM rejects compressed PTEs at a VA not aligned to the big page.
        return Err(Error::Contract(
            "compressed mapping must be 64 KiB aligned in offset, size and VA".into(),
        ));
    }
    Ok(MappingPlan {
        typed: if typed {
            let (memory_type, attr) = typed_va_request(kind, generic_kind);
            Some(abi::MemoryAlloc {
                flags: 0x80000 | 0x100 | if address.is_some() { 0x10 } else { 0 },
                memory_type,
                attr,
                attr2: 1,
                format: kind.unwrap_or(0),
                size: length,
                alignment: 4096,
                offset: address.unwrap_or(0),
                limit: length - 1,
                ..Default::default()
            })
        } else {
            None
        },
        // NVOS46 PAGE_SIZE (11:8): 4KB = 1, BIG = 2.
        flags: (if big { 2 << 8 } else { 1 << 8 })
            | (1 << 4)
            | if address.is_some() { 1 << 15 } else { 0 }
            | if typed {
                1 << 3
            } else if kind.is_some() {
                1 << 19
            } else {
                0
            },
        wire_kind: if typed { 0 } else { kind.unwrap_or(0) },
    })
}

pub struct AddressSpace {
    device: Rc<Device>,
    pub(crate) vaspace: Rc<Object>,
    dma: Rc<Object>,
}
impl AddressSpace {
    pub fn new(device: &Rc<Device>) -> Result<Self> {
        // Kernel RM owns its page tables. No external owner and no faulting.
        let mut parameters = abi::VaSpace {
            base: 0,
            size: 1 << 40,
            ..Default::default()
        };
        let vaspace = Object::new(device, device.handle, 0x90f1, &mut parameters, vec![])?;
        let mut params = abi::VirtualMemory {
            vaspace: vaspace.handle,
            ..Default::default()
        };
        let dma = Object::new(
            device,
            device.handle,
            0x70,
            &mut params,
            vec![Rc::clone(&vaspace)],
        )?;
        Ok(Self {
            device: Rc::clone(device),
            vaspace,
            dma,
        })
    }
    pub fn map(&self, memory: &Rc<Memory>) -> Result<GpuMapping> {
        self.map_internal(memory, 0, memory.size as u64, None, None)
    }
    /// Map an allocation range at a caller-managed VA. NVOS46 fixed-offset and
    /// page-kind override are public RM contracts (nvos.h), not relocations.
    pub(crate) fn map_range(
        &self,
        memory: &Rc<Memory>,
        offset: u64,
        length: u64,
        address: Option<u64>,
        kind: u32,
    ) -> Result<GpuMapping> {
        self.map_internal(memory, offset, length, address, Some(kind))
    }
    fn map_internal(
        &self,
        memory: &Rc<Memory>,
        offset: u64,
        length: u64,
        address: Option<u64>,
        kind: Option<u32>,
    ) -> Result<GpuMapping> {
        let requested = kind;
        let kind = match kind {
            Some(_) => effective_kind(kind, self.device.generic_kind()?, memory.compressed),
            None => None,
        };
        if requested == Some(KIND_GENERIC_COMPRESSIBLE)
            && std::env::var_os("NV_RM_DEBUG_COMPRESSION").is_some()
        {
            crate::diag::diag!(
                "nvk-rm: compressible image VA {address:?} len {length:#x}: kind {kind:?} (memory compressed={})",
                memory.compressed
            );
        }
        // Only kinded mappings need the architecture's generic kind (cached query).
        let generic_kind = if kind.is_some() {
            self.device.generic_kind()?
        } else {
            KIND_GENERIC
        };
        let plan = mapping_plan_paged(
            self.device.client.compat,
            offset,
            length,
            memory.size as u64,
            address,
            kind,
            memory.big_pages,
            generic_kind,
        )?;
        if !Rc::ptr_eq(&self.device, &memory.device) {
            return Err(Error::Contract(
                "cross-device mapping needs explicit peer support".into(),
            ));
        }
        // Before kindOverride, a typed virtual allocation carries the kind.
        // Public535 virtual_mem.c + dma.c: reservation belongs to NV50_MEMORY_VIRTUAL,
        // PAGE_KIND_VIRTUAL copies its kind into the mapped physical descriptor.
        let dma = if let Some(mut params) = plan.typed {
            params.owner = self.device.client.root;
            params.vaspace = self.vaspace.handle;
            let requested = params.format;
            let object = Object::new(
                &self.device,
                self.device.handle,
                TYPED_VA_CLASS,
                &mut params,
                vec![Rc::clone(&self.vaspace)],
            )?;
            // RM returns the kind it chose in `format`. Pre-Turing kinds are
            // layout-specific: a different kind is a wrong mapping, never a
            // fallback (the reservation is freed with `object`).
            typed_kind_check(generic_kind, requested, params.format)?;
            object
        } else {
            Rc::clone(&self.dma)
        };
        let mut call = abi::MapDma {
            client: self.device.client.root,
            device: self.device.handle,
            dma: dma.handle,
            memory: memory.handle,
            offset,
            length,
            address: address.unwrap_or(0),
            kind: plan.wire_kind,
            flags: plan.flags,
            ..Default::default()
        };
        self.device.client.map_dma(&mut call)?;
        let mapping = GpuMapping {
            memory: Rc::clone(memory),
            dma,
            pub_address: call.address,
            length,
            active: true,
        };
        if !call.address.is_multiple_of(4096)
            || call
                .address
                .checked_add(length)
                .is_none_or(|end| end > (1u64 << 40))
            || address.is_some_and(|expected| expected != call.address)
        {
            // Owned mapping rolls back before the reservation is freed.
            return Err(Error::Contract(
                "RM returned invalid or displaced GPU VA".into(),
            ));
        }
        Ok(mapping)
    }
}
/// Pre-Turing typed VA: RM's chosen kind must be the requested one.
fn typed_kind_check(generic_kind: u32, requested: u32, chosen: u32) -> Result<()> {
    if generic_kind != KIND_GENERIC && chosen != requested {
        return Err(Error::Contract(format!(
            "RM chose PTE kind {chosen:#x} for a typed VA that requested kind {requested:#x} \
             (driver without kindOverride, pre-Turing); refusing the mapping"
        )));
    }
    Ok(())
}
pub struct GpuMapping {
    memory: Rc<Memory>,
    dma: Rc<Object>,
    pub_address: u64,
    length: u64,
    active: bool,
}
impl GpuMapping {
    pub fn address(&self) -> u64 {
        self.pub_address
    }
    /// Mapped length in bytes (from the map request).
    pub(crate) fn length(&self) -> u64 {
        self.length
    }
    /// Explicit unbind may release a VA reservation only after RM succeeds.
    /// On failure the caller keeps this object and marks its device lost.
    pub(crate) fn try_unmap(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let device = &self.memory.device;
        let mut call = abi::UnmapDma {
            client: device.client.root,
            device: device.handle,
            dma: self.dma.handle,
            memory: self.memory.handle,
            address: self.pub_address,
            size: 0, // full mapping, including old NVOS47 without size
            ..Default::default()
        };
        device.client.unmap_dma(&mut call)?;
        self.active = false;
        Ok(())
    }
}
impl Drop for GpuMapping {
    fn drop(&mut self) {
        if let Err(error) = self.try_unmap() {
            diag!("GPU mapping cleanup: {error}");
        }
    }
}

pub(crate) use nv_rm_core::push::{hi_lo, method};

/// One isolated hardware-copy smoke operation. It owns every object and mapping
/// through completion or channel teardown. No shader compiler or Vulkan API yet.
pub fn copy_check(device: &Rc<Device>) -> Result<usize> {
    run_check(device, false)
}
pub fn compute_check(device: &Rc<Device>) -> Result<usize> {
    run_check(device, true)
}

fn run_check(device: &Rc<Device>, compute: bool) -> Result<usize> {
    let count = if compute { 256 } else { 16384 };
    let engine_class = if compute { 0xcec0 } else { 0xcab5 };
    const ENTRIES: usize = 1024;
    let classes = device.classes()?;
    for required in [0xc661, 0xc96f, engine_class] {
        if !classes.contains(&required) {
            return Err(Error::Contract(format!(
                "copy backend requires RM class {required:#x}, absent from device"
            )));
        }
    }
    let space = AddressSpace::new(device)?;
    let source = device.allocate_vram(count * 4)?;
    let destination = device.allocate_vram(count * 4)?;
    let queue_memory = device.allocate_memory(65536, true)?;
    let command_memory = device.allocate_vram(4096)?;
    let completion_memory = device.allocate_vram(4096)?;
    let src_va = space.map(&source)?;
    let dst_va = space.map(&destination)?;
    let queue_va = space.map(&queue_memory)?;
    let command_va = space.map(&command_memory)?;
    let completion_va = space.map(&completion_memory)?;
    let mut src = source.map()?;
    let mut dst = destination.map()?;
    let mut ring = queue_memory.map()?;
    let mut command = command_memory.map()?;
    let mut completion = completion_memory.map()?;
    let mut doorbell = device.map_usermode()?;
    for i in 0..count {
        src.write_u32(i * 4, (i as u32).wrapping_mul(0x12345) ^ 0x5aa5f00d)?;
        dst.write_u32(i * 4, 0)?;
    }
    for i in 0..65536 / 4 {
        ring.write_u32(i * 4, 0)?;
    }
    completion.write_u32(0, 0)?;
    let compute_job = if compute {
        Some(crate::compute::ComputeJob::new(
            device,
            &space,
            dst_va.address(),
            completion_va.address(),
        )?)
    } else {
        None
    };
    // Declaration order keeps the vote alive through channel teardown on errors.
    // TSG channels must use a context-share object; an explicit channel hVASpace
    // is rejected in kernel_channel.c:436 even when the TSG has that same VA.
    let mut group_params = abi::ChannelGroup {
        vaspace: space.vaspace.handle,
        engine: 1, // NV2080_ENGINE_TYPE_GRAPHICS; zero is not a valid TSG engine.
        ..Default::default()
    };
    let group = Object::new(
        device,
        device.handle,
        0xa06c,
        &mut group_params,
        vec![Rc::clone(&space.vaspace)],
    )?;
    let mut context_params = abi::ContextShare {
        vaspace: space.vaspace.handle,
        flags: 1,
        subcontext: 0,
    };
    let context = Object::new(
        device,
        group.handle,
        0x9067,
        &mut context_params,
        vec![Rc::clone(&group), Rc::clone(&space.vaspace)],
    )?;
    let mut channel_params = abi::Channel {
        fifo_address: queue_va.address(),
        entries: ENTRIES as u32,
        context_share: context.handle,
        ..Default::default()
    };
    channel_params.userd_memory[0] = queue_memory.handle;
    channel_params.userd_offsets[0] = (ENTRIES * 8) as u64;
    let channel = Object::new(
        device,
        group.handle,
        0xc96f,
        &mut channel_params,
        vec![Rc::clone(&group), Rc::clone(&context)],
    )?;
    let engine = Object::engine(device, &channel, 0xc96f, engine_class)?;
    let mut token = 0u32;
    device
        .client
        .control(channel.handle, 0xc36f0108, &mut token)?;
    device
        .client
        .control(group.handle, 0xa06c0101, &mut [1u8, 0, 0])?;

    let mut words = Vec::new();
    if let Some(job) = &compute_job {
        method(&mut words, 1, 0, &[engine_class]);
        method(&mut words, 1, 0x7b0, &hi_lo(crate::compute::LOCAL_WINDOW));
        method(&mut words, 1, 0x2a0, &hi_lo(crate::compute::SHARED_WINDOW));
        method(&mut words, 1, 0x2b4, &[(job.address >> 8) as u32]);
        method(&mut words, 1, 0x2c0, &[9]); // PREFETCH_SCHEDULE
    } else {
        method(&mut words, 4, 0, &[0xcab5]); // SET_OBJECT (class binding)
        let source_addr = hi_lo(src_va.address());
        let destination_addr = hi_lo(dst_va.address());
        method(
            &mut words,
            4,
            0x400,
            &[
                source_addr[0],
                source_addr[1],
                destination_addr[0],
                destination_addr[1],
            ],
        );
        method(&mut words, 4, 0x418, &[(count * 4) as u32]);
        method(&mut words, 4, 0x300, &[2 | (1 << 7) | (1 << 8)]); // non-pipelined, pitch layouts
        let completion_addr = hi_lo(completion_va.address());
        method(
            &mut words,
            4,
            0x240,
            &[completion_addr[0], completion_addr[1], 1],
        );
        method(&mut words, 4, 0x300, &[(1 << 2) | (1 << 3)]); // flush + release one-word semaphore
    }
    for (i, word) in words.iter().enumerate() {
        command.write_u32(i * 4, *word)?;
    }
    let entry = command_va.address() | ((words.len() as u64) << 42) | (1 << 41);
    ring.write_u32(0, entry as u32)?;
    ring.write_u32(4, (entry >> 32) as u32)?;
    crate::platform::device_full_barrier();
    ring.write_u32(ENTRIES * 8 + 0x8c, 1)?; // GPPut in BlackwellAControlGPFifo
    crate::platform::device_full_barrier();
    doorbell.write_u32(0x90, token)?;
    crate::platform::device_full_barrier();
    let deadline = Instant::now() + Duration::from_secs(2);
    while completion.read_u32(0)? != 1 {
        if Instant::now() >= deadline {
            return Err(Error::Contract("GPU work timed out after 2 seconds; channel will be torn down before memory is released".into()));
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    crate::platform::device_full_barrier();
    for i in 0..count {
        let expected = if compute {
            i as u32 * 17 + 123
        } else {
            (i as u32).wrapping_mul(0x12345) ^ 0x5aa5f00d
        };
        let actual = dst.read_u32(i * 4)?;
        if actual != expected {
            return Err(Error::Contract(format!(
                "GPU result mismatch word {i}: {actual:#x} != {expected:#x}"
            )));
        }
    }
    // Stop scheduling before destroying the channel and releasing its mappings.
    device
        .client
        .control(group.handle, 0xa06c0101, &mut [0u8, 0, 0])?;
    drop(engine);
    drop(channel);
    drop(context);
    drop(group);
    Ok(count)
}

/// Submit one bounded packet stream and wait for its GPU release. Buffers used
/// by the stream must remain alive in the caller until this returns. A failed
/// submission tears down its channel before returning; callers mark device lost.
pub(crate) fn submit_once(
    device: &Rc<Device>,
    space: &AddressSpace,
    engine_class: u32,
    completion: &rm::Mapping,
    words: &[u32],
) -> Result<()> {
    const ENTRIES: usize = 1024;
    if words.is_empty() || words.len() > 1024 {
        return Err(Error::Contract(
            "submission packet length outside command buffer".into(),
        ));
    }
    let classes = device.classes()?;
    for class in [0xc661, 0xc96f, engine_class] {
        if !classes.contains(&class) {
            return Err(Error::Contract(format!(
                "missing native engine class {class:#x}"
            )));
        }
    }
    let queue_memory = device.allocate_memory(65536, true)?;
    let command_memory = device.allocate_vram(4096)?;
    let queue_va = space.map(&queue_memory)?;
    let command_va = space.map(&command_memory)?;
    if command_va.address() >> 40 != 0 || command_va.address() & 3 != 0 {
        return Err(Error::Contract(
            "pushbuffer address outside GPFIFO encoding".into(),
        ));
    }
    let mut ring = queue_memory.map()?;
    let mut command = command_memory.map()?;
    let mut doorbell = device.map_usermode()?;
    for i in 0..65536 / 4 {
        ring.write_u32(i * 4, 0)?;
    }
    // Declaration order keeps the vote alive through channel teardown on errors.
    // TSG channels must use a context-share object; an explicit channel hVASpace
    // is rejected in kernel_channel.c:436 even when the TSG has that same VA.
    let mut group_params = abi::ChannelGroup {
        vaspace: space.vaspace.handle,
        engine: 1, // NV2080_ENGINE_TYPE_GRAPHICS; zero is not a valid TSG engine.
        ..Default::default()
    };
    let group = Object::new(
        device,
        device.handle,
        0xa06c,
        &mut group_params,
        vec![Rc::clone(&space.vaspace)],
    )?;
    let mut context_params = abi::ContextShare {
        vaspace: space.vaspace.handle,
        flags: 1,
        subcontext: 0,
    };
    let context = Object::new(
        device,
        group.handle,
        0x9067,
        &mut context_params,
        vec![Rc::clone(&group), Rc::clone(&space.vaspace)],
    )?;
    let mut channel_params = abi::Channel {
        fifo_address: queue_va.address(),
        entries: ENTRIES as u32,
        context_share: context.handle,
        ..Default::default()
    };
    channel_params.userd_memory[0] = queue_memory.handle;
    channel_params.userd_offsets[0] = (ENTRIES * 8) as u64;
    let channel = Object::new(
        device,
        group.handle,
        0xc96f,
        &mut channel_params,
        vec![Rc::clone(&group), Rc::clone(&context)],
    )?;
    let engine = Object::engine(device, &channel, 0xc96f, engine_class)?;
    let mut token = 0u32;
    device
        .client
        .control(channel.handle, 0xc36f0108, &mut token)?;
    device
        .client
        .control(group.handle, 0xa06c0101, &mut [1u8, 0, 0])?;

    for (i, word) in words.iter().enumerate() {
        command.write_u32(i * 4, *word)?;
    }
    let entry = command_va.address() | ((words.len() as u64) << 42) | (1 << 41);
    ring.write_u32(0, entry as u32)?;
    ring.write_u32(4, (entry >> 32) as u32)?;
    crate::platform::device_full_barrier();
    ring.write_u32(ENTRIES * 8 + 0x8c, 1)?; // GPPut in BlackwellAControlGPFifo
    crate::platform::device_full_barrier();
    doorbell.write_u32(0x90, token)?;
    crate::platform::device_full_barrier();
    let deadline = Instant::now() + Duration::from_secs(2);
    while completion.read_u32(0)? != 1 {
        if Instant::now() >= deadline {
            return Err(Error::Contract("GPU work timed out after 2 seconds; channel will be torn down before memory is released".into()));
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    crate::platform::device_full_barrier();
    device
        .client
        .control(group.handle, 0xa06c0101, &mut [0u8, 0, 0])?;
    drop(engine);
    drop(channel);
    drop(context);
    drop(group);
    Ok(())
}

/// Explicit host method families from published cl{a06f,a16f,b06f,c06f,c36f}.
/// Unknown classes fail instead of assuming numeric ordering means compatibility.
pub(crate) fn modern_host(channel: u32) -> Result<bool> {
    match channel {
        0xc36f | 0xc46f | 0xc56f | 0xc76f | 0xc86f | 0xc96f | 0xca6f => Ok(true),
        0xa06f | 0xa16f | 0xb06f | 0xc06f => Ok(false),
        _ => Err(Error::Contract(format!(
            "unimplemented host method class {channel:#x}"
        ))),
    }
}
fn host_visibility(words: &mut Vec<u32>, channel: u32) -> Result<()> {
    modern_host(channel)?;
    if channel == 0xa06f {
        method(words, 0, 0x50, &[0]);
    }
    // SET_REFERENCE implies WFI
    else {
        method(words, 0, 0x78, &[0]);
    }
    if matches!(channel, 0xa06f | 0xa16f) {
        method(words, 0, 0x28, &[0, 5 << 27]);
    } else if channel == 0xb06f {
        method(words, 0, 0x30, &[0, 5 << 27]);
    } else {
        method(words, 0, 0x28, &[0, 0, 0, 5 << 27]);
    }
    Ok(())
}
fn host_semaphore(
    words: &mut Vec<u32>,
    channel: u32,
    native64: bool,
    address: u64,
    value: u64,
    acquire: bool,
    timestamp: bool,
) -> Result<()> {
    if !address.is_multiple_of(16) || address >= 1 << 40 {
        return Err(Error::Contract(
            "unaligned or out-of-range host semaphore address".into(),
        ));
    }
    if modern_host(channel)? {
        let operation = if acquire {
            (if native64 { 2 } else { 3 }) | (1 << 12)
        } else {
            1 | (1 << 20) | if timestamp { 1 << 25 } else { 0 }
        };
        method(
            words,
            0,
            0x5c,
            &[
                address as u32,
                (address >> 32) as u32,
                value as u32,
                if native64 { (value >> 32) as u32 } else { 0 },
                operation | if native64 { 1 << 24 } else { 0 },
            ],
        );
    } else {
        if native64 {
            return Err(Error::Contract(
                "pre-Volta host lacks native64 semaphore payload".into(),
            ));
        }
        // Legacy RELEASE_WFI_EN is zero, unlike the modern SEM_EXECUTE bit.
        let operation = if acquire {
            4 | (1 << 12)
        } else {
            2 | (1 << 24)
        };
        method(
            words,
            0,
            0x10,
            &[
                (address >> 32) as u32,
                address as u32,
                value as u32,
                operation,
            ],
        );
    }
    Ok(())
}

/// A persistent graphics/compute channel for Mesa's nvkmd interface. Engine
/// context survives submissions; only the bounded GPFIFO ring is advanced.
/// Drop order is intentional: engine/channel handles precede their memory, and
/// the ledger's semaphore-slot references (last field) are released only after
/// the channel object is freed, so no GPU access to a recycled slot is possible.
/// NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC / _SPECIFIED (nvos.h).
const CTXSHARE_SYNC: u32 = 0;
const CTXSHARE_SPECIFIED: u32 = 2;
/// NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE (4:4) and
/// NVOS04_FLAGS_GROUP_CHANNEL_THREAD_ONE (29:28), alloc_channel.h.
const CHANNEL_RUNQUEUE_ONE: u32 = 1 << 4;
const CHANNEL_THREAD_ONE: u32 = 1 << 28;

/// One RM channel group (KEPLER_CHANNEL_GROUP_A) shared by every queue whose
/// channel sits in it. The `Rc` count is the group's reference count: RM
/// frees the TSG when the last queue holding it drops. `free_veids` is the
/// set of unused subcontext IDs, a 64-bit mask that starts full; a channel
/// takes its lowest set bit (stock 615 TSG block +0x10, initialised to all
/// ones at E 0x9e1400 and consumed with `tzcnt` at E 0x9e5f4f..0x9e5f6c).
/// RM bounds the VEID by the GPU's subcontext count and fails the
/// context-share allocation beyond it (kernel_ctxshare.c eheapAlloc).
///
/// VEID 0 belongs to the group, not to the queue that created it: the GR
/// engine saves and restores the group's graphics context state (the ZCULL
/// buffer among it) through subcontext 0's address space, so its context
/// share must live as long as any channel of the group. Measured on GB202:
/// freeing the VEID-0 share while a VEID-1 channel kept running faulted
/// GPCCLIENT_RAST with FAULT_PDE at the ZCULL buffer's VA (Xid 31).
/// Field order is the teardown order: the VEID-0 share is freed, then the
/// TSG object (and with it RM's GR context of the group), and only then is
/// the ZCULL buffer that context switches save into unmapped and freed.
pub(crate) struct Tsg {
    veid0_share: std::cell::RefCell<Option<Rc<Object>>>,
    object: Rc<Object>,
    free_veids: std::cell::Cell<u64>,
    /// ZCULL context-switch buffer of the group's GR context, bound by its
    /// graphics channel (VEID 0).
    zcull: std::cell::RefCell<Option<GpuMapping>>,
}
impl Tsg {
    fn new(object: Rc<Object>) -> Rc<Self> {
        Rc::new(Self {
            veid0_share: std::cell::RefCell::new(None),
            object,
            free_veids: std::cell::Cell::new(u64::MAX),
            zcull: std::cell::RefCell::new(None),
        })
    }
    fn keep_veid0_share(&self, share: Rc<Object>) -> Result<()> {
        let mut slot = self.veid0_share.borrow_mut();
        if slot.is_some() {
            return Err(Error::Contract(
                "TSG already has a VEID 0 context share".into(),
            ));
        }
        *slot = Some(share);
        Ok(())
    }
    fn keep_zcull(&self, mapping: GpuMapping) -> Result<()> {
        let mut zcull = self.zcull.borrow_mut();
        if zcull.is_some() {
            return Err(Error::Contract(
                "TSG already has a ZCULL context-switch buffer bound".into(),
            ));
        }
        *zcull = Some(mapping);
        Ok(())
    }
}
/// A queue's VEID in its TSG. Dropping it returns the VEID to the TSG's free
/// set and then releases the queue's TSG reference, after the queue's channel
/// and its context share were freed (stock release loop E 0x9e3178: share
/// free, VEID back into +0x10, refcount decrement, TSG free on the last).
/// VEID 0 is not returned: it stays with the TSG together with its share.
struct Subcontext {
    veid: u32,
    tsg: Rc<Tsg>,
}
impl Subcontext {
    fn take(tsg: Rc<Tsg>) -> Result<Self> {
        let (veid, rest) = lowest_free_veid(tsg.free_veids.get()).ok_or_else(|| {
            Error::Contract("TSG has no free subcontext (all 64 VEIDs in use)".into())
        })?;
        tsg.free_veids.set(rest);
        Ok(Self { veid, tsg })
    }
    /// Before Volta a channel group has no subcontexts: every channel of the
    /// group runs in the group's single GR context. Such a member holds the
    /// group without a VEID (0, never returned to the free set).
    fn legacy(tsg: Rc<Tsg>) -> Self {
        Self { veid: 0, tsg }
    }
}
/// Lowest free VEID of `free` and the set without it; `None` when empty.
fn lowest_free_veid(free: u64) -> Option<(u32, u64)> {
    (free != 0).then(|| {
        let veid = free.trailing_zeros();
        (veid, free & !(1u64 << veid))
    })
}
impl Drop for Subcontext {
    fn drop(&mut self) {
        if self.veid == 0 {
            return;
        }
        self.tsg
            .free_veids
            .set(self.tsg.free_veids.get() | 1u64 << self.veid);
    }
}

pub(crate) struct Queue {
    _events: crate::events::ChannelEvents,
    userd: Option<rm::Mapping>,
    channel_class: u32,
    completion: crate::fences::NativeFenceSurface,
    engines: Vec<Rc<Object>>,
    channel: Rc<Object>,
    /// This channel's subcontext (VEID) in its TSG. Declared after `channel`:
    /// the channel's RM free releases its context-share dependency first, so
    /// the VEID returns to the TSG only after RM freed the share, and the TSG
    /// reference drops last.
    subcontext: Subcontext,
    /// Compute class without 3D on this channel: every batch's tail starts
    /// with a compute-class WAIT_FOR_IDLE (see `submit_encoded`).
    compute_only: bool,
    error_notifier: rm::Mapping,
    ring: rm::Mapping,
    _ring_gpu: GpuMapping,
    tail: rm::Mapping,
    tail_gpu: GpuMapping,
    doorbell: Option<rm::Mapping>,
    /// Doorbell page reached through BAR1 (HOPPER_USERMODE_A bBar1Mapping),
    /// ordered behind the BAR1 GP_PUT write; otherwise a BAR0 page.
    bar1_doorbell: bool,
    token: u32,
    put: usize,
    shared: Arc<QueueShared>,
    lost: bool,
    stopped: bool,
    ledger: SubmissionLedger,
    scratch: Scratch,
    /// Perf capture only (capture.rs): per-submission start/end timestamp
    /// ring; None unless the process armed a capture. Drops after the channel.
    timing: Option<crate::capture::TimingRing>,
    /// Last: drops after the channel (see struct doc).
    held: HeldSlots,
}
/// Reused encode buffers (capacity retained: no steady-state allocation).
#[derive(Default)]
struct Scratch {
    prefix: Vec<u32>,
    signals: Vec<u32>,
    tail: Vec<u32>,
}

/// Lock-free view of a queue's private retirement word for caller CPU waits.
/// The word is a pool slot, so it stays mapped for the device lifetime even if
/// a racing waiter outlives the queue object.
pub(crate) struct QueueShared {
    pool: Arc<SlotPool>,
    retire: Slot,
    native64: bool,
    sequence: AtomicU64,
    lost: AtomicBool,
}
impl QueueShared {
    /// Completed private retirement value (sem32 reconstructed against the
    /// maximum issued sequence, published before every doorbell).
    pub(crate) fn completed(&self) -> Result<u64> {
        let sequence = self.sequence.load(Ordering::Acquire);
        if self.native64 {
            self.pool.read64_raw(self.retire)
        } else {
            crate::fences::extend_payload(self.pool.read32(self.retire)?, sequence)
        }
    }
    pub(crate) fn retired(&self, token: u64) -> Result<bool> {
        if self.lost.load(Ordering::Acquire) {
            return Err(Error::DeviceLost("persistent RM channel is lost".into()));
        }
        if token > self.sequence.load(Ordering::Acquire) {
            return Err(Error::Contract("unissued RM completion token".into()));
        }
        Ok(token == 0 || self.completed()? >= token)
    }
}
impl Drop for QueueShared {
    fn drop(&mut self) {
        self.pool.release(self.retire);
    }
}

/// Semaphore-slot references owned by accepted submissions, keyed by token.
/// Capacity is retained across frames: no steady-state allocation.
struct HeldSlots {
    pool: Arc<SlotPool>,
    slots: VecDeque<(u64, Slot)>,
}
impl HeldSlots {
    fn release_through(&mut self, retired: u64) {
        while self
            .slots
            .front()
            .is_some_and(|(token, _)| *token <= retired)
        {
            let (_, slot) = self.slots.pop_front().unwrap();
            self.pool.release(slot);
        }
    }
}
impl Drop for HeldSlots {
    fn drop(&mut self) {
        // Runs after the channel/engines were freed (field order): the GPU can
        // no longer reach these slots even if their batches never retired.
        for (_, slot) in self.slots.drain(..) {
            self.pool.release(slot);
        }
    }
}

/// One GPU semaphore operation in a submission (public host methods).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SemaphoreOp {
    /// Acquire a binary/gate slot == 1 (no GPU reset: the payload was detached).
    AcquireBinary(u64),
    /// Acquire timeline >= value (strict 64-bit GEQ, or circular 32-bit GEQ on
    /// a split-view low word for pre-Volta hosts).
    AcquireTimeline(u64, u64),
    /// Release 1 to a binary slot (after the work, with WFI).
    ReleaseBinary(u64),
    /// Release a timeline value (64-bit, or the low word on pre-Volta hosts).
    ReleaseTimeline(u64, u64),
}
fn encode_op(words: &mut Vec<u32>, channel: u32, native64: bool, op: SemaphoreOp) -> Result<()> {
    match op {
        // Binary payloads are 0/1: circular GEQ 1 (modern 0x1003 / legacy
        // ACQ_GEQ) passes exactly on 1, the same encoding gates always used.
        SemaphoreOp::AcquireBinary(address) => {
            host_semaphore(words, channel, false, address, 1, true, false)
        }
        SemaphoreOp::AcquireTimeline(address, value) => {
            host_semaphore(words, channel, native64, address, value, true, false)
        }
        SemaphoreOp::ReleaseBinary(address) => {
            host_semaphore(words, channel, false, address, 1, false, false)
        }
        SemaphoreOp::ReleaseTimeline(address, value) => {
            host_semaphore(words, channel, native64, address, value, false, false)
        }
    }
}

const QUEUE_ENTRIES: usize = 1024;
/// Per-batch command slot: [wait prefix 256 B][signals+completion ... | IRQ 8 B].
/// Typical frames (<= 10 waits, <= 7 signals) need no extra allocation; larger
/// batches put their prefix/signals in a retained overflow segment.
const TAIL_STRIDE: usize = 512;
const TAIL_PREFIX_BYTES: usize = 256;
const TAIL_IRQ_BYTES: usize = 8;

/// Public host NON_STALL_INTERRUPT, common to the validated channel classes.
/// Keep these words immutable: CPU retirement may precede their GPU fetch.
fn completion_interrupt(channel: u32) -> Result<[u32; 2]> {
    modern_host(channel)?;
    let mut words = Vec::new();
    method(&mut words, 0, 0x20, &[0]);
    Ok([words[0], words[1]])
}

/// Timeline release (WFI, 64-bit payload when `native64`) followed by the
/// host NON_STALL_INTERRUPT: the same pair a Queue uses for its public
/// completion word, for channels that are not Queues (NVDEC).
pub(crate) fn timeline_release(
    words: &mut Vec<u32>,
    channel: u32,
    native64: bool,
    address: u64,
    value: u64,
) -> Result<()> {
    host_semaphore(words, channel, native64, address, value, false, true)?;
    words.extend_from_slice(&completion_interrupt(channel)?);
    Ok(())
}

/// Right-align mutable methods before the slot's immutable interrupt suffix.
fn completion_tail_offset(words: usize) -> Result<usize> {
    let bytes = words
        .checked_mul(4)
        .ok_or_else(|| Error::Contract("host command size overflow".into()))?;
    let offset = (TAIL_STRIDE - TAIL_IRQ_BYTES)
        .checked_sub(bytes)
        .filter(|offset| *offset >= TAIL_PREFIX_BYTES)
        .ok_or_else(|| Error::Contract("host methods exceed reserved command slot".into()))?;
    Ok(offset)
}

/// Visibility + signal releases for a batch (empty signal list: visibility only).
fn signal_methods(
    words: &mut Vec<u32>,
    channel: u32,
    native64: bool,
    signals: &[SemaphoreOp],
) -> Result<()> {
    host_visibility(words, channel)?;
    for op in signals {
        encode_op(words, channel, native64, *op)?;
    }
    Ok(())
}

/// Public completion (SEMSURF payload, timestamped) then private retirement.
/// No mutable methods follow the retirement release; the IRQ is the immutable
/// suffix of the same GPFIFO entry.
fn final_methods(
    words: &mut Vec<u32>,
    channel: u32,
    native64: bool,
    public_address: u64,
    retirement_address: u64,
    token: u64,
) -> Result<()> {
    host_semaphore(words, channel, native64, public_address, token, false, true)?;
    host_semaphore(
        words,
        channel,
        native64,
        retirement_address,
        token,
        false,
        false,
    )
}

#[cfg(test)]
fn completion_tail(
    channel: u32,
    native64: bool,
    public_address: u64,
    retirement_address: u64,
    token: u64,
    signal_address: Option<u64>,
) -> Result<Vec<u32>> {
    let mut words = Vec::new();
    let signals: Vec<_> = signal_address
        .map(SemaphoreOp::ReleaseBinary)
        .into_iter()
        .collect();
    signal_methods(&mut words, channel, native64, &signals)?;
    final_methods(
        &mut words,
        channel,
        native64,
        public_address,
        retirement_address,
        token,
    )?;
    completion_tail_offset(words.len())?;
    Ok(words)
}

struct PendingSubmission {
    token: u64,
    entries: usize,
    slot: usize,
}
#[derive(Default)]
struct SubmissionLedger {
    sequence: u64,
    retired: u64,
    used_entries: usize,
    pending: VecDeque<PendingSubmission>,
    /// Overflow command segments retained until their batch retires (rare).
    overflow: VecDeque<(u64, rm::Mapping, GpuMapping)>,
}
impl SubmissionLedger {
    fn prepare(
        &self,
        entries: usize,
        waits: usize,
    ) -> Result<std::result::Result<(u64, usize), u64>> {
        if entries == 0 || entries >= QUEUE_ENTRIES || waits > 1024 {
            return Err(Error::Contract("batch exceeds queue capacity".into()));
        }
        // Leave one entry unused: PUT==GET must never describe a full ring.
        if self.used_entries + entries >= QUEUE_ENTRIES {
            return Ok(Err(self.pending.front().unwrap().token));
        }
        let token = self
            .sequence
            .checked_add(waits as u64)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::Contract("GPU completion sequence exhausted".into()))?;
        let slot = self
            .pending
            .back()
            .map_or(0, |p| (p.slot + 1) % QUEUE_ENTRIES);
        Ok(Ok((token, slot)))
    }
    fn prepare_for_mode(
        &self,
        entries: usize,
        waits: usize,
        native64: bool,
    ) -> Result<std::result::Result<(u64, usize), u64>> {
        let next = self.prepare(entries, waits)?;
        let Ok((token, _)) = next else {
            return Ok(next);
        };
        if !native64 {
            let wraps = (token >> 32) != (self.sequence >> 32);
            let too_far = token - self.retired > 0x7fff_ffff;
            if (wraps || too_far) && !self.pending.is_empty() {
                return Ok(Err(self.pending.front().unwrap().token));
            }
            if too_far {
                return Err(Error::Contract(
                    "sem32 submission exceeds half range".into(),
                ));
            }
        }
        Ok(next)
    }
    fn commit(&mut self, token: u64, entries: usize, slot: usize) {
        self.sequence = token;
        self.used_entries += entries;
        self.pending.push_back(PendingSubmission {
            token,
            entries,
            slot,
        });
    }
    fn retire(&mut self, completed: u64) -> Result<()> {
        if completed > self.sequence || completed < self.retired {
            return Err(Error::Contract(
                "GPU completion outside issued timeline".into(),
            ));
        }
        // Intermediate imported-wait values never retire a batch's storage.
        while self.pending.front().is_some_and(|p| p.token <= completed) {
            let p = self.pending.pop_front().unwrap();
            self.used_entries -= p.entries;
            self.retired = p.token;
        }
        while self
            .overflow
            .front()
            .is_some_and(|(token, _, _)| *token <= self.retired)
        {
            self.overflow.pop_front();
        }
        Ok(())
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SubmitStatus {
    Submitted(u64),
    Full(u64),
}
/// NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_ZCULL.
pub(crate) const ENGINE_GRAPHICS_ZCULL: u32 = 8;
/// NV2080_CTRL_CTXSW_ZCULL_MODE_SEPARATE_BUFFER.
const ZCULL_MODE_SEPARATE_BUFFER: u32 = 2;

/// ZCULL context-switch storage for one graphics channel, matching what the
/// stock driver does for every graphics context on every generation (stock
/// 615 G 0xae4760 E 0x9e726c..0x9e7412 + allocator G 0xae02c0,
/// 580 E a75a90 / a79b70; runtime trace `ctrl 0x801707 {8} -> {8, 0x1000,
/// 0x1b9600}` then `ctrl 0x20801208 {client, channel, VA, 2}` on GB202):
/// size = NV0080 FIFO_GET_ENGINE_CONTEXT_PROPERTIES(GRAPHICS_ZCULL).size on
/// the device (the returned alignment is not used), VRAM, mapped into the
/// channel's VA space, then NV2080 GR_CTXSW_ZCULL_BIND {hClient, hChannel =
/// the channel (not the TSG), vMemPtr, SEPARATE_BUFFER} on the subdevice.
/// Any failure fails channel creation (stock takes its error path). Stock
/// shares one device buffer across a device's graphics channels and keeps
/// it coherent with share walks that are not traced; each of our TSGs (one
/// graphics channel each) owns its buffer, which is stock's own unshared
/// branch (E 0x9e7468), and keeps it until RM freed the TSG.
fn bind_zcull(device: &Rc<Device>, space: &AddressSpace, channel: u32) -> Result<GpuMapping> {
    let mut properties = abi::EngineContextProperties {
        engine: ENGINE_GRAPHICS_ZCULL,
        ..Default::default()
    };
    device
        .client
        .control(device.handle, 0x801707, &mut properties)?;
    if properties.size == 0 {
        return Err(Error::Contract(
            "RM reports no GRAPHICS_ZCULL context size; graphics channel cannot bind ZCULL".into(),
        ));
    }
    let memory = device.allocate_vram(properties.size as usize)?;
    let mapping = space.map(&memory)?;
    if mapping
        .address()
        .checked_add(mapping.length())
        .is_none_or(|end| end >= 16 << 30)
    {
        return Err(Error::Contract(
            "internal GPU allocation collides with Mesa VA heap".into(),
        ));
    }
    let mut bind = abi::ZcullBind {
        client: device.client.root,
        channel,
        address: mapping.address(),
        mode: ZCULL_MODE_SEPARATE_BUFFER,
        padding: 0,
    };
    device
        .client
        .control(device.subdevice, 0x20801208, &mut bind)?;
    Ok(mapping)
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Push {
    pub address: u64,
    pub size: u32,
    pub flags: u32,
}
impl Queue {
    pub(crate) fn new_with_events(
        device: &Rc<Device>,
        space: &AddressSpace,
        mask: u32,
        events: &crate::events::RmEvents,
        pool: &Arc<SlotPool>,
        retire: Slot,
    ) -> Result<Self> {
        Self::create(device, space, mask, events, pool, retire, None)
    }
    /// A queue whose channel joins `share`'s TSG on the lowest free
    /// subcontext (VEID) instead of getting a TSG of its own. Channels of one
    /// TSG on different VEIDs run on the GR engine at the same time. This is
    /// how stock 615 places a compute-only queue: its channel goes into the
    /// graphics queue's TSG on its own VEID (ctxshare SPECIFIED, engine
    /// GRAPHICS, compute object only; libnvidia-eglcore.so.615.78.08
    /// E 0x9e5f35..0x9e5f6c, E 0x9e0170, E 0x9e17b0).
    pub(crate) fn new_in_group(
        device: &Rc<Device>,
        space: &AddressSpace,
        mask: u32,
        events: &crate::events::RmEvents,
        pool: &Arc<SlotPool>,
        retire: Slot,
        share: &Queue,
    ) -> Result<Self> {
        Self::create(device, space, mask, events, pool, retire, Some(share))
    }
    fn create(
        device: &Rc<Device>,
        space: &AddressSpace,
        mask: u32,
        events: &crate::events::RmEvents,
        pool: &Arc<SlotPool>,
        retire: Slot,
        share: Option<&Queue>,
    ) -> Result<Self> {
        // On error the caller still owns `retire`; on success QueueShared does.
        if mask == 0 || mask & !31 != 0 {
            return Err(Error::Contract("unsupported GPU engine mask".into()));
        }
        let caps = crate::capabilities::query(device)?;
        let modern = caps.usermode.is_some();
        if let Some(share) = share {
            // Stock's subcontext channels carry only the compute class (its
            // async-compute and compute-queue channels get one 0xcec0 object
            // on GB202, no 3D, 2D or copy object); 3D runs only on VEID 0.
            // Before Volta the joining channel shares the group's GR context
            // with the graphics channel, and it too carries the compute class
            // only (stock 580.178.04 on GP10x).
            if mask != 16 {
                return Err(Error::Contract(format!(
                    "a context in a shared TSG supports only the compute engine (mask 0x10), got {mask:#x}"
                )));
            }
            if modern {
                device.client.compat.require_constant(
                    "NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SPECIFIED",
                    CTXSHARE_SPECIFIED.into(),
                )?;
            }
            if !Rc::ptr_eq(&share.subcontext.tsg.object.device, device) {
                return Err(Error::Contract(
                    "a shared TSG belongs to another RM device".into(),
                ));
            }
        }
        // Public NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS.hObjectError accepts
        // a physical Memory handle. GSP requires FB backing for this notifier.
        // NvNotification is 16 bytes, error index 0, status at byte 14.
        let error_mem = device.allocate_vram(4096)?;
        let mut error_notifier = error_mem.map()?;
        for offset in (0..48).step_by(4) {
            error_notifier.write_u32(offset, 0)?;
        }
        let ring_mem = device.allocate_aligned(65536, 4096, true, !modern)?;
        let ring_gpu = space.map(&ring_mem)?;
        let mut ring = ring_mem.map()?;
        for i in 0..65536 / 4 {
            ring.write_u32(i * 4, 0)?;
        }
        // CPU rewrites the release sequence between submissions. Like Mesa's
        // command streams, this is CPU-cached/GPU-uncached system memory.
        let tail_mem = device.allocate_aligned(QUEUE_ENTRIES * TAIL_STRIDE, 4096, false, true)?;
        let tail_gpu = space.map(&tail_mem)?;
        let mut tail = tail_mem.map()?;
        let interrupt = completion_interrupt(caps.channel)?;
        for slot in 0..QUEUE_ENTRIES {
            for (word, value) in interrupt.iter().enumerate() {
                tail.write_u32(
                    slot * TAIL_STRIDE + TAIL_STRIDE - TAIL_IRQ_BYTES + word * 4,
                    *value,
                )?;
            }
        }
        for va in [&ring_gpu, &tail_gpu] {
            if va
                .address()
                .checked_add(va.length)
                .is_none_or(|end| end >= 16 << 30)
            {
                return Err(Error::Contract(
                    "internal GPU allocation collides with Mesa VA heap".into(),
                ));
            }
        }
        let tsg = match share {
            Some(share) => Rc::clone(&share.subcontext.tsg),
            None => {
                let mut group_params = abi::ChannelGroup {
                    vaspace: space.vaspace.handle,
                    engine: 1,
                    ..Default::default()
                };
                Tsg::new(Object::new(
                    device,
                    device.handle,
                    0xa06c,
                    &mut group_params,
                    vec![Rc::clone(&space.vaspace)],
                )?)
            }
        };
        // Declared before the context share and channel: on an error below
        // they drop (RM free) first and the VEID returns afterwards. Before
        // Volta there are no subcontexts and no context-share objects: a
        // channel joining the group is allocated under the TSG with
        // hContextShare 0 and runs in the group's GR context (stock
        // 580.178.04 skips the context-share allocation below Volta,
        // libnvidia-eglcore.so.580.178.04 E 0xa76fab, and still places the
        // compute queue's channel in the graphics queue's TSG, E 0xa7b167).
        let subcontext = if modern {
            Subcontext::take(tsg)?
        } else {
            Subcontext::legacy(tsg)
        };
        let veid = subcontext.veid;
        let group = Rc::clone(&subcontext.tsg.object);
        let context = if modern {
            // NV_CTXSHARE_ALLOCATION_PARAMETERS: VEID 0 = SUBCONTEXT_SYNC,
            // any other VEID = SUBCONTEXT_SPECIFIED with subctxId = VEID
            // (nvos.h NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT; stock 615
            // E 0x9e01c0..0x9e01fa: flags = (veid != 0) * 2).
            let mut parameters = abi::ContextShare {
                vaspace: space.vaspace.handle,
                flags: if veid == 0 {
                    CTXSHARE_SYNC
                } else {
                    CTXSHARE_SPECIFIED
                },
                subcontext: veid,
            };
            Some(Object::new(
                device,
                group.handle,
                0x9067,
                &mut parameters,
                vec![Rc::clone(&group), Rc::clone(&space.vaspace)],
            )?)
        } else {
            None
        };
        if let (0, Some(context)) = (veid, context.as_ref()) {
            subcontext.tsg.keep_veid0_share(Rc::clone(context))?;
        }
        let mut params = abi::Channel {
            error: error_mem.handle,
            fifo_address: ring_gpu.address(),
            entries: QUEUE_ENTRIES as u32,
            context_share: context.as_ref().map_or(0, |c| c.handle),
            // Every channel here is parented under our explicit TSG. Public
            // 535 kernel_channel.c rejects hVASpace on that parent regardless
            // of GPU generation. With hContextShare=0, SetLegacyMode creates
            // the legacy sync context and kernel_ctxshare.c inherits TSG pVAS.
            vaspace: 0,
            // A channel on a VEID other than 0 runs on TSG runqueue one and
            // SCG thread one, as stock 615 sets for every such channel
            // (E 0x9e1c70 `orl $0x10000010`; alloc_channel.h
            // NVOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE 4:4, _THREAD 29:28).
            flags: if veid == 0 {
                0
            } else {
                CHANNEL_RUNQUEUE_ONE | CHANNEL_THREAD_ONE
            },
            ..Default::default()
        };
        if modern {
            params.userd_memory[0] = ring_mem.handle;
            params.userd_offsets[0] = 8192;
        }
        let mut dependencies = vec![Rc::clone(&group), Rc::clone(&space.vaspace)];
        if let Some(context) = context {
            dependencies.push(context);
        }
        let channel = Object::new(
            device,
            group.handle,
            caps.channel,
            &mut params,
            dependencies,
        )?;
        let channel_events =
            events.attach_channel(channel.handle, caps.channel, channel.clone())?;
        let userd = if modern {
            None
        } else {
            Some(device.map_channel_userd(channel.handle, channel.clone())?)
        };
        let mut engines = Vec::new();
        for (bit, class) in [
            (4, caps.eng3d),
            (16, caps.compute),
            (1, caps.copy),
            (2, caps.eng2d),
            (8, caps.m2mf),
        ] {
            if mask & bit != 0 {
                engines.push(Object::engine(device, &channel, caps.channel, class)?);
            }
        }
        // Graphics contexts get their ZCULL context-switch buffer after the
        // engine objects, as stock does (objects, then ZCULL bind).
        // The buffer belongs to the TSG's GR context, not to this channel
        // (RM keeps one KernelGraphicsContext per channel group,
        // g_kernel_channel_group_api_nvoc.h hKernelGraphicsContext): the GR
        // context switch keeps saving and restoring it while any channel of
        // the TSG lives. It is therefore owned by the TSG and unmapped only
        // after RM freed the TSG.
        if mask & 4 != 0 {
            let zcull = bind_zcull(device, space, channel.handle)?;
            subcontext.tsg.keep_zcull(zcull)?;
        }
        let mut token = 0;
        let doorbell = if let Some(class) = caps.usermode {
            device
                .client
                .control(channel.handle, 0xc36f0108, &mut token)?;
            Some(device.map_usermode_class(class)?)
        } else {
            None
        };
        let bar1_doorbell = caps.usermode.is_some_and(|class| class >= 0xc661);
        let completion = crate::fences::NativeFenceSurface::new(device, space, channel.handle)?;
        if !modern && completion.native64() {
            return Err(Error::Contract(
                "pre-Volta channel and native64 surface capability disagree".into(),
            ));
        }
        // Retirement follows the public completion surface's width (as before).
        // Last fallible step before the slot reference moves into QueueShared.
        if share.is_none() {
            device
                .client
                .control(group.handle, 0xa06c0101, &mut [1u8, 0, 0])?;
        } else {
            // The TSG is already scheduled: RM put this channel on the TSG's
            // runlist at allocation (kernel_channel.c:926). Bind and schedule
            // the channel itself, as stock does for every channel
            // (NVA06F_CTRL_CMD_BIND {NV2080_ENGINE_TYPE_GRAPHICS}, then
            // NVA06F_CTRL_CMD_GPFIFO_SCHEDULE {bEnable 1}; stock 615
            // E 0x9e19c5 / E 0x9e1a01; kchannelCtrlCmdBind_IMPL covers a
            // channel added to a running TSG).
            device
                .client
                .control(channel.handle, 0xa06f0104, &mut 1u32)?;
            device
                .client
                .control(channel.handle, 0xa06f0103, &mut [1u8, 0, 0])?;
        }
        let shared = Arc::new(QueueShared {
            pool: Arc::clone(pool),
            retire,
            native64: completion.native64(),
            sequence: AtomicU64::new(0),
            lost: AtomicBool::new(false),
        });
        Ok(Self {
            _events: channel_events,
            userd,
            channel_class: caps.channel,
            engines,
            channel,
            subcontext,
            compute_only: mask & 16 != 0 && mask & 4 == 0,
            error_notifier,
            ring,
            _ring_gpu: ring_gpu,
            completion,
            tail,
            tail_gpu,
            doorbell,
            bar1_doorbell,
            token,
            put: 0,
            shared,
            lost: false,
            stopped: false,
            ledger: SubmissionLedger::default(),
            scratch: Scratch {
                prefix: Vec::with_capacity(TAIL_PREFIX_BYTES / 4),
                signals: Vec::with_capacity(TAIL_STRIDE / 4),
                tail: Vec::with_capacity(TAIL_STRIDE / 4),
            },
            timing: None,
            held: HeldSlots {
                pool: Arc::clone(pool),
                slots: VecDeque::with_capacity(4 * QUEUE_ENTRIES),
            },
        })
    }
    pub(crate) fn shared(&self) -> &Arc<QueueShared> {
        &self.shared
    }
    /// The channel's subcontext ID and its TSG's RM handle.
    #[cfg(test)]
    pub(crate) fn subcontext(&self) -> (u32, u32) {
        (self.subcontext.veid, self.subcontext.tsg.object.handle)
    }
    /// GPU VA range of the TSG's ZCULL buffer, if bound.
    #[cfg(test)]
    pub(crate) fn zcull_range(&self) -> Option<(u64, u64)> {
        self.subcontext
            .tsg
            .zcull
            .borrow()
            .as_ref()
            .map(|m| (m.address(), m.length()))
    }
    pub(crate) fn channel_class(&self) -> u32 {
        self.channel_class
    }
    /// Perf capture (capture.rs): add a timestamped host release at the start
    /// (after acquires, no WFI) and end (after the tail's SET_REFERENCE WFI +
    /// membar, RELEASE_WFI) of every later submission. Volta+ host only.
    pub(crate) fn enable_timing(&mut self, space: &AddressSpace) -> Result<()> {
        if !modern_host(self.channel_class)? {
            return Err(Error::Contract(
                "perf capture timestamps need a Volta+ host class (SEM_EXECUTE)".into(),
            ));
        }
        let bytes = crate::capture::TS_ENTRIES * 32;
        let mem = self
            .subcontext
            .tsg
            .object
            .device
            .allocate_aligned(bytes, 4096, false, true)?;
        let mut cpu = mem.map()?;
        for i in 0..bytes / 4 {
            cpu.write_u32(i * 4, 0)?;
        }
        let gpu = space.map(&mem)?;
        self.timing = Some(crate::capture::TimingRing {
            cpu,
            gpu,
            next: 1,
            pending: VecDeque::new(),
        });
        Ok(())
    }
    /// Completed (token, start_ns, end_ns) records since the last call.
    pub(crate) fn drain_timing(&mut self, out: &mut Vec<(u64, u64, u64)>) {
        if let Some(t) = self.timing.as_mut() {
            t.drain(out);
        }
    }
    pub(crate) fn last_token(&self) -> u64 {
        self.ledger.sequence
    }
    /// Read the authoritative RC notifier (VRAM). Called by the throttled loss
    /// scan (<= 1/s, or forced by RC readiness / failed submission), never per
    /// ordinary submission or status query.
    pub(crate) fn check_error(&mut self) -> Result<()> {
        if self.lost {
            return Err(Error::Contract("persistent RM channel is lost".into()));
        }
        crate::platform::device_read_barrier();
        if self.error_notifier.read_u32(12)? >> 16 == 0xffff {
            crate::platform::device_read_barrier();
            self.mark_lost();
            // Whole NvNotification (timeStamp @0, info32 @8, info16 @12,
            // status @14). RM puts the RC error type in info32; info16 and
            // timeStamp carry whatever its error path provides. On rented
            // hosts without dmesg this is the only per-channel fault record.
            let word12 = self.error_notifier.read_u32(12)?;
            let stamp = u64::from(self.error_notifier.read_u32(0)?)
                | (u64::from(self.error_notifier.read_u32(4)?) << 32);
            return Err(Error::Contract(format!(
                "GPU channel error {} at PUT={} sequence={}; device lost \
                 (host class {:#x}, notifier info16={:#06x} status={:#06x} timeStamp={stamp:#x})",
                self.error_notifier.read_u32(8)?,
                self.put,
                self.ledger.sequence,
                self.channel_class,
                word12 & 0xffff,
                word12 >> 16
            )));
        }
        Ok(())
    }
    fn mark_lost(&mut self) {
        self.lost = true;
        self.shared.lost.store(true, Ordering::Release);
    }
    /// SYS-memory-only retirement: pops retired batches, frees their ring
    /// entries, overflow segments and semaphore-slot references. Amortized on
    /// the submitting caller (stock retires temporaries at end of submit too).
    pub(crate) fn retire_completion(&mut self) -> Result<()> {
        if self.lost {
            return Err(Error::Contract("persistent RM channel is lost".into()));
        }
        if self.ledger.pending.is_empty() {
            return Ok(());
        }
        crate::platform::device_read_barrier();
        let completed = self.shared.completed()?;
        // Check the public semaphore too, catching impossible notification vs
        // retirement ordering rather than silently accepting a corrupt value.
        if completed > self.completion.read_completed()? {
            self.mark_lost();
            return Err(Error::Contract(
                "retirement preceded public completion".into(),
            ));
        }
        if let Err(e) = self.ledger.retire(completed) {
            self.mark_lost();
            return Err(e);
        }
        self.held.release_through(self.ledger.retired);
        crate::platform::device_full_barrier();
        Ok(())
    }
    /// Refuse further submissions on this channel (device loss). CPU state
    /// only: like stock (580 G b78280, 615 vkDestroyDevice), no TSG/channel
    /// schedule-off, preempt or disable control is ever sent; RM frees the
    /// channel directly when it is dropped.
    pub(crate) fn stop_submissions(&mut self) {
        self.stopped = true;
    }
    /// A sync_file that signals when `token` completes on this channel.
    ///
    /// # Safety
    /// `drm_fd` is negative (rejected) or an open descriptor for the whole call.
    pub(crate) unsafe fn context_fence(&self, drm_fd: i32, token: u64) -> Result<OwnedFd> {
        if self.lost || token == 0 || token > self.ledger.sequence {
            return Err(Error::Contract("invalid/lost RM completion token".into()));
        }
        // SAFETY: `drm_fd` is this function's, under its contract.
        unsafe { self.completion.context_fence(drm_fd, token) }
    }
    /// Write `words` into a dedicated retained SYS segment (rare overflow).
    fn overflow_segment(
        &mut self,
        space: &AddressSpace,
        words: &[u32],
    ) -> Result<(rm::Mapping, GpuMapping)> {
        let bytes = (words.len() * 4).next_multiple_of(4096);
        let mem = self
            .subcontext
            .tsg
            .object
            .device
            .allocate_aligned(bytes, 4096, false, true)?;
        let mut cpu = mem.map()?;
        let gpu = space.map(&mem)?;
        for (i, word) in words.iter().enumerate() {
            cpu.write_u32(i * 4, *word)?;
        }
        if gpu
            .address()
            .checked_add(bytes as u64)
            .is_none_or(|end| end >= 1 << 34)
        {
            return Err(Error::Contract(
                "GPU dependency commands collide with Mesa VA heap".into(),
            ));
        }
        Ok((cpu, gpu))
    }
    fn write_entry(&mut self, address: u64, bytes: usize, sync_wait: bool) -> Result<()> {
        // GP_ENTRY0/1 (public clc96f.h et al.): GET in [39:2], LEVEL bit 41,
        // LENGTH (words) in [62:42], SYNC_WAIT bit 63 — the existing encoding.
        let entry =
            address | ((bytes as u64 / 4) << 42) | (1 << 41) | if sync_wait { 1 << 63 } else { 0 };
        self.ring.write_u32(self.put * 8, entry as u32)?;
        self.ring
            .write_u32(self.put * 8 + 4, (entry >> 32) as u32)?;
        self.put = (self.put + 1) % QUEUE_ENTRIES;
        Ok(())
    }
    /// Waits, work and signals in one GPFIFO segment with one doorbell.
    ///
    /// `held` slots gain one reference each for the accepted batch and are
    /// released when its private retirement word passes. On Full nothing is
    /// written and no reference is taken. `before_kick(token)` runs after the
    /// batch is accepted and fully written, before the GPU can observe it
    /// (split-view shadows are raised there).
    pub(crate) fn submit(
        &mut self,
        pushes: &[Push],
        space: &AddressSpace,
        waits: &[SemaphoreOp],
        signals: &[SemaphoreOp],
        held: &[Slot],
        before_kick: impl FnOnce(u64) -> Result<()>,
    ) -> Result<SubmitStatus> {
        if self.stopped {
            return Err(Error::Contract(
                "cannot submit to a stopped RM channel".into(),
            ));
        }
        if pushes.len() > 1000
            || pushes.last().is_some_and(|p| p.flags & 2 != 0)
            || waits.len() > 1024
            || signals.len() > 65536
        {
            return Err(Error::Contract(
                "invalid/incomplete GPFIFO batch or waits".into(),
            ));
        }
        for p in pushes {
            if p.address & 3 != 0
                || p.address >= 1 << 40
                || p.address
                    .checked_add(u64::from(p.size))
                    .is_none_or(|end| end > (1u64 << 40))
                || p.size == 0
                || p.size & 3 != 0
                || p.size >= 1 << 23
                || p.flags & !3 != 0
            {
                return Err(Error::Contract(
                    "GPFIFO push outside hardware limits".into(),
                ));
            }
        }
        self.retire_completion()?;
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.submit_encoded(
            pushes,
            space,
            waits,
            signals,
            held,
            before_kick,
            &mut scratch,
        );
        self.scratch = scratch;
        if result.is_err() {
            self.mark_lost();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn submit_encoded(
        &mut self,
        pushes: &[Push],
        space: &AddressSpace,
        waits: &[SemaphoreOp],
        signals: &[SemaphoreOp],
        held: &[Slot],
        before_kick: impl FnOnce(u64) -> Result<()>,
        scratch: &mut Scratch,
    ) -> Result<SubmitStatus> {
        let native64 = self.completion.native64();
        let slot_native64 = self.shared.pool.view == crate::slots::TimelineView::Native64;
        scratch.prefix.clear();
        scratch.signals.clear();
        scratch.tail.clear();
        for op in waits {
            encode_op(&mut scratch.prefix, self.channel_class, slot_native64, *op)?;
        }
        if self.compute_only {
            // A channel with the compute class alone completes a batch the
            // way the proprietary driver completes its compute queue's
            // (615.78.08: compute-class WAIT_FOR_IDLE before the completion
            // release): the host WFI and RELEASE_WFI below do not wait for
            // compute grids launched on such a channel, and the 3D-class
            // subchannel switch that idles GR on a graphics channel does not
            // happen there. Measured on GB202 without it: the release came
            // first and a later grid wrote freed memory (Xid 31).
            // NVC7C0_WAIT_FOR_IDLE 0x0110 on the compute subchannel 1
            // (clc7c0.h; NVK binds the compute class there).
            method(&mut scratch.signals, 1, 0x110, &[0]);
        }
        signal_methods(
            &mut scratch.signals,
            self.channel_class,
            slot_native64,
            signals,
        )?;
        if let Some(t) = self.timing.as_ref() {
            // Perf capture: SEM_EXECUTE RELEASE | RELEASE_TIMESTAMP (bit 25),
            // 32-bit payload; start without RELEASE_WFI (host arrival time),
            // end through host_semaphore (RELEASE_WFI) after the tail's WFI.
            let address = t.entry_address(t.next);
            method(
                &mut scratch.prefix,
                0,
                0x5c,
                &[
                    address as u32,
                    (address >> 32) as u32,
                    t.next,
                    0,
                    1 | (1 << 25),
                ],
            );
            host_semaphore(
                &mut scratch.signals,
                self.channel_class,
                false,
                address + 16,
                u64::from(t.next),
                false,
                true,
            )?;
        }
        // Fixed-size completion methods (token only changes payload words).
        final_methods(&mut scratch.tail, self.channel_class, native64, 0, 0, 0)?;
        let final_words = scratch.tail.len();
        scratch.tail.clear();
        let prefix_inline = scratch.prefix.len() * 4 <= TAIL_PREFIX_BYTES;
        let signals_inline = (scratch.signals.len() + final_words) * 4
            <= TAIL_STRIDE - TAIL_IRQ_BYTES - TAIL_PREFIX_BYTES;
        let entries = pushes.len()
            + 1
            + usize::from(!scratch.prefix.is_empty())
            + usize::from(!signals_inline);
        let (token, slot) = match self.ledger.prepare_for_mode(entries, 0, native64)? {
            Ok(next) => next,
            Err(oldest) => return Ok(SubmitStatus::Full(oldest)),
        };
        let base = slot * TAIL_STRIDE;
        let prefix_segment = if !scratch.prefix.is_empty() && !prefix_inline {
            Some(self.overflow_segment(space, &scratch.prefix)?)
        } else {
            None
        };
        let signal_segment = if signals_inline {
            scratch.tail.extend_from_slice(&scratch.signals);
            None
        } else {
            Some(self.overflow_segment(space, &scratch.signals)?)
        };
        final_methods(
            &mut scratch.tail,
            self.channel_class,
            native64,
            self.completion.address(),
            self.shared.pool.gpu(self.shared.retire),
            token,
        )?;
        let tail_offset = completion_tail_offset(scratch.tail.len())?;
        if prefix_inline {
            for (i, word) in scratch.prefix.iter().enumerate() {
                self.tail.write_u32(base + i * 4, *word)?;
            }
        }
        for (i, word) in scratch.tail.iter().enumerate() {
            self.tail.write_u32(base + tail_offset + i * 4, *word)?;
        }
        // Accepted from here on (Full was the last non-error outcome).
        before_kick(token)?;
        // Publish the sem32 high-word tracker and the waiter-visible sequence
        // before the GPU can observe new work.
        self.completion.publish_maximum(token)?;
        self.shared.sequence.store(token, Ordering::Release);
        self.ledger.commit(token, entries, slot);
        if let Some(t) = self.timing.as_mut() {
            t.pending.push_back((t.next, token));
            t.next = t.next.wrapping_add(1).max(1);
        }
        for &held_slot in held {
            self.shared.pool.retain(held_slot);
            self.held.slots.push_back((token, held_slot));
        }
        if !scratch.prefix.is_empty() {
            let address = match &prefix_segment {
                Some((_, gpu)) => gpu.address(),
                None => self.tail_gpu.address() + base as u64,
            };
            self.write_entry(address, scratch.prefix.len() * 4, true)?;
        }
        for p in pushes {
            self.write_entry(p.address, p.size as usize, p.flags & 1 != 0)?;
        }
        if let Some((_, gpu)) = &signal_segment {
            let address = gpu.address();
            self.write_entry(address, scratch.signals.len() * 4, true)?;
        }
        let post = self.tail_gpu.address() + (base + tail_offset) as u64;
        self.write_entry(post, scratch.tail.len() * 4 + TAIL_IRQ_BYTES, true)?;
        for (cpu, gpu) in [prefix_segment, signal_segment].into_iter().flatten() {
            self.ledger.overflow.push_back((token, cpu, gpu));
        }
        self.kick()?;
        Ok(SubmitStatus::Submitted(token))
    }
    /// Publish GP_PUT and ring the doorbell.
    ///
    /// Volta+ (usermode doorbell): GPFIFO entries and GP_PUT (USERD at ring
    /// offset 8192) are VRAM written through BAR1 (write-combined). The
    /// doorbell must not reach the GPU host before the GP_PUT write is
    /// visible, or the host sees no new work and an idle queue's last batch
    /// is stranded until some later doorbell (measured: with a BAR0 doorbell
    /// and no flush, a batch stayed unexecuted > 2.1 s and ran 22 us after the
    /// next doorbell; build/fence-tracing/deferred-reclaim/kick-test-*).
    /// * Hopper+ (bar1_doorbell): the doorbell page is mapped through BAR1
    ///   (rm.rs map_usermode_class), the same posted-write path as GP_PUT, so
    ///   write ordering suffices: fence, GP_PUT, fence, doorbell. This is the
    ///   stock 615 sequence (sfence; GP_PUT stores; sfence; doorbell stores,
    ///   doorbell through a BAR1 usermode page — runtime-verified, see
    ///   stock-userd/) and UVM's (uvm_turing_host.c:231-235 with the
    ///   bBar1Mapping doorbell from nv_gpu_ops.c:5646-5659).
    /// * Volta..Ada (BAR0 doorbell page only): BAR0 and BAR1 writes reach
    ///   different GPU units, so GP_PUT is read back through BAR1 first (a
    ///   non-posted read completes only after earlier posted BAR1 writes
    ///   reached the GPU; UVM's flush, uvm_turing_host.c:224-229) and checked.
    ///
    /// Pre-Volta (no doorbell): writing GP_PUT to the RM USERD mapping is the
    /// kick; its read-back is retained unchanged.
    fn kick(&mut self) -> Result<()> {
        crate::platform::device_full_barrier();
        if let Some(userd) = self.userd.as_mut() {
            userd.write_u32(0x8c, self.put as u32)?;
            crate::platform::device_full_barrier();
            let _posted_write_flush = userd.read_u32(0x8c)?;
        } else {
            let put = self.put as u32;
            self.ring.write_u32(8192 + 0x8c, put)?;
            crate::platform::device_full_barrier();
            if !self.bar1_doorbell {
                let visible = self.ring.read_u32(8192 + 0x8c)?;
                if visible != put {
                    return Err(Error::Contract(format!(
                        "GP_PUT read back {visible} after writing {put}; channel state unknown"
                    )));
                }
                crate::platform::device_full_barrier();
            }
            self.doorbell
                .as_mut()
                .ok_or_else(|| Error::Contract("missing modern doorbell".into()))?
                .write_u32(0x90, self.token)?;
        }
        Ok(())
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        // Rc fields keep channel parents and all mappings alive until RM free.
        // No schedule-off before the free: stock frees channels directly
        // (stock 580 teardown order, libnvidia-glcore ELF b78280).
        // Field order closes the fence context/unbinds the semaphore while
        // the channel still exists, then frees engines/channel before buffers,
        // and finally releases held semaphore slots.
        let _ = (&self.engines, &self.channel, &self.tail, &self.held);
    }
}

#[cfg(test)]
mod subcontext_tests {
    use super::lowest_free_veid;
    #[test]
    fn veids_are_taken_lowest_first_and_returned() {
        let mut free = u64::MAX;
        for want in 0..64 {
            let (veid, rest) = lowest_free_veid(free).unwrap();
            assert_eq!(veid, want);
            free = rest;
        }
        assert_eq!(lowest_free_veid(free), None);
        // A returned VEID is the next one taken, even above a lower gap's end.
        free |= 1 << 37;
        free |= 1 << 5;
        assert_eq!(lowest_free_veid(free), Some((5, 1 << 37)));
    }
}

#[cfg(test)]
mod async_queue_tests {
    use super::*;
    #[test]
    fn private_retirement_is_last_mutable_packet_before_immutable_irq() {
        // Independent public C oracle: cla06f/cla16f/clb06f/clc06f/clc36f/
        // clc46f/clc56f NON_STALL_INTERRUPT = 0x20. Existing host_packet_tests
        // verify the release encodings against independent public C vectors.
        for channel in [0xa06f, 0xa16f, 0xb06f, 0xc06f, 0xc36f, 0xc46f, 0xc56f] {
            for signal in [None, Some(0x6000)] {
                let words = completion_tail(channel, false, 0x4000, 0x8000, 7, signal).unwrap();
                let expected: &[u32] = if modern_host(channel).unwrap() {
                    &[0x20050017, 0x8000, 0, 7, 0, 0x00100001]
                } else {
                    &[0x20040004, 0, 0x8000, 7, 0x01000002]
                };
                assert!(words.ends_with(expected));
                assert!(!words.contains(&0x20010008));
                assert_eq!(completion_interrupt(channel).unwrap(), [0x20010008, 0]);
            }
        }
    }
    #[test]
    fn reusing_mutable_tail_never_rewrites_irq_or_dependency_prefix() {
        let irq_offset = (TAIL_STRIDE - TAIL_IRQ_BYTES) / 4;
        let mut slot = [0xdead_beef; TAIL_STRIDE / 4];
        slot[irq_offset..].copy_from_slice(&completion_interrupt(0xc36f).unwrap());
        for token in 1..=4 {
            let words = completion_tail(
                0xc36f,
                true,
                0x4000,
                0x8000,
                token,
                (token % 2 == 0).then_some(0x6000),
            )
            .unwrap();
            let offset = completion_tail_offset(words.len()).unwrap() / 4;
            assert!(offset >= TAIL_PREFIX_BYTES / 4);
            slot[offset..irq_offset].copy_from_slice(&words);
            assert_eq!(
                &slot[..TAIL_PREFIX_BYTES / 4],
                &[0xdead_beef; TAIL_PREFIX_BYTES / 4]
            );
            assert_eq!(&slot[irq_offset..], &[0x20010008, 0]);
            // The submitted entry spans exactly the private release and IRQ;
            // no stale variable-length tail or gap is executed after it.
            assert_eq!(offset + words.len() + TAIL_IRQ_BYTES / 4, TAIL_STRIDE / 4);
        }
        assert!(completion_tail_offset(usize::MAX).is_err());
        assert!(
            completion_tail_offset((TAIL_STRIDE - TAIL_PREFIX_BYTES - TAIL_IRQ_BYTES) / 4 + 1)
                .is_err()
        );
    }
    #[test]
    fn sem32_wrap_drains_before_reusing_epoch_without_mutation() {
        let mut ledger = SubmissionLedger {
            sequence: 0xffff_fffe,
            retired: 0xffff_fffc,
            ..Default::default()
        };
        ledger.commit(0xffff_fffe, 3, 9);
        assert_eq!(
            ledger.prepare_for_mode(2, 3, false).unwrap(),
            Err(0xffff_fffe)
        );
        assert_eq!(ledger.sequence, 0xffff_fffe);
        assert_eq!(ledger.used_entries, 3);
        ledger.retire(0xffff_fffe).unwrap();
        assert_eq!(
            ledger.prepare_for_mode(2, 3, false).unwrap(),
            Ok((0x1_0000_0002, 0))
        );
        // A native64 queue has no sem32 epoch restriction.
        ledger.commit(0xffff_ffff, 3, 10);
        assert!(ledger.prepare_for_mode(2, 3, true).unwrap().is_ok());
        assert!(ledger.prepare_for_mode(2, usize::MAX, false).is_err());
    }
    #[test]
    fn imported_dependency_milestones_do_not_retire_storage() {
        let mut l = SubmissionLedger::default();
        let (token, slot) = l.prepare(19, 3).unwrap().unwrap();
        assert_eq!(token, 4);
        l.commit(token, 19, slot);
        for value in 0..4 {
            l.retire(value).unwrap();
            assert_eq!(l.used_entries, 19);
            assert_eq!(l.retired, 0);
        }
        l.retire(4).unwrap();
        assert_eq!(l.used_entries, 0);
        assert_eq!(l.retired, 4);
        assert!(l.retire(5).is_err());
        assert!(l.retire(3).is_err());
    }
    #[test]
    fn full_ring_returns_oldest_without_mutation_and_wraps_without_aliases() {
        use std::collections::HashSet;
        let mut l = SubmissionLedger::default();
        let mut live = HashSet::new();
        for _ in 0..1023 {
            let (token, slot) = l.prepare(1, 0).unwrap().unwrap();
            assert!(live.insert(slot));
            l.commit(token, 1, slot);
        }
        for _ in 0..4096 {
            let old = l.pending.front().unwrap().token;
            let old_slot = l.pending.front().unwrap().slot;
            let seq = l.sequence;
            assert_eq!(l.prepare(1, 0).unwrap(), Err(old));
            assert_eq!(l.sequence, seq);
            assert_eq!(l.used_entries, 1023);
            l.retire(old).unwrap();
            assert!(live.remove(&old_slot));
            let (token, slot) = l.prepare(1, 0).unwrap().unwrap();
            assert!(live.insert(slot), "live tail overwritten after ring wrap");
            l.commit(token, 1, slot);
        }
        l.retire(l.sequence).unwrap();
        assert_eq!(l.used_entries, 0);
        assert!(l.pending.is_empty());
    }
    #[test]
    fn batch_credit_and_u64_exhaustion_are_checked_before_commit() {
        let mut l = SubmissionLedger::default();
        let (token, slot) = l.prepare(1002, 1024).unwrap().unwrap();
        l.commit(token, 1002, slot);
        assert!(l.prepare(21, 0).unwrap().is_ok());
        assert_eq!(l.prepare(22, 0).unwrap(), Err(token));
        assert!(l.prepare(1024, 0).is_err());
        l.retire(token).unwrap();
        l.sequence = u64::MAX - 1;
        assert!(l.prepare(2, 1).is_err());
        assert!(l.pending.is_empty());
        assert_eq!(l.used_entries, 0);
    }
}

#[cfg(test)]
#[path = "host_packet_tests.rs"]
mod host_packet_tests;

#[cfg(test)]
mod compression_tests {
    use super::*;
    #[test]
    fn compressible_kind_kept_only_on_compressed_memory() {
        assert_eq!(effective_kind(Some(8), 6, true), Some(8));
        assert_eq!(effective_kind(Some(8), 6, false), Some(6));
        assert_eq!(effective_kind(Some(6), 6, true), Some(6));
        assert_eq!(effective_kind(Some(0), 6, true), Some(0));
        // Pre-Turing kind numbering is never rewritten.
        assert_eq!(effective_kind(Some(8), 0xfe, false), Some(8));
    }
    #[test]
    fn stock_compressible_request_attr() {
        // Stock 615 RM requests: 64 KiB/8 MiB 0x01002000, >= 16 MiB 0x09002000.
        assert_eq!(crate::rm::compressible_attr(64 << 10), 0x0100_2000);
        assert_eq!(crate::rm::compressible_attr(8 << 20), 0x0100_2000);
        assert_eq!(crate::rm::compressible_attr(16 << 20), 0x0900_2000);
        assert!(crate::rm::compression_granted(0x0900_1000)); // RM answer: COMPR REQUIRED
        assert!(!crate::rm::compression_granted(0x0900_0000));
    }
    /// Pascal (generic kind 0xfe) on 550 (typed VA) vs 580 (kindOverride):
    /// the typed VA must ask RM for a block-linear kind, the 580 plan and every
    /// Turing+ plan stay byte-identical to before.
    #[test]
    fn pascal_typed_va_requests_block_linear_kind() {
        const PASCAL_GENERIC: u32 = 0xfe;
        let old = crate::compat::Compat::for_version("550.54.14").unwrap();
        let new = crate::compat::Compat::for_version("580.178.04").unwrap();
        let plan = |compat, kind, generic| {
            mapping_plan_paged(
                compat,
                0,
                1 << 16,
                1 << 24,
                Some(1 << 34),
                kind,
                false,
                generic,
            )
            .unwrap()
        };
        // 550 Pascal block-linear colour: FORMAT BLOCK_LINEAR, DEPTH 32, 4 KiB.
        let p = plan(old, Some(PASCAL_GENERIC), PASCAL_GENERIC);
        let typed = p.typed.as_ref().unwrap();
        assert_eq!(typed.attr, (1 << 23) | (2 << 16) | 4);
        assert_eq!(typed.format, PASCAL_GENERIC);
        assert_eq!(typed.attr2, 1);
        assert_eq!(typed.flags, 0x80000 | 0x100 | 0x10);
        assert_eq!(p.flags, (1 << 8) | (1 << 4) | (1 << 15) | (1 << 3));
        assert_eq!(p.wire_kind, 0);
        // 550 Pascal pitch: unchanged FORMAT PITCH request.
        let p = plan(old, Some(0), PASCAL_GENERIC);
        assert_eq!(p.typed.as_ref().unwrap().attr, 1 << 23);
        assert_eq!(p.typed.as_ref().unwrap().format, 0);
        // 580 Pascal: kindOverride, no typed VA, kind on the wire (unchanged).
        let p = plan(new, Some(PASCAL_GENERIC), PASCAL_GENERIC);
        assert!(p.typed.is_none());
        assert_eq!(p.flags, (1 << 8) | (1 << 4) | (1 << 15) | (1 << 19));
        assert_eq!(p.wire_kind, PASCAL_GENERIC);
        // 550 Turing+: generic kind 0x06 keeps the old attr (RM ignores FORMAT).
        let p = plan(old, Some(KIND_GENERIC), KIND_GENERIC);
        assert_eq!(p.typed.as_ref().unwrap().attr, 1 << 23);
        // RM's kind answer: pre-Turing must match exactly; Turing+ unchecked.
        assert!(typed_kind_check(PASCAL_GENERIC, 0xfe, 0xfe).is_ok());
        assert!(typed_kind_check(PASCAL_GENERIC, 0xfe, 0x00).is_err());
        assert!(typed_kind_check(PASCAL_GENERIC, 0x00, 0xfe).is_err());
        assert!(typed_kind_check(KIND_GENERIC, 0x08, 0x06).is_ok());
    }
    /// Pre-Turing typed VA requests per PTE kind (no kindOverride): depth and
    /// stencil kinds ask RM through TYPE_DEPTH/STENCIL + Z attributes, colour
    /// keeps the block-linear generic request, Turing+ is unchanged.
    #[test]
    fn pascal_typed_va_requests_depth_kinds() {
        const G: u32 = 0xfe;
        let bl = ATTR_PAGE_4KB | ATTR_FORMAT_BLOCK_LINEAR;
        assert_eq!(typed_va_request(Some(G), G), (TYPE_IMAGE, bl | 4));
        assert_eq!(typed_va_request(Some(0), G), (TYPE_IMAGE, 1 << 23));
        assert_eq!(typed_va_request(None, G), (TYPE_IMAGE, 1 << 23));
        // Z16: FIXED, ZS_PACKING Z16 (7), DEPTH 16.
        assert_eq!(typed_va_request(Some(0x01), G), (1, bl | (7 << 19) | 2));
        // S8Z24 (0x11): FIXED, ZS_PACKING S8Z24 (1), DEPTH 32.
        assert_eq!(typed_va_request(Some(0x11), G), (1, bl | (1 << 19) | 4));
        // Z24S8 (0x46): FIXED, ZS_PACKING Z24S8 (0), DEPTH 32.
        assert_eq!(typed_va_request(Some(0x46), G), (1, bl | 4));
        // ZF32 (0x7b): FLOAT, ZS_PACKING Z32 (2), DEPTH 32.
        assert_eq!(
            typed_va_request(Some(0x7b), G),
            (1, bl | (1 << 18) | (2 << 19) | 4)
        );
        // ZF32_X24S8 (0xc3): FLOAT, ZS_PACKING Z32_X24S8 (5), DEPTH 64.
        assert_eq!(
            typed_va_request(Some(0xc3), G),
            (1, bl | (1 << 18) | (5 << 19) | 5)
        );
        // S8 (0x2a): TYPE_DEPTH, ZS_PACKING S8 (0), DEPTH 8 (TYPE_STENCIL is refused).
        assert_eq!(typed_va_request(Some(0x2a), G), (1, bl | 1));
        // Turing+ (generic 0x06): every kind keeps the plain request.
        for k in [0x06, 0x01, 0x11, 0x46, 0x7b, 0xc3, 0x2a] {
            assert_eq!(typed_va_request(Some(k), KIND_GENERIC), (0, 1 << 23));
        }
        // Wire-level: the plan carries type and attr; the kind check stays exact.
        let old = crate::compat::Compat::for_version("550.54.14").unwrap();
        let p = mapping_plan_paged(
            old,
            0,
            1 << 16,
            1 << 24,
            Some(1 << 34),
            Some(0x11),
            false,
            G,
        )
        .unwrap();
        let typed = p.typed.as_ref().unwrap();
        assert_eq!((typed.memory_type, typed.attr), (1, bl | (1 << 19) | 4));
        assert_eq!(typed.format, 0x11);
        assert!(typed_kind_check(G, 0x11, 0x11).is_ok());
        assert!(typed_kind_check(G, 0x11, 0xfe).is_err());
        // 580+ (kindOverride): pre-Turing S8 still goes through RM's chooser,
        // every other kind stays on the kindOverride wire; Turing+ unchanged.
        let new = crate::compat::Compat::for_version("580.178.04").unwrap();
        let plan = |kind, generic| {
            mapping_plan_paged(
                new,
                0,
                1 << 16,
                1 << 24,
                Some(1 << 34),
                Some(kind),
                false,
                generic,
            )
            .unwrap()
        };
        let s8 = plan(0x2a, G);
        let typed = s8.typed.as_ref().unwrap();
        assert_eq!(
            (typed.memory_type, typed.attr, typed.format),
            (1, bl | 1, 0x2a)
        );
        assert_eq!(s8.wire_kind, 0);
        assert_eq!(s8.flags & (1 << 19), 0);
        for k in [0xfe, 0x11, 0x46, 0x01] {
            let p = plan(k, G);
            assert!(p.typed.is_none());
            assert_eq!(p.wire_kind, k);
        }
        assert!(plan(0x2a, KIND_GENERIC).typed.is_none());
    }
    /// Hardware sweep (run on rentals with `--ignored --nocapture`): which PTE
    /// kind RM chooses for a typed VA per NVOS32 type/Z attributes, then the
    /// production map path for every Maxwell/Pascal depth/stencil kind. On a
    /// pre-Turing GPU without kindOverride the five Z kinds must map.
    #[test]
    #[ignore = "requires an NVIDIA RM device; VA reservations only, no GPU commands"]
    fn typed_va_kind_sweep() {
        use std::fs::OpenOptions;
        let client = crate::rm::Client::open().unwrap();
        let card = client
            .cards()
            .unwrap()
            .into_iter()
            .find(|c| {
                c.valid != 0
                    && c.pci.vendor == 0x10de
                    && OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(format!("/dev/nvidia{}", c.minor))
                        .is_ok()
            })
            .unwrap();
        let device = client.device(card).unwrap();
        let generic = device.generic_kind().unwrap();
        let override_ = client.compat.supports_kind_override();
        println!("{{\"generic_kind\":{generic},\"kind_override\":{override_}}}");
        let space = AddressSpace::new(&device).unwrap();
        for ty in [TYPE_IMAGE, TYPE_DEPTH, TYPE_STENCIL] {
            for z_float in 0..2u32 {
                for packing in 0..8u32 {
                    for depth in [1u32, 2, 3, 4, 5] {
                        let mut params = abi::MemoryAlloc {
                            owner: client.root,
                            vaspace: space.vaspace.handle,
                            flags: 0x80000 | 0x100,
                            memory_type: ty,
                            attr: ATTR_PAGE_4KB
                                | ATTR_FORMAT_BLOCK_LINEAR
                                | (z_float << 18)
                                | attr_zs_packing(packing)
                                | depth,
                            attr2: 1,
                            size: 1 << 16,
                            alignment: 4096,
                            limit: (1 << 16) - 1,
                            ..Default::default()
                        };
                        let r = Object::new(
                            &device,
                            device.handle,
                            TYPED_VA_CLASS,
                            &mut params,
                            vec![Rc::clone(&space.vaspace)],
                        );
                        let kind = match r {
                            Ok(_) => format!("{}", params.format),
                            Err(e) => format!("\"{}\"", e.to_string().replace('"', "'")),
                        };
                        println!(
                            "{{\"sweep\":1,\"type\":{ty},\"z_float\":{z_float},\"packing\":{packing},\"depth\":{depth},\"attr\":{},\"kind\":{kind}}}",
                            params.attr
                        );
                    }
                }
            }
        }
        let memory = device
            .allocate_aligned(1 << 16, 1 << 16, false, false)
            .unwrap();
        let mut failed = Vec::new();
        for kind in [
            generic,
            0,
            KIND_Z16,
            KIND_S8Z24,
            KIND_Z24S8,
            KIND_ZF32,
            KIND_ZF32_X24S8,
            KIND_S8,
        ] {
            let r = space.map_range(&memory, 0, 1 << 16, None, kind);
            let ok = r.is_ok();
            println!(
                "{{\"map\":1,\"kind\":{kind},\"ok\":{ok},\"error\":\"{}\"}}",
                r.err()
                    .map(|e| e.to_string().replace('"', "'"))
                    .unwrap_or_default()
            );
            if !ok && kind != KIND_S8 {
                failed.push(kind);
            }
        }
        if generic != KIND_GENERIC {
            assert!(failed.is_empty(), "pre-Turing kinds refused: {failed:x?}");
        }
    }
    #[test]
    fn big_pages_only_when_fully_aligned() {
        let compat = crate::compat::Compat::for_version("615.71.09").unwrap();
        let big = |offset, length, address| {
            mapping_plan_paged(
                compat,
                offset,
                length,
                1 << 24,
                Some(address),
                Some(6),
                true,
                KIND_GENERIC,
            )
            .unwrap()
            .flags
                & (0xf << 8)
        };
        assert_eq!(big(0, 1 << 16, 1 << 34), 2 << 8);
        assert_eq!(big(4096, 1 << 16, 1 << 34), 1 << 8);
        assert_eq!(big(0, 8192, 1 << 34), 1 << 8);
        assert_eq!(big(0, 1 << 16, (1 << 34) + 4096), 1 << 8);
        let small = mapping_plan_paged(
            compat,
            0,
            1 << 16,
            1 << 24,
            Some(1 << 34),
            Some(6),
            false,
            KIND_GENERIC,
        )
        .unwrap();
        assert_eq!(small.flags & (0xf << 8), 1 << 8);
        assert!(
            mapping_plan_paged(
                compat,
                4096,
                1 << 16,
                1 << 24,
                Some(1 << 34),
                Some(8),
                true,
                KIND_GENERIC
            )
            .is_err()
        );
    }
}
