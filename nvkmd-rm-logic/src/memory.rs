// SPDX-License-Identifier: MIT
//! Allocation policy: the RM placement, size and alignment of an nvkmd memory
//! request (`nvkmd_dev_alloc_mem`, `nvkmd_dev_alloc_tiled_mem`), and which
//! tiled (DRM modifier) requests the RM backend serves.
use nv_rm_core::ByteSize;
use thiserror::Error;

/// `enum nvkmd_mem_flags` (Mesa `src/nouveau/vulkan/nvkmd/nvkmd.h:42-65`, with
/// this repo's `NVKMD_MEM_POOL_BLOCK`), as NVK passes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemFlags(u32);

impl MemFlags {
    /// `NVKMD_MEM_LOCAL`.
    pub const LOCAL: Self = Self(1 << 0);
    /// `NVKMD_MEM_GART`: system memory.
    pub const GART: Self = Self(1 << 1);
    /// `NVKMD_MEM_VRAM`.
    pub const VRAM: Self = Self(1 << 2);
    /// `NVKMD_MEM_CAN_MAP`: the CPU may map it.
    pub const CAN_MAP: Self = Self(1 << 3);
    /// `NVKMD_MEM_SHARED`: it may be exported.
    pub const SHARED: Self = Self(1 << 4);
    /// `NVKMD_MEM_COHERENT`.
    pub const COHERENT: Self = Self(1 << 5);
    /// `NVKMD_MEM_POOL_BLOCK`: one GPU-internal pool block.
    pub const POOL_BLOCK: Self = Self(1 << 6);

    /// The flags NVK passed.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
    /// The bits as NVK reads them.
    pub const fn bits(self) -> u32 {
        self.0
    }
    /// Whether any flag of `other` is set.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
    /// Both sets of flags.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// A GPU page table entry kind (`NV_MMU_PTE_KIND_*`, open-gpu-kernel-modules
/// `src/common/inc/swref/published/*/dev_mmu.h`). 0 is pitch (plain) memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PteKind(u8);

impl PteKind {
    /// Plain (pitch) memory: RM picks the layout.
    pub const PITCH: Self = Self(0);
    /// The uncompressed generic block-linear kind from Turing on
    /// (`NV_MMU_PTE_KIND_GENERIC_MEMORY`, tu102 `dev_mmu.h`).
    pub const GENERIC_TURING: Self = Self(0x06);
    /// The uncompressed generic block-linear kind before Turing
    /// (`NV_MMU_PTE_KIND_GENERIC_16BX2`, gm107 `dev_mmu.h`).
    pub const GENERIC_FERMI: Self = Self(0xfe);

    /// The kind `value`.
    pub const fn new(value: u8) -> Self {
        Self(value)
    }
    /// The kind as the RM and NVK carry it.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Where the RM places an allocation (`NV_RM_PLACEMENT_*`, `mesa/nv_rm.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Device memory.
    Vram,
    /// System memory.
    Gart,
    /// Device memory, never CPU-mapped or exported, with compression asked of RM.
    VramCompressible,
    /// One 2 MiB GPU-internal pool block.
    VramPoolBlock,
}

/// What the RM is asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    /// The placement.
    pub placement: Placement,
    /// The size, rounded as the placement needs.
    pub size: ByteSize,
    /// The alignment, raised as the placement needs.
    pub alignment: ByteSize,
}

/// The size of one pool block (`NVKMD_POOL_BLOCK_SIZE`, nvkmd.h:67).
pub const POOL_BLOCK_SIZE: ByteSize = ByteSize::new(2 << 20);
/// The big page size device memory is allocated in (64 KiB, as the stock
/// driver requests BIG pages and NVK did up to Mesa 25.2,
/// `nvk_device_memory.c` `NVKMD_MEM_LOCAL` -> 64 KiB).
pub const BIG_PAGE: ByteSize = ByteSize::new(1 << 16);

/// A request the RM backend does not serve.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MemError {
    /// A pool block must be exactly [`POOL_BLOCK_SIZE`] of kind 0 device memory,
    /// neither system memory, nor mappable, nor shared.
    #[error(
        "pool block: {size} bytes kind {kind} flags {flags:#x} (needs exactly 2 MiB, kind 0, VRAM, \
         not mappable or shared)",
        kind = .kind.get(),
        flags = .flags.bits()
    )]
    PoolBlock {
        /// The requested size.
        size: ByteSize,
        /// The requested kind.
        kind: PteKind,
        /// The requested flags.
        flags: MemFlags,
    },
    /// Rounding the size up to the big page passes `u64::MAX`.
    #[error("{size} bytes of device memory cannot be rounded to 64 KiB pages")]
    SizeOverflow {
        /// The requested size.
        size: ByteSize,
    },
}

/// The placement, size and alignment the RM is asked for (`nvkmd_rm.c`
/// `rm_mem_alloc_kind`). `generic_kind` is the device's uncompressed generic
/// kind (`NvRmInfo::generic_kind`).
///
/// # Errors
/// [`MemError::PoolBlock`] for a pool block request outside its contract,
/// [`MemError::SizeOverflow`] when device memory cannot be rounded up.
pub fn place(
    size: ByteSize,
    alignment: ByteSize,
    flags: MemFlags,
    kind: PteKind,
    generic_kind: PteKind,
) -> Result<Placed, MemError> {
    let mut placed = Placed {
        placement: if flags.intersects(MemFlags::GART) {
            Placement::Gart
        } else {
            Placement::Vram
        },
        size,
        alignment,
    };
    if flags.intersects(MemFlags::POOL_BLOCK) {
        if size != POOL_BLOCK_SIZE
            || kind != PteKind::PITCH
            || flags.intersects(
                MemFlags::GART
                    .union(MemFlags::CAN_MAP)
                    .union(MemFlags::SHARED),
            )
        {
            return Err(MemError::PoolBlock { size, kind, flags });
        }
        placed.placement = Placement::VramPoolBlock;
        placed.alignment = POOL_BLOCK_SIZE;
    }
    if placed.placement == Placement::Vram {
        placed.alignment = placed.alignment.max(BIG_PAGE);
        placed.size = round_up(size, BIG_PAGE).ok_or(MemError::SizeOverflow { size })?;
    }
    if kind == PteKind::PITCH
        && placed.placement == Placement::Vram
        && !flags.intersects(MemFlags::CAN_MAP.union(MemFlags::SHARED))
        && generic_kind == PteKind::GENERIC_TURING
        && placed.size.get().is_multiple_of(BIG_PAGE.get())
    {
        placed.placement = Placement::VramCompressible;
    }
    Ok(placed)
}

/// `size` rounded up to a multiple of the power of two `to`.
fn round_up(size: ByteSize, to: ByteSize) -> Option<ByteSize> {
    let mask = to.get() - 1;
    size.get()
        .checked_add(mask)
        .map(|end| ByteSize::new(end & !mask))
}

/// How a tiled (DRM modifier) request is served (`nvkmd_rm.c` `rm_mem_tiled`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tiled {
    /// No kind and no tile mode: a plain allocation.
    Plain,
    /// Uncompressed block-linear memory of the device's generic kind.
    Kind(PteKind),
}

/// A tiled request the RM backend refuses.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum TiledError {
    /// Anything but NIL's uncompressed 2D modifier on generic memory.
    #[error("modifier allocation outside uncompressed 2D generic memory")]
    Unsupported,
    /// CPU-mappable block-linear device memory before Turing: BAR1 does not
    /// return such an image's bytes in the modifier's layout, so image memory
    /// types exclude host-visible VRAM there and this request breaks that.
    #[error(
        "internal contract violated: CPU-mappable block-linear VRAM before Turing (image \
         memoryTypeBits must exclude host-visible VRAM types)"
    )]
    MappableBlockLinearBeforeTuring,
}

/// Which tiled requests are served. `tile` is NIL's tile mode (block height
/// in bits 4..6).
///
/// # Errors
/// [`TiledError`] for a request outside the uncompressed generic 2D modifier.
pub fn tiled(
    kind: PteKind,
    tile: u16,
    flags: MemFlags,
    generic_kind: PteKind,
) -> Result<Tiled, TiledError> {
    if kind == PteKind::PITCH && tile == 0 {
        return Ok(Tiled::Plain);
    }
    if kind != generic_kind || tile & !0x70 != 0 || tile >> 4 > 5 {
        return Err(TiledError::Unsupported);
    }
    if generic_kind == PteKind::GENERIC_FERMI
        && flags.intersects(MemFlags::CAN_MAP)
        && !flags.intersects(MemFlags::GART)
    {
        return Err(TiledError::MappableBlockLinearBeforeTuring);
    }
    Ok(Tiled::Kind(kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;

    fn bytes(value: u64) -> ByteSize {
        ByteSize::new(value)
    }
    fn place_turing(size: u64, alignment: u64, flags: MemFlags) -> Result<Placed, MemError> {
        place(
            bytes(size),
            bytes(alignment),
            flags,
            PteKind::PITCH,
            PteKind::GENERIC_TURING,
        )
    }

    #[test]
    fn device_memory_is_rounded_to_big_pages() {
        // nvkmd_rm.c:807-810: VRAM size and alignment become 64 KiB multiples.
        let mappable = MemFlags::VRAM.union(MemFlags::CAN_MAP);
        assert_eq!(
            place_turing(4 * KIB, 4 * KIB, mappable),
            Ok(Placed {
                placement: Placement::Vram,
                size: bytes(64 * KIB),
                alignment: bytes(64 * KIB)
            })
        );
        assert_eq!(
            place_turing(64 * KIB + 1, 2 * MIB, mappable),
            Ok(Placed {
                placement: Placement::Vram,
                size: bytes(128 * KIB),
                alignment: bytes(2 * MIB)
            })
        );
        assert_eq!(
            place_turing(u64::MAX - 10, 4 * KIB, mappable),
            Err(MemError::SizeOverflow {
                size: bytes(u64::MAX - 10)
            })
        );
    }

    #[test]
    fn system_memory_keeps_the_request() {
        // nvkmd_rm.c:781-782: GART is neither rounded nor compressible.
        assert_eq!(
            place_turing(4 * KIB, 4 * KIB, MemFlags::GART),
            Ok(Placed {
                placement: Placement::Gart,
                size: bytes(4 * KIB),
                alignment: bytes(4 * KIB)
            })
        );
    }

    #[test]
    fn private_device_memory_is_compressible_from_turing() {
        // nvkmd_rm.c:816-820: kind 0, VRAM, neither mappable nor shared, generic kind 0x06.
        let placed = place_turing(100 * KIB, 4 * KIB, MemFlags::VRAM).map(|p| p.placement);
        assert_eq!(placed, Ok(Placement::VramCompressible));
        for flags in [MemFlags::CAN_MAP, MemFlags::SHARED] {
            let placed = place_turing(64 * KIB, 4 * KIB, flags).map(|p| p.placement);
            assert_eq!(placed, Ok(Placement::Vram), "{flags:?}");
        }
        let block_linear = place(
            bytes(64 * KIB),
            bytes(4 * KIB),
            MemFlags::VRAM,
            PteKind::GENERIC_TURING,
            PteKind::GENERIC_TURING,
        );
        assert_eq!(block_linear.map(|p| p.placement), Ok(Placement::Vram));
        let fermi = place(
            bytes(64 * KIB),
            bytes(4 * KIB),
            MemFlags::VRAM,
            PteKind::PITCH,
            PteKind::GENERIC_FERMI,
        );
        assert_eq!(fermi.map(|p| p.placement), Ok(Placement::Vram));
    }

    #[test]
    fn pool_blocks_are_exactly_two_mib_of_private_device_memory() {
        // nvkmd_rm.c:786-798.
        assert_eq!(
            place_turing(2 * MIB, 4 * KIB, MemFlags::POOL_BLOCK),
            Ok(Placed {
                placement: Placement::VramPoolBlock,
                size: bytes(2 * MIB),
                alignment: bytes(2 * MIB)
            })
        );
        for flags in [MemFlags::GART, MemFlags::CAN_MAP, MemFlags::SHARED] {
            let flags = MemFlags::POOL_BLOCK.union(flags);
            assert!(
                matches!(
                    place_turing(2 * MIB, 4 * KIB, flags),
                    Err(MemError::PoolBlock { .. })
                ),
                "{flags:?}"
            );
        }
        assert!(place_turing(MIB, 4 * KIB, MemFlags::POOL_BLOCK).is_err());
        let kinded = place(
            bytes(2 * MIB),
            bytes(4 * KIB),
            MemFlags::POOL_BLOCK,
            PteKind::GENERIC_TURING,
            PteKind::GENERIC_TURING,
        );
        assert!(kinded.is_err());
    }

    #[test]
    fn tiled_requests_are_the_generic_uncompressed_2d_modifier() {
        // nvkmd_rm.c:887-903.
        let turing = PteKind::GENERIC_TURING;
        assert_eq!(
            tiled(PteKind::PITCH, 0, MemFlags::VRAM, turing),
            Ok(Tiled::Plain)
        );
        for tile in [0x00, 0x10, 0x50] {
            assert_eq!(
                tiled(turing, tile, MemFlags::VRAM, turing),
                Ok(Tiled::Kind(turing))
            );
        }
        for (kind, tile) in [
            (turing, 0x60),
            (turing, 0x01),
            (turing, 0x80),
            (PteKind::new(0xdb), 0x10),
        ] {
            assert_eq!(
                tiled(kind, tile, MemFlags::VRAM, turing),
                Err(TiledError::Unsupported),
                "kind {kind:?} tile {tile:#x}"
            );
        }
        // The kind must be the device's generic kind even with tile mode 0.
        assert_eq!(
            tiled(PteKind::PITCH, 0x10, MemFlags::VRAM, turing),
            Err(TiledError::Unsupported)
        );
        let fermi = PteKind::GENERIC_FERMI;
        let mappable = MemFlags::VRAM.union(MemFlags::CAN_MAP);
        assert_eq!(
            tiled(fermi, 0x10, mappable, fermi),
            Err(TiledError::MappableBlockLinearBeforeTuring)
        );
        assert_eq!(
            tiled(fermi, 0x10, MemFlags::GART.union(MemFlags::CAN_MAP), fermi),
            Ok(Tiled::Kind(fermi))
        );
        assert_eq!(
            tiled(turing, 0x10, mappable, turing),
            Ok(Tiled::Kind(turing))
        );
    }
}
