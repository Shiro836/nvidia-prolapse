// SPDX-License-Identifier: MIT
//! GPU virtual address policy: which VA reservations are made from the
//! caller-owned window and from which of its two heaps, and which binds and
//! unbinds of a reservation are accepted (nvkmd `alloc_va`, va `bind_mem`,
//! va `unbind`).
//!
//! The window is split like `nvkmd_nouveau` splits its own
//! (`nvkmd_nouveau_dev.c:51-58`, `nvkmd_nouveau_va.c:20-60`): ordinary
//! reservations come from the lower half, capture/replay reservations
//! (`NVKMD_VA_REPLAY`) from the upper half, so no other reservation can take
//! an address an application captured and replays later.
use nv_rm_core::{ByteSize, GpuVa};
use thiserror::Error;

/// Start of the VA window NVK owns (`NV_RM_VA_START`, `mesa/nv_rm.h`).
pub const WINDOW_START: GpuVa = GpuVa::new(1 << 34);
/// End of the VA window NVK owns (`NV_RM_VA_END`).
pub const WINDOW_END: GpuVa = GpuVa::new(1 << 40);
/// Start of the capture/replay heap: the upper half of the window,
/// `[512 GiB, 1 TiB)`.
pub const REPLAY_START: GpuVa = GpuVa::new(1 << 39);
/// The CPU page size the RM maps in; VA reservations are made in it.
pub const PAGE: ByteSize = ByteSize::new(4096);

const _: () = assert!(
    WINDOW_START.get() < REPLAY_START.get() && REPLAY_START.get() < WINDOW_END.get(),
    "the replay heap lies inside the window"
);

/// `enum nvkmd_va_flags` (Mesa `src/nouveau/vulkan/nvkmd/nvkmd.h`), as NVK
/// passes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaFlags(u32);

impl VaFlags {
    /// `NVKMD_VA_SPARSE`: sparse residency.
    pub const SPARSE: Self = Self(1 << 0);
    /// `NVKMD_VA_REPLAY`: a capture/replay address (buffer device address
    /// capture, `VkBufferOpaqueCaptureAddressCreateInfo`).
    pub const REPLAY: Self = Self(1 << 1);
    /// `NVKMD_VA_ALLOC_FIXED`: at the caller's address.
    pub const ALLOC_FIXED: Self = Self(1 << 2);

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

/// A VA request the backend refuses.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VaError {
    /// Sparse residency is not supported (`has_sparse` is false).
    #[error("sparse residency")]
    Sparse,
    /// The size is 0 or larger than the window.
    #[error("{size} bytes of VA cannot be reserved")]
    Size {
        /// The requested size.
        size: ByteSize,
    },
    /// The alignment is not a power of two.
    #[error("unaligned VA")]
    Alignment,
    /// A fixed address that is not a replayed capture address: NVK asks for
    /// fixed addresses only to replay a capture (`nvkmd.h`
    /// `NVKMD_VA_ALLOC_FIXED`).
    #[error("fixed VA outside capture/replay")]
    FixedNotReplay,
    /// A bind outside the memory it binds.
    #[error("VA bind range outside allocation")]
    BindOutsideMemory,
    /// An unbind outside the reservation.
    #[error("VA unbind range outside reservation")]
    UnbindOutsideReservation,
    /// An unbind that covers part of an existing binding: the RM backend binds
    /// whole mappings only.
    #[error("partial VA unbind")]
    PartialUnbind,
}

/// One of the window's two heaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heap {
    /// `[WINDOW_START, REPLAY_START)`: every reservation that is not for
    /// capture/replay.
    Ordinary,
    /// `[REPLAY_START, WINDOW_END)`: capture/replay reservations.
    Replay,
}

impl Heap {
    /// The heap whose range holds `address`, which a released range goes back
    /// to.
    pub fn of(address: GpuVa) -> Self {
        if address >= REPLAY_START {
            Self::Replay
        } else {
            Self::Ordinary
        }
    }

    /// The heap's range as (start, size).
    pub const fn range(self) -> (GpuVa, ByteSize) {
        match self {
            Self::Ordinary => (
                WINDOW_START,
                ByteSize::new(REPLAY_START.get() - WINDOW_START.get()),
            ),
            Self::Replay => (
                REPLAY_START,
                ByteSize::new(WINDOW_END.get() - REPLAY_START.get()),
            ),
        }
    }
}

/// Where a reservation goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// The heap chooses the address.
    Heap(Heap),
    /// At the caller's address, a replayed capture address in the
    /// [`Heap::Replay`] range.
    Fixed(GpuVa),
}

/// A VA reservation request with its size and alignment validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    /// The size, rounded up to [`PAGE`].
    pub size: ByteSize,
    /// The alignment, at least [`PAGE`].
    pub alignment: ByteSize,
}

impl Reservation {
    /// Validates the size and alignment of an `alloc_va` request;
    /// [`Reservation::target`] then says where it goes.
    ///
    /// # Errors
    /// [`VaError::Sparse`] for sparse reservations, [`VaError::Size`] for an
    /// empty or oversized one, [`VaError::Alignment`] for an alignment that is
    /// not a power of two.
    pub fn new(size: ByteSize, alignment: ByteSize, flags: VaFlags) -> Result<Self, VaError> {
        if flags.intersects(VaFlags::SPARSE) {
            return Err(VaError::Sparse);
        }
        let window = WINDOW_END.get() - WINDOW_START.get();
        if size.get() == 0 || size.get() > window - (PAGE.get() - 1) {
            return Err(VaError::Size { size });
        }
        let mask = PAGE.get() - 1;
        let size = ByteSize::new((size.get() + mask) & !mask);
        let alignment = alignment.max(PAGE);
        if !alignment.get().is_power_of_two() {
            return Err(VaError::Alignment);
        }
        Ok(Self { size, alignment })
    }

    /// Where a reservation with `flags` goes: a fixed address only for a
    /// replayed capture, a heap-chosen one from the heap its flags select.
    ///
    /// # Errors
    /// [`VaError::FixedNotReplay`] for a fixed address without
    /// `NVKMD_VA_REPLAY`.
    pub fn target(flags: VaFlags, fixed: GpuVa) -> Result<Target, VaError> {
        if flags.intersects(VaFlags::ALLOC_FIXED) {
            if !flags.intersects(VaFlags::REPLAY) {
                return Err(VaError::FixedNotReplay);
            }
            return Ok(Target::Fixed(fixed));
        }
        Ok(Target::Heap(if flags.intersects(VaFlags::REPLAY) {
            Heap::Replay
        } else {
            Heap::Ordinary
        }))
    }

    /// Whether a fixed address lies in the replay heap with the whole
    /// reservation and is aligned. One that does not is an invalid capture
    /// address, as a taken one is (`VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS`,
    /// `nvkmd_nouveau_va.c:23-42`).
    pub fn fixed_fits(&self, address: GpuVa) -> bool {
        address >= REPLAY_START
            && address < WINDOW_END
            && self.size.get() <= WINDOW_END.get() - address.get()
            && address.is_aligned_to(self.alignment)
    }
}

/// A range of a reservation or of an allocation, as an offset and a size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    /// Offset from the start.
    pub offset: u64,
    /// Size in bytes.
    pub size: ByteSize,
}

impl Range {
    /// The range `[offset, offset + size)`.
    pub const fn new(offset: u64, size: ByteSize) -> Self {
        Self { offset, size }
    }
    /// Whether the range lies inside `[0, total)`. An empty range does when
    /// its offset does.
    pub const fn within(self, total: ByteSize) -> bool {
        self.offset <= total.get() && self.size.get() <= total.get() - self.offset
    }
    /// The end offset; callers keep ranges [`Range::within`] a reservation,
    /// whose size the window bounds below 2^40.
    const fn end(self) -> u64 {
        self.offset + self.size.get()
    }
    fn overlaps(self, other: Self) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }
    fn covers(self, other: Self) -> bool {
        other.offset >= self.offset && other.end() <= self.end()
    }
}

/// Checks a bind of `memory` (an allocation of `memory_size` bytes) into a
/// reservation. The VA range itself is checked by
/// the unbind of that range that precedes every bind.
///
/// # Errors
/// [`VaError::BindOutsideMemory`].
pub fn check_bind(memory: Range, memory_size: ByteSize) -> Result<(), VaError> {
    if memory.within(memory_size) {
        Ok(())
    } else {
        Err(VaError::BindOutsideMemory)
    }
}

/// An accepted unbind of a range of a reservation: which bindings it
/// removes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unbind(Range);

impl Unbind {
    /// Whether the unbind removes `binding`; [`unbind`] refused every partial
    /// overlap, so these are all the bindings the range overlaps.
    pub fn removes(self, binding: Range) -> bool {
        self.0.covers(binding)
    }
}

/// Checks an unbind of `range` from a reservation of `size` bytes holding
/// `bindings`. The bindings it removes are unmapped in list order, the order
/// they were bound in.
///
/// # Errors
/// [`VaError::UnbindOutsideReservation`] for an empty range or one past the
/// reservation, [`VaError::PartialUnbind`] when the range covers part of a
/// binding; nothing is removed then.
pub fn unbind(range: Range, size: ByteSize, bindings: &[Range]) -> Result<Unbind, VaError> {
    if range.size.get() == 0 || !range.within(size) {
        return Err(VaError::UnbindOutsideReservation);
    }
    if bindings
        .iter()
        .any(|binding| binding.overlaps(range) && !range.covers(*binding))
    {
        return Err(VaError::PartialUnbind);
    }
    Ok(Unbind(range))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1024;
    const GIB: u64 = 1024 * 1024 * KIB;

    fn bytes(value: u64) -> ByteSize {
        ByteSize::new(value)
    }
    fn reserve(size: u64, alignment: u64) -> Result<Reservation, VaError> {
        Reservation::new(bytes(size), bytes(alignment), VaFlags::default())
    }

    #[test]
    fn reservations_are_page_granular_and_fit_the_window() {
        assert_eq!(
            reserve(1, 0),
            Ok(Reservation {
                size: bytes(4 * KIB),
                alignment: bytes(4 * KIB),
            })
        );
        assert_eq!(
            reserve(8 * KIB, 64 * KIB).map(|r| r.alignment),
            Ok(bytes(64 * KIB))
        );
        assert_eq!(reserve(0, 0), Err(VaError::Size { size: bytes(0) }));
        let window = (1 << 40) - (1 << 34);
        assert_eq!(reserve(window - 4095, 0).map(|r| r.size), Ok(bytes(window)));
        assert_eq!(
            reserve(window - 4094, 0),
            Err(VaError::Size {
                size: bytes(window - 4094)
            })
        );
        assert_eq!(reserve(4 * KIB, 12 * KIB), Err(VaError::Alignment));
        assert_eq!(
            Reservation::new(bytes(4 * KIB), bytes(0), VaFlags::SPARSE),
            Err(VaError::Sparse)
        );
    }

    #[test]
    fn replay_reservations_come_from_the_upper_half() {
        // The heap choice of nvkmd_nouveau_dev.c:51-58.
        let none = GpuVa::new(0);
        assert_eq!(
            Reservation::target(VaFlags::default(), none),
            Ok(Target::Heap(Heap::Ordinary))
        );
        assert_eq!(
            Reservation::target(VaFlags::REPLAY, none),
            Ok(Target::Heap(Heap::Replay))
        );
        let fixed = GpuVa::new(600 * GIB);
        assert_eq!(
            Reservation::target(VaFlags::REPLAY.union(VaFlags::ALLOC_FIXED), fixed),
            Ok(Target::Fixed(fixed))
        );
        assert_eq!(
            Reservation::target(VaFlags::ALLOC_FIXED, fixed),
            Err(VaError::FixedNotReplay)
        );
        assert_eq!(
            Heap::Ordinary.range(),
            (GpuVa::new(16 * GIB), bytes(496 * GIB))
        );
        assert_eq!(
            Heap::Replay.range(),
            (GpuVa::new(512 * GIB), bytes(512 * GIB))
        );
    }

    #[test]
    fn released_ranges_return_to_the_heap_of_their_address() {
        assert_eq!(Heap::of(WINDOW_START), Heap::Ordinary);
        assert_eq!(
            Heap::of(GpuVa::new(REPLAY_START.get() - 4 * KIB)),
            Heap::Ordinary
        );
        assert_eq!(Heap::of(REPLAY_START), Heap::Replay);
        assert_eq!(
            Heap::of(GpuVa::new(WINDOW_END.get() - 4 * KIB)),
            Heap::Replay
        );
    }

    #[test]
    fn fixed_addresses_must_hold_the_whole_aligned_reservation_in_the_replay_heap() {
        let r = reserve(64 * KIB, 64 * KIB).unwrap();
        assert!(r.fixed_fits(REPLAY_START));
        assert!(r.fixed_fits(GpuVa::new(WINDOW_END.get() - 64 * KIB)));
        assert!(!r.fixed_fits(GpuVa::new(WINDOW_END.get() - 60 * KIB)));
        assert!(!r.fixed_fits(GpuVa::new(REPLAY_START.get() - 64 * KIB)));
        assert!(!r.fixed_fits(WINDOW_START));
        assert!(!r.fixed_fits(GpuVa::new(REPLAY_START.get() + 4 * KIB)));
        assert!(!r.fixed_fits(WINDOW_END));
    }

    #[test]
    fn binds_stay_inside_their_memory() {
        assert_eq!(check_bind(Range::new(0, bytes(8)), bytes(8)), Ok(()));
        assert_eq!(check_bind(Range::new(8, bytes(0)), bytes(8)), Ok(()));
        assert_eq!(
            check_bind(Range::new(9, bytes(0)), bytes(8)),
            Err(VaError::BindOutsideMemory)
        );
        assert_eq!(
            check_bind(Range::new(4, bytes(u64::MAX)), bytes(8)),
            Err(VaError::BindOutsideMemory)
        );
    }

    #[test]
    fn unbinds_remove_whole_bindings() {
        let size = bytes(64 * KIB);
        let bindings = [
            Range::new(32 * KIB, bytes(16 * KIB)),
            Range::new(0, bytes(16 * KIB)),
            Range::new(16 * KIB, bytes(16 * KIB)),
        ];
        let removed = |range: Range| -> Vec<usize> {
            let unbind = unbind(range, size, &bindings).unwrap();
            (0..bindings.len())
                .filter(|index| unbind.removes(bindings[*index]))
                .collect()
        };
        assert_eq!(removed(Range::new(0, bytes(48 * KIB))), [0, 1, 2]);
        assert_eq!(removed(Range::new(16 * KIB, bytes(16 * KIB))), [2]);
        assert_eq!(removed(Range::new(48 * KIB, bytes(16 * KIB))), [0_usize; 0]);
        assert_eq!(
            unbind(Range::new(8 * KIB, bytes(16 * KIB)), size, &bindings),
            Err(VaError::PartialUnbind)
        );
        assert_eq!(
            unbind(Range::new(0, bytes(0)), size, &bindings),
            Err(VaError::UnbindOutsideReservation)
        );
        assert_eq!(
            unbind(Range::new(48 * KIB, bytes(32 * KIB)), size, &bindings),
            Err(VaError::UnbindOutsideReservation)
        );
    }
}
