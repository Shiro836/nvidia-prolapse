// SPDX-License-Identifier: MIT
//! GPU virtual address policy: which VA reservations are made from the
//! caller-owned window, and which binds and unbinds of a reservation are
//! accepted (nvkmd `alloc_va`, va `bind_mem`, va `unbind`).
use nv_rm_core::{ByteSize, GpuVa};
use thiserror::Error;

/// Start of the VA window NVK owns (`NV_RM_VA_START`, `mesa/nv_rm.h`).
pub const WINDOW_START: GpuVa = GpuVa::new(1 << 34);
/// End of the VA window NVK owns (`NV_RM_VA_END`).
pub const WINDOW_END: GpuVa = GpuVa::new(1 << 40);
/// The CPU page size the RM maps in; VA reservations are made in it.
pub const PAGE: ByteSize = ByteSize::new(4096);

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

/// Where a reservation goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// The VA heap chooses the address.
    Heap,
    /// At the caller's address (`NVKMD_VA_ALLOC_FIXED`).
    Fixed(GpuVa),
}

/// A validated VA reservation request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    /// The size, rounded up to [`PAGE`].
    pub size: ByteSize,
    /// The alignment, at least [`PAGE`].
    pub alignment: ByteSize,
    /// Heap-chosen or fixed.
    pub placement: Placement,
}

impl Reservation {
    /// Validates an `alloc_va` request (`nvkmd_rm.c` `rm_va_alloc`).
    ///
    /// # Errors
    /// [`VaError::Sparse`] for sparse reservations, [`VaError::Size`] for an
    /// empty or oversized one, [`VaError::Alignment`] for an alignment that is
    /// not a power of two.
    pub fn new(
        size: ByteSize,
        alignment: ByteSize,
        sparse: Sparse,
        placement: Placement,
    ) -> Result<Self, VaError> {
        if sparse == Sparse::Yes {
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
        Ok(Self {
            size,
            alignment,
            placement,
        })
    }

    /// Whether a fixed address lies in the window with the whole reservation
    /// and is aligned; an address that does not is reported to NVK like a heap
    /// that has no room (`VK_ERROR_OUT_OF_DEVICE_MEMORY`, nvkmd_rm.c:622-633).
    pub fn fixed_fits(&self, address: GpuVa) -> bool {
        address >= WINDOW_START
            && address < WINDOW_END
            && self.size.get() <= WINDOW_END.get() - address.get()
            && address.is_aligned_to(self.alignment)
    }
}

/// Whether a reservation is sparse (`NVKMD_VA_SPARSE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sparse {
    /// Sparse residency.
    Yes,
    /// Every page is backed by a bind.
    No,
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
/// reservation (`nvkmd_rm.c` `rm_va_bind`). The VA range itself is checked by
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

/// The bindings an unbind of `range` removes, as indices into `bindings` in
/// list order, the order their RM unmaps run in (`nvkmd_rm.c` `rm_va_unbind`).
/// Nothing is removed when the request is refused.
///
/// # Errors
/// [`VaError::UnbindOutsideReservation`] for an empty range or one past the
/// reservation of `size` bytes, [`VaError::PartialUnbind`] when the range
/// covers part of a binding.
pub fn unbind(
    range: Range,
    size: ByteSize,
    bindings: &[Range],
) -> Result<impl Iterator<Item = usize>, VaError> {
    if range.size.get() == 0 || !range.within(size) {
        return Err(VaError::UnbindOutsideReservation);
    }
    if bindings
        .iter()
        .any(|binding| binding.overlaps(range) && !range.covers(*binding))
    {
        return Err(VaError::PartialUnbind);
    }
    Ok(bindings
        .iter()
        .enumerate()
        .filter(move |(_, binding)| range.covers(**binding))
        .map(|(index, _)| index))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1024;

    fn bytes(value: u64) -> ByteSize {
        ByteSize::new(value)
    }
    fn reserve(size: u64, alignment: u64) -> Result<Reservation, VaError> {
        Reservation::new(bytes(size), bytes(alignment), Sparse::No, Placement::Heap)
    }

    #[test]
    fn reservations_are_page_granular_and_fit_the_window() {
        // nvkmd_rm.c:608-615.
        assert_eq!(
            reserve(1, 0),
            Ok(Reservation {
                size: bytes(4 * KIB),
                alignment: bytes(4 * KIB),
                placement: Placement::Heap
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
            Reservation::new(bytes(4 * KIB), bytes(0), Sparse::Yes, Placement::Heap),
            Err(VaError::Sparse)
        );
    }

    #[test]
    fn fixed_addresses_must_hold_the_whole_aligned_reservation() {
        // nvkmd_rm.c:622-625.
        let r = reserve(64 * KIB, 64 * KIB).unwrap();
        assert!(r.fixed_fits(WINDOW_START));
        assert!(r.fixed_fits(GpuVa::new(WINDOW_END.get() - 64 * KIB)));
        assert!(!r.fixed_fits(GpuVa::new(WINDOW_END.get() - 60 * KIB)));
        assert!(!r.fixed_fits(GpuVa::new(WINDOW_START.get() - 64 * KIB)));
        assert!(!r.fixed_fits(GpuVa::new(WINDOW_START.get() + 4 * KIB)));
        assert!(!r.fixed_fits(WINDOW_END));
    }

    #[test]
    fn binds_stay_inside_their_memory() {
        // nvkmd_rm.c:689-690.
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
    fn unbinds_remove_whole_bindings_in_list_order() {
        // nvkmd_rm.c:652-676.
        let size = bytes(64 * KIB);
        let bindings = [
            Range::new(32 * KIB, bytes(16 * KIB)),
            Range::new(0, bytes(16 * KIB)),
            Range::new(16 * KIB, bytes(16 * KIB)),
        ];
        let removed: Vec<usize> = unbind(Range::new(0, bytes(48 * KIB)), size, &bindings)
            .unwrap()
            .collect();
        assert_eq!(removed, [0, 1, 2]);
        let removed: Vec<usize> = unbind(Range::new(16 * KIB, bytes(16 * KIB)), size, &bindings)
            .unwrap()
            .collect();
        assert_eq!(removed, [2]);
        let none = unbind(Range::new(48 * KIB, bytes(16 * KIB)), size, &bindings)
            .unwrap()
            .count();
        assert_eq!(none, 0);
        assert!(matches!(
            unbind(Range::new(8 * KIB, bytes(16 * KIB)), size, &bindings),
            Err(VaError::PartialUnbind)
        ));
        assert!(matches!(
            unbind(Range::new(0, bytes(0)), size, &bindings),
            Err(VaError::UnbindOutsideReservation)
        ));
        assert!(matches!(
            unbind(Range::new(48 * KIB, bytes(32 * KIB)), size, &bindings),
            Err(VaError::UnbindOutsideReservation)
        ));
    }
}
