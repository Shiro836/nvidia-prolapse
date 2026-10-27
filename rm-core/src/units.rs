// SPDX-License-Identifier: MIT
//! Distinct quantities of the RM interface as distinct types (RUST.md §5): a GPU virtual address,
//! a byte size, an RM object handle, an RM class id and a channel's completion (fence) value. Each
//! wraps the integer the kernel interface carries, with the same width, so a value converts to its
//! wire field with `get` at the ioctl boundary and nowhere else.
//!
//! Arithmetic is defined only where it means something: a GPU VA plus a byte size is a GPU VA
//! (`GpuVa::checked_add`), fence values are ordered and advance by a count. Everything is checked;
//! none of these wrap.
//!
//! Plain `Copy` data, no state; usable from any thread.
use std::fmt;

/// Formats the inner value in hex (`{:#x}`) and decimal (`{}`), so messages read as before.
macro_rules! integer_formatting {
    ($name:ident) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
        impl fmt::LowerHex for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::LowerHex::fmt(&self.0, f)
            }
        }
    };
}

/// A GPU virtual address in a channel's address space (`NV01_MEMORY_VIRTUAL` / `FERMI_VASPACE_A`;
/// 40 bits on every generation this backend drives).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GpuVa(u64);

impl GpuVa {
    /// The address `value`.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    /// The address as the `NvU64` an RM parameter block or a GPU method carries.
    pub const fn get(self) -> u64 {
        self.0
    }
    /// The address `size` bytes further on, or `None` past `u64::MAX`.
    pub const fn checked_add(self, size: ByteSize) -> Option<Self> {
        match self.0.checked_add(size.0) {
            Some(end) => Some(Self(end)),
            None => None,
        }
    }
    /// Whether the address is a multiple of `alignment` bytes.
    pub const fn is_aligned_to(self, alignment: ByteSize) -> bool {
        self.0.is_multiple_of(alignment.0)
    }
}
integer_formatting!(GpuVa);

/// A size in bytes of GPU-visible memory or of a GPU mapping (`NvU64` length fields of
/// `nvos.h` NVOS32/NVOS46).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ByteSize(u64);

impl ByteSize {
    /// `value` bytes.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    /// The size as the `NvU64` an RM parameter block carries.
    pub const fn get(self) -> u64 {
        self.0
    }
}
integer_formatting!(ByteSize);

/// An RM object handle (`NvHandle`, open-gpu-kernel-modules `nvtypes.h`): a client, device,
/// subdevice, memory, channel or any other object RM allocated. Zero is never a valid object;
/// the RM calls that return a handle refuse it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct RmHandle(u32);

impl RmHandle {
    /// The handle `value`.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
    /// The handle as the `NvHandle` an RM parameter block carries.
    pub const fn get(self) -> u32 {
        self.0
    }
}
integer_formatting!(RmHandle);

/// An RM class id (`NvU32` `hClass` of `nvos.h` NVOS21/NVOS64; the numbers of the public
/// `class/cl*.h` headers, e.g. `0xc96f` `BLACKWELL_CHANNEL_GPFIFO_A`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ClassId(u32);

impl ClassId {
    /// The class `value`.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
    /// The class as the `NvU32` an RM parameter block carries.
    pub const fn get(self) -> u32 {
        self.0
    }
}
integer_formatting!(ClassId);

/// A channel's completion value: the 64-bit sequence its semaphore releases after each batch
/// (zero = nothing submitted). Values only grow; a 32-bit semaphore payload is extended to this
/// width against the newest issued value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct FenceValue(u64);

impl FenceValue {
    /// The value `value`.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    /// The value as the `NvU64` a semaphore payload or a token in the C ABI carries.
    pub const fn get(self) -> u64 {
        self.0
    }
    /// The value `count` steps later, or `None` past `u64::MAX` (the sequence never wraps).
    pub const fn checked_add(self, count: u64) -> Option<Self> {
        match self.0.checked_add(count) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}
integer_formatting!(FenceValue);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layouts_are_the_wire_integers() {
        assert_eq!(size_of::<GpuVa>(), size_of::<u64>());
        assert_eq!(size_of::<ByteSize>(), size_of::<u64>());
        assert_eq!(size_of::<RmHandle>(), size_of::<u32>());
        assert_eq!(size_of::<ClassId>(), size_of::<u32>());
        assert_eq!(size_of::<FenceValue>(), size_of::<u64>());
    }
    #[test]
    fn address_arithmetic_is_checked() {
        let va = GpuVa::new(0x1000);
        assert_eq!(
            va.checked_add(ByteSize::new(0x10)),
            Some(GpuVa::new(0x1010))
        );
        assert_eq!(GpuVa::new(u64::MAX).checked_add(ByteSize::new(1)), None);
        assert!(va.is_aligned_to(ByteSize::new(4096)));
        assert!(!GpuVa::new(0x1010).is_aligned_to(ByteSize::new(4096)));
        assert_eq!(FenceValue::new(u64::MAX).checked_add(1), None);
        assert_eq!(
            format!("{:#x} {}", ClassId::new(0xc96f), RmHandle::new(7)),
            "0xc96f 7"
        );
    }
}
