// SPDX-License-Identifier: MIT
//! Integer conversions without a `From` impl (RUST.md §6), in one place for the workspace.
//!
//! Two kinds, each a named function so a call site states which one it means:
//! - **Lossless on every shipped target** (`x86_64-linux-android`, `i686-linux-android`, the 64-bit
//!   host): `usize` is 32 or 64 bits wide there, so `u32 -> usize` and `usize -> u64` never lose
//!   information. Std has no `From` for them because Rust also targets 16-bit pointers. Each
//!   function carries a compile-time assert on the pointer width it relies on; a target outside
//!   that range fails to build instead of truncating.
//! - **Named field extraction** (`low_u32`, `high_u32`, `low_u16`, `low_u8`): the part of a wider
//!   value that a hardware or bitstream field holds, written as a byte split so it needs no cast.
//!
//! - **Constant narrowing** (`const_u32_from_usize`): a `usize` constant into a `u32` field, checked
//!   when the constant is evaluated.
//!
//! Signed/unsigned and narrowing conversions of values that may not fit are not here: they use
//! `try_from` at the call site and handle the error.
//!
//! Pure functions, no state; usable from any thread.

// The lossless group (and the checked constant narrowing) is the one place in the workspace where `as` converts integers (RUST.md §6).
#![expect(
    clippy::as_conversions,
    reason = "lossless on every shipped target (each function asserts the pointer width it relies on) or checked before the cast"
)]

/// Shipped targets have 32- or 64-bit pointers (`x86_64`/`i686` Android, 64-bit hosts).
const POINTER_WIDTH_32_OR_64: () = assert!(usize::BITS == 32 || usize::BITS == 64);

/// `u32 -> usize`, lossless where `usize` has at least 32 bits.
#[inline]
pub const fn usize_from_u32(v: u32) -> usize {
    const { assert!(usize::BITS >= u32::BITS) };
    let () = POINTER_WIDTH_32_OR_64;
    v as usize
}

/// `usize -> u64`, lossless where `usize` has at most 64 bits.
#[inline]
pub const fn u64_from_usize(v: usize) -> u64 {
    const { assert!(usize::BITS <= u64::BITS) };
    let () = POINTER_WIDTH_32_OR_64;
    v as u64
}

/// `i32 -> isize`, lossless where `isize` has at least 32 bits.
#[inline]
pub const fn isize_from_i32(v: i32) -> isize {
    const { assert!(isize::BITS >= i32::BITS) };
    let () = POINTER_WIDTH_32_OR_64;
    v as isize
}

/// `isize -> i64`, lossless where `isize` has at most 64 bits.
#[inline]
pub const fn i64_from_isize(v: isize) -> i64 {
    const { assert!(isize::BITS <= i64::BITS) };
    let () = POINTER_WIDTH_32_OR_64;
    v as i64
}

/// `usize -> u32` for a constant (a ring size, a struct size): call it in a const context
/// (`const { const_u32_from_usize(N) }`), where a value beyond `u32::MAX` fails the build. At run
/// time the same check is an assert, so it is not for values that arrive from outside.
///
/// # Panics
/// If `v` exceeds `u32::MAX` (a build error in a const context).
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the assert above proves the value fits u32"
)]
pub const fn const_u32_from_usize(v: usize) -> u32 {
    assert!(v <= u32::MAX as usize, "constant does not fit u32");
    v as u32
}

/// Bits 31:0 of `v` (the `*_LO`/lower-dword half of a 64-bit address or value).
#[inline]
pub const fn low_u32(v: u64) -> u32 {
    let [b0, b1, b2, b3, ..] = v.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
}

/// Bits 63:32 of `v` (the `*_HI`/upper-dword half of a 64-bit address or value).
#[inline]
pub const fn high_u32(v: u64) -> u32 {
    let [.., b4, b5, b6, b7] = v.to_le_bytes();
    u32::from_le_bytes([b4, b5, b6, b7])
}

/// Bits 15:0 of `v`.
#[inline]
pub const fn low_u16(v: u32) -> u16 {
    let [b0, b1, ..] = v.to_le_bytes();
    u16::from_le_bytes([b0, b1])
}

/// Bits 7:0 of `v`.
#[inline]
pub const fn low_u8(v: u32) -> u8 {
    v.to_le_bytes()[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_round_trips_at_the_limits() {
        assert_eq!(usize_from_u32(u32::MAX), 0xffff_ffff);
        assert_eq!(
            u64_from_usize(usize::MAX),
            u64::try_from(usize::MAX).unwrap_or(0)
        );
        assert_eq!(isize_from_i32(i32::MIN), -0x8000_0000);
        assert_eq!(
            i64_from_isize(isize::MIN),
            i64::try_from(isize::MIN).unwrap_or(0)
        );
    }

    #[test]
    fn field_extraction_takes_the_named_bits() {
        let v = 0x0123_4567_89ab_cdef_u64;
        assert_eq!((high_u32(v), low_u32(v)), (0x0123_4567, 0x89ab_cdef));
        assert_eq!(low_u16(0x1234_5678), 0x5678);
        assert_eq!(low_u8(0x1234_5678), 0x78);
    }

    #[test]
    fn constant_narrowing_keeps_the_value() {
        const RING: u32 = const_u32_from_usize(1024);
        assert_eq!(RING, 1024);
        assert_eq!(const_u32_from_usize(0xffff_ffff), u32::MAX);
    }
}
