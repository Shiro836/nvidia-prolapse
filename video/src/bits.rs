// SPDX-License-Identifier: MIT
//! Annex-B NAL splitting, emulation-prevention removal and an RBSP bit reader (H.264 7.2/7.4.1,
//! H.265 7.2/7.4.2, Exp-Golomb 9.1 / 9.2).
use nv_rm_core::conv::usize_from_u32;

/// One NAL unit inside an Annex-B buffer: `start` is the offset of its start code
/// (`00 00 01` or `00 00 00 01`), `nal` the escaped NAL bytes (header included, trailing
/// zero bytes stripped).
#[derive(Debug, Clone, Copy)]
pub struct Nal<'a> {
    pub start: usize,
    pub nal: &'a [u8],
}

/// Splits an Annex-B byte stream at `00 00 01` start codes.
pub fn split_annexb(data: &[u8]) -> Vec<Nal<'_>> {
    let mut starts = Vec::new(); // (start code offset, payload offset)
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let sc = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            starts.push((sc, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &(sc, p)) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map_or(data.len(), |n| n.0);
        while end > p && data[end - 1] == 0 {
            end -= 1;
        }
        if end > p {
            out.push(Nal {
                start: sc,
                nal: &data[p..end],
            });
        }
    }
    out
}

/// Removes emulation-prevention bytes (`00 00 03` → `00 00`).
pub fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &b in nal {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// Why the RBSP bit reader stopped.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A read needs more bits than the NAL unit has left.
    #[error("read past the end of a NAL unit (bit {at} of {len})")]
    Truncated {
        /// Bit position of the read.
        at: usize,
        /// Bits in the NAL unit.
        len: usize,
    },
    /// A skip goes past the end of the NAL unit.
    #[error("skip past the end of a NAL unit ({bits} bits at bit {at})")]
    SkipPastEnd {
        /// Bit position of the skip.
        at: usize,
        /// Bits skipped.
        bits: usize,
    },
    /// More than 31 leading zero bits: the code does not fit 32 bits (H.264 9.1, H.265 9.2).
    #[error("Exp-Golomb code longer than 32 bits")]
    ExpGolombTooLong,
    /// `u(n)` with n above 32.
    #[error("u({0}) wider than 32 bits")]
    WidthTooLarge(u32),
    /// A `ue(v)` syntax element above its semantic maximum.
    #[error("{field} = {value} exceeds {max}")]
    AboveMax {
        /// The syntax element.
        field: &'static str,
        /// Its decoded value.
        value: u32,
        /// The inclusive maximum.
        max: u32,
    },
    /// An `se(v)` syntax element outside its semantic range.
    #[error("{field} = {value} outside [{min}, {max}]")]
    OutsideRange {
        /// The syntax element.
        field: &'static str,
        /// Its decoded value.
        value: i32,
        /// The inclusive minimum.
        min: i32,
        /// The inclusive maximum.
        max: i32,
    },
}

/// `Result` of the bit reader.
pub type Result<T> = std::result::Result<T, Error>;

/// MSB-first bit reader over an RBSP.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    /// `u(n)` (H.264 7.2 / H.265 7.2 `read_bits(n)`), `n <= 32`.
    ///
    /// # Errors
    /// [`Error::WidthTooLarge`] above 32 bits, [`Error::Truncated`] past the end.
    pub fn u(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(Error::WidthTooLarge(n));
        }
        let mut v: u32 = 0;
        for _ in 0..n {
            let Some(&byte) = self.data.get(self.pos >> 3) else {
                return Err(Error::Truncated {
                    at: self.pos,
                    len: self.data.len().saturating_mul(8),
                });
            };
            // n <= 32 bits: the shift never drops a set bit.
            v = (v << 1) | u32::from((byte >> (7 - (self.pos & 7))) & 1);
            self.pos += 1;
        }
        Ok(v)
    }
    /// One bit as a flag.
    ///
    /// # Errors
    /// [`Error::Truncated`] past the end.
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.u(1)? != 0)
    }
    /// Skips `n` bits.
    ///
    /// # Errors
    /// [`Error::SkipPastEnd`] past the end.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        let end = self.pos.checked_add(n);
        if end.is_none_or(|end| end.div_ceil(8) > self.data.len()) {
            return Err(Error::SkipPastEnd {
                at: self.pos,
                bits: n,
            });
        }
        self.pos += n;
        Ok(())
    }
    /// `ue(v)` (H.264 9.1, H.265 9.2): `2^zeros - 1 + read_bits(zeros)`, at most `2^32 - 2`.
    ///
    /// # Errors
    /// [`Error::ExpGolombTooLong`] or [`Error::Truncated`].
    pub fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros > 31 {
                return Err(Error::ExpGolombTooLong);
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        // zeros <= 31: (2^zeros - 1) + (2^zeros - 1) <= 2^32 - 2.
        Ok((1u32 << zeros) - 1 + self.u(zeros)?)
    }
    /// `se(v)` (H.264 9.1.1, H.265 9.2.2): k -> (-1)^(k+1) * Ceil(k / 2), within `i32` for every
    /// k of `ue(v)`.
    ///
    /// # Errors
    /// As [`Self::ue`].
    pub fn se(&mut self) -> Result<i32> {
        let k = self.ue()?;
        let magnitude = k.div_ceil(2);
        // Ceil((2^32 - 2) / 2) = 2^31 - 1: the magnitude always fits i32.
        let magnitude = i32::try_from(magnitude).map_err(|_overflow| Error::ExpGolombTooLong)?;
        Ok(if k & 1 == 1 { magnitude } else { -magnitude })
    }
    /// `ue(v)` checked against an inclusive maximum (named for the error).
    ///
    /// # Errors
    /// [`Error::AboveMax`], or as [`Self::ue`].
    pub fn ue_max(&mut self, max: u32, field: &'static str) -> Result<u32> {
        let value = self.ue()?;
        if value > max {
            return Err(Error::AboveMax { field, value, max });
        }
        Ok(value)
    }
    /// `se(v)` checked against an inclusive range (named for the error).
    ///
    /// # Errors
    /// [`Error::OutsideRange`], or as [`Self::ue`].
    pub fn se_range(&mut self, min: i32, max: i32, field: &'static str) -> Result<i32> {
        let value = self.se()?;
        if value < min || value > max {
            return Err(Error::OutsideRange {
                field,
                value,
                min,
                max,
            });
        }
        Ok(value)
    }
    /// True while more RBSP data precedes the `rbsp_stop_one_bit` (7.2 `more_rbsp_data()`).
    pub fn more_rbsp_data(&self) -> bool {
        // Position of the last 1 bit in the buffer.
        let last = self
            .data
            .iter()
            .rposition(|&b| b != 0)
            .map(|i| i * 8 + 7 - usize_from_u32(self.data[i].trailing_zeros()));
        last.is_some_and(|l| self.pos < l)
    }
}

/// Splits an Annex-B elementary stream into access units (H.264 7.4.1.2.3 / H.265 7.4.2.4.4):
/// a new unit starts at a parameter set / AUD / prefix SEI following a slice, or at the first
/// slice of a picture. `hevc` selects the NAL header format.
pub fn access_units(data: &[u8], hevc: bool) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let (mut start, mut has_slice) = (0usize, false);
    for n in split_annexb(data) {
        let (vcl, first, prefix) = if hevc {
            let t = (n.nal[0] >> 1) & 0x3f;
            let vcl = t < 32;
            (
                vcl,
                vcl && n.nal.len() > 2 && n.nal[2] & 0x80 != 0,
                matches!(t, 32..=35 | 39 | 41..=44 | 48..=55),
            )
        } else {
            let t = n.nal[0] & 0x1f;
            let vcl = t == 1 || t == 5;
            (
                vcl,
                vcl && n.nal.len() > 1 && n.nal[1] & 0x80 != 0,
                matches!(t, 6..=9 | 14..=18),
            )
        };
        if has_slice && (prefix || first) {
            out.push(&data[start..n.start]);
            start = n.start;
            has_slice = false;
        }
        has_slice |= vcl;
    }
    if start < data.len() {
        out.push(&data[start..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_golomb_and_split() {
        // ue: 1 → 0, 010 → 1, 011 → 2, 00100 → 3 ; se: 010 → +1, 011 → −1
        let d = [0b1010_0110, 0b0100_0100, 0b1100_0000];
        let mut r = BitReader::new(&d);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.ue().unwrap(), 2);
        assert_eq!(r.ue().unwrap(), 3);
        assert_eq!(r.se().unwrap(), 1);
        assert_eq!(r.se().unwrap(), -1);
        let s = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 0, 0, 3, 1, 0];
        let n = split_annexb(&s);
        assert_eq!(n.len(), 2);
        assert_eq!((n[0].start, n[0].nal), (0, &[0x67, 1, 2][..]));
        assert_eq!(n[1].start, 7);
        assert_eq!(unescape(n[1].nal), vec![0x68, 0, 0, 1]);
    }
}
