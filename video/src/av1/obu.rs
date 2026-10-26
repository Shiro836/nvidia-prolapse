// SPDX-License-Identifier: MIT
//! AV1 low-overhead bitstream format: bit reader, OBU framing, IVF container helper.
//! Written from the AV1 Bitstream & Decoding Process Specification (sections 4.10, 5.3, 5.11.1).
use nv_rm_core::conv::usize_from_u32;

/// Why the AV1 parser rejected a temporal unit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Av1Error {
    /// A syntax structure ends before its last element.
    #[error("AV1 bitstream truncated in {0}")]
    Truncated(&'static str),
    /// A syntax element violates a bitstream conformance requirement.
    #[error("AV1 bitstream invalid: {0}")]
    Invalid(&'static str),
    /// A syntax feature this parser does not implement yet; never silently ignored.
    #[error("AV1 feature not supported by the native NVDEC path: {0}")]
    Unsupported(&'static str),
    /// A frame header before any sequence header.
    #[error("AV1 frame before any sequence header")]
    NoSequenceHeader,
    /// A syntax-derived value does not fit the field it is stored in.
    #[error("AV1 bitstream invalid: {0}")]
    Range(#[from] crate::OutOfRange),
    /// A sequence whose format the session cannot hold: bit depth other than the session's,
    /// monochrome or not 4:2:0 (unsupported).
    #[error(
        "AV1 bit depth {bit_depth} / mono {mono_chrome} / subsampling {subsampling_x}{subsampling_y} in a {session_bit_depth}-bit 4:2:0 session (8- or 10-bit 4:2:0, session created with the stream's bit depth)"
    )]
    SessionFormat {
        /// BitDepth.
        bit_depth: u8,
        /// mono_chrome.
        mono_chrome: bool,
        /// subsampling_x.
        subsampling_x: u8,
        /// subsampling_y.
        subsampling_y: u8,
        /// The session's bit depth.
        session_bit_depth: u32,
    },
}

/// `Result` of the AV1 parser.
pub type Result<T> = std::result::Result<T, Av1Error>;

/// MSB-first bit reader (spec 4.10 f(n)).
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    pub fn bit_pos(&self) -> usize {
        self.pos
    }
    pub fn f(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self
                .data
                .get(self.pos >> 3)
                .ok_or(Av1Error::Truncated("f(n)"))?;
            v = (v << 1) | u32::from((byte >> (7 - (self.pos & 7))) & 1);
            self.pos += 1;
        }
        Ok(v)
    }
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.f(1)? != 0)
    }
    /// su(n) (spec 4.10.6): an n-bit two's complement value, 1 <= n <= 32.
    pub fn su(&mut self, n: u32) -> Result<i32> {
        if n == 0 || n > 32 {
            return Err(Av1Error::Invalid("su(n) width outside 1..=32"));
        }
        let v = i64::from(self.f(n)?);
        let sign = 1i64 << (n - 1);
        let value = if v & sign != 0 { v - 2 * sign } else { v };
        // -2^(n-1) <= value < 2^(n-1) with n <= 32.
        i32::try_from(value).map_err(|_overflow| Av1Error::Invalid("su(n) wider than 32 bits"))
    }
    /// su(n) for n <= 8, whose range -2^(n-1)..2^(n-1) fits `i8`.
    pub fn su_i8(&mut self, n: u32) -> Result<i8> {
        if n > 8 {
            return Err(Av1Error::Invalid("su(n) wider than 8 bits read as i8"));
        }
        i8::try_from(self.su(n)?).map_err(|_overflow| Av1Error::Invalid("su(n) wider than 8 bits"))
    }
    /// ns(n)
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        let w = 32 - n.leading_zeros(); // FloorLog2(n) + 1
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.f(1)?;
        Ok((v << 1) - m + extra)
    }
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut lz = 0;
        while !self.flag()? {
            lz += 1;
            if lz >= 32 {
                return Ok(u32::MAX);
            }
        }
        // lz <= 31: (2^lz - 1) + f(lz) <= 2^32 - 2 (spec 4.10.3).
        Ok(self.f(lz)? + ((1u32 << lz) - 1))
    }
    /// le(n): little-endian n bytes (byte aligned).
    pub fn le(&mut self, n: u32) -> Result<u32> {
        let mut t = 0u32;
        for i in 0..n {
            t |= self.f(8)? << (8 * i);
        }
        Ok(t)
    }
    pub fn byte_alignment(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
}

pub fn leb128(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for i in 0..8 {
        let b = *data.get(*pos).ok_or(Av1Error::Truncated("leb128"))?;
        *pos += 1;
        v |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(Av1Error::Invalid("leb128 longer than 8 bytes"))
}

pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub const OBU_FRAME_HEADER: u8 = 3;
pub const OBU_TILE_GROUP: u8 = 4;
pub const OBU_METADATA: u8 = 5;
pub const OBU_FRAME: u8 = 6;
pub const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
pub const OBU_TILE_LIST: u8 = 8;
pub const OBU_PADDING: u8 = 15;

#[derive(Debug, Clone, Copy)]
pub struct Obu<'a> {
    pub obu_type: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub payload: &'a [u8],
}

/// Split a temporal unit (low-overhead format, every OBU carries obu_size) into OBUs.
pub fn split_obus(tu: &[u8]) -> Result<Vec<Obu<'_>>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < tu.len() {
        let h = tu[pos];
        pos += 1;
        if h & 0x80 != 0 {
            return Err(Av1Error::Invalid("obu_forbidden_bit"));
        }
        let obu_type = (h >> 3) & 0xf;
        let ext = h & 4 != 0;
        let has_size = h & 2 != 0;
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if ext {
            let e = *tu
                .get(pos)
                .ok_or(Av1Error::Truncated("obu_extension_header"))?;
            pos += 1;
            temporal_id = e >> 5;
            spatial_id = (e >> 3) & 3;
        }
        let size = if has_size {
            usize::try_from(leb128(tu, &mut pos)?)
                .map_err(|_overflow| Av1Error::Truncated("obu payload"))?
        } else {
            tu.len() - pos
        };
        let payload = pos
            .checked_add(size)
            .and_then(|end| tu.get(pos..end))
            .ok_or(Av1Error::Truncated("obu payload"))?;
        pos += size;
        out.push(Obu {
            obu_type,
            temporal_id,
            spatial_id,
            payload,
        });
    }
    Ok(out)
}

/// IVF container: returns the frame payloads (one temporal unit each).
pub fn ivf_frames(data: &[u8]) -> Result<Vec<&[u8]>> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return Err(Av1Error::Invalid("not an IVF file"));
    }
    let mut pos = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut out = Vec::new();
    while let Some(&[s0, s1, s2, s3, ..]) = data.get(pos..).filter(|rest| rest.len() >= 12) {
        let sz = usize_from_u32(u32::from_le_bytes([s0, s1, s2, s3]));
        pos += 12;
        out.push(
            pos.checked_add(sz)
                .and_then(|end| data.get(pos..end))
                .ok_or(Av1Error::Truncated("ivf frame"))?,
        );
        pos += sz;
    }
    Ok(out)
}
