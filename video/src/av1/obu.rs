// SPDX-License-Identifier: MIT
//! AV1 low-overhead bitstream format: bit reader, OBU framing, IVF container helper.
//! Written from the AV1 Bitstream & Decoding Process Specification (sections 4.10, 5.3, 5.11.1).

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Av1Error {
    Truncated(&'static str),
    Invalid(&'static str),
    /// A syntax feature this parser does not implement yet; never silently ignored.
    Unsupported(&'static str),
    NoSequenceHeader,
}

impl fmt::Display for Av1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Av1Error::Truncated(w) => write!(f, "AV1 bitstream truncated in {w}"),
            Av1Error::Invalid(w) => write!(f, "AV1 bitstream invalid: {w}"),
            Av1Error::Unsupported(w) => {
                write!(f, "AV1 feature not supported by the native NVDEC path: {w}")
            }
            Av1Error::NoSequenceHeader => write!(f, "AV1 frame before any sequence header"),
        }
    }
}

impl std::error::Error for Av1Error {}

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
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1) as u32;
            self.pos += 1;
        }
        Ok(v)
    }
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.f(1)? != 0)
    }
    /// su(n)
    pub fn su(&mut self, n: u32) -> Result<i32> {
        let v = self.f(n)? as i32;
        let sign = 1i32 << (n - 1);
        Ok(if v & sign != 0 { v - 2 * sign } else { v })
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
        Ok(self.f(lz)? + ((1u64 << lz) - 1) as u32)
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
        v |= ((b & 0x7f) as u64) << (7 * i);
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
            leb128(tu, &mut pos)? as usize
        } else {
            tu.len() - pos
        };
        let payload = tu
            .get(pos..pos + size)
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
    let hdr = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut pos = hdr;
    let mut out = Vec::new();
    while pos + 12 <= data.len() {
        let sz = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        out.push(
            data.get(pos..pos + sz)
                .ok_or(Av1Error::Truncated("ivf frame"))?,
        );
        pos += sz;
    }
    Ok(out)
}
