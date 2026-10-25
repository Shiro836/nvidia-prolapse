//! Annex-B NAL splitting, emulation-prevention removal and an RBSP bit reader (H.264 7.2/7.4.1,
//! H.265 7.2/7.4.2, Exp-Golomb 9.1 / 9.2).
use super::{bitstream, VResult};

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
    pub fn bits_left(&self) -> isize {
        (self.data.len() * 8) as isize - self.pos as isize
    }
    pub fn u(&mut self, n: u32) -> VResult<u32> {
        debug_assert!(n <= 32);
        let mut v: u64 = 0;
        for _ in 0..n {
            let byte = match self.data.get(self.pos >> 3) {
                Some(b) => *b,
                None => return bitstream("read past the end of a NAL unit"),
            };
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1) as u64;
            self.pos += 1;
        }
        Ok(v as u32)
    }
    pub fn flag(&mut self) -> VResult<bool> {
        Ok(self.u(1)? != 0)
    }
    pub fn skip(&mut self, n: usize) -> VResult<()> {
        if self.pos + n > self.data.len() * 8 {
            return bitstream("skip past the end of a NAL unit");
        }
        self.pos += n;
        Ok(())
    }
    pub fn ue(&mut self) -> VResult<u32> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros > 31 {
                return bitstream("Exp-Golomb code longer than 32 bits");
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        Ok(((1u64 << zeros) - 1 + self.u(zeros)? as u64) as u32)
    }
    pub fn se(&mut self) -> VResult<i32> {
        let k = self.ue()? as i64;
        Ok(if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) } as i32)
    }
    /// `ue(v)` checked against an inclusive maximum (named for the error).
    pub fn ue_max(&mut self, max: u32, what: &str) -> VResult<u32> {
        let v = self.ue()?;
        if v > max {
            return bitstream(format!("{what} = {v} exceeds {max}"));
        }
        Ok(v)
    }
    pub fn se_range(&mut self, min: i32, max: i32, what: &str) -> VResult<i32> {
        let v = self.se()?;
        if v < min || v > max {
            return bitstream(format!("{what} = {v} outside [{min}, {max}]"));
        }
        Ok(v)
    }
    /// True while more RBSP data precedes the rbsp_stop_one_bit (7.2 more_rbsp_data()).
    pub fn more_rbsp_data(&self) -> bool {
        let total = self.data.len() * 8;
        if self.pos >= total {
            return false;
        }
        // position of the last 1 bit in the buffer
        let mut last = None;
        for (i, &b) in self.data.iter().enumerate().rev() {
            if b != 0 {
                last = Some(i * 8 + 7 - b.trailing_zeros() as usize);
                break;
            }
        }
        matches!(last, Some(l) if self.pos < l)
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
