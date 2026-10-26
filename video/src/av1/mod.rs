// SPDX-License-Identifier: MIT
//! Native NVDEC AV1 codec module: bitstream -> picture jobs (descriptor bytes, CDF buffer selection,
//! DPB slots, colMV/segment buffer selection). Pure functions over decoder state; the session layer owns the
//! buffers and emits the methods. NVDEC-specific conventions were matched against stock libnvcuvid
//! 615.71.09 captures.

pub mod cdf_defaults;
pub mod codec;
pub mod desc;
pub mod film_grain;
pub mod header;
pub mod mf;
pub mod obu;
#[cfg(test)]
mod tests;

use header::{FrameHeader, KEY_FRAME, PRIMARY_REF_NONE, ParserState, SequenceHeader, TileRange};
use obu::{Av1Error, Result};

pub const DPB_SLOTS: usize = 9;
pub const CDF_BUFFERS: usize = 10;
pub const CDF_BUFFER_SIZE: usize = 0x3000;

/// Default CDF q-context (spec get_qctx / libaom av1_get_q_ctx).
pub const fn default_cdf_qctx(base_q_idx: u8) -> u8 {
    match base_q_idx {
        0..=20 => 0,
        21..=60 => 1,
        61..=120 => 2,
        _ => 3,
    }
}

/// Write the spec default CDFs for `qctx` into an NVDEC CDF buffer (0x3000 bytes, inverse 15-bit form).
pub fn write_default_cdfs(qctx: u8, out: &mut [u8]) {
    assert!(out.len() >= CDF_BUFFER_SIZE && qctx < 4);
    out[..CDF_BUFFER_SIZE].fill(0);
    for p in cdf_defaults::PLACEMENTS {
        let t: &[u16] = match p.src {
            cdf_defaults::Src::Fixed(t) => t,
            cdf_defaults::Src::PerQ(q) => q[usize::from(qctx)],
        };
        for (r, row) in t.chunks_exact(p.vals).enumerate() {
            for (k, v) in row.iter().enumerate() {
                let a = p.off + r * p.stride + 2 * k;
                out[a..a + 2].copy_from_slice(&(32768 - *v).to_le_bytes());
            }
        }
    }
}

/// What the motion-field / segment logic needs to know about a decoded picture in a DPB slot.
#[derive(Debug, Clone, Copy, Default)]
pub struct PicRecord {
    pub order_hint: u32,
    /// Order hints of this picture's references, LAST..ALTREF (spec order).
    pub ref_hints: [u32; 7],
    pub upscaled_width: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub intra: bool,
    pub segmentation_enabled: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    in_use: bool,
    saved_cdf: Option<usize>,
    rec: PicRecord,
}

/// One picture the engine must decode.
#[derive(Debug, Clone)]
pub struct PictureJob {
    pub seq: SequenceHeader,
    pub hdr: FrameHeader,
    /// Tile data handed to NVDEC (IN_BUF), tile-group payloads concatenated.
    pub bitstream: Vec<u8>,
    pub tiles: Vec<TileRange>,
    /// DPB slot the picture decodes into.
    pub slot: usize,
    /// DPB slots of LAST..ALTREF (None on intra frames / missing references).
    pub ref_slots: [Option<usize>; 7],
    /// Records of the pictures in `ref_slots` at the time of this picture.
    pub ref_recs: [Option<PicRecord>; 7],
    pub cdf_read: usize,
    pub cdf_write: usize,
    /// Some(q): the read buffer must first be filled with `write_default_cdfs(q)`.
    pub cdf_default_qctx: Option<u8>,
    pub mf: desc::MotionField,
    /// colMV read slots (None = keep the previous binding, see `colmv_reads`).
    pub colmv_reads: [Option<usize>; 3],
    /// Segment-id read slot (None = pool base).
    pub seg_read: Option<usize>,
    pub apply_grain: bool,
}

/// AV1 decoder state across temporal units.
pub struct Av1Codec {
    pub parser: ParserState,
    slots: [Slot; DPB_SLOTS],
    ref_map: [Option<usize>; 8],
    cdf_stamp: [u64; CDF_BUFFERS],
    stamp: u64,
    pending: std::collections::VecDeque<Vec<u8>>,
    obu_pos: usize,
    cur: Option<(FrameHeader, Vec<u8>, Vec<TileRange>)>,
}

/// One output of the parser, in decode order.
pub enum Av1Out {
    Decode(Box<PictureJob>),
    /// show_existing_frame of the picture in DPB slot `slot`.
    ShowExisting {
        slot: usize,
        hdr: Box<FrameHeader>,
    },
}

impl Default for Av1Codec {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1Codec {
    pub fn new() -> Self {
        Self {
            parser: ParserState::default(),
            slots: [Slot::default(); DPB_SLOTS],
            ref_map: [None; 8],
            cdf_stamp: [0; CDF_BUFFERS],
            stamp: 0,
            pending: Default::default(),
            obu_pos: 0,
            cur: None,
        }
    }

    pub fn ref_map(&self) -> [Option<usize>; 8] {
        self.ref_map
    }

    fn alloc_cdf(&mut self, free: &mut [bool; CDF_BUFFERS]) -> Result<usize> {
        let mut best: Option<usize> = None;
        for (i, &f) in free.iter().enumerate() {
            if f && best.is_none_or(|b| self.cdf_stamp[i] < self.cdf_stamp[b]) {
                best = Some(i);
            }
        }
        let i = best.ok_or(Av1Error::Invalid("no free CDF buffer"))?;
        self.stamp += 1;
        self.cdf_stamp[i] = self.stamp;
        free[i] = false;
        Ok(i)
    }

    /// Parse one temporal unit; returns the pictures to decode in decode order.
    /// Parse one temporal unit completely (tests / tools); show_existing outputs are dropped.
    pub fn decode_temporal_unit(&mut self, tu: &[u8]) -> Result<Vec<PictureJob>> {
        self.push_unit(tu);
        let mut jobs = Vec::new();
        while let Some(o) = self.next_out()? {
            if let Av1Out::Decode(j) = o {
                jobs.push(*j);
            }
        }
        Ok(jobs)
    }

    /// Queue one temporal unit; outputs come from `next_out` (parsed lazily, one frame at a time, so
    /// reference state always reflects the frames handed out so far).
    pub fn push_unit(&mut self, tu: &[u8]) {
        self.pending.push_back(tu.to_vec());
    }

    pub fn next_out(&mut self) -> Result<Option<Av1Out>> {
        loop {
            let Some(tu) = self.pending.front().cloned() else {
                return Ok(None);
            };
            let obus = obu::split_obus(&tu)?;
            let Some(o) = obus.get(self.obu_pos).copied() else {
                if self.cur.is_some() {
                    self.cur = None;
                    return Err(Av1Error::Truncated("frame without all tile groups"));
                }
                self.pending.pop_front();
                self.obu_pos = 0;
                continue;
            };
            self.obu_pos += 1;
            match o.obu_type {
                obu::OBU_SEQUENCE_HEADER => {
                    self.parser.seq = Some(SequenceHeader::parse(o.payload)?)
                }
                obu::OBU_FRAME_HEADER | obu::OBU_FRAME => {
                    if self.cur.is_some() && o.obu_type == obu::OBU_FRAME_HEADER {
                        continue; // redundant copy of the header of the frame in progress
                    }
                    let h =
                        self.parser
                            .parse_frame_header(o.payload, o.temporal_id, o.spatial_id)?;
                    if h.show_existing_frame {
                        let slot = self.ref_map[usize::from(h.frame_to_show_map_idx)]
                            .ok_or(Av1Error::Invalid("show_existing_frame of an empty slot"))?;
                        self.show_existing(&h);
                        return Ok(Some(Av1Out::ShowExisting {
                            slot,
                            hdr: Box::new(h),
                        }));
                    }
                    let mut bs = Vec::new();
                    let mut tiles = Vec::new();
                    if o.obu_type == obu::OBU_FRAME {
                        let done = header::parse_tile_group(
                            &h,
                            &o.payload[h.header_bytes..],
                            &mut bs,
                            &mut tiles,
                        )?;
                        if done {
                            return Ok(Some(Av1Out::Decode(Box::new(
                                self.finish_frame(h, bs, tiles)?,
                            ))));
                        }
                    }
                    self.cur = Some((h, bs, tiles));
                }
                obu::OBU_TILE_GROUP => {
                    let (h, mut bs, mut tiles) = self
                        .cur
                        .take()
                        .ok_or(Av1Error::Invalid("tile group without header"))?;
                    if header::parse_tile_group(&h, o.payload, &mut bs, &mut tiles)? {
                        return Ok(Some(Av1Out::Decode(Box::new(
                            self.finish_frame(h, bs, tiles)?,
                        ))));
                    }
                    self.cur = Some((h, bs, tiles));
                }
                obu::OBU_TILE_LIST => {
                    return Err(Av1Error::Unsupported("large scale tile (OBU_TILE_LIST)"));
                }
                _ => {}
            }
        }
    }

    /// Drop all reference and parser state except the sequence header (seek).
    pub fn flush(&mut self) {
        let seq = self.parser.seq.take();
        *self = Self::new();
        self.parser.seq = seq;
    }

    /// Record of the picture in a DPB slot.
    pub fn slot_record(&self, slot: usize) -> PicRecord {
        self.slots[slot].rec
    }

    /// DPB slots currently held by the reference map.
    pub fn referenced_slots(&self) -> impl Iterator<Item = usize> + '_ {
        let mut seen = [false; DPB_SLOTS];
        self.ref_map
            .iter()
            .flatten()
            .copied()
            .filter(move |&s| !std::mem::replace(&mut seen[s], true))
    }

    fn show_existing(&mut self, h: &FrameHeader) {
        if h.frame_type == KEY_FRAME {
            let s = self.ref_map[usize::from(h.frame_to_show_map_idx)];
            self.ref_map = [s; 8];
        }
        self.parser.update_refs(h);
    }

    fn finish_frame(
        &mut self,
        h: FrameHeader,
        bitstream: Vec<u8>,
        tiles: Vec<TileRange>,
    ) -> Result<PictureJob> {
        let seq = self.parser.seq.clone().ok_or(Av1Error::NoSequenceHeader)?;
        // DPB: keep referenced slots, release the rest, take the first free one.
        let mut referenced = [false; DPB_SLOTS];
        for s in self.ref_map.iter().flatten() {
            referenced[*s] = true;
        }
        let mut free_cdf = [true; CDF_BUFFERS];
        for (i, slot) in self.slots.iter_mut().enumerate() {
            slot.in_use = referenced[i];
            if slot.in_use {
                if let Some(c) = slot.saved_cdf {
                    free_cdf[c] = false;
                }
            } else {
                slot.saved_cdf = None;
            }
        }
        let slot = self
            .slots
            .iter()
            .position(|s| !s.in_use)
            .ok_or(Av1Error::Invalid("DPB full"))?;
        let cdf_write = self.alloc_cdf(&mut free_cdf)?;
        let mut ref_slots = [None; 7];
        if !h.frame_is_intra {
            for (r, &idx) in ref_slots.iter_mut().zip(&h.ref_frame_idx) {
                *r = self.ref_map[usize::from(idx)];
            }
        }
        let primary_slot = if h.primary_ref_frame != PRIMARY_REF_NONE {
            ref_slots[usize::from(h.primary_ref_frame)]
        } else {
            None
        };
        let (cdf_read, cdf_default_qctx) = match primary_slot.and_then(|s| self.slots[s].saved_cdf)
        {
            Some(c) if h.frame_type & 1 == 1 => (c, None),
            _ => (
                self.alloc_cdf(&mut free_cdf)?,
                Some(default_cdf_qctx(h.base_q_idx)),
            ),
        };
        let recs: [Option<PicRecord>; 7] = ref_slots.map(|s| s.map(|s| self.slots[s].rec));
        let (mf, cand_slots) = mf::select(&seq, &h, &recs, &ref_slots);
        let mut colmv_reads = [None; 3];
        for j in 0..3 {
            if mf.use_temporal[j] {
                colmv_reads[j] = cand_slots[j];
            }
        }
        let seg_read = match primary_slot {
            Some(s) if h.segmentation_enabled && self.slots[s].rec.segmentation_enabled => {
                let r = &self.slots[s].rec;
                (r.mi_cols == h.mi_cols && r.mi_rows == h.mi_rows).then_some(s)
            }
            _ => None,
        };
        let rec = PicRecord {
            order_hint: h.order_hint,
            ref_hints: h.order_hints,
            upscaled_width: h.upscaled_width,
            frame_width: h.frame_width,
            frame_height: h.frame_height,
            mi_cols: h.mi_cols,
            mi_rows: h.mi_rows,
            intra: h.frame_is_intra,
            segmentation_enabled: h.segmentation_enabled,
        };
        self.slots[slot] = Slot {
            in_use: true,
            saved_cdf: Some(if h.disable_frame_end_update_cdf {
                cdf_read
            } else {
                cdf_write
            }),
            rec,
        };
        for i in 0..8 {
            if h.refresh_frame_flags & (1 << i) != 0 {
                self.ref_map[i] = Some(slot);
            }
        }
        self.parser.update_refs(&h);
        Ok(PictureJob {
            apply_grain: h.film_grain.apply_grain,
            seq,
            hdr: h,
            bitstream,
            tiles,
            slot,
            ref_slots,
            ref_recs: recs,
            cdf_read,
            cdf_write,
            cdf_default_qctx,
            mf,
            colmv_reads,
            seg_read,
        })
    }
}

impl PictureJob {
    /// Build the descriptor block (pic_s, tile info, global motion, film grain) into `block`
    /// (`desc::DESC_BLOCK_SIZE` bytes). `pitch` = luma pitch of the decode surface in bytes;
    /// `blackwell_b` = the class gate of the stat word and the +0x244 word (`desc::PicInputs`).
    pub fn build_descriptor(&self, pitch: u32, blackwell_b: bool, block: &mut [u8]) {
        let mut refs = [desc::RefGeom::default(); 7];
        for (g, rec) in refs.iter_mut().zip(&self.ref_recs) {
            if let Some(r) = rec {
                *g = desc::RefGeom {
                    width: r.upscaled_width,
                    height: r.frame_height,
                    order_hint: r.order_hint,
                };
            }
        }
        let pi = desc::PicInputs {
            seq: &self.seq,
            hdr: &self.hdr,
            stream_len: self.bitstream.len() as u32,
            refs,
            mf: self.mf,
            prev_segment_ids: self.seg_read.is_some(),
            apply_grain: self.apply_grain,
            pitch,
            blackwell_b,
        };
        block[..desc::DESC_BLOCK_SIZE].fill(0);
        block[..desc::PIC_S_SIZE].copy_from_slice(&desc::build_pic_s(&pi));
        if blackwell_b {
            let ext = desc::pic_ext_244(self.hdr.upscaled_width);
            block[desc::PIC_EXT_244..desc::PIC_EXT_244 + 4].copy_from_slice(&ext.to_le_bytes());
        }
        desc::build_tile_info(
            &self.hdr,
            &self.tiles,
            &mut block[desc::DESC_TILE_INFO..desc::DESC_GLOBAL_MOTION],
        );
        desc::build_global_motion(
            &self.hdr,
            &mut block[desc::DESC_GLOBAL_MOTION..desc::DESC_FILM_GRAIN],
        );
        if self.apply_grain {
            film_grain::build(
                &self.seq,
                &self.hdr.film_grain,
                &mut block[desc::DESC_FILM_GRAIN..desc::DESC_STATUS],
            );
        }
    }
}
