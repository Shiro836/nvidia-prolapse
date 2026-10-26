// SPDX-License-Identifier: MIT
//! NEW_H264 (`SET_APPLICATION_ID` 8) picture setup: 0x200 bytes, followed by the scaling
//! matrices at +0x200 (`H264_SET_SCALING_LIST_OFFSET`), status written by the engine at +0x700.
//!
//! The layout is not the public `nvdec_new_h264_pic_s` (that header struct is a copy of the
//! legacy one). It was established from stock libnvcuvid 615.71.09 (builder at ELF 0x3c460,
//! reference lists at 0x48800). This module is an
//! independent implementation of that layout from the field table, verified byte for byte
//! against stock captures (`tests.rs`).
//!
//! | off | field |
//! |---|---|
//! | 0x040 | 16 × DPB entry (16 B): u32 {b0-6 slot, b7-11 col slot, b12-13 used_for_reference, b14 long-term, b15 non-existing, b16 field, b17-20 top marking, b21-24 bottom marking, b25 output layout}, top POC, bottom POC, FrameIdx |
//! | 0x140 / 0x160 / 0x180 | initial RefPicList0 (P), RefPicList0 (B), RefPicList1 (B): 32 B, `pos | parity << 5`, unused 0x1f |
//! | 0x1a0 / 0x1b0 | MVC inter-view refs l0 / l1 (16 B, 0x1f = none) |
//! | 0x1c0 | gptimer = min(max(MBs, 8100) · 10000, limit) |
//! | 0x1c4 | bitstream length |
//! | 0x1c9 b6-7 / 0x1ca b0-1 | target-surface layout values |
//! | 0x1cc | b0-4 current slot, b5-9 surface id, b10 1, b12-14 0, b15 output layout, b18 1 |
//! | 0x1d0 / 0x1d4 (12 bit) | surface pitch / 16 ; 0x1d2 = 0xffff |
//! | 0x1d8 | b0-3 luma depth, b4-7 chroma depth, b8-17 width MBs, b18-27 height MBs, b28-29 chroma format |
//! | 0x1de | 64-bit SPS/PPS flag word |
//! | 0x1e6 | b0-9 colocated-MV row size (256 B units), 0x1e7 b2 !(direct_8x8 && frame_mbs_only), b3 qpprime bypass |
//! | 0x1e8 | frame_num ; 0x1ea picture flags ; 0x1ec / 0x1f0 POCs ; 0x1fc width term |
use super::ps::{Pps, Sps};

pub const SETUP_SIZE: usize = 0x200;
pub const SCALING_OFFSET: usize = 0x200;
pub const SCALING_SIZE: usize = 0xe0;
pub const STATUS_OFFSET: usize = 0x700;
pub const APP_ID_NEW_H264: u32 = 8;

/// Engine/session values the setup carries that do not come from the stream.
#[derive(Debug, Clone, Copy)]
pub struct EngineParams {
    /// Progressive 8-bit 4:2:0 pictures use NEW_H264 (application id 8); otherwise every picture uses
    /// the legacy setup (id 3). Stock: caps word & 0x1fe00000, Blackwell and Rubin only (stock
    /// ELF 0x450c0; Ada trace: id 3 for High progressive).
    pub new_h264: bool,
    /// Writes 0x1fc, 0x1c9/0x1ca and 0x1ce/0x1d2 (stock: caps word & 0x1ec00000; set on GB202).
    pub ext_fields: bool,
    /// gptimer upper bound.
    pub gptimer_limit: u32,
    /// Luma pitch of the target surfaces in bytes (0x1d0/0x1d4 carry pitch / 16). `None`: the
    /// block-linear pitch of a surface sized for the coded picture, align(width, 64) — what
    /// stock writes for its own surfaces (176 → 192, 352 → 384, 1920 → 1920 in the captures).
    pub surface_pitch: Option<u32>,
    /// Two 2-bit surface layout values at 0x1c9 b6-7 / 0x1ca b0-1 (stock asks the target surface
    /// object; GB202 block-linear NV12: 1 and 2).
    pub surface_layout: (u8, u8),
    /// Output memory layout bit (DPB byte 3 bit 1, 0x1cc bit 15).
    pub output_layout: bool,
}
impl EngineParams {
    pub fn pitch(&self, sps: &Sps) -> u32 {
        self.surface_pitch
            .unwrap_or_else(|| sps.width().next_multiple_of(64))
    }
    /// Values observed on GB202 (RTX 5090) with block-linear NV12 targets.
    pub fn blackwell(surface_pitch: Option<u32>) -> Self {
        Self::for_class(0xcfb0, surface_pitch)
    }
    /// The engine parameters of NVDEC class `class` (stock per-generation gates, `crate::NvdecGen`).
    pub fn for_class(class: u32, surface_pitch: Option<u32>) -> Self {
        let g = crate::NvdecGen::of(class);
        Self {
            new_h264: g.blackwell(),
            ext_fields: g.blackwell_b(),
            gptimer_limit: u32::MAX,
            surface_pitch,
            surface_layout: (1, 2),
            output_layout: false,
        }
    }
}

/// One reference as placed in the descriptor DPB.
#[derive(Debug, Clone, Copy, Default)]
pub struct DescRef {
    /// Colocated slot (None for a non-existing frame).
    pub slot: Option<u8>,
    /// used_for_reference: 1 top, 2 bottom, 3 both.
    pub state: u8,
    pub long_term: bool,
    pub non_existing: bool,
    pub poc: [i32; 2],
    /// frame_num (short-term) or LongTermFrameIdx (long-term).
    pub frame_idx: u32,
}

/// Everything the setup needs for one picture.
pub struct DescInput<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    /// DPB positions 0..16 (None = empty).
    pub dpb: &'a [Option<DescRef>; 16],
    /// Slot written into empty DPB positions (stock carries the last reference picture's slot).
    pub empty_slot: u8,
    pub lists: [&'a [u8]; 3],
    pub cur_slot: u8,
    pub surface_id: u32,
    pub bitstream_len: u32,
    pub frame_num: u32,
    pub field_pic: bool,
    pub bottom_field: bool,
    pub second_field: bool,
    pub ref_pic: bool,
    pub intra_pic: bool,
    pub poc: [i32; 2],
}

fn put32(d: &mut [u8], o: usize, v: u32) {
    d[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put16(d: &mut [u8], o: usize, v: u16) {
    d[o..o + 2].copy_from_slice(&v.to_le_bytes());
}

/// Builds setup (0x200) + scaling matrices (0xe0) = 0x2e0 bytes.
pub fn build(inp: &DescInput, ep: &EngineParams) -> Vec<u8> {
    let (sps, pps) = (inp.sps, inp.pps);
    let mut d = vec![0u8; SETUP_SIZE + SCALING_SIZE];
    let layout_bit = u32::from(ep.output_layout) << 25;
    for (i, e) in inp.dpb.iter().enumerate() {
        let o = 0x40 + 16 * i;
        let Some(e) = e else {
            let s = u32::from(inp.empty_slot);
            put32(&mut d, o, (s & 0x7f) | (s & 0x1f) << 7 | layout_bit);
            continue;
        };
        let s = e.slot.map_or(0x7f, u32::from);
        let mark = 1 + u32::from(e.long_term);
        let mut w = (s & 0x7f) | (s & 0x1f) << 7 | (u32::from(e.state) & 3) << 12;
        w |= u32::from(e.long_term) << 14 | u32::from(e.non_existing) << 15;
        if e.state & 1 != 0 {
            w |= mark << 17;
        }
        if e.state & 2 != 0 {
            w |= mark << 21;
        }
        put32(&mut d, o, w | layout_bit);
        put32(&mut d, o + 4, e.poc[0].cast_unsigned());
        put32(&mut d, o + 8, e.poc[1].cast_unsigned());
        put32(&mut d, o + 12, e.frame_idx);
    }
    for (k, l) in inp.lists.iter().enumerate() {
        let o = 0x140 + 0x20 * k;
        d[o..o + 0x20].fill(0x1f);
        d[o..o + l.len()].copy_from_slice(l);
    }
    d[0x1a0..0x1c0].fill(0x1f);
    let mbs = (sps.pic_width_in_mbs * sps.frame_height_in_mbs()).max(8100);
    put32(
        &mut d,
        0x1c0,
        mbs.saturating_mul(10000).min(ep.gptimer_limit),
    );
    put32(&mut d, 0x1c4, inp.bitstream_len);
    let w1cc = (u32::from(inp.cur_slot) & 0x1f)
        | (inp.surface_id & 0x1f) << 5
        | 1 << 10
        | u32::from(ep.output_layout) << 15
        | 1 << 18;
    put32(&mut d, 0x1cc, w1cc);
    let pitch16 = (((ep.pitch(sps) + 15) >> 4) & 0xfff) as u16;
    put16(&mut d, 0x1d0, pitch16);
    put16(&mut d, 0x1d4, pitch16);
    let luma = (sps.bit_depth_luma_minus8 + 8) & 0xf;
    let chroma = (sps.bit_depth_chroma_minus8 + 8) & 0xf;
    put32(
        &mut d,
        0x1d8,
        luma | chroma << 4
            | (sps.pic_width_in_mbs & 0x3ff) << 8
            | (sps.frame_height_in_mbs() & 0x3ff) << 18
            | (sps.chroma_format_idc & 3) << 28,
    );
    let b = |v: bool| u64::from(v);
    let m = |v: i32, mask: u64| u64::from(v.cast_unsigned()) & mask;
    let fl = m(sps.log2_max_frame_num_minus4.cast_signed(), 0xf)
        | m(sps.log2_max_poc_lsb_minus4.cast_signed(), 0xf) << 4
        | m(sps.pic_order_cnt_type.cast_signed(), 3) << 8
        | b(sps.delta_pic_order_always_zero) << 10
        | m(sps.max_num_ref_frames.cast_signed(), 0x1f) << 11
        | m(pps.num_ref_idx_l0_default_active_minus1.cast_signed(), 0x1f) << 16
        | m(pps.num_ref_idx_l1_default_active_minus1.cast_signed(), 0x1f) << 21
        | m(pps.chroma_qp_index_offset, 0x1f) << 26
        | m(pps.second_chroma_qp_index_offset, 0x1f) << 31
        | m(pps.pic_init_qp_minus26, 0x7f) << 36
        | b(pps.bottom_field_pic_order_in_frame_present) << 43
        | b(pps.deblocking_filter_control_present) << 44
        | b(pps.redundant_pic_cnt_present) << 45
        | b(pps.transform_8x8_mode) << 46
        | b(pps.constrained_intra_pred) << 47
        | b(pps.entropy_coding_mode) << 48
        | b(pps.weighted_pred) << 49
        | m(pps.weighted_bipred_idc.cast_signed(), 3) << 50
        | b(sps.direct_8x8_inference) << 62
        | b(sps.frame_mbs_only) << 63;
    d[0x1de..0x1e6].copy_from_slice(&fl.to_le_bytes());
    let coarse = sps.direct_8x8_inference && sps.frame_mbs_only;
    let per_mb: u32 = if coarse { 4 } else { 16 };
    let row = (sps.pic_width_in_mbs * per_mb * 8).div_ceil(256) & 0x3ff;
    let w1e6 = row as u16
        | u16::from(!coarse) << 10
        | u16::from(sps.qpprime_y_zero_transform_bypass) << 11;
    put16(&mut d, 0x1e6, w1e6);
    put16(&mut d, 0x1e8, inp.frame_num as u16);
    d[0x1ea] = u8::from(inp.field_pic)
        | u8::from(inp.bottom_field) << 1
        | u8::from(inp.second_field) << 2
        | u8::from(inp.ref_pic) << 3
        | u8::from(inp.intra_pic) << 4;
    if inp.field_pic {
        let p = inp.poc[usize::from(inp.bottom_field)].cast_unsigned();
        put32(&mut d, 0x1ec, p);
        put32(&mut d, 0x1f0, p);
    } else {
        put32(&mut d, 0x1ec, inp.poc[0].cast_unsigned());
        put32(&mut d, 0x1f0, inp.poc[1].cast_unsigned());
    }
    if ep.ext_fields {
        let w = (sps.pic_width_in_mbs & 0x3ff) << 4;
        put32(&mut d, 0x1fc, (((w * 2 + 0x7e) & 0xff80) + 0x13f) >> 8);
        d[0x1c9] |= ep.surface_layout.0 << 6;
        d[0x1ca] |= ep.surface_layout.1 & 3;
        put16(&mut d, 0x1d2, 0xffff);
    }
    // Scaling: 6 × 4x4 then 8x8 Y intra, Y inter, raster order.
    let s = &pps.scaling;
    for (i, m4) in s.m4.iter().enumerate() {
        d[SCALING_OFFSET + 16 * i..SCALING_OFFSET + 16 * (i + 1)].copy_from_slice(m4);
    }
    d[SCALING_OFFSET + 96..SCALING_OFFSET + 160].copy_from_slice(&s.m8[0]);
    d[SCALING_OFFSET + 160..SCALING_OFFSET + 224].copy_from_slice(&s.m8[1]);
    d
}

/// `SET_CONTROL_PARAMS` for NEW_H264 as stock emits it: codec 3, gptimer/err-conceal/mbtimer
/// enables (0x52053) with the error-concealment reference in bits 7-11.
pub fn control_params(err_conceal_slot: u8) -> u32 {
    0x52053 | ((u32::from(err_conceal_slot) & 0x1f) << 7)
}
