// SPDX-License-Identifier: MIT
//! Legacy H.264 setup (`SET_APPLICATION_ID` 3, public `nvdec_h264_pic_s`, 0x2fc bytes; slice
//! offsets at setup+0x300). Stock 615.71.09 uses it for interlaced SPSs (frame_mbs_only_flag =
//! 0; builder at ELF 0x3d070); this module follows the public struct
//! (`nvdec_drv.h`, MIT) and the values observed in stock captures of an MBAFF stream
//! (verified byte for byte in `tests.rs`).
//! Only frame pictures are produced (MBAFF or non-MBAFF frames of an interlaced SPS); field
//! pictures are rejected by the caller.
use super::desc::{DescInput, EngineParams};

pub const APP_ID_H264: u32 = 3;
pub const SETUP_SIZE: usize = 0x300;
pub const SLICE_OFFSETS_OFFSET: usize = 0x300;

/// Surface/side-buffer geometry the legacy setup carries. These describe OUR surfaces and
/// buffers; [`LegacyLayout::stock_fit`] reproduces the values stock wrote for its own surfaces
/// (fitted on one 352x288 capture: pitch 384, field offsets = pitch × field height aligned to
/// 16 rows, MBHIST = 128 B per MB row, history = 768 B per MB column).
#[derive(Debug, Clone, Copy)]
pub struct LegacyLayout {
    /// `tileFormat` (0 TBL, 1 KBL, 2 Tile16x16) and `gob_height` code.
    pub tile_format: u32,
    pub gob_height: u32,
    pub pitch_luma: u32,
    pub pitch_chroma: u32,
    /// Bottom-field plane offsets in 256-byte units (top and frame offsets are 0).
    pub luma_bot_offset: u32,
    pub chroma_bot_offset: u32,
    /// MBHIST buffer size in bytes and HISTORY buffer size in 256-byte units.
    pub mbhist_size: u32,
    pub hist_size: u32,
}
impl LegacyLayout {
    pub fn stock_fit(pitch: u32, width_mbs: u32, frame_height_mbs: u32) -> Self {
        let field_rows = frame_height_mbs * 8;
        Self {
            tile_format: 1,
            gob_height: 0,
            pitch_luma: pitch,
            pitch_chroma: pitch,
            luma_bot_offset: pitch * field_rows.next_multiple_of(16) / 256,
            chroma_bot_offset: pitch * (field_rows / 2).next_multiple_of(16) / 256,
            mbhist_size: frame_height_mbs * 128,
            hist_size: width_mbs * 3,
        }
    }
}

fn put32(d: &mut [u8], o: usize, v: u32) {
    d[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// Builds the 0x300-byte setup + the slice-offset table at +0x300 (one u32 per slice: the
/// offset of its start code in the bitstream).
///
/// # Panics
/// Never: there are fewer slices than bitstream bytes, whose count fits u32.
pub fn build(
    inp: &DescInput,
    ep: &EngineParams,
    lay: &LegacyLayout,
    mbaff_frame: bool,
    slice_offsets: &[u32],
) -> Vec<u8> {
    let (sps, pps) = (inp.sps, inp.pps);
    let mut d = vec![0u8; SETUP_SIZE + 4 * slice_offsets.len().max(1)];
    // eos[16]: two end-of-stream NAL start codes (00 00 01 0b); explicitEOSPresentFlag = 1.
    d[0x34..0x44].copy_from_slice(&[0, 0, 1, 0x0b, 0, 0, 0, 0, 0, 0, 1, 0x0b, 0, 0, 0, 0]);
    d[0x44] = 1;
    put32(&mut d, 0x48, inp.bitstream_len);
    // One offset per slice NAL of a bitstream whose length fits u32 (bitstream_len).
    put32(
        &mut d,
        0x4c,
        u32::try_from(slice_offsets.len()).expect("fewer slices than bitstream bytes"),
    );
    put32(&mut d, 0x50, lay.mbhist_size);
    let mbs = (sps.pic_width_in_mbs * sps.frame_height_in_mbs()).max(8100);
    put32(
        &mut d,
        0x54,
        mbs.saturating_mul(10000).min(ep.gptimer_limit),
    );
    put32(&mut d, 0x58, sps.log2_max_poc_lsb_minus4);
    put32(&mut d, 0x5c, u32::from(sps.delta_pic_order_always_zero));
    put32(&mut d, 0x60, u32::from(sps.frame_mbs_only));
    put32(&mut d, 0x64, sps.pic_width_in_mbs);
    put32(&mut d, 0x68, sps.frame_height_in_mbs());
    put32(
        &mut d,
        0x6c,
        (lay.tile_format & 3) | (lay.gob_height & 7) << 2,
    );
    put32(&mut d, 0x70, u32::from(pps.entropy_coding_mode));
    put32(
        &mut d,
        0x74,
        u32::from(pps.bottom_field_pic_order_in_frame_present),
    );
    put32(&mut d, 0x78, pps.num_ref_idx_l0_default_active_minus1);
    put32(&mut d, 0x7c, pps.num_ref_idx_l1_default_active_minus1);
    put32(
        &mut d,
        0x80,
        u32::from(pps.deblocking_filter_control_present),
    );
    put32(&mut d, 0x84, u32::from(pps.redundant_pic_cnt_present));
    put32(&mut d, 0x88, u32::from(pps.transform_8x8_mode));
    put32(&mut d, 0x8c, lay.pitch_luma);
    put32(&mut d, 0x90, lay.pitch_chroma);
    put32(&mut d, 0x98, lay.luma_bot_offset);
    put32(&mut d, 0xa4, lay.chroma_bot_offset);
    put32(&mut d, 0xac, lay.hist_size);
    let s5 = |v: i32| (v.cast_unsigned()) & 0x1f;
    let w = u32::from(mbaff_frame)
        | u32::from(sps.direct_8x8_inference) << 1
        | u32::from(pps.weighted_pred) << 2
        | u32::from(pps.constrained_intra_pred) << 3
        | u32::from(inp.ref_pic) << 4
        | u32::from(inp.field_pic) << 5
        | u32::from(inp.bottom_field) << 6
        | u32::from(inp.second_field) << 7
        | (sps.log2_max_frame_num_minus4 & 0xf) << 8
        | (sps.chroma_format_idc & 3) << 12
        | (sps.pic_order_cnt_type & 3) << 14
        | ((pps.pic_init_qp_minus26.cast_unsigned()) & 0x3f) << 16
        | s5(pps.chroma_qp_index_offset) << 22
        | s5(pps.second_chroma_qp_index_offset) << 27;
    put32(&mut d, 0xb0, w);
    let w = (pps.weighted_bipred_idc & 3)
        | (inp.surface_id & 0x7f) << 2
        | (u32::from(inp.cur_slot) & 0x1f) << 9
        | (inp.frame_num & 0xffff) << 14
        | u32::from(ep.output_layout) << 31;
    put32(&mut d, 0xb4, w);
    put32(&mut d, 0xb8, inp.poc[0].cast_unsigned());
    put32(&mut d, 0xbc, inp.poc[1].cast_unsigned());
    // DPB entries: same 16-byte nvdec_dpb_entry_s encoding as the NEW_H264 setup at +0x40.
    let new = super::desc::build(inp, ep);
    d[0xc0..0x1c0].copy_from_slice(&new[0x40..0x140]);
    let s = &pps.scaling;
    for (i, m4) in s.m4.iter().enumerate() {
        d[0x1c0 + 16 * i..0x1d0 + 16 * i].copy_from_slice(m4);
    }
    d[0x220..0x260].copy_from_slice(&s.m8[0]);
    d[0x260..0x2a0].copy_from_slice(&s.m8[1]);
    // MVC fields stay zero. displayPara (0x2d4): only the histogram window end is written.
    put32(
        &mut d,
        0x2ec,
        ((sps.width() - 1) & 0xfff) | ((sps.height() - 1) & 0xfff) << 12,
    );
    for (i, &o) in slice_offsets.iter().enumerate() {
        put32(&mut d, SLICE_OFFSETS_OFFSET + 4 * i, o);
    }
    d
}
