//! HEVC (H.265) header parser that builds NVDEC per-picture descriptors.
//!
//! Port of tinygrad's reference parser (MIT), `extra/hevc/hevc.py`
//! `parse_hevc_file_headers` and everything it uses: NAL splitting,
//! emulation-prevention removal, SPS/PPS/slice-segment header parsing,
//! short-term reference picture sets, tile column/row sizes and the
//! per-frame descriptor: `nvdec_hevc_pic_s` (276 bytes) + the 12-byte
//! Blackwell extension, zero padded to 0x200, then tile sizes at 0x200,
//! zero padded to 0x300 bytes per frame. It replaces the offline Python
//! fixture generator: for the two V0 fixtures (x265 640x360, comma 1952x1216
//! with 3 tile columns) the output is byte-identical (see the
//! `NVDEC_FIXTURE_DIRS` test).
//!
//! Struct layout: NVIDIA's MIT header
//! `src/nouveau/headers/nvidia/video/nvdec_drv.h` (`nvdec_hevc_pic_s`),
//! bit positions checked against a gcc compile of that header (the table in
//! the tests is that program's output).
//!
//! Where this differs from hevc.py (all are identical on both fixtures):
//! - `se(v)` follows H.265 9.2.2. hevc.py computes `(-1 ** (k + 1)) * (k // 2)`,
//!   which by Python precedence is always `-(k // 2)` and mis-decodes odd `k`.
//! - SPS/PPS fields hevc.py parses (or could) but never stores are written:
//!   `amp_enable_flag` (hevc.py assigns a misspelt `amp_enabled_flag`
//!   attribute that is not a struct field), `max_transform_hierarchy_depth_*`,
//!   `separate_colour_plane_flag`, `dependent_slice_segments_enabled_flag`,
//!   `output_flag_present_flag`, `num_extra_slice_header_bits`,
//!   `constrained_intra_pred_flag`, `transform_skip_enabled_flag`,
//!   `slice_segment_header_extension_present_flag`.
//! - Streams hevc.py would silently mis-describe fail with
//!   `Error::Contract`: SPS `scaling_list_enabled_flag` (hevc.py never sets
//!   `scalingListEnable` nor uploads matrices), chroma formats other than 4:2:0
//!   (the crop arithmetic assumes SubWidthC = SubHeightC = 2), VCL NAL types
//!   other than TRAIL_N/TRAIL_R/IDR_W_RADL/IDR_N_LP (hevc.py drops CRA, BLA,
//!   RASL, RADL, TSA, STSA pictures without a word), and any value that does
//!   not fit its descriptor field (hevc.py ORs overflowing bits into the
//!   neighbouring bitfields or wraps ctypes integers).
//! - A dependent slice segment header stops after
//!   `dependent_slice_segment_flag` (H.265 7.3.6.1); hevc.py forces the flag to
//!   0 and parses the remainder of a header that does not exist. Only the first
//!   slice segment of a picture feeds the descriptor in both.
//! - Negative `pps_cb_qp_offset`/`pps_cr_qp_offset` are stored as two's
//!   complement `char` (hevc.py raises `TypeError` on them).
//!
//! Faithful hevc.py behaviour kept on purpose (do not "fix" without a
//! stock-driver trace): the slice header is read from the escaped NAL payload
//! (no emulation-prevention removal), so `sw_hdr_skip_length` counts escaped
//! bits; `num_bits_short_term_ref_pics_in_slice = max(0, skip - 9)`; POCs are
//! raw `slice_pic_order_cnt_lsb` (no MSB tracking); the DPB is a FIFO of
//! `sps_max_dec_pic_buffering[0]` entries; SPS/PPS NAL units update the
//! descriptor immediately, so a picture still pending flush sees them.

use crate::{nvdec::HevcFrame, rm::Error};
use std::mem::size_of;

type Result<T> = std::result::Result<T, Error>;

fn contract<T>(message: String) -> Result<T> {
    Err(Error::Contract(format!("HEVC: {message}")))
}

/// Bytes per frame descriptor (pic_s + Blackwell extension + tile sizes).
pub const DESCRIPTOR_STRIDE: usize = 0x300;
/// Offset of the tile size table inside a descriptor.
pub const TILE_SIZES_OFFSET: usize = 0x200;
/// Blackwell extension appended right after `nvdec_hevc_pic_s` (hevc.py).
const BLACKWELL_EXTENSION: [u8; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0];

// NAL unit types, H.265 Table 7-1 (FFmpeg `HEVC_NAL_*`).
const NAL_TRAIL_N: u8 = 0;
const NAL_TRAIL_R: u8 = 1;
const NAL_BLA_W_LP: u8 = 16;
const NAL_IDR_W_RADL: u8 = 19;
const NAL_IDR_N_LP: u8 = 20;
const NAL_RSV_IRAP_VCL23: u8 = 23;
const NAL_SPS: u8 = 33;
const NAL_PPS: u8 = 34;
/// slice_type, H.265 Table 7-7 (FFmpeg `HEVC_SLICE_B`).
const SLICE_B: u32 = 0;

fn is_idr(nal_type: u8) -> bool {
    nal_type == NAL_IDR_W_RADL || nal_type == NAL_IDR_N_LP
}

fn is_irap(nal_type: u8) -> bool {
    (NAL_BLA_W_LP..=NAL_RSV_IRAP_VCL23).contains(&nal_type)
}

/// Picture is kept as a reference for later pictures (hevc.py `is_hist`).
fn is_reference(nal_type: u8) -> bool {
    nal_type == NAL_TRAIL_R || is_idr(nal_type)
}

const fn round_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

/// MSB-first bit reader over an RBSP (H.265 7.2 `read_bits`, 9.2 Exp-Golomb).
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits consumed so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// `u(n)`, n <= 32.
    pub fn u(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return contract(format!("u({n}) wider than 32 bits"));
        }
        let end = self.pos + n as usize;
        if end > self.data.len() * 8 {
            return contract(format!(
                "bitstream truncated: need bits {}..{end}, have {}",
                self.pos,
                self.data.len() * 8
            ));
        }
        let mut v = 0u64;
        for bit in self.pos..end {
            v = v << 1 | u64::from(self.data[bit / 8] >> (7 - bit % 8) & 1);
        }
        self.pos = end;
        Ok(v as u32)
    }

    fn flag(&mut self) -> Result<bool> {
        Ok(self.u(1)? == 1)
    }

    /// Skips `n` bits.
    pub fn skip(&mut self, mut n: u32) -> Result<()> {
        while n > 0 {
            let step = n.min(32);
            self.u(step)?;
            n -= step;
        }
        Ok(())
    }

    /// `ue(v)`, 9.2: 0th-order Exp-Golomb.
    pub fn ue(&mut self) -> Result<u32> {
        let mut leading_zero_bits = 0u32;
        while self.u(1)? == 0 {
            leading_zero_bits += 1;
            if leading_zero_bits > 31 {
                return contract("ue(v) longer than 32 bits".into());
            }
        }
        let rest = u64::from(self.u(leading_zero_bits)?);
        Ok(((1u64 << leading_zero_bits) - 1 + rest) as u32)
    }

    /// `se(v)`, 9.2.2: k -> (-1)^(k+1) * Ceil(k / 2).
    pub fn se(&mut self) -> Result<i64> {
        let k = i64::from(self.ue()?);
        Ok(if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) })
    }
}

/// 7.3.1.1 / 7.4.2: removes `emulation_prevention_three_byte`s
/// (`00 00 03` -> `00 00`), same scan as hevc.py `_hevc_get_rbsp`.
pub fn nal_to_rbsp(nal: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(nal.len());
    let mut off = 0;
    while off < nal.len() {
        if off + 2 < nal.len() && nal[off..off + 3] == [0, 0, 3] {
            rbsp.extend_from_slice(&[0, 0]);
            off += 3;
        } else {
            rbsp.push(nal[off]);
            off += 1;
        }
    }
    rbsp
}

fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    data.get(from..)?
        .windows(3)
        .position(|w| w == [0, 0, 1])
        .map(|p| from + p)
}

/// Explicitly coded short-term RPS sizes (num_negative_pics, num_positive_pics).
type RpsSize = (u32, u32);

/// 7.3.7 st_ref_pic_set(stRpsIdx). `last_sps_rps` is the size of the last
/// explicitly coded set of the SPS (hevc.py reads `sps.num_negative_pics` /
/// `sps.num_positive_pics`, i.e. the last set parsed in the SPS); `None` while
/// parsing the SPS itself, where hevc.py has no SPS to read and fails.
fn st_ref_pic_set(
    r: &mut BitReader,
    idx: u32,
    num_sets: u32,
    last_sps_rps: Option<RpsSize>,
) -> Result<Option<RpsSize>> {
    let inter_ref_pic_set_prediction_flag = idx != 0 && r.flag()?;
    if inter_ref_pic_set_prediction_flag {
        if idx == num_sets {
            r.ue()?; // delta_idx_minus1
        }
        r.u(1)?; // delta_rps_sign
        r.ue()?; // abs_delta_rps_minus1
        let Some((neg, pos)) = last_sps_rps else {
            return contract(
                "inter-predicted short-term RPS inside the SPS is not supported".into(),
            );
        };
        for _ in 0..=u64::from(neg) + u64::from(pos) {
            if !r.flag()? {
                r.u(1)?; // use_delta_flag
            }
        }
        Ok(None)
    } else {
        let neg = r.ue()?;
        let pos = r.ue()?;
        for _ in 0..u64::from(neg) + u64::from(pos) {
            r.ue()?; // delta_poc_sX_minus1
            r.u(1)?; // used_by_curr_pic_sX_flag
        }
        Ok(Some((neg, pos)))
    }
}

/// 7.3.2.2 seq_parameter_set_rbsp, up to `strong_intra_smoothing_enabled_flag`.
#[derive(Clone, Debug)]
struct Sps {
    chroma_format_idc: u32,
    separate_colour_plane_flag: u32,
    pic_width_in_luma_samples: u32,
    pic_height_in_luma_samples: u32,
    conf_win_offsets: [u32; 4], // left, right, top, bottom
    bit_depth_luma: u32,
    bit_depth_chroma: u32,
    log2_max_pic_order_cnt_lsb_minus4: u32,
    sps_max_dec_pic_buffering: u32,
    log2_min_luma_coding_block_size: u32,
    log2_max_luma_coding_block_size: u32,
    log2_min_transform_block_size: u32,
    log2_max_transform_block_size: u32,
    max_transform_hierarchy_depth_inter: u32,
    max_transform_hierarchy_depth_intra: u32,
    amp_enabled_flag: u32,
    sample_adaptive_offset_enabled_flag: u32,
    pcm_enabled_flag: u32,
    num_short_term_ref_pic_sets: u32,
    last_rps: Option<RpsSize>,
    sps_temporal_mvp_enabled_flag: u32,
    strong_intra_smoothing_enabled_flag: u32,
}

fn parse_sps(r: &mut BitReader) -> Result<Sps> {
    r.u(4)?; // sps_video_parameter_set_id
    let sps_max_sub_layers_minus1 = r.u(3)?;
    r.u(1)?; // sps_temporal_id_nesting_flag
             // 7.3.3 profile_tier_level(1, 0)
    if sps_max_sub_layers_minus1 != 0 {
        return contract(format!(
            "sps_max_sub_layers_minus1 = {sps_max_sub_layers_minus1}: sub-layers are not supported"
        ));
    }
    r.skip(88)?; // general profile/tier/flags
    r.u(8)?; // general_level_idc
    r.ue()?; // sps_seq_parameter_set_id
    let chroma_format_idc = r.ue()?;
    let separate_colour_plane_flag = if chroma_format_idc == 3 { r.u(1)? } else { 0 };
    let pic_width_in_luma_samples = r.ue()?;
    let pic_height_in_luma_samples = r.ue()?;
    let mut conf_win_offsets = [0; 4];
    if r.flag()? {
        for o in &mut conf_win_offsets {
            *o = r.ue()?;
        }
    }
    let bit_depth_luma = r.ue()? + 8;
    let bit_depth_chroma = r.ue()? + 8;
    let log2_max_pic_order_cnt_lsb_minus4 = r.ue()?;
    // sps_sub_layer_ordering_info_present_flag: with max_sub_layers_minus1 == 0
    // exactly one entry follows either way.
    r.u(1)?;
    let sps_max_dec_pic_buffering = r.ue()? + 1;
    r.ue()?; // sps_max_num_reorder_pics
    r.ue()?; // sps_max_latency_increase_plus1
    let log2_min_luma_coding_block_size = r.ue()? + 3;
    let log2_max_luma_coding_block_size = log2_min_luma_coding_block_size + r.ue()?;
    let log2_min_transform_block_size = r.ue()? + 2;
    let log2_max_transform_block_size = log2_min_transform_block_size + r.ue()?;
    let max_transform_hierarchy_depth_inter = r.ue()?;
    let max_transform_hierarchy_depth_intra = r.ue()?;
    if r.flag()? {
        let data_present = r.u(1)?;
        return contract(format!(
            "scaling_list_enabled_flag = 1 (sps_scaling_list_data_present_flag = {data_present}): \
             scaling lists are not supported"
        ));
    }
    let amp_enabled_flag = r.u(1)?;
    let sample_adaptive_offset_enabled_flag = r.u(1)?;
    let pcm_enabled_flag = r.u(1)?;
    if pcm_enabled_flag != 0 {
        return contract("pcm_enabled_flag = 1: PCM is not supported".into());
    }
    let num_short_term_ref_pic_sets = r.ue()?;
    let mut last_rps = None;
    for i in 0..num_short_term_ref_pic_sets {
        last_rps = st_ref_pic_set(r, i, num_short_term_ref_pic_sets, None)?;
    }
    if r.flag()? {
        return contract(
            "long_term_ref_pics_present_flag = 1: long-term references are not supported".into(),
        );
    }
    let sps_temporal_mvp_enabled_flag = r.u(1)?;
    let strong_intra_smoothing_enabled_flag = r.u(1)?;
    if chroma_format_idc != 1 {
        return contract(format!(
            "chroma_format_idc = {chroma_format_idc}: only 4:2:0 is supported"
        ));
    }
    if pic_width_in_luma_samples == 0 || pic_height_in_luma_samples == 0 {
        return contract("SPS with a zero picture dimension".into());
    }
    if log2_max_luma_coding_block_size > 6 {
        return contract(format!(
            "CtbLog2SizeY = {log2_max_luma_coding_block_size} is out of range"
        ));
    }
    Ok(Sps {
        chroma_format_idc,
        separate_colour_plane_flag,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        conf_win_offsets,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_pic_order_cnt_lsb_minus4,
        sps_max_dec_pic_buffering,
        log2_min_luma_coding_block_size,
        log2_max_luma_coding_block_size,
        log2_min_transform_block_size,
        log2_max_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        amp_enabled_flag,
        sample_adaptive_offset_enabled_flag,
        pcm_enabled_flag,
        num_short_term_ref_pic_sets,
        last_rps,
        sps_temporal_mvp_enabled_flag,
        strong_intra_smoothing_enabled_flag,
    })
}

#[derive(Clone, Debug, Default)]
struct Tiles {
    num_tile_columns_minus1: u32,
    num_tile_rows_minus1: u32,
    uniform_spacing_flag: bool,
    column_width_minus1: Vec<u32>,
    row_height_minus1: Vec<u32>,
    loop_filter_across_tiles_enabled_flag: u32,
}

/// 7.3.2.3 pic_parameter_set_rbsp, up to
/// `slice_segment_header_extension_present_flag`.
#[derive(Clone, Debug)]
struct Pps {
    dependent_slice_segments_enabled_flag: u32,
    output_flag_present_flag: u32,
    num_extra_slice_header_bits: u32,
    sign_data_hiding_enabled_flag: u32,
    cabac_init_present_flag: u32,
    num_ref_idx_l0_default_active: u32,
    num_ref_idx_l1_default_active: u32,
    init_qp: i64,
    constrained_intra_pred_flag: u32,
    transform_skip_enabled_flag: u32,
    cu_qp_delta_enabled_flag: u32,
    diff_cu_qp_delta_depth: u32,
    pps_cb_qp_offset: i64,
    pps_cr_qp_offset: i64,
    pps_slice_chroma_qp_offsets_present_flag: u32,
    weighted_pred_flag: u32,
    weighted_bipred_flag: u32,
    transquant_bypass_enabled_flag: u32,
    entropy_coding_sync_enabled_flag: u32,
    tiles: Option<Tiles>,
    loop_filter_across_slices_enabled_flag: u32,
    deblocking_filter_control_present_flag: u32,
    lists_modification_present_flag: u32,
    log2_parallel_merge_level: u32,
    slice_segment_header_extension_present_flag: u32,
}

fn parse_pps(r: &mut BitReader) -> Result<Pps> {
    r.ue()?; // pps_pic_parameter_set_id
    r.ue()?; // pps_seq_parameter_set_id
    let dependent_slice_segments_enabled_flag = r.u(1)?;
    let output_flag_present_flag = r.u(1)?;
    let num_extra_slice_header_bits = r.u(3)?;
    let sign_data_hiding_enabled_flag = r.u(1)?;
    let cabac_init_present_flag = r.u(1)?;
    let num_ref_idx_l0_default_active = r.ue()? + 1;
    let num_ref_idx_l1_default_active = r.ue()? + 1;
    let init_qp = r.se()? + 26;
    let constrained_intra_pred_flag = r.u(1)?;
    let transform_skip_enabled_flag = r.u(1)?;
    let cu_qp_delta_enabled_flag = r.u(1)?;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag != 0 {
        r.ue()?
    } else {
        0
    };
    let pps_cb_qp_offset = r.se()?;
    let pps_cr_qp_offset = r.se()?;
    let pps_slice_chroma_qp_offsets_present_flag = r.u(1)?;
    let weighted_pred_flag = r.u(1)?;
    let weighted_bipred_flag = r.u(1)?;
    let transquant_bypass_enabled_flag = r.u(1)?;
    let tiles_enabled_flag = r.flag()?;
    let entropy_coding_sync_enabled_flag = r.u(1)?;
    let tiles = if tiles_enabled_flag {
        let mut t = Tiles {
            num_tile_columns_minus1: r.ue()?,
            num_tile_rows_minus1: r.ue()?,
            uniform_spacing_flag: r.flag()?,
            ..Tiles::default()
        };
        if !t.uniform_spacing_flag {
            for _ in 0..t.num_tile_columns_minus1 {
                t.column_width_minus1.push(r.ue()?);
            }
            for _ in 0..t.num_tile_rows_minus1 {
                t.row_height_minus1.push(r.ue()?);
            }
        }
        t.loop_filter_across_tiles_enabled_flag = r.u(1)?;
        Some(t)
    } else {
        None
    };
    let loop_filter_across_slices_enabled_flag = r.u(1)?;
    let deblocking_filter_control_present_flag = r.u(1)?;
    if deblocking_filter_control_present_flag != 0 {
        return contract("deblocking_filter_control_present_flag = 1 is not supported".into());
    }
    if r.flag()? {
        return contract(
            "pps_scaling_list_data_present_flag = 1: scaling lists are not supported".into(),
        );
    }
    let lists_modification_present_flag = r.u(1)?;
    let log2_parallel_merge_level = r.ue()? + 2;
    let slice_segment_header_extension_present_flag = r.u(1)?;
    Ok(Pps {
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active,
        num_ref_idx_l1_default_active,
        init_qp,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        entropy_coding_sync_enabled_flag,
        tiles,
        loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        lists_modification_present_flag,
        log2_parallel_merge_level,
        slice_segment_header_extension_present_flag,
    })
}

/// 7.3.6.1 slice_segment_header, up to `num_ref_idx_active_override_flag`.
/// Bit positions count bits of the escaped NAL payload after the 2-byte NAL
/// header (hevc.py parses the slice header without emulation removal).
#[derive(Clone, Debug)]
struct SliceHeader {
    first_slice_segment_in_pic_flag: bool,
    slice_type: u32,
    slice_pic_order_cnt_lsb: u32,
    sw_skip_start: usize,
    sw_skip_end: usize,
}

fn parse_slice_header(
    r: &mut BitReader,
    nal_type: u8,
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceHeader> {
    let first_slice_segment_in_pic_flag = r.flag()?;
    if is_irap(nal_type) {
        r.u(1)?; // no_output_of_prior_pics_flag
    }
    r.ue()?; // slice_pic_parameter_set_id
    let mut dependent_slice_segment_flag = false;
    if !first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag != 0 {
            dependent_slice_segment_flag = r.flag()?;
        }
        r.ue()?; // slice_segment_address
    }
    let mut hdr = SliceHeader {
        first_slice_segment_in_pic_flag,
        slice_type: 0,
        slice_pic_order_cnt_lsb: 0,
        sw_skip_start: r.position(),
        sw_skip_end: r.position(),
    };
    if dependent_slice_segment_flag {
        return Ok(hdr);
    }
    r.u(pps.num_extra_slice_header_bits)?; // slice_reserved_flag[]
    hdr.slice_type = r.ue()?;
    hdr.sw_skip_start = r.position();
    if pps.output_flag_present_flag != 0 {
        r.u(1)?; // pic_output_flag
    }
    if sps.separate_colour_plane_flag != 0 {
        r.u(2)?; // colour_plane_id
    }
    if !is_idr(nal_type) {
        hdr.slice_pic_order_cnt_lsb = r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)?;
        if !r.flag()? {
            // short_term_ref_pic_set_sps_flag == 0
            st_ref_pic_set(
                r,
                sps.num_short_term_ref_pic_sets,
                sps.num_short_term_ref_pic_sets,
                sps.last_rps,
            )?;
        } else if sps.num_short_term_ref_pic_sets > 1 {
            return contract(
                "short_term_ref_pic_set_idx is not supported (num_short_term_ref_pic_sets > 1)"
                    .into(),
            );
        }
        hdr.sw_skip_end = r.position();
        if sps.sps_temporal_mvp_enabled_flag != 0 {
            r.u(1)?; // slice_temporal_mvp_enabled_flag
        }
    } else {
        hdr.sw_skip_end = hdr.sw_skip_start;
    }
    if sps.sample_adaptive_offset_enabled_flag != 0 {
        r.u(1)?; // slice_sao_luma_flag
        let chroma_array_type = if sps.separate_colour_plane_flag == 0 {
            sps.chroma_format_idc
        } else {
            0
        };
        if chroma_array_type != 0 {
            r.u(1)?; // slice_sao_chroma_flag
        }
    }
    // hevc.py tests `slice_type in {B, B}`: the override is only parsed for B
    // slices; nothing after it is used.
    if hdr.slice_type == SLICE_B && r.flag()? {
        r.ue()?; // num_ref_idx_l0_active_minus1
        r.ue()?; // num_ref_idx_l1_active_minus1
    }
    Ok(hdr)
}

/// A C bitfield inside a 32-bit (or 8-bit) storage unit of `nvdec_hevc_pic_s`.
#[derive(Clone, Copy, Debug)]
pub struct BitField {
    pub shift: u32,
    pub width: u32,
}

const fn bf(shift: u32, width: u32) -> BitField {
    BitField { shift, width }
}

/// Bitfields of `nvdec_hevc_pic_s` (nvdec_drv.h), by storage unit.
pub mod fields {
    use super::{bf, BitField};
    // `general_bits` (u8 at byte 64)
    pub const TILEFORMAT: BitField = bf(0, 2);
    pub const GOB_HEIGHT: BitField = bf(2, 3);
    pub const RESERVERD_SURFACE_FORMAT: BitField = bf(5, 3);
    // `encryption_bits` (byte 56)
    pub const KEY_INCREMENT: BitField = bf(0, 6);
    pub const ENCRYPTION_MODE: BitField = bf(6, 4);
    pub const KEY_SLOT_INDEX: BitField = bf(10, 4);
    pub const SSM_EN: BitField = bf(14, 1);
    pub const ENABLE_HISTOGRAM: BitField = bf(15, 1);
    pub const ENABLE_SUBSTREAM_DECODING: BitField = bf(16, 1);
    pub const RESERVED0: BitField = bf(17, 15);
    // `sps_bits0` (byte 92)
    pub const CHROMA_FORMAT_IDC: BitField = bf(0, 4);
    pub const BIT_DEPTH_LUMA: BitField = bf(4, 4);
    pub const BIT_DEPTH_CHROMA: BitField = bf(8, 4);
    pub const LOG2_MIN_LUMA_CODING_BLOCK_SIZE: BitField = bf(12, 4);
    pub const LOG2_MAX_LUMA_CODING_BLOCK_SIZE: BitField = bf(16, 4);
    pub const LOG2_MIN_TRANSFORM_BLOCK_SIZE: BitField = bf(20, 4);
    pub const LOG2_MAX_TRANSFORM_BLOCK_SIZE: BitField = bf(24, 4);
    pub const RESERVED2: BitField = bf(28, 4);
    // `sps_bits1` (byte 96)
    pub const MAX_TRANSFORM_HIERARCHY_DEPTH_INTER: BitField = bf(0, 3);
    pub const MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA: BitField = bf(3, 3);
    pub const SCALING_LIST_ENABLE: BitField = bf(6, 1);
    pub const AMP_ENABLE_FLAG: BitField = bf(7, 1);
    pub const SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG: BitField = bf(8, 1);
    pub const PCM_ENABLED_FLAG: BitField = bf(9, 1);
    pub const PCM_SAMPLE_BIT_DEPTH_LUMA: BitField = bf(10, 4);
    pub const PCM_SAMPLE_BIT_DEPTH_CHROMA: BitField = bf(14, 4);
    pub const LOG2_MIN_PCM_LUMA_CODING_BLOCK_SIZE: BitField = bf(18, 4);
    pub const LOG2_MAX_PCM_LUMA_CODING_BLOCK_SIZE: BitField = bf(22, 4);
    pub const PCM_LOOP_FILTER_DISABLED_FLAG: BitField = bf(26, 1);
    pub const SPS_TEMPORAL_MVP_ENABLED_FLAG: BitField = bf(27, 1);
    pub const STRONG_INTRA_SMOOTHING_ENABLED_FLAG: BitField = bf(28, 1);
    pub const RESERVED3: BitField = bf(29, 3);
    // `pps_bits0` (byte 100)
    pub const DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG: BitField = bf(0, 1);
    pub const OUTPUT_FLAG_PRESENT_FLAG: BitField = bf(1, 1);
    pub const NUM_EXTRA_SLICE_HEADER_BITS: BitField = bf(2, 3);
    pub const SIGN_DATA_HIDING_ENABLED_FLAG: BitField = bf(5, 1);
    pub const CABAC_INIT_PRESENT_FLAG: BitField = bf(6, 1);
    pub const NUM_REF_IDX_L0_DEFAULT_ACTIVE: BitField = bf(7, 4);
    pub const NUM_REF_IDX_L1_DEFAULT_ACTIVE: BitField = bf(11, 4);
    pub const INIT_QP: BitField = bf(15, 7);
    pub const CONSTRAINED_INTRA_PRED_FLAG: BitField = bf(22, 1);
    pub const TRANSFORM_SKIP_ENABLED_FLAG: BitField = bf(23, 1);
    pub const CU_QP_DELTA_ENABLED_FLAG: BitField = bf(24, 1);
    pub const DIFF_CU_QP_DELTA_DEPTH: BitField = bf(25, 2);
    pub const RESERVED4: BitField = bf(27, 5);
    // `pps_bits1` (byte 108)
    pub const PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG: BitField = bf(0, 1);
    pub const WEIGHTED_PRED_FLAG: BitField = bf(1, 1);
    pub const WEIGHTED_BIPRED_FLAG: BitField = bf(2, 1);
    pub const TRANSQUANT_BYPASS_ENABLED_FLAG: BitField = bf(3, 1);
    pub const TILES_ENABLED_FLAG: BitField = bf(4, 1);
    pub const ENTROPY_CODING_SYNC_ENABLED_FLAG: BitField = bf(5, 1);
    pub const NUM_TILE_COLUMNS: BitField = bf(6, 5);
    pub const NUM_TILE_ROWS: BitField = bf(11, 5);
    pub const LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG: BitField = bf(16, 1);
    pub const LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: BitField = bf(17, 1);
    pub const DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG: BitField = bf(18, 1);
    pub const DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG: BitField = bf(19, 1);
    pub const PPS_DEBLOCKING_FILTER_DISABLED_FLAG: BitField = bf(20, 1);
    pub const LISTS_MODIFICATION_PRESENT_FLAG: BitField = bf(21, 1);
    pub const LOG2_PARALLEL_MERGE_LEVEL: BitField = bf(22, 3);
    pub const SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG: BitField = bf(25, 1);
    pub const RESERVED5: BitField = bf(26, 6);
    // `dxva_bits` (byte 212)
    pub const SEPARATE_COLOUR_PLANE_FLAG: BitField = bf(0, 1);
    pub const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: BitField = bf(1, 4);
    pub const NUM_SHORT_TERM_REF_PIC_SETS: BitField = bf(5, 7);
    pub const NUM_LONG_TERM_REF_PICS_SPS: BitField = bf(12, 6);
    pub const BBIT_PARSING_DISABLE: BitField = bf(18, 1);
    pub const NUM_DELTA_POCS_OF_RPS_IDX: BitField = bf(19, 8);
    pub const LONG_TERM_REF_PICS_PRESENT_FLAG: BitField = bf(27, 1);
    pub const RESERVED_DXVA: BitField = bf(28, 4);
    // `v1_bits` (byte 248)
    pub const V1_SW_SKIP_START_LENGTH: BitField = bf(0, 14);
    pub const V1_EXTERNAL_REF_MEM_DIS: BitField = bf(14, 1);
    pub const V1_ERROR_RECOVERY_START_POS: BitField = bf(15, 2);
    pub const V1_ERROR_EXTERNAL_MV_EN: BitField = bf(17, 1);
    pub const V1_RESERVED0: BitField = bf(18, 14);
    // `v3_bits` (byte 256)
    pub const V3_SLICE_DECODING_ENABLE: BitField = bf(0, 1);
}

/// `nvdec_hevc_pic_s` (nvdec_drv.h, 276 bytes). Bitfield groups are stored as
/// their C storage units; the `fields` module documents bit positions.
/// Nested structs are flattened (`v1_*`, `v2_*`, `v3_*`); the never-written
/// `nvdec_ecdma_config_s` and `nvdec_pass2_otf_ext_s` are opaque words.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvdecHevcPicS {
    pub wrapped_session_key: [u32; 4],
    pub wrapped_content_key: [u32; 4],
    pub initialization_vector: [u32; 4],
    pub stream_len: u32,
    pub enable_encryption: u32,
    /// key_increment .. reserved0
    pub encryption_bits: u32,
    pub gptimer_timeout_value: u32,
    /// tileformat:2 gob_height:3 reserverd_surface_format:3
    pub general_bits: u8,
    pub sw_start_code_e: u8,
    pub disp_output_mode: u8,
    pub reserved1: u8,
    pub framestride: [u32; 2],
    pub col_mv_buffersize: u32,
    pub hevc_sao_buffer_offset: u32,
    pub hevc_bsd_ctrl_offset: u32,
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    /// chroma_format_idc .. reserved2
    pub sps_bits0: u32,
    /// max_transform_hierarchy_depth_inter .. reserved3
    pub sps_bits1: u32,
    /// dependent_slice_segments_enabled_flag .. reserved4
    pub pps_bits0: u32,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_beta_offset: i8,
    pub pps_tc_offset: i8,
    /// pps_slice_chroma_qp_offsets_present_flag .. reserved5
    pub pps_bits1: u32,
    pub num_ref_frames: u8,
    pub reserved6: u8,
    pub longtermflag: u16,
    pub initreflistidxl0: [u8; 16],
    pub initreflistidxl1: [u8; 16],
    pub ref_diff_pic_order_cnts: [i16; 16],
    pub idr_picture_flag: u8,
    pub rap_picture_flag: u8,
    pub curr_pic_idx: u8,
    pub pattern_id: u8,
    pub sw_hdr_skip_length: u16,
    pub reserved7: u16,
    /// nvdec_ecdma_config_s (24 bytes)
    pub ecdma_cfg: [u32; 6],
    /// separate_colour_plane_flag .. reserved_dxva
    pub dxva_bits: u32,
    pub num_bits_short_term_ref_pics_in_slice: u32,
    /// v1.hevc_main10_444_ext sps extension bitfields
    pub v1_main10_444_sps_bits: u32,
    /// v1.hevc_main10_444_ext pps extension bitfields
    pub v1_main10_444_pps_bits: u32,
    pub v1_cb_qp_adjustment: [i8; 6],
    pub v1_cr_qp_adjustment: [i8; 6],
    pub v1_hevc_flt_above_offset: u32,
    pub v1_hevc_sao_above_offset: u32,
    /// v1.sw_skip_start_length .. v1.reserved0
    pub v1_bits: u32,
    /// nvdec_hevc_pic_v2_s (mv-hevc bitfields)
    pub v2_bits: u32,
    /// v3.slice_decoding_enable .. v3.reserved
    pub v3_bits: u32,
    pub v3_hevc_slice_edge_offset: u32,
    /// nvdec_pass2_otf_ext_s (12 bytes)
    pub ssm: [u32; 3],
}

pub const NVDEC_HEVC_PIC_S_SIZE: usize = 276;
const _: () = assert!(size_of::<NvdecHevcPicS>() == NVDEC_HEVC_PIC_S_SIZE);
const _: () = assert!(
    cfg!(target_endian = "little"),
    "descriptor bytes are little-endian"
);

fn set_bits(unit: &mut u32, field: BitField, value: i64, name: &str) -> Result<()> {
    if value < 0 || value >= 1i64 << field.width {
        return contract(format!(
            "{name} = {value} does not fit its {}-bit descriptor field",
            field.width
        ));
    }
    let mask = ((1u64 << field.width) - 1) as u32;
    *unit = (*unit & !(mask << field.shift)) | ((value as u32) << field.shift);
    Ok(())
}

fn fit<T: TryFrom<i64>>(value: i64, name: &str) -> Result<T> {
    T::try_from(value).or_else(|_| {
        contract(format!(
            "{name} = {value} does not fit its {}-byte descriptor field",
            size_of::<T>()
        ))
    })
}

impl NvdecHevcPicS {
    /// hevc.py's initial context: `gptimer_timeout_value=92720000,
    /// tileformat=1, sw_start_code_e=1, pattern_id=2`, everything else zero.
    pub fn initial() -> Self {
        Self {
            gptimer_timeout_value: 92_720_000,
            general_bits: 1, // tileformat = 1 (KBL)
            sw_start_code_e: 1,
            pattern_id: 2,
            ..Self::default()
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C), only integer fields and arrays laid out without
        // padding (size asserted equal to the C sizeof), so every byte is
        // initialized; the borrow keeps `self` alive.
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u8>(), size_of::<Self>()) }
    }

    /// `fill_sps_into_dev_context` plus the SPS fields hevc.py leaves unset.
    fn fill_sps(&mut self, sps: &Sps) -> Result<()> {
        use fields::*;
        self.pic_width_in_luma_samples = fit(
            sps.pic_width_in_luma_samples.into(),
            "pic_width_in_luma_samples",
        )?;
        self.pic_height_in_luma_samples = fit(
            sps.pic_height_in_luma_samples.into(),
            "pic_height_in_luma_samples",
        )?;
        let s0: [(BitField, u32, &str); 7] = [
            (
                CHROMA_FORMAT_IDC,
                sps.chroma_format_idc,
                "chroma_format_idc",
            ),
            (BIT_DEPTH_LUMA, sps.bit_depth_luma, "bit_depth_luma"),
            (BIT_DEPTH_CHROMA, sps.bit_depth_chroma, "bit_depth_chroma"),
            (
                LOG2_MIN_LUMA_CODING_BLOCK_SIZE,
                sps.log2_min_luma_coding_block_size,
                "log2_min_luma_coding_block_size",
            ),
            (
                LOG2_MAX_LUMA_CODING_BLOCK_SIZE,
                sps.log2_max_luma_coding_block_size,
                "log2_max_luma_coding_block_size",
            ),
            (
                LOG2_MIN_TRANSFORM_BLOCK_SIZE,
                sps.log2_min_transform_block_size,
                "log2_min_transform_block_size",
            ),
            (
                LOG2_MAX_TRANSFORM_BLOCK_SIZE,
                sps.log2_max_transform_block_size,
                "log2_max_transform_block_size",
            ),
        ];
        for (f, v, n) in s0 {
            set_bits(&mut self.sps_bits0, f, v.into(), n)?;
        }
        let s1: [(BitField, u32, &str); 7] = [
            (
                MAX_TRANSFORM_HIERARCHY_DEPTH_INTER,
                sps.max_transform_hierarchy_depth_inter,
                "max_transform_hierarchy_depth_inter",
            ),
            (
                MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA,
                sps.max_transform_hierarchy_depth_intra,
                "max_transform_hierarchy_depth_intra",
            ),
            (AMP_ENABLE_FLAG, sps.amp_enabled_flag, "amp_enabled_flag"),
            (PCM_ENABLED_FLAG, sps.pcm_enabled_flag, "pcm_enabled_flag"),
            (
                SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG,
                sps.sample_adaptive_offset_enabled_flag,
                "sample_adaptive_offset_enabled_flag",
            ),
            (
                SPS_TEMPORAL_MVP_ENABLED_FLAG,
                sps.sps_temporal_mvp_enabled_flag,
                "sps_temporal_mvp_enabled_flag",
            ),
            (
                STRONG_INTRA_SMOOTHING_ENABLED_FLAG,
                sps.strong_intra_smoothing_enabled_flag,
                "strong_intra_smoothing_enabled_flag",
            ),
        ];
        for (f, v, n) in s1 {
            set_bits(&mut self.sps_bits1, f, v.into(), n)?;
        }
        set_bits(
            &mut self.dxva_bits,
            LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
            sps.log2_max_pic_order_cnt_lsb_minus4.into(),
            "log2_max_pic_order_cnt_lsb_minus4",
        )?;
        set_bits(
            &mut self.dxva_bits,
            SEPARATE_COLOUR_PLANE_FLAG,
            sps.separate_colour_plane_flag.into(),
            "separate_colour_plane_flag",
        )
    }

    /// `fill_pps_into_dev_context` plus the PPS fields hevc.py leaves unset.
    fn fill_pps(&mut self, pps: &Pps) -> Result<()> {
        use fields::*;
        let tiles = pps.tiles.as_ref();
        let p0: [(BitField, i64, &str); 11] = [
            (
                DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG,
                pps.dependent_slice_segments_enabled_flag.into(),
                "dependent_slice_segments_enabled_flag",
            ),
            (
                OUTPUT_FLAG_PRESENT_FLAG,
                pps.output_flag_present_flag.into(),
                "output_flag_present_flag",
            ),
            (
                NUM_EXTRA_SLICE_HEADER_BITS,
                pps.num_extra_slice_header_bits.into(),
                "num_extra_slice_header_bits",
            ),
            (
                SIGN_DATA_HIDING_ENABLED_FLAG,
                pps.sign_data_hiding_enabled_flag.into(),
                "sign_data_hiding_enabled_flag",
            ),
            (
                CABAC_INIT_PRESENT_FLAG,
                pps.cabac_init_present_flag.into(),
                "cabac_init_present_flag",
            ),
            (
                NUM_REF_IDX_L0_DEFAULT_ACTIVE,
                pps.num_ref_idx_l0_default_active.into(),
                "num_ref_idx_l0_default_active",
            ),
            (
                NUM_REF_IDX_L1_DEFAULT_ACTIVE,
                pps.num_ref_idx_l1_default_active.into(),
                "num_ref_idx_l1_default_active",
            ),
            (INIT_QP, pps.init_qp, "init_qp"),
            (
                CONSTRAINED_INTRA_PRED_FLAG,
                pps.constrained_intra_pred_flag.into(),
                "constrained_intra_pred_flag",
            ),
            (
                TRANSFORM_SKIP_ENABLED_FLAG,
                pps.transform_skip_enabled_flag.into(),
                "transform_skip_enabled_flag",
            ),
            (
                CU_QP_DELTA_ENABLED_FLAG,
                pps.cu_qp_delta_enabled_flag.into(),
                "cu_qp_delta_enabled_flag",
            ),
        ];
        for (f, v, n) in p0 {
            set_bits(&mut self.pps_bits0, f, v, n)?;
        }
        set_bits(
            &mut self.pps_bits0,
            DIFF_CU_QP_DELTA_DEPTH,
            pps.diff_cu_qp_delta_depth.into(),
            "diff_cu_qp_delta_depth",
        )?;
        self.pps_cb_qp_offset = fit(pps.pps_cb_qp_offset, "pps_cb_qp_offset")?;
        self.pps_cr_qp_offset = fit(pps.pps_cr_qp_offset, "pps_cr_qp_offset")?;
        let p1: [(BitField, u32, &str); 11] = [
            (
                PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG,
                pps.pps_slice_chroma_qp_offsets_present_flag,
                "pps_slice_chroma_qp_offsets_present_flag",
            ),
            (
                WEIGHTED_PRED_FLAG,
                pps.weighted_pred_flag,
                "weighted_pred_flag",
            ),
            (
                WEIGHTED_BIPRED_FLAG,
                pps.weighted_bipred_flag,
                "weighted_bipred_flag",
            ),
            (
                TRANSQUANT_BYPASS_ENABLED_FLAG,
                pps.transquant_bypass_enabled_flag,
                "transquant_bypass_enabled_flag",
            ),
            (
                TILES_ENABLED_FLAG,
                u32::from(tiles.is_some()),
                "tiles_enabled_flag",
            ),
            (
                ENTROPY_CODING_SYNC_ENABLED_FLAG,
                pps.entropy_coding_sync_enabled_flag,
                "entropy_coding_sync_enabled_flag",
            ),
            (
                LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
                pps.loop_filter_across_slices_enabled_flag,
                "loop_filter_across_slices_enabled_flag",
            ),
            (
                DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG,
                pps.deblocking_filter_control_present_flag,
                "deblocking_filter_control_present_flag",
            ),
            (
                LISTS_MODIFICATION_PRESENT_FLAG,
                pps.lists_modification_present_flag,
                "lists_modification_present_flag",
            ),
            (
                LOG2_PARALLEL_MERGE_LEVEL,
                pps.log2_parallel_merge_level,
                "log2_parallel_merge_level",
            ),
            (
                LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG,
                tiles.map_or(0, |t| t.loop_filter_across_tiles_enabled_flag),
                "loop_filter_across_tiles_enabled_flag",
            ),
        ];
        for (f, v, n) in p1 {
            set_bits(&mut self.pps_bits1, f, v.into(), n)?;
        }
        set_bits(
            &mut self.pps_bits1,
            SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG,
            pps.slice_segment_header_extension_present_flag.into(),
            "slice_segment_header_extension_present_flag",
        )
    }

    fn num_tile_columns(&self) -> u32 {
        self.pps_bits1 >> fields::NUM_TILE_COLUMNS.shift & 0x1f
    }

    fn num_tile_rows(&self) -> u32 {
        self.pps_bits1 >> fields::NUM_TILE_ROWS.shift & 0x1f
    }
}

/// Parsed stream: frame table, one `DESCRIPTOR_STRIDE`-byte descriptor per
/// frame, and geometry (`luma_*`: coded size, `width`/`height`: cropped).
pub struct HevcParsed {
    pub frames: Vec<HevcFrame>,
    /// Number of reference pictures (DPB entries) of each frame, hevc.py `hist`.
    pub hist: Vec<usize>,
    pub descriptors: Vec<u8>,
    pub luma_w: usize,
    pub luma_h: usize,
    pub width: usize,
    pub height: usize,
}

struct SliceNal {
    hdr: SliceHeader,
    nal_type: u8,
    offset: usize,
    len: usize,
}

struct Parser {
    ctx: NvdecHevcPicS,
    /// DPB: (curr_pic_idx, slice_pic_order_cnt_lsb), oldest first.
    history: Vec<(u8, u32)>,
    pending: Vec<SliceNal>,
    out: HevcParsed,
}

impl Parser {
    /// hevc.py `_flush_picture`, run with the SPS/PPS current at flush time.
    fn flush(&mut self, sps: &Sps, pps: &Pps) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let nals = std::mem::take(&mut self.pending);
        let hdr = &nals[0].hdr;
        let nal_type = nals[0].nal_type;
        if nals.iter().any(|n| n.nal_type != nal_type) {
            return contract(format!(
                "picture at offset {} mixes NAL unit types",
                nals[0].offset
            ));
        }
        let ctx = &mut self.ctx;
        let curr_pic_idx = (0u8..16)
            .find(|i| self.history.iter().all(|&(idx, _)| idx != *i))
            .ok_or_else(|| Error::Contract("HEVC: no free picture slot among 16".into()))?;
        ctx.curr_pic_idx = curr_pic_idx;
        if is_idr(nal_type) {
            self.history.clear();
        }
        ctx.num_ref_frames = fit(self.history.len() as i64, "num_ref_frames")?;
        ctx.idr_picture_flag = u8::from(is_idr(nal_type));
        ctx.rap_picture_flag = u8::from(is_irap(nal_type));
        ctx.ref_diff_pic_order_cnts = [0; 16];

        let width = u64::from(sps.pic_width_in_luma_samples);
        let height = u64::from(sps.pic_height_in_luma_samples);
        ctx.col_mv_buffersize = fit(
            (round_up(width, 64) * round_up(height, 64) / 16 / 256) as i64,
            "colMvBuffersize",
        )?;
        let stride: u32 = fit(round_up(width, 64) as i64, "framestride")?;
        ctx.framestride = [stride, stride];
        ctx.sw_hdr_skip_length = fit(
            (hdr.sw_skip_end - hdr.sw_skip_start) as i64,
            "sw_hdr_skip_length",
        )?;
        ctx.num_bits_short_term_ref_pics_in_slice =
            u32::from(ctx.sw_hdr_skip_length.saturating_sub(9));
        let stream_len: usize = nals.iter().map(|n| n.len).sum();
        ctx.stream_len = fit(stream_len as i64, "stream_len")?;
        if let Some(t) = &pps.tiles {
            set_bits(
                &mut ctx.pps_bits1,
                fields::NUM_TILE_COLUMNS,
                i64::from(t.num_tile_columns_minus1) + 1,
                "num_tile_columns",
            )?;
            set_bits(
                &mut ctx.pps_bits1,
                fields::NUM_TILE_ROWS,
                i64::from(t.num_tile_rows_minus1) + 1,
                "num_tile_rows",
            )?;
        }
        set_bits(
            &mut ctx.dxva_bits,
            fields::NUM_SHORT_TERM_REF_PIC_SETS,
            sps.num_short_term_ref_pic_sets.into(),
            "num_short_term_ref_pic_sets",
        )?;

        let luma_h_rounded = round_up(height, 64);
        let offset_256 = |prev: u32, bytes_per_row: u64, name: &str| -> Result<u32> {
            fit(
                (((u64::from(prev) << 8) + bytes_per_row * luma_h_rounded) >> 8) as i64,
                name,
            )
        };
        ctx.hevc_sao_buffer_offset = offset_256(0, 608, "HevcSaoBufferOffset")?;
        ctx.hevc_bsd_ctrl_offset =
            offset_256(ctx.hevc_sao_buffer_offset, 4864, "HevcBsdCtrlOffset")?;
        ctx.v1_hevc_flt_above_offset =
            offset_256(ctx.hevc_bsd_ctrl_offset, 152, "HevcFltAboveOffset")?;
        ctx.v1_hevc_sao_above_offset =
            offset_256(ctx.v1_hevc_flt_above_offset, 2000, "HevcSaoAboveOffset")?;
        ctx.v3_hevc_slice_edge_offset = ctx.v1_hevc_sao_above_offset;

        let poc = i64::from(hdr.slice_pic_order_cnt_lsb);
        let mut before = Vec::new();
        let mut after = Vec::new();
        for &(idx, ref_poc) in &self.history {
            let ref_poc = i64::from(ref_poc);
            ctx.ref_diff_pic_order_cnts[usize::from(idx)] =
                fit(poc - ref_poc, "RefDiffPicOrderCnts")?;
            if poc < ref_poc {
                after.push((ref_poc - poc, idx));
            } else {
                before.push((poc - ref_poc, idx));
            }
        }
        before.sort_unstable();
        after.sort_unstable();
        let list = |first: &[(i64, u8)], second: &[(i64, u8)]| -> Result<[u8; 16]> {
            let mut out = [0u8; 16];
            let all: Vec<u8> = first.iter().chain(second).map(|&(_, idx)| idx).collect();
            if all.len() > 16 {
                return contract(format!("{} reference pictures exceed 16", all.len()));
            }
            out[..all.len()].copy_from_slice(&all);
            Ok(out)
        };
        ctx.initreflistidxl0 = list(&before, &after)?;
        if hdr.slice_type == SLICE_B {
            ctx.initreflistidxl1 = list(&after, &before)?;
        }

        let mut desc = Vec::with_capacity(DESCRIPTOR_STRIDE);
        desc.extend_from_slice(ctx.as_bytes());
        desc.extend_from_slice(&BLACKWELL_EXTENSION);
        desc.resize(TILE_SIZES_OFFSET, 0);

        let ctb = 1u64 << sps.log2_max_luma_coding_block_size;
        let width_ctbs = width.div_ceil(ctb) as i64;
        let height_ctbs = height.div_ceil(ctb) as i64;
        let mut push_tile = |c: i64, r: i64| -> Result<()> {
            desc.extend_from_slice(&fit::<u16>(c, "tile column width")?.to_le_bytes());
            desc.extend_from_slice(&fit::<u16>(r, "tile row height")?.to_le_bytes());
            Ok(())
        };
        match &pps.tiles {
            Some(t) if t.uniform_spacing_flag => {
                if ctx.num_tile_columns() != 1 || ctx.num_tile_rows() != 1 {
                    return contract(
                        "uniform tile spacing with more than one tile is not supported".into(),
                    );
                }
                push_tile(width_ctbs, height_ctbs)?;
            }
            _ => {
                let (mut columns, mut rows): (Vec<i64>, Vec<i64>) = match &pps.tiles {
                    Some(t) => (
                        t.column_width_minus1
                            .iter()
                            .map(|&v| i64::from(v) + 1)
                            .collect(),
                        t.row_height_minus1
                            .iter()
                            .map(|&v| i64::from(v) + 1)
                            .collect(),
                    ),
                    None => (Vec::new(), Vec::new()),
                };
                columns.push(width_ctbs - columns.iter().sum::<i64>());
                rows.push(height_ctbs - rows.iter().sum::<i64>());
                for &c in &columns {
                    for &r in &rows {
                        push_tile(c, r)?;
                    }
                }
            }
        }
        if desc.len() > DESCRIPTOR_STRIDE {
            return contract(format!(
                "tile table needs {} bytes, descriptor holds {}",
                desc.len() - TILE_SIZES_OFFSET,
                DESCRIPTOR_STRIDE - TILE_SIZES_OFFSET
            ));
        }
        desc.resize(DESCRIPTOR_STRIDE, 0);
        self.out.descriptors.extend_from_slice(&desc);

        let is_hist = is_reference(nal_type);
        self.out.frames.push(HevcFrame {
            offset: nals[0].offset,
            size: stream_len,
            pos: u32::from(curr_pic_idx),
            is_hist,
        });
        self.out.hist.push(self.history.len());
        if is_hist {
            self.history
                .push((curr_pic_idx, hdr.slice_pic_order_cnt_lsb));
        }
        if self.history.len() >= sps.sps_max_dec_pic_buffering as usize {
            self.history.remove(0);
        }
        Ok(())
    }
}

/// Parses an Annex B HEVC elementary stream (starting with a 4-byte start
/// code) into the NVDEC frame table and per-frame descriptors.
pub fn parse_hevc(bitstream: &[u8]) -> Result<HevcParsed> {
    let mut p = Parser {
        ctx: NvdecHevcPicS::initial(),
        history: Vec::new(),
        pending: Vec::new(),
        out: HevcParsed {
            frames: Vec::new(),
            hist: Vec::new(),
            descriptors: Vec::new(),
            luma_w: 0,
            luma_h: 0,
            width: 0,
            height: 0,
        },
    };
    let mut sps: Option<Sps> = None;
    let mut pps: Option<Pps> = None;
    let mut start = 1usize;
    while start < bitstream.len() {
        if bitstream.get(start..start + 3) != Some(&[0, 0, 1][..]) {
            return contract(format!("NAL unit start code not found at offset {start}"));
        }
        let end = find_start_code(bitstream, start + 3).unwrap_or(bitstream.len());
        let len = end - start;
        let Some(&header) = bitstream.get(start + 3) else {
            return contract(format!("NAL unit at offset {start} has no header"));
        };
        let nal_type = header >> 1 & 0x3f;
        let payload = bitstream.get(start + 5..end).unwrap_or(&[]);
        match nal_type {
            NAL_SPS => {
                let s = parse_sps(&mut BitReader::new(&nal_to_rbsp(payload)))?;
                p.ctx.fill_sps(&s)?;
                sps = Some(s);
            }
            NAL_PPS => {
                let q = parse_pps(&mut BitReader::new(&nal_to_rbsp(payload)))?;
                p.ctx.fill_pps(&q)?;
                pps = Some(q);
            }
            NAL_TRAIL_N | NAL_TRAIL_R | NAL_IDR_W_RADL | NAL_IDR_N_LP => {
                let (Some(s), Some(q)) = (&sps, &pps) else {
                    return contract(format!("slice at offset {start} before SPS/PPS"));
                };
                let hdr = parse_slice_header(&mut BitReader::new(payload), nal_type, s, q)?;
                if hdr.first_slice_segment_in_pic_flag {
                    p.flush(s, q)?;
                }
                p.pending.push(SliceNal {
                    hdr,
                    nal_type,
                    offset: start,
                    len,
                });
            }
            0..=31 => {
                return contract(format!(
                    "VCL NAL unit type {nal_type} at offset {start} is not supported \
                     (only TRAIL_N, TRAIL_R, IDR_W_RADL, IDR_N_LP)"
                ));
            }
            _ => {}
        }
        start = end;
    }
    let (Some(s), Some(q)) = (&sps, &pps) else {
        return contract("stream has no SPS/PPS".into());
    };
    p.flush(s, q)?;
    if p.out.frames.is_empty() {
        return contract("stream has no pictures".into());
    }
    let [left, right, top, bottom] = s.conf_win_offsets.map(i64::from);
    let crop = |coded: u32, a: i64, b: i64, name: &str| -> Result<usize> {
        let v = i64::from(coded) - 2 * (a + b);
        if v <= 0 {
            return contract(format!("conformance window leaves {name} = {v}"));
        }
        Ok(v as usize)
    };
    p.out.width = crop(s.pic_width_in_luma_samples, left, right, "width")?;
    p.out.height = crop(s.pic_height_in_luma_samples, top, bottom, "height")?;
    p.out.luma_w = s.pic_width_in_luma_samples as usize;
    p.out.luma_h = s.pic_height_in_luma_samples as usize;
    Ok(p.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn exp_golomb() {
        // 1 | 010 | 011 | 00100 | 00101 | 00110 | 0001000 -> ue 0,1,2,3,4,5,7
        let bits = "10100110010000101001100001000";
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.bytes().enumerate() {
            bytes[i / 8] |= (b - b'0') << (7 - i % 8);
        }
        let mut r = BitReader::new(&bytes);
        let ue: Vec<u32> = (0..7).map(|_| r.ue().unwrap()).collect();
        assert_eq!(ue, [0, 1, 2, 3, 4, 5, 7]);
        assert_eq!(r.position(), bits.len());
        // se(v), Table 9-3: k = 0,1,2,3,4,5,7 -> 0,1,-1,2,-2,3,4
        let mut r = BitReader::new(&bytes);
        let se: Vec<i64> = (0..7).map(|_| r.se().unwrap()).collect();
        assert_eq!(se, [0, 1, -1, 2, -2, 3, 4]);
        let mut r = BitReader::new(&[0xa5]);
        assert_eq!(r.u(3).unwrap(), 0b101);
        assert_eq!(r.u(0).unwrap(), 0);
        assert_eq!(r.u(5).unwrap(), 0b00101);
        assert!(r.u(1).is_err(), "reading past the end must fail");
        // 33 leading zeros is not a valid 32-bit ue(v).
        assert!(BitReader::new(&[0, 0, 0, 0, 0]).ue().is_err());
    }

    #[test]
    fn emulation_prevention() {
        assert_eq!(nal_to_rbsp(&[0, 0, 3, 1]), [0, 0, 1]);
        assert_eq!(nal_to_rbsp(&[0, 0, 3, 0, 0, 3]), [0, 0, 0, 0]);
        assert_eq!(nal_to_rbsp(&[5, 0, 0, 3]), [5, 0, 0]);
        assert_eq!(nal_to_rbsp(&[0, 0, 0, 3, 7]), [0, 0, 0, 7]);
        assert_eq!(nal_to_rbsp(&[0, 3, 0, 0]), [0, 3, 0, 0]);
        assert_eq!(nal_to_rbsp(&[]), [] as [u8; 0]);
    }

    #[test]
    fn rejects_bad_streams() {
        assert!(parse_hevc(&[0, 0, 0, 2, 0x40]).is_err());
        assert!(parse_hevc(&[0, 0, 0, 1, 0x40, 1]).is_err(), "no SPS/PPS");
        // CRA (21) slice is refused, not skipped.
        let e = parse_hevc(&[0, 0, 0, 1, 21 << 1, 1, 0x80]).err().unwrap();
        assert!(e.to_string().contains("VCL NAL unit type 21"), "{e}");
    }

    #[test]
    fn bitfield_overflow_is_an_error() {
        let mut unit = 0u32;
        assert!(set_bits(&mut unit, fields::INIT_QP, 128, "init_qp").is_err());
        assert!(set_bits(&mut unit, fields::INIT_QP, -1, "init_qp").is_err());
        set_bits(&mut unit, fields::INIT_QP, 127, "init_qp").unwrap();
        assert_eq!(unit, 127 << 15);
    }

    /// Output of build/hevc-wt-scratch/layout.c: gcc on NVIDIA's MIT
    /// nvdec_drv.h, "<member> <bit offset> <bit width>" (sizeof 276).
    const C_LAYOUT: &str = "\
wrapped_session_key 0 128
wrapped_content_key 128 128
initialization_vector 256 128
stream_len 384 32
enable_encryption 416 32
key_increment 448 6
encryption_mode 454 4
key_slot_index 458 4
ssm_en 462 1
enable_histogram 463 1
enable_substream_decoding 464 1
reserved0 465 15
gptimer_timeout_value 480 32
tileformat 512 2
gob_height 514 3
reserverd_surface_format 517 3
sw_start_code_e 520 8
disp_output_mode 528 8
reserved1 536 8
framestride 544 64
colMvBuffersize 608 32
HevcSaoBufferOffset 640 32
HevcBsdCtrlOffset 672 32
pic_width_in_luma_samples 704 16
pic_height_in_luma_samples 720 16
chroma_format_idc 736 4
bit_depth_luma 740 4
bit_depth_chroma 744 4
log2_min_luma_coding_block_size 748 4
log2_max_luma_coding_block_size 752 4
log2_min_transform_block_size 756 4
log2_max_transform_block_size 760 4
reserved2 764 4
max_transform_hierarchy_depth_inter 768 3
max_transform_hierarchy_depth_intra 771 3
scalingListEnable 774 1
amp_enable_flag 775 1
sample_adaptive_offset_enabled_flag 776 1
pcm_enabled_flag 777 1
pcm_sample_bit_depth_luma 778 4
pcm_sample_bit_depth_chroma 782 4
log2_min_pcm_luma_coding_block_size 786 4
log2_max_pcm_luma_coding_block_size 790 4
pcm_loop_filter_disabled_flag 794 1
sps_temporal_mvp_enabled_flag 795 1
strong_intra_smoothing_enabled_flag 796 1
reserved3 797 3
dependent_slice_segments_enabled_flag 800 1
output_flag_present_flag 801 1
num_extra_slice_header_bits 802 3
sign_data_hiding_enabled_flag 805 1
cabac_init_present_flag 806 1
num_ref_idx_l0_default_active 807 4
num_ref_idx_l1_default_active 811 4
init_qp 815 7
constrained_intra_pred_flag 822 1
transform_skip_enabled_flag 823 1
cu_qp_delta_enabled_flag 824 1
diff_cu_qp_delta_depth 825 2
reserved4 827 5
pps_cb_qp_offset 832 8
pps_cr_qp_offset 840 8
pps_beta_offset 848 8
pps_tc_offset 856 8
pps_slice_chroma_qp_offsets_present_flag 864 1
weighted_pred_flag 865 1
weighted_bipred_flag 866 1
transquant_bypass_enabled_flag 867 1
tiles_enabled_flag 868 1
entropy_coding_sync_enabled_flag 869 1
num_tile_columns 870 5
num_tile_rows 875 5
loop_filter_across_tiles_enabled_flag 880 1
loop_filter_across_slices_enabled_flag 881 1
deblocking_filter_control_present_flag 882 1
deblocking_filter_override_enabled_flag 883 1
pps_deblocking_filter_disabled_flag 884 1
lists_modification_present_flag 885 1
log2_parallel_merge_level 886 3
slice_segment_header_extension_present_flag 889 1
reserved5 890 6
num_ref_frames 896 8
reserved6 904 8
longtermflag 912 16
initreflistidxl0 928 128
initreflistidxl1 1056 128
RefDiffPicOrderCnts 1184 256
IDR_picture_flag 1440 8
RAP_picture_flag 1448 8
curr_pic_idx 1456 8
pattern_id 1464 8
sw_hdr_skip_length 1472 16
reserved7 1488 16
ecdma_cfg 1504 192
separate_colour_plane_flag 1696 1
log2_max_pic_order_cnt_lsb_minus4 1697 4
num_short_term_ref_pic_sets 1701 7
num_long_term_ref_pics_sps 1708 6
bBitParsingDisable 1714 1
num_delta_pocs_of_rps_idx 1715 8
long_term_ref_pics_present_flag 1723 1
reserved_dxva 1724 4
num_bits_short_term_ref_pics_in_slice 1728 32
v1 1760 256
v1.hevc_main10_444_ext 1760 224
v1.hevc_main10_444_ext.cb_qp_adjustment 1824 48
v1.hevc_main10_444_ext.cr_qp_adjustment 1872 48
v1.hevc_main10_444_ext.HevcFltAboveOffset 1920 32
v1.hevc_main10_444_ext.HevcSaoAboveOffset 1952 32
v1.sw_skip_start_length 1984 14
v1.external_ref_mem_dis 1998 1
v1.error_recovery_start_pos 1999 2
v1.error_external_mv_en 2001 1
v1.reserved0 2002 14
v2 2016 32
v3 2048 64
v3.slice_decoding_enable 2048 1
v3.HevcSliceEdgeOffset 2080 32
ssm 2112 96
";

    #[test]
    fn layout_matches_c_header() {
        use fields::*;
        type P = NvdecHevcPicS;
        macro_rules! whole {
            ($f:ident) => {
                (offset_of!(P, $f) * 8, size_of_val(&P::default().$f) * 8)
            };
        }
        let bits = |unit: usize, f: BitField| (unit * 8 + f.shift as usize, f.width as usize);
        let (enc, r#gen) = (offset_of!(P, encryption_bits), offset_of!(P, general_bits));
        let (s0, s1) = (offset_of!(P, sps_bits0), offset_of!(P, sps_bits1));
        let (p0, p1) = (offset_of!(P, pps_bits0), offset_of!(P, pps_bits1));
        let (dx, v1b, v3b) = (
            offset_of!(P, dxva_bits),
            offset_of!(P, v1_bits),
            offset_of!(P, v3_bits),
        );
        let v1 = offset_of!(P, v1_main10_444_sps_bits);
        let rust: Vec<(&str, (usize, usize))> = vec![
            ("wrapped_session_key", whole!(wrapped_session_key)),
            ("wrapped_content_key", whole!(wrapped_content_key)),
            ("initialization_vector", whole!(initialization_vector)),
            ("stream_len", whole!(stream_len)),
            ("enable_encryption", whole!(enable_encryption)),
            ("key_increment", bits(enc, KEY_INCREMENT)),
            ("encryption_mode", bits(enc, ENCRYPTION_MODE)),
            ("key_slot_index", bits(enc, KEY_SLOT_INDEX)),
            ("ssm_en", bits(enc, SSM_EN)),
            ("enable_histogram", bits(enc, ENABLE_HISTOGRAM)),
            (
                "enable_substream_decoding",
                bits(enc, ENABLE_SUBSTREAM_DECODING),
            ),
            ("reserved0", bits(enc, RESERVED0)),
            ("gptimer_timeout_value", whole!(gptimer_timeout_value)),
            ("tileformat", bits(r#gen, TILEFORMAT)),
            ("gob_height", bits(r#gen, GOB_HEIGHT)),
            (
                "reserverd_surface_format",
                bits(r#gen, RESERVERD_SURFACE_FORMAT),
            ),
            ("sw_start_code_e", whole!(sw_start_code_e)),
            ("disp_output_mode", whole!(disp_output_mode)),
            ("reserved1", whole!(reserved1)),
            ("framestride", whole!(framestride)),
            ("colMvBuffersize", whole!(col_mv_buffersize)),
            ("HevcSaoBufferOffset", whole!(hevc_sao_buffer_offset)),
            ("HevcBsdCtrlOffset", whole!(hevc_bsd_ctrl_offset)),
            (
                "pic_width_in_luma_samples",
                whole!(pic_width_in_luma_samples),
            ),
            (
                "pic_height_in_luma_samples",
                whole!(pic_height_in_luma_samples),
            ),
            ("chroma_format_idc", bits(s0, CHROMA_FORMAT_IDC)),
            ("bit_depth_luma", bits(s0, BIT_DEPTH_LUMA)),
            ("bit_depth_chroma", bits(s0, BIT_DEPTH_CHROMA)),
            (
                "log2_min_luma_coding_block_size",
                bits(s0, LOG2_MIN_LUMA_CODING_BLOCK_SIZE),
            ),
            (
                "log2_max_luma_coding_block_size",
                bits(s0, LOG2_MAX_LUMA_CODING_BLOCK_SIZE),
            ),
            (
                "log2_min_transform_block_size",
                bits(s0, LOG2_MIN_TRANSFORM_BLOCK_SIZE),
            ),
            (
                "log2_max_transform_block_size",
                bits(s0, LOG2_MAX_TRANSFORM_BLOCK_SIZE),
            ),
            ("reserved2", bits(s0, RESERVED2)),
            (
                "max_transform_hierarchy_depth_inter",
                bits(s1, MAX_TRANSFORM_HIERARCHY_DEPTH_INTER),
            ),
            (
                "max_transform_hierarchy_depth_intra",
                bits(s1, MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA),
            ),
            ("scalingListEnable", bits(s1, SCALING_LIST_ENABLE)),
            ("amp_enable_flag", bits(s1, AMP_ENABLE_FLAG)),
            (
                "sample_adaptive_offset_enabled_flag",
                bits(s1, SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG),
            ),
            ("pcm_enabled_flag", bits(s1, PCM_ENABLED_FLAG)),
            (
                "pcm_sample_bit_depth_luma",
                bits(s1, PCM_SAMPLE_BIT_DEPTH_LUMA),
            ),
            (
                "pcm_sample_bit_depth_chroma",
                bits(s1, PCM_SAMPLE_BIT_DEPTH_CHROMA),
            ),
            (
                "log2_min_pcm_luma_coding_block_size",
                bits(s1, LOG2_MIN_PCM_LUMA_CODING_BLOCK_SIZE),
            ),
            (
                "log2_max_pcm_luma_coding_block_size",
                bits(s1, LOG2_MAX_PCM_LUMA_CODING_BLOCK_SIZE),
            ),
            (
                "pcm_loop_filter_disabled_flag",
                bits(s1, PCM_LOOP_FILTER_DISABLED_FLAG),
            ),
            (
                "sps_temporal_mvp_enabled_flag",
                bits(s1, SPS_TEMPORAL_MVP_ENABLED_FLAG),
            ),
            (
                "strong_intra_smoothing_enabled_flag",
                bits(s1, STRONG_INTRA_SMOOTHING_ENABLED_FLAG),
            ),
            ("reserved3", bits(s1, RESERVED3)),
            (
                "dependent_slice_segments_enabled_flag",
                bits(p0, DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG),
            ),
            (
                "output_flag_present_flag",
                bits(p0, OUTPUT_FLAG_PRESENT_FLAG),
            ),
            (
                "num_extra_slice_header_bits",
                bits(p0, NUM_EXTRA_SLICE_HEADER_BITS),
            ),
            (
                "sign_data_hiding_enabled_flag",
                bits(p0, SIGN_DATA_HIDING_ENABLED_FLAG),
            ),
            ("cabac_init_present_flag", bits(p0, CABAC_INIT_PRESENT_FLAG)),
            (
                "num_ref_idx_l0_default_active",
                bits(p0, NUM_REF_IDX_L0_DEFAULT_ACTIVE),
            ),
            (
                "num_ref_idx_l1_default_active",
                bits(p0, NUM_REF_IDX_L1_DEFAULT_ACTIVE),
            ),
            ("init_qp", bits(p0, INIT_QP)),
            (
                "constrained_intra_pred_flag",
                bits(p0, CONSTRAINED_INTRA_PRED_FLAG),
            ),
            (
                "transform_skip_enabled_flag",
                bits(p0, TRANSFORM_SKIP_ENABLED_FLAG),
            ),
            (
                "cu_qp_delta_enabled_flag",
                bits(p0, CU_QP_DELTA_ENABLED_FLAG),
            ),
            ("diff_cu_qp_delta_depth", bits(p0, DIFF_CU_QP_DELTA_DEPTH)),
            ("reserved4", bits(p0, RESERVED4)),
            ("pps_cb_qp_offset", whole!(pps_cb_qp_offset)),
            ("pps_cr_qp_offset", whole!(pps_cr_qp_offset)),
            ("pps_beta_offset", whole!(pps_beta_offset)),
            ("pps_tc_offset", whole!(pps_tc_offset)),
            (
                "pps_slice_chroma_qp_offsets_present_flag",
                bits(p1, PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG),
            ),
            ("weighted_pred_flag", bits(p1, WEIGHTED_PRED_FLAG)),
            ("weighted_bipred_flag", bits(p1, WEIGHTED_BIPRED_FLAG)),
            (
                "transquant_bypass_enabled_flag",
                bits(p1, TRANSQUANT_BYPASS_ENABLED_FLAG),
            ),
            ("tiles_enabled_flag", bits(p1, TILES_ENABLED_FLAG)),
            (
                "entropy_coding_sync_enabled_flag",
                bits(p1, ENTROPY_CODING_SYNC_ENABLED_FLAG),
            ),
            ("num_tile_columns", bits(p1, NUM_TILE_COLUMNS)),
            ("num_tile_rows", bits(p1, NUM_TILE_ROWS)),
            (
                "loop_filter_across_tiles_enabled_flag",
                bits(p1, LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG),
            ),
            (
                "loop_filter_across_slices_enabled_flag",
                bits(p1, LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG),
            ),
            (
                "deblocking_filter_control_present_flag",
                bits(p1, DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG),
            ),
            (
                "deblocking_filter_override_enabled_flag",
                bits(p1, DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG),
            ),
            (
                "pps_deblocking_filter_disabled_flag",
                bits(p1, PPS_DEBLOCKING_FILTER_DISABLED_FLAG),
            ),
            (
                "lists_modification_present_flag",
                bits(p1, LISTS_MODIFICATION_PRESENT_FLAG),
            ),
            (
                "log2_parallel_merge_level",
                bits(p1, LOG2_PARALLEL_MERGE_LEVEL),
            ),
            (
                "slice_segment_header_extension_present_flag",
                bits(p1, SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG),
            ),
            ("reserved5", bits(p1, RESERVED5)),
            ("num_ref_frames", whole!(num_ref_frames)),
            ("reserved6", whole!(reserved6)),
            ("longtermflag", whole!(longtermflag)),
            ("initreflistidxl0", whole!(initreflistidxl0)),
            ("initreflistidxl1", whole!(initreflistidxl1)),
            ("RefDiffPicOrderCnts", whole!(ref_diff_pic_order_cnts)),
            ("IDR_picture_flag", whole!(idr_picture_flag)),
            ("RAP_picture_flag", whole!(rap_picture_flag)),
            ("curr_pic_idx", whole!(curr_pic_idx)),
            ("pattern_id", whole!(pattern_id)),
            ("sw_hdr_skip_length", whole!(sw_hdr_skip_length)),
            ("reserved7", whole!(reserved7)),
            ("ecdma_cfg", whole!(ecdma_cfg)),
            (
                "separate_colour_plane_flag",
                bits(dx, SEPARATE_COLOUR_PLANE_FLAG),
            ),
            (
                "log2_max_pic_order_cnt_lsb_minus4",
                bits(dx, LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4),
            ),
            (
                "num_short_term_ref_pic_sets",
                bits(dx, NUM_SHORT_TERM_REF_PIC_SETS),
            ),
            (
                "num_long_term_ref_pics_sps",
                bits(dx, NUM_LONG_TERM_REF_PICS_SPS),
            ),
            ("bBitParsingDisable", bits(dx, BBIT_PARSING_DISABLE)),
            (
                "num_delta_pocs_of_rps_idx",
                bits(dx, NUM_DELTA_POCS_OF_RPS_IDX),
            ),
            (
                "long_term_ref_pics_present_flag",
                bits(dx, LONG_TERM_REF_PICS_PRESENT_FLAG),
            ),
            ("reserved_dxva", bits(dx, RESERVED_DXVA)),
            (
                "num_bits_short_term_ref_pics_in_slice",
                whole!(num_bits_short_term_ref_pics_in_slice),
            ),
            ("v1", (v1 * 8, (offset_of!(P, v2_bits) - v1) * 8)),
            ("v1.hevc_main10_444_ext", (v1 * 8, (v1b - v1) * 8)),
            (
                "v1.hevc_main10_444_ext.cb_qp_adjustment",
                whole!(v1_cb_qp_adjustment),
            ),
            (
                "v1.hevc_main10_444_ext.cr_qp_adjustment",
                whole!(v1_cr_qp_adjustment),
            ),
            (
                "v1.hevc_main10_444_ext.HevcFltAboveOffset",
                whole!(v1_hevc_flt_above_offset),
            ),
            (
                "v1.hevc_main10_444_ext.HevcSaoAboveOffset",
                whole!(v1_hevc_sao_above_offset),
            ),
            (
                "v1.sw_skip_start_length",
                bits(v1b, V1_SW_SKIP_START_LENGTH),
            ),
            (
                "v1.external_ref_mem_dis",
                bits(v1b, V1_EXTERNAL_REF_MEM_DIS),
            ),
            (
                "v1.error_recovery_start_pos",
                bits(v1b, V1_ERROR_RECOVERY_START_POS),
            ),
            (
                "v1.error_external_mv_en",
                bits(v1b, V1_ERROR_EXTERNAL_MV_EN),
            ),
            ("v1.reserved0", bits(v1b, V1_RESERVED0)),
            ("v2", whole!(v2_bits)),
            ("v3", (v3b * 8, (offset_of!(P, ssm) - v3b) * 8)),
            (
                "v3.slice_decoding_enable",
                bits(v3b, V3_SLICE_DECODING_ENABLE),
            ),
            ("v3.HevcSliceEdgeOffset", whole!(v3_hevc_slice_edge_offset)),
            ("ssm", whole!(ssm)),
        ];
        let c: Vec<(&str, (usize, usize))> = C_LAYOUT
            .lines()
            .map(|l| {
                let v: Vec<&str> = l.split_whitespace().collect();
                (v[0], (v[1].parse().unwrap(), v[2].parse().unwrap()))
            })
            .collect();
        assert_eq!(rust.len(), c.len());
        for (r, c) in rust.iter().zip(&c) {
            assert_eq!(r, c, "member layout differs from gcc on nvdec_drv.h");
        }
        assert_eq!(size_of::<P>(), 276);
    }

    /// Byte-identity against fixtures produced by the Python generator
    /// (tinygrad extra/hevc/hevc.py): `NVDEC_FIXTURE_DIRS=dir1:dir2`, each with
    /// stream.hevc, desc.bin and frames.txt. Skipped when unset.
    #[test]
    fn matches_python_fixtures() {
        let Ok(dirs) = std::env::var("NVDEC_FIXTURE_DIRS") else {
            eprintln!("NVDEC_FIXTURE_DIRS unset; fixture comparison skipped");
            return;
        };
        for dir in dirs.split(':').filter(|d| !d.is_empty()) {
            let dir = std::path::Path::new(dir);
            let stream = std::fs::read(dir.join("stream.hevc")).unwrap();
            let want_desc = std::fs::read(dir.join("desc.bin")).unwrap();
            let want_frames = std::fs::read_to_string(dir.join("frames.txt")).unwrap();
            let parsed = parse_hevc(&stream).unwrap();
            let mut frames = format!(
                "{} {} {} {}\n",
                parsed.luma_w, parsed.luma_h, parsed.width, parsed.height
            );
            for (f, hist) in parsed.frames.iter().zip(&parsed.hist) {
                frames += &format!(
                    "{} {} {} {} {}\n",
                    f.offset,
                    f.size,
                    f.pos,
                    hist,
                    u8::from(f.is_hist)
                );
            }
            assert_eq!(frames, want_frames, "{}: frames.txt", dir.display());
            assert_eq!(
                parsed.descriptors.len(),
                want_desc.len(),
                "{}",
                dir.display()
            );
            for (i, (got, want)) in parsed
                .descriptors
                .chunks(DESCRIPTOR_STRIDE)
                .zip(want_desc.chunks(DESCRIPTOR_STRIDE))
                .enumerate()
            {
                let first = got.iter().zip(want).position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "{}: frame {i} descriptor differs at byte {:#x}",
                    dir.display(),
                    first.unwrap_or(0)
                );
            }
            eprintln!(
                "{}: {} frames identical",
                dir.display(),
                parsed.frames.len()
            );
        }
    }
}
