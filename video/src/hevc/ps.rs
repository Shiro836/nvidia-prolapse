// SPDX-License-Identifier: MIT
//! HEVC parameter sets (H.265 7.3.2.2 SPS, 7.3.2.3 PPS, 7.3.4 scaling_list_data, 7.3.7
//! st_ref_pic_set) with the derived values the descriptor needs.
use crate::VResult;
use crate::bits::BitReader;
use crate::fit;
use crate::hevc::HevcError;
use nv_rm_core::conv::usize_from_u32;

/// Short-term RPS after 7.4.8 derivation: deltas relative to the current POC.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StRps {
    /// DeltaPocS0 (negative, decreasing) with UsedByCurrPicS0.
    pub s0: Vec<(i32, bool)>,
    /// DeltaPocS1 (positive, increasing) with UsedByCurrPicS1.
    pub s1: Vec<(i32, bool)>,
}
impl StRps {
    pub fn num_delta_pocs(&self) -> usize {
        self.s0.len() + self.s1.len()
    }
}

/// Parses `st_ref_pic_set(st_rps_idx)` (7.3.7) and derives it per 7.4.8. `sets` are the SPS sets
/// parsed so far (`num_short_term_ref_pic_sets` of them when called from a slice header).
/// Returns the set and, for inter-RPS prediction, NumDeltaPocs[RefRpsIdx].
pub fn st_ref_pic_set(
    r: &mut BitReader,
    idx: usize,
    sets: &[StRps],
    in_slice: bool,
) -> VResult<(StRps, Option<u32>)> {
    let inter = if idx != 0 { r.flag()? } else { false };
    if inter {
        let delta_idx = if in_slice {
            r.ue_max(fit::<u32, _>(idx, "RefRpsIdx")? - 1, "delta_idx_minus1")? + 1
        } else {
            1
        };
        let ref_idx = idx - usize_from_u32(delta_idx);
        let rf = &sets[ref_idx];
        let sign = r.flag()?;
        let abs = fit::<i32, _>(
            r.ue_max(1 << 15, "abs_delta_rps_minus1")?,
            "abs_delta_rps_minus1",
        )? + 1;
        let delta_rps = if sign { -abs } else { abs };
        let n = rf.num_delta_pocs();
        let mut used = vec![false; n + 1];
        let mut use_delta = vec![true; n + 1];
        for j in 0..=n {
            used[j] = r.flag()?;
            if !used[j] {
                use_delta[j] = r.flag()?;
            }
        }
        // 7.4.8 equations (7-61) and (7-62). dPoc over RefRps: S0 entries then S1 entries.
        let d = |j: usize| -> i32 {
            if j < rf.s0.len() {
                rf.s0[j].0
            } else {
                rf.s1[j - rf.s0.len()].0
            }
        };
        let n0 = rf.s0.len();
        let n1 = rf.s1.len();
        let mut s0 = Vec::new();
        // S0
        for j in (0..n1).rev() {
            let dpoc = d(n0 + j) + delta_rps;
            if dpoc < 0 && use_delta[n0 + j] {
                s0.push((dpoc, used[n0 + j]));
            }
        }
        if delta_rps < 0 && use_delta[n] {
            s0.push((delta_rps, used[n]));
        }
        for j in 0..n0 {
            let dpoc = d(j) + delta_rps;
            if dpoc < 0 && use_delta[j] {
                s0.push((dpoc, used[j]));
            }
        }
        // S1
        let mut s1 = Vec::new();
        for j in (0..n0).rev() {
            let dpoc = d(j) + delta_rps;
            if dpoc > 0 && use_delta[j] {
                s1.push((dpoc, used[j]));
            }
        }
        if delta_rps > 0 && use_delta[n] {
            s1.push((delta_rps, used[n]));
        }
        for j in 0..n1 {
            let dpoc = d(n0 + j) + delta_rps;
            if dpoc > 0 && use_delta[n0 + j] {
                s1.push((dpoc, used[n0 + j]));
            }
        }
        if s0.len() > 16 || s1.len() > 16 || s0.len() + s1.len() > 16 {
            return Err(HevcError::Malformed("st_ref_pic_set with more than 16 pictures").into());
        }
        Ok((StRps { s0, s1 }, Some(fit(n, "NumDeltaPocs")?)))
    } else {
        let n_neg = usize_from_u32(r.ue_max(16, "num_negative_pics")?);
        let n_pos = usize_from_u32(r.ue_max(
            16 - fit::<u32, _>(n_neg, "num_negative_pics")?,
            "num_positive_pics",
        )?);
        let mut s0 = Vec::with_capacity(n_neg);
        let mut poc = 0;
        for _ in 0..n_neg {
            poc -= fit::<i32, _>(
                r.ue_max(1 << 15, "delta_poc_s0_minus1")?,
                "delta_poc_s0_minus1",
            )? + 1;
            s0.push((poc, r.flag()?));
        }
        let mut s1 = Vec::with_capacity(n_pos);
        poc = 0;
        for _ in 0..n_pos {
            poc += fit::<i32, _>(
                r.ue_max(1 << 15, "delta_poc_s1_minus1")?,
                "delta_poc_s1_minus1",
            )? + 1;
            s1.push((poc, r.flag()?));
        }
        Ok((StRps { s0, s1 }, None))
    }
}

/// Scaling factors as coded (7.4.5), **coefficient order = up-right diagonal scan**, plus DC.
/// Index [sizeId][matrixId]; sizeId 0 has 16 coefficients, the others 64 (8x8 representative).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingList {
    pub lists: [[[u8; 64]; 6]; 4],
    /// scaling_list_dc_coef_minus8 + 8 for sizeId 2 (matrixId 0..5) and 3 (matrixId 0..5).
    pub dc: [[u8; 6]; 2],
}

/// Table 7-6 (diagonal order), intra (matrixId 0..2) and inter (3..5) defaults for sizeId 1..3.
const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];
const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];

impl ScalingList {
    /// 7.4.5 defaults (also what `scaling_list_enabled_flag` with no data present means).
    pub fn default_lists() -> Self {
        let mut s = ScalingList {
            lists: [[[16; 64]; 6]; 4],
            dc: [[16; 6]; 2],
        };
        for size in 1..4 {
            for m in 0..6 {
                s.lists[size][m] = if m < 3 { DEFAULT_INTRA } else { DEFAULT_INTER };
            }
        }
        s
    }
    fn default_for(size: usize, m: usize) -> [u8; 64] {
        if size == 0 {
            [16; 64]
        } else if m < 3 {
            DEFAULT_INTRA
        } else {
            DEFAULT_INTER
        }
    }
    /// scaling_list_data() (7.3.4, version 1 semantics: 32x32 has matrixId 0 and 3 only; the
    /// chroma 32x32 lists are inferred from the 16x16 ones as in 7.4.5 for ChromaArrayType 3,
    /// unused for 4:2:0).
    pub fn parse(r: &mut BitReader) -> VResult<Self> {
        let mut s = Self::default_lists();
        for size in 0..4usize {
            let step = if size == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                let pred_mode = r.flag()?;
                let coef_num = if size == 0 { 16 } else { 64 };
                if !pred_mode {
                    let delta = usize_from_u32(r.ue_max(
                        fit(m / step, "matrixId")?,
                        "scaling_list_pred_matrix_id_delta",
                    )?);
                    if delta == 0 {
                        s.lists[size][m] = Self::default_for(size, m);
                        if size >= 2 {
                            s.dc[size - 2][m] = 16;
                        }
                    } else {
                        let rm = m - delta * step;
                        s.lists[size][m] = s.lists[size][rm];
                        if size >= 2 {
                            s.dc[size - 2][m] = s.dc[size - 2][rm];
                        }
                    }
                } else {
                    let mut next: i32 = 8;
                    if size > 1 {
                        let dc = r.se_range(-7, 247, "scaling_list_dc_coef_minus8")? + 8;
                        next = dc;
                        s.dc[size - 2][m] = fit(dc, "scaling_list_dc_coef_minus8")?;
                    }
                    for i in 0..coef_num {
                        let d = r.se_range(-128, 127, "scaling_list_delta_coef")?;
                        next = (next + d + 256).rem_euclid(256);
                        if next == 0 {
                            return Err(HevcError::Malformed("scaling list coefficient 0").into());
                        }
                        s.lists[size][m][i] = fit(next, "ScalingList")?;
                    }
                }
                m += step;
            }
        }
        // 32x32 chroma (matrixId 1,2,4,5) are not coded for ChromaArrayType != 3; keep 16x16 copies.
        for m in [1, 2, 4, 5] {
            s.lists[3][m] = s.lists[2][m];
            s.dc[1][m] = s.dc[0][m];
        }
        Ok(s)
    }
}

/// Up-right diagonal scan (6.5.3) of a `blk`×`blk` block: scan position → (x, y).
pub fn diag_scan(blk: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(blk * blk);
    let (mut x, mut y): (isize, isize) = (0, 0);
    while out.len() < blk * blk {
        while y >= 0 {
            // x, y >= 0 here (y by the loop condition, x only grows from 0).
            if let (Ok(ux), Ok(uy)) = (usize::try_from(x), usize::try_from(y))
                && ux < blk
                && uy < blk
            {
                out.push((ux, uy));
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    out
}

#[derive(Debug, Clone)]
pub struct Sps {
    pub id: u32,
    pub profile_idc: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub width: u32,
    pub height: u32,
    pub conf_win: [u32; 4],
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_poc_lsb: u32,
    pub max_dec_pic_buffering: u32,
    /// sps_max_num_reorder_pics of the highest sub-layer (output bumping bound).
    pub max_num_reorder_pics: u32,
    pub log2_min_cb: u32,
    pub log2_ctb: u32,
    pub log2_min_tb: u32,
    pub log2_max_tb: u32,
    pub max_th_depth_inter: u32,
    pub max_th_depth_intra: u32,
    pub scaling_list_enabled: bool,
    /// SPS lists (defaults when enabled without data).
    pub scaling_list: Option<ScalingList>,
    pub amp: bool,
    pub sao: bool,
    pub pcm: bool,
    pub pcm_bit_depth_luma: u32,
    pub pcm_bit_depth_chroma: u32,
    pub log2_min_pcm_cb: u32,
    pub log2_max_pcm_cb: u32,
    pub pcm_loop_filter_disabled: bool,
    pub st_rps: Vec<StRps>,
    pub long_term_ref_pics_present: bool,
    pub num_long_term_ref_pics_sps: u32,
    pub temporal_mvp: bool,
    pub strong_intra_smoothing: bool,
}
impl Sps {
    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb
    }
    pub fn width_ctbs(&self) -> u32 {
        self.width.div_ceil(self.ctb_size())
    }
    pub fn height_ctbs(&self) -> u32 {
        self.height.div_ceil(self.ctb_size())
    }
}

fn profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> VResult<u32> {
    r.skip(2 + 1)?; // general_profile_space, general_tier_flag
    let profile_idc = r.u(5)?;
    let compat = r.u(32)?;
    r.skip(4 + 43 + 1)?; // progressive/interlaced/non_packed/frame_only, 43 reserved/constraint bits, inbld
    r.skip(8)?; // general_level_idc
    let mut prof = [false; 8];
    let mut lvl = [false; 8];
    let sub_layers = usize_from_u32(max_sub_layers_minus1);
    for i in 0..sub_layers {
        prof[i] = r.flag()?;
        lvl[i] = r.flag()?;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.skip(2)?;
        }
    }
    for i in 0..sub_layers {
        if prof[i] {
            r.skip(88)?;
        }
        if lvl[i] {
            r.skip(8)?;
        }
    }
    // profile_idc 0 with a Main/Main10 compatibility flag is common in the wild.
    let p = if profile_idc == 0 {
        (1..4).find(|&k| compat & (1 << (31 - k)) != 0).unwrap_or(0)
    } else {
        profile_idc
    };
    Ok(p)
}

pub fn parse_sps(rbsp: &[u8]) -> VResult<Sps> {
    let mut r = BitReader::new(super::payload(rbsp, "SPS")?);
    let r = &mut r;
    r.skip(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = r.u(3)?;
    if max_sub_layers_minus1 > 6 {
        return Err(HevcError::Malformed("sps_max_sub_layers_minus1 > 6").into());
    }
    r.skip(1)?;
    let profile_idc = profile_tier_level(r, max_sub_layers_minus1)?;
    // Main (1), Main 10 (2), Main Still Picture (3): VUI and SPS extensions carry nothing the
    // engine descriptor needs for these profiles, so parsing stops after the coding tools.
    if !(1..=3).contains(&profile_idc) {
        return Err(HevcError::Profile(profile_idc).into());
    }
    let id = r.ue_max(15, "sps_seq_parameter_set_id")?;
    let chroma_format_idc = r.ue_max(3, "chroma_format_idc")?;
    let separate_colour_plane = if chroma_format_idc == 3 {
        r.flag()?
    } else {
        false
    };
    if chroma_format_idc != 1 {
        return Err(HevcError::ChromaFormat(chroma_format_idc).into());
    }
    let width = r.ue_max(16888, "pic_width_in_luma_samples")?;
    let height = r.ue_max(16888, "pic_height_in_luma_samples")?;
    let mut conf_win = [0; 4];
    if r.flag()? {
        for c in &mut conf_win {
            *c = r.ue_max(16888, "conf_win_offset")?;
        }
    }
    let bit_depth_luma = r.ue_max(8, "bit_depth_luma_minus8")? + 8;
    let bit_depth_chroma = r.ue_max(8, "bit_depth_chroma_minus8")? + 8;
    // One sample container per surface: luma and chroma depth must agree (8 or 10).
    if bit_depth_luma > 10 || bit_depth_chroma != bit_depth_luma {
        return Err(HevcError::BitDepth {
            luma: bit_depth_luma,
            chroma: bit_depth_chroma,
        }
        .into());
    }
    let log2_max_poc_lsb = r.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")? + 4;
    let sub_info = r.flag()?;
    let mut max_dec_pic_buffering = 0;
    let mut max_num_reorder_pics = 0;
    let first = if sub_info { 0 } else { max_sub_layers_minus1 };
    for _ in first..=max_sub_layers_minus1 {
        max_dec_pic_buffering = r.ue_max(15, "sps_max_dec_pic_buffering_minus1")? + 1;
        max_num_reorder_pics = r.ue_max(15, "sps_max_num_reorder_pics")?;
        r.ue()?; // sps_max_latency_increase_plus1
    }
    let log2_min_cb = r.ue_max(3, "log2_min_luma_coding_block_size_minus3")? + 3;
    let log2_ctb = log2_min_cb + r.ue_max(3, "log2_diff_max_min_luma_coding_block_size")?;
    if !(4..=6).contains(&log2_ctb) {
        return Err(HevcError::CtbSize(log2_ctb).into());
    }
    let log2_min_tb = r.ue_max(3, "log2_min_luma_transform_block_size_minus2")? + 2;
    let log2_max_tb = log2_min_tb + r.ue_max(3, "log2_diff_max_min_luma_transform_block_size")?;
    let max_th_depth_inter = r.ue_max(4, "max_transform_hierarchy_depth_inter")?;
    let max_th_depth_intra = r.ue_max(4, "max_transform_hierarchy_depth_intra")?;
    let scaling_list_enabled = r.flag()?;
    let scaling_list = if scaling_list_enabled {
        Some(if r.flag()? {
            ScalingList::parse(r)?
        } else {
            ScalingList::default_lists()
        })
    } else {
        None
    };
    let amp = r.flag()?;
    let sao = r.flag()?;
    let pcm = r.flag()?;
    let (
        mut pcm_bit_depth_luma,
        mut pcm_bit_depth_chroma,
        mut log2_min_pcm_cb,
        mut log2_max_pcm_cb,
        mut pcm_lf,
    ) = (0, 0, 0, 0, false);
    if pcm {
        pcm_bit_depth_luma = r.u(4)? + 1;
        pcm_bit_depth_chroma = r.u(4)? + 1;
        log2_min_pcm_cb = r.ue_max(2, "log2_min_pcm_luma_coding_block_size_minus3")? + 3;
        log2_max_pcm_cb =
            log2_min_pcm_cb + r.ue_max(2, "log2_diff_max_min_pcm_luma_coding_block_size")?;
        pcm_lf = r.flag()?;
    }
    let n_sets = usize_from_u32(r.ue_max(64, "num_short_term_ref_pic_sets")?);
    let mut st_rps = Vec::with_capacity(n_sets);
    for i in 0..n_sets {
        let (s, _) = st_ref_pic_set(r, i, &st_rps, false)?;
        st_rps.push(s);
    }
    let long_term_ref_pics_present = r.flag()?;
    let mut num_long_term_ref_pics_sps = 0;
    if long_term_ref_pics_present {
        num_long_term_ref_pics_sps = r.ue_max(32, "num_long_term_ref_pics_sps")?;
        for _ in 0..num_long_term_ref_pics_sps {
            r.skip(usize_from_u32(log2_max_poc_lsb) + 1)?;
        }
    }
    let temporal_mvp = r.flag()?;
    let strong_intra_smoothing = r.flag()?;
    // vui_parameters_present_flag and the rest: not needed (see profile gate above).
    Ok(Sps {
        id,
        profile_idc,
        chroma_format_idc,
        separate_colour_plane,
        width,
        height,
        conf_win,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_poc_lsb,
        max_dec_pic_buffering,
        max_num_reorder_pics,
        log2_min_cb,
        log2_ctb,
        log2_min_tb,
        log2_max_tb,
        max_th_depth_inter,
        max_th_depth_intra,
        scaling_list_enabled,
        scaling_list,
        amp,
        sao,
        pcm,
        pcm_bit_depth_luma,
        pcm_bit_depth_chroma,
        log2_min_pcm_cb,
        log2_max_pcm_cb,
        pcm_loop_filter_disabled: pcm_lf,
        st_rps,
        long_term_ref_pics_present,
        num_long_term_ref_pics_sps,
        temporal_mvp,
        strong_intra_smoothing,
    })
}

#[derive(Debug, Clone)]
pub struct Tiles {
    pub uniform: bool,
    /// Explicit sizes (minus1) as coded; empty when uniform.
    pub col_minus1: Vec<u32>,
    pub row_minus1: Vec<u32>,
    pub cols: u32,
    pub rows: u32,
    pub loop_filter_across_tiles: bool,
}
impl Tiles {
    /// Column widths / row heights in CTBs (6.5.1, equations 6-3 and 6-4).
    pub fn sizes(&self, w_ctbs: u32, h_ctbs: u32) -> (Vec<u32>, Vec<u32>) {
        let derive = |n: u32, total: u32, explicit: &[u32]| -> Vec<u32> {
            if self.uniform {
                (0..n)
                    .map(|i| ((i + 1) * total) / n - (i * total) / n)
                    .collect()
            } else {
                let mut v: Vec<u32> = explicit.iter().map(|&x| x + 1).collect();
                let used: u32 = v.iter().sum();
                v.push(total.saturating_sub(used));
                v
            }
        };
        (
            derive(self.cols, w_ctbs, &self.col_minus1),
            derive(self.rows, h_ctbs, &self.row_minus1),
        )
    }
}

#[derive(Debug, Clone)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub init_qp_minus26: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip: bool,
    pub cu_qp_delta: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    pub tiles: Option<Tiles>,
    pub entropy_coding_sync: bool,
    pub loop_filter_across_slices: bool,
    pub deblocking_control_present: bool,
    pub deblocking_override_enabled: bool,
    pub deblocking_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub scaling_list: Option<ScalingList>,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level: u32,
    pub slice_header_extension_present: bool,
}

pub fn parse_pps(rbsp: &[u8]) -> VResult<Pps> {
    let mut r = BitReader::new(super::payload(rbsp, "PPS")?);
    let r = &mut r;
    let id = r.ue_max(63, "pps_pic_parameter_set_id")?;
    let sps_id = r.ue_max(15, "pps_seq_parameter_set_id")?;
    let dependent_slice_segments_enabled = r.flag()?;
    let output_flag_present = r.flag()?;
    let num_extra_slice_header_bits = r.u(3)?;
    let sign_data_hiding = r.flag()?;
    let cabac_init_present = r.flag()?;
    let num_ref_idx_l0_default = r.ue_max(14, "num_ref_idx_l0_default_active_minus1")? + 1;
    let num_ref_idx_l1_default = r.ue_max(14, "num_ref_idx_l1_default_active_minus1")? + 1;
    let init_qp_minus26 = r.se_range(-(26 + 12), 25, "init_qp_minus26")?;
    let constrained_intra_pred = r.flag()?;
    let transform_skip = r.flag()?;
    let cu_qp_delta = r.flag()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta {
        r.ue_max(3, "diff_cu_qp_delta_depth")?
    } else {
        0
    };
    let cb_qp_offset = r.se_range(-12, 12, "pps_cb_qp_offset")?;
    let cr_qp_offset = r.se_range(-12, 12, "pps_cr_qp_offset")?;
    let slice_chroma_qp_offsets_present = r.flag()?;
    let weighted_pred = r.flag()?;
    let weighted_bipred = r.flag()?;
    let transquant_bypass = r.flag()?;
    let tiles_enabled = r.flag()?;
    let entropy_coding_sync = r.flag()?;
    let tiles = if tiles_enabled {
        let cols = r.ue_max(19, "num_tile_columns_minus1")? + 1;
        let rows = r.ue_max(21, "num_tile_rows_minus1")? + 1;
        let uniform = r.flag()?;
        let (mut col_minus1, mut row_minus1) = (Vec::new(), Vec::new());
        if !uniform {
            for _ in 0..cols - 1 {
                col_minus1.push(r.ue_max(1 << 12, "column_width_minus1")?);
            }
            for _ in 0..rows - 1 {
                row_minus1.push(r.ue_max(1 << 12, "row_height_minus1")?);
            }
        }
        let loop_filter_across_tiles = r.flag()?;
        Some(Tiles {
            uniform,
            col_minus1,
            row_minus1,
            cols,
            rows,
            loop_filter_across_tiles,
        })
    } else {
        None
    };
    let loop_filter_across_slices = r.flag()?;
    let deblocking_control_present = r.flag()?;
    let (mut ovr, mut dis, mut beta, mut tc) = (false, false, 0, 0);
    if deblocking_control_present {
        ovr = r.flag()?;
        dis = r.flag()?;
        if !dis {
            beta = r.se_range(-6, 6, "pps_beta_offset_div2")?;
            tc = r.se_range(-6, 6, "pps_tc_offset_div2")?;
        }
    }
    let scaling_list = if r.flag()? {
        Some(ScalingList::parse(r)?)
    } else {
        None
    };
    let lists_modification_present = r.flag()?;
    let log2_parallel_merge_level = r.ue_max(4, "log2_parallel_merge_level_minus2")? + 2;
    let slice_header_extension_present = r.flag()?;
    if r.flag()? {
        // pps_extension_present_flag: range/multilayer/3D/SCC extensions are not Main/Main 10 tools.
        let range = r.flag()?;
        let others = r.u(3)?;
        if range || others != 0 {
            return Err(
                HevcError::Unsupported("HEVC PPS range/multilayer/3D/SCC extension").into(),
            );
        }
    }
    Ok(Pps {
        id,
        sps_id,
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding,
        cabac_init_present,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        init_qp_minus26,
        constrained_intra_pred,
        transform_skip,
        cu_qp_delta,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        weighted_pred,
        weighted_bipred,
        transquant_bypass,
        tiles,
        entropy_coding_sync,
        loop_filter_across_slices,
        deblocking_control_present,
        deblocking_override_enabled: ovr,
        deblocking_disabled: dis,
        beta_offset_div2: beta,
        tc_offset_div2: tc,
        scaling_list,
        lists_modification_present,
        log2_parallel_merge_level,
        slice_header_extension_present,
    })
}
