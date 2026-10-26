// SPDX-License-Identifier: MIT
//! H.264 parameter sets: SPS (7.3.2.1.1 + VUI E.1.1) and PPS (7.3.2.2), scaling-list fall-back
//! rules A/B (Table 7-2). Scaling matrices are stored in **raster** order, which is what the
//! engine consumes (CUVID `WeightScale4x4/8x8` are raster: FFmpeg h264_ps.c stores
//! `factors[zigzag[i]]`; the stock NEW_H264 builder at ELF 0x3c460 copies them verbatim to
//! setup+0x200).
use super::super::{VResult, bits::BitReader};
use crate::fit;
use crate::h264::H264Error;
use nv_rm_core::conv::low_u8;

/// 4x4 zig-zag (frame) scan: coded index → raster index (Table 8-13).
pub const ZIGZAG_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// 8x8 zig-zag (frame) scan: coded index → raster index (Table 8-14).
pub const ZIGZAG_8X8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
/// Table 7-3 / 7-4 defaults, in zig-zag (coded) order.
const DEFAULT_4X4_INTRA: [u8; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];
const DEFAULT_4X4_INTER: [u8; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];
const DEFAULT_8X8_INTRA: [u8; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25, 25,
    25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];
const DEFAULT_8X8_INTER: [u8; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];

fn raster4(zz: &[u8; 16]) -> [u8; 16] {
    let mut r = [0; 16];
    for i in 0..16 {
        r[usize::from(ZIGZAG_4X4[i])] = zz[i];
    }
    r
}
fn raster8(zz: &[u8; 64]) -> [u8; 64] {
    let mut r = [0; 64];
    for i in 0..64 {
        r[usize::from(ZIGZAG_8X8[i])] = zz[i];
    }
    r
}

/// Scaling matrices in raster order: 6 × 4x4 (Y/Cb/Cr intra, Y/Cb/Cr inter) and 6 × 8x8
/// (Y intra, Y inter, Cb intra, Cb inter, Cr intra, Cr inter — 7.4.2.1.1 i = 6..11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingMatrices {
    pub m4: [[u8; 16]; 6],
    pub m8: [[u8; 64]; 6],
}
impl ScalingMatrices {
    pub fn flat() -> Self {
        Self {
            m4: [[16; 16]; 6],
            m8: [[16; 64]; 6],
        }
    }
}

/// Parses `scaling_list()` (7.3.2.1.1.1). Returns the list in raster order, or `None` when
/// useDefaultScalingMatrixFlag is inferred.
fn scaling_list<const N: usize>(r: &mut BitReader, zigzag: &[u8]) -> VResult<Option<[u8; N]>> {
    let mut out = [0u8; N];
    let mut last = 8i32;
    let mut next = 8i32;
    for j in 0..N {
        if next != 0 {
            let delta = r.se_range(-128, 127, "delta_scale")?;
            next = (last + delta + 256) % 256;
            if j == 0 && next == 0 {
                return Ok(None);
            }
        }
        let v = if next == 0 { last } else { next };
        out[usize::from(zigzag[j])] = fit(v, "ScalingList")?;
        last = v;
    }
    Ok(Some(out))
}

/// Parses `n` scaling-list slots with fall-back rule A (SPS, `fallback = None`) or B (PPS,
/// `fallback = Some(sps lists)`).
fn scaling_matrices(
    r: &mut BitReader,
    n: usize,
    fallback: Option<&ScalingMatrices>,
) -> VResult<ScalingMatrices> {
    let mut m = ScalingMatrices::flat();
    for i in 0..12 {
        let present = if i < n { r.flag()? } else { false };
        if i < 6 {
            let parsed = if present {
                Some(scaling_list::<16>(r, &ZIGZAG_4X4)?)
            } else {
                None
            };
            m.m4[i] = match parsed {
                Some(Some(l)) => l,
                Some(None) => raster4(if i < 3 {
                    &DEFAULT_4X4_INTRA
                } else {
                    &DEFAULT_4X4_INTER
                }),
                None => match (i, fallback) {
                    (0, None) => raster4(&DEFAULT_4X4_INTRA),
                    (3, None) => raster4(&DEFAULT_4X4_INTER),
                    (0 | 3, Some(f)) => f.m4[i],
                    _ => m.m4[i - 1],
                },
            };
        } else {
            let k = i - 6;
            let parsed = if present {
                Some(scaling_list::<64>(r, &ZIGZAG_8X8)?)
            } else {
                None
            };
            m.m8[k] = match parsed {
                Some(Some(l)) => l,
                Some(None) => raster8(if k % 2 == 0 {
                    &DEFAULT_8X8_INTRA
                } else {
                    &DEFAULT_8X8_INTER
                }),
                None => match (k, fallback) {
                    (0, None) => raster8(&DEFAULT_8X8_INTRA),
                    (1, None) => raster8(&DEFAULT_8X8_INTER),
                    (0 | 1, Some(f)) => f.m8[k],
                    _ => m.m8[k - 2],
                },
            };
        }
    }
    Ok(m)
}

#[derive(Debug, Clone, Default)]
pub struct Vui {
    pub colour_primaries: u32,
    pub transfer_characteristics: u32,
    pub matrix_coefficients: u32,
    pub full_range: bool,
    /// bitstream_restriction: max_num_reorder_frames / max_dec_frame_buffering.
    pub max_num_reorder_frames: Option<u32>,
    pub max_dec_frame_buffering: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_flags: u8,
    pub level_idc: u8,
    pub id: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub qpprime_y_zero_transform_bypass: bool,
    pub seq_scaling_matrix_present: bool,
    pub scaling: ScalingMatrices,
    pub log2_max_frame_num_minus4: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_poc_lsb_minus4: u32,
    pub delta_pic_order_always_zero: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offset_for_ref_frame: Vec<i32>,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_allowed: bool,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only: bool,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,
    /// Crop in luma samples (left, right, top, bottom).
    pub crop: (u32, u32, u32, u32),
    pub vui: Vui,
}
impl Sps {
    pub fn max_frame_num(&self) -> u32 {
        1 << (self.log2_max_frame_num_minus4 + 4)
    }
    pub fn frame_height_in_mbs(&self) -> u32 {
        (2 - u32::from(self.frame_mbs_only)) * self.pic_height_in_map_units
    }
    pub fn width(&self) -> u32 {
        self.pic_width_in_mbs * 16
    }
    pub fn height(&self) -> u32 {
        self.frame_height_in_mbs() * 16
    }
    /// Size of the DPB in frames (VUI max_dec_frame_buffering, else max_num_ref_frames; A.3.1 h).
    pub fn dpb_frames(&self) -> u32 {
        self.vui
            .max_dec_frame_buffering
            .unwrap_or(16)
            .max(self.max_num_ref_frames)
            .max(1)
    }
}

fn hrd(r: &mut BitReader) -> VResult<()> {
    let cnt = r.ue_max(31, "cpb_cnt_minus1")? + 1;
    r.skip(8)?;
    for _ in 0..cnt {
        r.ue()?;
        r.ue()?;
        r.skip(1)?;
    }
    Ok(r.skip(20)?)
}

fn vui(r: &mut BitReader) -> VResult<Vui> {
    let mut v = Vui {
        colour_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
        ..Default::default()
    };
    if r.flag()? && r.u(8)? == 255 {
        r.skip(32)?; // Extended_SAR
    }
    if r.flag()? {
        r.skip(1)?; // overscan_appropriate
    }
    if r.flag()? {
        r.skip(3)?; // video_format
        v.full_range = r.flag()?;
        if r.flag()? {
            v.colour_primaries = r.u(8)?;
            v.transfer_characteristics = r.u(8)?;
            v.matrix_coefficients = r.u(8)?;
        }
    }
    if r.flag()? {
        r.ue()?;
        r.ue()?;
    }
    if r.flag()? {
        r.skip(65)?; // num_units_in_tick, time_scale, fixed_frame_rate
    }
    let nal_hrd = r.flag()?;
    if nal_hrd {
        hrd(r)?;
    }
    let vcl_hrd = r.flag()?;
    if vcl_hrd {
        hrd(r)?;
    }
    if nal_hrd || vcl_hrd {
        r.skip(1)?; // low_delay_hrd_flag
    }
    r.skip(1)?; // pic_struct_present_flag
    if r.flag()? {
        r.skip(1)?;
        for _ in 0..4 {
            r.ue()?;
        }
        v.max_num_reorder_frames = Some(r.ue_max(16, "max_num_reorder_frames")?);
        v.max_dec_frame_buffering = Some(r.ue_max(16, "max_dec_frame_buffering")?);
    }
    Ok(v)
}

/// `frame_crop_{left,right,top,bottom}_offset` times CropUnitX/Y (7.4.2.1.1, eq. 7-19..7-22),
/// checked: offsets are ue(v) up to 2^32 - 2, so the products and sums are checked before the
/// size test.
fn frame_crop(
    [l, rr, t, b]: [u32; 4],
    (cx, cy): (u32, u32),
    width: u32,
    height: u32,
) -> VResult<(u32, u32, u32, u32)> {
    let crop = (
        l.checked_mul(cx),
        rr.checked_mul(cx),
        t.checked_mul(cy),
        b.checked_mul(cy),
    );
    let (Some(cl), Some(cr), Some(ct), Some(cb)) = crop else {
        return Err(H264Error::CropTooLarge.into());
    };
    let fits = |a: u32, b: u32, size: u32| a.checked_add(b).is_some_and(|sum| sum < size);
    if !fits(cl, cr, width) || !fits(ct, cb, height) {
        return Err(H264Error::CropTooLarge.into());
    }
    Ok((cl, cr, ct, cb))
}

pub fn parse_sps(rbsp: &[u8]) -> VResult<Sps> {
    let mut r = BitReader::new(rbsp);
    r.skip(8)?; // NAL header
    let profile_idc = low_u8(r.u(8)?);
    let constraint_flags = low_u8(r.u(8)?);
    let level_idc = low_u8(r.u(8)?);
    let id = r.ue_max(31, "seq_parameter_set_id")?;
    let mut s = Sps {
        profile_idc,
        constraint_flags,
        level_idc,
        id,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        qpprime_y_zero_transform_bypass: false,
        seq_scaling_matrix_present: false,
        scaling: ScalingMatrices::flat(),
        log2_max_frame_num_minus4: 0,
        pic_order_cnt_type: 0,
        log2_max_poc_lsb_minus4: 0,
        delta_pic_order_always_zero: false,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        offset_for_ref_frame: Vec::new(),
        max_num_ref_frames: 0,
        gaps_in_frame_num_allowed: false,
        pic_width_in_mbs: 0,
        pic_height_in_map_units: 0,
        frame_mbs_only: true,
        mb_adaptive_frame_field: false,
        direct_8x8_inference: false,
        crop: (0, 0, 0, 0),
        vui: Vui {
            colour_primaries: 2,
            transfer_characteristics: 2,
            matrix_coefficients: 2,
            ..Default::default()
        },
    };
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        s.chroma_format_idc = r.ue_max(3, "chroma_format_idc")?;
        if s.chroma_format_idc == 3 {
            s.separate_colour_plane = r.flag()?;
        }
        s.bit_depth_luma_minus8 = r.ue_max(6, "bit_depth_luma_minus8")?;
        s.bit_depth_chroma_minus8 = r.ue_max(6, "bit_depth_chroma_minus8")?;
        s.qpprime_y_zero_transform_bypass = r.flag()?;
        s.seq_scaling_matrix_present = r.flag()?;
        if s.seq_scaling_matrix_present {
            let n = if s.chroma_format_idc != 3 { 8 } else { 12 };
            s.scaling = scaling_matrices(&mut r, n, None)?;
        }
    }
    s.log2_max_frame_num_minus4 = r.ue_max(12, "log2_max_frame_num_minus4")?;
    s.pic_order_cnt_type = r.ue_max(2, "pic_order_cnt_type")?;
    match s.pic_order_cnt_type {
        0 => s.log2_max_poc_lsb_minus4 = r.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")?,
        1 => {
            s.delta_pic_order_always_zero = r.flag()?;
            s.offset_for_non_ref_pic = r.se()?;
            s.offset_for_top_to_bottom_field = r.se()?;
            let n = r.ue_max(255, "num_ref_frames_in_pic_order_cnt_cycle")?;
            for _ in 0..n {
                s.offset_for_ref_frame.push(r.se()?);
            }
        }
        _ => {}
    }
    s.max_num_ref_frames = r.ue_max(16, "max_num_ref_frames")?;
    s.gaps_in_frame_num_allowed = r.flag()?;
    s.pic_width_in_mbs = r.ue_max(1023, "pic_width_in_mbs_minus1")? + 1;
    s.pic_height_in_map_units = r.ue_max(1023, "pic_height_in_map_units_minus1")? + 1;
    s.frame_mbs_only = r.flag()?;
    if !s.frame_mbs_only {
        s.mb_adaptive_frame_field = r.flag()?;
    }
    s.direct_8x8_inference = r.flag()?;
    if r.flag()? {
        let (cx, cy) = match (s.chroma_format_idc, s.separate_colour_plane) {
            (0, _) | (3, true) => (1, 2 - u32::from(s.frame_mbs_only)),
            (1, _) => (2, 2 * (2 - u32::from(s.frame_mbs_only))),
            (2, _) => (2, 2 - u32::from(s.frame_mbs_only)),
            _ => (1, 2 - u32::from(s.frame_mbs_only)),
        };
        let offsets = [r.ue()?, r.ue()?, r.ue()?, r.ue()?];
        s.crop = frame_crop(offsets, (cx, cy), s.width(), s.height())?;
    }
    if r.flag()? {
        s.vui = vui(&mut r)?;
    }
    Ok(s)
}

#[derive(Debug, Clone)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub entropy_coding_mode: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub second_chroma_qp_index_offset: i32,
    /// Effective matrices for pictures using this PPS (PPS lists, else the SPS lists, else Flat_16).
    pub scaling: ScalingMatrices,
}

/// `sps_lookup` resolves the referenced SPS (needed for chroma_format_idc and fall-back rule B).
pub fn parse_pps(rbsp: &[u8], sps_lookup: impl Fn(u32) -> Option<Sps>) -> VResult<Pps> {
    let mut r = BitReader::new(rbsp);
    r.skip(8)?;
    let id = r.ue_max(255, "pic_parameter_set_id")?;
    let sps_id = r.ue_max(31, "seq_parameter_set_id")?;
    let Some(sps) = sps_lookup(sps_id) else {
        return Err(H264Error::MissingSps {
            pps: id,
            sps: sps_id,
        }
        .into());
    };
    let entropy_coding_mode = r.flag()?;
    let bfpo = r.flag()?;
    if r.ue()? > 0 {
        return Err(H264Error::Unsupported(
            "H.264 slice groups (FMO, num_slice_groups_minus1 > 0)",
        )
        .into());
    }
    let l0 = r.ue_max(31, "num_ref_idx_l0_default_active_minus1")?;
    let l1 = r.ue_max(31, "num_ref_idx_l1_default_active_minus1")?;
    let weighted_pred = r.flag()?;
    let weighted_bipred_idc = r.u(2)?;
    let pic_init_qp_minus26 = r.se_range(
        -(26 + 6 * fit::<i32, _>(sps.bit_depth_luma_minus8, "bit_depth_luma_minus8")?),
        25,
        "pic_init_qp_minus26",
    )?;
    r.se()?; // pic_init_qs_minus26
    let chroma_qp_index_offset = r.se_range(-12, 12, "chroma_qp_index_offset")?;
    let deblocking_filter_control_present = r.flag()?;
    let constrained_intra_pred = r.flag()?;
    let redundant_pic_cnt_present = r.flag()?;
    let mut p = Pps {
        id,
        sps_id,
        entropy_coding_mode,
        bottom_field_pic_order_in_frame_present: bfpo,
        num_ref_idx_l0_default_active_minus1: l0,
        num_ref_idx_l1_default_active_minus1: l1,
        weighted_pred,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        chroma_qp_index_offset,
        deblocking_filter_control_present,
        constrained_intra_pred,
        redundant_pic_cnt_present,
        transform_8x8_mode: false,
        second_chroma_qp_index_offset: chroma_qp_index_offset,
        scaling: sps.scaling.clone(),
    };
    if r.more_rbsp_data() {
        p.transform_8x8_mode = r.flag()?;
        if r.flag()? {
            let n = 6 + if sps.chroma_format_idc != 3 { 2 } else { 6 }
                * usize::from(p.transform_8x8_mode);
            // 7.4.2.2: fall-back rule A when the SPS carries no matrices, rule B otherwise.
            let fallback = sps.seq_scaling_matrix_present.then_some(&sps.scaling);
            p.scaling = scaling_matrices(&mut r, n, fallback)?;
        }
        p.second_chroma_qp_index_offset = r.se_range(-12, 12, "second_chroma_qp_index_offset")?;
    }
    Ok(p)
}
