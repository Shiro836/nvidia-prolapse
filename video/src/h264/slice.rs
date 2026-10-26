// SPDX-License-Identifier: MIT
//! H.264 slice header (7.3.3) up to and including dec_ref_pic_marking(). The engine parses the
//! slice header again itself (NEW_H264 takes the Annex-B slices verbatim); we need it for POC,
//! frame_num, reference marking and the picture-level flags.
use super::super::{VResult, bits::BitReader};
use super::ps::{Pps, Sps};
use crate::h264::H264Error;
use nv_rm_core::conv::{low_u8, usize_from_u32};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mmco {
    /// 1: unmark short-term picNumX = CurrPicNum − (difference_of_pic_nums_minus1 + 1)
    ShortUnused(u32),
    /// 2: unmark long-term LongTermPicNum
    LongUnused(u32),
    /// 3: short-term (difference_of_pic_nums_minus1) → long-term LongTermFrameIdx
    ShortToLong(u32, u32),
    /// 4: max_long_term_frame_idx_plus1
    MaxLongTermIdx(u32),
    /// 5: unmark all, reset
    Reset,
    /// 6: current → long-term LongTermFrameIdx
    CurrentToLong(u32),
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub nal_unit_type: u8,
    pub nal_ref_idc: u8,
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub field_pic: bool,
    pub bottom_field: bool,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub redundant_pic_cnt: u32,
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub no_output_of_prior_pics: bool,
    pub long_term_reference_flag: bool,
    /// `Some` iff adaptive_ref_pic_marking_mode_flag.
    pub mmco: Option<Vec<Mmco>>,
}
impl SliceHeader {
    pub fn idr(&self) -> bool {
        self.nal_unit_type == 5
    }
}

fn skip_ref_pic_list_modification(r: &mut BitReader) -> VResult<()> {
    if r.flag()? {
        loop {
            match r.ue()? {
                0..=2 => {
                    r.ue()?;
                }
                3 => break,
                v => return Err(H264Error::ModificationIdc(v).into()),
            }
        }
    }
    Ok(())
}

fn skip_pred_weight_table(r: &mut BitReader, sps: &Sps, b: bool, l0: u32, l1: u32) -> VResult<()> {
    r.ue()?; // luma_log2_weight_denom
    let chroma = sps.chroma_format_idc != 0 && !sps.separate_colour_plane;
    if chroma {
        r.ue()?;
    }
    for n in [l0, if b { l1 } else { u32::MAX }] {
        if n == u32::MAX {
            break;
        }
        for _ in 0..=n {
            if r.flag()? {
                r.se()?;
                r.se()?;
            }
            if chroma && r.flag()? {
                for _ in 0..4 {
                    r.se()?;
                }
            }
        }
    }
    Ok(())
}

/// Parses one slice header from an unescaped slice NAL (header byte included).
pub fn parse_slice_header(
    rbsp: &[u8],
    sps_of: impl Fn(u32) -> Option<(Sps, Pps)>,
) -> VResult<SliceHeader> {
    let mut r = BitReader::new(rbsp);
    let hdr = r.u(8)?;
    let nal_unit_type = low_u8(hdr & 0x1f);
    let nal_ref_idc = low_u8((hdr >> 5) & 3);
    let first_mb_in_slice = r.ue()?;
    let st = r.ue_max(9, "slice_type")?;
    let slice_type = [
        SliceType::P,
        SliceType::B,
        SliceType::I,
        SliceType::Sp,
        SliceType::Si,
    ][usize_from_u32(st % 5)];
    let pps_id = r.ue_max(255, "pic_parameter_set_id")?;
    let Some((sps, pps)) = sps_of(pps_id) else {
        return Err(H264Error::MissingPps(pps_id).into());
    };
    if sps.separate_colour_plane {
        return Err(H264Error::Unsupported(
            "H.264 separate_colour_plane_flag (4:4:4 independent planes)",
        )
        .into());
    }
    let frame_num = r.u(sps.log2_max_frame_num_minus4 + 4)?;
    let (mut field_pic, mut bottom_field) = (false, false);
    if !sps.frame_mbs_only {
        field_pic = r.flag()?;
        if field_pic {
            bottom_field = r.flag()?;
        }
    }
    let idr_pic_id = if nal_unit_type == 5 {
        r.ue_max(65535, "idr_pic_id")?
    } else {
        0
    };
    let mut h = SliceHeader {
        nal_unit_type,
        nal_ref_idc,
        first_mb_in_slice,
        slice_type,
        pps_id,
        frame_num,
        field_pic,
        bottom_field,
        idr_pic_id,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt_bottom: 0,
        delta_pic_order_cnt: [0; 2],
        redundant_pic_cnt: 0,
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        no_output_of_prior_pics: false,
        long_term_reference_flag: false,
        mmco: None,
    };
    if sps.pic_order_cnt_type == 0 {
        h.pic_order_cnt_lsb = r.u(sps.log2_max_poc_lsb_minus4 + 4)?;
        if pps.bottom_field_pic_order_in_frame_present && !field_pic {
            h.delta_pic_order_cnt_bottom = r.se()?;
        }
    }
    if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero {
        h.delta_pic_order_cnt[0] = r.se()?;
        if pps.bottom_field_pic_order_in_frame_present && !field_pic {
            h.delta_pic_order_cnt[1] = r.se()?;
        }
    }
    if pps.redundant_pic_cnt_present {
        h.redundant_pic_cnt = r.ue_max(127, "redundant_pic_cnt")?;
    }
    let b = slice_type == SliceType::B;
    if b {
        r.skip(1)?; // direct_spatial_mv_pred_flag
    }
    if matches!(slice_type, SliceType::P | SliceType::Sp | SliceType::B) && r.flag()? {
        let max = if field_pic { 31 } else { 15 };
        h.num_ref_idx_l0_active_minus1 = r.ue_max(max, "num_ref_idx_l0_active_minus1")?;
        if b {
            h.num_ref_idx_l1_active_minus1 = r.ue_max(max, "num_ref_idx_l1_active_minus1")?;
        }
    }
    if nal_unit_type == 20 || nal_unit_type == 21 {
        return Err(H264Error::Unsupported("H.264 MVC/SVC slice extension").into());
    }
    if slice_type != SliceType::I && slice_type != SliceType::Si {
        skip_ref_pic_list_modification(&mut r)?;
        if b {
            skip_ref_pic_list_modification(&mut r)?;
        }
    }
    if (pps.weighted_pred && matches!(slice_type, SliceType::P | SliceType::Sp))
        || (pps.weighted_bipred_idc == 1 && b)
    {
        skip_pred_weight_table(
            &mut r,
            &sps,
            b,
            h.num_ref_idx_l0_active_minus1,
            h.num_ref_idx_l1_active_minus1,
        )?;
    }
    if nal_ref_idc != 0 {
        if nal_unit_type == 5 {
            h.no_output_of_prior_pics = r.flag()?;
            h.long_term_reference_flag = r.flag()?;
        } else if r.flag()? {
            let mut ops = Vec::new();
            loop {
                let op = match r.ue()? {
                    0 => break,
                    1 => Mmco::ShortUnused(r.ue()?),
                    2 => Mmco::LongUnused(r.ue()?),
                    3 => {
                        let d = r.ue()?;
                        Mmco::ShortToLong(d, r.ue()?)
                    }
                    4 => Mmco::MaxLongTermIdx(r.ue_max(16, "max_long_term_frame_idx_plus1")?),
                    5 => Mmco::Reset,
                    6 => Mmco::CurrentToLong(r.ue()?),
                    v => return Err(H264Error::MmcoValue(v).into()),
                };
                ops.push(op);
                if ops.len() > 66 {
                    return Err(H264Error::TooManyMmcos.into());
                }
            }
            h.mmco = Some(ops);
        }
    }
    Ok(h)
}
