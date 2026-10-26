// SPDX-License-Identifier: MIT
//! HEVC slice segment header (H.265 7.3.6.1), parsed up to the end of the reference picture set
//! syntax: everything after it (list modification, weights, QP, deblocking, entry points) is
//! parsed by the engine itself (`sw_hdr_skip_length` tells it what the driver consumed).
use super::ps::{Pps, Sps, StRps, st_ref_pic_set};
use crate::VResult;
use crate::bits::BitReader;
use crate::fit;
use crate::hevc::HevcError;
use nv_rm_core::conv::usize_from_u32;

pub const SLICE_B: u32 = 0;
pub const SLICE_P: u32 = 1;
pub const SLICE_I: u32 = 2;

#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_slice_segment_in_pic: bool,
    pub no_output_of_prior_pics: bool,
    pub pps_id: u32,
    pub dependent: bool,
    pub segment_address: u32,
    /// Fields below are only meaningful for independent slice segments.
    pub slice_type: u32,
    pub pic_output: bool,
    pub poc_lsb: u32,
    pub sps_rps_flag: bool,
    /// The RPS in effect (inline or SPS[idx]).
    pub rps: StRps,
    /// RBSP bits after `short_term_ref_pic_set_sps_flag`: the inline `st_ref_pic_set()` or the
    /// `short_term_ref_pic_set_idx` (FFmpeg `short_term_ref_pic_set_size`, passed as CUVID
    /// NumBitsForShortTermRPSInSlice, hevcdec.c:876-897); 0 for IDR.
    pub st_rps_bits: u32,
    /// NumDeltaPocs[RefRpsIdx] of an inline inter-predicted RPS, else 0.
    pub num_delta_pocs_ref: u32,
    /// RBSP bits from `slice_pic_order_cnt_lsb` to the end of the long-term syntax (0 for IDR).
    pub rps_syntax_bits: u32,
}

fn ceil_log2(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        32 - (v - 1).leading_zeros()
    }
}

/// `pps_of`/`sps_of` look up active parameter sets by id. `nal_type` is the NAL unit type.
pub fn parse_slice_header<'a>(
    rbsp: &[u8],
    nal_type: u8,
    pps_of: impl Fn(u32) -> Option<&'a Pps>,
    sps_of: impl Fn(u32) -> Option<&'a Sps>,
) -> VResult<SliceHeader> {
    let mut rd = BitReader::new(super::payload(rbsp, "slice segment")?);
    let r = &mut rd;
    let first = r.flag()?;
    let no_output_of_prior_pics = if (16..=23).contains(&nal_type) {
        r.flag()?
    } else {
        false
    };
    let pps_id = r.ue_max(63, "slice_pic_parameter_set_id")?;
    let Some(pps) = pps_of(pps_id) else {
        return Err(HevcError::MissingPps(pps_id).into());
    };
    let Some(sps) = sps_of(pps.sps_id) else {
        return Err(HevcError::MissingSps {
            pps: pps_id,
            sps: pps.sps_id,
        }
        .into());
    };
    let mut h = SliceHeader {
        first_slice_segment_in_pic: first,
        no_output_of_prior_pics,
        pps_id,
        dependent: false,
        segment_address: 0,
        slice_type: SLICE_I,
        pic_output: true,
        poc_lsb: 0,
        sps_rps_flag: false,
        rps: StRps::default(),
        st_rps_bits: 0,
        num_delta_pocs_ref: 0,
        rps_syntax_bits: 0,
    };
    if !first {
        if pps.dependent_slice_segments_enabled {
            h.dependent = r.flag()?;
        }
        let ctbs = sps.width_ctbs() * sps.height_ctbs();
        h.segment_address = r.u(ceil_log2(ctbs))?;
        if h.segment_address >= ctbs {
            return Err(HevcError::Malformed("slice_segment_address beyond the picture").into());
        }
    }
    if h.dependent {
        return Ok(h);
    }
    r.skip(usize_from_u32(pps.num_extra_slice_header_bits))?;
    h.slice_type = r.ue_max(2, "slice_type")?;
    if pps.output_flag_present {
        h.pic_output = r.flag()?;
    }
    if sps.separate_colour_plane {
        r.skip(2)?;
    }
    let idr = nal_type == 19 || nal_type == 20;
    if !idr {
        let start = r.pos();
        h.poc_lsb = r.u(sps.log2_max_poc_lsb)?;
        h.sps_rps_flag = r.flag()?;
        let n = sps.st_rps.len();
        let p = r.pos();
        if !h.sps_rps_flag {
            let (rps, ndp) = st_ref_pic_set(r, n, &sps.st_rps, true)?;
            h.num_delta_pocs_ref = ndp.unwrap_or(0);
            h.rps = rps;
        } else {
            if n == 0 {
                return Err(HevcError::Malformed(
                    "short_term_ref_pic_set_sps_flag with no SPS sets",
                )
                .into());
            }
            let idx = usize_from_u32(r.u(ceil_log2(fit(n, "num_short_term_ref_pic_sets")?))?);
            if idx >= n {
                return Err(HevcError::Malformed("short_term_ref_pic_set_idx out of range").into());
            }
            h.rps = sps.st_rps[idx].clone();
        }
        h.st_rps_bits = fit(r.pos() - p, "st_ref_pic_set bits")?;
        if sps.long_term_ref_pics_present {
            let num_lt_sps = if sps.num_long_term_ref_pics_sps > 0 {
                r.ue()?
            } else {
                0
            };
            let num_lt_pics = r.ue()?;
            if num_lt_sps + num_lt_pics > 0 {
                return Err(HevcError::Unsupported(
                    "HEVC long-term reference pictures in a slice header (sw_hdr_skip_length coverage of the \
                     long-term syntax and the longtermflag bit order are not verified against stock)",
                )
                .into());
            }
        }
        h.rps_syntax_bits = fit(r.pos() - start, "slice RPS syntax bits")?;
    }
    Ok(h)
}
