// SPDX-License-Identifier: MIT
//! HEVC (H.265) front end for NVDEC: Annex-B access unit → picture job (setup buffer, DPB slots,
//! engine bitstream, method values). Pure state machine; no device access.
//!
//! Spec processes implemented here: 8.1.3 (NoRaslOutputFlag, RASL skip after a random access
//! point), 8.3.1 (POC with MSB from prevTid0Pic), 8.3.2 (RPS-driven DPB marking), 8.3.4 initial
//! reference lists (repeated cyclically over all 16 entries as the stock driver does, ELF
//! 0x3e795).
//!
//! DPB indices: every picture gets the lowest slot (0..16) not held by a picture the current
//! RPS keeps; `curr_pic_idx`, the init lists and `RefDiffPicOrderCnts` use these slots (the
//! stock allocation observed in an 8-bit capture). The session binds each slot
//! to a surface (LUMA/CHROMA_OFFSET pairs) and colocated-MV area.
use nv_rm_core::conv::usize_from_u32;
pub mod codec;
pub mod desc;
pub mod ps;
pub mod slice;
#[cfg(test)]
mod tests;

use self::desc::{PicState, SessionInputs, StockOpaque};
use self::ps::{Pps, Sps};
use self::slice::parse_slice_header;
use crate::bits::{split_annexb, unescape};
use crate::{VResult, VideoError, bitstream, unsupported};

/// NVDEC `SET_APPLICATION_ID` for HEVC (clc9b0.h).
pub const APP_ID_HEVC: u32 = 7;
/// `SET_CONTROL_PARAMS` base observed for HEVC; the error-concealment reference index goes into
/// bits 7..11 (observed in hevc8.frames: = initreflistidxl0[0], 0 for intra pictures).
pub const CONTROL_PARAMS_BASE: u32 = 0x52057;

/// Method values the session emits for this picture (addresses are the session's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Methods {
    pub app_id: u32,
    pub control_params: u32,
    /// `SET_PICTURE_INDEX`: stock writes a running decode counter, not the DPB slot (observed 0..11
    /// in decode order in hevc8/hevc10.frames; the wrap point is not observed).
    pub picture_index: u32,
    /// HEVC_SET_SCALING_LIST = setup + this (only when the SPS enables scaling lists).
    pub scaling_list_offset: Option<u32>,
    /// HEVC_SET_TILE_SIZES = setup + this.
    pub tile_sizes_offset: u32,
}

#[derive(Debug, Clone)]
pub struct PictureJob {
    /// Monotonic id of the picture this job decodes.
    pub pic_id: u64,
    pub poc: i32,
    /// DPB slot of the current picture (= curr_pic_idx).
    pub slot: u8,
    /// Every DPB picture the engine may read: (slot, pic_id). The session programs these slots'
    /// surfaces; other slots get the current target.
    pub refs: Vec<(u8, u64)>,
    /// `desc::SETUP_SIZE` bytes.
    pub setup: Vec<u8>,
    /// Engine bitstream: each slice segment NAL (escaped) behind a `00 00 01` start code.
    pub bitstream: Vec<u8>,
    pub methods: Methods,
    pub keyframe: bool,
    /// pic_output_flag (skipped RASL pictures never produce a job).
    pub output: bool,
    pub width: u32,
    pub height: u32,
    /// Conformance window in luma samples: left, right, top, bottom.
    pub crop: [u32; 4],
    pub bit_depth: u32,
    /// sps_max_num_reorder_pics (display bumping bound).
    pub max_num_reorder: u32,
}

#[derive(Debug, Clone, Copy)]
struct DpbPic {
    slot: u8,
    poc: i32,
    id: u64,
}

pub struct Decoder {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Pps>>,
    dpb: Vec<DpbPic>,
    prev_tid0_poc: i32,
    /// Next picture starts a coded video sequence (start of stream, flush, end_of_seq).
    first_in_cvs: bool,
    /// The associated IRAP had NoRaslOutputFlag = 1: its RASL pictures are not decodable.
    skip_rasl: bool,
    next_id: u64,
    pub inputs: SessionInputs,
    /// Override of the stock-observed opaque bytes (None = [`StockOpaque::observed`]).
    pub opaque: Option<StockOpaque>,
}

struct SliceNal<'a> {
    nal: &'a [u8],
    hdr: slice::SliceHeader,
}

/// The RBSP after the two-byte NAL unit header (H.265 7.3.1.2) of a `what` NAL; a NAL shorter
/// than its header is a bitstream error.
fn payload<'a>(rbsp: &'a [u8], what: &str) -> VResult<&'a [u8]> {
    match rbsp.get(2..) {
        Some(p) => Ok(p),
        None => bitstream(format!(
            "HEVC {what} NAL of {} bytes has no NAL unit header",
            rbsp.len()
        )),
    }
}

fn bs_err<T>(m: impl Into<String>) -> Result<T, PicErr> {
    Err(PicErr::V(VideoError::Bitstream(m.into())))
}

enum PicErr {
    /// A RASL picture after a random access point: not decoded, not output (8.1.3).
    SkipRasl,
    V(VideoError),
}

impl Decoder {
    pub fn new(inputs: SessionInputs) -> Self {
        Self {
            sps: vec![None; 16],
            pps: vec![None; 64],
            dpb: Vec::new(),
            prev_tid0_poc: 0,
            first_in_cvs: true,
            skip_rasl: false,
            next_id: 0,
            inputs,
            opaque: None,
        }
    }

    /// Discards references (seek). The next picture must be an IRAP.
    pub fn flush(&mut self) {
        self.dpb.clear();
        self.first_in_cvs = true;
    }

    /// Decodes one access unit (Annex-B). Returns `None` for an AU without a decodable picture
    /// (parameter sets only, or a RASL picture skipped after a random access point).
    pub fn decode_au(&mut self, au: &[u8]) -> VResult<Option<PictureJob>> {
        let mut slices: Vec<SliceNal> = Vec::new();
        let mut nal_type0 = None;
        let mut tid0 = 0;
        for n in split_annexb(au) {
            let nal = n.nal;
            if nal.len() < 2 {
                return bitstream("NAL unit shorter than its header");
            }
            if nal[0] & 0x80 != 0 {
                return bitstream("forbidden_zero_bit set");
            }
            let t = (nal[0] >> 1) & 0x3f;
            let layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
            let tid_plus1 = nal[1] & 7;
            if tid_plus1 == 0 {
                return bitstream("nuh_temporal_id_plus1 = 0");
            }
            if layer != 0 {
                continue; // base-layer decoder: other layers are not decoded (F.8 / 8.1.1)
            }
            match t {
                33 => {
                    let s = ps::parse_sps(&unescape(nal))?;
                    let id = usize_from_u32(s.id);
                    self.sps[id] = Some(s);
                }
                34 => {
                    let p = ps::parse_pps(&unescape(nal))?;
                    let id = usize_from_u32(p.id);
                    self.pps[id] = Some(p);
                }
                36 => self.first_in_cvs = true, // end_of_seq: next picture is an IRAP with NoRaslOutputFlag
                0..=9 | 16..=21 => {
                    let rbsp = unescape(nal);
                    let mut hdr = parse_slice_header(
                        &rbsp,
                        t,
                        |i| self.pps[usize_from_u32(i)].as_ref(),
                        |i| self.sps[usize_from_u32(i)].as_ref(),
                    )?;
                    if hdr.first_slice_segment_in_pic {
                        if !slices.is_empty() {
                            return bitstream("two pictures in one access unit");
                        }
                        nal_type0 = Some(t);
                        tid0 = tid_plus1 - 1;
                    } else if slices.is_empty() {
                        return bitstream("access unit starts with a non-first slice segment");
                    } else if Some(t) != nal_type0 {
                        return bitstream(
                            "slice segments of one picture with different NAL unit types",
                        );
                    }
                    if hdr.dependent {
                        let Some(p) = slices
                            .iter()
                            .rev()
                            .find(|s| !s.hdr.dependent)
                            .map(|s| s.hdr.clone())
                        else {
                            return bitstream("dependent slice segment without an independent one");
                        };
                        hdr = slice::SliceHeader {
                            first_slice_segment_in_pic: false,
                            dependent: true,
                            segment_address: hdr.segment_address,
                            ..p
                        };
                    }
                    slices.push(SliceNal { nal, hdr });
                }
                10..=15 | 22 | 23 => {
                    return unsupported(format!("HEVC reserved VCL NAL unit type {t}"));
                }
                _ => {} // VPS, AUD, EOB, filler, SEI, reserved/unspecified non-VCL
            }
        }
        let Some(t) = nal_type0 else { return Ok(None) };
        match self.picture(t, tid0, &slices) {
            Ok(j) => Ok(Some(j)),
            Err(PicErr::SkipRasl) => Ok(None),
            Err(PicErr::V(e)) => Err(e),
        }
    }

    fn picture(&mut self, t: u8, tid: u8, slices: &[SliceNal]) -> Result<PictureJob, PicErr> {
        let h0 = &slices[0].hdr;
        let Some(pps) = self.pps[usize_from_u32(h0.pps_id)].clone() else {
            return bs_err("missing PPS");
        };
        let Some(sps) = self.sps[usize_from_u32(pps.sps_id)].clone() else {
            return bs_err("missing SPS");
        };
        let idr = t == 19 || t == 20;
        let bla = (16..=18).contains(&t);
        let irap = (16..=23).contains(&t);
        let rasl = t == 8 || t == 9;
        let radl = t == 6 || t == 7;
        // 8.1.3 NoRaslOutputFlag (HandleCraAsBlaFlag = 0).
        let no_rasl_output = irap && (idr || bla || self.first_in_cvs);
        if irap {
            self.skip_rasl = no_rasl_output;
        }
        if rasl && self.skip_rasl {
            return Err(PicErr::SkipRasl);
        }
        if self.first_in_cvs && !irap {
            return bs_err("coded video sequence does not start with an IRAP picture");
        }
        self.first_in_cvs = false;
        // 8.3.1 picture order count.
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let lsb = h0.poc_lsb.cast_signed();
        let msb = if irap && no_rasl_output {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        let slnr = t <= 14 && t.is_multiple_of(2);
        if tid == 0 && !rasl && !radl && !slnr {
            self.prev_tid0_poc = poc;
        }
        // 8.3.2 reference picture set and DPB marking.
        if irap && no_rasl_output {
            self.dpb.clear();
        }
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut keep: Vec<u64> = Vec::new();
        if !idr {
            for (list, out) in [(&h0.rps.s0, &mut before), (&h0.rps.s1, &mut after)] {
                for &(dp, used) in list.iter() {
                    match self.dpb.iter().find(|d| d.poc == poc + dp) {
                        Some(p) => {
                            if used {
                                out.push(*p);
                            }
                            keep.push(p.id);
                        }
                        None if used => {
                            return bs_err(format!(
                                "reference picture POC {} missing from the DPB",
                                poc + dp
                            ));
                        }
                        None => {} // "no reference picture" in a Foll list is allowed
                    }
                }
            }
        }
        self.dpb.retain(|d| keep.contains(&d.id));
        let Some(slot) = (0u8..16).find(|s| self.dpb.iter().all(|d| d.slot != *s)) else {
            return bs_err("DPB holds 16 pictures");
        };
        // 8.3.4: RefPicListTemp0 = StCurrBefore ++ StCurrAfter (LtCurr is rejected in the slice
        // parser), Temp1 = StCurrAfter ++ StCurrBefore, repeated over all 16 entries.
        let total = before.len() + after.len();
        let cyc = |a: &[DpbPic], b: &[DpbPic]| -> [u8; 16] {
            let v: Vec<u8> = a.iter().chain(b.iter()).map(|p| p.slot).collect();
            let mut l = [0u8; 16];
            if !v.is_empty() {
                for (i, e) in l.iter_mut().enumerate() {
                    *e = v[i % v.len()];
                }
            }
            l
        };
        let list0 = cyc(&before, &after);
        let list1 = cyc(&after, &before);
        let mut diff = [0i16; 16];
        for d in &self.dpb {
            let v = poc - d.poc;
            let Ok(v16) = i16::try_from(v) else {
                return bs_err(format!("POC distance {v} exceeds 16 bits"));
            };
            diff[usize::from(d.slot)] = v16;
        }
        let mut bs = Vec::new();
        for s in slices {
            bs.extend_from_slice(&[0, 0, 1]);
            bs.extend_from_slice(s.nal);
        }
        let st = PicState {
            stream_len: bs.len() as u32,
            idr,
            irap,
            curr_slot: slot,
            num_poc_total_curr: total as u32,
            list0,
            list1,
            diff,
            sw_hdr_skip_length: h0.rps_syntax_bits as u16,
            num_bits_st_rps: h0.st_rps_bits,
            num_delta_pocs_ref: h0.num_delta_pocs_ref,
        };
        let op = self
            .opaque
            .unwrap_or(StockOpaque::observed(sps.bit_depth_luma));
        let setup = desc::build(&sps, &pps, &st, &self.inputs, op).map_err(PicErr::V)?;
        let err_idx = if total > 0 { u32::from(list0[0]) } else { 0 };
        let id = self.next_id;
        self.next_id += 1;
        let refs = self.dpb.iter().map(|d| (d.slot, d.id)).collect();
        self.dpb.push(DpbPic { slot, poc, id });
        Ok(PictureJob {
            pic_id: id,
            poc,
            slot,
            refs,
            setup,
            bitstream: bs,
            methods: Methods {
                app_id: APP_ID_HEVC,
                control_params: CONTROL_PARAMS_BASE | (err_idx & 0x1f) << 7,
                picture_index: id as u32,
                scaling_list_offset: sps
                    .scaling_list_enabled
                    .then_some(desc::SCALING_LIST_OFFSET as u32),
                tile_sizes_offset: desc::TILE_SIZES_OFFSET as u32,
            },
            keyframe: irap,
            output: h0.pic_output,
            width: sps.width,
            height: sps.height,
            crop: [
                sps.conf_win[0] * 2,
                sps.conf_win[1] * 2,
                sps.conf_win[2] * 2,
                sps.conf_win[3] * 2,
            ],
            bit_depth: sps.bit_depth_luma,
            max_num_reorder: sps.max_num_reorder_pics,
        })
    }
}
