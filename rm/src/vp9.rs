//! VP9 bitstream front end for NVDEC: IVF/superframe splitting, the
//! uncompressed and compressed frame headers, probability contexts with
//! backward adaptation, and reference/surface bookkeeping.
//!
//! Written from the VP9 Bitstream & Decoding Process Specification v0.6
//! (sections 6.2-6.4, 8.4.2, 9.2) with libvpx (BSD) 63bcd588b50b as the
//! cross-check for adaptation (vp9_entropy.c adapt_coef_probs,
//! vp9_entropymode.c vp9_adapt_mode_probs, vp9_entropymv.c vp9_adapt_mv_probs,
//! vpx_dsp/prob.h merge_probs) and compressed-header order
//! (vp9_decodeframe.c read_compressed_header). Default tables: `vp9_tables`.
//! The engine-side count semantics (per-node inter-mode counts, coefficient
//! [ZERO, ONE, TWO+, EOB_MODEL] + separate eob-branch counts) follow the
//! public nvdec_drv.h struct (identical to the Hantro G2 layout, see Linux
//! drivers/media/platform/verisilicon/hantro_g2_vp9_dec.c hantro_g2_vp9_dec_done
//! for the same mapping onto libvpx counts).
use crate::nvdec_drv::{CoefCounts, CoefProbs, Pod, Vp9EntropyCounts, Vp9EntropyProbs};
use crate::rm::{Error, Result};
use crate::vp9_tables as t;

fn bad(msg: impl Into<String>) -> Error {
    Error::Contract(format!("VP9: {}", msg.into()))
}

/// Frames of an IVF file (32-byte header, 12-byte frame headers).
pub fn ivf_frames(data: &[u8]) -> Result<Vec<&[u8]>> {
    if data.len() < 32 || &data[0..4] != b"DKIF" || &data[8..12] != b"VP90" {
        return Err(bad("input is not an IVF VP90 file"));
    }
    let header = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut pos = header;
    let mut out = Vec::new();
    while pos + 12 <= data.len() {
        let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        let frame = data
            .get(pos..pos + size)
            .ok_or_else(|| bad("truncated IVF frame"))?;
        out.push(frame);
        pos += size;
    }
    Ok(out)
}

/// Split a VP9 superframe (spec Annex B) into its frames; a plain frame is
/// returned as is.
pub fn split_superframe(data: &[u8]) -> Result<Vec<&[u8]>> {
    let Some(&last) = data.last() else {
        return Err(bad("empty packet"));
    };
    if last & 0xe0 != 0xc0 {
        return Ok(vec![data]);
    }
    let frames = (last & 7) as usize + 1;
    let mag = ((last >> 3) & 3) as usize + 1;
    let index = 2 + mag * frames;
    if data.len() < index || data[data.len() - index] != last {
        return Ok(vec![data]);
    }
    let mut pos = data.len() - index + 1;
    let mut off = 0usize;
    let mut out = Vec::new();
    for _ in 0..frames {
        let mut size = 0usize;
        for b in 0..mag {
            size |= (data[pos + b] as usize) << (8 * b);
        }
        pos += mag;
        let frame = data
            .get(off..off + size)
            .ok_or_else(|| bad("superframe index overruns packet"))?;
        out.push(frame);
        off += size;
    }
    Ok(out)
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn bit(&mut self) -> Result<u32> {
        let byte = *self
            .data
            .get(self.pos / 8)
            .ok_or_else(|| bad("uncompressed header overruns frame"))?;
        let b = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Ok(b as u32)
    }
    fn f(&mut self, n: u32) -> Result<u32> {
        let mut v = 0;
        for _ in 0..n {
            v = v << 1 | self.bit()?;
        }
        Ok(v)
    }
    /// su(n): magnitude then sign.
    fn s(&mut self, n: u32) -> Result<i32> {
        let v = self.f(n)? as i32;
        Ok(if self.bit()? != 0 { -v } else { v })
    }
}

/// Boolean decoder, spec 9.2 (bit-serial form; bits past the end read 0).
struct BoolDecoder<'a> {
    data: &'a [u8],
    bit: usize,
    value: u32,
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(bad("empty compressed header"));
        }
        let mut d = Self {
            data,
            bit: 8,
            value: data[0] as u32,
            range: 255,
        };
        if d.read(128) != 0 {
            return Err(bad("compressed header marker bit set"));
        }
        Ok(d)
    }
    fn next_bit(&mut self) -> u32 {
        let b = self
            .data
            .get(self.bit / 8)
            .map_or(0, |byte| (byte >> (7 - self.bit % 8)) & 1);
        self.bit += 1;
        b as u32
    }
    fn read(&mut self, p: u8) -> u32 {
        let split = 1 + (((self.range - 1) * p as u32) >> 8);
        let bit = if self.value < split {
            self.range = split;
            0
        } else {
            self.range -= split;
            self.value -= split;
            1
        };
        while self.range < 128 {
            self.value = self.value << 1 | self.next_bit();
            self.range <<= 1;
        }
        bit
    }
    fn literal(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |v, _| v << 1 | self.read(128))
    }
    fn diff_update(&mut self, p: &mut u8) {
        if self.read(252) != 0 {
            let delp = if self.read(128) == 0 {
                self.literal(4)
            } else if self.read(128) == 0 {
                self.literal(4) + 16
            } else if self.read(128) == 0 {
                self.literal(5) + 32
            } else {
                let v = self.literal(7);
                (if v < 65 {
                    v
                } else {
                    (v << 1) - 65 + self.read(128)
                }) + 64
            };
            *p = inv_remap_prob(delp as usize, *p);
        }
    }
    fn mv_update(&mut self, p: &mut u8) {
        if self.read(252) != 0 {
            *p = (self.literal(7) << 1 | 1) as u8;
        }
    }
}

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

fn inv_remap_prob(delta: usize, prob: u8) -> u8 {
    let v = t::INV_MAP_TABLE[delta.min(254)] as i32;
    let m = prob as i32 - 1;
    if (m << 1) <= 255 {
        (1 + inv_recenter_nonneg(v, m)) as u8
    } else {
        (255 - inv_recenter_nonneg(v, 254 - m)) as u8
    }
}

/// Model coefficient probabilities of one transform size: [plane][ref][band][ctx][node].
pub type ModelCoefProbs = [[[[[u8; 3]; 6]; 6]; 2]; 2];

/// One probability context (spec "saved probabilities"), libvpx FRAME_CONTEXT
/// order.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FrameContext {
    pub tx8: [[u8; 1]; 2],
    pub tx16: [[u8; 2]; 2],
    pub tx32: [[u8; 3]; 2],
    pub coef: [ModelCoefProbs; 4],
    pub skip: [u8; 3],
    pub inter_mode: [[u8; 3]; 7],
    pub interp_filter: [[u8; 2]; 4],
    pub is_inter: [u8; 4],
    pub comp_inter: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub mv_joints: [u8; 3],
    pub mv: [MvComponent; 2],
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct MvComponent {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: [u8; 1],
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

fn fill<const N: usize>(dst: &mut [[u8; N]], src: &[u8]) {
    for (i, row) in dst.iter_mut().enumerate() {
        row.copy_from_slice(&src[i * N..(i + 1) * N]);
    }
}

impl Default for FrameContext {
    fn default() -> Self {
        let mut c = FrameContext {
            tx8: [[0; 1]; 2],
            tx16: [[0; 2]; 2],
            tx32: [[0; 3]; 2],
            coef: [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4],
            skip: t::DEFAULT_SKIP_PROBS,
            inter_mode: [[0; 3]; 7],
            interp_filter: [[0; 2]; 4],
            is_inter: t::DEFAULT_INTRA_INTER_P,
            comp_inter: t::DEFAULT_COMP_INTER_P,
            single_ref: [[0; 2]; 5],
            comp_ref: t::DEFAULT_COMP_REF_P,
            y_mode: [[0; 9]; 4],
            uv_mode: [[0; 9]; 10],
            partition: [[0; 3]; 16],
            mv_joints: t::DEFAULT_MV_JOINTS,
            mv: [MvComponent::default(); 2],
        };
        fill(&mut c.tx8, &t::DEFAULT_TX_P8);
        fill(&mut c.tx16, &t::DEFAULT_TX_P16);
        fill(&mut c.tx32, &t::DEFAULT_TX_P32);
        let coefs = [
            &t::DEFAULT_COEF_PROBS_4X4,
            &t::DEFAULT_COEF_PROBS_8X8,
            &t::DEFAULT_COEF_PROBS_16X16,
            &t::DEFAULT_COEF_PROBS_32X32,
        ];
        for (tx, src) in coefs.iter().enumerate() {
            let mut k = 0;
            for plane in c.coef[tx].iter_mut() {
                for r in plane.iter_mut() {
                    for band in r.iter_mut() {
                        for ctx in band.iter_mut() {
                            ctx.copy_from_slice(&src[k..k + 3]);
                            k += 3;
                        }
                    }
                }
            }
        }
        fill(&mut c.inter_mode, &t::DEFAULT_INTER_MODE_PROBS);
        fill(&mut c.interp_filter, &t::DEFAULT_SWITCHABLE_INTERP_PROB);
        fill(&mut c.single_ref, &t::DEFAULT_SINGLE_REF_P);
        fill(&mut c.y_mode, &t::DEFAULT_IF_Y_PROBS);
        fill(&mut c.uv_mode, &t::DEFAULT_IF_UV_PROBS);
        fill(&mut c.partition, &t::DEFAULT_PARTITION_PROBS);
        for (i, comp) in c.mv.iter_mut().enumerate() {
            let s = &t::DEFAULT_MV_COMPS[i * 33..(i + 1) * 33];
            comp.sign = s[0];
            comp.classes.copy_from_slice(&s[1..11]);
            comp.class0[0] = s[11];
            comp.bits.copy_from_slice(&s[12..22]);
            comp.class0_fp[0].copy_from_slice(&s[22..25]);
            comp.class0_fp[1].copy_from_slice(&s[25..28]);
            comp.fp.copy_from_slice(&s[28..31]);
            comp.class0_hp = s[31];
            comp.hp = s[32];
        }
        c
    }
}

pub const KEY_FRAME: u32 = 0;
pub const SWITCHABLE: u8 = 4;
/// Uncompressed-header interpolation filter literal -> libvpx enum
/// (EIGHTTAP 0, EIGHTTAP_SMOOTH 1, EIGHTTAP_SHARP 2, BILINEAR 3;
/// libvpx vp9_decodeframe.c literal_to_filter).
pub const LITERAL_TO_FILTER: [u8; 4] = [1, 0, 2, 3];
pub const TX_MODE_SELECT: u8 = 4;
pub const REFERENCE_MODE_SELECT: u8 = 2;

#[derive(Clone, Debug, Default)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub abs_delta: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    pub feature_enabled: [[bool; 4]; 8],
    pub feature_data: [[i16; 4]; 8],
}

/// Parsed frame header (uncompressed + compressed) of one coded frame.
#[derive(Clone, Debug, Default)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show: usize,
    pub frame_type: u32,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_frame_context: u32,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [usize; 3],
    /// Indexed by reference frame (0 = intra, 1 = last, 2 = golden, 3 = altref).
    pub sign_bias: [u8; 4],
    pub width: u32,
    pub height: u32,
    pub allow_high_precision_mv: bool,
    pub interp_filter: u8,
    pub refresh_frame_context: bool,
    pub frame_parallel: bool,
    pub frame_context_idx: usize,
    pub filter_level: u8,
    pub sharpness: u8,
    pub mode_ref_delta_enabled: bool,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub lossless: bool,
    pub log2_tile_cols: u32,
    pub log2_tile_rows: u32,
    pub header_size: usize,
    pub uncompressed_size: usize,
    pub tx_mode: u8,
    pub reference_mode: u8,
    pub comp_fixed_ref: u8,
    pub comp_var_ref: [u8; 2],
}

impl FrameHeader {
    pub fn is_intra(&self) -> bool {
        self.frame_type == KEY_FRAME || self.intra_only
    }
}

#[derive(Clone, Copy, Default)]
struct RefSlot {
    surface: Option<usize>,
    width: u32,
    height: u32,
}

/// Decoder state persisting across frames.
pub struct Vp9State {
    contexts: [FrameContext; 4],
    /// Probabilities of the frame being decoded (after forward updates).
    pub fc: FrameContext,
    slots: [RefSlot; 8],
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
    pub seg: Segmentation,
    pub last_frame_type: u32,
    pub last_show_frame: bool,
    pub last_size: Option<(u32, u32)>,
    /// (color_space, color_range) of the last key / intra-only frame.
    pub color: (u32, bool),
    /// Luma bit depth of the last key / intra-only frame (8; 10 or 12 in profile 2).
    pub bit_depth: u32,
}

impl Default for Vp9State {
    fn default() -> Self {
        Self {
            contexts: [FrameContext::default(); 4],
            fc: FrameContext::default(),
            slots: [RefSlot::default(); 8],
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
            seg: Segmentation::default(),
            last_frame_type: KEY_FRAME,
            last_show_frame: false,
            last_size: None,
            color: (0, false),
            bit_depth: 8,
        }
    }
}

impl Vp9State {
    /// Surface held by reference slot `i`.
    pub fn slot_surface(&self, i: usize) -> Option<usize> {
        self.slots[i].surface
    }
    pub fn slot_size(&self, i: usize) -> (u32, u32) {
        (self.slots[i].width, self.slots[i].height)
    }
    /// Surfaces referenced by any slot.
    pub fn referenced(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().filter_map(|s| s.surface)
    }

    fn setup_past_independence(&mut self, h: &FrameHeader) {
        self.seg.feature_enabled = [[false; 4]; 8];
        self.seg.feature_data = [[0; 4]; 8];
        self.seg.abs_delta = false;
        self.ref_deltas = [1, 0, -1, -1];
        self.mode_deltas = [0, 0];
        let defaults = FrameContext::default();
        if h.frame_type == KEY_FRAME || h.error_resilient || h.reset_frame_context == 3 {
            self.contexts = [defaults; 4];
        } else if h.reset_frame_context == 2 {
            self.contexts[h.frame_context_idx] = defaults;
        }
    }

    /// Parse the uncompressed and compressed headers of one frame and load the
    /// frame's probabilities into `self.fc`.
    pub fn parse(&mut self, data: &[u8]) -> Result<FrameHeader> {
        let mut r = BitReader { data, pos: 0 };
        let mut h = FrameHeader::default();
        if r.f(2)? != 2 {
            return Err(bad("frame marker"));
        }
        let profile = r.bit()? | r.bit()? << 1;
        if profile != 0 && profile != 2 {
            return Err(bad(format!(
                "profile {profile} unsupported (4:2:0 profiles 0 and 2 only)"
            )));
        }
        h.show_existing_frame = r.bit()? != 0;
        if h.show_existing_frame {
            h.frame_to_show = r.f(3)? as usize;
            if self.slots[h.frame_to_show].surface.is_none() {
                return Err(bad("show_existing_frame of an empty slot"));
            }
            return Ok(h);
        }
        h.frame_type = r.bit()?;
        h.show_frame = r.bit()? != 0;
        h.error_resilient = r.bit()? != 0;
        let sync = |r: &mut BitReader| -> Result<()> {
            if r.f(24)? != 0x498342 {
                return Err(bad("frame sync code"));
            }
            Ok(())
        };
        // color_config (spec 6.2.2): profile >= 2 starts with ten_or_twelve_bit; profiles 0 and
        // 2 are 4:2:0 for every colour space but RGB (which is 4:4:4, profiles 1/3 only).
        let color = |r: &mut BitReader| -> Result<(u32, bool, u32)> {
            let bit_depth = if profile >= 2 {
                if r.bit()? != 0 { 12 } else { 10 }
            } else {
                8
            };
            let cs = r.f(3)?;
            if cs == 7 {
                return Err(bad(format!(
                    "RGB colour space unsupported in profile {profile}"
                )));
            }
            let range = r.bit()?;
            Ok((cs, range != 0, bit_depth))
        };
        let frame_size = |r: &mut BitReader, h: &mut FrameHeader| -> Result<()> {
            h.width = r.f(16)? + 1;
            h.height = r.f(16)? + 1;
            Ok(())
        };
        let render_size = |r: &mut BitReader| -> Result<()> {
            if r.bit()? != 0 {
                r.f(32)?;
            }
            Ok(())
        };
        if h.frame_type == KEY_FRAME {
            sync(&mut r)?;
            let (cs, range, bd) = color(&mut r)?;
            (self.color, self.bit_depth) = ((cs, range), bd);
            frame_size(&mut r, &mut h)?;
            render_size(&mut r)?;
            h.refresh_frame_flags = 0xff;
        } else {
            h.intra_only = if h.show_frame { false } else { r.bit()? != 0 };
            h.reset_frame_context = if h.error_resilient { 0 } else { r.f(2)? };
            if h.intra_only {
                sync(&mut r)?;
                if profile > 0 {
                    let (cs, range, bd) = color(&mut r)?;
                    (self.color, self.bit_depth) = ((cs, range), bd);
                } else {
                    // Profile 0 intra-only frames carry no colour config: BT.601,
                    // studio range, 8-bit (spec 7.2, refresh_frame_flags follows).
                    (self.color, self.bit_depth) = ((1, false), 8);
                }
                h.refresh_frame_flags = r.f(8)? as u8;
                frame_size(&mut r, &mut h)?;
                render_size(&mut r)?;
            } else {
                h.refresh_frame_flags = r.f(8)? as u8;
                for i in 0..3 {
                    h.ref_frame_idx[i] = r.f(3)? as usize;
                    h.sign_bias[1 + i] = r.bit()? as u8;
                }
                let mut found = false;
                for i in 0..3 {
                    if r.bit()? != 0 {
                        let s = self.slots[h.ref_frame_idx[i]];
                        (h.width, h.height) = (s.width, s.height);
                        found = true;
                        break;
                    }
                }
                if !found {
                    frame_size(&mut r, &mut h)?;
                }
                render_size(&mut r)?;
                for i in 0..3 {
                    if self.slots[h.ref_frame_idx[i]].surface.is_none() {
                        return Err(bad("inter frame references an empty slot"));
                    }
                }
                h.allow_high_precision_mv = r.bit()? != 0;
                h.interp_filter = if r.bit()? != 0 {
                    SWITCHABLE
                } else {
                    LITERAL_TO_FILTER[r.f(2)? as usize]
                };
            }
        }
        if !h.error_resilient {
            h.refresh_frame_context = r.bit()? != 0;
            h.frame_parallel = r.bit()? != 0;
        } else {
            h.frame_parallel = true;
        }
        h.frame_context_idx = r.f(2)? as usize;
        if h.is_intra() || h.error_resilient {
            self.setup_past_independence(&h);
            h.frame_context_idx = 0;
            // libvpx vp9_setup_past_independence zeroes ref_frame_sign_bias
            // (vp9_entropymode.c:466), after the header read it; FFmpeg
            // vp9.c:594-598 likewise. Affects compound_allowed below.
            h.sign_bias = [0; 4];
        }
        // loop_filter_params
        h.filter_level = r.f(6)? as u8;
        h.sharpness = r.f(3)? as u8;
        h.mode_ref_delta_enabled = r.bit()? != 0;
        if h.mode_ref_delta_enabled && r.bit()? != 0 {
            for d in self.ref_deltas.iter_mut() {
                if r.bit()? != 0 {
                    *d = r.s(6)? as i8;
                }
            }
            for d in self.mode_deltas.iter_mut() {
                if r.bit()? != 0 {
                    *d = r.s(6)? as i8;
                }
            }
        }
        // quantization_params
        h.base_q_idx = r.f(8)? as u8;
        let mut dq = || -> Result<i8> { Ok(if r.bit()? != 0 { r.s(4)? as i8 } else { 0 }) };
        h.delta_q_y_dc = dq()?;
        h.delta_q_uv_dc = dq()?;
        h.delta_q_uv_ac = dq()?;
        h.lossless = h.base_q_idx == 0
            && h.delta_q_y_dc == 0
            && h.delta_q_uv_dc == 0
            && h.delta_q_uv_ac == 0;
        // segmentation_params
        let seg = &mut self.seg;
        seg.update_map = false;
        seg.temporal_update = false;
        seg.enabled = r.bit()? != 0;
        if seg.enabled {
            seg.update_map = r.bit()? != 0;
            if seg.update_map {
                for p in seg.tree_probs.iter_mut() {
                    *p = if r.bit()? != 0 { r.f(8)? as u8 } else { 255 };
                }
                seg.temporal_update = r.bit()? != 0;
                for p in seg.pred_probs.iter_mut() {
                    *p = if seg.temporal_update && r.bit()? != 0 {
                        r.f(8)? as u8
                    } else {
                        255
                    };
                }
            }
            if r.bit()? != 0 {
                seg.abs_delta = r.bit()? != 0;
                const BITS: [u32; 4] = [8, 6, 2, 0];
                const SIGNED: [bool; 4] = [true, true, false, false];
                for i in 0..8 {
                    for j in 0..4 {
                        let enabled = r.bit()? != 0;
                        let mut value = 0i16;
                        if enabled {
                            value = r.f(BITS[j])? as i16;
                            if SIGNED[j] && r.bit()? != 0 {
                                value = -value;
                            }
                        }
                        seg.feature_enabled[i][j] = enabled;
                        seg.feature_data[i][j] = value;
                    }
                }
            }
        }
        // tile_info
        let sb_cols = h.width.div_ceil(64);
        let mut min_log2 = 0;
        while (64 << min_log2) < sb_cols {
            min_log2 += 1;
        }
        let mut max_log2 = 1;
        while (sb_cols >> max_log2) >= 4 {
            max_log2 += 1;
        }
        max_log2 -= 1;
        h.log2_tile_cols = min_log2;
        while h.log2_tile_cols < max_log2 && r.bit()? != 0 {
            h.log2_tile_cols += 1;
        }
        h.log2_tile_rows = r.bit()?;
        if h.log2_tile_rows != 0 {
            h.log2_tile_rows += r.bit()?;
        }
        h.header_size = r.f(16)? as usize;
        if h.header_size == 0 {
            return Err(bad("compressed header size 0"));
        }
        h.uncompressed_size = r.pos.div_ceil(8);
        let comp = data
            .get(h.uncompressed_size..h.uncompressed_size + h.header_size)
            .ok_or_else(|| bad("compressed header overruns frame"))?;
        if !h.is_intra() {
            // setup_compound_reference_mode (libvpx vp9_setup_compound_reference_mode).
            let s = h.sign_bias;
            if s[1] == s[2] {
                (h.comp_fixed_ref, h.comp_var_ref) = (3, [1, 2]);
            } else if s[1] == s[3] {
                (h.comp_fixed_ref, h.comp_var_ref) = (2, [1, 3]);
            } else {
                (h.comp_fixed_ref, h.comp_var_ref) = (1, [2, 3]);
            }
        }
        self.fc = self.contexts[h.frame_context_idx];
        self.compressed_header(&mut h, comp)?;
        Ok(h)
    }

    fn compressed_header(&mut self, h: &mut FrameHeader, data: &[u8]) -> Result<()> {
        let mut d = BoolDecoder::new(data)?;
        let fc = &mut self.fc;
        h.tx_mode = if h.lossless {
            0
        } else {
            let mut m = d.literal(2) as u8;
            if m == 3 {
                m += d.literal(1) as u8;
            }
            m
        };
        if h.tx_mode == TX_MODE_SELECT {
            for p in fc.tx8.iter_mut().flatten() {
                d.diff_update(p);
            }
            for p in fc.tx16.iter_mut().flatten() {
                d.diff_update(p);
            }
            for p in fc.tx32.iter_mut().flatten() {
                d.diff_update(p);
            }
        }
        let max_tx = [0, 1, 2, 3, 3][h.tx_mode as usize];
        for tx in 0..=max_tx {
            if d.read(128) != 0 {
                for plane in fc.coef[tx].iter_mut() {
                    for r in plane.iter_mut() {
                        for (band, ctxs) in r.iter_mut().enumerate() {
                            let n = if band == 0 { 3 } else { 6 };
                            for ctx in ctxs.iter_mut().take(n) {
                                for p in ctx.iter_mut() {
                                    d.diff_update(p);
                                }
                            }
                        }
                    }
                }
            }
        }
        for p in fc.skip.iter_mut() {
            d.diff_update(p);
        }
        if !h.is_intra() {
            for p in fc.inter_mode.iter_mut().flatten() {
                d.diff_update(p);
            }
            if h.interp_filter == SWITCHABLE {
                for p in fc.interp_filter.iter_mut().flatten() {
                    d.diff_update(p);
                }
            }
            for p in fc.is_inter.iter_mut() {
                d.diff_update(p);
            }
            let s = h.sign_bias;
            let compound_allowed = s[2] != s[1] || s[3] != s[1];
            h.reference_mode = if compound_allowed && d.read(128) != 0 {
                if d.read(128) != 0 {
                    REFERENCE_MODE_SELECT
                } else {
                    1
                }
            } else {
                0
            };
            if h.reference_mode == REFERENCE_MODE_SELECT {
                for p in fc.comp_inter.iter_mut() {
                    d.diff_update(p);
                }
            }
            if h.reference_mode != 1 {
                for p in fc.single_ref.iter_mut().flatten() {
                    d.diff_update(p);
                }
            }
            if h.reference_mode != 0 {
                for p in fc.comp_ref.iter_mut() {
                    d.diff_update(p);
                }
            }
            for p in fc.y_mode.iter_mut().flatten() {
                d.diff_update(p);
            }
            for p in fc.partition.iter_mut().flatten() {
                d.diff_update(p);
            }
            for p in fc.mv_joints.iter_mut() {
                d.mv_update(p);
            }
            for c in fc.mv.iter_mut() {
                d.mv_update(&mut c.sign);
                for p in c.classes.iter_mut() {
                    d.mv_update(p);
                }
                d.mv_update(&mut c.class0[0]);
                for p in c.bits.iter_mut() {
                    d.mv_update(p);
                }
            }
            for c in fc.mv.iter_mut() {
                for p in c.class0_fp.iter_mut().flatten() {
                    d.mv_update(p);
                }
                for p in c.fp.iter_mut() {
                    d.mv_update(p);
                }
            }
            if h.allow_high_precision_mv {
                for c in fc.mv.iter_mut() {
                    d.mv_update(&mut c.class0_hp);
                    d.mv_update(&mut c.hp);
                }
            }
        }
        Ok(())
    }

    /// Engine probability table for the frame being decoded.
    pub fn entropy_table(&self) -> Vp9EntropyProbs {
        let mut p = Vp9EntropyProbs::zeroed();
        for i in 0..10 {
            for j in 0..10 {
                let k = (i * 10 + j) * 9;
                p.kf_bmode_prob[i][j].copy_from_slice(&t::KF_Y_MODE_PROB[k..k + 8]);
                p.kf_bmode_prob_b[i][j][0] = t::KF_Y_MODE_PROB[k + 8];
            }
            p.kf_uv_mode_prob[i].copy_from_slice(&t::KF_UV_MODE_PROB[i * 9..i * 9 + 8]);
            p.kf_uv_mode_prob_b[i][0] = t::KF_UV_MODE_PROB[i * 9 + 8];
        }
        p.mb_segment_tree_probs = self.seg.tree_probs;
        p.segment_pred_probs = self.seg.pred_probs;
        let fc = &self.fc;
        let a = &mut p.a;
        for i in 0..7 {
            a.inter_mode_prob[i][..3].copy_from_slice(&fc.inter_mode[i]);
        }
        a.intra_inter_prob = fc.is_inter;
        for i in 0..10 {
            a.uv_mode_prob[i].copy_from_slice(&fc.uv_mode[i][..8]);
            a.uv_mode_prob_b[i][0] = fc.uv_mode[i][8];
        }
        a.tx8x8_prob = fc.tx8;
        a.tx16x16_prob = fc.tx16;
        a.tx32x32_prob = fc.tx32;
        for i in 0..4 {
            a.sb_ymode_prob[i].copy_from_slice(&fc.y_mode[i][..8]);
            a.sb_ymode_prob_b[i][0] = fc.y_mode[i][8];
        }
        for i in 0..16 {
            a.partition_prob[0][i][..3].copy_from_slice(&t::KF_PARTITION_PROBS[i * 3..i * 3 + 3]);
            a.partition_prob[1][i][..3].copy_from_slice(&fc.partition[i]);
        }
        a.switchable_interp_prob = fc.interp_filter;
        a.comp_inter_prob = fc.comp_inter;
        a.mbskip_probs = fc.skip;
        let n = &mut a.nmvc;
        n.joints = fc.mv_joints;
        for i in 0..2 {
            let c = &fc.mv[i];
            n.sign[i] = c.sign;
            n.class0[i] = c.class0;
            n.fp[i] = c.fp;
            n.class0_hp[i] = c.class0_hp;
            n.hp[i] = c.hp;
            n.classes[i] = c.classes;
            n.class0_fp[i] = c.class0_fp;
            n.bits[i] = c.bits;
        }
        a.single_ref_prob = fc.single_ref;
        a.comp_ref_prob = fc.comp_ref;
        let dst: [&mut CoefProbs; 4] = [
            &mut a.prob_coeffs,
            &mut a.prob_coeffs8x8,
            &mut a.prob_coeffs16x16,
            &mut a.prob_coeffs32x32,
        ];
        for (tx, out) in dst.into_iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        for l in 0..6 {
                            out[i][j][k][l][..3].copy_from_slice(&fc.coef[tx][i][j][k][l]);
                        }
                    }
                }
            }
        }
        p
    }

    /// Post-decode: backward adaptation from the engine's counts, context
    /// save, reference refresh and per-frame history (spec 8.4.2, 8.10).
    pub fn finish_frame(
        &mut self,
        h: &FrameHeader,
        counts: Option<&Vp9EntropyCounts>,
        surface: usize,
    ) -> Result<()> {
        if !h.error_resilient && !h.frame_parallel {
            let c = counts.ok_or_else(|| bad("adaptation needs the engine counts"))?;
            let pre = self.contexts[h.frame_context_idx];
            adapt_coef(
                &mut self.fc,
                &pre,
                c,
                h.is_intra(),
                self.last_frame_type == KEY_FRAME,
            );
            if !h.is_intra() {
                adapt_mode(&mut self.fc, &pre, c, h);
            }
        }
        if h.refresh_frame_context {
            self.contexts[h.frame_context_idx] = self.fc;
        }
        for i in 0..8 {
            if h.refresh_frame_flags >> i & 1 != 0 {
                self.slots[i] = RefSlot {
                    surface: Some(surface),
                    width: h.width,
                    height: h.height,
                };
            }
        }
        self.last_frame_type = h.frame_type;
        self.last_show_frame = h.show_frame;
        self.last_size = Some((h.width, h.height));
        Ok(())
    }
}

fn get_prob(num: u32, den: u32) -> u8 {
    let p = ((num as u64 * 256 + (den as u64 >> 1)) / den as u64) as u32;
    p.clamp(1, 255) as u8
}

fn merge_probs(pre: u8, ct: [u32; 2], count_sat: u32, max_update: u32) -> u8 {
    let den = ct[0] + ct[1];
    let prob = if den == 0 { 128 } else { get_prob(ct[0], den) };
    let count = den.min(count_sat);
    let factor = max_update * count / count_sat;
    ((pre as u32 * (256 - factor) + prob as u32 * factor + 128) >> 8) as u8
}

const COUNT_TO_UPDATE_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];

fn mode_mv_merge(pre: u8, ct: [u32; 2]) -> u8 {
    let den = ct[0] + ct[1];
    if den == 0 {
        return pre;
    }
    let factor = COUNT_TO_UPDATE_FACTOR[den.min(20) as usize];
    let prob = get_prob(ct[0], den);
    ((pre as u32 * (256 - factor) + prob as u32 * factor + 128) >> 8) as u8
}

/// libvpx vpx_tree_merge_probs over a tree in libvpx index form.
fn tree_merge(tree: &[i8], pre: &[u8], counts: &[u32], out: &mut [u8]) {
    fn walk(i: usize, tree: &[i8], pre: &[u8], counts: &[u32], out: &mut [u8]) -> u32 {
        let side = |n: i8, out: &mut [u8]| {
            if n <= 0 {
                counts[(-n) as usize]
            } else {
                walk(n as usize, tree, pre, counts, out)
            }
        };
        let l = side(tree[i], out);
        let r = side(tree[i + 1], out);
        out[i >> 1] = mode_mv_merge(pre[i >> 1], [l, r]);
        l + r
    }
    walk(0, tree, pre, counts, out);
}

// libvpx trees (vp9_entropymode.c, vp9_entropymv.c).
const INTRA_MODE_TREE: [i8; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
const SWITCHABLE_INTERP_TREE: [i8; 4] = [0, 2, -1, -2];
const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
const MV_CLASS0_TREE: [i8; 2] = [0, -1];
const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

fn adapt_coef(
    fc: &mut FrameContext,
    pre: &FrameContext,
    c: &Vp9EntropyCounts,
    intra: bool,
    last_key: bool,
) {
    let update = if intra {
        112
    } else if last_key {
        128
    } else {
        112
    };
    let counts: [&CoefCounts; 4] = [
        &c.count_coeffs,
        &c.count_coeffs8x8,
        &c.count_coeffs16x16,
        &c.count_coeffs32x32,
    ];
    for tx in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    for l in 0..if k == 0 { 3 } else { 6 } {
                        let n = counts[tx][i][j][k][l];
                        let eob = c.count_eobs[tx][i][j][k][l];
                        let branch = [
                            [n[3], eob.wrapping_sub(n[3])],
                            [n[0], n[1] + n[2]],
                            [n[1], n[2]],
                        ];
                        for m in 0..3 {
                            fc.coef[tx][i][j][k][l][m] =
                                merge_probs(pre.coef[tx][i][j][k][l][m], branch[m], 24, update);
                        }
                    }
                }
            }
        }
    }
}

fn adapt_mode(fc: &mut FrameContext, pre: &FrameContext, c: &Vp9EntropyCounts, h: &FrameHeader) {
    for i in 0..4 {
        fc.is_inter[i] = mode_mv_merge(pre.is_inter[i], c.intra_inter_count[i]);
    }
    for i in 0..5 {
        fc.comp_inter[i] = mode_mv_merge(pre.comp_inter[i], c.comp_inter_count[i]);
        fc.comp_ref[i] = mode_mv_merge(pre.comp_ref[i], c.comp_ref_count[i]);
        for j in 0..2 {
            fc.single_ref[i][j] = mode_mv_merge(pre.single_ref[i][j], c.single_ref_count[i][j]);
        }
    }
    for i in 0..7 {
        // Engine counts per tree node; libvpx counts per mode (INTER_OFFSET
        // order NEAREST, NEAR, ZERO, NEW).
        let n = c.inter_mode_counts[i];
        let per_mode = [n[1][0], n[2][0], n[0][0], n[2][1]];
        tree_merge(
            &INTER_MODE_TREE,
            &pre.inter_mode[i],
            &per_mode,
            &mut fc.inter_mode[i],
        );
    }
    for i in 0..4 {
        tree_merge(
            &INTRA_MODE_TREE,
            &pre.y_mode[i],
            &c.sb_ymode_counts[i],
            &mut fc.y_mode[i],
        );
    }
    for i in 0..10 {
        tree_merge(
            &INTRA_MODE_TREE,
            &pre.uv_mode[i],
            &c.uv_mode_counts[i],
            &mut fc.uv_mode[i],
        );
    }
    for i in 0..16 {
        tree_merge(
            &PARTITION_TREE,
            &pre.partition[i],
            &c.partition_counts[i],
            &mut fc.partition[i],
        );
    }
    if h.interp_filter == SWITCHABLE {
        for i in 0..4 {
            tree_merge(
                &SWITCHABLE_INTERP_TREE,
                &pre.interp_filter[i],
                &c.switchable_interp_counts[i],
                &mut fc.interp_filter[i],
            );
        }
    }
    if h.tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            let t8 = c.tx8x8_count[i];
            fc.tx8[i][0] = mode_mv_merge(pre.tx8[i][0], [t8[0], t8[1]]);
            let t16 = c.tx16x16_count[i];
            let b16 = [[t16[0], t16[1] + t16[2]], [t16[1], t16[2]]];
            for j in 0..2 {
                fc.tx16[i][j] = mode_mv_merge(pre.tx16[i][j], b16[j]);
            }
            let t32 = c.tx32x32_count[i];
            let b32 = [
                [t32[0], t32[1] + t32[2] + t32[3]],
                [t32[1], t32[2] + t32[3]],
                [t32[2], t32[3]],
            ];
            for j in 0..3 {
                fc.tx32[i][j] = mode_mv_merge(pre.tx32[i][j], b32[j]);
            }
        }
    }
    for i in 0..3 {
        fc.skip[i] = mode_mv_merge(pre.skip[i], c.mbskip_count[i]);
    }
    // vp9_adapt_mv_probs
    let m = &c.nmvcount;
    tree_merge(&MV_JOINT_TREE, &pre.mv_joints, &m.joints, &mut fc.mv_joints);
    for i in 0..2 {
        let (p, o) = (&pre.mv[i], &mut fc.mv[i]);
        o.sign = mode_mv_merge(p.sign, m.sign[i]);
        tree_merge(&MV_CLASS_TREE, &p.classes, &m.classes[i], &mut o.classes);
        tree_merge(&MV_CLASS0_TREE, &p.class0, &m.class0[i], &mut o.class0);
        for j in 0..10 {
            o.bits[j] = mode_mv_merge(p.bits[j], m.bits[i][j]);
        }
        for j in 0..2 {
            tree_merge(
                &MV_FP_TREE,
                &p.class0_fp[j],
                &m.class0_fp[i][j],
                &mut o.class0_fp[j],
            );
        }
        tree_merge(&MV_FP_TREE, &p.fp, &m.fp[i], &mut o.fp);
        if h.allow_high_precision_mv {
            o.class0_hp = mode_mv_merge(p.class0_hp, m.class0_hp[i]);
            o.hp = mode_mv_merge(p.hp, m.hp[i]);
        }
    }
}

/// Tile (width, height) in SB64 units, raster order, as the engine reads them
/// at descriptor offset 0x200 (u16 pairs; layout observed in the stock trace).
pub fn tile_sizes(h: &FrameHeader) -> Vec<(u16, u16)> {
    let (sb_cols, sb_rows) = (h.width.div_ceil(64), h.height.div_ceil(64));
    let start = |i: u32, n: u32, log2: u32| (i * n) >> log2;
    let mut out = Vec::new();
    for row in 0..1u32 << h.log2_tile_rows {
        let hgt = start(row + 1, sb_rows, h.log2_tile_rows) - start(row, sb_rows, h.log2_tile_rows);
        for col in 0..1u32 << h.log2_tile_cols {
            let w =
                start(col + 1, sb_cols, h.log2_tile_cols) - start(col, sb_cols, h.log2_tile_cols);
            out.push((w as u16, hgt as u16));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superframe_index_splits() {
        // two frames of 3 and 2 bytes, 1-byte sizes: marker 0b11000_001
        let data = [1, 2, 3, 4, 5, 0xc1, 3, 2, 0xc1];
        let f = split_superframe(&data).unwrap();
        assert_eq!(f, vec![&[1u8, 2, 3][..], &[4, 5][..]]);
        assert_eq!(split_superframe(&[7, 8]).unwrap(), vec![&[7u8, 8][..]]);
    }

    #[test]
    fn inv_remap_matches_libvpx_examples() {
        // delta 0 maps through inv_map_table[0] = 7; m = 127 -> 1 + inv_recenter(7, 127) = 1 + 123
        assert_eq!(inv_remap_prob(0, 128), 124);
        // large m takes the mirrored branch
        assert_eq!(
            inv_remap_prob(0, 250),
            255 - inv_recenter_nonneg(7, 5) as u8
        );
    }

    #[test]
    fn merge_probs_matches_libvpx_formula() {
        assert_eq!(mode_mv_merge(100, [0, 0]), 100);
        // den 20 -> factor 128, prob 128: (100*128 + 128*128 + 128) >> 8 = 114
        assert_eq!(mode_mv_merge(100, [10, 10]), 114);
        assert_eq!(merge_probs(100, [0, 0], 24, 112), 100);
    }

    #[test]
    fn default_context_matches_spec_samples() {
        let c = FrameContext::default();
        assert_eq!(c.coef[0][0][0][0][0], [195, 29, 183]);
        assert_eq!(c.tx32[0], [3, 136, 37]);
        assert_eq!(c.tx8, [[100], [66]]);
        assert_eq!(c.mv[0].classes[0], 224);
        assert_eq!(c.mv[1].class0[0], 208);
        assert_eq!(c.partition[15], [10, 7, 6]);
    }

    #[test]
    fn tile_sizes_split_like_the_spec() {
        let h = FrameHeader {
            width: 1920,
            height: 1080,
            log2_tile_cols: 2,
            ..Default::default()
        };
        assert_eq!(tile_sizes(&h), vec![(7, 17), (8, 17), (7, 17), (8, 17)]);
    }
}
