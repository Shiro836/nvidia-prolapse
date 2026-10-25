//! repr(C) mirrors of the NVDEC driver/firmware interface structs from
//! NVIDIA's MIT-licensed `nvdec_drv.h` (vendored verbatim in
//! `vendor/nvidia/nvdec_drv.h`; Mesa main 0cd84d45c600
//! src/nouveau/headers/nvidia/video/nvdec_drv.h, identical to tinygrad's
//! extra/nv_gpu_driver copy). Hand-written, no bindgen: C bitfields are kept as
//! their storage units (`u32`/`u8`) with the bit positions documented, and the
//! sizes/offsets are checked against a gcc compile of the header (table in the
//! tests; the C program printing them is in the test comment).

/// `nvdec_vp9_pic_s` (256 bytes). Bitfield words:
/// - `bits_14`: key_increment[0:6] encryption_mode[6:10] sw_hdr_skip_length[10:24]
///   key_slot_index[24:28] ssm_en[28] enable_histogram[29]
/// - `surface`: tileformat[0:2] gob_height[2:5]
/// - `flags`: keyFrame[0] prevIsKeyFrame[1] resolutionChange[2] errorResilient[3]
///   prevShowFrame[4] intraOnly[5]
/// - `v1_bits`: external_ref_mem_dis[0] bit_depth[1:5] error_recovery_start_pos[5:7]
///   error_external_mv_en[7]
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Vp9PicS {
    pub wrapped_session_key: [u32; 4],
    pub wrapped_content_key: [u32; 4],
    pub initialization_vector: [u32; 4],
    pub stream_len: u32,
    pub enable_encryption: u32,
    pub bits_14: u32,
    pub gptimer_timeout_value: u32,
    pub surface: u8,
    pub reserved1: [u8; 3],
    /// BSD control buffer offset relative to the VP9 filter buffer, 256 B units.
    pub vp9_bsd_ctrl_offset: u32,
    /// [last, golden, altref]: (width, height, stride[2]).
    pub refs: [Vp9Dims; 3],
    pub current: Vp9Dims,
    pub flags: u8,
    pub reserved3: [u8; 3],
    pub ref_frame_sign_bias: [u8; 4],
    pub loop_filter_level: i8,
    pub loop_filter_sharpness: i8,
    pub qp_y_ac: u8,
    pub qp_y_dc: i8,
    pub qp_ch_ac: i8,
    pub qp_ch_dc: i8,
    pub lossless: i8,
    pub transform_mode: i8,
    pub allow_high_precision_mv: i8,
    pub mcomp_filter_type: i8,
    pub comp_pred_mode: i8,
    pub comp_fixed_ref: i8,
    pub comp_var_ref: [i8; 2],
    pub log2_tile_columns: i8,
    pub log2_tile_rows: i8,
    pub segment_enabled: u8,
    pub segment_map_update: u8,
    pub segment_map_temporal_update: u8,
    pub segment_feature_mode: u8,
    pub segment_feature_enable: [[u8; 4]; 8],
    pub segment_feature_data: [[i16; 4]; 8],
    pub mode_ref_lf_enabled: i8,
    pub mb_ref_lf_delta: [i8; 4],
    pub mb_mode_lf_delta: [i8; 2],
    pub reserved5: i8,
    /// v1: filter above offset relative to the filter buffer, 256 B units.
    pub vp9_flt_above_offset: u32,
    pub v1_bits: u32,
    /// nvdec_pass2_otf_ext_s (secure playback only).
    pub ssm: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Vp9Dims {
    pub width: u16,
    pub height: u16,
    pub stride: [u16; 2],
}

/// `nvdec_nmv_context` (69 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct NmvContext {
    pub joints: [u8; 3],
    pub sign: [u8; 2],
    pub class0: [[u8; 1]; 2],
    pub fp: [[u8; 3]; 2],
    pub class0_hp: [u8; 2],
    pub hp: [u8; 2],
    pub classes: [[u8; 10]; 2],
    pub class0_fp: [[[u8; 3]; 2]; 2],
    pub bits: [[u8; 10]; 2],
}

pub type CoefProbs = [[[[[u8; 4]; 6]; 6]; 2]; 2];

/// `nvdec_vp9AdaptiveEntropyProbs_t` (2720 bytes). 9-entry mode trees are
/// split into an 8-byte part and a 1-byte `*_b` tail; 3-node trees are padded
/// to 4.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vp9AdaptiveProbs {
    pub inter_mode_prob: [[u8; 4]; 7],
    pub intra_inter_prob: [u8; 4],
    pub uv_mode_prob: [[u8; 8]; 10],
    pub tx8x8_prob: [[u8; 1]; 2],
    pub tx16x16_prob: [[u8; 2]; 2],
    pub tx32x32_prob: [[u8; 3]; 2],
    pub sb_ymode_prob_b: [[u8; 1]; 4],
    pub sb_ymode_prob: [[u8; 8]; 4],
    /// [0] keyframe/intra-only defaults, [1] the adaptive context.
    pub partition_prob: [[[u8; 4]; 16]; 2],
    pub uv_mode_prob_b: [[u8; 1]; 10],
    pub switchable_interp_prob: [[u8; 2]; 4],
    pub comp_inter_prob: [u8; 5],
    pub mbskip_probs: [u8; 3],
    pub pad1: [u8; 1],
    pub nmvc: NmvContext,
    pub single_ref_prob: [[u8; 2]; 5],
    pub comp_ref_prob: [u8; 5],
    pub pad2: [u8; 17],
    pub prob_coeffs: CoefProbs,
    pub prob_coeffs8x8: CoefProbs,
    pub prob_coeffs16x16: CoefProbs,
    pub prob_coeffs32x32: CoefProbs,
}

/// `nvdec_vp9EntropyProbs_t` (3744 bytes), the VP9_SET_PROB_TAB_BUF target.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vp9EntropyProbs {
    pub kf_bmode_prob: [[[u8; 8]; 10]; 10],
    pub kf_bmode_prob_b: [[[u8; 1]; 10]; 10],
    pub ref_pred_probs: [u8; 3],
    pub mb_segment_tree_probs: [u8; 7],
    pub segment_pred_probs: [u8; 3],
    pub ref_scores: [u8; 4],
    pub prob_comppred: [u8; 2],
    pub pad1: [u8; 9],
    pub kf_uv_mode_prob: [[u8; 8]; 10],
    pub kf_uv_mode_prob_b: [[u8; 1]; 10],
    pub pad2: [u8; 6],
    pub a: Vp9AdaptiveProbs,
}

/// `nvdec_nmv_context_counts` (424 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NmvCounts {
    pub joints: [u32; 4],
    pub sign: [[u32; 2]; 2],
    pub classes: [[u32; 11]; 2],
    pub class0: [[u32; 2]; 2],
    pub bits: [[[u32; 2]; 10]; 2],
    pub class0_fp: [[[u32; 4]; 2]; 2],
    pub fp: [[u32; 4]; 2],
    pub class0_hp: [[u32; 2]; 2],
    pub hp: [[u32; 2]; 2],
}

pub type CoefCounts = [[[[[u32; 4]; 6]; 6]; 2]; 2];

/// `nvdec_vp9EntropyCounts_t` (13264 bytes), the VP9_SET_CTX_COUNTER_BUF
/// target written by the engine. Tree-coded symbols are counted per symbol
/// except `inter_mode_counts`, which holds per-node [bit] counts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vp9EntropyCounts {
    pub inter_mode_counts: [[[u32; 2]; 3]; 7],
    pub sb_ymode_counts: [[u32; 10]; 4],
    pub uv_mode_counts: [[u32; 10]; 10],
    pub partition_counts: [[u32; 4]; 16],
    pub switchable_interp_counts: [[u32; 3]; 4],
    pub intra_inter_count: [[u32; 2]; 4],
    pub comp_inter_count: [[u32; 2]; 5],
    pub single_ref_count: [[[u32; 2]; 2]; 5],
    pub comp_ref_count: [[u32; 2]; 5],
    pub tx32x32_count: [[u32; 4]; 2],
    pub tx16x16_count: [[u32; 3]; 2],
    pub tx8x8_count: [[u32; 2]; 2],
    pub mbskip_count: [[u32; 2]; 3],
    pub nmvcount: NmvCounts,
    /// [plane][ref][band][ctx][ZERO, ONE, TWO+, EOB_MODEL]
    pub count_coeffs: CoefCounts,
    pub count_coeffs8x8: CoefCounts,
    pub count_coeffs16x16: CoefCounts,
    pub count_coeffs32x32: CoefCounts,
    /// [tx][plane][ref][band][ctx]: times the more-coefficients branch was coded.
    pub count_eobs: [[[[[u32; 6]; 6]; 2]; 2]; 4],
}

const _: () = assert!(std::mem::size_of::<Vp9PicS>() == 256);
const _: () = assert!(std::mem::size_of::<NmvContext>() == 69);
const _: () = assert!(std::mem::size_of::<Vp9AdaptiveProbs>() == 2720);
const _: () = assert!(std::mem::size_of::<Vp9EntropyProbs>() == 3744);
const _: () = assert!(std::mem::size_of::<NmvCounts>() == 424);
const _: () = assert!(std::mem::size_of::<Vp9EntropyCounts>() == 13264);

/// Plain-old-data marker for the structs above (all-integer, no padding
/// bytes other than explicit `pad`/`reserved` fields).
///
/// # Safety
/// Implementors must be repr(C) structs of integer arrays without implicit
/// padding, valid for every bit pattern.
pub unsafe trait Pod: Copy + 'static {
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: Pod guarantees no implicit padding and plain integer fields.
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
    fn zeroed() -> Self {
        // SAFETY: all-zero is a valid value of every Pod type.
        unsafe { std::mem::zeroed() }
    }
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        // SAFETY: length checked; every bit pattern is valid; unaligned read.
        Some(unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Self) })
    }
}
// SAFETY: integer-only repr(C) structs; sizes asserted above equal the sum of
// their fields (no implicit padding) in the offset test.
unsafe impl Pod for Vp9PicS {}
unsafe impl Pod for Vp9EntropyProbs {}
unsafe impl Pod for Vp9EntropyCounts {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    /// Offsets printed by gcc 15 from the vendored header (offsetof on
    /// nvdec_vp9_pic_s, nvdec_vp9EntropyProbs_t, nvdec_vp9AdaptiveEntropyProbs_t,
    /// nvdec_vp9EntropyCounts_t), 2026-10-14.
    #[test]
    fn layout_matches_c_header() {
        assert_eq!(offset_of!(Vp9PicS, stream_len), 48);
        assert_eq!(offset_of!(Vp9PicS, gptimer_timeout_value), 60);
        assert_eq!(offset_of!(Vp9PicS, vp9_bsd_ctrl_offset), 68);
        assert_eq!(offset_of!(Vp9PicS, refs), 72);
        assert_eq!(offset_of!(Vp9PicS, current), 96);
        assert_eq!(offset_of!(Vp9PicS, ref_frame_sign_bias), 108);
        assert_eq!(offset_of!(Vp9PicS, loop_filter_level), 112);
        assert_eq!(offset_of!(Vp9PicS, qp_y_ac), 114);
        assert_eq!(offset_of!(Vp9PicS, lossless), 118);
        assert_eq!(offset_of!(Vp9PicS, log2_tile_columns), 126);
        assert_eq!(offset_of!(Vp9PicS, segment_enabled), 128);
        assert_eq!(offset_of!(Vp9PicS, segment_feature_enable), 132);
        assert_eq!(offset_of!(Vp9PicS, segment_feature_data), 164);
        assert_eq!(offset_of!(Vp9PicS, mode_ref_lf_enabled), 228);
        assert_eq!(offset_of!(Vp9PicS, mb_ref_lf_delta), 229);
        assert_eq!(offset_of!(Vp9PicS, mb_mode_lf_delta), 233);
        assert_eq!(offset_of!(Vp9PicS, vp9_flt_above_offset), 236);
        assert_eq!(offset_of!(Vp9PicS, ssm), 244);
        assert_eq!(offset_of!(Vp9EntropyProbs, kf_bmode_prob_b), 800);
        assert_eq!(offset_of!(Vp9EntropyProbs, ref_pred_probs), 900);
        assert_eq!(offset_of!(Vp9EntropyProbs, mb_segment_tree_probs), 903);
        assert_eq!(offset_of!(Vp9EntropyProbs, segment_pred_probs), 910);
        assert_eq!(offset_of!(Vp9EntropyProbs, kf_uv_mode_prob), 928);
        assert_eq!(offset_of!(Vp9EntropyProbs, a), 1024);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, uv_mode_prob), 32);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, tx8x8_prob), 112);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, sb_ymode_prob_b), 124);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, sb_ymode_prob), 128);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, partition_prob), 160);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, uv_mode_prob_b), 288);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, switchable_interp_prob), 298);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, comp_inter_prob), 306);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, mbskip_probs), 311);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, nmvc), 315);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, single_ref_prob), 384);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, comp_ref_prob), 394);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, prob_coeffs), 416);
        assert_eq!(offset_of!(Vp9AdaptiveProbs, prob_coeffs32x32), 2144);
        assert_eq!(offset_of!(Vp9EntropyCounts, sb_ymode_counts), 168);
        assert_eq!(offset_of!(Vp9EntropyCounts, nmvcount), 1320);
        assert_eq!(offset_of!(Vp9EntropyCounts, count_coeffs), 1744);
        assert_eq!(offset_of!(Vp9EntropyCounts, count_eobs), 10960);
    }
}
