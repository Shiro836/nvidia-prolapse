// SPDX-License-Identifier: MIT
//! Why the H.264 front end rejected a unit: a malformed stream ([`H264Error::is_unsupported`] is
//! false, `NV_VIDEO_E_BITSTREAM`) or a valid feature this decoder does not implement
//! (`NV_VIDEO_E_UNSUPPORTED`). Plain data.

/// An H.264 parse or reference-marking failure, with the syntax element and values involved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum H264Error {
    /// `frame_crop_*_offset` larger than the picture (7.4.2.1.1).
    #[error("frame cropping larger than the picture")]
    CropTooLarge,
    /// A PPS names an SPS that was not received.
    #[error("PPS {pps} references missing SPS {sps}")]
    MissingSps {
        /// `pic_parameter_set_id`.
        pps: u32,
        /// `seq_parameter_set_id`.
        sps: u32,
    },
    /// A slice names a PPS (or its SPS) that was not received.
    #[error("slice references missing PPS {0} (or its SPS)")]
    MissingPps(u32),
    /// `modification_of_pic_nums_idc` outside 0..=3 (7.4.3.1).
    #[error("modification_of_pic_nums_idc = {0}")]
    ModificationIdc(u32),
    /// `memory_management_control_operation` outside 0..=6 (7.4.3.3).
    #[error("memory_management_control_operation = {0}")]
    MmcoValue(u32),
    /// More memory management operations than the DPB can need.
    #[error("too many memory_management_control_operations")]
    TooManyMmcos,
    /// A reference frame beyond the 16-frame DPB.
    #[error("H.264 DPB overflow (more than 16 reference frames)")]
    DpbOverflow,
    /// The sliding window (8.2.5.3) found no short-term reference to drop.
    #[error("sliding window with no short-term reference to remove")]
    SlidingWindowEmpty,
    /// A frame_num gap without `gaps_in_frame_num_value_allowed_flag` (7.4.3).
    #[error(
        "frame_num gap {prev} -> {frame_num} without gaps_in_frame_num_value_allowed_flag (lost pictures)"
    )]
    FrameNumGap {
        /// PrevRefFrameNum.
        prev: u32,
        /// The slice's frame_num.
        frame_num: u32,
    },
    /// The first picture is not an IDR picture.
    #[error("H.264 stream does not start with an IDR picture")]
    NoIdr,
    /// The session's target surface is still a reference.
    #[error("target surface {0} is still a reference")]
    TargetReferenced(u32),
    /// MMCO 1 or 3 on a picture number no short-term reference has (8.2.5.4.1, 8.2.5.4.3).
    #[error("MMCO{mmco} on missing PicNum {cur_pic_num} - ({difference_minus1} + 1)")]
    MissingPicNum {
        /// The operation (1 or 3).
        mmco: u8,
        /// CurrPicNum.
        cur_pic_num: i32,
        /// `difference_of_pic_nums_minus1`.
        difference_minus1: u32,
    },
    /// MMCO 2 on a long-term picture number no reference has (8.2.5.4.2).
    #[error("MMCO2 on missing LongTermPicNum {0}")]
    MissingLongTermPicNum(u32),
    /// MMCO 3 or 6 with a LongTermFrameIdx above MaxLongTermFrameIdx (8.2.5.4.3, 8.2.5.4.6).
    #[error("MMCO{mmco} LongTermFrameIdx {idx} above MaxLongTermFrameIdx")]
    LongTermIdxAboveMax {
        /// The operation (3 or 6).
        mmco: u8,
        /// `long_term_frame_idx`.
        idx: u32,
    },
    /// Reference marking leaves more frames than `max_num_ref_frames`.
    #[error("reference marking leaves {frames} frames with max_num_ref_frames {max}")]
    TooManyReferences {
        /// Frames after marking.
        frames: u32,
        /// `max_num_ref_frames`.
        max: u32,
    },
    /// A valid H.264 feature this decoder does not implement (`NV_VIDEO_E_UNSUPPORTED`).
    #[error("{0}")]
    Unsupported(&'static str),
    /// A picture or session above the 4096x4096 this decoder supports.
    #[error("H.264 {what} {width}x{height} above 4096x4096")]
    TooLarge {
        /// "picture" or "session".
        what: &'static str,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// chroma_format_idc other than 1 (4:2:0).
    #[error(
        "H.264 chroma_format_idc {0} (only 4:2:0: the session's surfaces and copy-out are NV12/P010 \
         4:2:0, so 4:2:2/4:4:4 needs a new surface layout and copy path)"
    )]
    ChromaFormat(u32),
}

impl H264Error {
    /// True for valid streams this decoder does not implement.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::Unsupported(_) | Self::TooLarge { .. } | Self::ChromaFormat(_)
        )
    }
}
