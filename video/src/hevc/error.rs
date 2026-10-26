// SPDX-License-Identifier: MIT
//! Why the HEVC front end rejected a unit: a malformed stream (`NV_VIDEO_E_BITSTREAM`) or a valid
//! feature this decoder does not implement (`NV_VIDEO_E_UNSUPPORTED`, [`HevcError::is_unsupported`]).
//! Plain data.

/// An HEVC parse, reference-picture or descriptor failure, with the values involved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HevcError {
    /// A bitstream conformance requirement the stream breaks, named.
    #[error("{0}")]
    Malformed(&'static str),
    /// A NAL unit shorter than its two-byte header (7.3.1.2).
    #[error("HEVC {what} NAL of {len} bytes has no NAL unit header")]
    NoNalHeader {
        /// The NAL kind being parsed.
        what: &'static str,
        /// Its length in bytes.
        len: usize,
    },
    /// A slice names a PPS that was not received.
    #[error("slice references missing PPS {0}")]
    MissingPps(u32),
    /// A PPS names an SPS that was not received.
    #[error("PPS {pps} references missing SPS {sps}")]
    MissingSps {
        /// `pps_pic_parameter_set_id`.
        pps: u32,
        /// `pps_seq_parameter_set_id`.
        sps: u32,
    },
    /// `CtbLog2SizeY` outside 4..=6 (7.4.3.2.1).
    #[error("CtbLog2SizeY {0}")]
    CtbSize(u32),
    /// A reference picture of the RPS that the DPB does not hold (8.3.2).
    #[error("reference picture POC {0} missing from the DPB")]
    MissingReference(i32),
    /// A POC distance beyond the engine's 16-bit field.
    #[error("POC distance {0} exceeds 16 bits")]
    PocDistance(i32),
    /// A value beyond its descriptor field.
    #[error("{field} = {value} does not fit its {width}-bit descriptor field")]
    FieldWidth {
        /// The descriptor field.
        field: &'static str,
        /// The value.
        value: u32,
        /// The field width in bits.
        width: u32,
    },
    /// A valid HEVC feature this decoder does not implement, named.
    #[error("{0}")]
    Unsupported(&'static str),
    /// A profile other than Main, Main 10, Main Still Picture.
    #[error("HEVC profile_idc {0} (only Main, Main 10, Main Still Picture)")]
    Profile(u32),
    /// `chroma_format_idc` other than 1 (4:2:0).
    #[error("HEVC chroma_format_idc {0} (only 4:2:0)")]
    ChromaFormat(u32),
    /// Luma/chroma bit depths other than 8/8 and 10/10.
    #[error("HEVC bit depth {luma}/{chroma} (8/8 or 10/10)")]
    BitDepth {
        /// `BitDepthY`.
        luma: u32,
        /// `BitDepthC`.
        chroma: u32,
    },
    /// A reserved VCL NAL unit type (Table 7-1).
    #[error("HEVC reserved VCL NAL unit type {0}")]
    ReservedNalType(u8),
    /// A picture whose bit depth differs from the session's.
    #[error(
        "HEVC {picture}-bit picture in a {session}-bit session (create the session with the stream's bit depth)"
    )]
    SessionBitDepth {
        /// The picture's `BitDepthY`.
        picture: u32,
        /// The session's.
        session: u32,
    },
}

impl HevcError {
    /// True for valid streams this decoder does not implement.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::Unsupported(_)
                | Self::Profile(_)
                | Self::ChromaFormat(_)
                | Self::BitDepth { .. }
                | Self::ReservedNalType(_)
                | Self::SessionBitDepth { .. }
        )
    }
}
