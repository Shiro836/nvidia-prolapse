// SPDX-License-Identifier: MIT
//! The crate's API error, [`VideoError`], and the conversions from each subsystem's own error into
//! it: the shared bit reader (`bits::Error`), AV1 (`av1::obu::Av1Error`), the VP9 front end
//! (`nvdec_vp9::Vp9Error`), H.264 (`h264::H264Error`), HEVC (`hevc::HevcError`) and values that do not fit their field ([`OutOfRange`]). The C ABI
//! reports a [`VideoError`] as one `NV_VIDEO_E_*` code ([`VideoError::code`]) plus its message.
//!
//! Plain data; no state.
use std::borrow::Cow;
use std::fmt::Display;

/// `nv_video` error classes (`NvVideoError`). Every variant carries the reason that
/// `nv_video_last_error()` reports.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// `NV_VIDEO_E_INVALID`: the caller broke the API contract.
    #[error("{0}")]
    Invalid(String),
    /// `NV_VIDEO_E_UNSUPPORTED`: valid input this library or this GPU does not decode.
    #[error("{0}")]
    Unsupported(String),
    /// `NV_VIDEO_E_BITSTREAM`: the bitstream is malformed.
    #[error("{0}")]
    Bitstream(String),
    /// `NV_VIDEO_E_NO_MEMORY`.
    #[error("{0}")]
    NoMemory(String),
    /// `NV_VIDEO_E_DEVICE_LOST`: the session must be destroyed.
    #[error("{0}")]
    DeviceLost(String),
    /// `NV_VIDEO_E_AGAIN`: no surface free for this unit now.
    #[error("{0}")]
    Again(String),
    /// A random access point (key frame / IDR) that this session cannot decode at its
    /// size: the caller drains the session, creates one of at least `width` x `height`
    /// and submits the same unit to it (`NV_VIDEO_E_RESIZE`).
    #[error("{reason}")]
    Resize {
        /// Coded width the new session needs.
        width: u32,
        /// Coded height the new session needs.
        height: u32,
        /// Message for `nv_video_last_error()`.
        reason: String,
    },
    /// The H.264/HEVC bit reader ran out of data or read a value beyond its syntax limit
    /// (`NV_VIDEO_E_BITSTREAM`).
    #[error(transparent)]
    Bits(#[from] crate::bits::Error),
    /// A bitstream-derived value does not fit the field it goes into (`NV_VIDEO_E_BITSTREAM`).
    #[error(transparent)]
    Range(#[from] OutOfRange),
    /// AV1 parse failure: `NV_VIDEO_E_UNSUPPORTED` for an unimplemented feature, else
    /// `NV_VIDEO_E_BITSTREAM`.
    #[error("AV1: {0}")]
    Av1(#[from] crate::av1::obu::Av1Error),
    /// VP9 parse failure (`NV_VIDEO_E_BITSTREAM`).
    #[error(transparent)]
    Vp9(#[from] crate::nvdec_vp9::Vp9Error),
    /// H.264 parse or reference-marking failure: `NV_VIDEO_E_UNSUPPORTED` for an unimplemented
    /// feature, else `NV_VIDEO_E_BITSTREAM`.
    #[error(transparent)]
    H264(#[from] crate::h264::H264Error),
    /// HEVC parse, reference or descriptor failure: `NV_VIDEO_E_UNSUPPORTED` for an
    /// unimplemented feature, else `NV_VIDEO_E_BITSTREAM`.
    #[error(transparent)]
    Hevc(#[from] crate::hevc::HevcError),
    /// HDR static metadata scan failure (`NV_VIDEO_E_BITSTREAM`).
    #[error(transparent)]
    Hdr(#[from] crate::hdr::HdrError),
}

impl VideoError {
    /// The `NV_VIDEO_E_*` value of `nv_video.h` for this error.
    pub fn code(&self) -> i32 {
        match self {
            Self::Invalid(_) => -1,
            Self::Hevc(e) => {
                if e.is_unsupported() {
                    -2
                } else {
                    -3
                }
            }
            Self::H264(e) => {
                if e.is_unsupported() {
                    -2
                } else {
                    -3
                }
            }
            Self::Unsupported(_)
            | Self::Av1(
                crate::av1::obu::Av1Error::Unsupported(_)
                | crate::av1::obu::Av1Error::SessionFormat { .. },
            ) => -2,
            Self::Vp9(e) => {
                if e.is_unsupported() {
                    -2
                } else {
                    -3
                }
            }
            Self::Bitstream(_)
            | Self::Bits(_)
            | Self::Hdr(_)
            | Self::Range(_)
            | Self::Av1(
                crate::av1::obu::Av1Error::Truncated(_)
                | crate::av1::obu::Av1Error::Invalid(_)
                | crate::av1::obu::Av1Error::NoSequenceHeader
                | crate::av1::obu::Av1Error::Range(_),
            ) => -3,
            Self::NoMemory(_) => -4,
            Self::DeviceLost(_) => -5,
            Self::Again(_) => -6,
            Self::Resize { .. } => -7,
        }
    }
    /// The reason `nv_video_last_error()` reports.
    pub fn message(&self) -> Cow<'_, str> {
        match self {
            Self::Invalid(m)
            | Self::Unsupported(m)
            | Self::Bitstream(m)
            | Self::NoMemory(m)
            | Self::DeviceLost(m)
            | Self::Again(m)
            | Self::Resize { reason: m, .. } => Cow::Borrowed(m),
            Self::Bits(_)
            | Self::Range(_)
            | Self::Av1(_)
            | Self::Vp9(_)
            | Self::H264(_)
            | Self::Hevc(_)
            | Self::Hdr(_) => Cow::Owned(self.to_string()),
        }
    }
}

/// RM/channel failures inside a session are device-level: the session must be
/// destroyed (`NV_VIDEO_E_DEVICE_LOST`).
impl From<nv_rm_core::Error> for VideoError {
    fn from(e: nv_rm_core::Error) -> Self {
        Self::DeviceLost(e.to_string())
    }
}

/// `Result` with the crate's [`VideoError`].
pub type VResult<T> = std::result::Result<T, VideoError>;

/// A value derived from the bitstream (or from the session geometry) that does not fit the
/// integer type of the field it is stored in.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field} = {value} does not fit {target}")]
pub struct OutOfRange {
    /// The syntax element or descriptor field.
    pub field: &'static str,
    /// The value, formatted.
    pub value: String,
    /// The target type.
    pub target: &'static str,
}

/// `T::try_from(value)`, with a failure reported as [`OutOfRange`] for `field`.
pub(crate) fn fit<T, S>(value: S, field: &'static str) -> Result<T, OutOfRange>
where
    S: Copy + Display,
    T: TryFrom<S>,
{
    T::try_from(value).map_err(|_unrepresentable| OutOfRange {
        field,
        value: value.to_string(),
        target: std::any::type_name::<T>(),
    })
}

/// A picture larger than the session's maximum. At a random access point that starts a
/// new reference chain (H.264 IDR, HEVC IRAP, VP9 key frame, AV1 shown key frame) a
/// session of the new size decodes the stream from there on, so the caller re-creates
/// the session ([`VideoError::Resize`]) the way `FFmpeg` re-creates its hardware decoder
/// when the coded size changes (libavcodec/decode.c:1227 `ff_hwaccel_uninit` in
/// `ff_get_format`, reached from h264_slice.c:920, hevc/hevcdec.c:725, vp9.c:254,
/// av1dec.c:664). Anywhere else the references live in this session: refused.
pub fn too_large(
    codec: &str,
    width: u32,
    height: u32,
    max: (u32, u32),
    random_access: bool,
) -> VideoError {
    let what = format!(
        "{codec} picture {width}x{height} larger than the session max {}x{}",
        max.0, max.1
    );
    if random_access {
        VideoError::Resize {
            width,
            height,
            reason: format!("{what}: re-create the session at the new size"),
        }
    } else {
        VideoError::Unsupported(format!(
            "{what} on a picture that is not a random access point (references stay in this \
             session)"
        ))
    }
}

/// `Err(VideoError::Unsupported)` with `m`, for the codec paths' early returns.
///
/// # Errors
/// Always.
pub fn unsupported<T>(m: impl Into<String>) -> VResult<T> {
    Err(VideoError::Unsupported(m.into()))
}
/// `Err(VideoError::Bitstream)` with `m`, for the parsers' early returns.
///
/// # Errors
/// Always.
pub fn bitstream<T>(m: impl Into<String>) -> VResult<T> {
    Err(VideoError::Bitstream(m.into()))
}
