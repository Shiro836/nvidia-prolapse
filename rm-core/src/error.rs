// SPDX-License-Identifier: MIT
//! Error type of the RM backend: the failure of an ioctl, an RM status, a broken contract
//! or a lost device, with the operation that failed.
use std::io;

/// Why an RM backend operation failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A syscall or ioctl named by the first field failed.
    #[error("{0}: {1}")]
    Io(&'static str, #[source] io::Error),
    /// RM returned a non-zero `NV_STATUS` (open-gpu-kernel-modules `nvstatuscodes.h`).
    #[error("{operation}: RM status 0x{status:08x}")]
    Rm {
        /// The RM call that failed.
        operation: &'static str,
        /// Its raw status.
        status: u32,
    },
    /// The kernel, RM or the caller broke an interface contract.
    #[error("{0}")]
    Contract(String),
    /// The device or channel is unusable; the object must be destroyed.
    #[error("{0}")]
    DeviceLost(String),
    /// A CPU mapping would exceed the BAR aperture measured at device open.
    #[error(
        "BAR mapping budget exhausted: {used}+{requested} exceeds measured aperture {limit}; reduce live CPU maps or enable a larger BAR"
    )]
    BarPressure {
        /// Bytes mapped now.
        used: u64,
        /// Bytes this mapping asks for.
        requested: u64,
        /// The measured aperture.
        limit: u64,
    },
}

/// `Result` with the backend's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
