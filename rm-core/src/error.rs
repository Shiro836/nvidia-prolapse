// SPDX-License-Identifier: MIT
//! Error type of the RM backend: the failure of an ioctl, an RM status, a broken contract,
//! a lost device or a bounded wait that ran out, with the operation that failed.
use std::{io, time::Duration};

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
    /// A bounded wait on the GPU, the kernel or another process ran out.
    #[error(transparent)]
    Wait(#[from] WaitError),
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

/// A wait that RUST.md §7 bounds, and why it ended without the awaited state.
#[derive(Debug, thiserror::Error)]
pub enum WaitError {
    /// The awaited GPU work or event did not complete within `limit`.
    #[error("{operation}: {object} did not complete within {limit:?} (waited {elapsed:?})")]
    Timeout {
        /// The wait (the operation that issued the work).
        operation: &'static str,
        /// The awaited object and the state it was left in (channel, value, `GP_GET`).
        object: String,
        /// The bound.
        limit: Duration,
        /// Time from the start of the wait to the expiry check.
        elapsed: Duration,
    },
}

/// `Result` with the backend's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
