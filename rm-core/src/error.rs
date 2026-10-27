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
    /// A synchronization operation failed (payloads, the semaphore slot pool, the foreign
    /// watcher, CPU waiters).
    #[error(transparent)]
    Sync(#[from] SyncError),
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

impl Error {
    /// The device or channel is unusable and the object must be destroyed: what the C ABI
    /// reports as `VK_ERROR_DEVICE_LOST`.
    pub fn is_device_lost(&self) -> bool {
        match self {
            Self::DeviceLost(_) => true,
            Self::Sync(error) => error.is_device_lost(),
            Self::Io(..)
            | Self::Rm { .. }
            | Self::Contract(_)
            | Self::Wait(_)
            | Self::BarPressure { .. } => false,
        }
    }
}

/// Why a synchronization operation failed: the subsystem of payloads (binary, timeline,
/// imported `sync_file`, cross-process shared), their semaphore slot pool, the foreign
/// watcher thread and the CPU waiters.
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    /// A thread panicked while holding the named object's lock (the FFI guard marked the
    /// device lost); state behind a poisoned lock is never used again (RUST.md §8).
    #[error("{object} lock poisoned: a thread panicked while holding it")]
    Poisoned {
        /// The object whose lock is poisoned.
        object: &'static str,
    },
    /// The payload or watcher belongs to a device that was closed or lost.
    #[error("{0}: the device was closed or lost")]
    Closed(&'static str),
    /// An imported `sync_file` reported a producer error (a negative status, linux
    /// `sync_file.h` `sync_file_info.status`).
    #[error("imported sync_file producer error {status}")]
    ProducerError {
        /// The status the kernel reported.
        status: i32,
    },
    /// The producer of a cross-process shared payload marked it failed.
    #[error("shared synchronization producer failed")]
    SharedProducerFailed,
    /// The operation does not apply to this payload.
    #[error("{operation}: {reason}")]
    Invalid {
        /// The operation.
        operation: &'static str,
        /// Why it does not apply.
        reason: &'static str,
    },
    /// A binary payload's generation counter reached its limit.
    #[error("{operation}: binary payload generation exhausted")]
    GenerationExhausted {
        /// The operation that needed a new generation.
        operation: &'static str,
    },
}

impl SyncError {
    /// Whether this failure makes the device unusable (`Error::is_device_lost`).
    pub fn is_device_lost(&self) -> bool {
        match self {
            Self::Poisoned { .. }
            | Self::Closed(_)
            | Self::ProducerError { .. }
            | Self::SharedProducerFailed => true,
            Self::Invalid { .. } | Self::GenerationExhausted { .. } => false,
        }
    }
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
