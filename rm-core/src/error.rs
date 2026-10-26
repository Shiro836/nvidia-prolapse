// SPDX-License-Identifier: MIT
//! Error type of the RM backend: the failure of an ioctl, an RM status, a broken contract
//! or a lost device, with the operation that failed.
use std::io;

#[derive(Debug)]
pub enum Error {
    Io(&'static str, io::Error),
    Rm {
        operation: &'static str,
        status: u32,
    },
    Contract(String),
    DeviceLost(String),
    BarPressure {
        used: u64,
        requested: u64,
        limit: u64,
    },
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(operation, error) => write!(f, "{operation}: {error}"),
            Self::Rm { operation, status } => write!(f, "{operation}: RM status 0x{status:08x}"),
            Self::Contract(message) | Self::DeviceLost(message) => f.write_str(message),
            Self::BarPressure {
                used,
                requested,
                limit,
            } => write!(
                f,
                "BAR mapping budget exhausted: {used}+{requested} exceeds measured aperture {limit}; reduce live CPU maps or enable a larger BAR"
            ),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
