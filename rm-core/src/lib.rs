// SPDX-License-Identifier: MIT
//! Syscall-free core of the RM backend, shared by the boundary crate `rm` and the logic crates:
//! the backend's error type, push-buffer method encoding and the deferred-reclamation policy.
#![forbid(unsafe_code)]
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    missing_docs,
    reason = "pre-standard code; cleanup tracked"
)]
#![cfg_attr(
    test,
    expect(clippy::assert_is_empty, reason = "pre-standard code; cleanup tracked")
)]
pub mod error;
pub mod push;
pub mod reclaim;

pub use error::{Error, Result};
