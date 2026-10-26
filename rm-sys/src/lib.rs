// SPDX-License-Identifier: MIT
//! Raw bindings of the interfaces the RM backend talks to, no logic:
//! - `abi`: RM ioctl wire layouts (open-gpu-kernel-modules nvos.h, nv-ioctl.h, class/cl0080.h,
//!   ctrl0000gpu.h, ctrl0080gpu.h);
//! - `nvdec_drv`: NVDEC firmware interface structs (`nvdec_drv.h`, vendored in `rm/vendor/nvidia/`).
//!
//! Both are hand-written mirrors checked against the C headers; generating them with `bindgen` from the
//! pinned headers replaces them.
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::allow_attributes_without_reason,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    clippy::undocumented_unsafe_blocks,
    missing_debug_implementations,
    missing_docs,
    unused_qualifications,
    reason = "pre-standard code; cleanup tracked"
)]
pub mod abi;
pub mod nvdec_drv;
