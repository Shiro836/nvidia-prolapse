// SPDX-License-Identifier: MIT
//! Real native RM backend. Direct libc/ioctl calls; no NVIDIA userspace DSO,
//! glibc shim, CUDA, Vulkan loader or existing HAL is loaded by this library.
//! This is the backend foundation; a full Vulkan ICD is not implemented yet.
#![deny(unsafe_op_in_unsafe_fn)]
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::allow_attributes,
    clippy::allow_attributes_without_reason,
    clippy::as_conversions,
    clippy::bool_to_int_with_if,
    clippy::borrow_as_ptr,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::chunks_exact_to_as_chunks,
    clippy::collapsible_if,
    clippy::doc_markdown,
    clippy::explicit_iter_loop,
    clippy::items_after_statements,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::many_single_char_names,
    clippy::map_err_ignore,
    clippy::map_unwrap_or,
    clippy::mem_forget,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_continue,
    clippy::needless_pass_by_value,
    clippy::print_stderr,
    clippy::ptr_as_ptr,
    clippy::redundant_closure_for_method_calls,
    clippy::ref_as_ptr,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::struct_excessive_bools,
    clippy::struct_field_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::unwrap_used,
    clippy::used_underscore_binding,
    clippy::wildcard_enum_match_arm,
    let_underscore_drop,
    missing_debug_implementations,
    missing_docs,
    unreachable_pub,
    unused_qualifications,
    reason = "pre-standard code; cleanup tracked"
)]
#![cfg_attr(
    test,
    expect(
        clippy::assert_is_empty,
        clippy::multiple_unsafe_ops_per_block,
        clippy::undocumented_unsafe_blocks,
        reason = "pre-standard code; cleanup tracked"
    )
)]
pub use nv_rm_sys::abi;
pub mod backend;
pub mod capabilities;
mod capture;
pub mod compat;
pub mod compiler;
mod compute;
mod cpu_map_cache;
pub(crate) mod diag;
#[cfg(feature = "diagnostics")]
pub mod diagnostics;
pub(crate) mod events;
mod external;
mod fences;
mod ffi;
pub mod gpu;
pub mod mesa;
pub mod nvdec;
pub use nv_rm_sys::nvdec_drv;
pub use nv_video::nvdec_hevc;
#[cfg(feature = "diagnostics")]
pub mod perf_diagnostic;
pub mod platform;
mod reactor;
use nv_rm_core::reclaim;
pub mod rm;
mod slots;
mod sync_memory;
mod syncobj;
pub mod video;
pub use nv_video::nvdec_vp9 as vp9;
mod waiters;
mod watcher;

/// Version of the narrow diagnostic C ABI below, not a Vulkan ICD version.
/// No pointers, no state; callable from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn nv_native_backend_abi_version() -> u32 {
    ffi::guard_or_abort("nv_native_backend_abi_version", || 1)
}

/// Execute a real backend diagnostic: 0 = mapped VRAM, 1 = hardware copy,
/// 2 = source-built sm_120 compute shader. Returns verified word count, or -1
/// with a named error on stderr. Objects are destroyed before returning.
/// This narrow ABI retains the actual backend in the Android shared library.
/// No pointers; every object it creates is owned and released inside the call;
/// callable from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn nv_native_backend_check(mode: u32) -> i32 {
    ffi::guard(
        "nv_native_backend_check",
        None,
        || -1,
        || backend_check(mode),
    )
}
fn backend_check(mode: u32) -> i32 {
    fn check(mode: u32) -> rm::Result<usize> {
        if mode > 2 {
            return Err(rm::Error::Contract(format!(
                "unsupported diagnostic mode {mode}; expected 0, 1, or 2"
            )));
        }
        let client = rm::Client::open()?;
        let card = client
            .cards()?
            .into_iter()
            .next()
            .ok_or_else(|| rm::Error::Contract("no NVIDIA GPU reported by RM".into()))?;
        let device = client.device(card)?;
        match mode {
            0 => device.allocate_vram(65536)?.map()?.check_pattern(),
            1 => gpu::copy_check(&device),
            2 => gpu::compute_check(&device),
            _ => unreachable!(),
        }
    }
    match check(mode) {
        Ok(words) => words as i32,
        Err(error) => {
            diag::diag!("nv_native_backend_check: {error}");
            -1
        }
    }
}
