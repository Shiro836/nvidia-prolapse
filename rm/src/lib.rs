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
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::explicit_iter_loop,
    clippy::if_not_else,
    clippy::items_after_statements,
    clippy::let_underscore_must_use,
    clippy::let_underscore_untyped,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::map_err_ignore,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::mem_forget,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::multiple_unsafe_ops_per_block,
    clippy::needless_continue,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::print_stderr,
    clippy::ptr_as_ptr,
    clippy::ptr_cast_constness,
    clippy::redundant_closure_for_method_calls,
    clippy::ref_as_ptr,
    clippy::same_functions_in_if_condition,
    clippy::semicolon_if_nothing_returned,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::struct_excessive_bools,
    clippy::struct_field_names,
    clippy::too_many_lines,
    clippy::undocumented_unsafe_blocks,
    clippy::unnested_or_patterns,
    clippy::unreadable_literal,
    clippy::unused_self,
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
    not(test),
    expect(
        clippy::wildcard_imports,
        reason = "pre-standard code; cleanup tracked"
    )
)]
#![cfg_attr(
    test,
    expect(
        clippy::assert_is_empty,
        clippy::format_collect,
        clippy::format_push_string,
        clippy::ignore_without_reason,
        clippy::manual_is_multiple_of,
        clippy::stable_sort_primitive,
        reason = "pre-standard code; cleanup tracked"
    )
)]
#[cfg(all(not(target_pointer_width = "64"), feature = "prototype-vulkan"))]
compile_error!(
    "The standalone experimental Vulkan frontend is LP64; use the Mesa RM bridge on x86"
);
pub mod abi;
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
pub mod gpu;
pub mod mesa;
pub mod nvdec;
pub mod nvdec_drv;
pub mod nvdec_hevc;
#[cfg(feature = "diagnostics")]
pub mod perf_diagnostic;
pub mod platform;
mod reactor;
mod reclaim;
pub mod rm;
mod slots;
mod sync_memory;
mod syncobj;
// Fixed-shape probability/count tables are indexed in parallel like the spec.
pub mod video;
#[allow(clippy::needless_range_loop)]
pub mod vp9;
mod vp9_tables;
#[cfg(feature = "prototype-vulkan")]
pub mod vulkan;
mod waiters;
mod watcher;

/// Version of the narrow diagnostic C ABI below, not a Vulkan ICD version.
#[unsafe(no_mangle)]
pub extern "C" fn nv_native_backend_abi_version() -> u32 {
    1
}

/// Execute a real backend diagnostic: 0 = mapped VRAM, 1 = hardware copy,
/// 2 = source-built sm_120 compute shader. Returns verified word count, or -1
/// with a named error on stderr. Objects are destroyed before returning.
/// This narrow ABI retains the actual backend in the Android shared library.
#[unsafe(no_mangle)]
pub extern "C" fn nv_native_backend_check(mode: u32) -> i32 {
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
