// SPDX-License-Identifier: MIT
//! Real native RM backend. Direct libc/ioctl calls; no NVIDIA userspace DSO,
//! glibc shim, CUDA, Vulkan loader or existing HAL is loaded by this library.
//! This is the backend foundation; a full Vulkan ICD is not implemented yet.
//!
//! Process model (RUST.md §13). Every descriptor the library opens is
//! close-on-exec: std's `File`/`try_clone` (O_CLOEXEC, F_DUPFD_CLOEXEC),
//! memfd_create(MFD_CLOEXEC), eventfd(EFD_CLOEXEC), DRM PRIME export with
//! DRM_CLOEXEC, and the sync_files nvidia-drm creates with O_CLOEXEC
//! (open-gpu-kernel-modules kernel-open/nvidia-drm/nvidia-drm-os-interface.c,
//! nv_drm_create_sync_file). Nothing runs at load time; state is created by
//! the open entry points and owned by the handles they return. A handle used
//! in a forked child fails as lost and is never torn down there (process.rs).
//! The library supports dlclose once every handle is closed: closing joins
//! the device's watcher thread and releases its mappings and descriptors;
//! libc drops the atfork handler with the library, and the per-thread
//! nv_video_last_error buffer's destructor keeps the library loaded until
//! that thread exits (libc's thread-local destructor accounting).
#![deny(unsafe_op_in_unsafe_fn)]
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::allow_attributes,
    clippy::allow_attributes_without_reason,
    clippy::bool_to_int_with_if,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::mem_forget,
    clippy::needless_continue,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::used_underscore_binding,
    clippy::wildcard_enum_match_arm,
    missing_debug_implementations,
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
pub mod capabilities;
mod capture;
pub(crate) mod compat;
pub(crate) mod compiler;
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
pub(crate) mod mesa;
pub mod nvdec;
pub use nv_rm_sys::nvdec_drv;
pub use nv_video::nvdec_hevc;
#[cfg(feature = "diagnostics")]
pub mod perf_diagnostic;
pub(crate) mod platform;
mod process;
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
        Ok(words) => i32::try_from(words).expect("the checks verify at most 16384 words"),
        Err(error) => {
            diag::diag!("nv_native_backend_check: {error}");
            -1
        }
    }
}
