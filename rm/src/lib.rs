//! Real native RM backend. Direct libc/ioctl calls; no NVIDIA userspace DSO,
//! glibc shim, CUDA, Vulkan loader or existing HAL is loaded by this library.
//! This is the backend foundation; a full Vulkan ICD is not implemented yet.
#![deny(unsafe_op_in_unsafe_fn)]
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
