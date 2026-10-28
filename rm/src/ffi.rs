// SPDX-License-Identifier: MIT
//! The panic guard of every C entry point of this library (RUST.md §4): the
//! driver-wide guard of `nv-ffi`, reporting through this library's
//! diagnostics sink ([`crate::diag`]). Stateless; callable from any thread.
pub(crate) use nv_ffi::{FailFlag, Failable};

/// Panic reports go to this library's diagnostics sink.
struct Diag;
impl nv_ffi::PanicLog for Diag {
    fn write(message: std::fmt::Arguments<'_>) {
        crate::diag::write(message);
    }
}

/// [`nv_ffi::guard`] with this library's diagnostics sink.
pub(crate) fn guard<R>(
    entry: &'static str,
    object: Option<&dyn Failable>,
    failure: impl FnOnce() -> R,
    body: impl FnOnce() -> R,
) -> R {
    nv_ffi::guard::<Diag, R>(entry, object, failure, body)
}

/// [`nv_ffi::guard_or_abort`] with this library's diagnostics sink.
pub(crate) fn guard_or_abort<R>(entry: &'static str, body: impl FnOnce() -> R) -> R {
    nv_ffi::guard_or_abort::<Diag, R>(entry, body)
}
