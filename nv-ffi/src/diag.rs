// SPDX-License-Identifier: MIT
//! The diagnostics sink of the driver's libraries: error paths only.
//!
//! On Android, a graphics driver's stderr is not connected to anything the
//! platform keeps (zygote children inherit /dev/null), so Android builds write
//! each line to the platform log with liblog's public NDK entry point
//! `__android_log_write(int prio, const char* tag, const char* text)`
//! (NDK `<android/log.h>`; `ANDROID_LOG_ERROR` = 6), and also to stderr: bionic
//! processes without logd (test harnesses, rented hosts) would otherwise drop
//! every diagnostic silently (liblog's write fails ENOTCONN). GNU builds write
//! one line to stderr only. Formatting allocates; hot paths never call this.
//! Stateless; callable from any thread.
use std::{ffi::CStr, fmt};

#[cfg(target_os = "android")]
#[link(name = "log")]
unsafe extern "C" {
    fn __android_log_write(
        priority: std::ffi::c_int,
        tag: *const std::ffi::c_char,
        text: *const std::ffi::c_char,
    ) -> std::ffi::c_int;
}

/// Write one diagnostic line (no trailing newline needed) under the platform
/// log tag `tag`.
#[cfg_attr(
    not(target_os = "android"),
    expect(unused_variables, reason = "the tag only exists in the Android log")
)]
pub fn write(tag: &CStr, message: fmt::Arguments<'_>) {
    #[cfg(target_os = "android")]
    {
        // Interior NULs would truncate the C string; replace them instead.
        let text = message.to_string().replace('\0', "\u{fffd}");
        if let Ok(text) = std::ffi::CString::new(text) {
            const ANDROID_LOG_ERROR: std::ffi::c_int = 6;
            // SAFETY: both pointers are valid NUL-terminated strings for the
            // duration of the call; liblog copies them.
            unsafe { __android_log_write(ANDROID_LOG_ERROR, tag.as_ptr(), text.as_ptr()) };
        }
    }
    {
        use std::io::Write;
        // One locked write per line. A closed or full stderr has nowhere to
        // report to: the line is dropped, deliberately and only here.
        if let Err(error) = writeln!(std::io::stderr().lock(), "{message}") {
            drop(error);
        }
    }
}
