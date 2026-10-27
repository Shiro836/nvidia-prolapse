// SPDX-License-Identifier: MIT
//! Backend diagnostics sink.
//!
//! On Android, a graphics driver's stderr is not connected to anything the
//! platform keeps (zygote children inherit /dev/null), so loss, watcher and
//! cleanup diagnostics were invisible in logcat. Android builds write them to
//! the platform log with liblog's public NDK entry point
//! `__android_log_write(int prio, const char* tag, const char* text)`
//! (NDK <android/log.h>; ANDROID_LOG_ERROR = 6), tag `nvk-rm`, and also to
//! stderr: bionic processes without logd (test harnesses, rented hosts) would
//! otherwise drop every diagnostic silently (liblog's write fails ENOTCONN).
//! GNU builds write one line to stderr only. Diagnostics are error paths only: formatting
//! allocates, the hot paths never call this.

/// Platform log tag of every backend diagnostic.
#[cfg(target_os = "android")]
const TAG: &std::ffi::CStr = c"nvk-rm";

#[cfg(target_os = "android")]
#[link(name = "log")]
unsafe extern "C" {
    fn __android_log_write(
        priority: std::ffi::c_int,
        tag: *const std::ffi::c_char,
        text: *const std::ffi::c_char,
    ) -> std::ffi::c_int;
}

/// Write one diagnostic line (no trailing newline needed).
pub(crate) fn write(message: std::fmt::Arguments<'_>) {
    #[cfg(target_os = "android")]
    {
        // Interior NULs would truncate the C string; replace them instead.
        let text = message.to_string().replace('\0', "\u{fffd}");
        if let Ok(text) = std::ffi::CString::new(text) {
            const ANDROID_LOG_ERROR: std::ffi::c_int = 6;
            // SAFETY: both pointers are valid NUL-terminated strings for the
            // duration of the call; liblog copies them.
            unsafe { __android_log_write(ANDROID_LOG_ERROR, TAG.as_ptr(), text.as_ptr()) };
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

/// The `log` facade's sink in this library: records from the logic crates
/// (nv-video) go to the same place as [`write`]. Warnings and errors only, so
/// no per-frame path formats anything (RUST.md §7, §14).
struct Sink;
impl log::Log for Sink {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            write(format_args!("{}: {}", record.target(), record.args()));
        }
    }
    fn flush(&self) {}
}

/// Install [`Sink`] as this library's `log` logger, once, on the first open
/// entry point (no load-time work, RUST.md §13). The facade's state is this
/// cdylib's own copy, so another logger can only exist when the library is
/// linked into a Rust program that set one: that program's logger then gets
/// the records, which is the intended outcome, so the refusal is ignored.
pub(crate) fn install_logger() {
    static SINK: Sink = Sink;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if log::set_logger(&SINK).is_ok() {
            log::set_max_level(log::LevelFilter::Warn);
        }
    });
}

/// Log every panic of this library with its source location before the
/// C-boundary guard turns it into device loss (installed once, chaining the
/// previous hook; this cdylib's panic hook covers only its own code).
pub(crate) fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let location = info
                .location()
                .map_or_else(|| "unknown location".into(), ToString::to_string);
            write(format_args!("nvk Rust RM panic at {location}: {info}"));
            previous(info);
        }));
    });
}

/// `eprintln!`-compatible diagnostic routed to logcat on Android.
macro_rules! diag {
    ($($arg:tt)*) => {
        $crate::diag::write(format_args!($($arg)*))
    };
}
pub(crate) use diag;

#[cfg(test)]
mod tests {
    #[test]
    fn diagnostic_formats_like_eprintln() {
        // GNU sink: must not panic for ordinary text or embedded NULs.
        super::diag!("nvk-rm diagnostic test {} {:?}", 7, "a\0b");
    }
}
