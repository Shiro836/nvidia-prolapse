// SPDX-License-Identifier: MIT
//! Backend diagnostics sink.
//!
//! Loss, watcher and cleanup diagnostics go to the driver's shared sink
//! (`nv_ffi::diag`: the Android log and stderr) under the tag `nvk-rm`.
//! Diagnostics are error paths only: formatting allocates, the hot paths
//! never call this.

/// Platform log tag of every backend diagnostic.
const TAG: &std::ffi::CStr = c"nvk-rm";

/// Write one diagnostic line (no trailing newline needed).
pub(crate) fn write(message: std::fmt::Arguments<'_>) {
    nv_ffi::diag::write(TAG, message);
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
