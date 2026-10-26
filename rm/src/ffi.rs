// SPDX-License-Identifier: MIT
//! The panic guard shared by every C entry point of this library (RUST.md §4).
//!
//! A Rust panic must never unwind into C: unwinding through a C frame is undefined behaviour,
//! and the C caller (NVK's nvkmd, the Codec2 component) has no way to recover the state the
//! panic left behind. Every `extern "C"` function therefore runs its body inside [`guard`] (or
//! [`guard_or_abort`] when the C interface has no error return). On a panic the guard logs the
//! entry point and the panic message at error level, marks the object the call was made on as
//! failed, and returns the interface's most specific failure value. Later calls on a failed
//! object return that same value without running their body, so nothing touches state a panic
//! interrupted.
//!
//! The guard owns no state and takes no lock; the failed flag lives in the object (see
//! [`Failable`]). It may be called from any thread.
use crate::diag::diag;
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, Ordering},
};

/// An object handed to C whose entry points stop working after a panic interrupted one of them.
pub(crate) trait Failable {
    /// The object was marked failed by an earlier guarded panic.
    fn failed(&self) -> bool;
    /// Record that a panic interrupted a call on this object. Must not panic.
    fn mark_failed(&self);
}

/// The failed flag of one C-visible object.
#[derive(Debug, Default)]
pub(crate) struct FailFlag(AtomicBool);

impl FailFlag {
    /// Whether a guarded panic marked the owner failed.
    pub(crate) fn is_set(&self) -> bool {
        // Acquire pairs with the Release in `set`: a caller that sees the flag also sees every
        // write the failing thread made before marking it (its diagnostic state).
        self.0.load(Ordering::Acquire)
    }
    /// Mark the owner failed; never cleared.
    pub(crate) fn set(&self) {
        // Release: see `is_set`.
        self.0.store(true, Ordering::Release);
    }
}

/// Text of a panic payload (`panic!` with a literal or a formatted message).
fn message(payload: &(dyn Any + Send)) -> &str {
    if let Some(text) = payload.downcast_ref::<&str>() {
        text
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text
    } else {
        "non-string panic payload"
    }
}

/// Run the body of the C entry point `entry` for `object` (`None` for entry points that create
/// their object or have none).
///
/// Returns the body's value, or the value of `failure` when the body panicked or `object` was
/// already marked failed (`failure` runs only then, so it may allocate an error message). A panic
/// is logged with `entry` and its message, and marks `object` failed.
pub(crate) fn guard<R>(
    entry: &'static str,
    object: Option<&dyn Failable>,
    failure: impl FnOnce() -> R,
    body: impl FnOnce() -> R,
) -> R {
    if let Some(object) = object
        && object.failed()
    {
        return failure();
    }
    // AssertUnwindSafe: after a panic, the only state anyone reads again is the failed flag set
    // below; every other entry point on the object returns `failure` without running.
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            diag!(
                "{entry}: internal panic ({}); the object is marked failed",
                message(payload.as_ref())
            );
            if let Some(object) = object {
                object.mark_failed();
            }
            failure()
        }
    }
}

/// Run the body of the C entry point `entry`, whose interface has no error return. A panic is
/// logged with `entry` and its message, then the process aborts: the C caller cannot be told,
/// and unwinding into it is undefined behaviour.
pub(crate) fn guard_or_abort<R>(entry: &'static str, body: impl FnOnce() -> R) -> R {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            diag!(
                "{entry}: internal panic ({}) in an entry point without an error return; aborting",
                message(payload.as_ref())
            );
            std::process::abort()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FailFlag, Failable, guard, guard_or_abort};
    use std::cell::Cell;

    #[derive(Default)]
    struct Object {
        flag: FailFlag,
        marks: Cell<u32>,
    }
    impl Failable for Object {
        fn failed(&self) -> bool {
            self.flag.is_set()
        }
        fn mark_failed(&self) {
            self.marks.set(self.marks.get() + 1);
            self.flag.set();
        }
    }

    #[test]
    fn returns_the_body_value() {
        let object = Object::default();
        assert_eq!(guard("test_entry", Some(&object), || -4, || 7), 7);
        assert_eq!(guard("test_entry", None, || -4, || 0), 0);
        assert!(!object.failed());
        assert_eq!(guard_or_abort("test_entry", || 5), 5);
    }

    #[test]
    fn panic_returns_the_failure_value_and_marks_the_object() {
        let object = Object::default();
        assert_eq!(
            guard(
                "test_entry",
                Some(&object),
                || -4,
                || -> i32 { panic!("formatted {}", 1) }
            ),
            -4
        );
        assert!(object.failed());
        assert_eq!(object.marks.get(), 1);
        assert_eq!(
            guard("test_entry", None, || -1, || -> i32 { panic!("literal") }),
            -1
        );
    }

    #[test]
    fn later_calls_on_a_failed_object_fail_without_running() {
        let object = Object::default();
        guard(
            "test_entry",
            Some(&object),
            || -4,
            || -> i32 { std::panic::panic_any(3u8) },
        );
        let ran = Cell::new(false);
        let result = guard(
            "test_entry",
            Some(&object),
            || -4,
            || {
                ran.set(true);
                0
            },
        );
        assert_eq!(result, -4);
        assert!(!ran.get());
        assert_eq!(object.marks.get(), 1);
        // Other objects are unaffected.
        assert_eq!(
            guard("test_entry", Some(&Object::default()), || -4, || 0),
            0
        );
    }

    /// The abort path, observed from a child process running this test binary.
    #[test]
    fn panic_without_an_error_return_aborts() {
        const CHILD: &str = "NV_RM_FFI_GUARD_ABORT_CHILD";
        if std::env::var_os(CHILD).is_some() {
            guard_or_abort("test_entry", || panic!("child"));
            return;
        }
        let exe = std::env::current_exe().expect("test binary path is known");
        let status = std::process::Command::new(exe)
            .args([
                "--exact",
                "ffi::tests::panic_without_an_error_return_aborts",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("child test process runs");
        use std::os::unix::process::ExitStatusExt;
        // SIGABRT = 6 (Linux asm-generic/signal.h).
        assert_eq!(status.signal(), Some(6), "{status:?}");
    }
}
