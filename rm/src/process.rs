// SPDX-License-Identifier: MIT
//! Which process an object belongs to (RUST.md §13).
//!
//! The library is loaded into processes that may `fork` (Android's zygote, app code). A child
//! inherits the parent's descriptors, mappings and Rust objects, but not its threads, and the
//! RM objects behind those descriptors are still the parent's. A device or session handle
//! created before a `fork` must therefore never be used, or torn down, in the child.
//!
//! [`ProcessBound`] records the fork generation at creation; a `pthread_atfork` child handler
//! advances the generation in every child. Checking costs one atomic load and no syscall. The
//! handler is registered on the first [`ProcessBound::new`] (never at load time). Both bionic
//! and glibc remove a library's atfork handlers when it is `dlclose`d (they are registered
//! with the library's `__dso_handle`).
//!
//! No locks; callable from any thread.
use std::{
    ffi::c_int,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

/// Number of `fork`s since this library registered its handler, as seen by this process.
static FORKS: AtomicU64 = AtomicU64::new(0);
/// pthread_atfork's result, from the first registration.
static REGISTERED: OnceLock<c_int> = OnceLock::new();

unsafe extern "C" {
    // POSIX pthread_atfork; bionic and glibc both bind it to the calling DSO.
    fn pthread_atfork(
        prepare: Option<extern "C" fn()>,
        parent: Option<extern "C" fn()>,
        child: Option<extern "C" fn()>,
    ) -> c_int;
}

/// Runs in the child, on the forking thread, before `fork` returns: async-signal-safe (one
/// lock-free atomic add).
extern "C" fn in_child() {
    // Relaxed: the child has one thread, and every later load in it follows this store in
    // program order.
    FORKS.fetch_add(1, Ordering::Relaxed);
}

/// The process (fork generation) an object was created in.
#[derive(Debug)]
pub(crate) struct ProcessBound {
    generation: u64,
    pid: u32,
}

impl ProcessBound {
    /// Bind to the calling process. Fails only when the atfork handler cannot be registered
    /// (ENOMEM), with the libc error.
    pub(crate) fn new() -> std::io::Result<Self> {
        let status = *REGISTERED.get_or_init(|| {
            // SAFETY: registers a plain function of this library with no prepare/parent
            // handlers; libc removes it when the library is unloaded (module doc).
            unsafe { pthread_atfork(None, None, Some(in_child)) }
        });
        if status != 0 {
            return Err(std::io::Error::from_raw_os_error(status));
        }
        Ok(Self {
            // Relaxed: see `in_child`; a fork racing this load on another thread leaves the
            // object bound to the parent, which is the process it was created in.
            generation: FORKS.load(Ordering::Relaxed),
            pid: std::process::id(),
        })
    }
    /// The object was created in this process (not inherited across `fork`).
    #[inline]
    pub(crate) fn is_current(&self) -> bool {
        // Relaxed: see `in_child`.
        FORKS.load(Ordering::Relaxed) == self.generation
    }
    /// Pid of the creating process, for diagnostics.
    pub(crate) fn creator(&self) -> u32 {
        self.pid
    }
}

#[cfg(test)]
mod tests {
    use super::ProcessBound;

    #[test]
    fn an_object_from_before_fork_is_not_current_in_the_child() {
        unsafe extern "C" {
            fn fork() -> i32;
            fn waitpid(pid: i32, status: *mut i32, flags: i32) -> i32;
            fn _exit(status: i32) -> !;
        }
        let bound = ProcessBound::new().expect("atfork registration");
        assert!(bound.is_current());
        // SAFETY: the child only reads atomics and calls the async-signal-safe _exit.
        let pid = unsafe { fork() };
        assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
        if pid == 0 {
            let code = if bound.is_current() { 1 } else { 0 };
            // SAFETY: async-signal-safe; no destructors run in the child.
            unsafe { _exit(code) };
        }
        let mut status = -1;
        loop {
            // SAFETY: waits for our own child; `status` is a live local.
            let got = unsafe { waitpid(pid, &mut status, 0) };
            if got == pid {
                break;
            }
            assert_eq!(
                std::io::Error::last_os_error().kind(),
                std::io::ErrorKind::Interrupted
            );
        }
        assert_eq!(status, 0, "the child saw the parent's object as current");
        assert!(bound.is_current(), "the parent stays current");
        assert!(ProcessBound::new().expect("second binding").is_current());
    }
}
