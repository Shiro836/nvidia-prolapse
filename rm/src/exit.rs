// SPDX-License-Identifier: MIT
//! Frees this process's RM clients when the process exits (RUST.md §13).
//!
//! RM frees a client when the last reference to its control file (`/dev/nvidiactl`) is dropped
//! (open-gpu-kernel-modules kernel-open/nvidia/nv.c `nvidia_ctl_close` → osapi.c
//! `RmFreeUnusedClients`), in whichever task drops it. At exit that can be another process
//! holding the file (a `pidfd_getfd` copy), and RM's automatic unmap of the client's CPU
//! mappings then fails in that task's context (resserv rs_client.c auto-unmap: status
//! NV_ERR_INVALID_CLIENT and one kernel assertion per mapping). So an exit handler frees every
//! live client on the exiting thread, where RM's teardown unmaps and frees everything, as the
//! NVIDIA driver does (libEGL_nvidia.so.615.78.08 `__cxa_atexit` handler E 0x9f950,
//! libGLX_nvidia.so.615.78.08 DT_FINI E 0xe3000). A process killed by a signal runs no handler;
//! that case stays the kernel's.
//!
//! The handler is registered with `atexit` on the first client open, never at load time.
//! Bionic's `atexit` (crtbegin_so.c) and glibc's bind it to this library's `__dso_handle`, so
//! `dlclose` runs it (with every client closed: a no-op) and removes it. A client inherited
//! across `fork` belongs to the parent and is never freed by the child's handler. The handler
//! releases only RM state: the control fds stay with their [`crate::rm::Client`]s, and a call
//! on a handle the program still holds fails with RM's invalid-client status. Each client's
//! free first RM-unmaps (NVOS34) its live CPU mappings ([`crate::live_maps`]), as the NVIDIA
//! driver does before its frees; the VMAs stay mapped (their owners may still hold pointers)
//! and go with the process's address space.
//!
//! One mutex over the live list, never held across an ioctl and never held while a client's
//! mapping list is locked; callable from any thread.
use crate::{
    diag::diag,
    process::ProcessBound,
    rm::{Result, RmHandle},
};
use std::{
    ffi::c_int,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

unsafe extern "C" {
    // ISO C atexit; bionic and glibc both bind the handler to the calling DSO.
    fn atexit(handler: extern "C" fn()) -> c_int;
}

/// The RM free of one client's root, holding what it needs (its control fd).
pub(crate) type FreeRoot = Box<dyn FnOnce() -> Result<()> + Send>;

/// A client the exit handler frees unless its owner frees it first.
struct LiveClient {
    id: u64,
    root: RmHandle,
    free: FreeRoot,
    process: ProcessBound,
}

static LIVE: Mutex<Vec<LiveClient>> = Mutex::new(Vec::new());
/// `atexit`'s result, from the first registration.
static REGISTERED: OnceLock<c_int> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A client's entry in the exit handler's list; [`Registration::withdraw`] takes it back.
#[derive(Debug)]
pub(crate) struct Registration {
    id: u64,
}

impl Registration {
    /// Enter the client `root` into the list the exit handler frees with `free`, registering
    /// the handler on first use.
    pub(crate) fn new(root: RmHandle, free: FreeRoot) -> std::io::Result<Self> {
        let status = *REGISTERED.get_or_init(|| {
            // SAFETY: registers a plain function of this library that takes no arguments;
            // libc runs it at exit or removes it at dlclose (module doc).
            unsafe { atexit(free_live_clients) }
        });
        if status != 0 {
            return Err(std::io::Error::other(format!(
                "atexit registration failed ({status})"
            )));
        }
        let process = ProcessBound::new()?;
        // Relaxed: the id only has to be unique; it orders nothing.
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        live()?.push(LiveClient {
            id,
            root,
            free,
            process,
        });
        Ok(Self { id })
    }
    /// Remove the client from the exit handler's list. False when the exit handler already
    /// took it (and freed its root), so the owner must not free it again.
    pub(crate) fn withdraw(&self) -> bool {
        let mut list = match live() {
            Ok(list) => list,
            Err(error) => {
                diag!("RM client registry: {error}");
                return false;
            }
        };
        match list.iter().position(|entry| entry.id == self.id) {
            Some(index) => {
                list.swap_remove(index);
                true
            }
            None => false,
        }
    }
}

fn live() -> std::io::Result<std::sync::MutexGuard<'static, Vec<LiveClient>>> {
    // A panic while the list was locked leaves it unknown: no client is freed from it.
    LIVE.lock()
        .map_err(|poisoned| std::io::Error::other(format!("RM client registry: {poisoned}")))
}

/// Take the live clients `wanted` selects out of the list.
fn take(wanted: impl Fn(&LiveClient) -> bool) -> Vec<LiveClient> {
    let mut list = match live() {
        Ok(list) => list,
        Err(error) => {
            diag!("RM exit teardown: {error}");
            return Vec::new();
        }
    };
    let (current, inherited) = std::mem::take(&mut *list)
        .into_iter()
        .partition(|entry| wanted(entry));
    *list = inherited;
    current
}

extern "C" fn free_live_clients() {
    crate::ffi::guard_or_abort("RM exit teardown", || {
        // The parent's clients, inherited across a `fork`, stay.
        for entry in take(|entry| entry.process.is_current()) {
            if let Err(error) = (entry.free)() {
                diag!(
                    "RM exit teardown: free client {:#x}: {error}",
                    entry.root.get()
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static FREED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    fn recorder(root: u32) -> FreeRoot {
        Box::new(move || {
            FREED.lock().expect("test lock").push(root);
            Ok(())
        })
    }
    /// The exit path frees each registered client once; an owner that withdrew first is
    /// skipped, and an owner whose client the exit path took learns it must not free it.
    #[test]
    #[cfg_attr(miri, ignore = "atexit registration")]
    fn exit_frees_each_live_client_once() {
        let kept =
            Registration::new(RmHandle::new(0xc1d0_7001), recorder(0xc1d0_7001)).expect("register");
        let closed =
            Registration::new(RmHandle::new(0xc1d0_7002), recorder(0xc1d0_7002)).expect("register");
        assert!(closed.withdraw(), "an owner closing first takes its entry");
        assert!(!closed.withdraw(), "an entry is withdrawn once");
        // Only this test's entries: other tests' clients share the list.
        let ours = [kept.id, closed.id];
        for entry in take(|entry| ours.contains(&entry.id) && entry.process.is_current()) {
            (entry.free)().expect("record");
        }
        let freed = FREED.lock().expect("test lock").clone();
        assert_eq!(freed, [0xc1d0_7001]);
        assert!(
            !kept.withdraw(),
            "the exit path took the entry: the owner must not free it"
        );
    }
}
