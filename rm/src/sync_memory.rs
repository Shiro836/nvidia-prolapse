// SPDX-License-Identifier: MIT
//! CPU-only backing for this driver's private opaque-FD synchronization ABI.
//! A sealed memfd supplies normal cached shared memory and a kernel-enforced
//! fixed extent. It is not an NVIDIA RM object or a waitable DRM fence. GPU
//! completion points use separate RM allocations; this payload has no GPU map.
//! The protocol owner validates its header and orders all shared atomic access.
use crate::diag::diag;
use crate::{
    platform,
    rm::{Error, Result},
};
use nv_rm_core::conv::u64_from_usize;
use std::{
    fs::File,
    os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd},
    ptr::NonNull,
    sync::atomic::AtomicU64,
};

pub(crate) const PAYLOAD_SIZE: usize = 4096;
// Linux UAPI linux/fcntl.h. SEAL seals the seal set itself; GROW/SHRINK stop
// size changes (including SIGBUS attacks) after our fstat validation.
const REQUIRED_SEALS: u32 = 1 | 2 | 4;
const WRITE_SEALS: u32 = 8 | 16; // WRITE | FUTURE_WRITE forbid writable imports.

fn validate_properties(regular: bool, size: u64, seals: u32) -> Result<()> {
    if !regular || size != u64_from_usize(PAYLOAD_SIZE) {
        return Err(Error::Contract(format!(
            "sync memfd must be a regular {PAYLOAD_SIZE}-byte file (size {size})"
        )));
    }
    if seals & REQUIRED_SEALS != REQUIRED_SEALS || seals & WRITE_SEALS != 0 {
        return Err(Error::Contract(format!(
            "sync memfd needs immutable size/seals and writable contents (seals {seals:#x})"
        )));
    }
    Ok(())
}

/// Independently owned mapping and FD. Mapping lifetime never depends on the
/// exporter, borrowed import descriptor, RM device or another Rust wrapper.
pub(crate) struct SyncMemory {
    pointer: NonNull<u8>,
    file: File,
}
// SAFETY: an owned MAP_SHARED mapping of a sealed fixed-size memfd plus its
// File. All accesses go through word(), aligned AtomicU64 operations, so
// moving the owner between threads introduces no data race; the mapping is
// unmapped only by Drop, after every owner is gone.
unsafe impl Send for SyncMemory {}
// SAFETY: as Send: shared access is atomic-only.
unsafe impl Sync for SyncMemory {}
impl SyncMemory {
    pub(crate) fn new_cpu() -> Result<Self> {
        let file = File::from(
            platform::create_sealable_memfd().map_err(|e| Error::Io("create sync memfd", e))?,
        );
        file.set_len(u64_from_usize(PAYLOAD_SIZE))
            .map_err(|e| Error::Io("size sync memfd", e))?;
        platform::add_file_seals(file.as_raw_fd(), REQUIRED_SEALS)
            .map_err(|e| Error::Io("seal sync memfd extent", e))?;
        // A fresh memfd's newly allocated pages are zero initialized by Linux.
        Self::map(file)
    }
    pub(crate) fn import_cpu(fd: BorrowedFd<'_>) -> Result<Self> {
        // Pin the file description before validation. The caller retains its FD;
        // Vulkan's import layer decides whether successful import consumes it.
        let owned = fd
            .try_clone_to_owned()
            .map_err(|e| Error::Io("duplicate sync memfd", e))?;
        Self::map(File::from(owned))
    }
    fn map(file: File) -> Result<Self> {
        let seals = platform::file_seals(file.as_raw_fd())
            .map_err(|e| Error::Io("query sync memfd seals", e))?;
        // Require F_SEAL_SEAL before fstat: the producer can neither resize the
        // file nor add a write seal between these checks and mmap.
        if seals & REQUIRED_SEALS != REQUIRED_SEALS || seals & WRITE_SEALS != 0 {
            return Err(Error::Contract(format!(
                "sync memfd has unsafe seals {seals:#x}"
            )));
        }
        let stat = file
            .metadata()
            .map_err(|e| Error::Io("stat sync memfd", e))?;
        validate_properties(stat.file_type().is_file(), stat.len(), seals)?;
        // SAFETY: fixed sealed extent (no SIGBUS by truncation), offset zero,
        // read/write MAP_SHARED at a kernel-chosen address (NULL hint, no
        // MAP_FIXED): no existing mapping is replaced. Owned by Self from here.
        let raw = unsafe {
            platform::mmap(
                std::ptr::null_mut(),
                PAYLOAD_SIZE,
                3,
                1,
                file.as_raw_fd(),
                0,
            )
        };
        // MAP_FAILED is `(void *)-1` (sys/mman.h).
        if raw.addr() == usize::MAX {
            return Err(Error::Io("map sync memfd", std::io::Error::last_os_error()));
        }
        let Some(pointer) = NonNull::new(raw.cast::<u8>()) else {
            // SAFETY: even the unusual successful address-zero mapping is ours.
            unsafe {
                platform::munmap(raw, PAYLOAD_SIZE);
            }
            return Err(Error::Contract("sync memfd mapped at address zero".into()));
        };
        Ok(Self { pointer, file })
    }
    pub(crate) fn export(&self) -> Result<OwnedFd> {
        self.file
            .as_fd()
            .try_clone_to_owned()
            .map_err(|e| Error::Io("export sync memfd", e))
    }
    /// The 64-bit word at `offset` of the shared payload. Other processes map
    /// the same pages and update the words concurrently, so every access goes
    /// through the returned atomic and never through a plain reference.
    ///
    /// Panics when `offset` is not 8-aligned or leaves the payload (callers
    /// use the protocol's constant offsets).
    #[inline]
    pub(crate) fn word(&self, offset: usize) -> &AtomicU64 {
        assert!(
            offset.is_multiple_of(8) && offset <= PAYLOAD_SIZE - 8,
            "sync memfd word offset {offset} outside the payload"
        );
        // SAFETY: offset + 8 <= PAYLOAD_SIZE (asserted), inside this owned
        // mapping, which stays mapped while `self` is borrowed (Drop unmaps).
        let word = unsafe { self.pointer.as_ptr().add(offset) };
        // SAFETY: the page-aligned mapping plus an 8-aligned offset is
        // 8-aligned; AtomicU64 tolerates the other processes' concurrent
        // atomic writes.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "the offset is asserted 8-byte aligned above and the mapping is page aligned"
        )]
        let word = word.cast::<AtomicU64>();
        // SAFETY: as above: in bounds, aligned, atomic-only access.
        unsafe { &*word }
    }
}
impl Drop for SyncMemory {
    fn drop(&mut self) {
        // SAFETY: all references/pending users must retain this SyncMemory owner.
        if unsafe { platform::munmap(self.pointer.as_ptr().cast(), PAYLOAD_SIZE) } != 0 {
            diag!(
                "unmap sync memfd failed; VMA remains owned by kernel: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    fn word(memory: &SyncMemory, offset: usize) -> &AtomicU64 {
        memory.word(offset)
    }
    #[test]
    fn rejects_wrong_extent_type_and_seals() {
        assert!(validate_properties(true, 4096, REQUIRED_SEALS).is_ok());
        for size in [0, 1, 4095, 8192, u64::MAX] {
            assert!(validate_properties(true, size, REQUIRED_SEALS).is_err());
        }
        assert!(validate_properties(false, 4096, REQUIRED_SEALS).is_err());
        for seals in [0, 1, 2, 4, 3, 5, 6, 7 | 8, 7 | 16] {
            assert!(validate_properties(true, 4096, seals).is_err());
        }
    }
    #[test]
    fn independent_import_survives_exporter_and_fd_close() {
        use std::os::unix::fs::MetadataExt;
        let source = SyncMemory::new_cpu().unwrap();
        let identity = source.file.metadata().unwrap();
        // Count this allocation's inode, not every process FD: unrelated tests
        // may open files concurrently without making this leak check flaky.
        let references = || {
            std::fs::read_dir("/proc/self/fd")
                .unwrap()
                .filter_map(|entry| std::fs::metadata(entry.ok()?.path()).ok())
                .filter(|stat| stat.ino() == identity.ino() && stat.dev() == identity.dev())
                .count()
        };
        assert_eq!(references(), 1);
        for offset in (0..PAYLOAD_SIZE).step_by(8) {
            assert_eq!(word(&source, offset).load(Ordering::Acquire), 0);
        }
        let fd = source.export().unwrap();
        let target = SyncMemory::import_cpu(fd.as_fd()).unwrap();
        assert_eq!(references(), 3);
        word(&source, 0).store(123, Ordering::Release);
        assert_eq!(word(&target, 0).load(Ordering::Acquire), 123);
        word(&target, PAYLOAD_SIZE - 8).store(456, Ordering::Release);
        assert_eq!(word(&source, PAYLOAD_SIZE - 8).load(Ordering::Acquire), 456);
        drop(source);
        let another = SyncMemory::import_cpu(fd.as_fd()).unwrap();
        drop(fd);
        word(&another, 0).store(789, Ordering::Release);
        assert_eq!(word(&target, 0).load(Ordering::Acquire), 789);
        let file = File::from(target.export().unwrap());
        assert!(file.set_len(0).is_err());
        assert!(file.set_len(8192).is_err());
        assert!(platform::add_file_seals(file.as_raw_fd(), 8).is_err());
        drop((file, another, target));
        assert_eq!(references(), 0);
    }
    #[test]
    fn real_invalid_files_are_rejected_without_consuming_them() {
        let dev_null = File::open("/dev/null").unwrap();
        assert!(SyncMemory::import_cpu(dev_null.as_fd()).is_err());
        assert!(dev_null.metadata().is_ok());
        for (size, seals) in [
            (4096, 0),
            (4096, 2 | 4),
            (0, REQUIRED_SEALS),
            (8192, REQUIRED_SEALS),
            (4096, REQUIRED_SEALS | 8),
            (4096, REQUIRED_SEALS | 16),
        ] {
            let file = File::from(platform::create_sealable_memfd().unwrap());
            file.set_len(size).unwrap();
            if seals != 0 {
                platform::add_file_seals(file.as_raw_fd(), seals).unwrap();
            }
            for _ in 0..8 {
                assert!(SyncMemory::import_cpu(file.as_fd()).is_err());
            }
            assert_eq!(file.metadata().unwrap().len(), size);
        }
    }
    #[test]
    fn forked_process_observes_shared_atomics() {
        unsafe extern "C" {
            fn fork() -> i32;
            fn waitpid(pid: i32, status: *mut i32, flags: i32) -> i32;
            fn _exit(status: i32) -> !;
        }
        let memory = SyncMemory::new_cpu().unwrap();
        let counter = word(&memory, 0);
        counter.store(1, Ordering::Release);
        // SAFETY: child performs only a lock-free atomic and async-signal-safe
        // _exit, with no allocator/libc/Rust destructor calls after fork.
        let pid = unsafe { fork() };
        assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
        if pid == 0 {
            counter.fetch_add(1, Ordering::AcqRel);
            unsafe {
                _exit(0);
            }
        }
        let mut status = 0;
        loop {
            let got = unsafe { waitpid(pid, &raw mut status, 0) };
            if got == pid {
                break;
            }
            assert!(
                got < 0
                    && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
            );
        }
        assert_eq!(status, 0);
        assert_eq!(counter.load(Ordering::Acquire), 2);
    }
}
