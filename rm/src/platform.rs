// SPDX-License-Identifier: MIT
//! CPU/libc contracts for direct RM userspace.
//!
//! ABI authority: NDK r29 bits/ioctl.h, sys/mman.h and unistd.h; glibc
//! sys/ioctl.h, sys/mman.h and unistd.h. All offsets here are explicit off64_t.
//! Device barriers follow Linux arch/{x86,arm64,arm,riscv}/include/asm/barrier.h
//! hardware instructions, not its weaker CPU-only SMP barriers.
//! Barriers order accesses; they do not clean noncoherent CPU caches, drain a
//! remote PCIe endpoint, or replace the queue's posted-write readback.

use std::{
    ffi::{c_int, c_void},
    io,
};

#[cfg(not(any(target_os = "android", all(target_os = "linux", target_env = "gnu"))))]
compile_error!(
    "RM libc ABI is implemented only for Android bionic and Linux glibc; this libc needs independent ioctl/off64_t proof"
);
#[cfg(not(target_endian = "little"))]
compile_error!(
    "RM wire encoding is little-endian; big-endian targets need explicit wire conversion"
);
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "arm",
    target_arch = "riscv64"
)))]
compile_error!("RM device-memory barriers are not implemented for this CPU architecture");
#[cfg(all(target_arch = "arm", not(target_os = "android")))]
compile_error!(
    "32-bit ARM GNU targets need explicit ARMv7+ barrier and ABI validation; use the audited armv7-linux-androideabi target"
);
#[cfg(not(target_has_atomic = "64"))]
compile_error!("RM semaphore bookkeeping requires CPU AtomicU64 support on cached system memory");

#[cfg(target_os = "android")]
type IoctlRequest = c_int;
#[cfg(not(target_os = "android"))]
type IoctlRequest = std::ffi::c_ulong;

unsafe extern "C" {
    // NDK r29 sys/mman.h (API30+) and glibc: identical scalar arguments on
    // all five supported CPUs. fcntl seal commands use an int, never off_t.
    fn memfd_create(name: *const std::ffi::c_char, flags: u32) -> c_int;
    fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    #[link_name = "ioctl"]
    fn libc_ioctl(fd: c_int, request: IoctlRequest, ...) -> c_int;
    // LP32 glibc needs the same large-file symbol selection as LP32 bionic.
    // LP64 libc mmap/lseek already take signed 64-bit offsets.
    #[cfg_attr(target_pointer_width = "32", link_name = "mmap64")]
    #[cfg_attr(target_pointer_width = "64", link_name = "mmap")]
    fn libc_mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    #[link_name = "munmap"]
    fn libc_munmap(address: *mut c_void, length: usize) -> c_int;
    #[cfg_attr(target_pointer_width = "32", link_name = "lseek64")]
    #[cfg_attr(target_pointer_width = "64", link_name = "lseek")]
    fn libc_lseek(fd: c_int, offset: i64, whence: c_int) -> i64;
    fn getpagesize() -> c_int;
}

pub(crate) fn create_sealable_memfd() -> io::Result<std::os::fd::OwnedFd> {
    use std::os::fd::FromRawFd;
    // Linux UAPI linux/memfd.h: MFD_CLOEXEC | MFD_ALLOW_SEALING.
    // SAFETY: a static NUL-terminated name; memfd_create reads only it.
    let fd = unsafe { memfd_create(c"nv-native-sync".as_ptr(), 1 | 2) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful memfd_create transfers one new descriptor to us.
    Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) })
}

pub(crate) fn file_seals(fd: c_int) -> io::Result<u32> {
    // Linux UAPI linux/fcntl.h: F_GET_SEALS = F_LINUX_SPECIFIC_BASE + 10.
    // SAFETY: F_GET_SEALS takes no pointer; any fd value is a valid argument
    // (a bad one fails with EBADF).
    let result = unsafe { fcntl(fd, 1034, 0 as c_int) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result as u32)
    }
}

pub(crate) fn add_file_seals(fd: c_int, seals: u32) -> io::Result<()> {
    // F_ADD_SEALS = F_LINUX_SPECIFIC_BASE + 9; the third argument is an int.
    // SAFETY: F_ADD_SEALS takes an int, no pointer; a bad fd fails with EBADF.
    let result = unsafe { fcntl(fd, 1033, seals as c_int) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Issue a pointer-argument ioctl while preserving all 32 request bits.
///
/// # Safety
/// `argument` and any nested pointers must satisfy the request's exact wire
/// layout, initialized-byte and lifetime requirements. The libc variadic call
/// handles target calling conventions; never substitute a raw syscall here.
pub(crate) unsafe fn ioctl(fd: c_int, request: u32, argument: *mut c_void) -> c_int {
    // SAFETY: the caller's contract (# Safety) covers the argument buffer.
    unsafe { libc_ioctl(fd, request as IoctlRequest, argument) }
}

/// Map memory using an explicit signed 64-bit file offset on every ABI.
///
/// # Safety
/// The caller must enforce mmap's address, length, offset and ownership rules,
/// check MAP_FAILED, and avoid creating aliased Rust references to device data.
pub(crate) unsafe fn mmap(
    address: *mut c_void,
    length: usize,
    protection: c_int,
    flags: c_int,
    fd: c_int,
    offset: i64,
) -> *mut c_void {
    // SAFETY: the caller's contract (# Safety).
    unsafe { libc_mmap(address, length, protection, flags, fd, offset) }
}

/// # Safety
/// The caller owns this mapping and must stop all accesses before unmapping it.
pub(crate) unsafe fn munmap(address: *mut c_void, length: usize) -> c_int {
    // SAFETY: the caller's contract (# Safety).
    unsafe { libc_munmap(address, length) }
}

/// Seek with off64_t even on LP32; returns -1 with libc errno on failure.
///
/// # Safety
/// The caller must coordinate ownership of the file description and offset.
pub(crate) unsafe fn lseek(fd: c_int, offset: i64, whence: c_int) -> i64 {
    // SAFETY: the caller's contract (# Safety); no pointer arguments.
    unsafe { libc_lseek(fd, offset, whence) }
}

/// Validate CPU mapping granularity, independently of the GPU's 4-KiB pages.
pub(crate) fn validate_page_size(bytes: usize) -> io::Result<usize> {
    match bytes {
        4096 | 16384 | 65536 => Ok(bytes),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "unsupported CPU page size {bytes}; RM CPU mappings implement 4096, 16384 and 65536-byte pages"
            ),
        )),
    }
}

/// Validate the host before opening any driver device. This is not a GPU gate.
pub(crate) fn validate_runtime() -> io::Result<usize> {
    // SAFETY: no arguments, no side effects and no device access.
    let bytes = unsafe { getpagesize() };
    if bytes <= 0 {
        return Err(io::Error::other(format!(
            "libc getpagesize returned {bytes}"
        )));
    }
    validate_page_size(bytes as usize)
}

/// Compute whole CPU-page backing without overflow or zero-length mappings.
pub(crate) fn round_to_cpu_pages(bytes: usize, page_size: usize) -> io::Result<usize> {
    validate_page_size(page_size)?;
    if bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "zero-sized CPU mapping",
        ));
    }
    bytes
        .checked_add(page_size - 1)
        .map(|n| n & !(page_size - 1))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU mapping page-rounding overflow",
            )
        })
}

/// Order both device and ordinary memory reads/writes; not cache maintenance.
#[inline]
pub(crate) fn device_full_barrier() {
    // asm's default memory clobber is intentional: never use nomem/readonly.
    #[cfg(target_arch = "x86_64")]
    // SAFETY: mfence touches no memory, stack or flags; the default memory
    // clobber keeps the compiler from moving accesses across it.
    unsafe {
        std::arch::asm!("mfence", options(nostack, preserves_flags));
    }
    // A locked operation on a real local word also supports CPUs without SSE2.
    #[cfg(target_arch = "x86")]
    {
        let mut scratch = 0u32;
        // SAFETY: the locked add writes only the local `scratch` (adding 0);
        // the memory clobber orders every other access around it.
        unsafe {
            std::arch::asm!("lock add dword ptr [{word}], 0",
            word = in(reg) &mut scratch, options(nostack));
        }
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "riscv64")]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("fence iorw, iorw", options(nostack, preserves_flags));
    }
}

/// Order prior mapped-device/ordinary reads before subsequent reads.
#[inline]
pub(crate) fn device_read_barrier() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("lfence", options(nostack, preserves_flags));
    }
    #[cfg(any(target_arch = "x86", target_arch = "arm"))]
    device_full_barrier();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("dsb ld", options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "riscv64")]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("fence ir, ir", options(nostack, preserves_flags));
    }
}

/// Order prior mapped-device/ordinary writes before subsequent writes.
#[inline]
pub(crate) fn device_write_barrier() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("sfence", options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86")]
    device_full_barrier();
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("dsb st", options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "riscv64")]
    // SAFETY: a barrier instruction; no memory, stack or flags are changed.
    unsafe {
        std::arch::asm!("fence ow, ow", options(nostack, preserves_flags));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_cpu_pages_are_distinct_from_gpu_pages() {
        for page in [4096, 16384, 65536] {
            assert_eq!(validate_page_size(page).unwrap(), page);
            assert_eq!(round_to_cpu_pages(4096, page).unwrap(), page);
            for n in [1, page - 1, page, page + 1, 3 * page - 1] {
                let rounded = round_to_cpu_pages(n, page).unwrap();
                assert!(rounded >= n && rounded - n < page);
                assert_eq!(rounded % page, 0);
            }
        }
    }

    #[test]
    fn invalid_granularities_and_overflow_fail_loudly() {
        for page in [0, 1, 1024, 8192, 16385, usize::MAX] {
            assert_eq!(
                validate_page_size(page).unwrap_err().kind(),
                io::ErrorKind::Unsupported
            );
        }
        for page in [4096, 16384, 65536] {
            assert!(round_to_cpu_pages(0, page).is_err());
            assert!(round_to_cpu_pages(usize::MAX, page).is_err());
            let largest = usize::MAX & !(page - 1);
            assert_eq!(round_to_cpu_pages(largest, page).unwrap(), largest);
            assert!(round_to_cpu_pages(largest + 1, page).is_err());
        }
    }
}
