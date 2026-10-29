// SPDX-License-Identifier: MIT
//! Which push ranges a context accepts and how its pending pushes are cut
//! into submissions: at most [`MAX_PUSHES`] per `nv_rm_submit_batch`, never
//! between an incomplete push and the continuation that follows it
//! (`NV_RM_EXEC_INCOMPLETE`).
use nv_rm_abi::{NV_RM_EXEC_INCOMPLETE, NV_RM_EXEC_NO_PREFETCH, NvRmExec};
use thiserror::Error;

/// Pushes one `nv_rm_submit_batch` accepts (the RM backend refuses more:
/// `rm/src/gpu.rs` `Queue::submit`).
pub const MAX_PUSHES: usize = 1000;

/// Pushes are at most this many bytes, exclusive: a GPFIFO entry's length
/// field counts 32-bit words in 21 bits (`NV906F_GP_ENTRY1_LENGTH` 30:10,
/// open-gpu-kernel-modules `cl906f.h`; the RM refuses the same,
/// `rm/src/gpu.rs` `Queue::submit`).
pub const PUSH_SIZE_LIMIT: u32 = 1 << 23;

/// One push NVK hands a context (`nvkmd_ctx_exec`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exec {
    /// GPU VA of the push.
    pub addr: u64,
    /// Size in bytes.
    pub size: u32,
    /// The push continues in the next one.
    pub incomplete: bool,
    /// The GPU must not prefetch the push.
    pub no_prefetch: bool,
}

/// A push a context refuses.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ExecError {
    /// A push that is empty, not word-aligned in address or size, or at
    /// least [`PUSH_SIZE_LIMIT`] bytes.
    #[error("invalid push range")]
    InvalidRange {
        /// GPU VA of the push.
        addr: u64,
        /// Its size in bytes.
        size: u32,
    },
}

/// A push list that cannot be submitted.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PushError {
    /// [`MAX_PUSHES`] consecutive pushes are all incomplete: no submission can
    /// end on a complete push.
    #[error("push continuation chain exceeding 1000 entries")]
    ChainTooLong {
        /// Index of the first push of the chain.
        at: usize,
    },
}

/// The RM's form of `exec` (`nvkmd_rm.c` `rm_ctx_exec`).
///
/// # Errors
/// [`ExecError::InvalidRange`] for a push the GPU cannot fetch.
pub fn exec(exec: Exec) -> Result<NvRmExec, ExecError> {
    let Exec {
        addr,
        size,
        incomplete,
        no_prefetch,
    } = exec;
    if addr % 4 != 0 || size % 4 != 0 || size == 0 || size >= PUSH_SIZE_LIMIT {
        return Err(ExecError::InvalidRange { addr, size });
    }
    let mut flags = 0;
    if no_prefetch {
        flags |= NV_RM_EXEC_NO_PREFETCH;
    }
    if incomplete {
        flags |= NV_RM_EXEC_INCOMPLETE;
    }
    Ok(NvRmExec {
        address: addr,
        size_B: size,
        flags,
    })
}

/// Whether `push` continues in the next push.
#[must_use]
pub const fn is_incomplete(push: &NvRmExec) -> bool {
    push.flags & NV_RM_EXEC_INCOMPLETE != 0
}

/// Length of the first submission of `pending`: up to [`MAX_PUSHES`] pushes,
/// shortened so that it does not end on an incomplete push.
/// `incomplete(i)` says whether push `i` continues in push `i + 1`.
///
/// # Errors
/// [`PushError::ChainTooLong`] when no such length exists; also for an empty
/// list, which has no submission.
pub fn first_chunk(len: usize, incomplete: impl Fn(usize) -> bool) -> Result<usize, PushError> {
    let mut count = len.min(MAX_PUSHES);
    while count != 0 && incomplete(count - 1) {
        count -= 1;
    }
    if count == 0 {
        Err(PushError::ChainTooLong { at: 0 })
    } else {
        Ok(count)
    }
}

/// Checks every submission boundary of `len` pending pushes before any of
/// them is submitted, so a list that cannot be cut is refused whole
/// (`nvkmd_rm.c` `rm_ctx_flush_signals`, the validation loop).
///
/// # Errors
/// [`PushError::ChainTooLong`] with the index where no cut is possible.
pub fn check_chunks(len: usize, incomplete: impl Fn(usize) -> bool) -> Result<(), PushError> {
    let mut offset = 0;
    while offset < len {
        offset += first_chunk(len - offset, |i| incomplete(offset + i))
            .map_err(|PushError::ChainTooLong { .. }| PushError::ChainTooLong { at: offset })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(addr: u64, size: u32) -> Exec {
        Exec {
            addr,
            size,
            incomplete: false,
            no_prefetch: false,
        }
    }

    #[test]
    fn push_ranges_must_be_words_below_the_limit() {
        // nvkmd_rm.c rm_ctx_exec: 4-byte aligned address and size, 0 < size < 8 MiB.
        for (addr, size) in [(0x1002, 4), (0x1000, 6), (0x1000, 0), (0x1000, 1 << 23)] {
            assert_eq!(
                exec(push(addr, size)),
                Err(ExecError::InvalidRange { addr, size })
            );
        }
        assert_eq!(
            exec(push(0x1000, (1 << 23) - 4)),
            Ok(NvRmExec {
                address: 0x1000,
                size_B: (1 << 23) - 4,
                flags: 0
            })
        );
    }

    #[test]
    fn push_flags_map_to_the_rm_bits() {
        let flags = |incomplete, no_prefetch| {
            exec(Exec {
                incomplete,
                no_prefetch,
                ..push(0x1000, 4)
            })
            .map(|push| push.flags)
        };
        assert_eq!(flags(false, false), Ok(0));
        assert_eq!(flags(true, false), Ok(NV_RM_EXEC_INCOMPLETE));
        assert_eq!(flags(false, true), Ok(NV_RM_EXEC_NO_PREFETCH));
        assert_eq!(
            flags(true, true),
            Ok(NV_RM_EXEC_INCOMPLETE | NV_RM_EXEC_NO_PREFETCH)
        );
        let pushed = exec(Exec {
            incomplete: true,
            ..push(0x1000, 4)
        })
        .expect("a valid push");
        assert!(is_incomplete(&pushed));
    }

    /// Chunk lengths in submission order, as the C flush loop cuts them.
    fn chunks(incomplete: &[bool]) -> Result<Vec<usize>, PushError> {
        check_chunks(incomplete.len(), |i| incomplete[i])?;
        let mut out = Vec::new();
        let mut offset = 0;
        while offset < incomplete.len() {
            let count = first_chunk(incomplete.len() - offset, |i| incomplete[offset + i])?;
            out.push(count);
            offset += count;
        }
        Ok(out)
    }

    #[test]
    fn lists_are_cut_at_the_push_limit() {
        // nvkmd_rm.c rm_ctx_flush_signals with 0/1/999/1000/1001/2000 complete pushes.
        assert_eq!(chunks(&[]), Ok(vec![]));
        assert_eq!(chunks(&[false]), Ok(vec![1]));
        assert_eq!(chunks(&[false; 999]), Ok(vec![999]));
        assert_eq!(chunks(&[false; 1000]), Ok(vec![1000]));
        assert_eq!(chunks(&[false; 1001]), Ok(vec![1000, 1]));
        assert_eq!(chunks(&[false; 2000]), Ok(vec![1000, 1000]));
    }

    #[test]
    fn a_cut_never_separates_a_push_from_its_continuation() {
        // nvkmd_rm.c rm_ctx_flush_signals: both loops shorten a chunk ending on an incomplete push.
        let mut list = vec![false; 1001];
        list[998] = true;
        list[999] = true;
        assert_eq!(chunks(&list), Ok(vec![998, 3]));
        let mut list = vec![false; 1500];
        list[999] = true;
        assert_eq!(chunks(&list), Ok(vec![999, 501]));
    }

    #[test]
    fn an_uncuttable_chain_is_refused_before_anything_is_submitted() {
        // nvkmd_rm.c rm_ctx_flush_signals, the validation loop: a 1000-push incomplete chain, at the start or later.
        assert_eq!(
            chunks(&[true; 1000]),
            Err(PushError::ChainTooLong { at: 0 })
        );
        let mut list = vec![false; 2001];
        for incomplete in &mut list[500..1500] {
            *incomplete = true;
        }
        assert_eq!(chunks(&list), Err(PushError::ChainTooLong { at: 500 }));
        // A trailing incomplete push with nothing after it cannot end a submission.
        assert_eq!(chunks(&[true]), Err(PushError::ChainTooLong { at: 0 }));
    }
}
