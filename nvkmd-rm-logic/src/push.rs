// SPDX-License-Identifier: MIT
//! How a context's pending pushes are cut into submissions: at most
//! [`MAX_PUSHES`] per `nv_rm_submit_batch`, never between an incomplete push
//! and the continuation that follows it (`NV_RM_EXEC_INCOMPLETE`).
use thiserror::Error;

/// Pushes one `nv_rm_submit_batch` accepts (the RM backend refuses more:
/// `rm/src/gpu.rs` `Queue::submit`).
pub const MAX_PUSHES: usize = 1000;

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
        // nvkmd_rm.c:1111-1133 with 0/1/999/1000/1001/2000 complete pushes.
        assert_eq!(chunks(&[]), Ok(vec![]));
        assert_eq!(chunks(&[false]), Ok(vec![1]));
        assert_eq!(chunks(&[false; 999]), Ok(vec![999]));
        assert_eq!(chunks(&[false; 1000]), Ok(vec![1000]));
        assert_eq!(chunks(&[false; 1001]), Ok(vec![1000, 1]));
        assert_eq!(chunks(&[false; 2000]), Ok(vec![1000, 1000]));
    }

    #[test]
    fn a_cut_never_separates_a_push_from_its_continuation() {
        // nvkmd_rm.c:1113-1114, 1125-1126.
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
        // nvkmd_rm.c:1115-1116: a 1000-push incomplete chain, at the start or later.
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
