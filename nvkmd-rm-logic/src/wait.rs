// SPDX-License-Identifier: MIT
//! Which of a context's waits become GPU acquires (`nvkmd_ctx_wait` on a
//! context with a channel; a bind context waits on the CPU instead).
use nv_rm_abi::{NvRmSyncId, NvRmSyncPoint};
use thiserror::Error;

/// The payload behind a `vk_sync` NVK asks to wait on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitSync {
    /// One of this backend's sync types, with the RM payload it holds.
    Rm(NvRmSyncId),
    /// Any other `vk_sync_type`: the RM cannot acquire it.
    Foreign,
}

/// One `vk_sync_wait`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    /// The payload.
    pub sync: WaitSync,
    /// Whether the sync is a timeline (`VK_SYNC_IS_TIMELINE`).
    pub timeline: bool,
    /// The value to wait for; ignored for a binary sync.
    pub value: u64,
}

/// A wait the RM backend cannot encode.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum WaitError {
    /// The payload is not one of this backend's sync types.
    #[error("non-RM synchronization payload")]
    Foreign,
}

/// The acquire for `wait`, or `None` when there is nothing to wait for: a
/// timeline value is a `u64`, so a wait for 0 is already satisfied. That
/// check comes first, so a foreign timeline at value 0 is skipped too
/// (`nvkmd_rm.c` `rm_ctx_wait`).
///
/// # Errors
/// [`WaitError::Foreign`] for a payload the RM does not own.
pub fn acquire(wait: Wait) -> Result<Option<NvRmSyncPoint>, WaitError> {
    if wait.timeline && wait.value == 0 {
        return Ok(None);
    }
    match wait.sync {
        WaitSync::Rm(sync) => Ok(Some(NvRmSyncPoint {
            sync,
            value: wait.value,
        })),
        WaitSync::Foreign => Err(WaitError::Foreign),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait(sync: WaitSync, timeline: bool, value: u64) -> Wait {
        Wait {
            sync,
            timeline,
            value,
        }
    }

    #[test]
    fn timeline_value_zero_is_skipped_before_the_type_check() {
        // nvkmd_rm.c:1183-1184.
        let rm = WaitSync::Rm(NvRmSyncId::new(7));
        assert_eq!(acquire(wait(rm, true, 0)), Ok(None));
        assert_eq!(acquire(wait(WaitSync::Foreign, true, 0)), Ok(None));
    }

    #[test]
    fn rm_payloads_become_acquires_with_their_value() {
        // nvkmd_rm.c:1187-1189: binary syncs keep the value NVK passed (0).
        let id = NvRmSyncId::new(7);
        assert_eq!(
            acquire(wait(WaitSync::Rm(id), true, 5)),
            Ok(Some(NvRmSyncPoint { sync: id, value: 5 }))
        );
        assert_eq!(
            acquire(wait(WaitSync::Rm(id), false, 0)),
            Ok(Some(NvRmSyncPoint { sync: id, value: 0 }))
        );
    }

    #[test]
    fn foreign_payloads_are_refused() {
        // nvkmd_rm.c:1185-1186.
        assert_eq!(
            acquire(wait(WaitSync::Foreign, false, 0)),
            Err(WaitError::Foreign)
        );
        assert_eq!(
            acquire(wait(WaitSync::Foreign, true, 3)),
            Err(WaitError::Foreign)
        );
    }
}
