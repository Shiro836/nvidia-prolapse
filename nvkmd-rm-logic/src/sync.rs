// SPDX-License-Identifier: MIT
//! The rules of the sync payload callbacks (`vk_sync_type` `wait_many`,
//! `reserve_signal`, `move`) that are not plain pass-through to the RM: the
//! relative timeout a CPU wait hands the RM, what the RM's wait status means,
//! and how the RM's answers become the callbacks' results.
/// The infinite deadline and timeout of Vulkan's waits (`UINT64_MAX`).
pub const FOREVER: u64 = u64::MAX;

/// The relative timeout of a wait whose absolute `deadline` (Mesa's
/// monotonic clock, `os_time_get_nano`) is `deadline`: infinite stays
/// infinite, a past deadline polls once.
/// The RM's wait owns the deadline; nothing here shortens or extends it.
/// `now` reads the clock, only for a finite deadline: an infinite wait (the
/// common `UINT64_MAX` one) costs no clock read (RUST.md §14).
#[must_use]
pub fn wait_timeout(deadline: u64, now: impl FnOnce() -> u64) -> u64 {
    if deadline == FOREVER {
        return FOREVER;
    }
    // Saturating: a deadline already passed is a zero timeout.
    deadline.saturating_sub(now())
}

/// Mesa's clock reading (`int64_t os_time_get_nano()`, `CLOCK_MONOTONIC`) as
/// the unsigned time the deadlines are expressed in (`uint64_t` in the
/// `vk_sync` interface); a monotonic reading is never negative, and the
/// conversion is the bit reinterpretation C's implicit conversion performs.
#[must_use]
pub const fn clock_reading(now: i64) -> u64 {
    now.cast_unsigned()
}

/// What `nv_rm_sync_wait_many` returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitStatus {
    /// 0: the points completed (all, or any with `NV_RM_SYNC_WAIT_ANY`).
    Complete,
    /// 2: the caller's timeout ran out first (`VK_TIMEOUT`).
    Timeout,
    /// Anything else: the RM's status, which is the `VkResult` itself.
    Failed(i32),
}

impl WaitStatus {
    /// The status `nv_rm_sync_wait_many` returned (`nv_rm.h`: 0 completion,
    /// 2 caller timeout, negative failure).
    #[must_use]
    pub const fn from_rm(status: i32) -> Self {
        match status {
            0 => Self::Complete,
            2 => Self::Timeout,
            failed => Self::Failed(failed),
        }
    }
}

/// `move`'s `shared_source` argument: whether `src` is shared with another
/// process (`VK_SYNC_IS_SHARED`), as the RM's 0/1 flag.
#[must_use]
pub const fn shared_source(shared: bool) -> u32 {
    if shared { 1 } else { 0 }
}

/// `reserve_signal`'s answer: reserved only when the RM call succeeded and
/// reported the reservation.
#[must_use]
pub const fn reserved(succeeded: bool, done: u32) -> bool {
    succeeded && done != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infinite_deadlines_stay_infinite() {
        let unread = || -> u64 { panic!("an infinite wait reads no clock") };
        assert_eq!(wait_timeout(FOREVER, unread), FOREVER);
    }

    #[test]
    fn finite_deadlines_become_the_remaining_time() {
        // deadline > now ? deadline - now : 0.
        assert_eq!(wait_timeout(1_002_000_000, || 1_000_000_000), 2_000_000);
        assert_eq!(wait_timeout(1_000_000_001, || 1_000_000_000), 1);
        assert_eq!(wait_timeout(1_000_000_000, || 1_000_000_000), 0);
        assert_eq!(wait_timeout(0, || 1_000_000_000), 0);
        assert_eq!(wait_timeout(FOREVER - 1, || 0), FOREVER - 1);
    }

    #[test]
    fn clock_readings_convert_like_c() {
        assert_eq!(clock_reading(0), 0);
        assert_eq!(clock_reading(1_000_000_000), 1_000_000_000);
        assert_eq!(clock_reading(i64::MAX), FOREVER >> 1);
        assert_eq!(clock_reading(-1), FOREVER);
    }

    #[test]
    fn wait_statuses() {
        // nv_rm.h nv_rm_sync_wait_many.
        assert_eq!(WaitStatus::from_rm(0), WaitStatus::Complete);
        assert_eq!(WaitStatus::from_rm(2), WaitStatus::Timeout);
        assert_eq!(WaitStatus::from_rm(-4), WaitStatus::Failed(-4));
        assert_eq!(WaitStatus::from_rm(-13), WaitStatus::Failed(-13));
        assert_eq!(WaitStatus::from_rm(1), WaitStatus::Failed(1));
    }

    #[test]
    fn shared_sources() {
        // !!(src->flags & VK_SYNC_IS_SHARED).
        assert_eq!(shared_source(false), 0);
        assert_eq!(shared_source(true), 1);
    }

    #[test]
    fn reservations_need_success_and_the_rm_answer() {
        assert!(reserved(true, 1));
        assert!(reserved(true, 7));
        assert!(!reserved(true, 0));
        assert!(!reserved(false, 1));
        assert!(!reserved(false, 0));
    }
}
