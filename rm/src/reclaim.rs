//! Deferred reclamation keyed by per-queue completion-token snapshots.
//!
//! Freeing memory or a VA range must not wait for GPU work: that work can be
//! gated on a foreign sync_file whose producer is the freeing thread itself
//! (Android buffer release cycle: build/fence-tracing/caller-release/
//! FINDINGS.md). Kernel drivers solve the same problem by keeping a freed
//! object alive until the fences that were outstanding at free time signal;
//! the slot pool here already does it for semaphore slots (HeldSlots in
//! gpu.rs, keyed by token). This is the same for RM mappings, memory objects
//! and VA ranges: at free time the caller records, for every queue with
//! unretired work, that queue's last submitted token; the object is
//! reclaimed once every recorded token has retired.
//!
//! Groups are FIFO. Per queue, tokens are monotonic in submission order and
//! queue indices are never reused, so if a group's snapshot has not retired,
//! no later group's has either: reclamation stops at the first unretired
//! group, costing one comparison per recorded queue. Storage is three deques
//! whose capacity is retained (no steady-state allocation on the free path).
use std::collections::VecDeque;

pub(crate) struct Reclaim<T> {
    /// (snapshot token count, item count) per group, oldest first.
    groups: VecDeque<(u32, u32)>,
    /// Flattened snapshots: (queue index, last submitted token).
    tokens: VecDeque<(u32, u64)>,
    /// Flattened items, in the order they must be reclaimed.
    items: VecDeque<T>,
    /// Snapshot tokens appended by `begin` and not yet committed.
    open: u32,
}
impl<T> Default for Reclaim<T> {
    fn default() -> Self {
        Self {
            groups: VecDeque::new(),
            tokens: VecDeque::new(),
            items: VecDeque::new(),
            open: 0,
        }
    }
}
impl<T> Reclaim<T> {
    pub(crate) fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
    #[cfg(test)]
    pub(crate) fn groups(&self) -> usize {
        self.groups.len()
    }
    /// Record the snapshot of a new group: one (queue, token) pair for every
    /// queue whose last submitted token has not retired. Returns how many
    /// were recorded; zero means nothing can still use the object and the
    /// caller reclaims it immediately (no group is opened then).
    pub(crate) fn begin(&mut self, snapshot: impl Iterator<Item = (u32, u64)>) -> u32 {
        debug_assert_eq!(self.open, 0, "reclaim group already open");
        let before = self.tokens.len();
        self.tokens.extend(snapshot);
        let count = (self.tokens.len() - before) as u32;
        self.open = count;
        count
    }
    /// Close the group opened by a non-zero `begin` with its items.
    pub(crate) fn commit(&mut self, items: impl IntoIterator<Item = T>) {
        debug_assert_ne!(self.open, 0, "commit without an open snapshot");
        let before = self.items.len();
        self.items.extend(items);
        let count = (self.items.len() - before) as u32;
        self.groups.push_back((self.open, count));
        self.open = 0;
    }
    /// Move the oldest group's items into `out` (in free order) if every
    /// token of its snapshot satisfies `retired`, or unconditionally with
    /// `force` (device close after every channel was freed).
    pub(crate) fn pop_retired(
        &mut self,
        force: bool,
        mut retired: impl FnMut(u32, u64) -> bool,
        out: &mut Vec<T>,
    ) -> bool {
        debug_assert_eq!(self.open, 0, "reclaim while a group is open");
        let Some(&(tokens, items)) = self.groups.front() else {
            return false;
        };
        if !force
            && !self
                .tokens
                .iter()
                .take(tokens as usize)
                .all(|&(queue, token)| retired(queue, token))
        {
            return false;
        }
        self.groups.pop_front();
        self.tokens.drain(..tokens as usize);
        out.extend(self.items.drain(..items as usize));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_opens_no_group() {
        let mut r = Reclaim::<u32>::default();
        assert_eq!(r.begin(std::iter::empty()), 0);
        assert!(r.is_empty());
        let mut out = Vec::new();
        assert!(!r.pop_retired(false, |_, _| true, &mut out));
    }

    #[test]
    fn groups_reclaim_in_order_only_after_every_snapshot_token_retires() {
        let mut r = Reclaim::default();
        // Queue 0 at token 5, queue 2 at token 9.
        assert_eq!(r.begin([(0, 5), (2, 9)].into_iter()), 2);
        r.commit([10, 11]);
        assert_eq!(r.begin([(0, 6)].into_iter()), 1);
        r.commit([20]);
        let mut out = Vec::new();
        let completed = std::cell::Cell::new([4u64, 0, 9]);
        let retired = |q: u32, t: u64| completed.get()[q as usize] >= t;
        assert!(!r.pop_retired(false, retired, &mut out), "queue 0 not at 5");
        completed.set([5, 0, 9]);
        assert!(r.pop_retired(false, retired, &mut out));
        assert_eq!(out, [10, 11]);
        assert!(!r.pop_retired(false, retired, &mut out), "queue 0 not at 6");
        completed.set([6, 0, 9]);
        assert!(r.pop_retired(false, retired, &mut out));
        assert_eq!(out, [10, 11, 20]);
        assert!(r.is_empty());
    }

    #[test]
    fn later_group_never_overtakes_an_unretired_earlier_one() {
        let mut r = Reclaim::default();
        r.begin([(1, 3)].into_iter());
        r.commit(["first"]);
        r.begin([(0, 1)].into_iter());
        r.commit(["second"]);
        let mut out = Vec::new();
        // Only queue 0 retired: FIFO keeps "second" behind "first".
        assert!(!r.pop_retired(false, |q, _| q == 0, &mut out));
        assert!(out.is_empty());
        assert_eq!(r.groups(), 2);
    }

    #[test]
    fn force_reclaims_everything_at_close() {
        let mut r = Reclaim::default();
        for i in 0..5u32 {
            r.begin([(0, u64::from(i) + 100)].into_iter());
            r.commit([i]);
        }
        let mut out = Vec::new();
        while r.pop_retired(true, |_, _| false, &mut out) {}
        assert_eq!(out, [0, 1, 2, 3, 4]);
        assert!(r.is_empty());
    }

    #[test]
    fn steady_state_free_path_does_not_grow_storage() {
        let mut r = Reclaim::default();
        let mut out = Vec::with_capacity(8);
        let mut capacity = None;
        for frame in 0..10_000u64 {
            // Three queues busy at free time, two items per free, reclaimed
            // two frames later (pipelined frames).
            r.begin([(0, frame), (1, frame), (3, frame)].into_iter());
            r.commit([frame, frame + 1]);
            while r.pop_retired(false, |_, t| t + 2 <= frame, &mut out) {}
            out.clear();
            if frame == 100 {
                capacity = Some((r.groups.capacity(), r.tokens.capacity(), r.items.capacity()));
            }
        }
        assert_eq!(
            capacity,
            Some((r.groups.capacity(), r.tokens.capacity(), r.items.capacity()))
        );
        assert!(r.groups() <= 3);
    }
}
