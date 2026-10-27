// SPDX-License-Identifier: MIT
//! Per-device pool of 16-byte GPU semaphore slots in RM system memory.
//!
//! The mapped word is the only truth for ordinary synchronization: CPU status,
//! reset, host signal and GPU acquire/release all address the same slot. There
//! is no CPU shadow counter for local payloads (pre-Volta split timelines keep
//! the public 64-bit shadow next to their 32-bit GPU word, see below).
//!
//! Memory contract: pages are RM SYS allocations requested CPU-cached and
//! GPU-uncached (rm.rs allocate_aligned; public nvos.h ATTR/ATTR2), CPU-mapped
//! once for the device lifetime and GPU-mapped below the Mesa VA heap. The RM
//! owners of each page (Memory/Mapping/GpuMapping, Rc-confined) live in the
//! Bridge's locked state; this pool only records their plain addresses. Pages
//! are released only at device teardown, after every channel is freed.
//!
//! Lifetime: each slot has a reference count. One reference belongs to the
//! CPU owner (a sync payload, a gate registration or a queue retirement word);
//! every accepted GPU submission that encodes an acquire/release on the slot
//! holds one more until that submission's private retirement word is observed.
//! A slot returns to the free list only when the count reaches zero, so a slot
//! captured by in-flight GPU work can never be recycled or re-initialized.
//! Allocation zeroes all 16 bytes before the slot is handed out.
use crate::rm::{Error, Result};
use nv_rm_core::conv::{low_u32, u64_from_usize, usize_from_u32};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence},
};

pub(crate) const SLOT_BYTES: usize = 16;
/// One RM system-memory allocation per page; 4096 slots each.
pub(crate) const PAGE_BYTES: usize = 64 * 1024;
pub(crate) const SLOTS_PER_PAGE: usize = PAGE_BYTES / SLOT_BYTES;
/// 256 pages = 1M slots (16 MiB SYS). Exhaustion is a named error.
pub(crate) const MAX_PAGES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Slot(pub(crate) u32);

/// One installed page. `cpu` is the address of a PAGE_BYTES CPU mapping the
/// GPU writes concurrently (semaphore releases); it is only ever accessed
/// through the atomics `atomic32`/`atomic64` hand out (install_page's
/// contract), never through a plain reference or a non-atomic access.
struct Page {
    cpu: usize,
    gpu: u64,
    refs: Box<[AtomicU32]>,
}

/// Which word of a 16-byte slot: the GPU word at +0 or the second 64-bit
/// word at +8 (the Split32 shadow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Word {
    Low,
    High,
}

/// Byte offset of a slot word inside its page. Only `of` makes one: the slot
/// base is a multiple of SLOT_BYTES below PAGE_BYTES and the word adds 0 or
/// 8, so an 8-byte access at the offset is aligned and inside the page.
#[derive(Clone, Copy, Debug)]
struct PageOffset(usize);
const _: () = assert!(PAGE_BYTES.is_multiple_of(SLOT_BYTES) && SLOT_BYTES == 16);
impl PageOffset {
    #[inline]
    fn of(slot: Slot, word: Word) -> Self {
        let within = match word {
            Word::Low => 0,
            Word::High => 8,
        };
        Self((usize_from_u32(slot.0) % SLOTS_PER_PAGE) * SLOT_BYTES + within)
    }
}

impl Page {
    #[inline]
    fn atomic32(&self, at: PageOffset) -> &AtomicU32 {
        // SAFETY: `at` is 8-aligned and at + 8 <= PAGE_BYTES (PageOffset), and
        // install_page's contract keeps `cpu..cpu + PAGE_BYTES` mapped and
        // 4096-aligned until the pool is closed; the pool outlives `&self`.
        // Atomics have interior mutability, so the GPU's concurrent writes do
        // not conflict with the shared reference.
        unsafe { &*std::ptr::with_exposed_provenance::<AtomicU32>(self.cpu + at.0) }
    }
    #[inline]
    fn atomic64(&self, at: PageOffset) -> &AtomicU64 {
        // SAFETY: as atomic32; `at` is 8-byte aligned.
        unsafe { &*std::ptr::with_exposed_provenance::<AtomicU64>(self.cpu + at.0) }
    }
}

struct FreeList {
    free: Vec<u32>,
    pages: usize,
}

/// Timeline representation of a slot.
///
/// Native64: one 64-bit GPU-visible word at +0, read with a real 64-bit atomic
/// (cmpxchg8b/SSE on i686 via AtomicU64), released/acquired with 64-bit host
/// SEM_EXECUTE (public clc36f+ PAYLOAD_SIZE_64BIT, ACQ_STRICT_GEQ).
///
/// Split32 (pre-Volta hosts a06f/a16f/b06f/c06f have only 32-bit SEMAPHORE[A-D]):
/// GPU word = low 32 bits at +0, 64-bit shadow at +8 = maximum value ever
/// submitted for signalling or host-signalled. The shadow is raised (CAS-max)
/// BEFORE any GPU release of its low word can execute, so a reader that reads
/// the low word first and the shadow second always reconstructs the epoch
/// correctly while fewer than 2^31 values are outstanding (same rule as the
/// public RM sem_surf.c reconstruction used for the queue sem32 surface).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimelineView {
    Native64,
    Split32,
}

pub(crate) struct SlotPool {
    pages: Box<[OnceLock<Page>]>,
    free: Mutex<FreeList>,
    closed: AtomicBool,
    pub(crate) view: TimelineView,
}

/// Reconstruct a full-width value from its GPU low word and the 64-bit shadow
/// (maximum submitted). Identical arithmetic to fences::extend_payload.
pub(crate) fn split_value(low: u32, shadow: u64) -> Result<u64> {
    crate::fences::extend_payload(low, shadow)
}

impl SlotPool {
    pub(crate) fn new(view: TimelineView) -> Self {
        Self {
            pages: (0..MAX_PAGES).map(|_| OnceLock::new()).collect(),
            free: Mutex::new(FreeList {
                free: Vec::new(),
                pages: 0,
            }),
            closed: AtomicBool::new(false),
            view,
        }
    }

    /// Next page index to install, or None at the hard limit.
    pub(crate) fn next_page(&self) -> Option<usize> {
        let free = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (free.pages < MAX_PAGES).then_some(free.pages)
    }

    pub(crate) fn has_free(&self) -> bool {
        !self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .free
            .is_empty()
    }

    /// Register a mapped page. `cpu` must stay mapped and `gpu` must stay
    /// GPU-mapped until close(); both must be 16-byte aligned PAGE_BYTES ranges.
    ///
    /// # Safety
    /// `cpu..cpu + PAGE_BYTES` is a readable and writable CPU mapping that
    /// stays mapped until close() and that nothing accesses except through
    /// this pool's atomics (the GPU through `gpu`) while installed.
    pub(crate) unsafe fn install_page(&self, cpu: usize, gpu: u64) -> Result<()> {
        if cpu == 0 || !cpu.is_multiple_of(4096) || !gpu.is_multiple_of(4096) {
            return Err(Error::Contract("unaligned semaphore pool page".into()));
        }
        let mut free = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = free.pages;
        if index >= MAX_PAGES {
            return Err(Error::Contract(
                "semaphore slot pool exhausted (1M live slots)".into(),
            ));
        }
        let page = Page {
            cpu,
            gpu,
            refs: (0..SLOTS_PER_PAGE).map(|_| AtomicU32::new(0)).collect(),
        };
        if self.pages[index].set(page).is_err() {
            return Err(Error::Contract(
                "semaphore pool page installed twice".into(),
            ));
        }
        free.pages += 1;
        let slots = |page: usize| {
            page.checked_mul(SLOTS_PER_PAGE)
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| {
                    Error::Contract(format!("semaphore pool page {index} beyond u32 slot ids"))
                })
        };
        let (base, end) = (slots(index)?, slots(index + 1)?);
        // Hand out low indices first (LIFO pop takes from the end).
        free.free.reserve(SLOTS_PER_PAGE);
        free.free.extend((base..end).rev());
        Ok(())
    }

    #[inline]
    fn page(&self, slot: Slot) -> &Page {
        let index = usize_from_u32(slot.0) / SLOTS_PER_PAGE;
        self.pages[index]
            .get()
            .expect("slot index refers to an installed page")
    }
    #[inline]
    fn offset(slot: Slot) -> usize {
        (usize_from_u32(slot.0) % SLOTS_PER_PAGE) * SLOT_BYTES
    }

    /// Pop a zeroed slot with one reference, or None if a new page is needed.
    pub(crate) fn try_allocate(&self) -> Option<Slot> {
        let slot = Slot(
            self.free
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .free
                .pop()?,
        );
        let page = self.page(slot);
        let refs = &page.refs[usize_from_u32(slot.0) % SLOTS_PER_PAGE];
        debug_assert_eq!(refs.load(Ordering::Relaxed), 0);
        refs.store(1, Ordering::Relaxed);
        // No GPU reference exists (count was zero: every submission that
        // touched it has retired), so plain initialization cannot race a
        // GPU write. Order it before the slot address escapes.
        page.atomic64(PageOffset::of(slot, Word::Low))
            .store(0, Ordering::Relaxed);
        page.atomic64(PageOffset::of(slot, Word::High))
            .store(0, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        Some(slot)
    }

    pub(crate) fn retain(&self, slot: Slot) {
        let previous = self.page(slot).refs[usize_from_u32(slot.0) % SLOTS_PER_PAGE]
            .fetch_add(1, Ordering::Relaxed);
        debug_assert!(previous != 0, "retain of a free semaphore slot");
    }

    /// Drop one reference; the last one returns the slot to the free list.
    pub(crate) fn release(&self, slot: Slot) {
        let refs = &self.page(slot).refs[usize_from_u32(slot.0) % SLOTS_PER_PAGE];
        let previous = refs.fetch_sub(1, Ordering::Release);
        debug_assert!(previous != 0, "release of a free semaphore slot");
        if previous == 1 {
            // Synchronize with every other owner's final use (Release above).
            fence(Ordering::Acquire);
            self.free
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .free
                .push(slot.0);
        }
    }

    pub(crate) fn refs(&self, slot: Slot) -> u32 {
        self.page(slot).refs[usize_from_u32(slot.0) % SLOTS_PER_PAGE].load(Ordering::Acquire)
    }

    pub(crate) fn gpu(&self, slot: Slot) -> u64 {
        self.page(slot).gpu + u64_from_usize(Self::offset(slot))
    }

    /// Device teardown: no CPU access to page memory is valid afterwards.
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
    #[inline]
    fn live(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(Error::DeviceLost(
                "synchronization payload used after device close".into(),
            ))
        } else {
            Ok(())
        }
    }
    #[inline]
    fn word32(&self, slot: Slot) -> &AtomicU32 {
        self.page(slot).atomic32(PageOffset::of(slot, Word::Low))
    }
    #[inline]
    fn word64(&self, slot: Slot, word: Word) -> &AtomicU64 {
        self.page(slot).atomic64(PageOffset::of(slot, word))
    }

    /// Binary payload: 1 = signalled. GPU consume writes 0 in its own stream.
    #[inline]
    pub(crate) fn read32(&self, slot: Slot) -> Result<u32> {
        self.live()?;
        let value = self.word32(slot).load(Ordering::Acquire);
        Ok(value)
    }
    /// CPU binary write (reset 0 / signal 1 / gate release 1). Callers prove
    /// no GPU write to this slot is outstanding (see SyncObj::reset).
    pub(crate) fn store32(&self, slot: Slot, value: u32) -> Result<()> {
        self.live()?;
        self.word32(slot).store(value, Ordering::SeqCst);
        // Drain the store buffer toward the coherent GPU snoop path.
        crate::platform::device_write_barrier();
        Ok(())
    }

    /// Raw 64-bit word at +0 regardless of the timeline view (queue retirement
    /// words released with 64-bit host semaphores). Real 64-bit atomic load.
    pub(crate) fn read64_raw(&self, slot: Slot) -> Result<u64> {
        self.live()?;
        Ok(self.word64(slot, Word::Low).load(Ordering::Acquire))
    }

    /// Full-width timeline value of the slot (see TimelineView).
    pub(crate) fn timeline_value(&self, slot: Slot) -> Result<u64> {
        self.live()?;
        match self.view {
            TimelineView::Native64 => Ok(self.word64(slot, Word::Low).load(Ordering::Acquire)),
            TimelineView::Split32 => {
                // Low word strictly before the shadow (see TimelineView).
                let low = self.word32(slot).load(Ordering::Acquire);
                let shadow = self.word64(slot, Word::High).load(Ordering::Acquire);
                split_value(low, shadow)
            }
        }
    }
    /// Initial value for a fresh timeline slot (no GPU reference exists).
    pub(crate) fn init_timeline(&self, slot: Slot, value: u64) -> Result<()> {
        self.live()?;
        match self.view {
            TimelineView::Native64 => self.word64(slot, Word::Low).store(value, Ordering::SeqCst),
            TimelineView::Split32 => {
                self.word64(slot, Word::High).store(value, Ordering::SeqCst);
                self.word32(slot).store(low_u32(value), Ordering::SeqCst);
            }
        }
        crate::platform::device_write_barrier();
        Ok(())
    }
    /// Host timeline signal: monotonic CAS-max of the mapped word. Returns
    /// whether the value increased. Vulkan requires a host-signalled value to
    /// exceed the current value and every pending signal, so the unconditional
    /// low-word store of the split view cannot regress a valid timeline.
    pub(crate) fn host_signal(&self, slot: Slot, value: u64) -> Result<bool> {
        self.live()?;
        let changed = match self.view {
            TimelineView::Native64 => {
                self.word64(slot, Word::Low)
                    .fetch_max(value, Ordering::SeqCst)
                    < value
            }
            TimelineView::Split32 => {
                // Reconstruct with the shadow as it was, never after raising it:
                // a jump >= 2^31 cannot be represented by the 32-bit GPU word.
                let low = self.word32(slot).load(Ordering::Acquire);
                let shadow = self.word64(slot, Word::High).load(Ordering::Acquire);
                let current = split_value(low, shadow)?;
                if value <= current {
                    false
                } else if value - current >= 0x8000_0000 {
                    return Err(Error::Contract(format!(
                        "32-bit host semaphore cannot represent timeline step {current} -> {value} (2^31 window)"
                    )));
                } else {
                    // Shadow first (readers read low then shadow), then the
                    // GPU-visible low word. A pending higher GPU signal keeps
                    // its larger shadow (VUID-VkSemaphoreSignalInfo-value-03259).
                    self.word64(slot, Word::High)
                        .fetch_max(value, Ordering::SeqCst);
                    self.word32(slot).store(low_u32(value), Ordering::SeqCst);
                    true
                }
            }
        };
        crate::platform::device_write_barrier();
        Ok(changed)
    }
    /// Split view only: raise the shadow before a GPU release of `value` can
    /// execute (called after the batch is accepted, before the doorbell).
    pub(crate) fn raise_shadow(&self, slot: Slot, value: u64) -> Result<()> {
        self.live()?;
        if self.view == TimelineView::Split32 {
            self.word64(slot, Word::High)
                .fetch_max(value, Ordering::SeqCst);
        }
        Ok(())
    }
    /// Split view: the circular 32-bit compare encoded by the GPU acquire is
    /// only correct while target and every reachable value stay within 2^31.
    pub(crate) fn check_split_distance(&self, slot: Slot, target: u64) -> Result<()> {
        if self.view == TimelineView::Split32 {
            let shadow = self.word64(slot, Word::High).load(Ordering::Acquire);
            if target.abs_diff(shadow) >= 0x8000_0000 {
                return Err(Error::Contract(format!(
                    "32-bit host semaphore cannot order timeline wait {target} against pending {shadow} (2^31 window)"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;

    /// CPU-only page standing in for an RM SYS mapping (same 16-byte layout).
    #[repr(C, align(4096))]
    pub(crate) struct HostPage([u64; PAGE_BYTES / 8]);
    /// Owns the pages a test pool maps. They are held as the raw pointers
    /// `Box::into_raw` returned, never as `Box`es: moving a `Box` asserts
    /// unique access, which would revoke the provenance the pool uses.
    pub(crate) struct HostPages(Vec<*mut HostPage>);
    impl HostPages {
        pub(crate) fn pool(view: TimelineView, pages: usize) -> (Arc<SlotPool>, Self) {
            let pool = Arc::new(SlotPool::new(view));
            let mut owned = Self(Vec::new());
            for i in 0..pages {
                // SAFETY: an all-zero u64 array is a valid HostPage.
                let page: Box<HostPage> = unsafe { Box::new_zeroed().assume_init() };
                let page = Box::into_raw(page);
                owned.0.push(page);
                let address = page.expose_provenance();
                // SAFETY: the page stays allocated until the returned
                // HostPages drops, and every test keeps it alongside the
                // pool; only the pool's atomics access it.
                unsafe { pool.install_page(address, 0x1000_0000 + u64_from_usize(i * PAGE_BYTES)) }
                    .unwrap();
            }
            (pool, owned)
        }
    }
    impl Drop for HostPages {
        fn drop(&mut self) {
            for &page in &self.0 {
                // SAFETY: each pointer came from Box::into_raw in `pool` and
                // is freed exactly once, here.
                drop(unsafe { Box::from_raw(page) });
            }
        }
    }
    impl SlotPool {
        /// Test stand-in for a GPU 32-bit release (no CPU ordering claims).
        pub(crate) fn gpu_write32(&self, slot: Slot, value: u32) {
            self.word32(slot).store(value, Ordering::SeqCst);
        }
        pub(crate) fn gpu_write64(&self, slot: Slot, value: u64) {
            self.word64(slot, Word::Low).store(value, Ordering::SeqCst);
        }
    }

    #[test]
    fn slot_lifetime_requires_cpu_and_every_gpu_reference() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let a = pool.try_allocate().unwrap();
        pool.store32(a, 1).unwrap();
        // Two in-flight GPU submissions capture the slot.
        pool.retain(a);
        pool.retain(a);
        pool.release(a); // CPU owner gone (handle destroyed/reset away)
        assert_eq!(pool.refs(a), 2);
        // Not recyclable: the next allocations must be different slots.
        let mut seen = vec![];
        for _ in 0..64 {
            let s = pool.try_allocate().unwrap();
            assert_ne!(s, a);
            seen.push(s);
        }
        pool.release(a); // first submission retired
        assert!(
            pool.read32(a).unwrap() == 1,
            "not re-initialized while captured"
        );
        pool.release(a); // last submission retired → freed
        assert_eq!(pool.refs(a), 0);
        let again = pool.try_allocate().unwrap();
        assert_eq!(again, a, "LIFO reuse after final release");
        assert_eq!(pool.read32(again).unwrap(), 0, "zeroed on allocation");
        for s in seen {
            pool.release(s);
        }
    }

    #[test]
    fn exhaustion_and_close_are_named_errors() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let all: Vec<_> = (0..SLOTS_PER_PAGE)
            .map(|_| pool.try_allocate().unwrap())
            .collect();
        assert!(pool.try_allocate().is_none());
        pool.release(all[7]);
        assert_eq!(pool.try_allocate(), Some(all[7]));
        pool.close();
        assert!(matches!(pool.read32(all[0]), Err(Error::DeviceLost(_))));
        assert!(pool.store32(all[0], 1).is_err());
        assert!(pool.timeline_value(all[0]).is_err());
    }

    #[test]
    fn native64_host_signal_is_monotonic_full_width() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 1);
        let s = pool.try_allocate().unwrap();
        pool.init_timeline(s, 0xffff_fffe).unwrap();
        assert!(pool.host_signal(s, 0x1_0000_0001).unwrap());
        assert_eq!(pool.timeline_value(s).unwrap(), 0x1_0000_0001);
        assert!(!pool.host_signal(s, 5).unwrap());
        assert_eq!(pool.timeline_value(s).unwrap(), 0x1_0000_0001);
        pool.gpu_write64(s, 0x2_0000_0000);
        assert_eq!(pool.timeline_value(s).unwrap(), 0x2_0000_0000);
    }

    #[test]
    fn split_view_wraps_epochs_and_rejects_out_of_window_waits() {
        let (pool, _pages) = HostPages::pool(TimelineView::Split32, 1);
        let s = pool.try_allocate().unwrap();
        pool.init_timeline(s, 0xffff_fff0).unwrap();
        assert_eq!(pool.timeline_value(s).unwrap(), 0xffff_fff0);
        // GPU signal of a wrapped value: shadow raised at encode, then the
        // GPU writes only the low word. Before the write the old value reads.
        pool.raise_shadow(s, 0x1_0000_0005).unwrap();
        assert_eq!(pool.timeline_value(s).unwrap(), 0xffff_fff0);
        pool.gpu_write32(s, 0x0000_0005);
        assert_eq!(pool.timeline_value(s).unwrap(), 0x1_0000_0005);
        // Host signals walk across another epoch in representable steps.
        assert!(pool.host_signal(s, 0x1_7000_0000).unwrap());
        assert!(pool.host_signal(s, 0x1_efff_fff0).unwrap());
        assert!(pool.host_signal(s, 0x2_0000_0001).unwrap());
        assert_eq!(pool.timeline_value(s).unwrap(), 0x2_0000_0001);
        assert!(
            !pool.host_signal(s, 0x2_0000_0001).unwrap(),
            "equal is quiet"
        );
        // A single step of 2^31 or more is a named error and changes nothing.
        assert!(pool.host_signal(s, 0x2_8000_0001).is_err());
        assert_eq!(pool.timeline_value(s).unwrap(), 0x2_0000_0001);
        // A host signal below a pending (higher, submitted) GPU signal keeps
        // the larger shadow, then the GPU word reaches the pending value.
        pool.raise_shadow(s, 0x2_0000_0100).unwrap();
        assert!(pool.host_signal(s, 0x2_0000_0010).unwrap());
        assert_eq!(pool.timeline_value(s).unwrap(), 0x2_0000_0010);
        pool.gpu_write32(s, 0x0000_0100);
        assert_eq!(pool.timeline_value(s).unwrap(), 0x2_0000_0100);
        // Waits within the 2^31 window are encodable; beyond it they are not.
        pool.check_split_distance(s, 0x2_7fff_ffff).unwrap();
        assert!(pool.check_split_distance(s, 0x2_8000_0100).is_err());
        assert!(pool.check_split_distance(s, 0x1_8000_0001).is_err());
        // A corrupt distance (low far behind shadow) is detected.
        pool.raise_shadow(s, 0x2_9000_0000).unwrap();
        assert!(pool.timeline_value(s).is_err());
    }

    #[test]
    fn concurrent_allocation_and_release_never_duplicate_live_slots() {
        let (pool, _pages) = HostPages::pool(TimelineView::Native64, 2);
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    let mut live = Vec::new();
                    for i in 0..20_000u32 {
                        if i % 3 != 2 {
                            let s = pool.try_allocate().unwrap();
                            // A live slot's count is exactly ours.
                            assert_eq!(pool.refs(s), 1);
                            pool.store32(s, i).unwrap();
                            live.push((s, i));
                        } else if let Some((s, v)) = live.pop() {
                            assert_eq!(pool.read32(s).unwrap(), v, "slot reused while live");
                            pool.release(s);
                        }
                        if live.len() > 200 {
                            for (s, v) in live.drain(..) {
                                assert_eq!(pool.read32(s).unwrap(), v);
                                pool.release(s);
                            }
                        }
                    }
                    for (s, _) in live {
                        pool.release(s);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let free = pool.free.lock().unwrap();
        assert_eq!(free.free.len(), 2 * SLOTS_PER_PAGE);
        let mut unique = free.free.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), free.free.len());
    }
}
