//! Owned CPU mappings are lazy and may survive logical unmap until allocation
//! free. Idle leases are reclaimable under CPU-VA/RM/BAR pressure; active leases
//! are never indexed. One weak FIFO gives O(1) transitions without an Rc cycle.
use crate::rm::{Error, Result};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

pub(crate) trait Lease {
    fn retain_idle(&self) -> bool;
    fn release(&mut self) -> Result<()>;
}
struct Entry<M> {
    mapping: Option<M>,
    refs: u32,
    failed: bool,
}
struct Links<M> {
    previous: Weak<Slot<M>>,
    next: Weak<Slot<M>>,
    linked: bool,
}
impl<M> Default for Links<M> {
    fn default() -> Self {
        Self {
            previous: Weak::new(),
            next: Weak::new(),
            linked: false,
        }
    }
}
pub(crate) struct Slot<M> {
    entry: RefCell<Entry<M>>,
    links: RefCell<Links<M>>,
    registry: Weak<Registry<M>>,
    identity: Weak<Self>,
}
impl<M> Slot<M> {
    fn unlink(&self) {
        let links = std::mem::take(&mut *self.links.borrow_mut());
        if !links.linked {
            return;
        }
        // Drop unlinks before dropping Mapping, which can release the final
        // Device/registry owner. Weak neighbors never extend buffer lifetime.
        if let Some(registry) = self.registry.upgrade() {
            match links.previous.upgrade() {
                Some(previous) => {
                    previous.links.borrow_mut().next = links.next.clone();
                }
                _ => {
                    registry.ends.borrow_mut().head = links.next.clone();
                }
            }
            match links.next.upgrade() {
                Some(next) => {
                    next.links.borrow_mut().previous = links.previous;
                }
                _ => {
                    registry.ends.borrow_mut().tail = links.previous;
                }
            }
        }
    }
    fn enqueue_idle(&self) {
        assert!(!self.links.borrow().linked, "idle CPU map already indexed");
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let tail = registry.ends.borrow().tail.clone();
        match tail.upgrade() {
            Some(previous) => {
                previous.links.borrow_mut().next = self.identity.clone();
            }
            _ => {
                registry.ends.borrow_mut().head = self.identity.clone();
            }
        }
        *self.links.borrow_mut() = Links {
            previous: tail,
            next: Weak::new(),
            linked: true,
        };
        registry.ends.borrow_mut().tail = self.identity.clone();
    }
}
impl<M> Drop for Slot<M> {
    fn drop(&mut self) {
        self.unlink();
    }
}
impl<M: Lease> Slot<M> {
    pub fn acquire<T>(
        &self,
        make: impl FnOnce() -> Result<M>,
        address: impl FnOnce(&M) -> T,
    ) -> Result<T> {
        if self.entry.borrow().failed {
            return Err(Error::DeviceLost(
                "CPU mapping slot has incomplete cleanup".into(),
            ));
        }
        let refs = self
            .entry
            .borrow()
            .refs
            .checked_add(1)
            .ok_or_else(|| Error::Contract("CPU mapping reference overflow".into()))?;
        // Remove an idle lease before it can be exposed to a caller. No registry
        // or slot borrow crosses make: it can reclaim another idle mapping.
        self.unlink();
        if self.entry.borrow().mapping.is_none() {
            let mapping = make()?;
            self.entry.borrow_mut().mapping = Some(mapping);
        }
        let mut entry = self.entry.borrow_mut();
        entry.refs = refs;
        Ok(address(entry.mapping.as_ref().unwrap()))
    }
    pub fn release(&self) -> Result<()> {
        let mut entry = self.entry.borrow_mut();
        let refs = entry
            .refs
            .checked_sub(1)
            .ok_or_else(|| Error::Contract("unmap RM memory without a CPU map".into()))?;
        let mapping = entry
            .mapping
            .as_mut()
            .ok_or_else(|| Error::Contract("CPU map reference has no owned mapping".into()))?;
        if refs == 0 && !mapping.retain_idle() {
            if let Err(error) = mapping.release() {
                entry.failed = true;
                return Err(error);
            }
            entry.mapping.take();
            entry.failed = false;
        }
        // Failed cleanup preserves the original reference and partial lease.
        entry.refs = refs;
        let idle = refs == 0 && entry.mapping.is_some();
        drop(entry);
        if idle {
            self.enqueue_idle();
        }
        Ok(())
    }
    pub fn refs(&self) -> u32 {
        self.entry.borrow().refs
    }
    pub fn evict(&self) -> Result<bool> {
        let mut entry = self.entry.borrow_mut();
        if entry.refs != 0 {
            return Ok(false);
        }
        let Some(mapping) = entry.mapping.as_mut() else {
            return Ok(false);
        };
        if let Err(error) = mapping.release() {
            entry.failed = true;
            return Err(error);
        }
        entry.mapping.take();
        entry.failed = false;
        drop(entry);
        self.unlink();
        Ok(true)
    }
}
struct Ends<M> {
    head: Weak<Slot<M>>,
    tail: Weak<Slot<M>>,
}
pub(crate) struct Registry<M> {
    ends: RefCell<Ends<M>>,
}
impl<M> Default for Registry<M> {
    fn default() -> Self {
        Self {
            ends: RefCell::new(Ends {
                head: Weak::new(),
                tail: Weak::new(),
            }),
        }
    }
}
impl<M: Lease> Registry<M> {
    pub fn slot(self: &Rc<Self>) -> Rc<Slot<M>> {
        Rc::new_cyclic(|identity| Slot {
            entry: RefCell::new(Entry {
                mapping: None,
                refs: 0,
                failed: false,
            }),
            links: RefCell::new(Links::default()),
            registry: Rc::downgrade(self),
            identity: identity.clone(),
        })
    }
    pub fn evict_one(&self) -> Result<bool> {
        // O(1), no allocation/scan or registry borrow across RM cleanup. A
        // failed head remains owned/indexed; Device poison prevents reuse.
        let head = self.ends.borrow().head.upgrade();
        match head {
            Some(slot) => slot.evict(),
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    struct Fake {
        keep: bool,
        released: Rc<Cell<usize>>,
        fail: Rc<Cell<bool>>,
        _owner: Rc<Registry<Fake>>,
    }
    impl Lease for Fake {
        fn retain_idle(&self) -> bool {
            self.keep
        }
        fn release(&mut self) -> Result<()> {
            if self.fail.get() {
                return Err(Error::Contract("injected RM unmap failure".into()));
            }
            self.released.set(self.released.get() + 1);
            Ok(())
        }
    }
    fn acquire(
        slot: &Slot<Fake>,
        owner: &Rc<Registry<Fake>>,
        released: &Rc<Cell<usize>>,
        fail: &Rc<Cell<bool>>,
        keep: bool,
    ) {
        slot.acquire(
            || {
                Ok(Fake {
                    keep,
                    released: Rc::clone(released),
                    fail: Rc::clone(fail),
                    _owner: Rc::clone(owner),
                })
            },
            |_| (),
        )
        .unwrap();
    }
    #[test]
    fn reuse_protects_overlapping_logical_maps_and_evicts_only_idle() {
        let registry = Rc::new(Registry::default());
        let slot = registry.slot();
        let released = Rc::new(Cell::new(0));
        let fail = Rc::new(Cell::new(false));
        acquire(&slot, &registry, &released, &fail, true);
        slot.acquire(|| panic!("overlapping map must reuse the lease"), |_| ())
            .unwrap();
        slot.release().unwrap();
        assert_eq!(slot.refs(), 1);
        assert!(!registry.evict_one().unwrap());
        slot.release().unwrap();
        assert_eq!(released.get(), 0);
        slot.acquire(|| panic!("idle remap must reuse the lease"), |_| ())
            .unwrap();
        assert!(!registry.evict_one().unwrap());
        slot.release().unwrap();
        assert!(registry.evict_one().unwrap());
        assert_eq!(released.get(), 1);
        assert!(!registry.evict_one().unwrap());
        assert!(slot.release().is_err());
        acquire(&slot, &registry, &released, &fail, true);
        assert_eq!(slot.refs(), 1);
    }
    #[test]
    fn special_maps_release_at_last_reference_and_failures_keep_ownership() {
        let registry = Rc::new(Registry::default());
        let slot = registry.slot();
        let released = Rc::new(Cell::new(0));
        let fail = Rc::new(Cell::new(true));
        acquire(&slot, &registry, &released, &fail, false);
        assert!(slot.release().is_err());
        assert_eq!(slot.refs(), 1);
        assert!(slot.entry.borrow().mapping.is_some());
        assert!(matches!(
            slot.acquire(
                || panic!("failed slot must not remap"),
                |_| panic!("failed slot must not expose pointer")
            ),
            Err(Error::DeviceLost(_))
        ));
        assert!(!registry.evict_one().unwrap());
        fail.set(false);
        slot.release().unwrap();
        assert_eq!(released.get(), 1);
        assert!(slot.entry.borrow().mapping.is_none());
        acquire(&slot, &registry, &released, &fail, true);
        slot.release().unwrap();
        fail.set(true);
        assert!(registry.evict_one().is_err());
        assert_eq!(slot.refs(), 0);
        assert!(slot.entry.borrow().mapping.is_some());
        assert!(matches!(
            slot.acquire(
                || panic!("failed idle slot must not remap"),
                |_| panic!("partially unmapped idle pointer must not escape")
            ),
            Err(Error::DeviceLost(_))
        ));
        fail.set(false);
        assert!(registry.evict_one().unwrap());
        assert_eq!(released.get(), 2);
    }
    #[test]
    fn weak_registry_cannot_keep_device_or_stale_buffer_alive() {
        let registry = Rc::new(Registry::default());
        let owner = Rc::downgrade(&registry);
        let slot = registry.slot();
        let released = Rc::new(Cell::new(0));
        let fail = Rc::new(Cell::new(false));
        acquire(&slot, &registry, &released, &fail, true);
        slot.release().unwrap();
        let weak_slot = Rc::downgrade(&slot);
        drop(registry); // Cached mapping owns Device/registry, exactly as production.
        assert!(owner.upgrade().is_some());
        drop(slot);
        assert!(weak_slot.upgrade().is_none());
        assert!(
            owner.upgrade().is_none(),
            "weak index must not form an Rc cycle"
        );
    }
    #[test]
    fn failed_map_and_reference_overflow_do_not_publish_a_logical_map() {
        let registry = Rc::new(Registry::<Fake>::default());
        let slot = registry.slot();
        assert!(
            slot.acquire(
                || Err(Error::Contract("injected map failure".into())),
                |_| ()
            )
            .is_err()
        );
        assert_eq!(slot.refs(), 0);
        assert!(slot.entry.borrow().mapping.is_none());
        slot.entry.borrow_mut().refs = u32::MAX;
        assert!(
            slot.acquire(|| panic!("overflow must reject before allocation"), |_| ())
                .is_err()
        );
        assert_eq!(slot.refs(), u32::MAX);
    }
    #[test]
    fn fifo_reacquire_and_explicit_free_preserve_other_idle_leases() {
        let registry = Rc::new(Registry::default());
        let slots: Vec<_> = (0..4).map(|_| registry.slot()).collect();
        let counters: Vec<_> = (0..4).map(|_| Rc::new(Cell::new(0))).collect();
        let fail = Rc::new(Cell::new(false));
        for (slot, counter) in slots.iter().zip(&counters) {
            acquire(slot, &registry, counter, &fail, true);
            slot.release().unwrap();
        }
        // Reacquire the head and a middle lease: neither may be reclaimed.
        for i in [0, 2] {
            slots[i]
                .acquire(|| panic!("retained lease lost"), |_| ())
                .unwrap();
        }
        assert!(registry.evict_one().unwrap());
        assert_eq!(counters[1].get(), 1);
        // Explicit free of the tail must repair both ends of the list.
        assert!(slots[3].evict().unwrap());
        assert!(!registry.evict_one().unwrap());
        slots[2].release().unwrap();
        slots[0].release().unwrap();
        assert!(registry.evict_one().unwrap());
        assert_eq!(counters[2].get(), 1);
        assert_eq!(counters[0].get(), 0);
        assert!(registry.evict_one().unwrap());
        assert_eq!(counters[0].get(), 1);
        assert!(!registry.evict_one().unwrap());
    }
    #[test]
    fn dropping_idle_head_middle_and_tail_does_not_sever_fifo() {
        for removed in 0..3 {
            let registry = Rc::new(Registry::default());
            let mut slots: Vec<_> = (0..3).map(|_| Some(registry.slot())).collect();
            let counters: Vec<_> = (0..3).map(|_| Rc::new(Cell::new(0))).collect();
            let fail = Rc::new(Cell::new(false));
            for (slot, counter) in slots.iter().zip(&counters) {
                let slot = slot.as_ref().unwrap();
                acquire(slot, &registry, counter, &fail, true);
                slot.release().unwrap();
            }
            let weak = Rc::downgrade(slots[removed].as_ref().unwrap());
            drop(slots[removed].take());
            assert!(weak.upgrade().is_none());
            for (index, counter) in counters.iter().enumerate() {
                if index != removed {
                    assert!(registry.evict_one().unwrap());
                    assert_eq!(counter.get(), 1);
                }
            }
            assert!(!registry.evict_one().unwrap());
            // An empty queue remains reusable after its former endpoints die.
            let extra = registry.slot();
            acquire(&extra, &registry, &counters[removed], &fail, true);
            extra.release().unwrap();
            assert!(registry.evict_one().unwrap());
        }
    }
    #[test]
    fn mapping_under_pressure_can_reclaim_other_slots_without_borrow_or_active_eviction() {
        let registry = Rc::new(Registry::default());
        let active = registry.slot();
        let idle = registry.slot();
        let incoming = registry.slot();
        let released = Rc::new(Cell::new(0));
        let fail = Rc::new(Cell::new(false));
        acquire(&active, &registry, &released, &fail, true);
        acquire(&idle, &registry, &released, &fail, true);
        idle.release().unwrap();
        incoming
            .acquire(
                || {
                    assert!(registry.evict_one()?);
                    assert_eq!(released.get(), 1);
                    assert!(!registry.evict_one()?);
                    Ok(Fake {
                        keep: true,
                        released: Rc::clone(&released),
                        fail: Rc::clone(&fail),
                        _owner: Rc::clone(&registry),
                    })
                },
                |_| (),
            )
            .unwrap();
        assert_eq!(active.refs(), 1);
        assert_eq!(incoming.refs(), 1);
        assert_eq!(idle.refs(), 0);
    }
}
