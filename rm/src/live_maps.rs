// SPDX-License-Identifier: MIT
//! The RM CPU mappings live on one client: every NVOS33 whose NVOS34 has not run, with the
//! NVOS34 parameters that release it.
//!
//! A CPU mapping is released before anything it depends on is freed: RM unmap (NVOS34), then
//! `munmap`, then the RM free of the memory, its parent, its device or the client. That is the
//! order the NVIDIA driver frees host-visible memory in, also from its exit handler, and RM's
//! BAR1 mapping-reuse cache relies on it: destroying video memory that is still BAR1-mapped
//! drops its reuse submap as if it were empty (open-gpu-kernel-modules
//! `src/nvidia/src/kernel/gpu/bus/arch/maxwell/kern_bus_gm107.c` `_kbusDestroyMemdescBar1Cb`), and
//! a later mapping that hits a stale entry gives the CPU other pages than the GPU uses.
//!
//! Two parties release a mapping: its owner (the lease in `rm.rs`, on the client's thread) and
//! the exit handler (`exit.rs`, any thread). Whoever takes the entry out of the list issues its
//! NVOS34; the other finds it gone and skips.
//!
//! Locking: one mutex per client, taken by nothing else while it is held (no other lock is
//! acquired under it). The owner holds it across its own NVOS33, UPDATE_DEVICE_MAPPING_INFO and
//! NVOS34, so the exit handler never misses a mapping RM holds, never unmaps one under a key RM
//! no longer uses, and never frees the client while an owner's NVOS34 is still in flight.
//! These are RM ioctls that do not wait on the GPU, and the client is single-threaded (`Rc`),
//! so only the exit handler ever contends for the lock. The exit handler closes the list and
//! takes every entry under the lock, then issues the NVOS34s after releasing it. A poisoned
//! lock fails every operation on the list (RUST.md §8).
use crate::abi;
use crate::rm::{Error, MemoryError, Result, RmHandle, SyncError};
use std::{
    collections::BTreeMap,
    sync::{Mutex, MutexGuard},
};

/// A mapping's entry in its client's [`LiveMaps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MapId(u64);

/// One live RM CPU mapping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LiveMap {
    /// The NVOS34 that releases it. RM keys the mapping by `linear_address`: NVOS33's
    /// returned address, or for video memory the CPU address UPDATE_DEVICE_MAPPING_INFO set.
    pub(crate) unmap: abi::UnmapMemory,
    /// The mapped object's parent.
    pub(crate) parent: RmHandle,
}

#[derive(Debug, Default)]
struct State {
    /// The exit handler took the list: no new mapping is made on this client.
    closed: bool,
    next: u64,
    maps: BTreeMap<u64, LiveMap>,
}

/// The live RM CPU mappings of one client (module doc).
#[derive(Debug, Default)]
pub(crate) struct LiveMaps(Mutex<State>);

impl LiveMaps {
    fn state(&self) -> Result<MutexGuard<'_, State>> {
        self.0.lock().map_err(|_poisoned| {
            SyncError::Poisoned {
                object: "RM CPU mapping list",
            }
            .into()
        })
    }

    /// Issue `map` (the NVOS33, returning the NVOS34 that releases it) and record the mapping,
    /// both under the lock.
    ///
    /// # Errors
    /// The exit handler closed the list (no NVOS33 is issued), `map` fails, or the lock is
    /// poisoned.
    pub(crate) fn map(
        &self,
        parent: RmHandle,
        map: impl FnOnce() -> Result<abi::UnmapMemory>,
    ) -> Result<MapId> {
        let mut state = self.state()?;
        if state.closed {
            return Err(torn_down());
        }
        let id = state.next;
        state.next = id.checked_add(1).ok_or(MemoryError::CpuMapping {
            reason: "RM CPU mapping ids exhausted",
        })?;
        let unmap = map()?;
        state.maps.insert(id, LiveMap { unmap, parent });
        Ok(MapId(id))
    }

    /// Issue `update` (UPDATE_DEVICE_MAPPING_INFO, given the mapping's current NVOS34) and
    /// release the mapping by `address` from then on, both under the lock.
    ///
    /// # Errors
    /// The exit handler already released the mapping, `update` fails (the key stays), or the
    /// lock is poisoned.
    pub(crate) fn rekey(
        &self,
        id: MapId,
        address: u64,
        update: impl FnOnce(&abi::UnmapMemory) -> Result<()>,
    ) -> Result<()> {
        let mut state = self.state()?;
        let closed = state.closed;
        let Some(map) = state.maps.get_mut(&id.0) else {
            return Err(if closed { torn_down() } else { unknown() });
        };
        update(&map.unmap)?;
        map.unmap.linear_address = address;
        Ok(())
    }

    /// Issue `unmap` (the NVOS34) under the lock and forget the mapping once it succeeded. A
    /// mapping the exit handler took is skipped: the handler issued its NVOS34.
    ///
    /// # Errors
    /// `unmap` fails (the mapping stays listed), the mapping was never listed, or the lock is
    /// poisoned.
    pub(crate) fn unmap(
        &self,
        id: MapId,
        unmap: impl FnOnce(&abi::UnmapMemory) -> Result<()>,
    ) -> Result<()> {
        let mut state = self.state()?;
        let Some(map) = state.maps.get(&id.0) else {
            return if state.closed { Ok(()) } else { Err(unknown()) };
        };
        unmap(&map.unmap)?;
        state.maps.remove(&id.0);
        Ok(())
    }

    /// Refuse an RM free of `object` while a mapping depends on it: it is the mapped object,
    /// its parent, its device, or the client `root`.
    ///
    /// # Errors
    /// [`MemoryError::FreeWhileCpuMapped`] naming the first such mapping, or a poisoned lock.
    pub(crate) fn check_free(&self, root: RmHandle, object: RmHandle) -> Result<()> {
        let state = self.state()?;
        let raw = object.get();
        let blocking = state.maps.values().find(|map| {
            object == root
                || map.unmap.memory == raw
                || map.parent == object
                || map.unmap.device == raw
        });
        match blocking {
            Some(map) => Err(MemoryError::FreeWhileCpuMapped {
                object,
                mapped: RmHandle::new(map.unmap.memory),
                parent: map.parent,
                device: RmHandle::new(map.unmap.device),
            }
            .into()),
            None => Ok(()),
        }
    }

    /// Close the list for the exit handler and hand it every live mapping, whose NVOS34 the
    /// caller issues before it frees the client. Owners find their entries gone from here on,
    /// and new mappings are refused.
    ///
    /// # Errors
    /// The lock is poisoned.
    pub(crate) fn close(&self) -> Result<Vec<LiveMap>> {
        let mut state = self.state()?;
        state.closed = true;
        Ok(std::mem::take(&mut state.maps).into_values().collect())
    }
}

fn torn_down() -> Error {
    MemoryError::NotLive {
        reason: "the RM client was torn down by the process exit handler",
    }
    .into()
}

fn unknown() -> Error {
    MemoryError::NotLive {
        reason: "RM CPU mapping missing from its client's list",
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    const ROOT: RmHandle = RmHandle::new(0xc1d0_0001);
    const DEVICE: u32 = 0xcaf0_0000;
    const PARENT: RmHandle = RmHandle::new(DEVICE);

    fn nvos34(memory: u32, linear_address: u64) -> abi::UnmapMemory {
        abi::UnmapMemory {
            client: ROOT.get(),
            device: DEVICE,
            memory,
            linear_address,
            ..Default::default()
        }
    }

    fn mapped(maps: &LiveMaps, memory: u32, cookie: u64) -> MapId {
        maps.map(PARENT, || Ok(nvos34(memory, cookie)))
            .expect("map")
    }

    /// A free of the mapped object, its parent, its device or the client is refused while
    /// the mapping is live, and allowed once its NVOS34 succeeded.
    #[test]
    fn live_mapping_refuses_free_of_what_it_depends_on() {
        let maps = LiveMaps::default();
        let (memory, other) = (0xbeef_0001, 0xbeef_0002);
        let id = mapped(&maps, memory, 0x1000);
        for object in [memory, DEVICE, ROOT.get()] {
            let error = maps
                .check_free(ROOT, RmHandle::new(object))
                .expect_err("free refused while mapped");
            assert!(matches!(
                error,
                Error::Memory(MemoryError::FreeWhileCpuMapped { mapped, .. })
                    if mapped == RmHandle::new(memory)
            ));
        }
        maps.check_free(ROOT, RmHandle::new(other))
            .expect("unrelated object frees");
        let failed: Result<()> = Err(Error::Contract("injected NVOS34 failure".into()));
        assert!(maps.unmap(id, |_| failed).is_err());
        assert!(
            maps.check_free(ROOT, RmHandle::new(memory)).is_err(),
            "a failed NVOS34 leaves the mapping live"
        );
        maps.unmap(id, |_| Ok(())).expect("unmap");
        for object in [memory, DEVICE, ROOT.get(), other] {
            maps.check_free(ROOT, RmHandle::new(object))
                .expect("free allowed after unmap");
        }
        assert!(
            maps.unmap(id, |_| panic!("no second NVOS34")).is_err(),
            "an unlisted mapping on an open client is an error"
        );
    }

    /// The exit path unmaps every live mapping by its current RM key (the CPU address after
    /// UPDATE_DEVICE_MAPPING_INFO), refuses new mappings, and owners skip what it took.
    #[test]
    fn exit_unmaps_every_live_mapping_by_its_current_key() {
        let maps = LiveMaps::default();
        let sysmem = mapped(&maps, 0x51, 0x1000);
        let vidmem = mapped(&maps, 0x52, 0x2000);
        let released = mapped(&maps, 0x53, 0x3000);
        maps.rekey(vidmem, 0x7f00_0000_0000, |unmap| {
            assert_eq!(unmap.linear_address, 0x2000, "UPDATE gets NVOS33's address");
            Ok(())
        })
        .expect("rekey");
        let failed: Result<()> = Err(Error::Contract("injected UPDATE failure".into()));
        assert!(maps.rekey(sysmem, 0x7f00_0001_0000, |_| failed).is_err());
        maps.unmap(released, |_| Ok(())).expect("unmap");

        let mut taken: Vec<_> = maps
            .close()
            .expect("close")
            .into_iter()
            .map(|map| (map.unmap.memory, map.unmap.linear_address))
            .collect();
        taken.sort_unstable();
        assert_eq!(taken, [(0x51, 0x1000), (0x52, 0x7f00_0000_0000)]);

        maps.unmap(sysmem, |_| panic!("the exit path issued this NVOS34"))
            .expect("skip");
        assert!(maps.rekey(vidmem, 1, |_| panic!("no UPDATE")).is_err());
        assert!(maps.map(PARENT, || panic!("no NVOS33 after exit")).is_err());
        assert!(maps.close().expect("close").is_empty());
    }

    /// An owner unmapping while the exit handler closes the list: exactly one NVOS34 per
    /// mapping in every interleaving, and the exit path's NVOS34s come after every owner NVOS34
    /// that started first (the root free follows them).
    #[test]
    fn owner_and_exit_issue_one_nvos34_per_mapping() {
        const MAPPINGS: u32 = 8;
        let rounds = if cfg!(miri) { 4 } else { 200 };
        for _ in 0..rounds {
            let maps = Arc::new(LiveMaps::default());
            let ids: Vec<_> = (0..MAPPINGS)
                .map(|memory| mapped(&maps, memory, u64::from(memory) << 12))
                .collect();
            let issued = Arc::new(Mutex::new(Vec::new()));
            let start = Arc::new(Barrier::new(2));
            let owner = {
                let (maps, issued, start) =
                    (Arc::clone(&maps), Arc::clone(&issued), Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    for id in ids {
                        maps.unmap(id, |unmap| {
                            issued
                                .lock()
                                .expect("test lock")
                                .push(("owner", unmap.memory));
                            Ok(())
                        })
                        .expect("owner unmap");
                    }
                })
            };
            start.wait();
            for map in maps.close().expect("close") {
                issued
                    .lock()
                    .expect("test lock")
                    .push(("exit", map.unmap.memory));
            }
            owner.join().expect("owner thread");
            let issued = issued.lock().expect("test lock").clone();
            let mut memories: Vec<_> = issued.iter().map(|(_, memory)| *memory).collect();
            memories.sort_unstable();
            assert_eq!(memories, (0..MAPPINGS).collect::<Vec<_>>());
            if let Some(first_exit) = issued.iter().position(|(by, _)| *by == "exit") {
                assert!(
                    issued[first_exit..].iter().all(|(by, _)| *by == "exit"),
                    "no owner NVOS34 after the exit path took the list: {issued:?}"
                );
            }
        }
    }
}
