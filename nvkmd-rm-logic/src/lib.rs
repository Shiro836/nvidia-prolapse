// SPDX-License-Identifier: MIT
//! Policy of NVK's nvkmd backend over the RM, without any I/O: the device
//! information NVK sees, which RM placement, size and alignment an allocation
//! request gets, which VA requests and binds are accepted, which bindings an
//! unbind covers, which waits become GPU acquires, how CPU waits on sync
//! payloads are timed and answered, and how pending pushes are cut into
//! submissions.
//!
//! The boundary crate turns NVK's arguments into these types, calls the
//! functions here and carries out the result through the RM's C ABI.
//! Pure functions on `Copy` data: no state, usable from any thread.
#![forbid(unsafe_code)]

pub mod info;
pub mod memory;
pub mod push;
pub mod sync;
pub mod va;
pub mod wait;
