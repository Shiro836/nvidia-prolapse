// SPDX-License-Identifier: MIT
//! Policy of NVK's nvkmd backend over the RM, without any I/O: the device
//! information NVK sees, which RM placement, size and alignment an allocation
//! request gets, which VA requests and binds are accepted, which bindings an
//! unbind covers, which waits become GPU acquires, and how pending pushes are
//! cut into submissions.
//!
//! The boundary crate turns NVK's arguments into these types, calls the
//! functions here and carries out the result through the RM's C ABI. The rules
//! are those of the C backend `mesa/nvkmd_rm.c`; each test names the line it pins.
//! Pure functions on `Copy` data: no state, usable from any thread.
#![forbid(unsafe_code)]

pub mod info;
pub mod memory;
pub mod push;
pub mod va;
pub mod wait;
