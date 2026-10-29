// SPDX-License-Identifier: MIT
//! NVK's nvkmd backend over the RM, built as a static library that Mesa links
//! into `libvulkan_nouveau.so` (meson option `nvk-rm-backend=rust`). It reaches Mesa's hidden-visibility functions at that static
//! link and the RM through the `nv_rm_*` entries of `libnv_bionic_native.so`,
//! the one RM instance of the process.
//!
//! Every `extern "C"` entry and callback lives in [`ffi`] and runs inside the
//! shared panic guard (`nv-ffi`); the policy it applies lives in
//! `nvkmd-rm-logic`.
mod ffi;
