// SPDX-License-Identifier: MIT
//! Fuzz target: see `nv_video_fuzz::decoder`.
#![no_main]
libfuzzer_sys::fuzz_target!(|data: &[u8]| nv_video_fuzz::decoder(data));
