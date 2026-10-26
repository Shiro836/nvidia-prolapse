// SPDX-License-Identifier: MIT
//! Fuzz target: see `nv_video_fuzz::h264_headers`.
#![no_main]
libfuzzer_sys::fuzz_target!(|data: &[u8]| nv_video_fuzz::h264_headers(data));
