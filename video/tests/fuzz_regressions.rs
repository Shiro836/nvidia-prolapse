// SPDX-License-Identifier: MIT
//! Inputs the fuzz targets (`video/fuzz`) found panicking, replayed through the same drivers. Each
//! file under `fuzz-regressions/` is named after the target and the panic site it hit before the
//! fix; every one must now be rejected with an error, not a panic.

#[path = "../fuzz/src/lib.rs"]
pub mod drivers;

fn replay(target: fn(&[u8]), name: &str) {
    let path = format!(
        "{}/tests/fuzz-regressions/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read(&path).expect("regression input is checked in");
    target(&data);
}

/// HEVC SPS, PPS and slice NALs shorter than the two-byte NAL unit header (H.265 7.3.1.2): the
/// parsers sliced `rbsp[2..]` and panicked.
#[test]
fn hevc_nal_shorter_than_its_header() {
    replay(drivers::hevc_headers, "hevc_headers-sps_without_nal_header");
    replay(drivers::hevc_headers, "hevc_headers-pps_without_nal_header");
    replay(
        drivers::hevc_headers,
        "hevc_headers-slice_without_nal_header",
    );
}
