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

/// H.264/HEVC NALs without an SEI payload (shorter than or equal to the NAL unit header, or empty
/// between two start codes): the HDR scan sliced past the header before checking the length.
#[test]
fn hdr_nal_without_payload() {
    replay(drivers::hdr_metadata, "hdr_metadata-nal_without_payload");
    replay(drivers::hdr_metadata, "hdr_metadata-empty_nal");
}

/// H.264 SPS frame cropping offsets whose product with CropUnitX/Y (7.4.2.1.1) overflows `u32`:
/// the multiplication panicked under overflow checks (release builds keep them).
#[test]
fn h264_crop_offsets_overflow() {
    replay(drivers::h264_headers, "h264_headers-crop_left_overflow");
    replay(drivers::h264_headers, "h264_headers-crop_top_overflow");
    replay(drivers::decoder, "decoder-h264_crop_bottom_overflow");
}

/// AV1 frame of size 0 (a frame_size_with_refs copy of an empty reference slot): tile_info's
/// non-uniform spacing divided by its widest tile, 0.
#[test]
fn av1_tile_info_zero_size() {
    replay(drivers::av1_headers, "av1_headers-tile_info_zero_size");
    replay(drivers::decoder, "decoder-av1_tile_info_zero_size");
}
