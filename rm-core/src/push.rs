// SPDX-License-Identifier: MIT
//! Push-buffer method encoding shared by every channel (host, copy, compute, NVDEC).
use crate::conv::{high_u32, low_u32};

/// Largest dword count one method header carries (`NV906F_DMA_METHOD_COUNT`, bits 28:16:
/// open-gpu-kernel-modules class/cl906f.h:175).
pub const METHOD_COUNT_MAX: usize = 0x1fff;

/// Appends one incrementing method header for `values.len()` dwords at `address` on `subchannel`,
/// followed by the values (`NV906F_DMA_SEC_OP_INC_METHOD` in bits 31:29, `NV906F_DMA_METHOD_COUNT`
/// in 28:16: open-gpu-kernel-modules class/cl906f.h:173-177).
///
/// # Panics
/// If `values` holds more than [`METHOD_COUNT_MAX`] dwords: the header cannot encode that count,
/// and every caller passes a value list of fixed, short length.
pub fn method(words: &mut Vec<u32>, subchannel: u32, address: u32, values: &[u32]) {
    assert!(
        values.len() <= METHOD_COUNT_MAX,
        "method 0x{address:x}: {} dwords exceed NV906F_DMA_METHOD_COUNT",
        values.len()
    );
    let count = u32::try_from(values.len()).expect("count <= METHOD_COUNT_MAX fits u32");
    words.push((2 << 28) | (count << 16) | (subchannel << 13) | (address >> 2));
    words.extend_from_slice(values);
}
/// Splits a 64-bit GPU address into the (upper, lower) dword pair that the `*_A`/`*_B`
/// method pairs of the NVIDIA classes take, upper first.
pub fn hi_lo(address: u64) -> [u32; 2] {
    [high_u32(address), low_u32(address)]
}
