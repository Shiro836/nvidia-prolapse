// SPDX-License-Identifier: MIT
//! Push-buffer method encoding shared by every channel (host, copy, compute, NVDEC).
/// Appends one incrementing method header for `values.len()` dwords at `address` on `subchannel`,
/// followed by the values (`NV906F_DMA_SEC_OP_INC_METHOD` in bits 31:29, `NV906F_DMA_METHOD_COUNT`
/// in 28:16: open-gpu-kernel-modules class/cl906f.h:173-177).
pub fn method(words: &mut Vec<u32>, subchannel: u32, address: u32, values: &[u32]) {
    words.push((2 << 28) | ((values.len() as u32) << 16) | (subchannel << 13) | (address >> 2));
    words.extend_from_slice(values);
}
/// Splits a 64-bit GPU address into the (upper, lower) dword pair that the `*_A`/`*_B`
/// method pairs of the NVIDIA classes take, upper first.
pub fn hi_lo(address: u64) -> [u32; 2] {
    [(address >> 32) as u32, address as u32]
}
