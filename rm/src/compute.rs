// SPDX-License-Identifier: MIT
//! Public Blackwell QMD v5 descriptor construction for runtime-compiled shaders.
//! NAK supplies code and resource metadata; local stacks remain unsupported.
use crate::{
    gpu::{AddressSpace, GpuMapping},
    rm::{Device, Error, Mapping, Result},
};
use std::rc::Rc;

pub(crate) const LOCAL_WINDOW: u64 = 0x729300000000;
pub(crate) const SHARED_WINDOW: u64 = 0x729400000000;

/// Bit ranges are inclusive, from NVIDIA's MIT clcec0qmd.h (QMDV05_00).
struct Qmd([u32; 96]);
impl Qmd {
    fn set(&mut self, high: usize, low: usize, value: u64) -> Result<()> {
        if high < low
            || high >= 3072
            || high - low >= 64
            || (high - low < 63 && value >> (high - low + 1) != 0)
        {
            return Err(Error::Contract(format!(
                "QMD value {value:#x} does not fit {high}:{low}"
            )));
        }
        for bit in low..=high {
            self.0[bit / 32] = (self.0[bit / 32] & !(1 << (bit % 32)))
                | (((value >> (bit - low)) as u32 & 1) << (bit % 32));
        }
        Ok(())
    }
}

pub(crate) struct ComputeJob {
    pub address: u64,
    // Keep mappings, memory objects and code alive until after channel teardown.
    _code_va: GpuMapping,
    _descriptor_va: GpuMapping,
    _code_cpu: Mapping,
    _descriptor_cpu: Mapping,
}
impl ComputeJob {
    pub fn new(
        device: &Rc<Device>,
        space: &AddressSpace,
        output: u64,
        completion: u64,
    ) -> Result<Self> {
        let words: Vec<u32> = include_bytes!("../shaders/pattern.spv")
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let program = crate::compiler::compile(&words, c"main")?;
        let mut constants = vec![0u8; 1152];
        constants[..8].copy_from_slice(&output.to_le_bytes());
        constants[8..12].copy_from_slice(&1024u32.to_le_bytes());
        Self::compiled(device, space, completion, &program, &constants, [1, 1, 1])
    }
    pub(crate) fn compiled(
        device: &Rc<Device>,
        space: &AddressSpace,
        completion: u64,
        program: &crate::compiler::Shader,
        constants: &[u8],
        groups: [u32; 3],
    ) -> Result<Self> {
        program.validate()?;
        if groups.iter().any(|&v| v > 65535) || constants.len() > 3072 {
            return Err(Error::Contract(
                "dispatch grid or constant buffer exceeds implemented limits".into(),
            ));
        }
        let sm = device.sm_version()?;
        if sm != 0xa04 {
            return Err(Error::Contract(format!(
                "native shader backend requires RM SM version 0xa04, got {sm:#x}; compile and validate the matching shader/QMD first"
            )));
        }
        let code = device.allocate_vram(program.code.len().next_multiple_of(4096) + 4096)?; // Extra mapped page for instruction prefetch.
        let descriptor = device.allocate_vram(4096)?;
        let code_va = space.map(&code)?;
        let descriptor_va = space.map(&descriptor)?;
        let mut code_cpu = code.map()?;
        let mut descriptor_cpu = descriptor.map()?;
        for i in 0..code.size / 4 {
            code_cpu.write_u32(i * 4, 0)?;
        }
        for i in 0..4096 / 4 {
            descriptor_cpu.write_u32(i * 4, 0)?;
        }
        for (i, chunk) in program.code.chunks_exact(4).enumerate() {
            code_cpu.write_u32(i * 4, u32::from_le_bytes(chunk.try_into().unwrap()))?;
        }
        let address = descriptor_va.address();
        if address & 255 != 0 || address >> 40 != 0 {
            return Err(Error::Contract(
                "QMD launch address must be 256-byte aligned and below 1 TiB".into(),
            ));
        }
        let cbuf = address + 512;
        let mut qmd = Qmd([0; 96]);
        for (high, low, value) in [
            (149, 144, 0x3f), // group id
            (153, 151, 2),    // GRID_CTA
            (288, 288, 1),
            (290, 289, 2),
            (291, 291, 1),    // release0, two words, system membar
            (455, 448, 0xa4), // sm_120 ISA: NV2080 SM version 0xa04 -> SASS 0xa4
            (456, 456, 1),
            (457, 457, 1),
            (471, 468, 5),
            (472, 472, 1),
            (473, 473, 1),
            (474, 474, 1),
            (475, 475, 1),
            (476, 476, 1),
            (477, 477, 1),
            (511, 480, completion & 0xffffffff),
            (536, 512, completion >> 32),
            (575, 544, 1), // 64-bit completion payload, upper already zero
            (625, 624, 1), // CWD L1_SYSMEMBAR
            (1055, 1024, (code_va.address() >> 4) & 0xffffffff),
            (1076, 1056, code_va.address() >> 36),
            (1085, 1077, ((program.code.len() >> 8).min(0x1ff)) as u64),
            (1103, 1088, program.workgroup[0] as u64),
            (1119, 1104, program.workgroup[1] as u64),
            (1127, 1120, program.workgroup[2] as u64),
            (1136, 1128, program.registers as u64),
            (1141, 1137, program.barriers as u64),
            (
                1162,
                1152,
                program.shared_size.next_multiple_of(128) as u64 >> 7,
            ),
            (1168, 1163, 9),
            (1174, 1169, 0x1a),
            (1180, 1175, 9),
            (1279, 1248, groups[0] as u64),
            (1295, 1280, groups[1] as u64),
            (1327, 1312, groups[2] as u64),
            (1375, 1344, (cbuf >> 6) & 0xffffffff),
            (1394, 1376, cbuf >> 38),
            (1407, 1395, constants.len().next_multiple_of(16) as u64 >> 4), // bytes rounded to 16-byte QMD size units
            (1856, 1856, 1),
            (1859, 1859, 1),
            (1919, 1888, (code_va.address() >> 8) & 0xffffffff),
            (1936, 1920, code_va.address() >> 40),
        ] {
            qmd.set(high, low, value)?;
        }
        for (i, word) in qmd.0.into_iter().enumerate() {
            descriptor_cpu.write_u32(i * 4, word)?;
        }
        for (i, chunk) in constants.chunks(4).enumerate() {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            descriptor_cpu.write_u32(512 + i * 4, u32::from_le_bytes(bytes))?;
        }
        Ok(Self {
            address,
            _code_va: code_va,
            _descriptor_va: descriptor_va,
            _code_cpu: code_cpu,
            _descriptor_cpu: descriptor_cpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qmd_fields_cross_words_and_preserve_neighbors() {
        let mut qmd = Qmd([u32::MAX; 96]);
        qmd.set(70, 29, 0x123456789ab).unwrap();
        let raw: u128 =
            u128::from(qmd.0[0]) | (u128::from(qmd.0[1]) << 32) | (u128::from(qmd.0[2]) << 64);
        assert_eq!((raw >> 29) & ((1u128 << 42) - 1), 0x123456789ab);
        assert_eq!(raw & ((1u128 << 29) - 1), (1u128 << 29) - 1);
        assert_eq!(raw >> 71, (1u128 << 25) - 1);
        assert!(qmd.set(3, 0, 16).is_err());
        assert!(qmd.set(3072, 3072, 0).is_err());
        assert!(qmd.set(0, 1, 0).is_err());
    }
}
