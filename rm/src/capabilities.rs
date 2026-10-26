// SPDX-License-Identifier: MIT
//! Explicit RM-advertised/public-header/Mesa backend intersection.
//!
//! Authorities: OGKM 615.71.09 class/cl* and ctrl2080gr.h SM_VERSION;
//! Mesa 0b97dcdd9424fce130054e1972a610b765644003 classes/cl*,
//! winsys/nouveau_device.c::sm_for_chipset, nvk_physical_device.c and
//! nil/image.rs::choose_pte_kind. These are implementation contracts, not
//! evidence that any untested GPU/driver combination works.
use crate::{
    abi,
    rm::{Device, Error, Result},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submission {
    PreVolta,
    Volta,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub sm: u32,
    pub channel: u32,
    pub usermode: Option<u32>,
    pub copy: u32,
    pub eng2d: u32,
    pub eng3d: u32,
    pub m2mf: u32,
    pub compute: u32,
    pub generic_kind: u32,
    pub submission: Submission,
}

pub(crate) const COPY_CLASSES: &[u32] = &[
    0xcab5, 0xc9b5, 0xc7b5, 0xc6b5, 0xc5b5, 0xc3b5, 0xc1b5, 0xc0b5, 0xb0b5,
];

/// Select an advertised graphics partner, never guess a physical CE instance.
/// Kepler+ normally has one partner. Validate per-engine class membership so
/// reused CE class IDs cannot silently select an unrelated engine.
pub(crate) fn copy_engine_parameters(
    compat: crate::compat::Compat,
    class: u32,
    partners: &abi::EnginePartners,
    mut supports_class: impl FnMut(u32) -> Result<bool>,
) -> Result<abi::CopyEngine> {
    if !COPY_CLASSES.contains(&class) {
        return Err(Error::Contract(format!(
            "unimplemented copy allocation class {class:#x}"
        )));
    }
    compat.require_constant("NVB0B5_ALLOCATION_PARAMETERS_VERSION_1", 1)?;
    if partners.count as usize > partners.partners.len() {
        return Err(Error::Contract(
            "RM copy partner count exceeds public array capacity".into(),
        ));
    }
    let mut selected = None;
    let mut seen = Vec::new();
    for &engine in &partners.partners[..partners.count as usize] {
        if seen.contains(&engine) {
            return Err(Error::Contract(
                "RM returned duplicate engine partners".into(),
            ));
        }
        seen.push(engine);
        // COPY10+ is non-contiguous and absent from older headers. Test exact
        // profile constants instead of accepting a numeric interval or suffix.
        let is_copy = (0..20).any(|i| {
            compat.constant(&format!("NV2080_ENGINE_TYPE_COPY{i}")) == Some(engine as u64)
        });
        if is_copy && supports_class(engine)? && selected.replace(engine).is_some() {
            return Err(Error::Contract(format!(
                "multiple graphics copy partners advertise class {class:#x}; select a documented runqueue"
            )));
        }
    }
    let engine_type = selected.ok_or_else(|| {
        Error::Contract(format!(
            "no graphics copy partner advertises class {class:#x}"
        ))
    })?;
    Ok(abi::CopyEngine {
        version: 1,
        engine_type,
    })
}

fn advertised(classes: &[u32], known: &[u32], name: &str) -> Result<u32> {
    known
        .iter()
        .copied()
        .find(|c| classes.contains(c))
        .ok_or_else(|| {
            Error::Contract(format!(
                "RM advertises no implemented {name} class; queried {classes:x?}"
            ))
        })
}

/// The RM value names an ISA revision, not a CUDA compute capability decimal.
/// In particular GA100 uses 8_02 for SM80 and GB20x's transition spelling
/// 10_04 and final 12_00 both describe SM120 (ctrl2080gr.h Bug 4333440).
/// Unknown encodings are rejected instead of extrapolating future hardware.
pub fn select(classes: &[u32], rm_sm: u32) -> Result<Capabilities> {
    let (sm, eng3d, compute) = match rm_sm {
        0x500 => (50, 0xb097, 0xb0c0),
        0x502 => (52, 0xb197, 0xb1c0),
        0x503 => (53, 0xb197, 0xb1c0),
        0x600 => (60, 0xc097, 0xc0c0),
        0x601 => (61, 0xc197, 0xc1c0),
        0x602 => (62, 0xc197, 0xc1c0),
        0x700 => (70, 0xc397, 0xc3c0),
        0x702 => (72, 0xc397, 0xc3c0),
        // TU116/TU117 report RM ISA7_03, but use Mesa's Turing SM75 compiler
        // contract. Keep the real chipset in mesa::Info: NVK separately masks
        // TU11x features (nvk_physical_device.c::nvk_get_device_features).
        0x703 | 0x705 => (75, 0xc597, 0xc5c0),
        0x802 => (80, 0xc697, 0xc6c0),
        0x806 => (86, 0xc797, 0xc7c0),
        0x807 => (87, 0xc797, 0xc7c0),
        0x809 => (89, 0xc997, 0xc9c0),
        0xa04 | 0xc00 => (120, 0xce97, 0xcec0),
        _ => {
            return Err(Error::Contract(format!(
                "RM SM ISA {rm_sm:#x} has no enabled pinned Mesa frontend contract; add a sourced compiler/engine mapping before enabling this GPU"
            )));
        }
    };
    let eng3d = advertised(classes, &[eng3d], "3D for queried SM")?;
    let compute = advertised(classes, &[compute], "compute for queried SM")?;
    // Finite preference lists: never accept an arbitrary class with a familiar
    // low byte. Public GPFIFO families through Blackwell share the selected
    // pre-Volta or Volta submission contracts implemented by gpu.rs.
    let channel = if sm < 70 {
        advertised(classes, &[0xc06f, 0xb06f], "pre-Volta GPFIFO")?
    } else {
        advertised(
            classes,
            &[0xc96f, 0xc86f, 0xc76f, 0xc56f, 0xc46f, 0xc36f],
            "Volta+ GPFIFO",
        )?
    };
    let usermode = if sm < 70 {
        None
    } else {
        Some(advertised(
            classes,
            &[0xc661, 0xc561, 0xc461, 0xc361],
            "USERMODE doorbell",
        )?)
    };
    Ok(Capabilities {
        sm,
        channel,
        usermode,
        eng3d,
        compute,
        copy: advertised(classes, COPY_CLASSES, "DMA copy")?,
        eng2d: advertised(classes, &[0x902d], "2D")?,
        // CD40 has the common inline-to-memory method ABI in the public RM
        // header; Mesa's emitter uses the same A140 method subset.
        m2mf: advertised(classes, &[0xcd40, 0xa140], "inline-to-memory")?,
        generic_kind: if sm < 75 { 0xfe } else { 0x06 },
        submission: if sm < 70 {
            Submission::PreVolta
        } else {
            Submission::Volta
        },
    })
}

pub fn query(device: &Device) -> Result<Capabilities> {
    let mut entries = [[0xcu32, 0]];
    let mut query = abi::GrInfo {
        count: 1,
        entries: entries.as_mut_ptr() as u64,
        ..Default::default()
    };
    device
        .client
        .control(device.subdevice, 0x20801201, &mut query)?;
    select(&device.classes()?, entries[0][1])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn classes() -> Vec<u32> {
        vec![0xc96f, 0xc661, 0xcab5, 0x902d, 0xce97, 0xcd40, 0xcec0]
    }
    fn partners(values: &[u32]) -> abi::EnginePartners {
        let mut out = abi::EnginePartners {
            count: values.len() as u32,
            ..Default::default()
        };
        out.partners[..values.len()].copy_from_slice(values);
        out
    }
    #[test]
    fn copy_parameters_use_discovered_partner_and_exact_class() {
        for version in [
            "515.43.04",
            "535.309.01",
            "550.54.14",
            "580.178.04",
            "615.71.09",
        ] {
            let compat = crate::compat::Compat::for_version(version).unwrap();
            for class in COPY_CLASSES {
                let mut queries = Vec::new();
                let p = copy_engine_parameters(compat, *class, &partners(&[1, 11]), |e| {
                    queries.push(e);
                    Ok(true)
                })
                .unwrap();
                // Independent public C oracle: version1 means engine ordinal,
                // so COPY2 is11, not physical index2 and not default COPY0=9.
                assert_eq!((p.version, p.engine_type), (1, 11));
                assert_eq!(queries, [11]);
            }
        }
    }
    #[test]
    fn copy_partner_selection_rejects_missing_ambiguous_or_malformed_reports() {
        let compat = crate::compat::Compat::for_version("580.178.04").unwrap();
        for values in [&[][..], &[1], &[0xdead], &[9, 10], &[9, 9]] {
            assert!(
                copy_engine_parameters(compat, 0xb0b5, &partners(values), |_| Ok(true)).is_err()
            );
        }
        assert!(copy_engine_parameters(compat, 0xc0b5, &partners(&[9]), |_| Ok(false)).is_err());
        assert!(
            copy_engine_parameters(compat, 0xffffb5, &partners(&[9]), |_| panic!(
                "unknown class queried"
            ))
            .is_err()
        );
        let mut oversized = partners(&[9]);
        oversized.count = 33;
        assert!(
            copy_engine_parameters(compat, 0xb0b5, &oversized, |_| panic!(
                "oversized report queried"
            ))
            .is_err()
        );
        assert!(
            copy_engine_parameters(compat, 0xb0b5, &partners(&[9]), |_| Err(Error::Contract(
                "class query failed".into()
            )))
            .is_err()
        );
        // Multiple returned partners are okay only when exactly one advertises
        // our selected class. Never choose the first matching ordinal blindly.
        let p =
            copy_engine_parameters(compat, 0xc0b5, &partners(&[9, 10]), |e| Ok(e == 10)).unwrap();
        assert_eq!(p.engine_type, 10);
    }
    #[test]
    fn later_copy_ordinals_are_profile_scoped() {
        let old = crate::compat::Compat::for_version("535.309.01").unwrap();
        let new = crate::compat::Compat::for_version("580.178.04").unwrap();
        // Public535 has COPY0..9 only; public580 COPY10 is0x34 (not0x13).
        assert!(
            copy_engine_parameters(old, 0xc6b5, &partners(&[0x34]), |_| panic!(
                "unmeasured ordinal queried"
            ))
            .is_err()
        );
        assert_eq!(
            copy_engine_parameters(new, 0xc6b5, &partners(&[0x34]), |_| Ok(true))
                .unwrap()
                .engine_type,
            0x34
        );
        let unmeasured = crate::compat::Compat::for_version("390.116").unwrap();
        assert!(
            copy_engine_parameters(unmeasured, 0xb0b5, &partners(&[9]), |_| panic!(
                "unmeasured profile queried"
            ))
            .is_err()
        );
    }
    #[test]
    fn blackwell_transition_isa_is_sm120() {
        let caps = select(&classes(), 0xa04).unwrap();
        assert_eq!(caps.sm, 120);
        assert_eq!(caps.copy, 0xcab5);
        assert_eq!(caps.m2mf, 0xcd40);
        assert_eq!(caps.eng3d, 0xce97);
        assert_eq!(caps.compute, 0xcec0);
        assert_eq!(select(&classes(), 0xc00).unwrap().sm, 120);
    }
    #[test]
    fn tu11x_isa703_uses_turing_shader_model_and_advertised_classes() {
        let c = [0xc46f, 0xc461, 0xc5b5, 0x902d, 0xc597, 0xa140, 0xc5c0];
        let caps = select(&c, 0x703).unwrap();
        assert_eq!(caps, select(&c, 0x705).unwrap());
        assert_eq!(caps.sm, 75);
        assert_eq!(caps.generic_kind, 6);
        assert_eq!(caps.submission, Submission::Volta);
        // RM ISA encodings are an explicit table, never a numeric SM formula.
        for unknown in [0x701, 0x704, 0x706] {
            assert!(select(&c, unknown).is_err());
        }
        for missing in [0xc597, 0xc5c0] {
            let absent: Vec<_> = c.into_iter().filter(|v| *v != missing).collect();
            assert!(select(&absent, 0x703).is_err());
        }
    }
    #[test]
    fn refuses_unknown_classes_and_missing_engines() {
        let mut c = classes();
        c[0] = 0xffff6f;
        assert!(select(&c, 0xa04).is_err());
        c = classes();
        c[1] = 0xffff61;
        assert!(select(&c, 0xa04).is_err());
        assert!(select(&classes(), 0x900).is_err());
        assert!(select(&classes(), 0xc01).is_err());
        assert!(select(&classes(), 0x806).is_err());
    }
    #[test]
    fn maxwell_has_no_usermode_and_old_generic_kind() {
        let c = [0xb06f, 0xb0b5, 0x902d, 0xb197, 0xa140, 0xb1c0];
        let caps = select(&c, 0x502).unwrap();
        assert_eq!(caps.sm, 52);
        assert_eq!(caps.usermode, None);
        assert_eq!(caps.generic_kind, 0xfe);
        assert_eq!(caps.submission, Submission::PreVolta);
    }
    #[test]
    fn ga100_isa_802_is_not_sm82() {
        let c = [0xc56f, 0xc561, 0xc6b5, 0x902d, 0xc697, 0xa140, 0xc6c0];
        assert_eq!(select(&c, 0x802).unwrap().sm, 80);
    }
}
