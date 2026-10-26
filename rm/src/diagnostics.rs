// SPDX-License-Identifier: MIT
//! Hardware-invariant diagnostics for the vast.ai harness (feature
//! `diagnostics` only; production bridges never compile this module).
//! Read-only queries of the same public RM controls the backend itself uses.
//! Nothing here relaxes a contract: every value is reported exactly, and
//! decisions remain those of `capabilities`, `fences` and `compat`.
use crate::{
    compat::Compat,
    rm::{Device, Result},
};

/// Exact semaphore-surface reply (NV2080_CTRL_CMD_FB_GET_SEMAPHORE_SURFACE_LAYOUT)
/// plus the payload mode `fences::payload_mode` derives from it.
#[derive(Debug, Clone)]
pub struct SemaphoreLayout {
    pub max_submitted_offset: u64,
    pub monitored_offset: u64,
    pub stride: u64,
    pub caps: u32,
    /// "native64", "tracked32@<offset>", or the named contract error.
    pub mode: std::result::Result<String, String>,
}

pub fn semaphore_layout(device: &Device) -> Result<SemaphoreLayout> {
    let raw = crate::fences::query_layout(device)?;
    Ok(SemaphoreLayout {
        max_submitted_offset: raw.0,
        monitored_offset: raw.1,
        stride: raw.2,
        caps: raw.3,
        mode: crate::fences::layout_mode(raw).map_err(|e| e.to_string()),
    })
}

/// Every exact release compiled into this backend with its static predicates.
pub struct VersionFacts {
    pub version: &'static str,
    pub client_abi: std::result::Result<(), String>,
    pub semsurf_fences: std::result::Result<(), String>,
    pub semsurf_bind: bool,
    pub kind_override: bool,
    pub external_memory: bool,
    pub identify_object: bool,
}

fn facts(compat: Compat) -> VersionFacts {
    VersionFacts {
        version: compat.version(),
        client_abi: compat.require_client_abi().map_err(|e| e.to_string()),
        semsurf_fences: crate::fences::validate_compat(compat).map_err(|e| e.to_string()),
        semsurf_bind: compat.supports_semsurf_bind(),
        kind_override: compat.supports_kind_override(),
        external_memory: compat.supports_external_memory(),
        identify_object: compat.supports_identify_object(),
    }
}

pub fn recognized_versions() -> Vec<VersionFacts> {
    crate::compat::measured_versions().map(facts).collect()
}

/// Static facts for one version string, or the exact refusal Client::open gives.
pub fn version_facts(version: &str) -> std::result::Result<VersionFacts, String> {
    Compat::for_version(version)
        .map(facts)
        .map_err(|e| e.to_string())
}
