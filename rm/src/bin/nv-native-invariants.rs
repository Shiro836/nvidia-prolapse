// SPDX-License-Identifier: MIT
//! vast.ai hardware-invariant probe (feature `diagnostics`).
//! Emits exactly one JSON object on stdout; exit 0 = pass, 1 = fail.
//!   versions        every exact release compiled in, with static predicates
//!   version VER     static facts for VER, or the exact refusal
//!   caps            running driver: class list, SM ISA, capability selection,
//!                   semaphore-surface layout/caps, BAR1 (card info fb_size)
use nv_bionic_native::{capabilities, diagnostics, rm};

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
fn res(r: &Result<(), String>) -> String {
    match r {
        Ok(()) => "\"ok\"".into(),
        Err(e) => esc(e),
    }
}
fn facts_json(f: &diagnostics::VersionFacts) -> String {
    format!(
        "{{\"version\":{},\"client_abi\":{},\"semsurf_fences\":{},\"semsurf_bind\":{},\"kind_override\":{},\"external_memory\":{},\"identify_object\":{}}}",
        esc(f.version),
        res(&f.client_abi),
        res(&f.semsurf_fences),
        f.semsurf_bind,
        f.kind_override,
        f.external_memory,
        f.identify_object
    )
}

fn caps() -> (bool, String) {
    let libc = if cfg!(target_os = "android") {
        "bionic"
    } else {
        "gnu"
    };
    let arch = std::env::consts::ARCH;
    let head = format!(
        "\"probe\":\"inv-caps\",\"libc\":{},\"arch\":{}",
        esc(libc),
        esc(arch)
    );
    let version = match rm::runtime_version() {
        Ok(v) => v,
        Err(e) => {
            return (
                false,
                format!(
                    "{{{head},\"verdict\":\"fail\",\"stage\":\"read_version\",\"error\":{}}}",
                    esc(&e.to_string())
                ),
            );
        }
    };
    let statics = match diagnostics::version_facts(&version) {
        Ok(f) => facts_json(&f),
        Err(e) => format!("{{\"refused\":{}}}", esc(&e)),
    };
    let head = format!(
        "{head},\"running_version\":{},\"static\":{statics}",
        esc(&version)
    );
    let client = match rm::Client::open() {
        Ok(c) => c,
        Err(e) => {
            return (
                false,
                format!(
                    "{{{head},\"verdict\":\"fail\",\"stage\":\"client_open\",\"error\":{}}}",
                    esc(&e.to_string())
                ),
            );
        }
    };
    let cards = match client.cards() {
        Ok(c) => c,
        Err(e) => {
            return (
                false,
                format!(
                    "{{{head},\"verdict\":\"fail\",\"stage\":\"cards\",\"error\":{}}}",
                    esc(&e.to_string())
                ),
            );
        }
    };
    if cards.is_empty() {
        return (
            false,
            format!(
                "{{{head},\"verdict\":\"fail\",\"stage\":\"cards\",\"error\":\"RM enumerated no NVIDIA GPU\"}}"
            ),
        );
    }
    let mut ok = true;
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for card in cards {
        // Containers often list every host GPU but grant one /dev/nvidiaN.
        let node = format!("/dev/nvidia{}", card.minor);
        if let Err(e) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&node)
        {
            if matches!(
                e.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::NotFound
            ) {
                skipped.push(format!(
                    "{{\"minor\":{},\"error\":{}}}",
                    card.minor,
                    esc(&e.to_string())
                ));
                continue;
            }
        }
        let pci = format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            card.pci.domain, card.pci.bus, card.pci.slot, card.pci.function
        );
        let mut fields = vec![
            format!("\"pci\":{}", esc(&pci)),
            format!(
                "\"vendor_id\":\"0x{:04x}\",\"device_id\":\"0x{:04x}\"",
                card.pci.vendor, card.pci.device
            ),
            format!(
                "\"gpu_id\":\"0x{:x}\",\"minor\":{}",
                card.gpu_id, card.minor
            ),
            format!("\"card_fb_size\":{}", card.framebuffer_size),
        ];
        match client.device(card) {
            Err(e) => {
                ok = false;
                fields.push(format!("\"error\":{}", esc(&format!("device: {e}"))));
            }
            Ok(device) => {
                let sm = device.sm_version();
                let classes = device.classes();
                match &sm {
                    Ok(v) => fields.push(format!("\"rm_sm_isa\":\"0x{v:x}\"")),
                    Err(e) => {
                        ok = false;
                        fields.push(format!("\"rm_sm_isa_error\":{}", esc(&e.to_string())))
                    }
                }
                match &classes {
                    Ok(c) => {
                        let mut c = c.clone();
                        c.sort_unstable();
                        let list: Vec<String> = c.iter().map(|x| format!("\"0x{x:x}\"")).collect();
                        fields.push(format!(
                            "\"class_count\":{},\"classes\":[{}]",
                            c.len(),
                            list.join(",")
                        ));
                    }
                    Err(e) => {
                        ok = false;
                        fields.push(format!("\"classes_error\":{}", esc(&e.to_string())))
                    }
                }
                if let (Ok(sm), Ok(classes)) = (&sm, &classes) {
                    match capabilities::select(classes, *sm) {
                        Ok(c) => fields.push(format!(
                            "\"selection\":{{\"sm\":{},\"channel\":\"0x{:x}\",\"usermode\":{},\"copy\":\"0x{:x}\",\"eng2d\":\"0x{:x}\",\"eng3d\":\"0x{:x}\",\"m2mf\":\"0x{:x}\",\"compute\":\"0x{:x}\",\"generic_kind\":\"0x{:x}\",\"submission\":\"{:?}\"}}",
                            c.sm,
                            c.channel,
                            c.usermode.map_or("null".to_string(), |u| format!("\"0x{u:x}\"")),
                            c.copy,
                            c.eng2d,
                            c.eng3d,
                            c.m2mf,
                            c.compute,
                            c.generic_kind,
                            c.submission
                        )),
                        Err(e) => {
                            ok = false;
                            fields.push(format!("\"selection_error\":{}", esc(&e.to_string())))
                        }
                    }
                }
                match diagnostics::semaphore_layout(&device) {
                    Ok(l) => {
                        let mode = match &l.mode {
                            Ok(m) => format!("\"mode\":{}", esc(m)),
                            Err(e) => {
                                ok = false;
                                format!("\"mode_error\":{}", esc(e))
                            }
                        };
                        fields.push(format!(
                            "\"semaphore\":{{\"caps\":\"0x{:x}\",\"monitored_cap\":{},\"native64_cap\":{},\"max_submitted_offset\":{},\"monitored_offset\":{},\"stride\":{},{mode}}}",
                            l.caps,
                            l.caps & 1 != 0,
                            l.caps & 2 != 0,
                            l.max_submitted_offset,
                            l.monitored_offset,
                            l.stride
                        ));
                    }
                    Err(e) => {
                        ok = false;
                        fields.push(format!("\"semaphore_error\":{}", esc(&e.to_string())))
                    }
                }
            }
        }
        out.push(format!("{{{}}}", fields.join(",")));
    }
    if out.is_empty() {
        ok = false;
    }
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let blob = maps.lines().any(|l| {
        l.contains("libnvidia-") || l.contains("libEGL_nvidia") || l.contains("libnvglibc")
    });
    if blob {
        ok = false;
    }
    (
        ok,
        format!(
            "{{{head},\"verdict\":\"{}\",\"nvidia_userspace_mapped\":{blob},\"gpus\":[{}],\"skipped_unassigned\":[{}]}}",
            if ok { "pass" } else { "fail" },
            out.join(","),
            skipped.join(",")
        ),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (ok, text) = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["versions"] => {
            let all: Vec<String> = diagnostics::recognized_versions()
                .iter()
                .map(facts_json)
                .collect();
            (true, format!("[{}]", all.join(",\n")))
        }
        ["version", v] => match diagnostics::version_facts(v) {
            Ok(f) => (true, facts_json(&f)),
            Err(e) => (
                false,
                format!("{{\"version\":{},\"refused\":{}}}", esc(v), esc(&e)),
            ),
        },
        ["caps"] => caps(),
        _ => {
            eprintln!("usage: nv-native-invariants versions | version VER | caps");
            std::process::exit(2);
        }
    };
    println!("{text}");
    std::process::exit(if ok { 0 } else { 1 });
}
