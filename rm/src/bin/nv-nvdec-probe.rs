// SPDX-License-Identifier: MIT
//! V0 NVDEC spike driver: decode an HEVC fixture through the native RM
//! backend and compare against a software NV12 reference.
//! usage: nv-nvdec-probe <fixture-dir> <out-dir> [max-frames] [reference.nv12]
//!        nv-nvdec-probe vp9 <in.ivf> <reference.nv12|-> [descriptor-dump]
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::format_push_string,
    clippy::naive_bytecount,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "pre-standard code; cleanup tracked"
)]
use nv_bionic_native::{
    nvdec::{self, HevcFrame, HevcStream},
    rm::{Client, Error},
};

fn io<T>(what: &'static str, r: std::io::Result<T>) -> Result<T, Error> {
    r.map_err(|e| Error::Io(what, e))
}

fn open_device() -> Result<std::rc::Rc<nv_bionic_native::rm::Device>, Error> {
    let client = Client::open()?;
    let card = client
        .cards()?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Contract("no NVIDIA card".into()))?;
    println!(
        "driver {}; GPU device {:04x}",
        client.version, card.pci.device
    );
    client.device(card)
}

/// VP9: decode an IVF file, compare every shown frame against a software
/// NV12 reference, report per-frame decode times.
fn run_vp9(args: &[String]) -> Result<(), Error> {
    if args.len() < 2 {
        return Err(Error::Contract(
            "usage: nv-nvdec-probe vp9 <in.ivf> <reference.nv12|-> [descriptor-dump]".into(),
        ));
    }
    let data = io("read ivf", std::fs::read(&args[0]))?;
    let packets = nv_bionic_native::vp9::ivf_frames(&data)?;
    let device = open_device()?;
    let start = std::time::Instant::now();
    let report = nvdec::decode_vp9(&device, &packets)?;
    let wall = start.elapsed();
    if let Some(p) = args.get(2) {
        io(
            "write descriptors",
            std::fs::write(p, report.descriptors.concat()),
        )?;
    }
    let reference = if args[1] == "-" {
        None
    } else {
        Some(io("read reference", std::fs::read(&args[1]))?)
    };
    let mut off = 0usize;
    let mut bad_frames = 0usize;
    let mut diff_total = 0usize;
    for (i, o) in report.outputs.iter().enumerate() {
        let nv12 = nvdec::untile_nv12(&o.surface, o.width, o.height, o.stride, o.chroma_offset);
        if i == 0 {
            let zeros = nv12.iter().filter(|b| **b == 0).count();
            println!("frame 0: {zeros} zero bytes of {}", nv12.len());
            if let Ok(p) = std::env::var("NVDEC_VP9_DUMP") {
                io("dump", std::fs::write(&p, &nv12))?;
                io("dump raw", std::fs::write(format!("{p}.raw"), &o.surface))?;
            }
        }
        if let Some(r) = &reference {
            let refp = r
                .get(off..off + nv12.len())
                .ok_or_else(|| Error::Contract(format!("reference shorter than frame {i}")))?;
            let diff = nv12.iter().zip(refp).filter(|(a, b)| a != b).count();
            if diff != 0 {
                bad_frames += 1;
                diff_total += diff;
                if bad_frames <= 5 {
                    println!("frame {i}: {diff} of {} bytes differ", nv12.len());
                }
            }
        }
        off += nv12.len();
    }
    for (i, sw) in report.status_words.iter().enumerate().take(3) {
        println!("decoded frame {i} status {sw:08x?}");
    }
    let mut times: Vec<_> = report.frame_times.clone();
    times.sort();
    let total: std::time::Duration = times.iter().sum();
    println!(
        "decoder class {:#x}: {} coded frames decoded, {} shown, {}x{}; per-frame submit->done mean {:?} median {:?} max {:?}; wall incl. parse+readback {:?}",
        report.decoder_class,
        report.decoded,
        report.outputs.len(),
        report.outputs.first().map_or(0, |o| o.width),
        report.outputs.first().map_or(0, |o| o.height),
        total / times.len().max(1) as u32,
        times.get(times.len() / 2).copied().unwrap_or_default(),
        times.last().copied().unwrap_or_default(),
        wall
    );
    if let Some(r) = &reference {
        if off != r.len() {
            println!("MISMATCH: decoded {off} bytes, reference has {}", r.len());
        } else if bad_frames == 0 {
            println!(
                "PASS: all {} frames bit-exact against software decode",
                report.outputs.len()
            );
        } else {
            println!("MISMATCH: {bad_frames} frames differ ({diff_total} bytes)");
        }
    }
    Ok(())
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("vp9") {
        return run_vp9(&args[1..]);
    }
    if args.len() < 2 {
        return Err(Error::Contract(
            "usage: nv-nvdec-probe <fixture-dir> <out-dir> [max-frames] [reference.nv12]".into(),
        ));
    }
    let dir = std::path::Path::new(&args[0]);
    let out = std::path::Path::new(&args[1]);
    io("create out dir", std::fs::create_dir_all(out))?;
    // Raw .hevc input is parsed by the Rust HEVC parser; a directory is a
    // V0 offline fixture (stream.hevc, desc.bin, frames.txt).
    let (bitstream, descriptors, mut frames, luma_w, luma_h, w, h);
    if dir.is_file() {
        bitstream = io("read stream", std::fs::read(dir))?;
        let p = nv_bionic_native::nvdec_hevc::parse_hevc(&bitstream)?;
        (descriptors, frames) = (p.descriptors, p.frames);
        (luma_w, luma_h, w, h) = (p.luma_w, p.luma_h, p.width, p.height);
    } else {
        bitstream = io("read stream", std::fs::read(dir.join("stream.hevc")))?;
        descriptors = io("read desc", std::fs::read(dir.join("desc.bin")))?;
        let table = io(
            "read frames",
            std::fs::read_to_string(dir.join("frames.txt")),
        )?;
        let mut lines = table.lines();
        let geom: Vec<usize> = lines
            .next()
            .unwrap_or("")
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        (luma_w, luma_h, w, h) = (geom[0], geom[1], geom[2], geom[3]);
        frames = lines
            .map(|l| {
                let v: Vec<usize> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
                HevcFrame {
                    offset: v[0],
                    size: v[1],
                    pos: v[2] as u32,
                    is_hist: v[4] != 0,
                }
            })
            .collect();
    }
    if let Some(max) = args.get(2) {
        frames.truncate(max.parse().unwrap());
    }
    let client = Client::open()?;
    let card = client
        .cards()?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Contract("no NVIDIA card".into()))?;
    println!(
        "driver {}; GPU device {:04x}",
        client.version, card.pci.device
    );
    let device = client.device(card)?;
    let stream = HevcStream {
        bitstream: &bitstream,
        descriptors: &descriptors,
        frames: &frames,
        luma_w,
        luma_h,
    };
    let report = nvdec::decode_hevc(&device, &stream)?;
    println!(
        "decoder class {:#x}: {} frames decoded",
        report.decoder_class,
        report.surfaces.len()
    );
    let reference = match args.get(3) {
        Some(p) => Some(io("read reference", std::fs::read(p))?),
        None => None,
    };
    let frame_bytes = w * h + w * h.div_ceil(2);
    let mut all_exact = reference.is_some();
    for (i, surface) in report.surfaces.iter().enumerate() {
        let nv12 = nvdec::untile_nv12(surface, w, h, report.stride, report.chroma_offset);
        io(
            "write frame",
            std::fs::write(out.join(format!("frame{i:02}.nv12")), &nv12),
        )?;
        let st = report.status_words[i];
        let mut line = format!("frame {i}: {:?} status {:08x?}", report.frame_times[i], st);
        if let Some(r) = &reference {
            let refp = &r[i * frame_bytes..(i + 1) * frame_bytes];
            let mut se = 0f64;
            let mut diff = 0usize;
            let mut maxd = 0i32;
            for (a, b) in nv12.iter().zip(refp) {
                let d = *a as i32 - *b as i32;
                if d != 0 {
                    diff += 1;
                    maxd = maxd.max(d.abs());
                }
                se += (d * d) as f64;
            }
            let mse = se / nv12.len() as f64;
            let psnr = if mse == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (255.0f64 * 255.0 / mse).log10()
            };
            all_exact &= diff == 0;
            line += &format!(
                "; vs software: {diff} of {} bytes differ, max |d| {maxd}, PSNR {psnr:.2} dB",
                nv12.len()
            );
        }
        println!("{line}");
    }
    if reference.is_some() {
        println!(
            "{}",
            if all_exact {
                "PASS: bit-exact against software decode"
            } else {
                "MISMATCH against software decode"
            }
        );
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("nv-nvdec-probe: {e}");
        std::process::exit(1);
    }
}
