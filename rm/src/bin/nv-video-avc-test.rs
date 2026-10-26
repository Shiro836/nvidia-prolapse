// SPDX-License-Identifier: MIT
//! GNU host test of nv_video.h through its C ABI: every frame goes through
//! nv_video_decode -> nv_video_next_picture -> nv_video_copy_out (LINEAR NV12
//! dma-buf target) -> wait release fence -> CPU readback -> compare with a
//! software (FFmpeg) NV12 reference. Annex-B H.264 (.h264/.264) or HEVC (.hevc/.265), one
//! access unit per nv_video_decode() call.
//! usage: nv-video-avc-test <in.h264|in.hevc> <reference.nv12|-> <max_w> <max_h>
//! NV_VIDEO_TARGET=yuv420p: copy out to a planar Y/Cb/Cr target (YV12 plane order and pitches)
//! and compare it repacked as NV12.
//! NV_VIDEO_BIT_DEPTH=10: create a 10-bit session and copy out P010 (references and dumps are then
//! p010le: 16-bit little-endian samples, 10 MSBs).
//! NV_VIDEO_DUMP=<out.nv12>: also write every output picture (display order,
//! display window, tightly packed NV12, concatenated) to that file.
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::as_conversions,
    clippy::borrow_as_ptr,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "pre-standard code; cleanup tracked"
)]
#![cfg_attr(
    not(test),
    expect(
        clippy::wildcard_imports,
        reason = "pre-standard code; cleanup tracked"
    )
)]
use nv_bionic_native::video::{
    CODEC_H264, CODEC_HEVC, FORMAT_NV12, FORMAT_P010, FORMAT_YUV420P, TestDmaBuf, VideoDevice,
    bits::access_units, ffi::*,
};
use std::time::{Duration, Instant};

#[path = "common/nv_video.rs"]
mod nv_video;
use nv_video::{Device, Session, wait_fd, wait_fd_keep};
use std::os::fd::OwnedFd;

fn check(r: i32, what: &str) {
    if r < 0 {
        eprintln!("{what}: error {r}: {}", nv_video::last_error());
        std::process::exit(2);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = std::fs::read(&args[0]).expect("read stream");
    let hevc = args[0].ends_with(".hevc") || args[0].ends_with(".265");
    let packets = access_units(&data, hevc);
    let reference = (args.get(1).map(String::as_str) != Some("-"))
        .then(|| std::fs::read(&args[1]).expect("read reference"));
    let max_w: u32 = args[2].parse().unwrap();
    let max_h: u32 = args[3].parse().unwrap();

    let (r, opened) = Device::open();
    check(r, "device_open");
    let (dev, info) = opened.expect("nv_video_device_open returned a device");
    let name = String::from_utf8_lossy(&info.name);
    println!(
        "device {:04x} {} class {:#x} engines {}",
        info.pci_device,
        name.trim_end_matches('\0'),
        info.nvdec_class,
        info.nvdec_engines
    );
    // NV_VIDEO_REPORT=1: per-unit picture counts (nv_video_decode's ABI 2 return) and the
    // HDR static metadata of the first picture that carries any.
    let report = std::env::var_os("NV_VIDEO_REPORT").is_some();
    let bit_depth: u32 =
        std::env::var("NV_VIDEO_BIT_DEPTH").map_or(8, |v| v.parse().expect("NV_VIDEO_BIT_DEPTH"));
    let bps = if bit_depth > 8 { 2usize } else { 1 };
    let planar = match std::env::var("NV_VIDEO_TARGET").as_deref() {
        Ok("yuv420p") => true,
        Ok("nv12") | Err(_) => false,
        Ok(other) => panic!("NV_VIDEO_TARGET={other}: nv12 or yuv420p"),
    };
    let si = NvVideoSessionInfo {
        codec: if hevc { CODEC_HEVC } else { CODEC_H264 },
        max_width: max_w,
        max_height: max_h,
        bit_depth,
        extra_output_slots: 2,
        codec_config: std::ptr::null(),
        codec_config_len: 0,
        reserved: [0; 8],
    };
    let (r, s) = Session::create(&dev, si, &[]);
    check(r, "session_create");
    let mut s = s.expect("nv_video_session_create returned a session");
    let vdev = VideoDevice::open().expect("device");
    let pitch = (max_w * bps as u32).next_multiple_of(256);
    let luma = pitch as usize * max_h as usize;
    let size = luma + pitch as usize * max_h.div_ceil(2) as usize;
    let target = TestDmaBuf::new(&vdev, size).expect("dma-buf");
    use std::os::fd::{AsFd, AsRawFd};
    let tfd = target.fd.as_raw_fd();

    let mut off = 0usize;
    let (mut frames, mut bad, mut diff_total) = (0usize, 0usize, 0usize);
    let mut dec_times = Vec::new();
    let mut copy_times = Vec::new();
    // NV_VIDEO_MEASURE_ACQUIRE=1: pass the previous picture's (already signalled) release
    // fence back as the acquire fence, the steady state of a media pipeline, and time the
    // copy_out call alone. The wait on a signalled sync_file is one poll(2).
    let measure_acquire = std::env::var_os("NV_VIDEO_MEASURE_ACQUIRE").is_some();
    let mut acquire: Option<OwnedFd> = None;
    let mut copy_call: Vec<Duration> = Vec::new();
    let mut out = vec![0u8; size];
    let mut dump = std::env::var_os("NV_VIDEO_DUMP")
        .map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("create NV_VIDEO_DUMP")));
    let mut handle = |s: &mut Session<'_>,
                      frames: &mut usize,
                      bad: &mut usize,
                      diff_total: &mut usize,
                      off: &mut usize,
                      copy_times: &mut Vec<Duration>,
                      acquire: &mut Option<OwnedFd>,
                      copy_call: &mut Vec<Duration>| {
        loop {
            let (r, pic) = s.next_picture();
            check(r, "next_picture");
            if r == 0 {
                break;
            }
            if report {
                let (r, hdr) = s.picture_hdr(pic.id);
                check(r, "picture_hdr");
                if hdr.has_mastering_display != 0 || hdr.has_content_light_level != 0 {
                    println!(
                        "picture {} HDR: mdcv {} primaries (SEI order G,B,R) ({},{}) ({},{}) ({},{}) WP({},{}) \
                         L({},{}) CLL {} ({},{})",
                        pic.id,
                        hdr.has_mastering_display,
                        hdr.mastering_display.display_primaries_x[0],
                        hdr.mastering_display.display_primaries_y[0],
                        hdr.mastering_display.display_primaries_x[1],
                        hdr.mastering_display.display_primaries_y[1],
                        hdr.mastering_display.display_primaries_x[2],
                        hdr.mastering_display.display_primaries_y[2],
                        hdr.mastering_display.white_point_x,
                        hdr.mastering_display.white_point_y,
                        hdr.mastering_display.max_luminance,
                        hdr.mastering_display.min_luminance,
                        hdr.has_content_light_level,
                        hdr.max_content_light_level,
                        hdr.max_pic_average_light_level
                    );
                }
            }
            let (w, h) = (pic.crop_width as usize, pic.crop_height as usize);
            target.fill(0xa5).unwrap();
            // NV_VIDEO_TARGET=yuv420p: planar target laid out like an Android YV12 buffer (Y, then
            // Cr, then Cb; chroma pitch = luma pitch / 2), repacked to NV12 for the comparison.
            let cpitch = pitch / 2;
            let cr = luma as u64;
            let cb = cr + u64::from(cpitch) * u64::from(max_h.div_ceil(2));
            let t = NvVideoTarget {
                dmabuf_fd: [tfd, tfd],
                offset: [0, if planar { cb } else { luma as u64 }],
                pitch: [pitch, if planar { cpitch } else { pitch }],
                modifier: 0,
                format: if planar {
                    FORMAT_YUV420P
                } else if bps == 2 {
                    FORMAT_P010
                } else {
                    FORMAT_NV12
                },
                width: max_w,
                height: max_h,
                acquire_fence_fd: match acquire.as_ref() {
                    Some(fd) if measure_acquire => fd.as_raw_fd(),
                    _ => -1,
                },
                cr_dmabuf_fd: if planar { tfd } else { -1 },
                cr_pitch: if planar { cpitch } else { 0 },
                cr_offset: if planar { cr } else { 0 },
                reserved: [0; 2],
            };
            let lent: Vec<_> = std::iter::once(target.fd.as_fd())
                .chain(acquire.as_ref().map(AsFd::as_fd))
                .collect();
            let t0 = Instant::now();
            let (r, fence) = s.copy_out(pic.id, &t, &lent);
            check(r, "copy_out");
            copy_call.push(t0.elapsed());
            drop(lent);
            if measure_acquire {
                // The acquire fd is borrowed for the call, ours afterwards.
                drop(acquire.take());
                let Some(fence) = fence else {
                    eprintln!("release fence did not signal");
                    std::process::exit(3);
                };
                wait_fd_keep(fence.as_fd());
                *acquire = Some(fence);
            } else {
                wait_fd(fence);
            }
            copy_times.push(t0.elapsed());
            target.read(0, &mut out).unwrap();
            let mut nv12 = Vec::with_capacity(w * h * 3 / 2 * bps);
            for y in 0..h {
                nv12.extend_from_slice(&out[y * pitch as usize..][..w * bps]);
            }
            for y in 0..h.div_ceil(2) {
                if planar {
                    let row =
                        |base: u64| &out[base as usize + y * cpitch as usize..][..w.div_ceil(2)];
                    for (u, v) in row(cb).iter().zip(row(cr)) {
                        nv12.extend_from_slice(&[*u, *v]);
                    }
                } else {
                    nv12.extend_from_slice(
                        &out[luma + y * pitch as usize..][..w.div_ceil(2) * 2 * bps],
                    );
                }
            }
            if let Some(d) = dump.as_mut() {
                std::io::Write::write_all(d, &nv12).expect("write NV_VIDEO_DUMP");
            }
            if let Some(r) = &reference {
                let refp = &r[*off..*off + nv12.len()];
                let d = nv12.iter().zip(refp).filter(|(a, b)| a != b).count();
                if d != 0 {
                    *bad += 1;
                    *diff_total += d;
                    if *bad <= 5 {
                        println!(
                            "frame {}: {d} of {} bytes differ ({w}x{h})",
                            *frames,
                            nv12.len()
                        );
                    }
                }
            }
            *off += nv12.len();
            *frames += 1;
            check(s.picture_release(pic.id), "release");
        }
    };
    let start = Instant::now();
    let mut empty_units = 0usize;
    for (i, p) in packets.iter().enumerate() {
        let t0 = Instant::now();
        let shown = s.decode(Some(p), i as u64, 0);
        check(shown, "decode");
        if shown == 0 {
            empty_units += 1;
        }
        if report {
            println!("unit {i}: {shown} displayable pictures, {} bytes", p.len());
        }
        dec_times.push(t0.elapsed());
        handle(
            &mut s,
            &mut frames,
            &mut bad,
            &mut diff_total,
            &mut off,
            &mut copy_times,
            &mut acquire,
            &mut copy_call,
        );
    }
    check(s.decode(None, 0, 1), "drain");
    handle(
        &mut s,
        &mut frames,
        &mut bad,
        &mut diff_total,
        &mut off,
        &mut copy_times,
        &mut acquire,
        &mut copy_call,
    );
    let wall = start.elapsed();
    if let Some(d) = dump.as_mut() {
        std::io::Write::flush(d).expect("flush NV_VIDEO_DUMP");
    }
    let stat = |v: &mut Vec<Duration>| {
        v.sort();
        let total: Duration = v.iter().sum();
        (
            total / v.len().max(1) as u32,
            v.get(v.len() / 2).copied().unwrap_or_default(),
            v.last().copied().unwrap_or_default(),
        )
    };
    let (dm, dmed, dmax) = stat(&mut dec_times);
    let (cm, cmed, cmax) = stat(&mut copy_times);
    let (mm, mmed, _) = stat(&mut copy_call.clone());
    println!(
        "copy_out call mean {mm:?} median {mmed:?}{}",
        if measure_acquire {
            " (acquire fence = the previous picture's release fence, already signalled)"
        } else {
            ""
        }
    );
    println!(
        "{} packets ({empty_units} without a displayable picture), {frames} pictures, max {max_w}x{max_h}; nv_video_decode mean {dm:?} median {dmed:?} max {dmax:?}; copy_out+fence mean {cm:?} median {cmed:?} max {cmax:?}; wall {wall:?}",
        packets.len()
    );
    drop(s);
    drop(dev);
    if let Some(r) = &reference {
        if off != r.len() {
            println!("MISMATCH: produced {off} bytes, reference {}", r.len());
            std::process::exit(1);
        } else if bad == 0 {
            println!("PASS: all {frames} pictures bit-exact against the FFmpeg software decode");
        } else {
            println!("MISMATCH: {bad} pictures differ ({diff_total} bytes)");
            std::process::exit(1);
        }
    }
}
