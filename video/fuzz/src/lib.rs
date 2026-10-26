// SPDX-License-Identifier: MIT
//! Drivers shared by the fuzz targets and by the regression tests in `nv-video`'s test suite: each
//! takes the raw fuzz input and runs one parser family over it. A driver never panics on its own;
//! every panic it surfaces is a parser bug.
#![forbid(unsafe_code)]

use nv_video::{
    Allocator, BuildContext, CodecDecoder, Job, SurfaceRef, VResult, av1, bits, h264, hdr, hevc,
    nvdec_hevc, nvdec_vp9,
};

/// H.264 SPS, PPS and slice headers (7.3.2.1, 7.3.2.2, 7.3.3) of every NAL in an Annex-B stream,
/// each against the last parameter sets that parsed.
pub fn h264_headers(data: &[u8]) {
    let mut sps = None;
    let mut pps = None;
    for n in bits::split_annexb(data) {
        let rbsp = bits::unescape(n.nal);
        match n.nal[0] & 0x1f {
            7 => sps = h264::ps::parse_sps(&rbsp).ok(),
            8 => {
                pps = h264::ps::parse_pps(&rbsp, |_| sps.clone())
                    .ok()
                    .zip(sps.clone());
            }
            1 | 5 => {
                let _hdr =
                    h264::slice::parse_slice_header(&rbsp, |_| pps.clone().map(|(p, s)| (s, p)));
            }
            _ => {}
        }
    }
}

/// HEVC SPS, PPS and slice segment headers (7.3.2.2, 7.3.2.3, 7.3.6) of every NAL in an Annex-B
/// stream, each against the last parameter sets that parsed.
pub fn hevc_headers(data: &[u8]) {
    let mut sps = None;
    let mut pps = None;
    for n in bits::split_annexb(data) {
        let rbsp = bits::unescape(n.nal);
        let nal_type = (n.nal[0] >> 1) & 0x3f;
        match nal_type {
            33 => sps = hevc::ps::parse_sps(&rbsp).ok(),
            34 => pps = hevc::ps::parse_pps(&rbsp).ok(),
            0..=31 => {
                let _hdr = hevc::slice::parse_slice_header(
                    &rbsp,
                    nal_type,
                    |_| pps.as_ref(),
                    |_| sps.as_ref(),
                );
            }
            _ => {}
        }
    }
}

/// The reference HEVC descriptor parser over a whole Annex-B stream.
pub fn hevc_reference(data: &[u8]) {
    let _parsed = nvdec_hevc::parse_hevc(data);
}

/// VP9 uncompressed + compressed headers (6.2, 6.3) of every frame of every superframe, with the
/// reference state carried across frames; the input is an IVF file or one raw chunk.
pub fn vp9_headers(data: &[u8]) {
    let mut state = nvdec_vp9::Vp9State::default();
    for packet in packets(data) {
        let Ok(frames) = nvdec_vp9::split_superframe(packet) else {
            continue;
        };
        for frame in frames {
            let _hdr = state.parse(frame);
        }
    }
}

/// AV1 OBU framing (5.3), sequence headers (5.5) and frame headers (5.9) with the parser state
/// carried across temporal units; the input is an IVF file or one raw temporal unit.
pub fn av1_headers(data: &[u8]) {
    let mut parser = av1::header::ParserState::default();
    for tu in packets(data) {
        let Ok(obus) = av1::obu::split_obus(tu) else {
            continue;
        };
        for o in obus {
            match o.obu_type {
                av1::obu::OBU_SEQUENCE_HEADER => {
                    if let Ok(s) = av1::header::SequenceHeader::parse(o.payload) {
                        parser.seq = Some(s);
                    }
                }
                av1::obu::OBU_FRAME_HEADER | av1::obu::OBU_FRAME => {
                    if let Ok(h) = parser.parse_frame_header(o.payload, o.temporal_id, o.spatial_id)
                    {
                        parser.update_refs(&h);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Container splitters: IVF (VP9 and AV1 readers), VP9 superframe index (Annex B), AV1 OBU
/// framing, Annex-B access units.
pub fn containers(data: &[u8]) {
    let _vp9_ivf = nvdec_vp9::ivf_frames(data);
    let _av1_ivf = av1::obu::ivf_frames(data);
    let _superframe = nvdec_vp9::split_superframe(data);
    let _obus = av1::obu::split_obus(data);
    let _avc = bits::access_units(data, false);
    let _hevc = bits::access_units(data, true);
}

/// HDR static metadata (H.264/HEVC SEI, AV1 metadata OBUs) of one unit for every codec; the first
/// byte selects the codec.
pub fn hdr_metadata(data: &[u8]) {
    let Some((&sel, unit)) = data.split_first() else {
        return;
    };
    let _meta = hdr::parse_unit(u32::from(sel % 4) + 1, unit);
}

/// IVF packets, or the whole input as one packet.
fn packets(data: &[u8]) -> Vec<&[u8]> {
    if data.starts_with(b"DKIF")
        && let Ok(p) = nvdec_vp9::ivf_frames(data)
    {
        return p;
    }
    vec![data]
}

/// Side-buffer allocator handing out distinct fake addresses; writes are bounds-checked like the
/// session's.
struct FakeVram {
    next: u64,
    sizes: Vec<(u64, usize)>,
}

impl Allocator for FakeVram {
    fn vram(&mut self, size: usize) -> VResult<u64> {
        let va = self.next;
        let Ok(size64) = u64::try_from(size.next_multiple_of(256)) else {
            return nv_video::bitstream("side buffer larger than the address space");
        };
        self.next += size64 + 256;
        self.sizes.push((va, size));
        Ok(va)
    }
    fn write(&mut self, va: u64, offset: usize, data: &[u8]) -> VResult<()> {
        let Some(&(_, size)) = self.sizes.iter().find(|(v, _)| *v == va) else {
            return nv_video::bitstream("write to an unknown side buffer");
        };
        if offset + data.len() > size {
            return nv_video::bitstream("side-buffer write out of range");
        }
        Ok(())
    }
}

fn boxed(d: impl CodecDecoder + 'static) -> Box<dyn CodecDecoder> {
    Box::new(d)
}

/// Session surfaces of the fuzzed decoders.
const SURFACES: usize = 24;

/// A whole decoder (parse, admission, descriptor and method build, reference update) the way the
/// session drives it, without a GPU: the first byte selects the codec, the rest is the stream.
pub fn decoder(data: &[u8]) {
    let Some((&sel, stream)) = data.split_first() else {
        return;
    };
    let (w, h) = (640, 480);
    let class = 0xcfb0;
    let dec: VResult<Box<dyn CodecDecoder>> = match sel % 4 {
        0 => h264::codec::H264Decoder::new(class, w, h, &[]).map(boxed),
        1 => hevc::codec::HevcDecoder::new(class, w, h, 8, &[]).map(boxed),
        2 => Ok(boxed(nv_video::vp9::Vp9Decoder::new(class, w, h, 8))),
        _ => av1::codec::Av1Decoder::new(w, h, 8, &[]).map(boxed),
    };
    let Ok(mut dec) = dec else { return };
    let mut alloc = FakeVram {
        next: 1 << 32,
        sizes: Vec::new(),
    };
    if dec.allocate(&mut alloc, w, h).is_err() {
        return;
    }
    let layout = dec.surface_layout(w, h);
    let Ok(surface_bytes) = u64::try_from(layout.size) else {
        return;
    };
    let surfaces: Vec<SurfaceRef> = (0..SURFACES)
        .scan(1u64 << 36, |base, _| {
            let luma = *base;
            *base += surface_bytes;
            Some(SurfaceRef {
                luma,
                chroma: luma + layout.chroma_offset,
            })
        })
        .collect();
    let units: Vec<&[u8]> = match sel % 4 {
        0 => bits::access_units(stream, false),
        1 => bits::access_units(stream, true),
        _ => packets(stream),
    };
    let mut picture_index = 0u32;
    for unit in units {
        if dec.surfaces_needed(unit).is_err() || dec.push_unit(unit).is_err() {
            continue;
        }
        for _ in 0..64 {
            let job = match dec.next_job() {
                Ok(Some(Job::Decode(job))) => job,
                Ok(Some(Job::ShowExisting { .. })) => continue,
                Ok(None) | Err(_) => break,
            };
            let referenced = dec.referenced();
            let Some(current) = (0..SURFACES).find(|s| !referenced.contains(s)) else {
                break;
            };
            let ctx = BuildContext {
                decoder_class: class,
                host_class: 0xc96f,
                descriptor: 1 << 34,
                bitstream: 1 << 35,
                bitstream_len: job.bitstream.len(),
                picture_index,
                current,
                surfaces: &surfaces,
                layout,
            };
            picture_index = picture_index.wrapping_add(1);
            let mut desc = vec![0u8; 0x5000];
            if dec.build(&job, &ctx, &mut desc).is_err() {
                break;
            }
            let readback = job
                .readback
                .clone()
                .map(|r| desc.get(r).map(<[u8]>::to_vec));
            let readback = match readback {
                Some(Some(bytes)) => Some(bytes),
                Some(None) => break,
                None => None,
            };
            if dec.finish(&job, current, readback.as_deref()).is_err() {
                break;
            }
        }
    }
}
