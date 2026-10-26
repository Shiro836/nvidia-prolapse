// SPDX-License-Identifier: MIT
//! AV1 on the nv_video session layer (`CodecDecoder`). Method list and buffer selection follow stock
//! libnvcuvid 615.71.09 as observed in RM traces; descriptor builder in `desc`.
//!
//! Deviation from stock (behaviour-neutral): stock fills an LRU CDF buffer with the default tables on the
//! CPU for every picture without a primary reference. Session side buffers are VRAM written once at
//! creation, so the four default q-context tables live in four constant buffers and an LRU index that
//! stock would have filled with defaults is aliased to the constant buffer (the engine never writes a
//! read buffer, so saving that index as a context stays valid).
use super::obu::Av1Error;
use crate::fit;
use nv_rm_core::conv::{u64_from_usize, usize_from_u32};

use super::{Av1Codec, Av1Out, CDF_BUFFER_SIZE, CDF_BUFFERS, DPB_SLOTS, PictureJob, desc};
use crate::{
    Allocator, BuildContext, CODEC_AV1, CodecDecoder, DecodeJob, Job, PictureInfo, VResult,
    VideoError,
};
use nv_rm_core::push::method;

const APPLICATION_ID_AV1: u32 = 0x12;
const CONTROL_PARAMS_AV1: u32 = 0x5205a;

fn align(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}
/// AV1_TEMPORAL_MV_BUF_SIZE / AV1_SEGMENT_ID_BUF_SIZE (nvdec_drv.h).
fn mv_seg_sizes(w: u32, h: u32) -> (u64, u64) {
    let sb64 = align(u64::from(w), 128) * align(u64::from(h), 128) / 4096;
    (align(sb64 * 256, 4096), align(sb64 * 128, 4096))
}

/// Where the pictures of a DPB slot live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surf {
    /// A session surface: reference and displayed picture are the same.
    Session(usize),
    /// Film-grain frame: the grain-free reference is the codec-private surface of the DPB slot (OFFSET8)
    /// and the session surface received the grain-applied picture (SET_DISPLAY_BUF 0x428/0x42c, as
    /// stock does). `display` is kept while the slot is referenced when the frame is
    /// showable, because a later show_existing_frame outputs it again: film grain is part of output, and
    /// show_existing_frame loads the stored grain parameters (spec 7.21 load_grain_params), so the grain
    /// output of the decode is exactly what the spec shows then. Stock under FFmpeg does the same: every
    /// grain frame, hidden ones included, gets 0x428/0x42c (traced on the BBB SVT film-grain=12 stream,
    /// 23 pictures), and show_existing_frame issues no engine work.
    Grain { display: Option<usize> },
}

pub struct Av1Decoder {
    c: Av1Codec,
    max: (u32, u32),
    /// Sequence header from codec_config declares film grain: grain-free reference surfaces exist.
    grain_capable: bool,
    private: [u64; DPB_SLOTS],
    chroma_offset: u64,
    slot_surface: [Option<Surf>; DPB_SLOTS],
    current: Option<Box<PictureJob>>,
    pending_refs: Vec<usize>,
    cdf: u64,
    cdf_default: [u64; 4],
    cdf_alias: [Option<u8>; CDF_BUFFERS],
    pool: u64,
    mv_size: u64,
    slot_size: u64,
    filter: u64,
    /// Session bit depth (8 or 10); surfaces are sized for it.
    bit_depth: u32,
}

impl Av1Decoder {
    /// `codec_config`: optional sequence header (raw OBUs or an `av1C` record). Film-grain streams need it:
    /// the grain-free reference surfaces are allocated at session creation.
    pub fn new(
        max_width: u32,
        max_height: u32,
        bit_depth: u32,
        codec_config: &[u8],
    ) -> VResult<Self> {
        let mut c = Av1Codec::new();
        let obus = if codec_config.first() == Some(&0x81) {
            &codec_config[4.min(codec_config.len())..]
        } else {
            codec_config
        };
        for o in super::obu::split_obus(obus)? {
            if o.obu_type == super::obu::OBU_SEQUENCE_HEADER {
                c.parser.seq = Some(super::header::SequenceHeader::parse(o.payload)?);
            }
        }
        let grain_capable = c
            .parser
            .seq
            .as_ref()
            .is_some_and(|s| s.film_grain_params_present);
        Ok(Self::with_codec(
            c,
            max_width,
            max_height,
            bit_depth,
            grain_capable,
        ))
    }

    fn with_codec(
        c: Av1Codec,
        max_width: u32,
        max_height: u32,
        bit_depth: u32,
        grain_capable: bool,
    ) -> Self {
        Self {
            c,
            bit_depth,
            grain_capable,
            private: [0; DPB_SLOTS],
            chroma_offset: 0,
            max: (max_width, max_height),
            slot_surface: [None; DPB_SLOTS],
            current: None,
            pending_refs: Vec::new(),
            cdf: 0,
            cdf_default: [0; 4],
            cdf_alias: [None; CDF_BUFFERS],
            pool: 0,
            mv_size: 0,
            slot_size: 0,
            filter: 0,
        }
    }

    fn info(&self, seq: &super::header::SequenceHeader, w: u32, h: u32, key: bool) -> PictureInfo {
        PictureInfo {
            width: w,
            height: h,
            crop: [0, 0, w, h],
            bit_depth: u32::from(seq.bit_depth),
            color_primaries: 2,
            transfer: 2,
            matrix: 2,
            full_range: seq.color_range,
            keyframe: key,
        }
    }

    fn cdf_addr(&self, idx: usize) -> u64 {
        match self.cdf_alias[idx] {
            Some(q) => self.cdf_default[usize::from(q)],
            None => self.cdf + u64_from_usize(idx * CDF_BUFFER_SIZE),
        }
    }
}

/// Count the frames a temporal unit decodes (OBU_FRAME / OBU_FRAME_HEADER without show_existing_frame).
fn frames_in_unit(data: &[u8]) -> VResult<usize> {
    let obus = super::obu::split_obus(data)?;
    let mut n = 0;
    for o in obus {
        if matches!(
            o.obu_type,
            super::obu::OBU_FRAME | super::obu::OBU_FRAME_HEADER
        ) && o.payload.first().is_some_and(|b| b & 0x80 == 0)
        {
            n += 1;
        }
    }
    Ok(n)
}

impl CodecDecoder for Av1Decoder {
    fn codec(&self) -> u32 {
        CODEC_AV1
    }
    fn bit_depth(&self) -> u32 {
        self.bit_depth
    }
    fn dpb_size(&self) -> usize {
        8
    }
    fn allocate(
        &mut self,
        alloc: &mut dyn Allocator,
        max_width: u32,
        max_height: u32,
    ) -> VResult<()> {
        self.cdf = alloc.vram(CDF_BUFFERS * CDF_BUFFER_SIZE)?;
        let mut b = vec![0u8; CDF_BUFFER_SIZE];
        for q in 0..4u8 {
            self.cdf_default[usize::from(q)] = alloc.vram(CDF_BUFFER_SIZE)?;
            super::write_default_cdfs(q, &mut b);
            alloc.write(self.cdf_default[usize::from(q)], 0, &b)?;
        }
        let (mv, seg) = mv_seg_sizes(max_width, max_height);
        self.mv_size = mv;
        self.slot_size = mv + seg;
        self.pool = alloc.vram(fit(
            self.slot_size * u64_from_usize(DPB_SLOTS),
            "DPB pool size",
        )?)?;
        self.filter = alloc.vram(fit(
            desc::filter_layout(max_width, max_height).total,
            "filter size",
        )?)?;
        if self.grain_capable {
            let l = self.surface_layout(max_width, max_height);
            self.chroma_offset = l.chroma_offset;
            for p in self.private.iter_mut() {
                *p = alloc.vram(l.size)?;
            }
        }
        Ok(())
    }
    fn surfaces_needed(&self, data: &[u8]) -> VResult<usize> {
        frames_in_unit(data)
    }
    fn push_unit(&mut self, data: &[u8]) -> VResult<()> {
        self.c.push_unit(data);
        Ok(())
    }
    fn next_job(&mut self) -> VResult<Option<Job>> {
        let Some(out) = self.c.next_out()? else {
            return Ok(None);
        };
        let seq = self
            .c
            .parser
            .seq
            .clone()
            .ok_or(Av1Error::NoSequenceHeader)?;
        if u32::from(seq.bit_depth) != self.bit_depth
            || seq.mono_chrome
            || !(seq.subsampling_x && seq.subsampling_y)
        {
            return Err(Av1Error::SessionFormat {
                bit_depth: seq.bit_depth,
                mono_chrome: seq.mono_chrome,
                subsampling_x: u8::from(seq.subsampling_x),
                subsampling_y: u8::from(seq.subsampling_y),
                session_bit_depth: self.bit_depth,
            }
            .into());
        }
        match out {
            Av1Out::ShowExisting { slot, hdr } => {
                let surface = match self.slot_surface[slot] {
                    Some(Surf::Session(s)) if !hdr.film_grain.apply_grain => s,
                    Some(Surf::Grain { display: Some(s) }) if hdr.film_grain.apply_grain => s,
                    Some(Surf::Grain { display: None }) => {
                        return Err(Av1Error::Invalid(
                            "show_existing_frame of a frame that is not showable",
                        )
                        .into());
                    }
                    Some(_) => {
                        return Err(VideoError::Invalid(
                            "AV1: film-grain state of the shown slot differs from its decode"
                                .into(),
                        ));
                    }
                    None => {
                        return Err(Av1Error::Invalid("show_existing of an undecoded slot").into());
                    }
                };
                let r = self.c.slot_record(slot);
                let info = self.info(&seq, r.upscaled_width, r.frame_height, hdr.frame_type == 0);
                Ok(Some(Job::ShowExisting { surface, info }))
            }
            Av1Out::Decode(j) => {
                let h = &j.hdr;
                if h.use_superres && !h.frame_is_intra {
                    // Stock libnvcuvid 615.78.08 decodes superres inter frames. This decoder now
                    // reproduces its picture descriptor for them (current frame info carries the
                    // coded width, the reference scale factors are upscaled-ref to coded-current,
                    // and the motion-field size gate finds no candidate), verified byte-for-byte
                    // against a runtime capture of the installed library; the engine still
                    // reports error_status 0xd on the fifth picture of the capture stream, so a
                    // field outside the compared region is still missing. Refuse rather than
                    // submit a descriptor that is known incomplete.
                    return Err(Av1Error::Unsupported("superres on inter frames").into());
                }
                if h.upscaled_width > self.max.0 || h.frame_height > self.max.1 {
                    // A shown key frame refreshes every reference slot (spec 5.9.2:
                    // refresh_frame_flags = allFrames).
                    let random_access = h.frame_type == super::header::KEY_FRAME && h.show_frame;
                    return Err(crate::too_large(
                        "AV1",
                        h.upscaled_width,
                        h.frame_height,
                        self.max,
                        random_access,
                    ));
                }
                if j.apply_grain && !self.grain_capable {
                    return Err(Av1Error::Unsupported(
                        "film grain: pass the sequence header as codec_config (reference surfaces are \
                         allocated at session creation)",
                    )
                    .into());
                }
                self.pending_refs = j
                    .ref_slots
                    .iter()
                    .flatten()
                    .filter_map(|s| match self.slot_surface[*s] {
                        Some(Surf::Session(x)) => Some(x),
                        // Grain references are read from the private surfaces.
                        _ => None,
                    })
                    .collect();
                self.slot_surface[j.slot] = None;
                let info = self.info(&seq, h.upscaled_width, h.frame_height, h.frame_type == 0);
                let job = DecodeJob {
                    info,
                    show: h.show_frame,
                    display_key: None,
                    bitstream: j.bitstream.clone(),
                    readback: None,
                };
                self.current = Some(j);
                Ok(Some(Job::Decode(job)))
            }
        }
    }
    fn build(&mut self, _job: &DecodeJob, ctx: &BuildContext, d: &mut [u8]) -> VResult<Vec<u32>> {
        let j = self
            .current
            .take()
            .ok_or_else(|| VideoError::Invalid("AV1 build without a parsed frame".into()))?;
        if ctx.bitstream_len != j.bitstream.len() {
            return Err(VideoError::Invalid("AV1 bitstream length mismatch".into()));
        }
        self.slot_surface[j.slot] = Some(if j.apply_grain {
            Surf::Grain {
                display: j.hdr.showable_frame.then_some(ctx.current),
            }
        } else {
            Surf::Session(ctx.current)
        });
        j.build_descriptor(ctx.layout.pitch, ctx.r#gen().blackwell_b(), d)?;
        if let Some(q) = j.cdf_default_qctx {
            self.cdf_alias[j.cdf_read] = Some(q);
        }
        let cdf_read = self.cdf_addr(j.cdf_read);
        self.cdf_alias[j.cdf_write] = None;
        let cdf_write = self.cdf_addr(j.cdf_write);
        let da = ctx.descriptor;
        let mut w = Vec::new();
        // Relocation encoding per host class (stock reloc_4a8a0).
        let r = |w: &mut Vec<u32>, m: u32, va: u64| ctx.reloc(w, m, va);
        let cur = ctx.current;
        let slot_ref = |sl: usize| -> crate::SurfaceRef {
            match self.slot_surface[sl] {
                Some(Surf::Session(x)) => ctx.surfaces[x],
                _ => {
                    let va = self.private[sl];
                    crate::SurfaceRef {
                        luma: va,
                        chroma: va + self.chroma_offset,
                    }
                }
            }
        };
        // Decode target (OFFSET8): the slot's reference surface.
        let target = slot_ref(j.slot);
        let ref_surface = |k: usize| j.ref_slots[k].map_or(target, slot_ref);
        // ERROR_FRM_IDX: picture index of LAST (stock); our surfaces have no CUVID index, use LAST's slot.
        let err_idx = if j.hdr.frame_is_intra {
            0
        } else {
            fit(
                j.ref_slots[0].unwrap_or(0),
                "error-concealment reference slot",
            )?
        };
        method(&mut w, 4, 0x200, &[APPLICATION_ID_AV1]);
        method(
            &mut w,
            4,
            0x400,
            &[CONTROL_PARAMS_AV1 | (err_idx & 0x1f) << 7],
        );
        r(&mut w, 0x404, da);
        r(&mut w, 0x408, ctx.bitstream);
        for k in 0..7u32 {
            let sr = ref_surface(usize_from_u32(k));
            r(&mut w, 0x430 + 4 * k, sr.luma);
            r(&mut w, 0x474 + 4 * k, sr.chroma);
        }
        r(&mut w, 0x414, self.pool);
        r(&mut w, 0x424, da + u64_from_usize(desc::DESC_STATUS));
        method(&mut w, 4, 0x40c, &[ctx.picture_index]);
        r(&mut w, 0x4cc, da + u64_from_usize(desc::DESC_TILE_INFO));
        r(&mut w, 0x660, da + u64_from_usize(desc::DESC_GLOBAL_MOTION));
        r(&mut w, 0x664, da + u64_from_usize(desc::DESC_FILM_GRAIN));
        let slot_va = |sl: usize| self.pool + u64_from_usize(sl) * self.slot_size;
        r(
            &mut w,
            0x648,
            j.seg_read
                .map_or(self.pool, |sl| slot_va(sl) + self.mv_size),
        );
        r(&mut w, 0x64c, slot_va(j.slot) + self.mv_size);
        for k in 0..3u32 {
            r(
                &mut w,
                0x650 + 4 * k,
                j.colmv_reads[usize_from_u32(k)].map_or(self.pool, slot_va),
            );
        }
        r(&mut w, 0x65c, slot_va(j.slot));
        let fl = desc::filter_layout(j.hdr.upscaled_width, j.hdr.frame_height);
        r(&mut w, 0x4d0, self.filter);
        r(&mut w, 0x4c8, self.filter + fl.intra_top);
        r(&mut w, 0x640, cdf_read);
        r(&mut w, 0x644, cdf_write);
        // 0x4dc on blackwell_b classes only (gens.rs DECODE_GATES 0x44daf; Ada trace: none).
        if ctx.r#gen().blackwell_b() {
            r(&mut w, 0x4dc, self.filter + fl.extra_4dc);
        }
        r(&mut w, 0x450, target.luma);
        r(&mut w, 0x494, target.chroma);
        if j.apply_grain {
            r(&mut w, 0x428, ctx.surfaces[cur].luma);
            r(&mut w, 0x42c, ctx.surfaces[cur].chroma);
        }
        method(&mut w, 4, 0x300, &[0]); // EXECUTE
        Ok(w)
    }
    fn finish(
        &mut self,
        _job: &DecodeJob,
        _current: usize,
        _readback: Option<&[u8]>,
    ) -> VResult<()> {
        self.pending_refs.clear();
        Ok(())
    }
    fn referenced(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .c
            .referenced_slots()
            .filter_map(|s| match self.slot_surface[s] {
                Some(Surf::Session(x)) | Some(Surf::Grain { display: Some(x) }) => Some(x),
                _ => None,
            })
            .collect();
        v.extend(self.pending_refs.iter().copied());
        v.sort_unstable();
        v.dedup();
        v
    }
    fn max_referenced(&self) -> Option<usize> {
        // The 8 reference_frame slots; the 9th DPB slot is the picture being decoded.
        Some(8)
    }
    fn status_offset(&self) -> Option<usize> {
        Some(desc::DESC_STATUS)
    }
    fn flush(&mut self) {
        self.c.flush();
        self.slot_surface = [None; DPB_SLOTS];
        self.current = None;
        self.pending_refs.clear();
        self.cdf_alias = [None; CDF_BUFFERS];
    }
}
