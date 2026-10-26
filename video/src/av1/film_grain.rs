// SPDX-License-Identifier: MIT
//! Film grain side buffer (`AV1FilmGrainMemory`, descriptor +0x1400): scaling LUTs and the cropped grain
//! templates, generated on the CPU exactly as AV1 spec 7.18.3.3 (generate_grain) and 7.18.3.4
//! (scaling lookup). Stock libnvcuvid writes the same bytes (verified on 8/8 captured
//! frames).

#[path = "gaussian.rs"]
mod gaussian;
use super::header::{FilmGrainParams, SequenceHeader};
use gaussian::GAUSSIAN_SEQUENCE;
use nv_rm_core::conv::low_u8;

pub const FILM_GRAIN_MEMORY_SIZE: usize = 3 * 256 + 2 * (4096 + 2 * 1024);

fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

struct Rng(u16);
impl Rng {
    fn next(&mut self, bits: u32) -> usize {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        let r = (r >> 1) | (bit << 15);
        self.0 = r;
        usize::from((r >> (16 - bits)) & ((1 << bits) - 1))
    }
}

fn scaling_lut(num: u8, value: &[u8], scaling: &[u8], lut: &mut [u8]) {
    let n = usize::from(num);
    if n == 0 {
        lut[..256].fill(0);
        return;
    }
    lut[..usize::from(value[0])].fill(scaling[0]);
    for i in 0..n - 1 {
        let dy = i32::from(scaling[i + 1]) - i32::from(scaling[i]);
        let dx = i32::from(value[i + 1]) - i32::from(value[i]);
        // Points must increase (6.8.20); a segment of width 0 or less has no samples to fill,
        // and its slope would divide by zero.
        if dx <= 0 {
            continue;
        }
        let delta = dy * ((65536 + (dx >> 1)) / dx);
        for (k, x) in (0..dx).enumerate() {
            // The interpolated scale is stored into a u8 LUT entry, keeping its low 8 bits as the
            // reference decoders' uint8_t LUTs do (dav1d src/filmgrain_tmpl.c generate_scaling).
            lut[usize::from(value[i]) + k] =
                low_u8((i32::from(scaling[i]) + ((x * delta + 32768) >> 16)).cast_unsigned());
        }
    }
    lut[usize::from(value[n - 1])..256].fill(scaling[n - 1]);
}

/// A grain template sample as the engine's 16-bit entry: every sample is a rounded Gaussian
/// value (|v| < 2^11) or was clamped to GrainMin..=GrainMax (at most 2^11 in magnitude for 12-bit).
fn grain_i16(v: i32) -> i16 {
    i16::try_from(v).expect("grain samples are within -2^11..2^11")
}

/// Fill `out` (`FILM_GRAIN_MEMORY_SIZE` bytes) for one picture.
///
/// # Panics
/// Never: grain samples are bounded by the clamp (see `grain_i16`).
pub fn build(seq: &SequenceHeader, g: &FilmGrainParams, out: &mut [u8]) {
    out[..FILM_GRAIN_MEMORY_SIZE].fill(0);
    let bd = u32::from(seq.bit_depth);
    let (sx, sy) = (
        usize::from(seq.subsampling_x),
        usize::from(seq.subsampling_y),
    );
    // Scaling LUTs (8-bit index).
    scaling_lut(
        g.num_y_points,
        &g.point_y_value,
        &g.point_y_scaling,
        &mut out[0..256],
    );
    if g.chroma_scaling_from_luma {
        let (a, b) = out.split_at_mut(256);
        b[..256].copy_from_slice(&a[..256]);
        b[256..512].copy_from_slice(&a[..256]);
    } else {
        scaling_lut(
            g.num_cb_points,
            &g.point_cb_value,
            &g.point_cb_scaling,
            &mut out[256..512],
        );
        scaling_lut(
            g.num_cr_points,
            &g.point_cr_value,
            &g.point_cr_scaling,
            &mut out[512..768],
        );
    }
    // Grain templates.
    let center = 128i32 << (bd - 8);
    let (gmin, gmax) = (-center, (256i32 << (bd - 8)) - 1 - center);
    let mut luma = vec![[0i32; 82]; 73];
    let shift = 12 - bd + u32::from(g.grain_scale_shift);
    let mut rng = Rng(g.grain_seed);
    for row in luma.iter_mut() {
        for v in row.iter_mut() {
            let x = if g.num_y_points > 0 {
                i32::from(GAUSSIAN_SEQUENCE[rng.next(11)])
            } else {
                0
            };
            *v = round2(x, shift);
        }
    }
    let ar_shift = u32::from(g.ar_coeff_shift_minus_6) + 6;
    let lag = isize::from(g.ar_coeff_lag);
    // The templates keep a 3-sample border and ar_coeff_lag <= 3, so y + dr and x + dc never go
    // below 0 (spec 7.18.3.3).
    let at = |v: usize, d: isize| v.checked_add_signed(d).expect("border 3 >= ar_coeff_lag");
    for y in 3..73 {
        for x in 3..82 - 3 {
            let (mut sum, mut pos) = (0i32, 0usize);
            'o: for dr in -lag..=0 {
                for dc in -lag..=lag {
                    if dr == 0 && dc == 0 {
                        break 'o;
                    }
                    let c = i32::from(g.ar_coeffs_y_plus_128[pos]) - 128;
                    sum += luma[at(y, dr)][at(x, dc)] * c;
                    pos += 1;
                }
            }
            luma[y][x] = (luma[y][x] + round2(sum, ar_shift)).clamp(gmin, gmax);
        }
    }
    let (cw, ch) = (if sx == 1 { 44 } else { 82 }, if sy == 1 { 38 } else { 73 });
    let mut cb = vec![vec![0i32; cw]; ch];
    let mut cr = vec![vec![0i32; cw]; ch];
    for (plane, seed_xor, n) in [
        (0, 0xb524u16, g.num_cb_points),
        (1, 0x49d8u16, g.num_cr_points),
    ] {
        let mut rng = Rng(g.grain_seed ^ seed_xor);
        let p = if plane == 0 { &mut cb } else { &mut cr };
        for row in p.iter_mut() {
            for v in row.iter_mut() {
                let x = if n > 0 || g.chroma_scaling_from_luma {
                    i32::from(GAUSSIAN_SEQUENCE[rng.next(11)])
                } else {
                    0
                };
                *v = round2(x, shift);
            }
        }
    }
    for y in 3..ch {
        for x in 3..cw - 3 {
            let (mut s0, mut s1, mut pos) = (0i32, 0i32, 0usize);
            'o: for dr in -lag..=0 {
                for dc in -lag..=lag {
                    let c0 = i32::from(g.ar_coeffs_cb_plus_128[pos]) - 128;
                    let c1 = i32::from(g.ar_coeffs_cr_plus_128[pos]) - 128;
                    if dr == 0 && dc == 0 {
                        if g.num_y_points > 0 {
                            let mut l = 0;
                            let lx = ((x - 3) << sx) + 3;
                            let ly = ((y - 3) << sy) + 3;
                            for i in 0..=sy {
                                for j in 0..=sx {
                                    l += luma[ly + i][lx + j];
                                }
                            }
                            let l = round2(
                                l,
                                u32::try_from(sx + sy).expect("subsampling shifts are 0 or 1"),
                            );
                            s0 += l * c0;
                            s1 += l * c1;
                        }
                        break 'o;
                    }
                    s0 += c0 * cb[at(y, dr)][at(x, dc)];
                    s1 += c1 * cr[at(y, dr)][at(x, dc)];
                    pos += 1;
                }
            }
            let (yy, xx) = (y, x);
            if g.num_cb_points > 0 || g.chroma_scaling_from_luma {
                cb[yy][xx] = (cb[yy][xx] + round2(s0, ar_shift)).clamp(gmin, gmax);
            }
            if g.num_cr_points > 0 || g.chroma_scaling_from_luma {
                cr[yy][xx] = (cr[yy][xx] + round2(s1, ar_shift)).clamp(gmin, gmax);
            }
        }
    }
    // Cropped templates: luma [9..73][9..73], chroma [6..38][6..38] (4:2:0).
    let mut o = 768;
    for row in luma.iter().skip(9).take(64) {
        for v in &row[9..73] {
            out[o..o + 2].copy_from_slice(&grain_i16(*v).to_le_bytes());
            o += 2;
        }
    }
    for p in [&cb, &cr] {
        for row in p.iter().skip(6).take(32) {
            for v in &row[6..38] {
                out[o..o + 2].copy_from_slice(&grain_i16(*v).to_le_bytes());
                o += 2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two film-grain points with the same value (a non-conformant stream, 6.8.20): the scaling
    /// LUT divided by the zero segment width and panicked. The segment is empty and skipped.
    #[test]
    fn scaling_lut_with_repeated_point_value() {
        let mut lut = [0u8; 256];
        scaling_lut(2, &[10, 10], &[20, 30], &mut lut);
        assert!(lut[..10].iter().all(|&v| v == 20));
        assert!(lut[10..].iter().all(|&v| v == 30));
    }
}
