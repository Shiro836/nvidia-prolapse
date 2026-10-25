/* The reference ASTC decoder for test/astc-roundtrip.c: Mesa's CPU decoder for GL_KHR_texture_compression_astc_ldr
 * (src/mesa/main/texcompress_astc.cpp, "copied from the OASTC library written by Philip Taylor"), compiled from the
 * pinned Mesa commit (out/gen/mesa, Makefile MESA_SHA) with the stand-in headers of test/astc-ref/. It is an
 * implementation independent of the GPU decoder under test (Granite's astc_decoder.glsl), run in the same
 * unorm8 output mode. */
#include <cstdint>
#include <cstdlib>

#include "texcompress_astc.cpp"

extern "C" uint16_t _mesa_uint16_div_64k_to_half(uint16_t v) {
    (void)v;
    abort(); /* only reachable outside the unorm8 output mode */
}

/* decodes one 16-byte block of bw x bh texels into rgba (bw*bh*4 bytes, row-major); returns Mesa's
 * decode_error::type (0 = ok; otherwise rgba is the error colour) */
extern "C" int astc_ref_decode(const uint8_t *block, unsigned bw, unsigned bh, int srgb, uint8_t *rgba) {
    Decoder dec((int)bw, (int)bh, 1, srgb != 0, true);
    uint16_t out[12 * 12 * 4];
    int err = (int)dec.decode(block, out);
    for (unsigned i = 0; i < bw * bh * 4; i++) rgba[i] = (uint8_t)out[i];
    return err;
}
