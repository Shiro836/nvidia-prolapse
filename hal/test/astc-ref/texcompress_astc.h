/* Test-only stand-in for Mesa's src/mesa/main/texcompress_astc.h (+ texcompress.h / formats.h): just enough
 * for Mesa's CPU ASTC decoder, src/mesa/main/texcompress_astc.cpp at the pinned commit, to compile as the
 * reference of test/astc-roundtrip.c. Only its Decoder class is used (astc-ref.cpp); the mesa_format helpers
 * below serve _mesa_unpack_astc_2d_ldr(), which the test never calls. */
#ifndef ASTC_REF_TEXCOMPRESS_ASTC_H
#define ASTC_REF_TEXCOMPRESS_ASTC_H
#include <inttypes.h>
#include <stdbool.h>
typedef int mesa_format;
static inline bool _mesa_is_format_astc_2d(mesa_format f) { (void)f; return true; }
static inline bool _mesa_is_format_srgb(mesa_format f) { (void)f; return false; }
static inline void _mesa_get_format_block_size(mesa_format f, unsigned *w, unsigned *h) { (void)f; *w = *h = 4; }
#ifdef __cplusplus
extern "C" {
#endif
void _mesa_unpack_astc_2d_ldr(uint8_t *dst_row, unsigned dst_stride, const uint8_t *src_row, unsigned src_stride,
                              unsigned src_width, unsigned src_height, mesa_format format);
#ifdef __cplusplus
}
#endif
#endif
