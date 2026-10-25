/* Test-only stand-in for Mesa's src/util/half_float.h: the reference decoder runs in its unorm8 output mode,
 * which never converts to FP16 (astc-ref.cpp aborts if it ever did). */
#ifndef ASTC_REF_HALF_FLOAT_H
#define ASTC_REF_HALF_FLOAT_H
#include <stdint.h>
#define FP16_ONE ((uint16_t)0x3c00)
#define FP16_ZERO ((uint16_t)0)
#ifdef __cplusplus
extern "C"
#endif
uint16_t _mesa_uint16_div_64k_to_half(uint16_t v);
#endif
