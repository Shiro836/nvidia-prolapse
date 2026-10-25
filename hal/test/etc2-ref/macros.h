/* Test-only stand-in for Mesa's src/mesa/main/macros.h: the helpers texcompress_etc.c uses. */
#ifndef ETC2_REF_MACROS_H
#define ETC2_REF_MACROS_H
#include <assert.h>
#include <stdlib.h>
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))
/* only the single-texel fetch functions use these, which the test never calls */
#define UBYTE_TO_FLOAT(u) ((float)(u) / 255.0f)
#define USHORT_TO_FLOAT(u) ((float)(u) / 65535.0f)
#define SHORT_TO_FLOAT(s) ((float)(s) / 32767.0f)
#define RCOMP 0
#define GCOMP 1
#define BCOMP 2
#define ACOMP 3
#define unreachable(msg) abort()
#endif
