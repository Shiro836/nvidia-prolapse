/* Test-only stand-in for Mesa's src/util/format_srgb.h: only the single-texel fetch functions, which the test
 * never calls, use it. */
#ifndef ETC2_REF_FORMAT_SRGB_H
#define ETC2_REF_FORMAT_SRGB_H
#include <stdlib.h>
static inline float util_format_srgb_8unorm_to_linear_float(unsigned char v) {
    (void)v;
    abort();
}
#endif
