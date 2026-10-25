/* Test-only stand-in for Mesa's src/mesa/main/macros.h: the helpers texcompress_astc.cpp uses. */
#ifndef ASTC_REF_MACROS_H
#define ASTC_REF_MACROS_H
#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define unreachable(msg) abort()
#endif
