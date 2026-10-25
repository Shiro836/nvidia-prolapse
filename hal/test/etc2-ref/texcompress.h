/* Test-only stand-in for the Mesa headers src/mesa/main/texcompress_etc.c includes (texcompress.h,
 * texcompress_etc.h, texstore.h, util/glheader.h): the GL types, the mesa_format values and the texstore/fetch
 * signatures it names. Only _mesa_unpack_etc2_format() is called (test/etc2-ref.c). */
#ifndef ETC2_REF_TEXCOMPRESS_H
#define ETC2_REF_TEXCOMPRESS_H
#include <stdbool.h>
#include <stdint.h>
typedef unsigned char GLboolean;
typedef signed char GLbyte;
typedef unsigned char GLubyte;
typedef short GLshort;
typedef unsigned short GLushort;
typedef int GLint;
typedef unsigned int GLuint;
typedef unsigned int GLenum;
typedef float GLfloat;
typedef void GLvoid;
#define GL_FALSE 0
#define GL_TRUE 1
typedef enum {
    MESA_FORMAT_NONE = 0,
    MESA_FORMAT_ETC1_RGB8,
    MESA_FORMAT_ETC2_RGB8,
    MESA_FORMAT_ETC2_SRGB8,
    MESA_FORMAT_ETC2_RGBA8_EAC,
    MESA_FORMAT_ETC2_SRGB8_ALPHA8_EAC,
    MESA_FORMAT_ETC2_R11_EAC,
    MESA_FORMAT_ETC2_RG11_EAC,
    MESA_FORMAT_ETC2_SIGNED_R11_EAC,
    MESA_FORMAT_ETC2_SIGNED_RG11_EAC,
    MESA_FORMAT_ETC2_RGB8_PUNCHTHROUGH_ALPHA1,
    MESA_FORMAT_ETC2_SRGB8_PUNCHTHROUGH_ALPHA1,
} mesa_format;
struct gl_context;
typedef void (*compressed_fetch_func)(const GLubyte *map, GLint rowStride, GLint i, GLint j, GLfloat *texel);
#define UNUSED __attribute__((unused))
#define UNUSED_TEXSTORE_PARAMS                                                                                    \
    UNUSED struct gl_context *ctx, UNUSED GLuint dims, UNUSED GLenum baseInternalFormat,                          \
        UNUSED mesa_format dstFormat, UNUSED GLint dstRowStride, UNUSED GLubyte **dstSlices, UNUSED GLint srcWidth, \
        UNUSED GLint srcHeight, UNUSED GLint srcDepth, UNUSED GLenum srcFormat, UNUSED GLenum srcType,           \
        UNUSED const GLvoid *srcAddr, UNUSED const void *srcPacking
#endif
