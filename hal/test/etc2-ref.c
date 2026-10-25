/* The reference ETC2/EAC decoder for test/etc2-roundtrip.c: Mesa's CPU decoder (src/mesa/main/texcompress_etc.c,
 * _mesa_unpack_etc2_format) compiled from the pinned Mesa commit (out/gen/mesa, Makefile MESA_SHA) with the
 * stand-in headers of test/etc2-ref/. It is an implementation independent of the GPU decoder under test
 * (src/etc2_decoder.comp). */
#include <stdint.h>
#include <stdlib.h>

#include "texcompress_etc.c"

/* decodes one block (8 or 16 bytes) of the ETC2/EAC VkFormat vkfmt (VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK = 147 ...
 * VK_FORMAT_EAC_R11G11_SNORM_BLOCK = 156) into 4x4 texels, row-major: 4 bytes per texel (RGBA8) for the ETC2
 * formats, 2 bytes (R16) or 4 bytes (R16G16) for EAC, as Mesa's unpack writes them; returns -1 for another format */
int etc2_ref_decode(const uint8_t *block, int vkfmt, uint8_t *out) {
    static const mesa_format map[10] = {
        MESA_FORMAT_ETC2_RGB8, MESA_FORMAT_ETC2_SRGB8, MESA_FORMAT_ETC2_RGB8_PUNCHTHROUGH_ALPHA1,
        MESA_FORMAT_ETC2_SRGB8_PUNCHTHROUGH_ALPHA1, MESA_FORMAT_ETC2_RGBA8_EAC, MESA_FORMAT_ETC2_SRGB8_ALPHA8_EAC,
        MESA_FORMAT_ETC2_R11_EAC, MESA_FORMAT_ETC2_SIGNED_R11_EAC, MESA_FORMAT_ETC2_RG11_EAC,
        MESA_FORMAT_ETC2_SIGNED_RG11_EAC};
    if (vkfmt < 147 || vkfmt > 156) return -1;
    const unsigned i = (unsigned)(vkfmt - 147);
    const unsigned texel = i < 6 ? 4 : i < 8 ? 2 : 4;
    const unsigned bytes = i == 0 || i == 1 || i == 2 || i == 3 || i == 6 || i == 7 ? 8 : 16;
    _mesa_unpack_etc2_format(out, 4 * texel, block, bytes, 4, 4, map[i], false);
    return 0;
}
