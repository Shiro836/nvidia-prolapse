/* The ASTC decode compute shader as SPIR-V words. out/gen/astc_spv.h is produced at build time by
 *   glslangValidator -V -S comp -x -o out/gen/astc_spv.h out/gen/mesa/src/compiler/glsl/astc_decoder.glsl
 * from Mesa's own source at the pinned commit (Makefile MESA_SHA; the command is the one in Mesa's
 * src/vulkan/runtime/meson.build, target 'astc_spv.h'); the glslang version that compiled it is
 * recorded in out/gen/astc_spv.version. Nothing here is checked in as a binary. */
#include "nvhal_astc.h"

const uint32_t nvhal_astc_spv[] = {
#include "astc_spv.h"
};
const size_t nvhal_astc_spv_size = sizeof nvhal_astc_spv;
