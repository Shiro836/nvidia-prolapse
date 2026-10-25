/* The ETC2/EAC decode compute shader as SPIR-V words. out/gen/etc2_spv.h is produced at build time by
 *   glslangValidator -V -S comp -x -o out/gen/etc2_spv.h src/etc2_decoder.comp
 * (Makefile), validated with spirv-val; the glslang version that compiled it is recorded in
 * out/gen/etc2_spv.version. Nothing here is checked in as a binary. */
#include "nvhal_astc.h"

const uint32_t nvhal_etc2_spv[] = {
#include "etc2_spv.h"
};
const size_t nvhal_etc2_spv_size = sizeof nvhal_etc2_spv;
