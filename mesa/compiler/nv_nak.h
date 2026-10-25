/* Native driver compiler bridge. SPDX-License-Identifier: MIT */
#ifndef NV_NATIVE_NAK_H
#define NV_NATIVE_NAK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Cbuf0: 8 records { uint64_t address; uint32_t size, reserved; } at
 * binding*16; push constants at byte 1024, maximum 128 bytes.
 * Only descriptor set 0, one SSBO per binding; no descriptor arrays yet. */
struct nv_nak_result {
    uint8_t *code;
    size_t code_size;
    uint32_t registers;
    uint32_t local_bytes;
    uint32_t shared_bytes;
    uint32_t barriers;
    uint32_t workgroup[3];
    uint32_t cbuf_bytes;
};
/* 0 success, -1 invalid/unsupported or compile failure. out is zeroed on error.
 * error is NUL-terminated when error_size>0. Allocation belongs to the bridge.
 * All inputs are borrowed for this call. sm must currently equal 120.
 * Callers must serialize compiler calls (Mesa global GLSL type cache). */
int nv_nak_compile(const uint32_t *spirv, size_t word_count,
                   const char *entrypoint, uint32_t sm,
                   struct nv_nak_result *out, char *error, size_t error_size);
void nv_nak_free(struct nv_nak_result *result);
#ifdef __cplusplus
}
#endif
#endif
