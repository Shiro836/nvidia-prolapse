/* SPDX-License-Identifier: MIT */
#include "nv_nak.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
/* Independent nv_nak.h layout checks on LP64 and LP32. */
#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(struct nv_nak_result) == 48, "Rust LP64 result ABI size");
_Static_assert(_Alignof(struct nv_nak_result) == 8, "LP64 result alignment");
_Static_assert(offsetof(struct nv_nak_result, code_size) == 8, "code size offset");
_Static_assert(offsetof(struct nv_nak_result, registers) == 16, "register offset");
_Static_assert(offsetof(struct nv_nak_result, workgroup) == 32, "workgroup offset");
_Static_assert(offsetof(struct nv_nak_result, cbuf_bytes) == 44, "cbuf offset");
#elif UINTPTR_MAX == UINT32_MAX
_Static_assert(sizeof(struct nv_nak_result) == 40, "Rust LP32 result ABI size");
_Static_assert(_Alignof(struct nv_nak_result) == 4, "LP32 result alignment");
_Static_assert(offsetof(struct nv_nak_result, code_size) == 4, "code size offset");
_Static_assert(offsetof(struct nv_nak_result, registers) == 8, "register offset");
_Static_assert(offsetof(struct nv_nak_result, workgroup) == 24, "workgroup offset");
_Static_assert(offsetof(struct nv_nak_result, cbuf_bytes) == 36, "cbuf offset");
#else
#error "nv_nak_result ABI needs independent layout verification for this pointer width"
#endif
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s shader.spv [...]\n", argv[0]); return 2; }
    for (int arg = 1; arg < argc; arg++) {
        FILE *f = fopen(argv[arg], "rb");
        if (!f || fseek(f, 0, SEEK_END)) return 2;
        long size = ftell(f);
        if (size < 20 || size % 4 || fseek(f, 0, SEEK_SET)) return 2;
        uint32_t *words = malloc((size_t)size);
        if (!words || fread(words, 1, (size_t)size, f) != (size_t)size) return 2;
        fclose(f);
        struct nv_nak_result out;
        char error[4096];
        int ret = nv_nak_compile(words, (size_t)size / 4, "main", 120, &out, error, sizeof(error));
        free(words);
        if (ret) { fprintf(stderr, "%s: %s\n", argv[arg], error); return 1; }
        printf("%s: code=%zu registers=%u local=%u shared=%u barriers=%u workgroup=%u,%u,%u cbuf=%u\n",
               argv[arg], out.code_size, out.registers, out.local_bytes, out.shared_bytes,
               out.barriers, out.workgroup[0], out.workgroup[1], out.workgroup[2], out.cbuf_bytes);
        nv_nak_free(&out);
        if (out.code || out.code_size) return 1;
    }
    uint32_t invalid[] = {0x07230203, 0x10000, 0, 1, 0, 0};
    struct nv_nak_result out;
    char error[128];
    if (!nv_nak_compile(invalid, 6, "main", 120, &out, error, sizeof(error)) ||
        !error[0] || out.code) return 1;
    printf("%zu-byte result ABI and malformed-input rejection: PASS\n", sizeof(struct nv_nak_result));
    return 0;
}
