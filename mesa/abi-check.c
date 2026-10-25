/* SPDX-License-Identifier: MIT */
#include "nv_rm.h"
#include <stdio.h>
_Static_assert(sizeof(struct NvRmInfo) == 224, "Rust Info size");
_Static_assert(offsetof(struct NvRmInfo, max_shared_ctx) == 216, "Rust Info shared context offset");
_Static_assert(offsetof(struct NvRmInfo, zcull) == 152, "Rust Info ZCULL offset");
_Static_assert(offsetof(struct NvRmInfo, zcull_ctxsw_size) == 192, "Rust Info ZCULL ctxsw offset");
_Static_assert(offsetof(struct NvRmInfo, has_zcull_info) == 200, "Rust Info ZCULL flag offset");
_Static_assert(offsetof(struct NvRmInfo, sm_isa) == 204, "Rust Info SM ISA offset");
_Static_assert(offsetof(struct NvRmInfo, shader_pipe_count) == 208, "Rust Info shader pipe offset");
_Static_assert(offsetof(struct NvRmInfo, shader_pipe_sub_count) == 212, "Rust Info shader pipe sub offset");
_Static_assert(offsetof(struct NvRmInfo, generic_kind) == 68, "Rust Info kind offset");
_Static_assert(offsetof(struct NvRmInfo, vram_size) == 72, "Rust Info VRAM offset");
_Static_assert(offsetof(struct NvRmInfo, bar_size) == 80, "Rust Info BAR offset");
_Static_assert(offsetof(struct NvRmInfo, name) == 88, "Rust Info name offset");
_Static_assert(offsetof(struct NvRmAllocation, cpu_address) == 16, "Rust allocation address offset");
#define SIZE(T) printf(#T ".size=%zu align=%zu\n", sizeof(struct T), _Alignof(struct T))
#define OFFSET(T, F) printf(#T "." #F "=%zu\n", offsetof(struct T, F))
int main(void) {
   SIZE(NvRmInfo);
   OFFSET(NvRmInfo, generic_kind);
   OFFSET(NvRmInfo, vram_size);
   OFFSET(NvRmInfo, bar_size);
   OFFSET(NvRmInfo, name);
   OFFSET(NvRmInfo, zcull);
   OFFSET(NvRmInfo, zcull_ctxsw_size);
   OFFSET(NvRmInfo, has_zcull_info);
   OFFSET(NvRmInfo, sm_isa);
   OFFSET(NvRmInfo, shader_pipe_count);
   OFFSET(NvRmInfo, shader_pipe_sub_count);
   OFFSET(NvRmInfo, max_shared_ctx);
   SIZE(NvRmAllocation);
   OFFSET(NvRmAllocation, id);
   OFFSET(NvRmAllocation, size);
   OFFSET(NvRmAllocation, cpu_address);
   SIZE(NvRmExec);
   OFFSET(NvRmExec, size_B);
   OFFSET(NvRmExec, flags);
   return 0;
}
