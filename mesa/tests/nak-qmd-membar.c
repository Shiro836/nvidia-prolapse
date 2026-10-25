/* QMD CWD_MEMBAR_TYPE of the built NVK: calls its exported nak_fill_qmd for a minimal compute shader on
 * every compute class from Volta to Blackwell B and checks CWD_MEMBAR_TYPE: L1_MEMBAR (3) from Turing on,
 * as NVIDIA's driver builds every QMD (libnvidia-eglcore.so 615.78.08 QMD template ELF 0xaa12a0, tails
 * ELF 0xaa1338 / 0xaa14c7 / 0xaa13d3), L1_NONE (0) on Volta. Field: MW(369:368) in V02_02 / V03_00
 * (clc3c0qmd.h, clc6c0qmd.h), MW(1137:1136) in V04_00 (clcbc0qmd.h), MW(625:624) in V05_00 (clcdc0qmd.h).
 * Usage: nak-qmd-membar LIBVULKAN_NOUVEAU.so. Build/run: tests/README.md "nak-qmd-membar".
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "nv_device_info.h"
#include "nak.h"
int main(int argc, char **argv) {
   void *h = dlopen(argv[1], RTLD_NOW);
   if (!h) { printf("%s\n", dlerror()); return 2; }
   void (*fill)(const struct nv_device_info *, const struct nak_shader_info *, const struct nak_qmd_info *, void *, size_t) = dlsym(h, "nak_fill_qmd");
   uint32_t (*size)(const struct nv_device_info *) = dlsym(h, "nak_qmd_size_B");
   static const struct { uint16_t cc, c3d; uint8_t sm; unsigned dw, want; } cls[] = {
      { 0xc3c0, 0xc397, 70, 11, 0 }, { 0xc5c0, 0xc597, 75, 11, 3 }, { 0xc6c0, 0xc697, 80, 11, 3 },
      { 0xc7c0, 0xc797, 86, 11, 3 }, { 0xc9c0, 0xc997, 89, 11, 3 }, { 0xcbc0, 0xcb97, 90, 35, 3 },
      { 0xcdc0, 0xcd97, 100, 19, 3 }, { 0xcec0, 0xce97, 120, 19, 3 },
   };
   int bad = 0;
   for (unsigned k = 0; k < sizeof(cls) / sizeof(cls[0]); k++) {
      struct nv_device_info dev; memset(&dev, 0, sizeof(dev));
      dev.cls_compute = cls[k].cc; dev.cls_eng3d = cls[k].c3d;
      dev.sm = cls[k].sm; dev.max_smem_per_wg_kB = 48; dev.sm_smem_size_count = 2; dev.sm_smem_sizes_kB[0] = 32; dev.sm_smem_sizes_kB[1] = 100;
      struct nak_shader_info info; memset(&info, 0, sizeof(info));
      info.stage = MESA_SHADER_COMPUTE; info.max_warps_per_sm = 48; info.num_gprs = 16;
      info.cs.local_size[0] = 64; info.cs.local_size[1] = 1; info.cs.local_size[2] = 1;
      struct nak_qmd_info qi; memset(&qi, 0, sizeof(qi));
      qi.addr = 0x100000; qi.global_size[0] = 4; qi.global_size[1] = 1; qi.global_size[2] = 1;
      uint32_t q[96] = {0};
      uint32_t sz = size(&dev);
      fill(&dev, &info, &qi, q, sz);
      unsigned m = (q[cls[k].dw] >> 16) & 3;
      printf("cls_compute 0x%x qmd %u B: CWD_MEMBAR_TYPE = %u (dword%u = 0x%08x)\n", dev.cls_compute, sz, m, cls[k].dw, q[cls[k].dw]);
      if (m != cls[k].want) bad = 1;
   }
   printf("%s\n", bad ? "FAIL" : "PASS");
   return bad;
}
