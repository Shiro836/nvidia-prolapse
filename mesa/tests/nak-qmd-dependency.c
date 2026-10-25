/* QMD dependency fields of the built NVK: calls its exported nak_qmd_set_dependence_counter,
 * nak_qmd_set_dependent_qmd (slots 0 and 1), nak_qmd_dependent_qmd_enabled and nak_qmd_set_invalidates
 * on a zeroed QMD for every class they serve and compares every word with the bits NVIDIA's driver writes
 * for that layout (libnvidia-eglcore.so 615.78.08: DEPENDENCE_COUNTER ELF 0xd32f60, DEPENDENT_QMD0
 * ELF 0xd33340, DEPENDENT_QMD1 ELF 0xd334a0, invalidates ELF 0xd341b0; ACTION 4 = DECREMENT_DEPENDENCE,
 * PREFETCH 1):
 *   V03_00 (c7c0, c9c0): counter word 4 bits 29:14; QMD0 word 16 bits 0 / 3:1 / 4, pointer word 15;
 *     QMD1 word 16 bits 5 / 8:6 / 9, pointer word 9; invalidates word 5 bits 28 / 29 / 31.
 *   V04_00 (cbc0): counter word 0 bits 15:0; QMD0 word 11 bits 16 / 19:17 / 20, pointer word 14;
 *     QMD1 word 11 bits 21 / 24:22 / 25, pointer word 15; invalidates word 18 bits 28 / 29 / 31.
 *   V05_00 (cdc0 in NAK, cec0): counter word 4 bits 15:0; QMD0 word 10 bits 16 / 19:17 / 20, pointer
 *     word 12; QMD1 word 10 bits 24 / 27:25 / 28, pointer word 13; invalidates word 14 bits 26 / 27 / 29.
 * Usage: nak-qmd-dependency LIBVULKAN_NOUVEAU.so. Build/run: tests/README.md "nak-qmd-dependency".
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "nv_device_info.h"
#include "nak.h"
struct want { unsigned w; uint32_t v; };
int main(int argc, char **argv) {
   if (argc < 2) return 2;
   void *h = dlopen(argv[1], RTLD_NOW);
   if (!h) { printf("%s\n", dlerror()); return 2; }
   uint32_t (*size)(const struct nv_device_info *) = dlsym(h, "nak_qmd_size_B");
   void (*counter)(const struct nv_device_info *, void *, size_t, uint32_t) = dlsym(h, "nak_qmd_set_dependence_counter");
   void (*dep)(const struct nv_device_info *, void *, size_t, uint32_t, uint64_t) = dlsym(h, "nak_qmd_set_dependent_qmd");
   bool (*enabled)(const struct nv_device_info *, void *, size_t, uint32_t) = dlsym(h, "nak_qmd_dependent_qmd_enabled");
   void (*inval)(const struct nv_device_info *, void *, size_t, bool, bool, bool) = dlsym(h, "nak_qmd_set_invalidates");
   if (!size || !counter || !dep || !enabled || !inval) { printf("missing export\n"); return 2; }
   const uint64_t a0 = 0x12345600, a1 = 0xabcdef00;
   const uint32_t cnt = 0x1234;
   const struct want v3[] = { { 4, cnt << 14 }, { 5, 0xb0000000 }, { 9, a1 >> 8 }, { 15, a0 >> 8 },
                              { 16, 0x1 | 4 << 1 | 1 << 4 | 1 << 5 | 4 << 6 | 1 << 9 } };
   const struct want v4[] = { { 0, cnt }, { 11, 1 << 16 | 4 << 17 | 1 << 20 | 1 << 21 | 4 << 22 | 1 << 25 },
                              { 14, a0 >> 8 }, { 15, a1 >> 8 }, { 18, 0xb0000000 } };
   const struct want v5[] = { { 4, cnt }, { 10, 1 << 16 | 4 << 17 | 1 << 20 | 1 << 24 | 4 << 25 | 1u << 28 },
                              { 12, a0 >> 8 }, { 13, a1 >> 8 }, { 14, 0x2c000000 } };
   const struct { uint16_t cc, c3d; const struct want *w; unsigned n; } cls[] = {
      { 0xc7c0, 0xc797, v3, 5 }, { 0xc9c0, 0xc997, v3, 5 }, { 0xcbc0, 0xcb97, v4, 5 },
      { 0xcdc0, 0xcd97, v5, 5 }, { 0xcec0, 0xce97, v5, 5 },
   };
   int bad = 0;
   for (unsigned k = 0; k < sizeof(cls) / sizeof(cls[0]); k++) {
      struct nv_device_info dev; memset(&dev, 0, sizeof(dev));
      dev.cls_compute = cls[k].cc; dev.cls_eng3d = cls[k].c3d;
      uint32_t sz = size(&dev), q[96] = {0}, want[96] = {0};
      int e0 = enabled(&dev, q, sz, 0), e1 = enabled(&dev, q, sz, 1);
      counter(&dev, q, sz, cnt);
      dep(&dev, q, sz, 0, a0);
      int e0b = enabled(&dev, q, sz, 0), e1b = enabled(&dev, q, sz, 1);
      dep(&dev, q, sz, 1, a1);
      int e1c = enabled(&dev, q, sz, 1);
      inval(&dev, q, sz, true, true, true);
      for (unsigned i = 0; i < cls[k].n; i++) want[cls[k].w[i].w] = cls[k].w[i].v;
      int ok = !e0 && !e1 && e0b && !e1b && e1c;
      for (unsigned i = 0; i < sz / 4; i++) {
         if (q[i] != want[i]) { ok = 0; printf("  0x%x word %u = 0x%08x, want 0x%08x\n", cls[k].cc, i, q[i], want[i]); }
      }
      printf("cls_compute 0x%x qmd %u B: %s\n", cls[k].cc, sz, ok ? "ok" : "MISMATCH");
      if (!ok) bad = 1;
   }
   printf("%s\n", bad ? "FAIL" : "PASS");
   return bad;
}
