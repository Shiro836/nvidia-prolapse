/* PASCAL_A clear bracket: method words and class selection.
 *
 * Build/run (repository root, after build.py configured the pinned source):
 *   M=build/mesa-rm-source; cc -std=c11 -Wall -Wextra -Werror \
 *     -I$M/src/nouveau/vulkan -I$M/src/nouveau/headers -I$M/src \
 *     -I$M/include -I$M/src/nouveau/headers/nvidia/classes \
 *     nvidia-prolapse/driver-source/native/mesa/tests/pascal-a-clear.c -o pascal-a-clear && ./pascal-a-clear
 *
 * Expected words are the stock 580.178.04 eglcore pushes (objdump of
 * libnvidia-eglcore at ELF 0xb5b8f7 and 0xb5b881): only PASCAL_A gets them.
 */
#include <stdio.h>
#include <stdlib.h>
#include "nvk_pascal_a.h"

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static void
one(uint16_t cls, int expect_bracket)
{
   uint32_t buf[8] = {0};
   struct nv_push p;
   nv_push_init(&p, buf, 8, SUBC_MASK_ALL);
   nvk_push_pascal_a_clear_begin(&p, cls);
   size_t begin = p.end - p.start;
   nvk_push_pascal_a_clear_end(&p, cls);
   size_t total = p.end - p.start;
   CHECK(begin == nvk_pascal_a_clear_begin_dw(cls));
   CHECK(total - begin == nvk_pascal_a_clear_end_dw(cls));
   if (expect_bracket) {
      CHECK(total == 3);
      CHECK(buf[0] == 0x800103d8); /* 0xF60 := 1 */
      CHECK(buf[1] == 0x800103e0); /* 0xF80 := 1 */
      CHECK(buf[2] == 0x800003d8); /* 0xF60 := 0 */
   } else {
      CHECK(total == 0);
   }
   printf("class %#06x bracket=%d words=%zu\n", cls, expect_bracket, total);
}

int
main(void)
{
   one(0xb097, 0); /* MAXWELL_A */
   one(0xb197, 0); /* MAXWELL_B */
   one(0xc097, 1); /* PASCAL_A: GP100, GP10B */
   one(0xc197, 0); /* PASCAL_B: GP102..GP108 */
   one(0xc397, 0); /* VOLTA_A */
   one(0xc597, 0); /* TURING_A */
   one(0xce97, 0); /* BLACKWELL_B */
   if (failures) {
      fprintf(stderr, "%d failures\n", failures);
      return 1;
   }
   printf("PASS pascal-a-clear\n");
   return 0;
}
