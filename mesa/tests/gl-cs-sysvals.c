/* SPDX-License-Identifier: MIT
 *
 * Compute system values on nvrm: gl_LocalInvocationIndex, gl_LocalInvocationID,
 * gl_GlobalInvocationID, gl_WorkGroupID, gl_NumWorkGroups and gl_WorkGroupSize,
 * each invocation writing them to an SSBO slot of its own (indexed by the
 * global invocation id, not by gl_LocalInvocationIndex), checked against the
 * CPU for several workgroup shapes: 16x16x1 (Minecraft's GLSL176 shape),
 * 4x2x3, 64x1x1, 1x1x1 and 7x5x3, with 2x3x2 workgroups each; plus a shared
 * memory reduction that indexes shared memory by gl_LocalInvocationIndex.
 *
 * Mesa 26's NAK stopped lowering load_local_invocation_index itself (25.2 set
 * lower_cs_local_index_to_id in its nir_options): without the driver-side
 * nir_lower_compute_system_values(lower_local_invocation_index) NAK panics
 * "Unsupported intrinsic instruction: load_local_invocation_index" at link time.
 *
 *   cc -std=c11 -I<mesa>/include gl-cs-sysvals.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -o gl-cs-sysvals
 *   LIBGL_DRIVERS_PATH=<build>/src/gallium/targets/dri ./gl-cs-sysvals
 * PASS = every value of every invocation matches; also passes on llvmpipe
 * (__EGL_VENDOR_LIBRARY_FILENAMES=.../50_mesa.json LIBGL_ALWAYS_SOFTWARE=1).
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static GLuint
program_cs(const char *src)
{
   GLuint s = glCreateShader(GL_COMPUTE_SHADER);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("FAIL: compile:\n%s\n%s\n", log, src);
      exit(1);
   }
   GLuint p = glCreateProgram();
   glAttachShader(p, s);
   glLinkProgram(p);
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      printf("FAIL: link:\n%s\n", log);
      exit(1);
   }
   glDeleteShader(s);
   return p;
}

#define WORDS 16 /* per invocation */

static void
sysvals_case(unsigned lx, unsigned ly, unsigned lz, unsigned gx, unsigned gy, unsigned gz)
{
   char src[2048];
   snprintf(src, sizeof(src),
            "#version 310 es\n"
            "layout(local_size_x = %u, local_size_y = %u, local_size_z = %u) in;\n"
            "layout(std430, binding = 0) buffer Out { uint v[]; };\n"
            "void main() {\n"
            "   uvec3 g = gl_GlobalInvocationID;\n"
            "   uvec3 dim = gl_NumWorkGroups * gl_WorkGroupSize;\n"
            "   uint slot = ((g.z * dim.y + g.y) * dim.x + g.x) * %uu;\n"
            "   v[slot + 0u] = gl_LocalInvocationIndex;\n"
            "   v[slot + 1u] = gl_LocalInvocationID.x;\n"
            "   v[slot + 2u] = gl_LocalInvocationID.y;\n"
            "   v[slot + 3u] = gl_LocalInvocationID.z;\n"
            "   v[slot + 4u] = g.x; v[slot + 5u] = g.y; v[slot + 6u] = g.z;\n"
            "   v[slot + 7u] = gl_WorkGroupID.x; v[slot + 8u] = gl_WorkGroupID.y;\n"
            "   v[slot + 9u] = gl_WorkGroupID.z;\n"
            "   v[slot + 10u] = gl_NumWorkGroups.x; v[slot + 11u] = gl_NumWorkGroups.y;\n"
            "   v[slot + 12u] = gl_NumWorkGroups.z;\n"
            "   v[slot + 13u] = gl_WorkGroupSize.x * 10000u + gl_WorkGroupSize.y * 100u +\n"
            "                   gl_WorkGroupSize.z;\n"
            "   v[slot + 14u] = 0xc0ffeeu;\n"
            "   v[slot + 15u] = gl_LocalInvocationIndex * 3u + 1u;\n"
            "}\n", lx, ly, lz, WORDS);
   GLuint p = program_cs(src);
   const unsigned dx = lx * gx, dy = ly * gy, dz = lz * gz;
   const unsigned n = dx * dy * dz;
   const size_t size = (size_t)n * WORDS * sizeof(uint32_t);
   GLuint buf;
   glGenBuffers(1, &buf);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
   uint32_t *init = malloc(size);
   memset(init, 0xab, size);
   glBufferData(GL_SHADER_STORAGE_BUFFER, size, init, GL_DYNAMIC_READ);
   free(init);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
   glUseProgram(p);
   glDispatchCompute(gx, gy, gz);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   const uint32_t *v = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_READ_BIT);
   unsigned bad = 0;
   for (unsigned z = 0; z < dz; z++)
      for (unsigned y = 0; y < dy; y++)
         for (unsigned x = 0; x < dx; x++) {
            const uint32_t *s = v + ((size_t)(z * dy + y) * dx + x) * WORDS;
            const unsigned l[3] = { x % lx, y % ly, z % lz };
            const uint32_t idx = (l[2] * ly + l[1]) * lx + l[0];
            const uint32_t want[WORDS] = {
               idx, l[0], l[1], l[2], x, y, z, x / lx, y / ly, z / lz, gx, gy, gz,
               lx * 10000u + ly * 100u + lz, 0xc0ffeeu, idx * 3u + 1u,
            };
            for (unsigned w = 0; w < WORDS; w++) {
               if (s[w] != want[w]) {
                  if (bad++ < 8)
                     printf("  invocation (%u,%u,%u) word %u: got %u want %u\n", x, y, z, w,
                            s[w], want[w]);
               }
            }
         }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   printf("%s: local %ux%ux%u groups %ux%ux%u: %u invocations, %u wrong values\n",
          bad ? "FAIL" : "ok", lx, ly, lz, gx, gy, gz, n, bad);
   failures += bad != 0;
   glDeleteBuffers(1, &buf);
   glDeleteProgram(p);
}

/* Shared-memory reduction keyed by gl_LocalInvocationIndex (the pattern of
 * compute shaders that build per-workgroup histograms / prefix sums). */
static void
shared_case(void)
{
   static const char *src =
      "#version 310 es\n"
      "layout(local_size_x = 16, local_size_y = 16) in;\n"
      "layout(std430, binding = 0) buffer Out { uint v[]; };\n"
      "shared uint s[256];\n"
      "void main() {\n"
      "   s[gl_LocalInvocationIndex] = gl_LocalInvocationIndex + 1u;\n"
      "   barrier();\n"
      "   for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
      "      if (gl_LocalInvocationIndex < stride)\n"
      "         s[gl_LocalInvocationIndex] += s[gl_LocalInvocationIndex + stride];\n"
      "      barrier();\n"
      "   }\n"
      "   if (gl_LocalInvocationIndex == 0u)\n"
      "      v[gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x] = s[0];\n"
      "}\n";
   GLuint p = program_cs(src);
   GLuint buf;
   glGenBuffers(1, &buf);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
   glBufferData(GL_SHADER_STORAGE_BUFFER, 6 * 4, NULL, GL_DYNAMIC_READ);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
   glUseProgram(p);
   glDispatchCompute(3, 2, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   const uint32_t *v = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 6 * 4, GL_MAP_READ_BIT);
   unsigned bad = 0;
   for (unsigned i = 0; i < 6; i++)
      bad += v[i] != 256u * 257u / 2u;
   printf("%s: shared reduction by gl_LocalInvocationIndex (16x16, 3x2 groups): %u %u %u %u %u %u (want %u)\n",
          bad ? "FAIL" : "ok", v[0], v[1], v[2], v[3], v[4], v[5], 256u * 257u / 2u);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   failures += bad != 0;
   glDeleteBuffers(1, &buf);
   glDeleteProgram(p);
}

int
main(void)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(dpy, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API)) {
      printf("FAIL: EGL init\n");
      return 1;
   }
   static const EGLint es31[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, es31);
   if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: ES 3.1 context\n");
      return 1;
   }
   printf("%s / %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   sysvals_case(16, 16, 1, 2, 3, 2);
   sysvals_case(4, 2, 3, 2, 3, 2);
   sysvals_case(64, 1, 1, 2, 3, 2);
   sysvals_case(1, 1, 1, 2, 3, 2);
   sysvals_case(7, 5, 3, 2, 3, 2);
   shared_case();
   GLenum err = glGetError();
   if (err != GL_NO_ERROR) {
      printf("FAIL: GL error 0x%x\n", err);
      failures++;
   }
   printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   return failures != 0;
}
