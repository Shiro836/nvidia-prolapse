/* SPDX-License-Identifier: MIT
 *
 * nvrm GLES 3.1 part A gate: compute dispatch, SSBO read/write from compute and
 * fragment shaders, SSBO atomics, atomic counters (lowered to SSBOs), shared
 * memory with barrier(), a 32 KiB shared array, compute UBO blocks past the
 * default block, gl_NumWorkGroups, dispatch -> glMemoryBarrier -> dispatch and
 * dispatch -> glMemoryBarrier -> draw chains, and (argument "indirect")
 * glDispatchComputeIndirect. Every result is checked against values computed on
 * the CPU; the printed digest lets two drivers be compared (llvmpipe).
 *
 *   cc -std=c11 -I<mesa>/include gl-es31-compute.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -o gl-es31-compute
 *   nvrm (ES 3.0 until part B lands): MESA_GLES_VERSION_OVERRIDE=3.1 ./gl-es31-compute
 *   llvmpipe: __EGL_VENDOR_LIBRARY_FILENAMES=.../50_mesa.json LIBGL_ALWAYS_SOFTWARE=1
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static uint32_t digest = 2166136261u;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static void
hash(const void *data, size_t n)
{
   const uint8_t *b = data;
   for (size_t i = 0; i < n; i++)
      digest = (digest ^ b[i]) * 16777619u;
}

static GLuint
compile(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096] = "";
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("FAIL: compile:\n%s\n", log);
      exit(1);
   }
   return s;
}

static GLuint
program(GLuint a, GLuint b)
{
   GLuint p = glCreateProgram();
   glAttachShader(p, a);
   if (b)
      glAttachShader(p, b);
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096] = "";
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      printf("FAIL: link:\n%s\n", log);
      exit(1);
   }
   return p;
}

static GLuint
buffer(GLenum target, const void *data, size_t size)
{
   GLuint b;
   glGenBuffers(1, &b);
   glBindBuffer(target, b);
   glBufferData(target, size, data, GL_DYNAMIC_COPY);
   return b;
}

static void *
map_read(GLuint b, size_t size)
{
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
   return glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_READ_BIT);
}

#define N 1024
#define GROUP 64

static const char *cs_main =
   "#version 310 es\n"
   "layout(local_size_x = 64) in;\n"
   "layout(std430, binding = 0) readonly buffer In { uint a[]; };\n"
   "layout(std430, binding = 1) writeonly buffer Out { uint b[]; };\n"
   "layout(std430, binding = 2) buffer Ctr { uint count; uint sum; uint maxv; };\n"
   "layout(std140, binding = 1) uniform P { uint add; };\n"
   "layout(binding = 0, offset = 0) uniform atomic_uint ac;\n"
   "uniform uint mul;\n"
   "shared uint s[64];\n"
   "void main() {\n"
   "  uint lid = gl_LocalInvocationID.x;\n"
   "  uint idx = gl_GlobalInvocationID.x;\n"
   "  s[lid] = a[idx] * mul;\n"
   "  barrier();\n"
   "  b[idx] = s[63u - lid] + add + gl_NumWorkGroups.x * 100000u;\n"
   "  atomicAdd(count, 1u);\n"
   "  atomicAdd(sum, a[idx]);\n"
   "  atomicMax(maxv, a[idx]);\n"
   "  atomicCounterIncrement(ac);\n"
   "}\n";

/* 32 KiB shared array: per-group tree reduction. */
static const char *cs_big_shared =
   "#version 310 es\n"
   "layout(local_size_x = 256) in;\n"
   "layout(std430, binding = 0) buffer Out { uint r[]; };\n"
   "shared uint big[8192];\n"
   "void main() {\n"
   "  uint lid = gl_LocalInvocationID.x;\n"
   "  for (uint i = 0u; i < 32u; i++) big[lid * 32u + i] = lid * 32u + i + gl_WorkGroupID.x;\n"
   "  barrier();\n"
   "  for (uint stride = 4096u; stride > 0u; stride >>= 1u) {\n"
   "    for (uint i = lid; i < stride; i += 256u) big[i] += big[i + stride];\n"
   "    barrier();\n"
   "  }\n"
   "  if (lid == 0u) r[gl_WorkGroupID.x] = big[0];\n"
   "}\n";

/* Second dispatch of a chain: reads what the first wrote. */
static const char *cs_chain =
   "#version 310 es\n"
   "layout(local_size_x = 32, local_size_y = 2) in;\n"
   "layout(std430, binding = 0) readonly buffer In { uint b[]; };\n"
   "layout(std430, binding = 3) buffer Out { uint c[]; };\n"
   "void main() {\n"
   "  uint idx = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * 32u) + gl_GlobalInvocationID.x;\n"
   "  c[idx] = b[idx] ^ 0x5a5a5a5au;\n"
   "}\n";

static const char *vs =
   "#version 310 es\n"
   "void main() {\n"
   "  vec2 p = vec2(float((gl_VertexID & 1) * 4 - 1), float((gl_VertexID & 2) * 2 - 1));\n"
   "  gl_Position = vec4(p, 0.0, 1.0);\n"
   "}\n";

/* Five SSBO blocks in a fragment shader (Minecraft's maximum), plus an atomic
 * per fragment and a per-pixel store. */
static const char *fs =
   "#version 310 es\n"
   "precision highp float;\n"
   "precision highp int;\n"
   "layout(std430, binding = 0) readonly buffer B0 { uint v0[]; };\n"
   "layout(std430, binding = 1) readonly buffer B1 { vec4 v1; };\n"
   "layout(std430, binding = 2) readonly buffer B2 { float v2[4]; };\n"
   "layout(std430, binding = 3) buffer B3 { uint frags; };\n"
   "layout(std430, binding = 4) writeonly buffer B4 { uint px[]; };\n"
   "layout(location = 0) out vec4 color;\n"
   "void main() {\n"
   "  uint x = uint(gl_FragCoord.x), y = uint(gl_FragCoord.y);\n"
   "  uint i = y * 16u + x;\n"
   "  atomicAdd(frags, 1u);\n"
   "  px[i] = v0[i] + 7u;\n"
   "  color = vec4(float(v0[i] & 255u) / 255.0, v1.y, v2[2], v1.w);\n"
   "}\n";


/* 16 SSBO blocks in one compute and one fragment shader (Mesa's
 * MAX_SHADER_STORAGE_BUFFERS; stock GLES caps lower) + a uniform block at
 * binding 14 (no hardware cbuf slot in nvrm: descriptor-table path). */
#define SSBO16_DECL \
   "layout(std430, binding = 0) buffer S0 { uint s0; };\n" \
   "layout(std430, binding = 1) buffer S1 { uint s1; };\n" \
   "layout(std430, binding = 2) buffer S2 { uint s2; };\n" \
   "layout(std430, binding = 3) buffer S3 { uint s3; };\n" \
   "layout(std430, binding = 4) buffer S4 { uint s4; };\n" \
   "layout(std430, binding = 5) buffer S5 { uint s5; };\n" \
   "layout(std430, binding = 6) buffer S6 { uint s6; };\n" \
   "layout(std430, binding = 7) buffer S7 { uint s7; };\n" \
   "layout(std430, binding = 8) buffer S8 { uint s8; };\n" \
   "layout(std430, binding = 9) buffer S9 { uint s9; };\n" \
   "layout(std430, binding = 10) buffer S10 { uint s10; };\n" \
   "layout(std430, binding = 11) buffer S11 { uint s11; };\n" \
   "layout(std430, binding = 12) buffer S12 { uint s12; };\n" \
   "layout(std430, binding = 13) buffer S13 { uint s13; };\n" \
   "layout(std430, binding = 14) buffer S14 { uint s14; };\n" \
   "layout(std430, binding = 15) buffer S15 { uint r[]; };\n" \
   "layout(std140, binding = 14) uniform U14 { uint u14; };\n"
#define SSBO16_SUM "s0 + s1 * 2u + s2 * 3u + s3 * 4u + s4 * 5u + s5 * 6u + s6 * 7u + s7 * 8u + " \
   "s8 * 9u + s9 * 10u + s10 * 11u + s11 * 12u + s12 * 13u + s13 * 14u + s14 * 15u + u14"
static const char *cs_ssbo16 =
   "#version 310 es\n"
   "layout(local_size_x = 1) in;\n"
   SSBO16_DECL
   "void main() { r[0] = " SSBO16_SUM "; atomicAdd(r[257], 1u); }\n";
static const char *fs_ssbo16 =
   "#version 310 es\n"
   "precision highp float;\n"
   "precision highp int;\n"
   SSBO16_DECL
   "layout(location = 0) out vec4 color;\n"
   "void main() {\n"
   "  uint i = uint(gl_FragCoord.y) * 16u + uint(gl_FragCoord.x);\n"
   "  r[1u + i] = " SSBO16_SUM " + i;\n"
   "  atomicAdd(r[257], 1u);\n"
   "  color = vec4(1.0);\n"
   "}\n";

/* Texture sampling in a compute shader (two units, texelFetch + texture). */
static const char *cs_tex =
   "#version 310 es\n"
   "layout(local_size_x = 4, local_size_y = 4) in;\n"
   "uniform highp sampler2D t0;\n"
   "uniform highp usampler2D t1;\n"
   "layout(std430, binding = 0) buffer Out { uvec4 o[]; };\n"
   "void main() {\n"
   "  ivec2 c = ivec2(gl_LocalInvocationID.xy);\n"
   "  vec4 a = texture(t0, (vec2(c) + 0.5) / 4.0);\n"
   "  uvec4 b = texelFetch(t1, c, 0);\n"
   "  o[c.y * 4 + c.x] = uvec4(a * 255.0 + 0.5) + b * 1000u;\n"
   "}\n";

int
main(int argc, char **argv)
{
   const int indirect = argc > 1 && !strcmp(argv[1], "indirect");
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(dpy, NULL, NULL)) {
      printf("FAIL: eglInitialize\n");
      return 1;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   static const EGLint es31[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, es31);
   if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: no OpenGL ES 3.1 context\n");
      return 1;
   }
   printf("GL_RENDERER: %s\nGL_VERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   GLint lim[8];
   glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &lim[0]);
   glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &lim[1]);
   glGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &lim[2]);
   glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &lim[3]);
   glGetIntegerv(GL_MAX_COMPUTE_ATOMIC_COUNTERS, &lim[4]);
   glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &lim[5]);
   glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, &lim[6]);
   printf("limits: invocations %d shared %d fs-ssbo %d cs-ssbo %d cs-atomic-counters %d cs-ubo %d fs-ubo %d\n",
          lim[0], lim[1], lim[2], lim[3], lim[4], lim[5], lim[6]);
   CHECK(lim[0] >= 1024 && lim[1] >= 32768 && lim[2] >= 16 && lim[3] >= 16 && lim[4] >= 8 &&
         lim[5] >= 15 && lim[6] >= 15, "limits below Mesa's ceilings (16 SSBOs, 15 UBOs)");

   /* --- 1: dispatch with SSBOs, shared memory, atomics, counters, UBO --- */
   uint32_t in[N];
   for (unsigned i = 0; i < N; i++)
      in[i] = (i * 2654435761u) >> 12;
   const uint32_t ubo[4] = { 77, 0, 0, 0 };
   const uint32_t zero[4] = { 0 };
   GLuint b_in = buffer(GL_SHADER_STORAGE_BUFFER, in, sizeof(in));
   GLuint b_out = buffer(GL_SHADER_STORAGE_BUFFER, NULL, sizeof(in));
   GLuint b_ctr = buffer(GL_SHADER_STORAGE_BUFFER, zero, sizeof(zero));
   GLuint b_ubo = buffer(GL_UNIFORM_BUFFER, ubo, sizeof(ubo));
   GLuint b_ac = buffer(GL_ATOMIC_COUNTER_BUFFER, zero, sizeof(zero));
   GLuint p_main = program(compile(GL_COMPUTE_SHADER, cs_main), 0);
   glUseProgram(p_main);
   glUniform1ui(glGetUniformLocation(p_main, "mul"), 3);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_in);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_out);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, b_ctr);
   glBindBufferBase(GL_UNIFORM_BUFFER, 1, b_ubo);
   glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, b_ac);
   if (indirect) {
      const uint32_t cmd[3] = { N / GROUP, 1, 1 };
      GLuint b_cmd = buffer(GL_DISPATCH_INDIRECT_BUFFER, cmd, sizeof(cmd));
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, b_cmd);
      glDispatchComputeIndirect(0);
   } else {
      glDispatchCompute(N / GROUP, 1, 1);
   }
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   CHECK(glGetError() == GL_NO_ERROR, "GL error after dispatch 1");
   const uint32_t *out = map_read(b_out, sizeof(in));
   unsigned bad = 0;
   uint64_t sum = 0;
   uint32_t maxv = 0;
   for (unsigned i = 0; out && i < N; i++) {
      const unsigned g = i / GROUP, l = i % GROUP;
      const uint32_t want = in[g * GROUP + 63 - l] * 3 + 77 + (N / GROUP) * 100000u;
      bad += out[i] != want;
      sum += in[i];
      maxv = in[i] > maxv ? in[i] : maxv;
   }
   CHECK(out && !bad, "dispatch 1: %u of %u outputs wrong (shared memory/UBO/NumWorkGroups)", bad, N);
   if (out)
      hash(out, sizeof(in));
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   const uint32_t *ctr = map_read(b_ctr, 12);
   CHECK(ctr && ctr[0] == N && ctr[1] == (uint32_t)sum && ctr[2] == maxv,
         "SSBO atomics: count %u (want %u) sum %u (want %u) max %u (want %u)",
         ctr ? ctr[0] : 0, N, ctr ? ctr[1] : 0, (uint32_t)sum, ctr ? ctr[2] : 0, maxv);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, b_ac);
   const uint32_t *ac = glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, 4, GL_MAP_READ_BIT);
   CHECK(ac && ac[0] == N, "atomic counter %u (want %u)", ac ? ac[0] : 0, N);
   glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
   if (indirect) {
      printf("%s: indirect dispatch\n", failures ? "FAIL" : "PASS");
      return failures != 0;
   }

   /* --- 2: dispatch -> barrier -> dispatch chain (2D workgroups) --- */
   GLuint b_chain = buffer(GL_SHADER_STORAGE_BUFFER, NULL, sizeof(in));
   GLuint p_chain = program(compile(GL_COMPUTE_SHADER, cs_chain), 0);
   glUseProgram(p_main);
   glDispatchCompute(N / GROUP, 1, 1); /* rewrites b_out with the same values */
   glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
   glUseProgram(p_chain);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_out);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, b_chain);
   glDispatchCompute(N / 64, 1, 1); /* 16 x (32x2) */
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   const uint32_t *c = map_read(b_chain, sizeof(in));
   bad = 0;
   /* idx = gy * (16*32) + gx with gx in [0,512), gy in [0,2): covers 0..1023. */
   for (unsigned i = 0; c && out && i < N; i++) {
      const unsigned g = i / GROUP, l = i % GROUP;
      const uint32_t b = in[g * GROUP + 63 - l] * 3 + 77 + (N / GROUP) * 100000u;
      bad += c[i] != (b ^ 0x5a5a5a5au);
   }
   CHECK(c && !bad, "dispatch chain: %u of %u wrong", bad, N);
   if (c)
      hash(c, sizeof(in));
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   /* --- 3: 32 KiB shared memory --- */
   GLuint b_red = buffer(GL_SHADER_STORAGE_BUFFER, NULL, 4 * 4);
   GLuint p_big = program(compile(GL_COMPUTE_SHADER, cs_big_shared), 0);
   glUseProgram(p_big);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_red);
   glDispatchCompute(4, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   const uint32_t *red = map_read(b_red, 16);
   for (unsigned g = 0; red && g < 4; g++) {
      const uint32_t want = 8191u * 8192u / 2u + 8192u * g;
      CHECK(red[g] == want, "32 KiB shared reduction group %u: %u (want %u)", g, red[g], want);
   }
   if (red)
      hash(red, 16);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   /* --- 4: compute -> barrier -> draw with 5 fragment SSBOs + atomics --- */
   const float v1[4] = { 0.0f, 0.5f, 0.0f, 1.0f };
   const float v2[4] = { 0.0f, 0.0f, 0.25f, 0.0f };
   GLuint b_v1 = buffer(GL_SHADER_STORAGE_BUFFER, v1, sizeof(v1));
   GLuint b_v2 = buffer(GL_SHADER_STORAGE_BUFFER, v2, sizeof(v2));
   GLuint b_frags = buffer(GL_SHADER_STORAGE_BUFFER, zero, sizeof(zero));
   GLuint b_px = buffer(GL_SHADER_STORAGE_BUFFER, NULL, 256 * 4);
   GLuint tex, fbo;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   GLuint p_draw = program(compile(GL_VERTEX_SHADER, vs), compile(GL_FRAGMENT_SHADER, fs));
   glUseProgram(p_draw);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_chain); /* written by dispatch 2 */
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_v1);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, b_v2);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, b_frags);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, b_px);
   glViewport(0, 0, 16, 16);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   uint8_t pix[256 * 4];
   glReadPixels(0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, pix);
   const uint32_t *chain = map_read(b_chain, sizeof(in));
   uint32_t chain_copy[256];
   memcpy(chain_copy, chain, sizeof(chain_copy));
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   bad = 0;
   for (unsigned i = 0; i < 256; i++) {
      const uint8_t want[4] = { chain_copy[i] & 255, 128, 64, 255 };
      for (unsigned k = 0; k < 4; k++)
         bad += abs((int)pix[i * 4 + k] - (int)want[k]) > 1;
   }
   CHECK(!bad, "fragment SSBO reads: %u of 1024 channels wrong", bad);
   hash(pix, sizeof(pix));
   const uint32_t *frags = map_read(b_frags, 4);
   CHECK(frags && frags[0] == 256, "fragment atomics: %u (want 256)", frags ? frags[0] : 0);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   const uint32_t *px = map_read(b_px, 256 * 4);
   bad = 0;
   for (unsigned i = 0; px && i < 256; i++)
      bad += px[i] != chain_copy[i] + 7u;
   CHECK(px && !bad, "fragment SSBO stores: %u of 256 wrong (px[0] %u want %u)", bad,
         px ? px[0] : 0, chain_copy[0] + 7u);
   if (px)
      hash(px, 256 * 4);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   /* --- 5: 16 SSBOs + UBO binding 14 in compute and fragment shaders --- */
   {
      GLuint sb[16];
      uint32_t want = 0;
      for (unsigned k = 0; k < 15; k++) {
         const uint32_t v = 1000u + k * 37u;
         sb[k] = buffer(GL_SHADER_STORAGE_BUFFER, &v, 4);
         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, k, sb[k]);
      }
      {
         uint32_t z[258] = { 0 };
         sb[15] = buffer(GL_SHADER_STORAGE_BUFFER, z, sizeof(z));
      }
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, sb[15]);
      const uint32_t u14[4] = { 4242, 0, 0, 0 };
      GLuint b_u14 = buffer(GL_UNIFORM_BUFFER, u14, sizeof(u14));
      glBindBufferBase(GL_UNIFORM_BUFFER, 14, b_u14);
      for (unsigned k = 0; k < 15; k++)
         want += (1000u + k * 37u) * (k + 1);
      want += 4242;
      GLuint p16 = program(compile(GL_COMPUTE_SHADER, cs_ssbo16), 0);
      glUseProgram(p16);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      const uint32_t want_fs = want;
      GLuint pf16 = program(compile(GL_VERTEX_SHADER, vs), compile(GL_FRAGMENT_SHADER, fs_ssbo16));
      glUseProgram(pf16);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
      const uint32_t *r = map_read(sb[15], 258 * 4);
      CHECK(r && r[0] == want, "compute 16 SSBOs + UBO 14: %u (want %u)", r ? r[0] : 0, want);
      bad = 0;
      for (unsigned i = 0; r && i < 256; i++)
         bad += r[1 + i] != want_fs + i;
      CHECK(r && !bad, "fragment 16 SSBOs + UBO 14: %u of 256 wrong (r[1] %u want %u)", bad,
            r ? r[1] : 0, want_fs);
      if (r)
         hash(r, 258 * 4);
      CHECK(r && r[257] == 257, "16-SSBO atomics: %u (want 257)", r ? r[257] : 0);
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   }

   /* --- 6: textures in a compute shader --- */
   {
      uint8_t texel[64];
      uint32_t utexel[64];
      for (unsigned i = 0; i < 64; i++) {
         texel[i] = (uint8_t)(i * 13 + 5);
         utexel[i] = i * 7 + 1;
      }
      GLuint t[2];
      glGenTextures(2, t);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, t[0]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, texel);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, t[1]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, 4, 4);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA_INTEGER, GL_UNSIGNED_INT, utexel);
      /* Integer textures are incomplete with LINEAR filtering. */
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      GLuint pt = program(compile(GL_COMPUTE_SHADER, cs_tex), 0);
      glUseProgram(pt);
      glUniform1i(glGetUniformLocation(pt, "t0"), 0);
      glUniform1i(glGetUniformLocation(pt, "t1"), 1);
      GLuint b_t = buffer(GL_SHADER_STORAGE_BUFFER, NULL, 64 * 4);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_t);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
      const uint32_t *o = map_read(b_t, 64 * 4);
      bad = 0;
      for (unsigned i = 0; o && i < 64; i++)
         bad += o[i] != texel[i] + utexel[i] * 1000u;
      CHECK(o && !bad, "compute textures: %u of 64 wrong (o[0] %u want %u)", bad,
            o ? o[0] : 0, texel[0] + utexel[0] * 1000u);
      if (o)
         hash(o, 64 * 4);
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   }
   CHECK(glGetError() == GL_NO_ERROR, "GL error at the end");

   printf("digest %08x\n%s\n", digest, failures ? "FAIL" : "PASS");
   return failures != 0;
}
