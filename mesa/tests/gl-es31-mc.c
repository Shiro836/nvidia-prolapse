/* SPDX-License-Identifier: MIT
 *
 * GLES 3.1 app gate modelled on Minecraft Bedrock's renderer (RenderDragon/bgfx).
 *
 * Measured 2026-10-15 (build/nvrm-mc): every one of the 14,290 shaders in MC's
 * assets/renderer/materials/<name>.material.bin is "#version 310 es" (ESSL_310 is the
 * only platform shipped); fragment shaders use up to 5 std430 SSBO blocks (1,915),
 * textureGather (2,163), samplerCubeArray via GL_EXT_texture_cube_map_array (1,702),
 * centroid varyings; 44 compute shaders use image load/store, SSBOs and atomicAdd.
 * On an ES 3.0 context bgfx's init fails glBindBuffer(GL_SHADER_STORAGE_BUFFER) and
 * glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY) with GL errors, MC never calls
 * glCompileShader for a material and presents bgfx clears only: a black screen.
 *
 * This test checks the same feature set with our own shaders (not MC's code):
 *   1. an ES 3.1 context (GLSL ES 3.10) and the limits MC needs,
 *   2. a fragment shader reading 5 SSBO blocks + textureGather + samplerCubeArray
 *      + a centroid varying, rendered into RGBA32F and read back,
 *   3. a compute dispatch doing imageStore + SSBO writes + atomicAdd.
 * Exit 0 only if all pass. On an ES 3.0 driver it reports what is missing.
 *
 *   cc -std=c11 -I<mesa>/include gl-es31-mc.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -o gl-es31-mc
 *   ./gl-es31-mc            (llvmpipe comparator: LIBGL_ALWAYS_SOFTWARE=1 with Mesa's libEGL)
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_TEXTURE_CUBE_MAP_ARRAY_EXT
#define GL_TEXTURE_CUBE_MAP_ARRAY_EXT 0x9009
#endif

static int failures;

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static int
has_ext(const char *name)
{
   GLint n = 0;
   glGetIntegerv(GL_NUM_EXTENSIONS, &n);
   for (GLint i = 0; i < n; i++)
      if (!strcmp((const char *)glGetStringi(GL_EXTENSIONS, i), name))
         return 1;
   return 0;
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
      printf("FAIL: compile %s shader:\n%s\n", type == GL_COMPUTE_SHADER ? "compute" :
             type == GL_VERTEX_SHADER ? "vertex" : "fragment", log);
      failures++;
      return 0;
   }
   return s;
}

static GLuint
link(GLuint a, GLuint b)
{
   if (!a || (b == 0 && a == 0))
      return 0;
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
      failures++;
      return 0;
   }
   return p;
}

static const char *vs_src =
   "#version 310 es\n"
   "layout(location = 0) in vec2 pos;\n"
   "centroid out highp vec2 v_uv;\n"
   "void main() { v_uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";

static const char *fs_src =
   "#version 310 es\n"
   "#extension GL_EXT_texture_cube_map_array : require\n"
   "precision highp float;\n"
   "layout(std430, binding = 0) readonly buffer B0 { vec4 v; } b0;\n"
   "layout(std430, binding = 1) readonly buffer B1 { vec4 v; } b1;\n"
   "layout(std430, binding = 2) readonly buffer B2 { vec4 v; } b2;\n"
   "layout(std430, binding = 3) readonly buffer B3 { vec4 v; } b3;\n"
   "layout(std430, binding = 4) readonly buffer B4 { vec4 v; } b4;\n"
   "layout(binding = 0) uniform highp sampler2D s_gather;\n"
   "layout(binding = 1) uniform highp samplerCubeArray s_cubes;\n"
   "centroid in highp vec2 v_uv;\n"
   "layout(location = 0) out vec4 o;\n"
   "void main() {\n"
   "   float ssbo = b0.v.x + b1.v.x + b2.v.x + b3.v.x + b4.v.x;\n"
   "   vec4 g = textureGather(s_gather, vec2(0.5));\n"
   "   float cube = texture(s_cubes, vec4(1.0, 0.0, 0.0, 1.0)).r;\n"
   "   o = vec4(ssbo, g.x + g.y + g.z + g.w, cube, v_uv.x > 0.0 ? 1.0 : 0.0);\n"
   "}\n";

static const char *cs_src =
   "#version 310 es\n"
   "layout(local_size_x = 4, local_size_y = 4) in;\n"
   "layout(rgba8, binding = 0) writeonly uniform highp image2D img;\n"
   "layout(std430, binding = 0) buffer C { uint count; uint vals[]; } c;\n"
   "void main() {\n"
   "   uvec2 id = gl_GlobalInvocationID.xy;\n"
   "   uint idx = id.y * 16u + id.x;\n"
   "   imageStore(img, ivec2(id), vec4(float(id.x) / 15.0, float(id.y) / 15.0, 0.0, 1.0));\n"
   "   c.vals[idx] = idx * 3u;\n"
   "   atomicAdd(c.count, 1u);\n"
   "}\n";

static void
fragment_test(void)
{
   GLuint p = link(compile(GL_VERTEX_SHADER, vs_src), compile(GL_FRAGMENT_SHADER, fs_src));
   if (!p)
      return;

   GLuint ssbo[5];
   glGenBuffers(5, ssbo);
   for (int i = 0; i < 5; i++) {
      float v[4] = { 0.1f * (i + 1), 0, 0, 0 };
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, ssbo[i]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
   }

   GLuint tex[2];
   glGenTextures(2, tex);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex[0]);
   const unsigned char g[4] = { 10, 20, 30, 40 };
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, 2, 2);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 2, GL_RED, GL_UNSIGNED_BYTE, g);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, tex[1]);
   glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, 1, GL_RGBA8, 1, 1, 12);
   for (int slice = 0; slice < 12; slice++) {
      const unsigned char px[4] = { (unsigned char)(slice * 16), 0, 0, 255 };
      glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, 0, 0, 0, slice, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
   }
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

   GLuint rb, fb;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, 8, 8);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "RGBA32F FBO incomplete");
   glViewport(0, 0, 8, 8);

   static const float tri[] = { -1, -1, 3, -1, -1, 3 };
   GLuint vb, va;
   glGenVertexArrays(1, &va);
   glBindVertexArray(va);
   glGenBuffers(1, &vb);
   glBindBuffer(GL_ARRAY_BUFFER, vb);
   glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   float out[4] = { -1, -1, -1, -1 };
   glReadPixels(4, 4, 1, 1, GL_RGBA, GL_FLOAT, out);
   const float want[4] = { 1.5f, 100.0f / 255.0f, 96.0f / 255.0f, 1.0f };
   const char *what[4] = { "5 SSBO blocks", "textureGather", "samplerCubeArray layer 1 +X", "centroid varying" };
   for (int i = 0; i < 4; i++)
      CHECK(fabsf(out[i] - want[i]) < 2e-3f, "fragment %s: got %f want %f", what[i], out[i], want[i]);
   if (!failures)
      printf("PASS: fragment SSBOx5 %.4f, gather %.4f, cube array %.4f\n", out[0], out[1], out[2]);
   GLenum e = glGetError();
   CHECK(e == GL_NO_ERROR, "fragment test GL error 0x%x", e);
}

static void
compute_test(void)
{
   GLuint p = link(compile(GL_COMPUTE_SHADER, cs_src), 0);
   if (!p)
      return;
   GLuint tex, buf;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
   glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
   unsigned zero[1 + 256];
   memset(zero, 0, sizeof(zero));
   glGenBuffers(1, &buf);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
   glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), zero, GL_DYNAMIC_COPY);
   glUseProgram(p);
   glDispatchCompute(4, 4, 1);
   glMemoryBarrier(GL_ALL_BARRIER_BITS);

   const unsigned *c = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zero), GL_MAP_READ_BIT);
   if (!c) {
      CHECK(0, "compute: map SSBO");
      return;
   }
   CHECK(c[0] == 256, "compute atomicAdd count %u want 256", c[0]);
   int bad = 0;
   for (unsigned i = 0; i < 256; i++)
      bad += c[1 + i] != i * 3;
   CHECK(bad == 0, "compute SSBO writes: %d of 256 wrong", bad);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   GLuint fb;
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   unsigned char px[16 * 16 * 4];
   glReadPixels(0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, px);
   int badpx = 0;
   for (int y = 0; y < 16; y++)
      for (int x = 0; x < 16; x++) {
         const unsigned char *q = px + (y * 16 + x) * 4;
         int wr = (int)lroundf(x / 15.0f * 255), wg = (int)lroundf(y / 15.0f * 255);
         badpx += abs(q[0] - wr) > 1 || abs(q[1] - wg) > 1 || q[3] != 255;
      }
   CHECK(badpx == 0, "compute imageStore: %d of 256 texels wrong", badpx);
   GLenum e = glGetError();
   CHECK(e == GL_NO_ERROR, "compute test GL error 0x%x", e);
   if (!bad && !badpx)
      printf("PASS: compute imageStore 256/256, SSBO 256/256, atomic count\n");
}

int
main(void)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = get_display ? get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL)
                                : EGL_NO_DISPLAY;
   EGLint major, minor;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &major, &minor) || !eglBindAPI(EGL_OPENGL_ES_API)) {
      printf("FAIL: EGL surfaceless init (0x%x)\n", eglGetError());
      return 1;
   }
   static const EGLint es31[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   static const EGLint es30[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, es31);
   int have31 = ctx != EGL_NO_CONTEXT;
   if (!have31)
      ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, es30);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: no ES 3.x context (0x%x)\n", eglGetError());
      return 1;
   }
   printf("GL_RENDERER %s\nGL_VERSION %s\nGLSL %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION),
          glGetString(GL_SHADING_LANGUAGE_VERSION));

   CHECK(have31, "no OpenGL ES 3.1 context: MC's materials are all #version 310 es");
   CHECK(has_ext("GL_EXT_texture_cube_map_array") || has_ext("GL_OES_texture_cube_map_array"),
         "GL_EXT_texture_cube_map_array missing (MC PBR fragment shaders #extension ... : require)");
   if (!have31) {
      printf("RESULT: FAIL (%d)\n", failures);
      return 1;
   }
   GLint fs_ssbo = 0, cs_ssbo = 0, cs_img = 0;
   glGetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &fs_ssbo);
   glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &cs_ssbo);
   glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &cs_img);
   printf("limits: fragment SSBO blocks %d, compute SSBO blocks %d, compute images %d\n", fs_ssbo, cs_ssbo, cs_img);
   CHECK(fs_ssbo >= 5, "GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS %d < 5 (MC fragment shaders use up to 5)", fs_ssbo);
   CHECK(cs_ssbo >= 5, "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS %d < 5", cs_ssbo);

   fragment_test();
   compute_test();
   printf("RESULT: %s (%d)\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
