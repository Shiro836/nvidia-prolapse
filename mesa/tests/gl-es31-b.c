/* SPDX-License-Identifier: MIT
 *
 * GLES 3.1 part B on nvrm: shader images (store/load/atomics from fragment
 * shaders), glMemoryBarrier, framebuffer_no_attachment, textureGather
 * (+Offset, +Offsets, shadow), stencil texturing (DEPTH_STENCIL_TEXTURE_MODE),
 * texelFetch on sampler2DMS + GL_SAMPLE_POSITION.
 * Prints one line per value; PASS/FAIL lines for checks with a CPU
 * expectation. The value lines are compared against llvmpipe (diff).
 * Needs an ES 3.1 context: until compute (part A) lands, run with
 * MESA_GLES_VERSION_OVERRIDE=3.1 on nvrm.
 *
 *   cc -std=c11 -I<mesa>/include gl-es31-b.c -L$B/src/egl -lEGL \
 *      -L$B/src/mesa/glapi/es2api -lGLESv2 -Wl,-rpath,... -o gl-es31-b
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
                           else { printf("PASS: " __VA_ARGS__); printf("\n"); } } while (0)

static GLuint
prog(const char *vs, const char *fs)
{
   const char *src[2] = { vs, fs };
   GLenum types[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
   GLuint p = glCreateProgram();
   for (int i = 0; i < 2; i++) {
      GLuint s = glCreateShader(types[i]);
      glShaderSource(s, 1, &src[i], NULL);
      glCompileShader(s);
      GLint ok = 0;
      glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[4096];
         glGetShaderInfoLog(s, sizeof(log), NULL, log);
         printf("FAIL: compile: %s\n%s\n", log, src[i]);
         exit(1);
      }
      glAttachShader(p, s);
   }
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      printf("FAIL: link: %s\n", log);
      exit(1);
   }
   return p;
}

static const char *VS =
   "#version 310 es\n"
   "void main() {\n"
   "  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);\n"
   "  gl_Position = vec4(p, 0.0, 1.0);\n"
   "}\n";

static void
draw_fullscreen(void)
{
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* Render `fs` (writing vec4 o) into a w x h RGBA32F target, print values. */
static void
run_rgba32f(const char *name, GLuint p, int w, int h, float *out)
{
   GLuint tex, fbo;
   /* Unit 7: never a sampled unit here, so the target does not replace
    * the texture under test on unit 0. */
   glActiveTexture(GL_TEXTURE7);
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
   glActiveTexture(GL_TEXTURE0);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   glViewport(0, 0, w, h);
   glUseProgram(p);
   draw_fullscreen();
   glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, out);
   for (int i = 0; i < w * h; i++)
      printf("%s[%d] %.6f %.6f %.6f %.6f\n", name, i, out[i * 4], out[i * 4 + 1],
             out[i * 4 + 2], out[i * 4 + 3]);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &tex);
}

static void
test_images_no_attachment(void)
{
   const int W = 16, H = 8;
   GLuint img, cnt;
   glGenTextures(1, &img);
   glBindTexture(GL_TEXTURE_2D, img);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, W, H);
   glGenTextures(1, &cnt);
   glBindTexture(GL_TEXTURE_2D, cnt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
   const GLuint zero = 0;
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
   glBindImageTexture(0, img, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
   glBindImageTexture(1, cnt, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);

   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, W);
   glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, H);
   CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
         "no-attachment FBO complete");
   GLuint p = prog(VS,
      "#version 310 es\n"
      "#extension GL_OES_shader_image_atomic : require\n"
      "precision highp float;\n"
      "layout(r32ui, binding = 0) writeonly uniform highp uimage2D img;\n"
      "layout(r32ui, binding = 1) uniform highp uimage2D cnt;\n"
      "void main() {\n"
      "  ivec2 c = ivec2(gl_FragCoord.xy);\n"
      "  imageStore(img, c, uvec4(uint(c.x + c.y * 100 + 7)));\n"
      "  imageAtomicAdd(cnt, ivec2(0), 1u);\n"
      "}\n");
   glViewport(0, 0, W, H);
   glUseProgram(p);
   draw_fullscreen();
   glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                   GL_TEXTURE_FETCH_BARRIER_BIT);

   /* Read the image back through a colour attachment. */
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, img, 0);
   GLuint px[16 * 8 * 4];
   glReadPixels(0, 0, W, H, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
   int bad = 0;
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
         bad += px[(y * W + x) * 4] != (GLuint)(x + y * 100 + 7);
   CHECK(bad == 0, "imageStore from FS, no-attachment FBO %dx%d: %d wrong texels", W, H, bad);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cnt, 0);
   glReadPixels(0, 0, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
   CHECK(px[0] == (GLuint)(W * H), "imageAtomicAdd fragment count %u (want %d)", px[0], W * H);

   /* imageLoad (formatted r32ui + rgba8) into a colour output. */
   GLuint rgba;
   glGenTextures(1, &rgba);
   glBindTexture(GL_TEXTURE_2D, rgba);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
   unsigned char t[4 * 4 * 4];
   for (int i = 0; i < 64; i++)
      t[i] = (unsigned char)(i * 4 + 1);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, t);
   glBindImageTexture(2, rgba, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
   glBindImageTexture(0, img, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
   GLuint p2 = prog(VS,
      "#version 310 es\n"
      "precision highp float;\n"
      "layout(rgba8, binding = 2) readonly uniform highp image2D tex;\n"
      "layout(r32ui, binding = 0) readonly uniform highp uimage2D img;\n"
      "out vec4 o;\n"
      "void main() {\n"
      "  ivec2 c = ivec2(gl_FragCoord.xy);\n"
      "  vec4 v = imageLoad(tex, c);\n"
      "  o = vec4(v.rgb, float(imageLoad(img, c * 3).r));\n"
      "}\n");
   float out[4 * 4 * 4];
   run_rgba32f("imageload", p2, 4, 4, out);
   bad = 0;
   for (int i = 0; i < 16; i++) {
      const int x = i % 4, y = i / 4;
      for (int c = 0; c < 3; c++)
         bad += out[i * 4 + c] - t[i * 4 + c] / 255.0f > 1e-6f ||
                t[i * 4 + c] / 255.0f - out[i * 4 + c] > 1e-6f;
      /* imageLoad outside the 16x8 image returns 0 (ES 3.1 8.26). */
      bad += out[i * 4 + 3] != (y * 3 < 8 ? (float)(x * 3 + y * 3 * 100 + 7) : 0.0f);
   }
   CHECK(bad == 0, "imageLoad rgba8 + r32ui: %d wrong", bad);
   glDeleteFramebuffers(1, &fbo);
}

/* Layered / non-layered image views from a fragment shader: 2D array
 * layer as image2D, 3D level as image3D and one slice as image2D, cube
 * (layered) as imageCube, imageSize of each, r32f atomic exchange and
 * r32i compare-swap. Read back per layer through FBO layer attachments. */
static void
test_image_views(void)
{
   const int N = 4;
   GLuint arr, vol, cube, fi;
   glGenTextures(1, &arr);
   glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
   glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, N, N, 3);
   glGenTextures(1, &vol);
   glBindTexture(GL_TEXTURE_3D, vol);
   glTexStorage3D(GL_TEXTURE_3D, 2, GL_R32UI, N, N, 4);
   glGenTextures(1, &cube);
   glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
   glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, N, N);
   glGenTextures(1, &fi);
   glBindTexture(GL_TEXTURE_2D, fi);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, 2, 1);
   const GLint init[2] = { 5, 9 };
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 1, GL_RED_INTEGER, GL_INT, init);

   glBindImageTexture(0, arr, 0, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);  /* layer 1 */
   glBindImageTexture(1, vol, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);   /* whole level 0 */
   glBindImageTexture(2, vol, 1, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);  /* level 1 slice 1 */
   glBindImageTexture(3, cube, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
   glBindImageTexture(4, fi, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, N);
   glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, N);
   GLuint p = prog(VS,
      "#version 310 es\n"
      "#extension GL_OES_shader_image_atomic : require\n"
      "precision highp float;\n"
      "layout(r32ui, binding = 0) writeonly uniform highp uimage2D a;\n"
      "layout(r32ui, binding = 1) writeonly uniform highp uimage3D v;\n"
      "layout(r32ui, binding = 2) writeonly uniform highp uimage2D vs;\n"
      "layout(r32ui, binding = 3) writeonly uniform highp uimageCube c;\n"
      "layout(r32i, binding = 4) uniform highp iimage2D f;\n"
      "void main() {\n"
      "  ivec2 p = ivec2(gl_FragCoord.xy);\n"
      "  uint id = uint(p.x + p.y * 4);\n"
      "  imageStore(a, p, uvec4(1000u + id));\n"
      "  for (int z = 0; z < 4; z++) imageStore(v, ivec3(p, z), uvec4(2000u + id + uint(z) * 100u));\n"
      "  if (p.x < 2 && p.y < 2) imageStore(vs, p, uvec4(3000u + id));\n"
      "  for (int l = 0; l < 6; l++) imageStore(c, ivec3(p, l), uvec4(4000u + id + uint(l) * 100u));\n"
      "  ivec2 sa = imageSize(a); ivec3 sv = imageSize(v); ivec2 ss = imageSize(vs); ivec2 sc = imageSize(c);\n"
      "  if (p == ivec2(0)) {\n"
      "    imageAtomicCompSwap(f, ivec2(0), 5, sa.x * 1000 + sv.z * 100 + ss.x * 10 + sc.x);\n"
      "    imageAtomicExchange(f, ivec2(1, 0), 77);\n"
      "  }\n"
      "}\n");
   glViewport(0, 0, N, N);
   glUseProgram(p);
   draw_fullscreen();
   glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
   GLuint px[4 * 4 * 4];
   int bad = 0;
   glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, 0, 1);
   glReadPixels(0, 0, N, N, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
   for (int i = 0; i < 16; i++) bad += px[i * 4] != 1000u + i;
   CHECK(bad == 0, "image2D on 2D array layer 1: %d wrong", bad);
   bad = 0;
   for (int z = 0; z < 4; z++) {
      glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vol, 0, z);
      glReadPixels(0, 0, N, N, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
      for (int i = 0; i < 16; i++) bad += px[i * 4] != 2000u + i + z * 100u;
   }
   CHECK(bad == 0, "layered image3D level 0: %d wrong", bad);
   glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, vol, 1, 1);
   glReadPixels(0, 0, 2, 2, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
   bad = 0;
   for (int y = 0; y < 2; y++)
      for (int x = 0; x < 2; x++) bad += px[(y * 2 + x) * 4] != 3000u + x + y * 4;
   CHECK(bad == 0, "image2D on 3D level 1 slice 1: %d wrong", bad);
   bad = 0;
   for (int l = 0; l < 6; l++) {
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_CUBE_MAP_POSITIVE_X + l, cube, 0);
      glReadPixels(0, 0, N, N, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
      for (int i = 0; i < 16; i++) bad += px[i * 4] != 4000u + i + l * 100u;
   }
   CHECK(bad == 0, "layered imageCube: %d wrong", bad);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fi, 0);
   GLint ipx[2 * 4];
   glReadPixels(0, 0, 2, 1, GL_RGBA_INTEGER, GL_INT, ipx);
   CHECK(ipx[0] == 4 * 1000 + 4 * 100 + 2 * 10 + 4 && ipx[4] == 77,
         "imageSize (2D layer, 3D, 3D slice, cube) via compswap = %d, exchange = %d", ipx[0], ipx[4]);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);
}

/* More image units than stock NVIDIA GLES's 8 (MAX_IMAGE_UNITS 8 broke
 * Minecraft Vibrant Visuals under nvnative): one fragment shader writes
 * `n` (16, 32) separately bound R32UI images, then reads each back. */
static void
test_many_images(int n)
{
   GLint max = 0;
   glGetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS, &max);
   if (max < n) {
      printf("FAIL: MAX_FRAGMENT_IMAGE_UNIFORMS %d < %d\n", max, n);
      failures++;
      return;
   }
   GLuint tex[32];
   glGenTextures(n, tex);
   for (int i = 0; i < n; i++) {
      glBindTexture(GL_TEXTURE_2D, tex[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 4, 4);
      glBindImageTexture(i, tex[i], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
   }
   static char fs[16384];
   int o = snprintf(fs, sizeof(fs), "#version 310 es\nprecision highp float;\n");
   for (int i = 0; i < n; i++)
      o += snprintf(fs + o, sizeof(fs) - o,
                    "layout(r32ui, binding = %d) writeonly uniform highp uimage2D i%d;\n", i, i);
   o += snprintf(fs + o, sizeof(fs) - o,
                 "void main() {\n  ivec2 p = ivec2(gl_FragCoord.xy);\n"
                 "  uint id = uint(p.x + p.y * 4);\n");
   for (int i = 0; i < n; i++)
      o += snprintf(fs + o, sizeof(fs) - o, "  imageStore(i%d, p, uvec4(id * 1000u + %du));\n", i, i);
   snprintf(fs + o, sizeof(fs) - o, "}\n");
   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, 4);
   glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, 4);
   glViewport(0, 0, 4, 4);
   glUseProgram(prog(VS, fs));
   draw_fullscreen();
   glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
   int bad = 0;
   GLuint px[16 * 4];
   for (int i = 0; i < n; i++) {
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[i], 0);
      glReadPixels(0, 0, 4, 4, GL_RGBA_INTEGER, GL_UNSIGNED_INT, px);
      for (int t = 0; t < 16; t++) {
         printf("many%d[%d.%d] %u\n", n, i, t, px[t * 4]);
         bad += px[t * 4] != (GLuint)(t * 1000 + i);
      }
   }
   CHECK(bad == 0, "%d image units in one fragment shader: %d wrong", n, bad);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(n, tex);
}

static void
test_gather(void)
{
   GLuint tex;
   glGenTextures(1, &tex);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   unsigned char t[8 * 8 * 4];
   for (int i = 0; i < 64; i++) {
      t[i * 4 + 0] = (unsigned char)(i * 3);
      t[i * 4 + 1] = (unsigned char)(255 - i * 2);
      t[i * 4 + 2] = (unsigned char)(i * 5 % 256);
      t[i * 4 + 3] = (unsigned char)(i * 7 % 256);
   }
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, t);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   GLuint p = prog(VS,
      "#version 310 es\n"
      "precision highp float;\n"
      "uniform highp sampler2D s;\n"
      "uniform int mode;\n"
      "out vec4 o;\n"
      "void main() {\n"
      "  vec2 uv = (gl_FragCoord.xy * 2.0 + 0.5) / 8.0;\n"
      "  if (mode == 0) o = textureGather(s, uv);\n"
      "  else if (mode == 1) o = textureGather(s, uv, 1);\n"
      "  else if (mode == 2) o = textureGather(s, uv, 3);\n"
      "  else o = textureGatherOffset(s, uv, ivec2(-1, 1), 2);\n"
      "}\n");
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "s"), 0);
   float out[4 * 4 * 4];
   char name[32];
   for (int m = 0; m < 4; m++) {
      glUseProgram(p);
      glUniform1i(glGetUniformLocation(p, "mode"), m);
      snprintf(name, sizeof(name), "gather%d", m);
      run_rgba32f(name, p, 4, 4, out);
   }
   /* CPU check of mode 0 (component 0), wrap REPEAT: footprint at
    * texel (2x+0.5+... ) -> i0 = floor(u*8 - 0.5). */
   int bad = 0;
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "mode"), 0);
   run_rgba32f("gather0-check", p, 4, 4, out);
   for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++) {
         const int i0 = (int)((2 * x + 0.5f) * 2 + 0.5f) / 2; /* unused */
         (void)i0;
         const float u = ((x + 0.5f) * 2 + 0.5f) / 8 * 8 - 0.5f;
         const float v = ((y + 0.5f) * 2 + 0.5f) / 8 * 8 - 0.5f;
         const int x0 = (int)u, y0 = (int)v;
         const int xs[4] = { x0, x0 + 1, x0 + 1, x0 };
         const int ys[4] = { y0 + 1, y0 + 1, y0, y0 };
         for (int c = 0; c < 4; c++) {
            const int tx = xs[c] & 7, ty = ys[c] & 7;
            const float want = t[(ty * 8 + tx) * 4] / 255.0f, got = out[(y * 4 + x) * 4 + c];
            bad += got - want > 1e-6f || want - got > 1e-6f; /* llvmpipe's unorm is 1 ulp off */
         }
      }
   CHECK(bad == 0, "textureGather component 0 footprint: %d wrong", bad);

   /* Shadow gather on a DEPTH_COMPONENT32F texture, compare LEQUAL. */
   GLuint dt;
   glGenTextures(1, &dt);
   glBindTexture(GL_TEXTURE_2D, dt);
   float d[64];
   for (int i = 0; i < 64; i++)
      d[i] = (i % 9) / 8.0f;
   glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 8, 8, 0, GL_DEPTH_COMPONENT, GL_FLOAT, d);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
   GLuint ps = prog(VS,
      "#version 310 es\n"
      "precision highp float;\n"
      "uniform highp sampler2DShadow s;\n"
      "out vec4 o;\n"
      "void main() {\n"
      "  vec2 uv = (gl_FragCoord.xy * 2.0 + 0.5) / 8.0;\n"
      "  o = textureGather(s, uv, 0.5) + 2.0 * textureGatherOffset(s, uv, 0.3, ivec2(1, -1));\n"
      "}\n");
   run_rgba32f("gathershadow", ps, 4, 4, out);
}

static void
test_stencil_texturing(void)
{
   GLuint tex;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   GLuint ds[4 * 4];
   for (int i = 0; i < 16; i++)
      ds[i] = ((GLuint)(i * 1000) << 8) | (GLuint)(i * 13 + 3);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, 4, 4, 0, GL_DEPTH_STENCIL,
                GL_UNSIGNED_INT_24_8, ds);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX);
   GLuint p = prog(VS,
      "#version 310 es\n"
      "precision highp float;\n"
      "uniform highp usampler2D s;\n"
      "out vec4 o;\n"
      "void main() { o = vec4(texelFetch(s, ivec2(gl_FragCoord.xy), 0)); }\n");
   float out[4 * 4 * 4];
   run_rgba32f("stencil", p, 4, 4, out);
   int bad = 0;
   for (int i = 0; i < 16; i++)
      bad += out[i * 4] != (float)(i * 13 + 3);
   CHECK(bad == 0, "stencil texturing D24S8 (DEPTH_STENCIL_TEXTURE_MODE): %d wrong", bad);
   /* Back to depth: same texture sampled as depth. */
   glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_DEPTH_COMPONENT);
   GLuint pd = prog(VS,
      "#version 310 es\n"
      "precision highp float;\n"
      "uniform highp sampler2D s;\n"
      "out vec4 o;\n"
      "void main() { o = texelFetch(s, ivec2(gl_FragCoord.xy), 0); }\n");
   run_rgba32f("stencil-depthmode", pd, 4, 4, out);
}

static void
test_ms_fetch(void)
{
   GLuint ms, fbo;
   glGenTextures(1, &ms);
   glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
   glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, 4, 4, GL_TRUE);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, ms, 0);
   CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "MS texture FBO");
   for (int i = 0; i < 4; i++) {
      GLfloat pos[2];
      glGetMultisamplefv(GL_SAMPLE_POSITION, i, pos);
      printf("samplepos[%d] %.4f %.4f\n", i, pos[0], pos[1]);
   }
   glViewport(0, 0, 4, 4);
   glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   /* Diagonal triangle: covers some samples of the diagonal pixels. */
   GLuint pt = prog(
      "#version 310 es\n"
      "void main() {\n"
      "  vec2 p = vec2(gl_VertexID == 1 ? 1.0 : -1.0, gl_VertexID == 2 ? 1.0 : -1.0);\n"
      "  gl_Position = vec4(p, 0.0, 1.0);\n"
      "}\n",
      "#version 310 es\n"
      "precision highp float; out vec4 o; void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }\n");
   glUseProgram(pt);
   draw_fullscreen();
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   GLuint p = prog(VS,
      "#version 310 es\n"
      "precision highp float;\n"
      "uniform highp sampler2DMS s;\n"
      "out vec4 o;\n"
      "void main() {\n"
      "  ivec2 c = ivec2(gl_FragCoord.xy);\n"
      "  o = vec4(texelFetch(s, c, 0).r, texelFetch(s, c, 1).r, texelFetch(s, c, 2).r,\n"
      "           texelFetch(s, c, 3).r + float(textureSize(s).x) * 10.0);\n"
      "}\n");
   float out[4 * 4 * 4];
   run_rgba32f("msfetch", p, 4, 4, out);
}

int
main(void)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(dpy, NULL, NULL))
      return 2;
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint ca[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: no ES 3.1 context (0x%x)\n", eglGetError());
      return 1;
   }
   printf("# GL_RENDERER %s / %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
   GLint v;
   glGetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS, &v);
   printf("# MAX_FRAGMENT_IMAGE_UNIFORMS %d\n", v);
   glGetIntegerv(GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET, &v);
   printf("# MAX_PROGRAM_TEXTURE_GATHER_OFFSET %d\n", v);
   glGetIntegerv(GL_MAX_FRAMEBUFFER_SAMPLES, &v);
   printf("# MAX_FRAMEBUFFER_SAMPLES %d\n", v);
   test_images_no_attachment();
   test_image_views();
   test_many_images(16);
   test_many_images(32);
   test_gather();
   test_stencil_texturing();
   test_ms_fetch();
   GLenum err = glGetError();
   CHECK(err == GL_NO_ERROR, "glGetError 0x%x", err);
   printf("%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
