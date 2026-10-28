/*
 * Minecraft visual bugs on nvrm (2026-10-15): clip control, depth clamp and
 * copy_image (ES 3.1 context). Self-checking against analytic expectations,
 * so the same binary run on llvmpipe validates the expectations.
 *
 * Why: Minecraft Bedrock (bgfx/RenderDragon; libminecraftpe.so carries the
 * strings EXT_clip_control, EXT_depth_clamp, OES/EXT_copy_image,
 * glClipControl, glCopyImageSubDataEXT, glCopyTexSubImage2D) cut distant
 * terrain, flashed a black sky and logged ~1,034 "glCopyTexSubImage2D(invalid
 * readbuffer)" per 75 s with "bad format" on the read FBO's colour0 on nvrm,
 * which exposed none of these extensions (ANGLE exposes all of them).
 *
 * Sections:
 *   ext     GL_EXT_clip_control, GL_EXT_depth_clamp, GL_OES/EXT_copy_image.
 *   mc      bgfx's blit fallback without copy_image: a DEPTH texture attached
 *           as GL_COLOR_ATTACHMENT0 of the read FBO + glCopyTexSubImage2D =
 *           INCOMPLETE_ATTACHMENT + GL_INVALID_FRAMEBUFFER_OPERATION on every
 *           Mesa driver (Mesa fbobject.c "bad format" -> teximage.c "invalid
 *           readbuffer"); glCopyImageSubDataEXT of the same depth texture works.
 *   clip    8 strips at clip z/w = -1.5 .. 1.5 into D32F and D24 targets
 *           under {NEGATIVE_ONE,ZERO}_TO_ONE x depth clamp off/on, depth
 *           ranges (0,1), (0.25,0.75), (0.9,0.1): visibility per strip from
 *           colour, stored depth read by sampling the depth texture.
 *   fragz   gl_FragDepth = 1.5 / -0.5 is stored clamped to [0,1] for float
 *           and unorm depth, depth clamp off and on.
 *   copy    glCopyImageSubDataEXT: RGBA8 mip->level sub-rect, D32F, D24,
 *           D24S8 (depth + stencil sampled), 2D array layers -> 3D slices,
 *           cube face -> 2D, RGTC1/BC1 (8 B blocks) <-> RGBA16UI both
 *           directions, render -> copy -> sample ordering x64 without
 *           flushes, compute imageStore -> barrier -> copy -> sample,
 *           glCopyBufferSubData of 300,000 B across 128 KiB CE lines.
 *
 *   cc -std=c11 -I<mesa>/include gl-mc-vis.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -lm -o gl-mc-vis
 *   LIBGL_DRIVERS_PATH=<build>/src/gallium/targets/dri ./gl-mc-vis
 *   cc -std=c11 gl-mc-vis.c -lEGL -lGLESv2 -lm -o gl-mc-vis-sys
 *   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
 *      LIBGL_ALWAYS_SOFTWARE=1 ./gl-mc-vis-sys
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_LOWER_LEFT_EXT
#define GL_LOWER_LEFT_EXT 0x8CA1
#define GL_NEGATIVE_ONE_TO_ONE_EXT 0x935E
#define GL_ZERO_TO_ONE_EXT 0x935F
#endif
#ifndef GL_DEPTH_CLAMP_EXT
#define GL_DEPTH_CLAMP_EXT 0x864F
#endif
#ifndef GL_COMPRESSED_RED_RGTC1_EXT
#define GL_COMPRESSED_RED_RGTC1_EXT 0x8DBB
#endif
#ifndef GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#endif
#ifndef GL_DEPTH_STENCIL_TEXTURE_MODE
#define GL_DEPTH_STENCIL_TEXTURE_MODE 0x90EA
#endif

typedef void (*clip_control_fn)(GLenum, GLenum);
typedef void (*copy_image_fn)(GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum,
                              GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei);
static clip_control_fn ClipControl;
static copy_image_fn CopyImage;

static int failures;
static const char *only;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
                           else if (getenv("VERBOSE")) { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)
#define SECTION(n) (!only || !strcmp(only, n))

static GLuint
prog(const char *vs, const char *fs)
{
   GLuint p = glCreateProgram();
   const char *src[2] = {vs, fs};
   const GLenum ty[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   for (int i = 0; i < 2; i++) {
      GLuint s = glCreateShader(ty[i]);
      glShaderSource(s, 1, &src[i], NULL);
      glCompileShader(s);
      GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
      if (!ok) { char log[2048]; glGetShaderInfoLog(s, sizeof log, NULL, log); printf("shader: %s\n", log); exit(2); }
      glAttachShader(p, s);
   }
   glLinkProgram(p);
   GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) { printf("link failed\n"); exit(2); }
   return p;
}

static GLuint
cprog(const char *cs)
{
   GLuint p = glCreateProgram(), s = glCreateShader(GL_COMPUTE_SHADER);
   glShaderSource(s, 1, &cs, NULL); glCompileShader(s);
   GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) { char log[2048]; glGetShaderInfoLog(s, sizeof log, NULL, log); printf("cs: %s\n", log); exit(2); }
   glAttachShader(p, s); glLinkProgram(p);
   return p;
}

/* Full-viewport quad helpers: attribute-less, gl_VertexID. */
#define VS_QUAD "#version 310 es\n" \
   "uniform vec4 r; uniform float z;\n" \
   "void main(){ vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1);\n" \
   " gl_Position = vec4(mix(r.xy, r.zw, c), z, 1.0); }\n"

static GLuint p_strip, p_fragz, p_sample_depth, p_sample_color, p_sample_stencil, p_sample_3d;

static void
quad(GLuint p, float x0, float y0, float x1, float y1, float z)
{
   glUseProgram(p);
   glUniform4f(glGetUniformLocation(p, "r"), x0, y0, x1, y1);
   glUniform1f(glGetUniformLocation(p, "z"), z);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static GLuint
tex2d(GLenum ifmt, int w, int h, int levels)
{
   GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
   glTexStorage2D(GL_TEXTURE_2D, levels, ifmt, w, h);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   return t;
}

static GLuint
fbo_color(GLuint tex, int level)
{
   GLuint f; glGenFramebuffers(1, &f); glBindFramebuffer(GL_FRAMEBUFFER, f);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, level);
   return f;
}

/* Sample a depth (or colour) texture 1:1 into an RGBA32F target and read it. */
static void
read_tex_f(GLuint p, GLuint tex, int w, int h, float *out /* w*h*4 */)
{
   GLuint t = tex2d(GL_RGBA32F, w, h, 1), f = fbo_color(t, 0);
   glViewport(0, 0, w, h);
   glDisable(GL_DEPTH_TEST); glDisable(GL_STENCIL_TEST);
   glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "s"), 0);
   quad(p, -1, -1, 1, 1, 0);
   glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, out);
   glDeleteFramebuffers(1, &f); glDeleteTextures(1, &t);
}

static void
read_rgba8(GLuint tex, int level, int w, int h, unsigned char *out)
{
   GLuint f = fbo_color(tex, level);
   glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, out);
   glDeleteFramebuffers(1, &f);
}

/* ------------------------------------------------------------------ mc */

static void
test_mc(void)
{
   const int W = 32;
   GLuint dt = tex2d(GL_DEPTH_COMPONENT32F, W, W, 1);
   float dz[32 * 32];
   for (int i = 0; i < W * W; i++) dz[i] = (float)i / (W * W);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, W, GL_DEPTH_COMPONENT, GL_FLOAT, dz);
   GLuint dst = tex2d(GL_DEPTH_COMPONENT32F, W, W, 1);
   while (glGetError()) {}

   /* bgfx's fallback: src at the read FBO's colour 0, glCopyTexSubImage2D. */
   GLuint rf; glGenFramebuffers(1, &rf);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, rf);
   glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dt, 0);
   GLenum st = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
   glBindTexture(GL_TEXTURE_2D, dst);
   glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, W, W);
   GLenum e = glGetError();
   CHECK(st == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT && e == GL_INVALID_FRAMEBUFFER_OPERATION,
         "mc: depth at COLOR_ATTACHMENT0 status 0x%x, CopyTexSubImage2D error 0x%x "
         "(expected 0x8cd6 / 0x506 on any driver)", st, e);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &rf);

   /* The copy_image path bgfx takes when the extension exists. */
   CopyImage(dt, GL_TEXTURE_2D, 0, 0, 0, 0, dst, GL_TEXTURE_2D, 0, 0, 0, 0, W, W, 1);
   e = glGetError();
   float *o = malloc(W * W * 16);
   read_tex_f(p_sample_depth, dst, W, W, o);
   int bad = 0;
   for (int i = 0; i < W * W; i++) bad += o[i * 4] != dz[i];
   CHECK(e == 0 && bad == 0, "mc: glCopyImageSubDataEXT depth32f error 0x%x, %d wrong", e, bad);
   free(o);
}

/* ------------------------------------------------------------------ clip */

static const float strip_z[8] = {-1.5f, -1.02f, -0.75f, -0.25f, 0.25f, 0.75f, 0.98f, 1.5f};

static void
test_clip_one(GLenum dfmt, GLenum origin_depth, int clamp, float n, float f)
{
   const int W = 64, H = 16;
   GLuint ct = tex2d(GL_RGBA8, W, H, 1), dt = tex2d(dfmt, W, H, 1);
   GLuint fb = fbo_color(ct, 0);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dt, 0);
   glViewport(0, 0, W, H);
   glClearColor(0, 0, 0, 0); glClearDepthf(0.5f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   ClipControl(GL_LOWER_LEFT_EXT, origin_depth);
   if (clamp) glEnable(GL_DEPTH_CLAMP_EXT); else glDisable(GL_DEPTH_CLAMP_EXT);
   glDepthRangef(n, f);
   glEnable(GL_DEPTH_TEST); glDepthFunc(GL_ALWAYS); glDepthMask(GL_TRUE);
   for (int i = 0; i < 8; i++)
      quad(p_strip, -1 + i * 0.25f, -1, -1 + (i + 1) * 0.25f, 1, strip_z[i]);
   unsigned char px[64 * 16 * 4];
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   ClipControl(GL_LOWER_LEFT_EXT, GL_NEGATIVE_ONE_TO_ONE_EXT);
   glDisable(GL_DEPTH_CLAMP_EXT); glDepthRangef(0, 1); glDisable(GL_DEPTH_TEST);
   float *dz = malloc(W * H * 16);
   read_tex_f(p_sample_depth, dt, W, H, dz);

   const int halfz = origin_depth == GL_ZERO_TO_ONE_EXT;
   const char *fn = dfmt == GL_DEPTH_COMPONENT32F ? "d32f" : "d24";
   for (int i = 0; i < 8; i++) {
      const float z = strip_z[i];
      const int visible = clamp || (halfz ? (z >= 0 && z <= 1) : (z >= -1 && z <= 1));
      float d = halfz ? n + (f - n) * z : n + (f - n) * (z + 1) / 2;
      if (clamp) {
         const float lo = fminf(n, f), hi = fmaxf(n, f);
         d = fminf(fmaxf(d, lo), hi);
      }
      d = fminf(fmaxf(d, 0), 1);
      const int x = i * 8 + 4, y = 8;
      const int lit = px[(y * W + x) * 4 + 0] == 255;
      const float got = dz[(y * W + x) * 4];
      const float want = visible ? d : 0.5f;
      CHECK(lit == visible && fabsf(got - want) < 2e-5f,
            "clip %s %s clamp=%d range(%.2f,%.2f) z=%.2f: lit %d (want %d) depth %.6f (want %.6f)",
            fn, halfz ? "ZERO_TO_ONE" : "NEG_ONE_TO_ONE", clamp, n, f, z, lit, visible, got, want);
   }
   free(dz);
   glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &ct); glDeleteTextures(1, &dt);
}

static void
test_clip(void)
{
   const GLenum dfmt[2] = {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT24};
   const float ranges[3][2] = {{0, 1}, {0.25f, 0.75f}, {0.9f, 0.1f}};
   for (int d = 0; d < 2; d++)
      for (int h = 0; h < 2; h++)
         for (int c = 0; c < 2; c++)
            for (int r = 0; r < 3; r++)
               test_clip_one(dfmt[d], h ? GL_ZERO_TO_ONE_EXT : GL_NEGATIVE_ONE_TO_ONE_EXT, c,
                             ranges[r][0], ranges[r][1]);
}

static void
test_fragz(void)
{
   const GLenum dfmt[2] = {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT24};
   const float fz[2] = {1.5f, -0.5f};
   for (int d = 0; d < 2; d++)
      for (int c = 0; c < 2; c++)
         for (int k = 0; k < 2; k++) {
            const int W = 16;
            GLuint ct = tex2d(GL_RGBA8, W, W, 1), dt = tex2d(dfmt[d], W, W, 1);
            GLuint fb = fbo_color(ct, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dt, 0);
            glViewport(0, 0, W, W);
            glClearDepthf(0.5f); glClear(GL_DEPTH_BUFFER_BIT);
            if (c) glEnable(GL_DEPTH_CLAMP_EXT);
            glEnable(GL_DEPTH_TEST); glDepthFunc(GL_ALWAYS);
            glUseProgram(p_fragz);
            glUniform1f(glGetUniformLocation(p_fragz, "fz"), fz[k]);
            quad(p_fragz, -1, -1, 1, 1, 0);
            glDisable(GL_DEPTH_CLAMP_EXT); glDisable(GL_DEPTH_TEST);
            float o[16 * 16 * 4];
            read_tex_f(p_sample_depth, dt, W, W, o);
            const float want = fz[k] > 1 ? 1.0f : 0.0f;
            CHECK(fabsf(o[(8 * W + 8) * 4] - want) < 1e-6f,
                  "fragz %s clamp=%d gl_FragDepth=%.1f stored %.6f (want %.1f)",
                  d ? "d24" : "d32f", c, fz[k], o[(8 * W + 8) * 4], want);
            glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &ct); glDeleteTextures(1, &dt);
         }
}

/* ------------------------------------------------------------------ copy */

static unsigned char
pat(int x, int y, int l, int c)
{
   return (unsigned char)(x * 7 + y * 13 + l * 31 + c * 59 + 1);
}

static void
test_copy_rgba8(void)
{
   GLuint a = tex2d(GL_RGBA8, 64, 64, 3), b = tex2d(GL_RGBA8, 64, 64, 1);
   static unsigned char src[64 * 64 * 4], dst0[64 * 64 * 4], out[64 * 64 * 4];
   for (int l = 0; l < 3; l++) {
      const int s = 64 >> l;
      for (int y = 0; y < s; y++)
         for (int x = 0; x < s; x++)
            for (int c = 0; c < 4; c++)
               src[(y * s + x) * 4 + c] = pat(x, y, l, c);
      glBindTexture(GL_TEXTURE_2D, a);
      glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, s, s, GL_RGBA, GL_UNSIGNED_BYTE, src);
   }
   memset(dst0, 0x5a, sizeof dst0);
   glBindTexture(GL_TEXTURE_2D, b);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, dst0);
   CopyImage(a, GL_TEXTURE_2D, 1, 3, 5, 0, b, GL_TEXTURE_2D, 0, 10, 7, 0, 17, 11, 1);
   GLenum e = glGetError();
   read_rgba8(b, 0, 64, 64, out);
   int bad = 0;
   for (int y = 0; y < 64; y++)
      for (int x = 0; x < 64; x++)
         for (int c = 0; c < 4; c++) {
            const int in = x >= 10 && x < 27 && y >= 7 && y < 18;
            const unsigned char want = in ? pat(x - 10 + 3, y - 7 + 5, 1, c) : 0x5a;
            bad += out[(y * 64 + x) * 4 + c] != want;
         }
   CHECK(e == 0 && bad == 0, "copy rgba8 level1 (3,5)->level0 (10,7) 17x11: err 0x%x, %d wrong", e, bad);
   glDeleteTextures(1, &a); glDeleteTextures(1, &b);
}

static void
test_copy_depth(GLenum ifmt, GLenum type, const char *name)
{
   const int W = 48, H = 40;
   GLuint a = tex2d(ifmt, W, H, 1), b = tex2d(ifmt, W, H, 1);
   static uint32_t da[48 * 40 * 2], db[48 * 40 * 2];
   float want_d[48 * 40], base_d[48 * 40];
   unsigned want_s[48 * 40], base_s[48 * 40];
   const int has_s = type != GL_FLOAT && type != GL_UNSIGNED_INT;
   for (int i = 0; i < W * H; i++) {
      const float d1 = (float)((i * 2654435761u) % 4096) / 4096.0f, d2 = 0.125f;
      const unsigned s1 = (i * 37 + 11) & 0xff, s2 = 0x3c;
      want_d[i] = d1; base_d[i] = d2; want_s[i] = s1; base_s[i] = s2;
      if (type == GL_FLOAT) { memcpy(&da[i], &d1, 4); memcpy(&db[i], &d2, 4); }
      else if (type == GL_UNSIGNED_INT) { da[i] = (uint32_t)(d1 * 4294967295.0); db[i] = (uint32_t)(d2 * 4294967295.0); }
      else if (type == GL_UNSIGNED_INT_24_8) {
         da[i] = ((uint32_t)(d1 * 16777215.0) << 8) | s1;
         db[i] = ((uint32_t)(d2 * 16777215.0) << 8) | s2;
      } else { /* FLOAT_32_UNSIGNED_INT_24_8_REV */
         memcpy(&da[2 * i], &d1, 4); da[2 * i + 1] = s1;
         memcpy(&db[2 * i], &d2, 4); db[2 * i + 1] = s2;
      }
   }
   const GLenum fmt = has_s ? GL_DEPTH_STENCIL : GL_DEPTH_COMPONENT;
   glBindTexture(GL_TEXTURE_2D, a); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, fmt, type, da);
   glBindTexture(GL_TEXTURE_2D, b); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, fmt, type, db);
   const int sx = 5, sy = 3, dx = 9, dy = 14, cw = 31, ch = 19;
   CopyImage(a, GL_TEXTURE_2D, 0, sx, sy, 0, b, GL_TEXTURE_2D, 0, dx, dy, 0, cw, ch, 1);
   GLenum e = glGetError();
   float *o = malloc(W * H * 16);
   read_tex_f(p_sample_depth, b, W, H, o);
   int bad = 0, sbad = 0;
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         const int in = x >= dx && x < dx + cw && y >= dy && y < dy + ch;
         const float want = in ? want_d[(y - dy + sy) * W + (x - dx + sx)] : base_d[y * W + x];
         bad += fabsf(o[(y * W + x) * 4] - want) > 1e-6f;
      }
   if (has_s) {
      glBindTexture(GL_TEXTURE_2D, b);
      glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX);
      read_tex_f(p_sample_stencil, b, W, H, o);
      for (int y = 0; y < H; y++)
         for (int x = 0; x < W; x++) {
            const int in = x >= dx && x < dx + cw && y >= dy && y < dy + ch;
            const unsigned want = in ? want_s[(y - dy + sy) * W + (x - dx + sx)] : base_s[y * W + x];
            sbad += (unsigned)o[(y * W + x) * 4] != want;
         }
   }
   CHECK(e == 0 && bad == 0 && sbad == 0, "copy %s sub-rect: err 0x%x, %d depth / %d stencil wrong",
         name, e, bad, sbad);
   free(o);
   glDeleteTextures(1, &a); glDeleteTextures(1, &b);
}

static void
test_copy_array_3d_cube(void)
{
   const int S = 16;
   static unsigned char buf[16 * 16 * 4 * 6], out[16 * 16 * 4];
   GLuint arr; glGenTextures(1, &arr); glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
   glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, S, S, 4);
   for (int l = 0; l < 4; l++)
      for (int i = 0; i < S * S * 4; i++) buf[l * S * S * 4 + i] = pat(i / 4 % S, i / 4 / S, l, i % 4);
   glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, S, S, 4, GL_RGBA, GL_UNSIGNED_BYTE, buf);
   GLuint t3; glGenTextures(1, &t3); glBindTexture(GL_TEXTURE_3D, t3);
   glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, S, S, 3);
   memset(buf, 0, sizeof buf);
   glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, S, S, 3, GL_RGBA, GL_UNSIGNED_BYTE, buf);
   /* layers 1..2 -> slices 1..2, offset (2,4) */
   CopyImage(arr, GL_TEXTURE_2D_ARRAY, 0, 0, 0, 1, t3, GL_TEXTURE_3D, 0, 2, 4, 1, 12, 10, 2);
   GLenum e = glGetError();
   int bad = 0;
   GLuint f; glGenFramebuffers(1, &f); glBindFramebuffer(GL_FRAMEBUFFER, f);
   for (int z = 0; z < 3; z++) {
      glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t3, 0, z);
      glReadPixels(0, 0, S, S, GL_RGBA, GL_UNSIGNED_BYTE, out);
      for (int y = 0; y < S; y++)
         for (int x = 0; x < S; x++)
            for (int c = 0; c < 4; c++) {
               const int in = z >= 1 && x >= 2 && x < 14 && y >= 4 && y < 14;
               bad += out[(y * S + x) * 4 + c] != (in ? pat(x - 2, y - 4, z, c) : 0);
            }
   }
   CHECK(e == 0 && bad == 0, "copy 2D array layers 1-2 -> 3D slices 1-2: err 0x%x, %d wrong", e, bad);

   /* 3D slice 2 -> cube face -Y (layer 3), then cube face -> 2D. */
   GLuint cube; glGenTextures(1, &cube); glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
   glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA8, S, S);
   CopyImage(t3, GL_TEXTURE_3D, 0, 0, 0, 2, cube, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 3, S, S, 1);
   GLuint t2 = tex2d(GL_RGBA8, S, S, 1);
   CopyImage(cube, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 3, t2, GL_TEXTURE_2D, 0, 0, 0, 0, S, S, 1);
   e = glGetError();
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t2, 0);
   glReadPixels(0, 0, S, S, GL_RGBA, GL_UNSIGNED_BYTE, out);
   bad = 0;
   for (int y = 0; y < S; y++)
      for (int x = 0; x < S; x++)
         for (int c = 0; c < 4; c++) {
            const int in = x >= 2 && x < 14 && y >= 4 && y < 14;
            bad += out[(y * S + x) * 4 + c] != (in ? pat(x - 2, y - 4, 2, c) : 0);
         }
   CHECK(e == 0 && bad == 0, "copy 3D slice -> cube face -> 2D: err 0x%x, %d wrong", e, bad);
   glDeleteFramebuffers(1, &f);
   GLuint ts[4] = {arr, t3, cube, t2}; glDeleteTextures(4, ts);
}

/* 8-byte compressed blocks <-> RGBA16UI texels (one block = one texel). */
static void
test_copy_compressed(GLenum cfmt, const char *name)
{
   const int W = 32, H = 32, BW = W / 4, BH = H / 4;
   static uint16_t blocks[8 * 8 * 4], out[8 * 8 * 4 * 2];
   for (int i = 0; i < BW * BH * 4; i++) blocks[i] = (uint16_t)(i * 2654435761u >> 7);
   GLuint c = tex2d(cfmt, W, H, 1);
   glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, cfmt, BW * BH * 8, blocks);
   GLuint u = tex2d(GL_RGBA16UI, BW, BH, 1);
   static const uint16_t zero[8 * 8 * 4];
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, BW, BH, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, zero);
   /* compressed (8,4) 16x12 px = 4x3 blocks -> plain (1,2) */
   CopyImage(c, GL_TEXTURE_2D, 0, 8, 4, 0, u, GL_TEXTURE_2D, 0, 1, 2, 0, 16, 12, 1);
   /* round trip: plain -> another compressed texture -> plain */
   GLuint c2 = tex2d(cfmt, W, H, 1), u2 = tex2d(GL_RGBA16UI, BW, BH, 1);
   CopyImage(u, GL_TEXTURE_2D, 0, 0, 0, 0, c2, GL_TEXTURE_2D, 0, 0, 0, 0, BW, BH, 1);
   CopyImage(c2, GL_TEXTURE_2D, 0, 0, 0, 0, u2, GL_TEXTURE_2D, 0, 0, 0, 0, W, H, 1);
   GLenum e = glGetError();
   int bad = 0;
   GLuint tex[2] = {u, u2};
   for (int t = 0; t < 2; t++) {
      GLuint f = fbo_color(tex[t], 0);
      static GLuint o32[8 * 8 * 4];
      glReadPixels(0, 0, BW, BH, GL_RGBA_INTEGER, GL_UNSIGNED_INT, o32);
      glDeleteFramebuffers(1, &f);
      for (int y = 0; y < BH; y++)
         for (int x = 0; x < BW; x++)
            for (int k = 0; k < 4; k++) {
               const int in = x >= 1 && x < 5 && y >= 2 && y < 5;
               const unsigned want = in ? blocks[((y - 2 + 1) * BW + (x - 1 + 2)) * 4 + k] : 0;
               bad += o32[(y * BW + x) * 4 + k] != want;
            }
   }
   (void)out;
   CHECK(e == 0 && bad == 0, "copy %s <-> RGBA16UI (blocks, round trip): err 0x%x, %d wrong", name, e, bad);
   GLuint ts[4] = {c, u, c2, u2}; glDeleteTextures(4, ts);
}

/* Render into A, copy A->B, sample B into C; 64 rounds without any flush or
 * barrier call: GL orders all of it. */
static void
test_copy_ordering(void)
{
   const int S = 32;
   GLuint a = tex2d(GL_RGBA8, S, S, 1), b = tex2d(GL_RGBA8, S, S, 1), c = tex2d(GL_RGBA8, S, S, 1);
   GLuint fa = fbo_color(a, 0), fc = fbo_color(c, 0);
   int bad_rounds = 0;
   for (int r = 0; r < 64; r++) {
      const float v = (float)((r * 37) % 251 + 1) / 255.0f;
      glBindFramebuffer(GL_FRAMEBUFFER, fa); glViewport(0, 0, S, S);
      glClearColor(v, 1 - v, v * 0.5f, 1); glClear(GL_COLOR_BUFFER_BIT);
      glUseProgram(p_strip); quad(p_strip, -0.5f, -0.5f, 0.5f, 0.5f, 0); /* white centre */
      CopyImage(a, GL_TEXTURE_2D, 0, 0, 0, 0, b, GL_TEXTURE_2D, 0, 0, 0, 0, S, S, 1);
      glBindFramebuffer(GL_FRAMEBUFFER, fc);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, b);
      glUseProgram(p_sample_color); glUniform1i(glGetUniformLocation(p_sample_color, "s"), 0);
      quad(p_sample_color, -1, -1, 1, 1, 0);
      unsigned char px[2][4];
      glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px[0]);
      glReadPixels(16, 16, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px[1]);
      const int want = (int)lrintf(v * 255);
      if (abs(px[0][0] - want) > 1 || px[1][0] != 255 || px[1][1] != 255)
         bad_rounds++;
   }
   CHECK(bad_rounds == 0, "copy ordering render->copy->sample x64: %d bad rounds", bad_rounds);
   glDeleteFramebuffers(1, &fa); glDeleteFramebuffers(1, &fc);
   GLuint ts[3] = {a, b, c}; glDeleteTextures(3, ts);
}

static void
test_copy_compute(void)
{
   const int S = 32;
   GLuint a = tex2d(GL_RGBA8, S, S, 1), b = tex2d(GL_RGBA8, S, S, 1);
   GLuint cp = cprog("#version 310 es\nlayout(local_size_x=8,local_size_y=8) in;\n"
                     "layout(rgba8, binding=0) writeonly uniform highp image2D img;\nuniform float k;\n"
                     "void main(){ ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
                     " imageStore(img, p, vec4(float(p.x)/31.0, float(p.y)/31.0, k, 1.0)); }\n");
   int bad = 0;
   for (int r = 0; r < 8; r++) {
      const float k = (float)(r * 30 + 10) / 255.0f;
      glUseProgram(cp); glUniform1f(glGetUniformLocation(cp, "k"), k);
      glBindImageTexture(0, a, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
      glDispatchCompute(S / 8, S / 8, 1);
      glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
      CopyImage(a, GL_TEXTURE_2D, 0, 0, 0, 0, b, GL_TEXTURE_2D, 0, 0, 0, 0, S, S, 1);
      unsigned char px[32 * 32 * 4];
      read_rgba8(b, 0, S, S, px);
      for (int y = 0; y < S; y++)
         for (int x = 0; x < S; x++) {
            const unsigned char *q = &px[(y * S + x) * 4];
            bad += abs(q[0] - (int)lrintf(x / 31.0f * 255)) > 1 || abs(q[1] - (int)lrintf(y / 31.0f * 255)) > 1 ||
                   abs(q[2] - (int)lrintf(k * 255)) > 1;
         }
   }
   CHECK(bad == 0, "compute imageStore -> barrier -> copy -> read x8: %d wrong", bad);
   glDeleteProgram(cp);
   GLuint ts[2] = {a, b}; glDeleteTextures(2, ts);
}

static void
test_copy_buffer(void)
{
   const int N = 300000;
   unsigned char *d = malloc(N + 64), *o;
   for (int i = 0; i < N + 64; i++) d[i] = (unsigned char)(i * 131 + 7);
   GLuint b[2]; glGenBuffers(2, b);
   glBindBuffer(GL_COPY_READ_BUFFER, b[0]); glBufferData(GL_COPY_READ_BUFFER, N + 64, d, GL_STATIC_DRAW);
   glBindBuffer(GL_COPY_WRITE_BUFFER, b[1]); glBufferData(GL_COPY_WRITE_BUFFER, N + 64, NULL, GL_STATIC_DRAW);
   unsigned char *z = calloc(1, N + 64);
   glBufferSubData(GL_COPY_WRITE_BUFFER, 0, N + 64, z);
   glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 13, 40, N);
   o = glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, N + 64, GL_MAP_READ_BIT);
   int bad = 0;
   for (int i = 0; i < N + 64; i++)
      bad += o[i] != ((i >= 40 && i < 40 + N) ? d[i - 40 + 13] : 0);
   glUnmapBuffer(GL_COPY_WRITE_BUFFER);
   CHECK(glGetError() == 0 && bad == 0, "glCopyBufferSubData 300000 B (13 -> 40): %d wrong", bad);
   glDeleteBuffers(2, b); free(d); free(z);
}

int
main(int argc, char **argv)
{
   only = argc > 1 ? argv[1] : NULL;
   PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   eglInitialize(d, NULL, NULL);
   eglBindAPI(EGL_OPENGL_ES_API);
   EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
   if (!c || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) { printf("FAIL: context\n"); return 1; }
   printf("%s / %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

   const char *need[] = {"GL_EXT_clip_control", "GL_EXT_depth_clamp", "GL_OES_copy_image",
                         "GL_EXT_copy_image", "GL_EXT_color_buffer_float"};
   int have[5] = {0};
   GLint n; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
   for (int i = 0; i < n; i++)
      for (int k = 0; k < 5; k++)
         have[k] |= !strcmp((const char *)glGetStringi(GL_EXTENSIONS, i), need[k]);
   for (int k = 0; k < 5; k++)
      CHECK(have[k], "ext %s", need[k]);
   ClipControl = (clip_control_fn)eglGetProcAddress("glClipControlEXT");
   CopyImage = (copy_image_fn)eglGetProcAddress("glCopyImageSubDataEXT");
   if (!have[0] || !have[1] || !have[3] || !have[4] || !ClipControl || !CopyImage) {
      printf("FAIL (%d failures): required extensions missing\n", failures + 1);
      return 1;
   }

   p_strip = prog(VS_QUAD, "#version 310 es\nprecision highp float;\nout vec4 c;\nvoid main(){c=vec4(1);}\n");
   p_fragz = prog(VS_QUAD, "#version 310 es\nprecision highp float;\nuniform float fz;\nout vec4 c;\n"
                           "void main(){c=vec4(1); gl_FragDepth = fz;}\n");
   p_sample_depth = prog(VS_QUAD, "#version 310 es\nprecision highp float;\nuniform highp sampler2D s;\n"
                                  "out vec4 c;\nvoid main(){c=vec4(texelFetch(s, ivec2(gl_FragCoord.xy), 0).r);}\n");
   p_sample_stencil = prog(VS_QUAD, "#version 310 es\nprecision highp float;\nuniform highp usampler2D s;\n"
                                    "out vec4 c;\nvoid main(){c=vec4(float(texelFetch(s, ivec2(gl_FragCoord.xy), 0).r));}\n");
   p_sample_color = prog(VS_QUAD, "#version 310 es\nprecision highp float;\nuniform highp sampler2D s;\n"
                                  "out vec4 c;\nvoid main(){c=texelFetch(s, ivec2(gl_FragCoord.xy), 0);}\n");
   (void)p_sample_3d;
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);

   if (SECTION("mc")) test_mc();
   if (SECTION("clip")) test_clip();
   if (SECTION("fragz")) test_fragz();
   if (SECTION("copy")) {
      test_copy_rgba8();
      test_copy_depth(GL_DEPTH_COMPONENT32F, GL_FLOAT, "d32f");
      test_copy_depth(GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT, "d24");
      test_copy_depth(GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8, "d24s8");
      test_copy_depth(GL_DEPTH32F_STENCIL8, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, "d32fs8");
      test_copy_array_3d_cube();
      test_copy_compressed(GL_COMPRESSED_RED_RGTC1_EXT, "rgtc1");
      test_copy_compressed(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, "bc1");
      test_copy_ordering();
      test_copy_compute();
      test_copy_buffer();
   }
   GLenum e = glGetError();
   CHECK(e == 0, "final glGetError 0x%x", e);
   printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
   return failures != 0;
}
