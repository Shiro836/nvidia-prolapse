/* SPDX-License-Identifier: MIT
 *
 * gl-texpool-rebind: texture fetches of a batch that is still executing when
 * the NEXT batch on the channel rebinds the texture header / sampler pool
 * (GTA SA flicker, 2026-10-16: whole final pass of a frame sampled zeros).
 *
 * nvrm gives every submission batch its own TIC/TSC pool (a 4-slot ring,
 * nvrm_texture.c) and binds it with SET_TEX_HEADER_POOL at the batch's first
 * draw. Stock GL and NVK never change the pool contents a running draw can
 * see: stock binds its driver pools once per channel, NVK rebinds only a grown copy of the same table
 * (nvk_descriptor_table.c, nvk_queue.c:83-128). This test makes the race
 * certain instead of waiting for it:
 *
 *   batch 1  slow full-screen draw into FBO A sampling texture RED (pool of
 *            ring slot k, header slot 0); the sample is taken after a long
 *            ALU loop so every warp fetches the header late; glFlush.
 *   batch 2  immediately: a draw into FBO B sampling texture GREEN (ring slot
 *            k+1: header slot 0 = GREEN); glFlush; glFinish.
 *
 * If batch 2's pool binding reaches the texture units while batch 1's draw is
 * still running, part of A reads GREEN (or whatever the new pool holds).
 * PASS = every pixel of A is RED in every round, and batch 1 alone takes long
 * enough that the overlap really happened (printed).
 *
 * Variant "zero": batch 2 draws without textures and a fresh context makes
 * ring slot k+1's pool all zeros (never written), the GTA signature (texture
 * fetches return 0).
 *
 * Variant "wrap": batch 2 is followed by more one-draw GREEN batches, so the
 * ring comes back to batch 1's slot while batch 1 still runs: the flush that
 * rotates onto it must wait for batch 1 before slot 0 is rewritten GREEN.
 *
 * Variant "ctx2": batch 2 comes from a second (shared) context on the same
 * screen channel, which binds its own pools and handle cbuf.
 *
 * Variant "free": right after batch 1 is submitted, RED is deleted (sampler
 * view, resource and memory released while batch 1 still samples it) and 32
 * new GREEN textures are created and drawn in batch 2, so freed memory/VA or
 * pool slots reused too early would show GREEN (or zeros) in A.
 *
 * Build as gl-g0a-clear.c (README); run:
 * ./gl-texpool-rebind [rounds] [iters] [zero|wrap|ctx2|free]
 */
#define _POSIX_C_SOURCE 200809L
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W 2048
#define H 2048

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static GLuint
shader(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "compile: %s\n", log);
      exit(1);
   }
   return s;
}

static GLuint
program(const char *vs, const char *fs)
{
   GLuint p = glCreateProgram();
   glAttachShader(p, shader(GL_VERTEX_SHADER, vs));
   glAttachShader(p, shader(GL_FRAGMENT_SHADER, fs));
   glBindAttribLocation(p, 0, "pos");
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      fprintf(stderr, "link failed\n");
      exit(1);
   }
   return p;
}

static const char *vs_src =
   "#version 300 es\n"
   "in vec2 pos;\n"
   "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";

/* Long dependent ALU loop, then one texture fetch whose coordinate depends
 * on the loop (the fetch cannot be hoisted above it). k = 0 at run time. */
static const char *fs_slow_src =
   "#version 300 es\n"
   "precision highp float;\n"
   "uniform sampler2D tex;\n"
   "uniform int iters;\n"
   "uniform float k;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  float a = gl_FragCoord.x * 0.001 + gl_FragCoord.y * 0.002;\n"
   "  for (int i = 0; i < iters; i++)\n"
   "    a = fract(a * 1.37 + 0.11);\n"
   "  o = texture(tex, vec2(0.5) + vec2(a * k));\n"
   "}\n";

static const char *fs_tex_src =
   "#version 300 es\n"
   "precision highp float;\n"
   "uniform sampler2D tex;\n"
   "out vec4 o;\n"
   "void main() { o = texture(tex, vec2(0.5)); }\n";

static const char *fs_flat_src =
   "#version 300 es\n"
   "precision highp float;\n"
   "out vec4 o;\n"
   "void main() { o = vec4(0.0, 0.0, 1.0, 1.0); }\n";

static GLuint
solid_texture(uint32_t rgba)
{
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_2D, t);
   uint32_t texels[16];
   for (int i = 0; i < 16; i++)
      texels[i] = rgba;
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   return t;
}

static GLuint
target(GLsizei w, GLsizei h)
{
   GLuint rt, fb;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete FBO\n");
      exit(1);
   }
   return fb;
}

int
main(int argc, char **argv)
{
   const int rounds = argc > 1 ? atoi(argv[1]) : 8;
   const int iters = argc > 2 ? atoi(argv[2]) : 4000;
   const int zero = argc > 3 && !strcmp(argv[3], "zero");
   const int wrap = argc > 3 && !strcmp(argv[3], "wrap");
   const int ctx2_mode = argc > 3 && !strcmp(argv[3], "ctx2");
   const int free_mode = argc > 3 && !strcmp(argv[3], "free");

   EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) {
      fprintf(stderr, "eglInitialize failed\n");
      return 1;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint ctx_attr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      fprintf(stderr, "no ES 3.0 context\n");
      return 1;
   }
   printf("renderer: %s\n", glGetString(GL_RENDERER));

   GLuint red = solid_texture(0xff0000ffu);   /* R=255 A=255 */
   const GLuint green = solid_texture(0xff00ff00u); /* G=255 A=255 */
   const GLuint fb_a = target(W, H);
   const GLuint fb_b = target(16, 16);
   const GLuint p_slow = program(vs_src, fs_slow_src);
   const GLuint p_tex = program(vs_src, fs_tex_src);
   const GLuint p_flat = program(vs_src, fs_flat_src);
   glUseProgram(p_slow);
   glUniform1i(glGetUniformLocation(p_slow, "tex"), 0);
   glUniform1i(glGetUniformLocation(p_slow, "iters"), iters);
   glUniform1f(glGetUniformLocation(p_slow, "k"), 0.0f);
   glUseProgram(p_tex);
   glUniform1i(glGetUniformLocation(p_tex, "tex"), 0);

   static const float quad[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
   GLuint vb;
   glGenBuffers(1, &vb);
   glBindBuffer(GL_ARRAY_BUFFER, vb);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glActiveTexture(GL_TEXTURE0);

   /* Second context sharing textures with the first (variant ctx2). */
   EGLContext ctx2 = EGL_NO_CONTEXT;
   GLuint fb_b2 = 0, p_tex2 = 0;
   if (ctx2_mode) {
      ctx2 = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, ctx, ctx_attr);
      if (ctx2 == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx2)) {
         fprintf(stderr, "no second context\n");
         return 1;
      }
      fb_b2 = target(16, 16);
      p_tex2 = program(vs_src, fs_tex_src);
      glUseProgram(p_tex2);
      glUniform1i(glGetUniformLocation(p_tex2, "tex"), 0);
      GLuint vb2;
      glGenBuffers(1, &vb2);
      glBindBuffer(GL_ARRAY_BUFFER, vb2);
      glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glEnableVertexAttribArray(0);
      glActiveTexture(GL_TEXTURE0);
      glFinish();
      eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
   }
   glFinish();

   /* Batch 1 alone: how long the slow draw keeps the GPU busy. */
   double t0 = now_ms();
   glBindFramebuffer(GL_FRAMEBUFFER, fb_a);
   glViewport(0, 0, W, H);
   glUseProgram(p_slow);
   glBindTexture(GL_TEXTURE_2D, red);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glFinish();
   printf("slow batch alone: %.2f ms (iters %d, %dx%d)\n", now_ms() - t0, iters, W, H);

   uint32_t *px = malloc((size_t)W * H * 4);
   long total_bad = 0;
   for (int r = 0; r < rounds; r++) {
      glBindFramebuffer(GL_FRAMEBUFFER, fb_a);
      glViewport(0, 0, W, H);
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);
      glUseProgram(p_slow);
      glBindTexture(GL_TEXTURE_2D, red);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFlush(); /* batch 1 submitted, still running */
      const double t_sub = now_ms();
      GLuint fresh[32];
      if (free_mode) {
         glDeleteTextures(1, &red); /* batch 1 still samples it */
         glBindFramebuffer(GL_FRAMEBUFFER, fb_b);
         glViewport(0, 0, 16, 16);
         glUseProgram(p_tex);
         for (int i = 0; i < 32; i++) {
            fresh[i] = solid_texture(0xff00ff00u);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
         }
         glFlush();
      }
      if (ctx2_mode)
         eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx2);
      glBindFramebuffer(GL_FRAMEBUFFER, ctx2_mode ? fb_b2 : fb_b);
      glViewport(0, 0, 16, 16);
      if (ctx2_mode) {
         glUseProgram(p_tex2);
         glBindTexture(GL_TEXTURE_2D, green);
      } else if (zero) {
         glUseProgram(p_flat);
      } else {
         glUseProgram(p_tex);
         glBindTexture(GL_TEXTURE_2D, green);
      }
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFlush(); /* batch 2: its first draw binds the next ring slot's pools */
      for (int b = 0; wrap && b < 6; b++) {
         glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
         glFlush(); /* batches 3..8: the ring passes batch 1's slot */
      }
      const double t_b2 = now_ms() - t_sub;
      glFinish();
      if (ctx2_mode)
         eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
      if (free_mode) {
         glDeleteTextures(32, fresh);
         red = solid_texture(0xff0000ffu);
      }
      glBindFramebuffer(GL_FRAMEBUFFER, fb_a);
      glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
      long bad = 0;
      uint32_t first_bad = 0;
      for (long i = 0; i < (long)W * H; i++) {
         if (px[i] != 0xff0000ffu) {
            if (!bad)
               first_bad = px[i];
            bad++;
         }
      }
      printf("round %d: batch 2 submitted %.3f ms after batch 1; %ld of %d pixels not RED "
             "(first 0x%08x)\n", r, t_b2, bad, W * H, first_bad);
      total_bad += bad;
      /* Fresh ring position per round for the "zero" variant is not
       * guaranteed after round 0: later rounds see pools written before. */
   }
   free(px);
   printf("%s: %ld wrong pixels over %d rounds\n", total_bad ? "FAIL" : "PASS", total_bad,
          rounds);
   return total_bad ? 1 : 0;
}
