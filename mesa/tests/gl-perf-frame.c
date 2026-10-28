/* SPDX-License-Identifier: MIT
 *
 * gl-perf-frame: directional per-frame cost of nvrm's memory placement and
 * synchronisation (not a benchmark: another GPU client changes the numbers;
 * measure FPS on an idle GPU).
 *
 * One frame = DRAWS draws into a 1024x1024 RGBA8 FBO, each:
 *  - glBufferSubData of a 256 B per-draw UBO (the pattern that made every
 *    buffer map flush + wait while the UBO was still referenced),
 *  - a static VBO of VERTS vertices (position + colour),
 *  - a vertex shader indexing a 64 KiB UBO table and a fragment shader
 *    reading a 1 MiB SSBO and sampling a 1024x1024 texture
 *    (GPU fetch traffic: system memory vs VRAM placement).
 * glFinish ends every frame. Reported per frame: wall ms (GPU-bound here, so
 * ~ GPU time), process CPU ms, and wall - CPU = time the CPU spent waiting.
 * A checksum of the last frame guards against "fast because broken".
 *
 * Build as gl-es31-mc.c (README); run: ./gl-perf-frame [frames] [draws] [pipelined]
 * "pipelined": glFlush per frame and one glFinish at the end (like a game
 * that never waits per frame); reported per-frame numbers are averages over
 * the whole run.
 */
#define _POSIX_C_SOURCE 200809L
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W 1024
#define H 1024
#define VERTS (3 * 20000)

static double
now_ms(clockid_t c)
{
   struct timespec ts;
   clock_gettime(c, &ts);
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

static const char *vs_src =
   "#version 310 es\n"
   "layout(location=0) in vec2 pos;\n"
   "layout(location=1) in vec4 col;\n"
   "layout(std140, binding=0) uniform Table { vec4 t[4096]; };\n"
   "layout(std140, binding=1) uniform Draw { vec4 d[16]; };\n"
   "out vec4 vcol;\n"
   "void main() {\n"
   "  vec4 a = t[(gl_VertexID * 7) & 4095];\n"
   "  gl_Position = vec4(pos * d[0].xy + d[0].zw + a.xy * 0.0001, 0.0, 1.0);\n"
   "  vcol = col * d[1] + a * 0.001;\n"
   "}\n";

static const char *fs_src =
   "#version 310 es\n"
   "precision highp float;\n"
   "layout(std430, binding=0) readonly buffer S { vec4 s[]; };\n"
   "uniform sampler2D tex;\n"
   "in vec4 vcol;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  ivec2 p = ivec2(gl_FragCoord.xy);\n"
   "  vec4 v = s[(p.x * 31 + p.y * 17) & 65535];\n"
   "  o = vcol * 0.5 + texture(tex, gl_FragCoord.xy / 1024.0) * 0.25 + v * 0.25;\n"
   "}\n";

int
main(int argc, char **argv)
{
   const int frames = argc > 1 ? atoi(argv[1]) : 60;
   const int draws = argc > 2 ? atoi(argv[2]) : 200;
   const int pipelined = argc > 3 && !strcmp(argv[3], "pipelined");

   EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (dpy == EGL_NO_DISPLAY)
      dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!eglInitialize(dpy, NULL, NULL)) {
      fprintf(stderr, "eglInitialize failed\n");
      return 1;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint cfg_attr[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
   EGLConfig cfg;
   EGLint n;
   eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n);
   const EGLint ctx_attr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, n ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      fprintf(stderr, "no ES 3.1 context\n");
      return 1;
   }
   printf("renderer: %s\n", glGetString(GL_RENDERER));

   GLuint fb, rt;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);

   GLuint tex;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   uint32_t *texels = malloc(W * H * 4);
   for (int i = 0; i < W * H; i++)
      texels[i] = 0xff000000u | (i * 2654435761u >> 8);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, texels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   free(texels);

   float *v = malloc(VERTS * 6 * sizeof(float));
   for (int i = 0; i < VERTS; i++) {
      const float x = ((i * 7919) % 1000) / 500.0f - 1.0f;
      const float y = ((i * 104729) % 1000) / 500.0f - 1.0f;
      v[i * 6 + 0] = x;
      v[i * 6 + 1] = y;
      for (int c = 0; c < 4; c++)
         v[i * 6 + 2 + c] = ((i + c) % 5) / 5.0f;
   }
   GLuint vbo, vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, VERTS * 6 * sizeof(float), v, GL_STATIC_DRAW);
   free(v);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, (void *)0);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, (void *)8);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   GLuint ubo_table, ubo_draw, ssbo;
   float *tab = malloc(65536);
   for (int i = 0; i < 16384; i++)
      tab[i] = (i % 97) / 97.0f;
   glGenBuffers(1, &ubo_table);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo_table);
   glBufferData(GL_UNIFORM_BUFFER, 65536, tab, GL_STATIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo_table);
   glGenBuffers(1, &ubo_draw);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo_draw);
   glBufferData(GL_UNIFORM_BUFFER, 256, NULL, GL_DYNAMIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER, 1, ubo_draw);
   float *sb = malloc(1 << 20);
   for (int i = 0; i < (1 << 18); i++)
      sb[i] = (i % 89) / 89.0f;
   glGenBuffers(1, &ssbo);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
   glBufferData(GL_SHADER_STORAGE_BUFFER, 1 << 20, sb, GL_STATIC_DRAW);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
   free(tab);
   free(sb);

   GLuint prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER, vs_src));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs_src));
   glLinkProgram(prog);
   GLint ok;
   glGetProgramiv(prog, GL_LINK_STATUS, &ok);
   if (!ok) {
      fprintf(stderr, "link failed\n");
      return 1;
   }
   glUseProgram(prog);
   glUniform1i(glGetUniformLocation(prog, "tex"), 0);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   glViewport(0, 0, W, H);
   glEnable(GL_BLEND);
   glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

   double wall_sum = 0, cpu_sum = 0, wall_min = 1e9;
   const int warm = 3;
   double pw0 = 0, pc0 = 0;
   for (int f = 0; f < frames + warm; f++) {
      if (pipelined && f == warm) {
         glFinish();
         pw0 = now_ms(CLOCK_MONOTONIC);
         pc0 = now_ms(CLOCK_PROCESS_CPUTIME_ID);
      }
      const double w0 = now_ms(CLOCK_MONOTONIC), c0 = now_ms(CLOCK_PROCESS_CPUTIME_ID);
      glClearColor(0, 0, 0, 1);
      glClear(GL_COLOR_BUFFER_BIT);
      for (int d = 0; d < draws; d++) {
         float u[64] = { 0 };
         u[0] = 0.5f; u[1] = 0.5f;
         u[2] = ((d * 37) % 100) / 100.0f - 0.5f;
         u[3] = ((d * 53) % 100) / 100.0f - 0.5f;
         u[4] = u[5] = u[6] = 0.2f + (d % 7) / 10.0f;
         u[7] = 0.1f;
         glBindBuffer(GL_UNIFORM_BUFFER, ubo_draw);
         glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(u), u);
         glDrawArrays(GL_TRIANGLES, 0, VERTS / 8);
      }
      if (pipelined)
         glFlush();
      else
         glFinish();
      const double wall = now_ms(CLOCK_MONOTONIC) - w0;
      const double cpu = now_ms(CLOCK_PROCESS_CPUTIME_ID) - c0;
      if (f >= warm) {
         wall_sum += wall;
         cpu_sum += cpu;
         if (wall < wall_min)
            wall_min = wall;
      }
   }
   if (pipelined) {
      glFinish();
      wall_sum = now_ms(CLOCK_MONOTONIC) - pw0;
      cpu_sum = now_ms(CLOCK_PROCESS_CPUTIME_ID) - pc0;
      wall_min = 0;
   }
   uint8_t *px = malloc(W * H * 4);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   uint32_t h = 2166136261u;
   for (int i = 0; i < W * H * 4; i++)
      h = (h ^ px[i]) * 16777619u;
   free(px);
   printf("frames %d draws/frame %d: wall %.2f ms/frame (min %.2f), CPU %.2f ms/frame, "
          "CPU waiting %.2f ms/frame, checksum %08x, GL error 0x%x\n",
          frames, draws, wall_sum / frames, wall_min, cpu_sum / frames,
          (wall_sum - cpu_sum) / frames, h, glGetError());
   /* Screen destruction prints NVRM_DEBUG_STATS totals: run two frame
    * counts and difference them for per-frame submissions / CPU waits. */
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   return 0;
}
