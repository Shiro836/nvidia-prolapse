/* SPDX-License-Identifier: MIT
 *
 * gl-perf-mc: a host workload with Minecraft Bedrock's per-frame GL pattern
 * as captured on nvrm (~4.4k draws/frame,
 * a default-block uniform update and texture binds between most draws, UBO
 * range re-pointing, several FBO passes, a compute pass, a streamed texture
 * upload). It exists to count what the driver emits per frame
 * (NV_RM_CAPTURE + rm/tools/nvcap.py) and the syscalls it makes
 * (strace -c), not to time anything: directional only.
 *
 * One frame (DRAWS draws in pass A):
 *  - glTexSubImage2D 64x64 into texture 0 (streamed texel data);
 *  - pass A, 1024x1024 RGBA8 + D24S8 FBO: per draw glUniformMatrix4fv +
 *    glUniform4f (default block), a texture bind every 3rd draw (8 textures,
 *    2 sampler objects), glBindBufferRange of a 64 KiB UBO table every 16th
 *    draw, glBufferSubData of a busy vertex buffer every 100th draw,
 *    indexed draws of 36 indices;
 *  - pass B, 512x512: 64 draws sampling pass A's colour target;
 *  - compute: 16 work groups write a 4 KiB SSBO, glMemoryBarrier;
 *  - pass C, 1024x1024: 16 draws reading the SSBO and sampling pass B.
 * glFinish ends the frame (= one capture frame: the context fence).
 * Prints per-frame wall/CPU ms and an FNV-1a checksum of pass C's last frame.
 *
 * Build as gl-es31-mc.c (README); run: ./gl-perf-mc [frames] [draws]
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
#define NTEX 8

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

static GLuint
program(const char *vs, const char *fs, const char *cs)
{
   GLuint prog = glCreateProgram();
   if (cs) {
      glAttachShader(prog, shader(GL_COMPUTE_SHADER, cs));
   } else {
      glAttachShader(prog, shader(GL_VERTEX_SHADER, vs));
      glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs));
   }
   glLinkProgram(prog);
   GLint ok;
   glGetProgramiv(prog, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[2048];
      glGetProgramInfoLog(prog, sizeof(log), NULL, log);
      fprintf(stderr, "link: %s\n", log);
      exit(1);
   }
   return prog;
}

static const char *vs_a =
   "#version 310 es\n"
   "layout(location=0) in vec3 pos;\n"
   "layout(location=1) in vec2 uv;\n"
   "layout(std140, binding=0) uniform Frame { mat4 viewproj; vec4 fog; };\n"
   "layout(std140, binding=1) uniform Chunk { vec4 chunk[64]; };\n"
   "uniform mat4 u_model;\n"
   "out vec2 v_uv;\n"
   "out vec4 v_fog;\n"
   "void main() {\n"
   "  gl_Position = viewproj * u_model * vec4(pos + chunk[gl_VertexID & 63].xyz * 0.01, 1.0);\n"
   "  v_uv = uv;\n"
   "  v_fog = fog;\n"
   "}\n";

static const char *fs_a =
   "#version 310 es\n"
   "precision highp float;\n"
   "uniform sampler2D u_tex;\n"
   "uniform vec4 u_tint;\n"
   "in vec2 v_uv;\n"
   "in vec4 v_fog;\n"
   "out vec4 o;\n"
   "void main() { o = texture(u_tex, v_uv) * u_tint + v_fog * 0.125; }\n";

static const char *vs_post =
   "#version 310 es\n"
   "layout(location=0) in vec3 pos;\n"
   "layout(location=1) in vec2 uv;\n"
   "uniform vec4 u_rect;\n"
   "out vec2 v_uv;\n"
   "void main() {\n"
   "  gl_Position = vec4(pos.xy * u_rect.xy + u_rect.zw, 0.0, 1.0);\n"
   "  v_uv = uv;\n"
   "}\n";

static const char *fs_b =
   "#version 310 es\n"
   "precision highp float;\n"
   "uniform sampler2D u_src;\n"
   "uniform vec4 u_tint;\n"
   "in vec2 v_uv;\n"
   "out vec4 o;\n"
   "void main() { o = texture(u_src, v_uv) * u_tint; }\n";

static const char *fs_c =
   "#version 310 es\n"
   "precision highp float;\n"
   "layout(std430, binding=0) readonly buffer S { vec4 s[256]; };\n"
   "uniform sampler2D u_src;\n"
   "uniform vec4 u_tint;\n"
   "in vec2 v_uv;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  ivec2 p = ivec2(gl_FragCoord.xy);\n"
   "  o = texture(u_src, v_uv) * u_tint + s[(p.x + p.y * 7) & 255] * 0.25;\n"
   "}\n";

static const char *cs_src =
   "#version 310 es\n"
   "layout(local_size_x = 16) in;\n"
   "layout(std430, binding=0) buffer S { vec4 s[256]; };\n"
   "uniform float u_t;\n"
   "void main() {\n"
   "  uint i = gl_GlobalInvocationID.x;\n"
   "  s[i] = vec4(float(i & 7u) / 8.0, u_t, float(i) / 256.0, 1.0);\n"
   "}\n";

static GLuint
color_target(int w, int h)
{
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_2D, t);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   return t;
}

static GLuint
fbo(GLuint color, GLuint depth_rb)
{
   GLuint f;
   glGenFramebuffers(1, &f);
   glBindFramebuffer(GL_FRAMEBUFFER, f);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
   if (depth_rb)
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                                depth_rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete framebuffer\n");
      exit(1);
   }
   return f;
}

int
main(int argc, char **argv)
{
   const int frames = argc > 1 ? atoi(argv[1]) : 30;
   const int draws = argc > 2 ? atoi(argv[2]) : 1000;

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

   /* Render targets. */
   GLuint rt_a = color_target(W, H), rt_b = color_target(W / 2, H / 2), rt_c = color_target(W, H);
   GLuint ds;
   glGenRenderbuffers(1, &ds);
   glBindRenderbuffer(GL_RENDERBUFFER, ds);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
   GLuint fb_a = fbo(rt_a, ds), fb_b = fbo(rt_b, 0), fb_c = fbo(rt_c, 0);

   /* Sampled textures (mipmapped), two sampler objects. */
   GLuint tex[NTEX];
   glGenTextures(NTEX, tex);
   uint32_t *texels = malloc(256 * 256 * 4);
   for (int t = 0; t < NTEX; t++) {
      for (int i = 0; i < 256 * 256; i++)
         texels[i] = 0xff000000u | ((i * 2654435761u + t * 40503u) >> 8);
      glBindTexture(GL_TEXTURE_2D, tex[t]);
      glTexStorage2D(GL_TEXTURE_2D, 9, GL_RGBA8, 256, 256);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 256, GL_RGBA, GL_UNSIGNED_BYTE, texels);
      glGenerateMipmap(GL_TEXTURE_2D);
   }
   GLuint samp[2];
   glGenSamplers(2, samp);
   glSamplerParameteri(samp[0], GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
   glSamplerParameteri(samp[0], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glSamplerParameteri(samp[1], GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
   glSamplerParameteri(samp[1], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glSamplerParameteri(samp[1], GL_TEXTURE_WRAP_S, GL_REPEAT);

   /* Geometry: a cube (24 vertices, 36 indices) in a dynamic VBO. */
   float v[24 * 5];
   for (int i = 0; i < 24; i++) {
      const int face = i / 4, c = i % 4;
      const float a = (c & 1) ? 0.5f : -0.5f, b = (c & 2) ? 0.5f : -0.5f;
      const float s = (face & 1) ? 0.5f : -0.5f;
      float *o = &v[i * 5];
      o[0] = face < 2 ? s : a;
      o[1] = face < 2 ? a : face < 4 ? s : b;
      o[2] = face < 4 ? b : s;
      o[3] = (c & 1) ? 1.0f : 0.0f;
      o[4] = (c & 2) ? 1.0f : 0.0f;
   }
   uint16_t idx[36];
   for (int f = 0; f < 6; f++) {
      const uint16_t b = f * 4;
      const uint16_t q[6] = { b, b + 1, b + 2, b + 2, b + 1, b + 3 };
      memcpy(&idx[f * 6], q, sizeof(q));
   }
   GLuint vao, vbo, ibo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_DYNAMIC_DRAW);
   glGenBuffers(1, &ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, (void *)0);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, (void *)12);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   /* UBOs: per-frame block, 64 KiB chunk table re-pointed by range. */
   GLint ubo_align = 256;
   glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &ubo_align);
   GLuint ubo_frame, ubo_chunks, ssbo;
   glGenBuffers(1, &ubo_frame);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo_frame);
   glBufferData(GL_UNIFORM_BUFFER, 256, NULL, GL_DYNAMIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo_frame);
   float *tab = malloc(65536);
   for (int i = 0; i < 16384; i++)
      tab[i] = (i % 61) / 61.0f;
   glGenBuffers(1, &ubo_chunks);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo_chunks);
   glBufferData(GL_UNIFORM_BUFFER, 65536, tab, GL_STATIC_DRAW);
   free(tab);
   glGenBuffers(1, &ssbo);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
   glBufferData(GL_SHADER_STORAGE_BUFFER, 4096, NULL, GL_DYNAMIC_DRAW);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

   GLuint pa = program(vs_a, fs_a, NULL), pb = program(vs_post, fs_b, NULL);
   GLuint pc = program(vs_post, fs_c, NULL), pcs = program(NULL, NULL, cs_src);
   const GLint a_model = glGetUniformLocation(pa, "u_model");
   const GLint a_tint = glGetUniformLocation(pa, "u_tint");
   const GLint b_rect = glGetUniformLocation(pb, "u_rect"), b_tint = glGetUniformLocation(pb, "u_tint");
   const GLint c_rect = glGetUniformLocation(pc, "u_rect"), c_tint = glGetUniformLocation(pc, "u_tint");
   const GLint cs_t = glGetUniformLocation(pcs, "u_t");
   glUseProgram(pa);
   glUniform1i(glGetUniformLocation(pa, "u_tex"), 0);
   glUseProgram(pb);
   glUniform1i(glGetUniformLocation(pb, "u_src"), 0);
   glUseProgram(pc);
   glUniform1i(glGetUniformLocation(pc, "u_src"), 0);

   double wall_sum = 0, cpu_sum = 0;
   const int warm = 3;
   uint32_t stream[64 * 64];
   for (int f = 0; f < frames + warm; f++) {
      const double w0 = now_ms(CLOCK_MONOTONIC), c0 = now_ms(CLOCK_PROCESS_CPUTIME_ID);
      const int fr = f % 7; /* the content repeats, the checksum is of the last frame */

      /* Streamed texture update. */
      for (int i = 0; i < 64 * 64; i++)
         stream[i] = 0xff000000u | ((i * 97u + fr * 131u) & 0xffffffu);
      glBindTexture(GL_TEXTURE_2D, tex[0]);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 32, 32, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, stream);

      /* Pass A. */
      float frame_ub[20] = { 0 };
      frame_ub[0] = frame_ub[5] = 0.9f;
      frame_ub[10] = -0.5f;
      frame_ub[15] = 1.0f;
      frame_ub[16] = 0.1f * fr;
      frame_ub[17] = 0.2f;
      frame_ub[18] = 0.3f;
      glBindBuffer(GL_UNIFORM_BUFFER, ubo_frame);
      glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(frame_ub), frame_ub);
      glBindFramebuffer(GL_FRAMEBUFFER, fb_a);
      glViewport(0, 0, W, H);
      glEnable(GL_DEPTH_TEST);
      glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
      glUseProgram(pa);
      glBindVertexArray(vao);
      glActiveTexture(GL_TEXTURE0);
      for (int d = 0; d < draws; d++) {
         if (d % 100 == 50) {
            /* Busy vertex buffer write (in-place in stock). */
            float wob = 0.5f + 0.01f * ((d / 100 + fr) % 5);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float), &wob);
         }
         if (d % 16 == 0)
            glBindBufferRange(GL_UNIFORM_BUFFER, 1, ubo_chunks,
                              (GLintptr)((d / 16) % 15) * 4096 / ubo_align * ubo_align, 1024);
         if (d % 3 == 0) {
            glBindTexture(GL_TEXTURE_2D, tex[(d / 3) % NTEX]);
            glBindSampler(0, samp[(d / 3) & 1]);
         }
         float m[16] = { 0 };
         const float s = 0.05f + 0.02f * (d % 5);
         m[0] = m[5] = m[10] = s;
         m[12] = ((d * 37) % 100) / 50.0f - 1.0f;
         m[13] = ((d * 53 + fr * 11) % 100) / 50.0f - 1.0f;
         m[14] = ((d * 17) % 100) / 100.0f;
         m[15] = 1.0f;
         glUniformMatrix4fv(a_model, 1, GL_FALSE, m);
         glUniform4f(a_tint, 0.5f + (d % 3) * 0.2f, 0.6f, 0.7f + (d % 4) * 0.05f, 1.0f);
         glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
      }
      glDisable(GL_DEPTH_TEST);
      glBindSampler(0, 0);

      /* Pass B: sample A. */
      glBindFramebuffer(GL_FRAMEBUFFER, fb_b);
      glViewport(0, 0, W / 2, H / 2);
      glUseProgram(pb);
      glBindTexture(GL_TEXTURE_2D, rt_a);
      for (int d = 0; d < 64; d++) {
         glUniform4f(b_rect, 0.25f, 0.25f, (d % 8) / 4.0f - 0.875f, (d / 8) / 4.0f - 0.875f);
         glUniform4f(b_tint, 1.0f, 0.9f + (d % 2) * 0.1f, 1.0f, 1.0f);
         glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)(uintptr_t)(4 * 6 * 2));
      }

      /* Compute pass writing the SSBO pass C reads. */
      glUseProgram(pcs);
      glUniform1f(cs_t, 0.125f * fr);
      glDispatchCompute(16, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      /* Pass C: SSBO + sample B. */
      glBindFramebuffer(GL_FRAMEBUFFER, fb_c);
      glViewport(0, 0, W, H);
      glUseProgram(pc);
      glBindTexture(GL_TEXTURE_2D, rt_b);
      for (int d = 0; d < 16; d++) {
         glUniform4f(c_rect, 0.5f, 0.5f, (d % 4) / 2.0f - 0.75f, (d / 4) / 2.0f - 0.75f);
         glUniform4f(c_tint, 1.0f, 1.0f, 0.8f + (d % 3) * 0.1f, 1.0f);
         glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)(uintptr_t)(4 * 6 * 2));
      }
      glFinish();
      const double wall = now_ms(CLOCK_MONOTONIC) - w0;
      const double cpu = now_ms(CLOCK_PROCESS_CPUTIME_ID) - c0;
      if (f >= warm) {
         wall_sum += wall;
         cpu_sum += cpu;
      }
   }
   uint8_t *px = malloc(W * H * 4);
   uint32_t h = 2166136261u;
   GLuint fbs[2] = { fb_a, fb_c };
   for (int k = 0; k < 2; k++) {
      glBindFramebuffer(GL_FRAMEBUFFER, fbs[k]);
      glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
      for (int i = 0; i < W * H * 4; i++)
         h = (h ^ px[i]) * 16777619u;
   }
   free(px);
   free(texels);
   printf("frames %d draws/frame %d: wall %.2f ms/frame, CPU %.2f ms/frame, checksum %08x, "
          "GL error 0x%x\n", frames, draws, wall_sum / frames, cpu_sum / frames, h, glGetError());
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   return 0;
}
