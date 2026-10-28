/* SPDX-License-Identifier: MIT
 *
 * D2: empty vertex streams and index ranges. Pre-Turing 3D classes have no
 * stream/index size methods, only end addresses; an empty range is bound to
 * the zero page with end = zero_page - 1, as NVK's nvk_mme_bind_vb /
 * nvk_mme_bind_ib (nvk_cmd_draw.c:3641-3668, 3740-3777).
 *
 *   gl-zero-vb
 *
 * 1. attribute 1 sourced at offset == buffer size (an empty stream) and
 *    attribute 2 from a zero-size buffer object, both read by the vertex
 *    shader but multiplied by 0: the quad must come out exactly green;
 * 2. glDrawElements whose index offset == the index buffer size, and user
 *    indices in a stream upload, then a normal indexed draw: exact pixels.
 * Run with the kernel log watched: 0 new NVRM Xid lines is part of PASS.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 32
#define H 32

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static GLuint
shader(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("FAIL: shader: %s\n", log);
      exit(1);
   }
   return s;
}

static int
count_not(const unsigned char *px, unsigned r, unsigned g, unsigned b, unsigned a)
{
   int n = 0;
   for (int i = 0; i < W * H; i++)
      n += px[i * 4] != r || px[i * 4 + 1] != g || px[i * 4 + 2] != b || px[i * 4 + 3] != a;
   return n;
}

int
main(void)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_dpy =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = get_dpy(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   eglInitialize(dpy, NULL, NULL);
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint cattr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, cattr);
   if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: context\n");
      return 1;
   }
   printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

   GLuint rt, fbo;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
   glViewport(0, 0, W, H);

   GLuint prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER,
      "#version 300 es\nlayout(location=0) in vec2 p;\nlayout(location=1) in vec4 a1;\n"
      "layout(location=2) in vec4 a2;\nout vec4 c;\n"
      "void main() { c = vec4(0.0, 1.0, 0.0, 1.0) + 0.0 * (a1 + a2);\n"
      "              gl_Position = vec4(p, 0.0, 1.0); }\n"));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER,
      "#version 300 es\nprecision mediump float;\nin vec4 c;\nout vec4 o;\n"
      "void main() { o = c; }\n"));
   glLinkProgram(prog);
   glUseProgram(prog);

   static const float quad[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
   GLuint vbo, vb1, vb2, ibo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   /* attribute 1: offset == size, an empty stream */
   glGenBuffers(1, &vb1);
   glBindBuffer(GL_ARRAY_BUFFER, vb1);
   glBufferData(GL_ARRAY_BUFFER, 64, NULL, GL_STATIC_DRAW);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, (const void *)64);
   glEnableVertexAttribArray(1);
   /* attribute 2: a zero-size buffer object */
   glGenBuffers(1, &vb2);
   glBindBuffer(GL_ARRAY_BUFFER, vb2);
   glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW);
   glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, 0);
   glEnableVertexAttribArray(2);

   unsigned char px[W * H * 4];
   glClearColor(0, 0, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   CHECK(glGetError() == GL_NO_ERROR, "empty streams: no GL error");
   CHECK(count_not(px, 0, 255, 0, 255) == 0, "empty streams: quad exactly green (%d other px)",
         count_not(px, 0, 255, 0, 255));

   /* 2: index offset == index buffer size (an empty index range). */
   static const unsigned short idx[] = { 0, 1, 2, 2, 1, 3 };
   glGenBuffers(1, &ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)sizeof(idx));
   glFinish();
   CHECK(glGetError() == GL_NO_ERROR, "index offset == size: no GL error, no fault");
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   CHECK(count_not(px, 0, 255, 0, 255) == 0, "indexed draw after it: exactly green (%d other px)",
         count_not(px, 0, 255, 0, 255));
   /* user indices (stream upload, nonzero first index) */
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
   glClear(GL_COLOR_BUFFER_BIT);
   static const unsigned short uidx[] = { 9, 9, 0, 1, 2, 2, 1, 3 };
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, uidx + 2);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   CHECK(count_not(px, 0, 255, 0, 255) == 0, "user indices: exactly green (%d other px)",
         count_not(px, 0, 255, 0, 255));
   printf("%s\n", failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
