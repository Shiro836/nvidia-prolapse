/* SPDX-License-Identifier: MIT
 *
 * G0b: EGL surfaceless + GLES2 on the nvrm gallium driver draws through
 * NAK-compiled shaders; the readback is compared with a reference image
 * produced by the same program on another driver (llvmpipe).
 *
 *   gl-g0b-triangle [--es3] [--flat] <vendor-substring> <out.rgba> [<reference.rgba>]
 *
 * --es3 asks for an ES 3.0 context (EGL_CONTEXT_CLIENT_VERSION 3) and fails
 * unless GL_VERSION is "OpenGL ES 3.x"; the scene is the same.
 * --flat gives the three vertices of each primitive the same colour, so the
 * expected colour does not depend on varying-interpolation precision (which
 * GL leaves to the implementation: llvmpipe and NVIDIA's own GLES differ by
 * up to 1 LSB on the gradient scene). Coverage is compared in both modes.
 * Tints avoid x.5 UNORM8 ties (0.5 * 255): NVIDIA rounds 127.5 to 127,
 * llvmpipe to 128.
 *
 * Scene (64x64 RGBA8 renderbuffer FBO):
 *  1. glClear (16,32,48,255)
 *  2. glDrawArrays(GL_TRIANGLES) from a VBO: vec2 position + vec3 colour per
 *     vertex; the VS adds uniform u_offset, the FS multiplies by u_tint
 *     (uniforms in both stages -> constant buffer slot 0 of both groups).
 *  3. glDrawElements(GL_TRIANGLES, GL_UNSIGNED_SHORT) with client-side
 *     indices (user index upload) for a quad, with new uniform values.
 * Vertex positions are chosen so that no edge passes through a pixel
 * centre (coverage does not depend on the fill-rule tie break).
 *
 * Exit 0 if the vendor matches, no GL error occurred, and (with a
 * reference) every pixel matches exactly.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 64

static const char *vs_src =
   "attribute vec2 a_pos;\n"
   "attribute vec3 a_color;\n"
   "uniform vec4 u_offset;\n"
   "varying vec3 v_color;\n"
   "void main() {\n"
   "   v_color = a_color;\n"
   "   gl_Position = vec4(a_pos + u_offset.xy, 0.0, 1.0);\n"
   "}\n";

static const char *fs_src =
   "precision mediump float;\n"
   "varying vec3 v_color;\n"
   "uniform vec4 u_tint;\n"
   "void main() {\n"
   "   gl_FragColor = vec4(v_color, 1.0) * u_tint;\n"
   "}\n";

static GLuint
compile(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "FAIL: shader compile: %s\n", log);
      exit(1);
   }
   return s;
}

int
main(int argc, char **argv)
{
   bool flat = false, es3 = false;
   if (argc > 1 && !strcmp(argv[1], "--es3")) {
      es3 = true;
      argv++;
      argc--;
   }
   if (argc > 1 && !strcmp(argv[1], "--flat")) {
      flat = true;
      argv++;
      argc--;
   }
   if (argc < 3) {
      fprintf(stderr, "usage: %s [--es3] [--flat] <vendor-substring> <out.rgba> [<reference.rgba>]\n", argv[0]);
      return 2;
   }
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   if (!get_display) {
      fprintf(stderr, "FAIL: no eglGetPlatformDisplayEXT\n");
      return 1;
   }
   EGLDisplay dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   EGLint major, minor;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &major, &minor)) {
      fprintf(stderr, "FAIL: eglInitialize (0x%x)\n", eglGetError());
      return 1;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, es3 ? 3 : 2, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attribs);
   if (ctx == EGL_NO_CONTEXT ||
       !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      fprintf(stderr, "FAIL: context (0x%x)\n", eglGetError());
      return 1;
   }
   const char *vendor = (const char *)glGetString(GL_VENDOR);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   const char *version = (const char *)glGetString(GL_VERSION);
   printf("GL_VENDOR '%s'\nGL_RENDERER '%s'\nGL_VERSION '%s'\n", vendor, renderer, version);
   if (es3 && (!version || strncmp(version, "OpenGL ES 3.", 12))) {
      fprintf(stderr, "FAIL: --es3 but the context is '%s'\n", version ? version : "(null)");
      return 1;
   }
   if (!vendor || !renderer ||
       (!strstr(vendor, argv[1]) && !strstr(renderer, argv[1]))) {
      fprintf(stderr, "FAIL: context is not on '%s'\n", argv[1]);
      return 1;
   }

   GLuint rb, fbo;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8_OES, W, H);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FAIL: framebuffer incomplete\n");
      return 1;
   }
   glViewport(0, 0, W, H);

   GLuint prog = glCreateProgram();
   glAttachShader(prog, compile(GL_VERTEX_SHADER, vs_src));
   glAttachShader(prog, compile(GL_FRAGMENT_SHADER, fs_src));
   glBindAttribLocation(prog, 0, "a_pos");
   glBindAttribLocation(prog, 1, "a_color");
   glLinkProgram(prog);
   GLint ok = 0;
   glGetProgramiv(prog, GL_LINK_STATUS, &ok);
   if (!ok) {
      fprintf(stderr, "FAIL: link\n");
      return 1;
   }
   glUseProgram(prog);
   const GLint u_offset = glGetUniformLocation(prog, "u_offset");
   const GLint u_tint = glGetUniformLocation(prog, "u_tint");

   /* x, y, r, g, b: triangle then quad corners. */
   float verts[] = {
      -0.83f, -0.71f, 1.0f, 0.0f, 0.0f,
       0.77f, -0.52f, 0.0f, 1.0f, 0.0f,
       0.05f,  0.89f, 0.0f, 0.0f, 1.0f,
       0.52f,  0.48f, 1.0f, 1.0f, 0.0f,
       0.91f,  0.48f, 0.0f, 1.0f, 1.0f,
       0.91f,  0.87f, 1.0f, 0.0f, 1.0f,
       0.52f,  0.87f, 1.0f, 1.0f, 1.0f,
   };
   if (flat) {
      /* Triangle: colour of vertex 0; quad: colour of vertex 3. */
      for (unsigned v = 1; v < 7; v++) {
         const unsigned src = v < 3 ? 0 : 3;
         memcpy(&verts[v * 5 + 2], &verts[src * 5 + 2], 3 * sizeof(float));
      }
   }
   GLuint vbo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                         (void *)(2 * sizeof(float)));
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   glClearColor(16 / 255.0f, 32 / 255.0f, 48 / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   glUniform4f(u_offset, 0.0f, 0.0f, 0.0f, 0.0f);
   glUniform4f(u_tint, 1.0f, 0.75f, 0.4f, 1.0f);
   glDrawArrays(GL_TRIANGLES, 0, 3);

   static const GLushort quad[] = { 3, 4, 5, 3, 5, 6 };
   glUniform4f(u_offset, -0.07f, 0.03f, 0.0f, 0.0f);
   glUniform4f(u_tint, 0.4f, 1.0f, 0.25f, 1.0f);
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, quad);

   static uint8_t px[W * H * 4];
   memset(px, 0xcd, sizeof(px));
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   GLenum err = glGetError();
   if (err != GL_NO_ERROR) {
      fprintf(stderr, "FAIL: GL error 0x%x\n", err);
      return 1;
   }
   FILE *f = fopen(argv[2], "wb");
   if (!f || fwrite(px, sizeof(px), 1, f) != 1) {
      fprintf(stderr, "FAIL: cannot write %s\n", argv[2]);
      return 1;
   }
   fclose(f);

   unsigned background = 0;
   for (unsigned i = 0; i < W * H; i++)
      background += px[i * 4] == 16 && px[i * 4 + 1] == 32 && px[i * 4 + 2] == 48;
   printf("covered pixels %u/%u, centre (32,32) = (%u,%u,%u,%u)\n", W * H - background,
          W * H, px[(32 * W + 32) * 4], px[(32 * W + 32) * 4 + 1],
          px[(32 * W + 32) * 4 + 2], px[(32 * W + 32) * 4 + 3]);

   int fails = 0;
   if (argc > 3) {
      static uint8_t ref[W * H * 4];
      FILE *r = fopen(argv[3], "rb");
      if (!r || fread(ref, sizeof(ref), 1, r) != 1) {
         fprintf(stderr, "FAIL: cannot read %s\n", argv[3]);
         return 1;
      }
      fclose(r);
      unsigned bad = 0, coverage = 0, maxd = 0;
      for (unsigned i = 0; i < W * H; i++) {
         const uint8_t *a = &px[i * 4], *b = &ref[i * 4];
         const bool bg_a = a[0] == 16 && a[1] == 32 && a[2] == 48;
         const bool bg_b = b[0] == 16 && b[1] == 32 && b[2] == 48;
         coverage += bg_a != bg_b;
         if (memcmp(a, b, 4)) {
            bad++;
            for (unsigned c = 0; c < 4; c++) {
               unsigned d = a[c] > b[c] ? a[c] - b[c] : b[c] - a[c];
               if (d > maxd)
                  maxd = d;
            }
            if (bad <= 5)
               printf("  mismatch (%u,%u): got (%u,%u,%u,%u) ref (%u,%u,%u,%u)\n",
                      i % W, i / W, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
         }
      }
      printf("vs reference: mismatching pixels %u/%u, coverage differences %u, "
             "max channel difference %u\n", bad, W * H, coverage, maxd);
      fails = bad != 0;
   }

   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   printf("%s\n", fails ? "G0b FAIL" : "G0b PASS");
   return fails ? 1 : 0;
}
