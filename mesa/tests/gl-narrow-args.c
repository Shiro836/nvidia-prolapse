/* gl-narrow-args: GL entry points with integer parameters narrower than 32
 * bits (GLboolean here) must use only the parameter's own bits.
 *
 * The x86-64 SysV and i386 psABIs leave the bits above a char/short argument
 * unspecified; clang-built callees assume they were sign/zero-extended to 32
 * bits. Binary translators bridging another ABI (Android's ARM translation)
 * pass GLboolean arguments with garbage above bit 7: an arm64 Unity game
 * called glColorMask(0, 0, 0, 1) with the registers r=0x0 g=0x10000 b=0x100
 * a=0x1, and an entry point that tests the whole register turns that into
 * G|B|A instead of alpha only.
 *
 * The test calls glColorMask, glColorMaski and glDepthMask through function
 * pointers typed with 32-bit parameters (so the garbage reaches the callee),
 * both the exported symbols and the eglGetProcAddress results, then checks
 * GL_COLOR_WRITEMASK / GL_DEPTH_WRITEMASK and the pixels a masked clear and a
 * masked draw leave in an RGBA8 framebuffer.
 *
 * Build and run like gl-g0a-clear.c (tests/README.md). Exit 0 = PASS.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void (*wide4_fn)(uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*wide5_fn)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*wide1_fn)(uint32_t);

static int failures;

static void
check(int ok, const char *what)
{
   printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
   if (!ok)
      failures++;
}

static GLuint
program(void)
{
   static const char *vs =
      "#version 300 es\n"
      "void main() {\n"
      "   vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);\n"
      "   gl_Position = vec4(p, 0.0, 1.0);\n"
      "}\n";
   static const char *fs =
      "#version 300 es\n"
      "precision mediump float;\n"
      "out vec4 c;\n"
      "void main() { c = vec4(0.0, 0.0, 0.0, 0.25); }\n";
   GLuint p = glCreateProgram();
   const char *src[2] = { vs, fs };
   const GLenum type[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
   for (int i = 0; i < 2; i++) {
      GLuint s = glCreateShader(type[i]);
      glShaderSource(s, 1, &src[i], NULL);
      glCompileShader(s);
      glAttachShader(p, s);
   }
   glLinkProgram(p);
   GLint linked = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &linked);
   check(linked, "program links");
   return p;
}

/* One full round on `mask4` (glColorMask-shaped) and `depth` entry points. */
static void
round_masks(const char *via, wide4_fn mask4, wide1_fn depth, GLuint prog)
{
   char what[160];
   GLboolean m[4];

   /* White, full mask. */
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glClearColor(1, 1, 1, 1);
   glClear(GL_COLOR_BUFFER_BIT);

   /* The arm64 game's call: alpha only, garbage above bit 7 of g and b. */
   mask4(0x0, 0x10000, 0x100, 0x1);
   glGetBooleanv(GL_COLOR_WRITEMASK, m);
   snprintf(what, sizeof(what), "%s: glColorMask(0, 0x10000, 0x100, 1) -> writemask %d%d%d%d (want 0001)",
            via, m[0], m[1], m[2], m[3]);
   check(!m[0] && !m[1] && !m[2] && m[3], what);

   /* Masked clear and masked draw both may write alpha only. */
   glClearColor(0, 0, 0, 0.5f);
   glClear(GL_COLOR_BUFFER_BIT);
   uint8_t px[4];
   glReadPixels(8, 8, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
   snprintf(what, sizeof(what), "%s: masked clear -> %u %u %u %u (want 255 255 255 128)",
            via, px[0], px[1], px[2], px[3]);
   check(px[0] == 255 && px[1] == 255 && px[2] == 255 && px[3] >= 127 && px[3] <= 128, what);

   glUseProgram(prog);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glReadPixels(8, 8, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
   snprintf(what, sizeof(what), "%s: masked draw -> %u %u %u %u (want 255 255 255 64)",
            via, px[0], px[1], px[2], px[3]);
   check(px[0] == 255 && px[1] == 255 && px[2] == 255 && px[3] >= 63 && px[3] <= 64, what);

   /* Garbage that is "true" in the low byte stays true. */
   mask4(0x101, 0x10001, 0x201, 0x0);
   glGetBooleanv(GL_COLOR_WRITEMASK, m);
   snprintf(what, sizeof(what), "%s: glColorMask(0x101, 0x10001, 0x201, 0) -> writemask %d%d%d%d (want 1110)",
            via, m[0], m[1], m[2], m[3]);
   check(m[0] && m[1] && m[2] && !m[3], what);

   GLboolean d;
   depth(0x100);
   glGetBooleanv(GL_DEPTH_WRITEMASK, &d);
   snprintf(what, sizeof(what), "%s: glDepthMask(0x100) -> %d (want 0)", via, d);
   check(!d, what);
   depth(0x1);
   glGetBooleanv(GL_DEPTH_WRITEMASK, &d);
   snprintf(what, sizeof(what), "%s: glDepthMask(1) -> %d (want 1)", via, d);
   check(d, what);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}

int
main(void)
{
   EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(dpy, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API)) {
      printf("FAIL: EGL init\n");
      return 1;
   }
   static const EGLint ctx_attr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1,
                                      EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: context\n");
      return 1;
   }
   printf("GL_RENDERER %s\n", glGetString(GL_RENDERER));

   GLuint rb, fb;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 64, 64);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "framebuffer complete");
   glViewport(0, 0, 64, 64);
   GLuint prog = program();

   /* Exported entry points (linked against libGLESv2). */
   round_masks("export", (wide4_fn)(void (*)(void))glColorMask,
               (wide1_fn)(void (*)(void))glDepthMask, prog);
   /* eglGetProcAddress entry points (the dispatch library's table). */
   round_masks("getproc", (wide4_fn)eglGetProcAddress("glColorMask"),
               (wide1_fn)eglGetProcAddress("glDepthMask"), prog);

   /* glColorMaski (ES 3.2 / OES_draw_buffers_indexed) through GetProcAddress. */
   const char *names[] = { "glColorMaski", "glColorMaskiOES", "glColorMaskiEXT" };
   for (unsigned i = 0; i < 3; i++) {
      wide5_fn maski = (wide5_fn)eglGetProcAddress(names[i]);
      if (!maski || !strstr((const char *)glGetString(GL_EXTENSIONS), "draw_buffers_indexed"))
         continue;
      GLboolean m[4];
      maski(0, 0x0, 0x10000, 0x100, 0x1);
      glGetBooleani_v(GL_COLOR_WRITEMASK, 0, m);
      char what[160];
      snprintf(what, sizeof(what), "%s(0, 0, 0x10000, 0x100, 1) -> writemask %d%d%d%d (want 0001)",
               names[i], m[0], m[1], m[2], m[3]);
      check(!m[0] && !m[1] && !m[2] && m[3], what);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   }

   check(glGetError() == GL_NO_ERROR, "no GL error");
   printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
