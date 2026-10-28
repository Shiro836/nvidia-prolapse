/* SPDX-License-Identifier: MIT
 *
 * G0a: EGL surfaceless + GLES2 context on the nvrm gallium driver,
 * glClear of a renderbuffer, glReadPixels must return the exact colour.
 *
 *   cc -std=c11 -I<mesa>/include gl-g0a-clear.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -o gl-g0a-clear
 *
 * Exit 0 only if every pixel of two successive clears matches exactly and
 * GL_RENDERER is the nvrm screen (vendor string checked).
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 48

static int
check_clear(const uint8_t want[4])
{
   glClearColor(want[0] / 255.0f, want[1] / 255.0f, want[2] / 255.0f, want[3] / 255.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   static uint8_t px[W * H * 4];
   memset(px, 0xcd, sizeof(px));
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   GLenum err = glGetError();
   if (err != GL_NO_ERROR) {
      fprintf(stderr, "FAIL: GL error 0x%x\n", err);
      return 1;
   }
   unsigned bad = 0;
   for (unsigned i = 0; i < W * H; i++)
      if (memcmp(&px[i * 4], want, 4))
         bad++;
   printf("clear (%u,%u,%u,%u): pixel[0]=(%u,%u,%u,%u) pixel[last]=(%u,%u,%u,%u) "
          "mismatches %u/%u\n",
          want[0], want[1], want[2], want[3], px[0], px[1], px[2], px[3],
          px[(W * H - 1) * 4], px[(W * H - 1) * 4 + 1], px[(W * H - 1) * 4 + 2],
          px[(W * H - 1) * 4 + 3], bad, W * H);
   return bad != 0;
}

int
main(void)
{
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
   printf("EGL %d.%d vendor '%s'\n", major, minor, eglQueryString(dpy, EGL_VENDOR));
   if (!eglBindAPI(EGL_OPENGL_ES_API)) {
      fprintf(stderr, "FAIL: eglBindAPI\n");
      return 1;
   }
   static const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attribs);
   if (ctx == EGL_NO_CONTEXT) {
      fprintf(stderr, "FAIL: eglCreateContext (0x%x)\n", eglGetError());
      return 1;
   }
   if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      fprintf(stderr, "FAIL: eglMakeCurrent (0x%x)\n", eglGetError());
      return 1;
   }
   const char *vendor = (const char *)glGetString(GL_VENDOR);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("GL_VENDOR '%s'\nGL_RENDERER '%s'\nGL_VERSION '%s'\n", vendor, renderer,
          (const char *)glGetString(GL_VERSION));
   if (!vendor || !strstr(vendor, "nvrm")) {
      fprintf(stderr, "FAIL: context is not on the nvrm driver\n");
      return 1;
   }

   GLuint rb, fbo;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8_OES, W, H);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   if (status != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FAIL: framebuffer status 0x%x\n", status);
      return 1;
   }
   glViewport(0, 0, W, H);

   static const uint8_t c1[4] = { 51, 102, 153, 255 };
   static const uint8_t c2[4] = { 255, 0, 128, 64 };
   int fails = check_clear(c1) + check_clear(c2);

   glDeleteFramebuffers(1, &fbo);
   glDeleteRenderbuffers(1, &rb);
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   printf("%s\n", fails ? "G0a FAIL" : "G0a PASS");
   return fails ? 1 : 0;
}
