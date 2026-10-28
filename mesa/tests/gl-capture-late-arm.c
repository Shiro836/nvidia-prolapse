/* SPDX-License-Identifier: MIT
 *
 * gl-capture-late-arm: the Rust RM perf capture (rm/src/capture.rs) armed
 * on a process that is already rendering, without a restart (2026-10-16, for
 * catching an intermittent bug such as the GTA SA flicker in the process that
 * shows it). On Android the trigger is `setprop debug.nvrm.capture <dir>`; on
 * the host the same poll reads the environment, which this program changes at
 * run time with setenv().
 *
 * Steps: render frames with no capture configured; setenv NV_RM_CAPTURE=DIR
 * (+ MODE push); keep rendering for > 1 s (the backend polls once per
 * second from submit, then arms); create DIR/go; render more frames; exit
 * (the capture file is closed at device teardown). PASS = the program saw
 * DIR/go removed or a cap-*.bin appear (checked by the caller with
 * tools/nvcap_seq.py: submits with pushbuffer words).
 *
 * Build as gl-g0a-clear.c (README); run: ./gl-capture-late-arm DIR
 */
#define _POSIX_C_SOURCE 200809L
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
now_s(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void
frame(int i)
{
   glClearColor((i & 1) ? 1.0f : 0.0f, 0.5f, 0.25f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();
}

static int
count_caps(const char *dir)
{
   DIR *d = opendir(dir);
   if (!d)
      return 0;
   int n = 0;
   struct dirent *e;
   while ((e = readdir(d)))
      n += !strncmp(e->d_name, "cap-", 4);
   closedir(d);
   return n;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s DIR\n", argv[0]);
      return 2;
   }
   const char *dir = argv[1];
   if (getenv("NV_RM_CAPTURE")) {
      fprintf(stderr, "NV_RM_CAPTURE must not be set at start\n");
      return 2;
   }
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
   GLuint rt, fb;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 64, 64);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);

   for (int i = 0; i < 20; i++)
      frame(i);
   const int before = count_caps(dir);

   /* Configure the capture now, as `setprop` would on Android. */
   setenv("NV_RM_CAPTURE_MODE", "push", 1);
   setenv("NV_RM_CAPTURE", dir, 1);
   const double t_arm = now_s();
   int i = 0;
   while (now_s() - t_arm < 1.5)
      frame(i++);
   char go[4096];
   snprintf(go, sizeof(go), "%s/go", dir);
   FILE *f = fopen(go, "w");
   if (!f) {
      perror("go");
      return 1;
   }
   fputs("push frames=4\n", f);
   fclose(f);
   const double t_go = now_s();
   while (now_s() - t_go < 0.6)
      frame(i++);
   const int during = count_caps(dir);
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   printf("capture files: %d before arming, %d after go (%d frames rendered)\n", before,
          during, i);
   const int pass = during > before;
   printf("%s\n", pass ? "PASS: late-armed capture started" : "FAIL: no capture file");
   return pass ? 0 : 1;
}
