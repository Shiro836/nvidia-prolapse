/* SPDX-License-Identifier: MIT
 *
 * gl-shader-async: nvrm's asynchronous shader compile (compile queue, waits
 * at first use, drop on delete) seen from the GL API, with shader objects
 * shared across a share group.
 *
 * usage: gl-shader-async DIR [ROUNDS]
 *   DIR holds pairs.txt ("a.vert b.frag" per line) and the sources (the
 *   shbench / thrbench format; MC set A from matdump.py makes the NAK jobs
 *   long enough to still be queued or running when the GL calls below arrive).
 *
 * Per round:
 *   1. drop: A links every pair and deletes the program at once (the compile
 *      job is dropped or waited for, never run on freed memory). In round 0
 *      the jobs are cold NAK compiles; later phases hit the program cache.
 *   2. cross: context A (main thread) links every pair and hands the program
 *      to context B (same share group, second thread) without using it; B
 *      uses, draws, finishes and deletes it. The first wait and the delete
 *      happen in another context than the one that created the shaders.
 *   3. status: glMaxShaderCompilerThreadsKHR(0xffffffff), link every pair,
 *      poll GL_COMPLETION_STATUS_KHR until all report done (bounded), then
 *      draw each one.
 * PASS = every link succeeds, no GL error after any draw, every program
 * reaches COMPLETION_STATUS within 60 s, and the process exits normally. */
#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef GL_COMPLETION_STATUS_KHR
#define GL_COMPLETION_STATUS_KHR 0x91B1
#endif

#define MAXP 128
static char *vsrc[MAXP], *fsrc[MAXP];
static int np, errors;
static EGLDisplay dpy;
static EGLConfig cfg;
static EGLContext ctx_a;

static double
now(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static char *
slurp(const char *dir, const char *name)
{
   char path[4096];
   snprintf(path, sizeof(path), "%s/%s", dir, name);
   FILE *f = fopen(path, "rb");
   if (!f) {
      perror(path);
      exit(2);
   }
   fseek(f, 0, SEEK_END);
   long len = ftell(f);
   rewind(f);
   char *buf = malloc(len + 1);
   if (fread(buf, 1, len, f) != (size_t)len)
      exit(2);
   buf[len] = 0;
   fclose(f);
   return buf;
}

static EGLContext
make_context(EGLContext share)
{
   const EGLint attribs[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   EGLContext c = eglCreateContext(dpy, cfg, share, attribs);
   if (!c || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) {
      printf("FAIL context creation 0x%x\n", eglGetError());
      exit(1);
   }
   GLuint tex, fbo;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   glViewport(0, 0, 4, 4);
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   return c;
}

static GLuint
compile(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[512];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("FAIL compile: %s\n", log);
      exit(1);
   }
   return s;
}

static GLuint
link_pair(int i, bool check)
{
   GLuint p = glCreateProgram();
   GLuint v = compile(GL_VERTEX_SHADER, vsrc[i]), f = compile(GL_FRAGMENT_SHADER, fsrc[i]);
   glAttachShader(p, v);
   glAttachShader(p, f);
   glLinkProgram(p);
   glDeleteShader(v);
   glDeleteShader(f);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (check && !ok) {
      printf("FAIL link of pair %d\n", i);
      errors++;
   }
   return p;
}

static int
is_sampler(GLenum t)
{
   switch (t) {
   case GL_SAMPLER_2D: case GL_SAMPLER_3D: case GL_SAMPLER_CUBE: case GL_SAMPLER_2D_SHADOW:
   case GL_SAMPLER_2D_ARRAY: case GL_SAMPLER_2D_ARRAY_SHADOW: case GL_SAMPLER_CUBE_SHADOW:
   case GL_INT_SAMPLER_2D: case GL_INT_SAMPLER_3D: case GL_INT_SAMPLER_CUBE: case GL_INT_SAMPLER_2D_ARRAY:
   case GL_UNSIGNED_INT_SAMPLER_2D: case GL_UNSIGNED_INT_SAMPLER_3D: case GL_UNSIGNED_INT_SAMPLER_CUBE:
   case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY: case GL_SAMPLER_2D_MULTISAMPLE: case GL_SAMPLER_CUBE_MAP_ARRAY:
   case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW: case GL_SAMPLER_BUFFER:
      return 1;
   }
   return 0;
}

/* Use + draw + finish; every sampler on its own unit (MC programs leave all
 * of them on unit 0, which fails draw validation: thrbench2). */
static void
draw_program(GLuint p, const char *phase, int i)
{
   glUseProgram(p);
   GLint n = 0;
   glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &n);
   int unit = 0;
   for (GLint u = 0; u < n; u++) {
      char name[256];
      GLint size;
      GLenum type;
      glGetActiveUniform(p, u, sizeof(name), NULL, &size, &type, name);
      if (!is_sampler(type))
         continue;
      GLint loc = glGetUniformLocation(p, name);
      for (GLint k = 0; k < size; k++) {
         if (loc >= 0)
            glUniform1i(loc + k, unit);
         unit++;
      }
   }
   glDrawArrays(GL_TRIANGLES, 0, 3);
   GLenum e = glGetError();
   if (e) {
      printf("FAIL %s: GL error 0x%x drawing pair %d\n", phase, e, i);
      errors++;
   }
   glFinish();
   glUseProgram(0);
}

/* Phase 1 hand-off A -> B. */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static GLuint handed[MAXP];
static int handed_n, handed_done;

static void *
context_b(void *arg)
{
   (void)arg;
   EGLContext c = make_context(ctx_a);
   for (int i = 0; i < np; i++) {
      pthread_mutex_lock(&lock);
      while (handed_n <= i)
         pthread_cond_wait(&cond, &lock);
      GLuint p = handed[i];
      pthread_mutex_unlock(&lock);
      draw_program(p, "cross", i);
      glDeleteProgram(p);
   }
   glFinish();
   pthread_mutex_lock(&lock);
   handed_done = 1;
   pthread_mutex_unlock(&lock);
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, c);
   return NULL;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: gl-shader-async DIR [ROUNDS]\n");
      return 2;
   }
   const char *dir = argv[1];
   const int rounds = argc > 2 ? atoi(argv[2]) : 1;
   char path[4096];
   snprintf(path, sizeof(path), "%s/pairs.txt", dir);
   FILE *f = fopen(path, "r");
   if (!f) {
      perror(path);
      return 2;
   }
   char a[64], b[64];
   while (np < MAXP && fscanf(f, "%63s %63s", a, b) == 2) {
      vsrc[np] = slurp(dir, a);
      fsrc[np] = slurp(dir, b);
      np++;
   }
   fclose(f);

   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(dpy, NULL, NULL)) {
      printf("FAIL eglInitialize\n");
      return 1;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint cattr[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
                            EGL_OPENGL_ES3_BIT, EGL_NONE };
   EGLint n = 0;
   if (!eglChooseConfig(dpy, cattr, &cfg, 1, &n) || !n)
      cfg = EGL_NO_CONFIG_KHR;
   ctx_a = make_context(EGL_NO_CONTEXT);
   printf("renderer: %s | %s, %d pairs\n", glGetString(GL_RENDERER), glGetString(GL_VERSION), np);
   void (*max_threads)(GLuint) =
      (void (*)(GLuint))eglGetProcAddress("glMaxShaderCompilerThreadsKHR");
   if (!max_threads) {
      printf("FAIL no glMaxShaderCompilerThreadsKHR\n");
      return 1;
   }

   for (int r = 0; r < rounds; r++) {
      /* 1. delete right after link (first: in round 0 the jobs are cold NAK
       *    compiles, still queued or running at the delete) */
      double t0 = now();
      for (int i = 0; i < np; i++)
         glDeleteProgram(link_pair(i, true));
      glFinish();
      printf("round %d drop: %d link+delete in %.1f ms\n", r, np, now() - t0);

      /* 2. cross-context first use and delete */
      handed_n = handed_done = 0;
      pthread_t tb;
      pthread_create(&tb, NULL, context_b, NULL);
      t0 = now();
      for (int i = 0; i < np; i++) {
         GLuint p = link_pair(i, true);
         pthread_mutex_lock(&lock);
         handed[handed_n++] = p;
         pthread_cond_signal(&cond);
         pthread_mutex_unlock(&lock);
      }
      double t_links = now() - t0;
      pthread_join(tb, NULL);
      printf("round %d cross: %d links on A in %.1f ms, B done at %.1f ms\n", r, np, t_links, now() - t0);

      /* 3. COMPLETION_STATUS polling, then draw */
      max_threads(0xffffffff);
      GLuint progs[MAXP];
      t0 = now();
      for (int i = 0; i < np; i++)
         progs[i] = link_pair(i, true);
      double t_links3 = now() - t0;
      int pending = np, polls = 0;
      while (pending && now() - t0 < 60000) {
         pending = 0;
         for (int i = 0; i < np; i++) {
            GLint done = 0;
            glGetProgramiv(progs[i], GL_COMPLETION_STATUS_KHR, &done);
            pending += !done;
         }
         polls++;
      }
      double t_status = now() - t0;
      if (pending) {
         printf("FAIL status: %d programs not complete after 60 s\n", pending);
         errors++;
      }
      for (int i = 0; i < np; i++) {
         draw_program(progs[i], "status", i);
         glDeleteProgram(progs[i]);
      }
      printf("round %d status: links %.1f ms, all complete at %.1f ms (%d polls)\n",
             r, t_links3, t_status, polls);
   }
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx_a);
   eglTerminate(dpy);
   printf("%s errors=%d\n", errors ? "FAIL" : "PASS", errors);
   return errors != 0;
}
