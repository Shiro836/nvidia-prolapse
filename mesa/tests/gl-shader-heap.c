/*
 * nvrm shader heap growth (Minecraft Bedrock in-world, 2026-10-15: loading a
 * world aborted with "nvrm: shader heap (4194304 B) exhausted" — the heap was
 * one fixed 4 MiB BO; now NVK's growable contiguous arena).
 *
 * Links N distinct programs whose fragment shaders are long unrolled integer chains
 * (each a few KiB of SASS, together well past 4 MiB), keeps them all alive and
 * draws with every one into its own pixel of an RGBA8 FBO, then checks each
 * pixel against the value computed on the CPU. Programs uploaded across memory
 * boundaries (allocations straddling two arena memories) are exercised too.
 * Then deletes half, flushes, links N/2 more and checks again (reclaim).
 *
 *   cc -std=c11 -I<mesa>/include gl-shader-heap.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -lm -o gl-shader-heap
 *   LIBGL_DRIVERS_PATH=<build>/src/gallium/targets/dri ./gl-shader-heap [N]
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHAIN 400
static const char *vs = "#version 300 es\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0,1);}\n";

static unsigned expect(int k)
{
   /* Exact integer chain, must match the shader: x = k; x = (x ^ x >> 13) * 1664525 + c_i
    * (the xorshift keeps NIR from reassociating the chain into one IMAD). */
   unsigned x = (unsigned)k;
   for (int i = 0; i < CHAIN; i++)
      x = (x ^ (x >> 13)) * 1664525u + (unsigned)((i * 7919 + k * 31) % 65521);
   return x >> 24;
}

static GLuint make(int k)
{
   static char src[CHAIN * 64 + 512];
   char *s = src;
   s += sprintf(s, "#version 300 es\nprecision highp float;\nuniform float u;\nout vec4 c;\n"
                   "void main(){ uint x = %du + uint(u);\n", k);
   for (int i = 0; i < CHAIN; i++)
      s += sprintf(s, "x = (x ^ (x >> 13u)) * 1664525u + %du;\n", (i * 7919 + k * 31) % 65521);
   sprintf(s, "c = vec4(float(x >> 24u) / 255.0, float(%d %% 256) / 255.0, 0.0, 1.0); }\n", k);
   GLuint f = glCreateShader(GL_FRAGMENT_SHADER), v = glCreateShader(GL_VERTEX_SHADER);
   const char *fp = src;
   glShaderSource(f, 1, &fp, NULL); glCompileShader(f);
   glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
   GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f);
   glBindAttribLocation(p, 0, "p"); glLinkProgram(p);
   GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
   glDeleteShader(v); glDeleteShader(f);
   if (!ok) { printf("FAIL: link %d\n", k); exit(1); }
   return p;
}

static int check(GLuint *progs, int n, int base)
{
   int bad = 0;
   glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
   for (int i = 0; i < n; i++) {
      if (!progs[i]) continue;
      glUseProgram(progs[i]);
      glUniform1f(glGetUniformLocation(progs[i], "u"), 0.0f);
      glScissor(i % 64, i / 64, 1, 1); glEnable(GL_SCISSOR_TEST);
      glDrawArrays(GL_TRIANGLES, 0, 3);
   }
   glDisable(GL_SCISSOR_TEST);
   static unsigned char px[64 * 64 * 4];
   glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, px);
   for (int i = 0; i < n; i++) {
      if (!progs[i]) continue;
      const int k = base + i;
      const unsigned char *q = &px[i * 4];
      const int want_r = (int)expect(k);
      if (q[0] != want_r || q[1] != k % 256) {
         if (bad++ < 8) printf("  prog %d: got (%u,%u) want (%d,%d)\n", k, q[0], q[1], want_r, k % 256);
      }
   }
   return bad;
}

int main(int argc, char **argv)
{
   const int n = argc > 1 ? atoi(argv[1]) : 400;
   if (n > 4096) return 2;
   PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   eglInitialize(d, NULL, NULL); eglBindAPI(EGL_OPENGL_ES_API);
   EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
   EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c);
   printf("%s\n", glGetString(GL_RENDERER));
   GLuint tex, fbo; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 64, 64);
   glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   glViewport(0, 0, 64, 64);
   const float v[] = {-1, -1, 3, -1, -1, 3};
   GLuint vb; glGenBuffers(1, &vb); glBindBuffer(GL_ARRAY_BUFFER, vb);
   glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

   GLuint *progs = calloc(n, sizeof *progs);
   for (int i = 0; i < n; i++) progs[i] = make(i);
   int bad = check(progs, n, 0);
   printf("%s: %d programs alive, %d wrong pixels\n", bad ? "FAIL" : "ok", n, bad);
   /* reclaim: delete every other program, draw (flush frees), relink new ones */
   for (int i = 0; i < n; i += 2) { glDeleteProgram(progs[i]); progs[i] = 0; }
   glFinish();
   int bad2 = check(progs, n, 0);
   for (int i = 0; i < n; i += 2) progs[i] = make(i);
   bad2 += check(progs, n, 0);
   printf("%s: after delete/relink of %d, %d wrong pixels\n", bad2 ? "FAIL" : "ok", n / 2, bad2);
   GLenum e = glGetError();
   printf("%s (GL error 0x%x)\n", bad || bad2 || e ? "FAIL" : "PASS", e);
   return bad || bad2 || e;
}
