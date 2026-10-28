/* SPDX-License-Identifier: MIT
 *
 * D4: border colour with texture swizzles, luminance/alpha formats and sRGB
 * (OES_texture_border_clamp, GL_CLAMP_TO_BORDER). The sampler's border
 * colour is returned without the texture header's swizzle, so it must be
 * pre-swizzled per view (NVK nvk_sampler.c:250-283; nvrm uses st's
 * PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_NV50 like nvc0).
 *
 *   gl-border-swizzle <out.rgba> [ref.rgba]
 *
 * One pixel per case: a fragment shader samples outside [0,1] (the border)
 * of a 4x4 texture; GL_RGBA8 output. With ref (llvmpipe), prints each
 * differing case and exits 1 if any channel differs by more than 1 LSB.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GL_CLAMP_TO_BORDER_OES 0x812D
#define GL_TEXTURE_BORDER_COLOR_OES 0x1004

struct tcase {
   const char *name;
   GLenum ifmt, fmt, type;
   GLint swz[4];
   bool integer;
};

#define ID { GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA }
static const struct tcase cases[] = {
   { "rgba8 id",          GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ID },
   { "rgba8 bgra",        GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, { GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA } },
   { "rgba8 a1r0",        GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, { GL_ALPHA, GL_ONE, GL_RED, GL_ZERO } },
   { "rgba8 gbar",        GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, { GL_GREEN, GL_BLUE, GL_ALPHA, GL_RED } },
   { "r8 rrr1",           GL_R8, GL_RED, GL_UNSIGNED_BYTE, { GL_RED, GL_RED, GL_RED, GL_ONE } },
   { "rg8 ggr1",          GL_RG8, GL_RG, GL_UNSIGNED_BYTE, { GL_GREEN, GL_GREEN, GL_RED, GL_ONE } },
   { "lum id",            GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE, ID },
   { "alpha id",          GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE, ID },
   { "lumalpha id",       GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, ID },
   { "lumalpha agr1",     GL_LUMINANCE_ALPHA, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, { GL_ALPHA, GL_GREEN, GL_RED, GL_ONE } },
   { "alpha aaa0",        GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE, { GL_ALPHA, GL_ALPHA, GL_ALPHA, GL_ZERO } },
   { "srgb8a8 id",        GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ID },
   { "srgb8a8 bgra",      GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, { GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA } },
   { "srgb8a8 aaaa",      GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, { GL_ALPHA, GL_ALPHA, GL_ALPHA, GL_ALPHA } },
   { "rgba16f bgra",      GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, { GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA } },
};
#define N (int)(sizeof(cases) / sizeof(cases[0]))

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

int
main(int argc, char **argv)
{
   if (argc < 2)
      return 2;
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
   if (!strstr((const char *)glGetString(GL_EXTENSIONS), "GL_OES_texture_border_clamp")) {
      printf("FAIL: no GL_OES_texture_border_clamp\n");
      return 1;
   }
   GLuint prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER,
      "#version 300 es\nlayout(location=0) in vec2 p;\n"
      "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n"));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER,
      "#version 300 es\nprecision highp float;\nuniform sampler2D t;\nout vec4 o;\n"
      "void main() { o = texture(t, vec2(1.75, -0.75)); }\n"));
   glLinkProgram(prog);
   glUseProgram(prog);

   GLuint rt, fbo;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, N, 1);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);

   static const float quad[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, quad);
   glEnableVertexAttribArray(0);
   const float border[4] = { 0.125f, 0.5f, 0.75f, 0.25f };
   static const unsigned char texels[4 * 4 * 4] = { 0 };
   static const unsigned short texels16[4 * 4 * 4] = { 0 };
   for (int i = 0; i < N; i++) {
      GLuint tex;
      glGenTextures(1, &tex);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexImage2D(GL_TEXTURE_2D, 0, cases[i].ifmt, 4, 4, 0, cases[i].fmt, cases[i].type,
                   cases[i].type == GL_HALF_FLOAT ? (const void *)texels16 : texels);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_OES);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_OES);
      glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR_OES, border);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, cases[i].swz[0]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, cases[i].swz[1]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, cases[i].swz[2]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, cases[i].swz[3]);
      glViewport(i, 0, 1, 1);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      if (glGetError() != GL_NO_ERROR)
         printf("FAIL: GL error in case %s\n", cases[i].name);
   }
   unsigned char px[N * 4];
   glReadPixels(0, 0, N, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
   FILE *f = fopen(argv[1], "wb");
   fwrite(px, 1, sizeof(px), f);
   fclose(f);
   int bad = 0;
   unsigned char ref[N * 4];
   if (argc > 2) {
      f = fopen(argv[2], "rb");
      if (!f || fread(ref, 1, sizeof(ref), f) != sizeof(ref)) {
         printf("FAIL: reference\n");
         return 1;
      }
      fclose(f);
   }
   for (int i = 0; i < N; i++) {
      int maxd = 0;
      for (int c = 0; c < 4 && argc > 2; c++)
         maxd = abs(px[i * 4 + c] - ref[i * 4 + c]) > maxd ? abs(px[i * 4 + c] - ref[i * 4 + c]) : maxd;
      printf("%-16s %3u %3u %3u %3u", cases[i].name, px[i * 4], px[i * 4 + 1], px[i * 4 + 2], px[i * 4 + 3]);
      if (argc > 2)
         printf("   ref %3u %3u %3u %3u  %s", ref[i * 4], ref[i * 4 + 1], ref[i * 4 + 2], ref[i * 4 + 3],
                maxd > 1 ? "DIFF" : "ok");
      printf("\n");
      bad += maxd > 1;
   }
   if (argc > 2)
      printf("%s (%d/%d cases differ)\n", bad ? "FAIL" : "PASS", bad, N);
   return bad ? 1 : 0;
}
