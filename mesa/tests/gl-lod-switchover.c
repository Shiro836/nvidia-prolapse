/*
 * Measures the minification/magnification switch-over point c of the GL
 * implementation for explicit-LOD vertex-shader lookups (texture2DLod and
 * textureCubeLod). GL 2.x / ES 2.0 allow (GL 3.1+: "may") c = 0.5 when MAG
 * is LINEAR and MIN is NEAREST_MIPMAP_*; dEQP's reference sampler uses c = 0.
 *
 * Level 0 of every face/texture is a column pattern black|white|black|white
 * (4x4); the lookup point sits on the black/white texel boundary, so
 * NEAREST gives 0 or 255 and LINEAR gives ~128. Level 1+ are solid green.
 * Each output pixel is one point with its own LOD.
 *
 * Build: same line as gl-g0a-clear.c (README). Run: ./gl-lod-switchover
 * Output: one line per (target, min, mag, lod) with the red channel.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
      fprintf(stderr, "FAIL: shader: %s\n", log);
      exit(1);
   }
   return s;
}

static GLuint
program(const char *vs, const char *fs)
{
   GLuint p = glCreateProgram();
   glAttachShader(p, shader(GL_VERTEX_SHADER, vs));
   glAttachShader(p, shader(GL_FRAGMENT_SHADER, fs));
   glBindAttribLocation(p, 0, "a_pos");
   glBindAttribLocation(p, 1, "a_lod");
   glLinkProgram(p);
   return p;
}

static void
upload_levels(GLenum target)
{
   unsigned char l0[4 * 4 * 4], l1[2 * 2 * 4], l2[4];
   for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++) {
         unsigned char v = (x & 1) ? 255 : 0;
         unsigned char *p = &l0[(y * 4 + x) * 4];
         p[0] = v; p[1] = v; p[2] = v; p[3] = 255;
      }
   for (int i = 0; i < 4; i++) {
      unsigned char *p = &l1[i * 4];
      p[0] = 0; p[1] = 255; p[2] = 0; p[3] = 255;
   }
   l2[0] = 0; l2[1] = 255; l2[2] = 0; l2[3] = 255;
   glTexImage2D(target, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, l0);
   glTexImage2D(target, 1, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, l1);
   glTexImage2D(target, 2, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, l2);
}

int
main(void)
{
   EGLDisplay dpy = EGL_NO_DISPLAY;
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   if (get_display)
      dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   EGLint major, minor;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &major, &minor)) {
      dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (!eglInitialize(dpy, &major, &minor)) {
         fprintf(stderr, "FAIL: eglInitialize (0x%x)\n", eglGetError());
         return 1;
      }
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   static const EGLint cfg_attribs[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                         EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                                         EGL_RED_SIZE, 8, EGL_NONE };
   EGLConfig cfg;
   EGLint n = 0;
   eglChooseConfig(dpy, cfg_attribs, &cfg, 1, &n);
   static const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, n ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attribs);
   EGLSurface surf = EGL_NO_SURFACE;
   if (n) {
      static const EGLint pb[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
      surf = eglCreatePbufferSurface(dpy, cfg, pb);
   }
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, surf, surf, ctx)) {
      fprintf(stderr, "FAIL: context (0x%x)\n", eglGetError());
      return 1;
   }
   printf("GL_RENDERER %s\n", glGetString(GL_RENDERER));

   GLuint rb, fbo;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, 16, 1);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "FAIL: fbo\n");
      return 1;
   }
   glViewport(0, 0, 16, 1);

   static const char *fs =
      "precision mediump float; varying vec4 v; void main() { gl_FragColor = v; }\n";
   /* x = 0.25: boundary between texel 0 (black) and texel 1 (white). */
   GLuint p2d = program(
      "attribute vec2 a_pos; attribute float a_lod; uniform sampler2D t; varying vec4 v;\n"
      "void main() { v = texture2DLod(t, vec2(0.25, 0.375), a_lod);\n"
      "  gl_Position = vec4(a_pos, 0.0, 1.0); gl_PointSize = 1.0; }\n", fs);
   /* +Z face: s = x/|z|, t = -y/|z| mapped to [0,1]: x = -0.5 -> s = 0.25. */
   GLuint pcube = program(
      "attribute vec2 a_pos; attribute float a_lod; uniform samplerCube t; varying vec4 v;\n"
      "void main() { v = textureCubeLod(t, vec3(-0.5, -0.25, 1.0), a_lod);\n"
      "  gl_Position = vec4(a_pos, 0.0, 1.0); gl_PointSize = 1.0; }\n", fs);

   GLuint tex2d, texcube;
   glGenTextures(1, &tex2d);
   glBindTexture(GL_TEXTURE_2D, tex2d);
   upload_levels(GL_TEXTURE_2D);
   glGenTextures(1, &texcube);
   glBindTexture(GL_TEXTURE_CUBE_MAP, texcube);
   for (int f = 0; f < 6; f++)
      upload_levels(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f);

   static const float lods[] = { -0.5f, 0.0f, 0.1f, 0.19f, 0.3f, 0.45f, 0.49f, 0.51f, 0.6f, 0.9f, 1.2f };
   const int nl = sizeof(lods) / sizeof(lods[0]);
   float verts[16 * 3];
   for (int i = 0; i < nl; i++) {
      verts[i * 3 + 0] = (i + 0.5f) / 8.0f - 1.0f;
      verts[i * 3 + 1] = 0.0f;
      verts[i * 3 + 2] = lods[i];
   }
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 12, verts);
   glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 12, verts + 2);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);

   static const struct { GLenum min, mag; const char *name; } modes[] = {
      { GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR,  "nearest_mipmap_nearest/linear" },
      { GL_NEAREST_MIPMAP_LINEAR,  GL_LINEAR,  "nearest_mipmap_linear/linear" },
      { GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST, "nearest_mipmap_nearest/nearest" },
      { GL_LINEAR_MIPMAP_NEAREST,  GL_NEAREST, "linear_mipmap_nearest/nearest" },
   };
   for (int t = 0; t < 2; t++) {
      GLenum target = t ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
      glUseProgram(t ? pcube : p2d);
      for (unsigned m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
         glTexParameteri(target, GL_TEXTURE_MIN_FILTER, modes[m].min);
         glTexParameteri(target, GL_TEXTURE_MAG_FILTER, modes[m].mag);
         glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
         glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
         glClearColor(0, 0, 1, 1);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawArrays(GL_POINTS, 0, nl);
         unsigned char px[16 * 4];
         glReadPixels(0, 0, 16, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
         printf("%-4s %-32s", t ? "cube" : "2d", modes[m].name);
         for (int i = 0; i < nl; i++)
            printf(" %5.2f:%3u/%3u", lods[i], px[i * 4], px[i * 4 + 1]);
         printf("\n");
      }
   }
   printf("glGetError 0x%x\n", glGetError());
   return 0;
}
