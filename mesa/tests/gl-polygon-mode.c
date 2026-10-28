/*
 * GL_NV_polygon_mode on nvrm (ES 3.1 context).
 *
 * Why: Minecraft Bedrock (RenderDragon/bgfx, libminecraftpe.so imports the name
 * "glPolygonMode" and no polygon_mode extension string) calls
 * glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) once per frame on GLES. Mesa's
 * eglGetProcAddress returns the PolygonMode dispatch stub, which on ES had no
 * implementation: "GL_INVALID_OPERATION in glPolygonMode(invalid call)" per frame
 * (2026-10-15, ~1,034 per 75 s). With NV_polygon_mode the slot is installed on ES.
 *
 * Checks (analytic, no comparator driver: llvmpipe does not expose the extension):
 *   1. GL_NV_polygon_mode listed; glPolygonModeNV and glPolygonMode resolve.
 *   2. MC's call glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) raises no error.
 *   3. GL_FRONT / GL_BACK -> GL_INVALID_ENUM (extension: FRONT_AND_BACK only).
 *   4. Triangle in 64x64 RGBA8 FBO: FILL covers the interior; LINE leaves the
 *      interior empty but draws edges; POINT draws only the 3 vertex pixels.
 *   5. glGetIntegerv(GL_POLYGON_MODE_NV) == mode; POLYGON_OFFSET_{POINT,LINE}_NV
 *      enable/IsEnabled without error.
 *
 *   cc -std=c11 -I<mesa>/include gl-polygon-mode.c -L<build>/src/egl -lEGL \
 *      -L<build>/src/mesa/glapi/es2api -lGLESv2 -o gl-polygon-mode
 *   LIBGL_DRIVERS_PATH=<build>/src/gallium/targets/dri ./gl-polygon-mode
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <stdio.h>
#include <string.h>

#ifndef GL_POLYGON_MODE_NV
#define GL_POLYGON_MODE_NV 0x0B40
#define GL_POLYGON_OFFSET_POINT_NV 0x2A01
#define GL_POLYGON_OFFSET_LINE_NV 0x2A02
#define GL_POINT_NV 0x1B00
#define GL_LINE_NV 0x1B01
#define GL_FILL_NV 0x1B02
#endif
#define GL_FRONT_ 0x0404

typedef void (*polygon_mode_fn)(GLenum, GLenum);
static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static const char *vs = "#version 310 es\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0,1);gl_PointSize=1.0;}\n";
static const char *fs = "#version 310 es\nprecision mediump float;\nout vec4 c;\nvoid main(){c=vec4(1);}\n";

static GLuint sh(GLenum t, const char *s)
{
   GLuint o = glCreateShader(t); glShaderSource(o, 1, &s, NULL); glCompileShader(o); return o;
}

static int count_lit(const unsigned char *px, int x0, int y0, int x1, int y1)
{
   int n = 0;
   for (int y = y0; y < y1; y++)
      for (int x = x0; x < x1; x++)
         n += px[(y * 64 + x) * 4] != 0;
   return n;
}

static void draw(polygon_mode_fn pm, GLenum mode, unsigned char *px)
{
   pm(GL_FRONT_AND_BACK, mode);
   glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, px);
}

int main(void)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   eglInitialize(d, NULL, NULL);
   eglBindAPI(EGL_OPENGL_ES_API);
   EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
   EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
   if (!c || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) { printf("FAIL: context\n"); return 1; }
   printf("%s / %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

   int has = 0; GLint n; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
   for (int i = 0; i < n; i++) has |= !strcmp((const char *)glGetStringi(GL_EXTENSIONS, i), "GL_NV_polygon_mode");
   CHECK(has, "GL_NV_polygon_mode exposed");
   polygon_mode_fn nv = (polygon_mode_fn)eglGetProcAddress("glPolygonModeNV");
   polygon_mode_fn core = (polygon_mode_fn)eglGetProcAddress("glPolygonMode");
   CHECK(nv && core, "glPolygonModeNV %p / glPolygonMode %p resolve", (void *)nv, (void *)core);
   if (!nv || !core) return 1;

   while (glGetError()) ;
   core(GL_FRONT_AND_BACK, GL_FILL_NV);
   CHECK(glGetError() == GL_NO_ERROR, "glPolygonMode(FRONT_AND_BACK, FILL) (Minecraft's call) no error");
   nv(GL_FRONT_, GL_LINE_NV);
   CHECK(glGetError() == GL_INVALID_ENUM, "glPolygonModeNV(GL_FRONT) -> GL_INVALID_ENUM");
   nv(GL_FRONT_AND_BACK, GL_FILL_RECTANGLE_NV);
   CHECK(glGetError() == GL_INVALID_ENUM, "mode FILL_RECTANGLE_NV -> GL_INVALID_ENUM");

   GLuint tex, fbo; glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 64, 64);
   glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   glViewport(0, 0, 64, 64);
   GLuint prog = glCreateProgram();
   glAttachShader(prog, sh(GL_VERTEX_SHADER, vs)); glAttachShader(prog, sh(GL_FRAGMENT_SHADER, fs));
   glBindAttribLocation(prog, 0, "p"); glLinkProgram(prog); glUseProgram(prog);
   /* pixel-centre vertices (8,8) (56,8) (8,56) */
   const float v[] = {-0.734375f, -0.734375f, 0.765625f, -0.734375f, -0.734375f, 0.765625f};
   GLuint vb; glGenBuffers(1, &vb); glBindBuffer(GL_ARRAY_BUFFER, vb);
   glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

   static unsigned char px[64 * 64 * 4];
   draw(nv, GL_FILL_NV, px);
   int fill_in = count_lit(px, 14, 14, 30, 30), fill_all = count_lit(px, 0, 0, 64, 64);
   CHECK(fill_in == 256 && fill_all > 1000, "FILL: interior 16x16 lit %d/256, total %d", fill_in, fill_all);
   draw(nv, GL_LINE_NV, px);
   int line_in = count_lit(px, 14, 14, 30, 30), line_all = count_lit(px, 0, 0, 64, 64);
   CHECK(line_in == 0 && line_all >= 100 && line_all < 250, "LINE: interior lit %d (0), total %d (edges ~144)", line_in, line_all);
   GLint mode = 0; glGetIntegerv(GL_POLYGON_MODE_NV, &mode);
   CHECK(mode == GL_LINE_NV, "glGetIntegerv(POLYGON_MODE_NV) = 0x%x", mode);
   draw(nv, GL_POINT_NV, px);
   int pt_all = count_lit(px, 0, 0, 64, 64);
   CHECK(pt_all == 3 && px[(8 * 64 + 8) * 4] && px[(8 * 64 + 56) * 4] && px[(56 * 64 + 8) * 4],
         "POINT: %d lit (3 vertices)", pt_all);
   core(GL_FRONT_AND_BACK, GL_FILL_NV);

   glEnable(GL_POLYGON_OFFSET_POINT_NV); glEnable(GL_POLYGON_OFFSET_LINE_NV);
   CHECK(glIsEnabled(GL_POLYGON_OFFSET_LINE_NV) && glIsEnabled(GL_POLYGON_OFFSET_POINT_NV) &&
         glGetError() == GL_NO_ERROR, "POLYGON_OFFSET_POINT/LINE_NV enable");
   glDisable(GL_POLYGON_OFFSET_POINT_NV); glDisable(GL_POLYGON_OFFSET_LINE_NV);
   CHECK(glGetError() == GL_NO_ERROR, "no GL error at end");
   printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
   return failures != 0;
}
