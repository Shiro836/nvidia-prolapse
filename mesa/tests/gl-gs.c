/* gl-gs.c: geometry shaders (OES/EXT_geometry_shader) on an ES 3.1 context.
 *
 * Cases (each checked analytically, and the raw pixels go to an optional
 * dump file so two drivers can be compared byte for byte):
 *   points   point -> 8x8 quad expansion, colour from gl_PrimitiveIDIn
 *   layered  layout(invocations = 4), gl_Layer = gl_InvocationID into a
 *            4-layer 2D-array FBO attached with glFramebufferTexture
 *   tri_adj  GL_TRIANGLES_ADJACENCY: colour encodes the ids of the three
 *            adjacent vertices (1, 3, 5)
 *   line_adj GL_LINES_ADJACENCY: colour encodes the ids of vertices 0..3
 *   strip_adj GL_TRIANGLE_STRIP_ADJACENCY, 2 triangles: per-primitive
 *            check of the adjacency vertex order the spec defines
 *   cube_arr samplerCubeArray in a fragment shader (#extension
 *            GL_EXT_texture_cube_map_array : require)
 *
 * usage: gl-gs <label> [dump-file]
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_GEOMETRY_SHADER_EXT
#define GL_GEOMETRY_SHADER_EXT 0x8DD9
#endif
#ifndef GL_LINES_ADJACENCY_EXT
#define GL_LINES_ADJACENCY_EXT 0x000A
#define GL_TRIANGLES_ADJACENCY_EXT 0x000C
#define GL_TRIANGLE_STRIP_ADJACENCY_EXT 0x000D
#endif
#ifndef GL_TEXTURE_CUBE_MAP_ARRAY_EXT
#define GL_TEXTURE_CUBE_MAP_ARRAY_EXT 0x9009
#endif
#ifndef GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT
#define GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT 0x8DE0
#define GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS_EXT 0x8DE1
#define GL_MAX_GEOMETRY_SHADER_INVOCATIONS_EXT 0x8E5A
#endif

typedef void (*PFNFBTEX)(GLenum, GLenum, GLuint, GLint);

#define W 64
static int failures;
static FILE *dump;

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static GLuint
shader(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("compile failed:\n%s\n%s\n", src, log);
      exit(2);
   }
   return s;
}

static GLuint
program(const char *vs, const char *gs, const char *fs)
{
   GLuint p = glCreateProgram();
   glAttachShader(p, shader(GL_VERTEX_SHADER, vs));
   if (gs)
      glAttachShader(p, shader(GL_GEOMETRY_SHADER_EXT, gs));
   glAttachShader(p, shader(GL_FRAGMENT_SHADER, fs));
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      printf("link failed: %s\n", log);
      exit(2);
   }
   return p;
}

#define HDR "#version 310 es\n#extension GL_EXT_geometry_shader : require\n"

static const char *vs_src =
   "#version 310 es\n"
   "layout(location = 0) in vec4 pos;\n"
   "out float vid;\n"
   "void main() { gl_Position = vec4(pos.xy, 0.0, 1.0); vid = pos.z; }\n";

static const char *fs_src =
   "#version 310 es\nprecision highp float;\n"
   "in vec4 col;\nout vec4 o;\nvoid main() { o = col; }\n";

static GLuint fbo2d, tex2d;

static void
read_rgba(uint8_t *px, const char *name)
{
   glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, px);
   if (dump) {
      fprintf(dump, "%s\n", name);
      fwrite(px, 1, W * W * 4, dump);
   }
}

static void
draw_setup(const float *verts, int n)
{
   static GLuint vbo;
   if (!vbo)
      glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, n * 4 * sizeof(float), verts, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo2d);
   glViewport(0, 0, W, W);
   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT);
}

static void
case_points(void)
{
   const char *gs =
      HDR "layout(points) in;\nlayout(triangle_strip, max_vertices = 4) out;\n"
      "out vec4 col;\n"
      "void main() {\n"
      "  vec4 c = vec4(float(gl_PrimitiveIDIn + 1) / 8.0, 0.25, 1.0, 1.0);\n"
      "  vec2 h = vec2(4.0 / 32.0);\n" /* 4 px half size on a 64 px target */
      "  for (int i = 0; i < 4; i++) {\n"
      "    vec2 o = vec2(i & 1, i >> 1) * 2.0 - 1.0;\n"
      "    gl_Position = gl_in[0].gl_Position + vec4(o * h, 0.0, 0.0);\n"
      "    col = c; EmitVertex();\n"
      "  }\n"
      "  EndPrimitive();\n"
      "}\n";
   GLuint p = program(vs_src, gs, fs_src);
   glUseProgram(p);
   /* centres on pixel corners: (8+16k)/64 -> NDC */
   float v[4 * 4];
   for (int k = 0; k < 4; k++) {
      v[k * 4 + 0] = (16.0f * k + 8.0f) / 32.0f - 1.0f;
      v[k * 4 + 1] = (16.0f * k + 8.0f) / 32.0f - 1.0f;
      v[k * 4 + 2] = k;
      v[k * 4 + 3] = 1;
   }
   draw_setup(v, 4);
   glDrawArrays(GL_POINTS, 0, 4);
   uint8_t px[W * W * 4];
   read_rgba(px, "points");
   int lit = 0, bad = 0;
   for (int y = 0; y < W; y++)
      for (int x = 0; x < W; x++) {
         const uint8_t *c = &px[(y * W + x) * 4];
         int k = -1;
         for (int j = 0; j < 4; j++)
            if (x >= 16 * j + 4 && x < 16 * j + 12 && y >= 16 * j + 4 && y < 16 * j + 12)
               k = j;
         if (k < 0) {
            bad += c[3] != 0;
         } else {
            lit++;
            const int r = (int)((k + 1) / 8.0 * 255 + 0.5);
            bad += abs(c[0] - r) > 1 || abs(c[1] - 64) > 1 || c[2] != 255 || c[3] != 255;
         }
      }
   CHECK(bad == 0, "points: %d of %d pixels wrong (expected 4 quads of 8x8)", bad, W * W);
   if (!bad)
      printf("PASS: points (4 points -> 4 quads, %d px, gl_PrimitiveIDIn)\n", lit);
}

static void
case_layered(PFNFBTEX fbtex)
{
   const char *gs =
      HDR "layout(triangles, invocations = 4) in;\n"
      "layout(triangle_strip, max_vertices = 3) out;\n"
      "out vec4 col;\n"
      "void main() {\n"
      "  for (int i = 0; i < 3; i++) {\n"
      "    gl_Position = gl_in[i].gl_Position;\n"
      "    gl_Layer = gl_InvocationID;\n"
      "    col = vec4(float(gl_InvocationID + 1) / 4.0, 0.5, 0.0, 1.0);\n"
      "    EmitVertex();\n"
      "  }\n"
      "}\n";
   GLuint p = program(vs_src, gs, fs_src);
   GLuint arr, fbo, rfbo;
   glGenTextures(1, &arr);
   glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
   glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, W, W, 4);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   fbtex(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, 0);
   CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
         "layered: framebuffer incomplete 0x%x", glCheckFramebufferStatus(GL_FRAMEBUFFER));
   glViewport(0, 0, W, W);
   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT); /* clears all 4 layers */
   const float v[] = { -1, -1, 0, 1,  3, -1, 1, 1,  -1, 3, 2, 1 };
   static GLuint vbo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glUseProgram(p);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glGenFramebuffers(1, &rfbo);
   glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
   int bad_layers = 0;
   for (int l = 0; l < 4; l++) {
      glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, 0, l);
      uint8_t px[W * W * 4];
      char name[32];
      snprintf(name, sizeof(name), "layered%d", l);
      read_rgba(px, name);
      const int r = (int)((l + 1) / 4.0 * 255 + 0.5);
      int bad = 0;
      for (int i = 0; i < W * W; i++)
         bad += abs(px[i * 4] - r) > 1 || abs(px[i * 4 + 1] - 128) > 1 ||
                px[i * 4 + 2] != 0 || px[i * 4 + 3] != 255;
      CHECK(bad == 0, "layered: layer %d has %d wrong pixels (first %u,%u,%u,%u, want %d,128,0,255)",
            l, bad, px[0], px[1], px[2], px[3], r);
      bad_layers += bad != 0;
   }
   if (!bad_layers)
      printf("PASS: layered (invocations=4, gl_Layer -> 4 layers of a 2D-array FBO)\n");
}

/* One fullscreen triangle per input primitive; colour from vertex ids. */
static void
case_adj(const char *name, const char *layout_in, int n_in, GLenum mode,
         const float *ids, int n, const int *sel, int nsel, int nprims)
{
   char gs[2048];
   char expr[256] = "";
   for (int i = 0; i < nsel; i++)
      snprintf(expr + strlen(expr), sizeof(expr) - strlen(expr), "%s vid_in[%d] / 16.0",
               i ? "," : "", sel[i]);
   snprintf(gs, sizeof(gs),
      HDR "layout(%s) in;\nlayout(triangle_strip, max_vertices = 4) out;\n"
      "in float vid[];\nout vec4 col;\n"
      "void main() {\n"
      "  float vid_in[%d];\n"
      "  for (int i = 0; i < %d; i++) vid_in[i] = vid[i];\n"
      "  vec4 c = vec4(%s%s);\n"
      "  float w = 2.0 / float(%d);\n"
      "  float x0 = -1.0 + w * float(gl_PrimitiveIDIn);\n"
      "  for (int i = 0; i < 4; i++) {\n"
      "    gl_Position = vec4(x0 + w * float(i & 1), float(i >> 1) * 2.0 - 1.0, 0.0, 1.0);\n"
      "    col = c; EmitVertex();\n"
      "  }\n"
      "}\n", layout_in, n_in, n_in, expr, nsel == 3 ? ", 1.0" : "", nprims);
   GLuint p = program(vs_src, gs, fs_src);
   glUseProgram(p);
   float v[16 * 4];
   for (int i = 0; i < n; i++) {
      v[i * 4 + 0] = 0;
      v[i * 4 + 1] = 0;
      v[i * 4 + 2] = ids[i];
      v[i * 4 + 3] = 1;
   }
   draw_setup(v, n);
   glDrawArrays(mode, 0, n);
   uint8_t px[W * W * 4];
   read_rgba(px, name);
}

static int
expect_quad_colors(const char *name, const uint8_t *px, int nprims, const int (*want)[4])
{
   int bad = 0;
   for (int prim = 0; prim < nprims; prim++) {
      const int x = (prim * 2 + 1) * W / (2 * nprims), y = W / 2;
      const uint8_t *c = &px[(y * W + x) * 4];
      for (int k = 0; k < 4; k++)
         if (abs(c[k] - want[prim][k]) > 1) {
            printf("FAIL: %s prim %d: got %u,%u,%u,%u want %d,%d,%d,%d\n", name, prim,
                   c[0], c[1], c[2], c[3], want[prim][0], want[prim][1], want[prim][2],
                   want[prim][3]);
            bad++;
            failures++;
            break;
         }
   }
   return bad;
}

static int
c16(float id)
{
   return (int)(id / 16.0f * 255.0f + 0.5f);
}

static void
case_cube_array(void)
{
   const char *fs =
      "#version 310 es\n#extension GL_EXT_texture_cube_map_array : require\n"
      "precision highp float;\n"
      "uniform highp samplerCubeArray s;\nout vec4 o;\n"
      "void main() { o = texture(s, vec4(1.0, 0.0, 0.0, 1.0)); }\n";
   const char *vs =
      "#version 310 es\nlayout(location = 0) in vec4 pos;\n"
      "void main() { gl_Position = vec4(pos.xy, 0.0, 1.0); }\n";
   GLuint p = program(vs, NULL, fs);
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, t);
   glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, 1, GL_RGBA8, 4, 4, 12);
   uint8_t data[4 * 4 * 4 * 12];
   for (int layer = 0; layer < 12; layer++)
      for (int i = 0; i < 16; i++) {
         uint8_t *d = &data[(layer * 16 + i) * 4];
         d[0] = 10 * layer; d[1] = 200; d[2] = 7; d[3] = 255;
      }
   glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, 0, 0, 0, 0, 4, 4, 12, GL_RGBA,
                   GL_UNSIGNED_BYTE, data);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glUseProgram(p);
   const float v[] = { -1, -1, 0, 1,  3, -1, 0, 1,  -1, 3, 0, 1 };
   draw_setup(v, 3);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   uint8_t px[W * W * 4];
   read_rgba(px, "cube_arr");
   /* +X face of cube 1 = layer-face 6 */
   const uint8_t *c = &px[(W / 2 * W + W / 2) * 4];
   CHECK(c[0] == 60 && c[1] == 200 && c[2] == 7, "cube_arr: got %u,%u,%u want 60,200,7",
         c[0], c[1], c[2]);
   if (c[0] == 60 && c[1] == 200 && c[2] == 7)
      printf("PASS: cube_arr (samplerCubeArray, cube 1 +X)\n");
}

int
main(int argc, char **argv)
{
   const char *label = argc > 1 ? argv[1] : "?";
   if (argc > 2)
      dump = fopen(argv[2], "wb");
   EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   eglInitialize(dpy, NULL, NULL);
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint cfg_attr[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                               EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
   EGLConfig cfg;
   EGLint n;
   eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n);
   const EGLint ctx_attr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
   const EGLint pb[] = { EGL_WIDTH, W, EGL_HEIGHT, W, EGL_NONE };
   EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pb);
   if (!ctx || !eglMakeCurrent(dpy, surf, surf, ctx)) {
      printf("RESULT: FAIL (no OpenGL ES 3.1 context)\n");
      return 1;
   }
   printf("%s: %s / %s\n", label, glGetString(GL_RENDERER), glGetString(GL_VERSION));
   const char *ext = (const char *)glGetString(GL_EXTENSIONS);
   const int has_gs = strstr(ext, "GL_EXT_geometry_shader") || strstr(ext, "GL_OES_geometry_shader");
   const int has_ca = strstr(ext, "GL_EXT_texture_cube_map_array") != NULL;
   printf("GL_EXT_geometry_shader %d, GL_EXT_texture_cube_map_array %d\n", has_gs, has_ca);
   if (!has_gs || !has_ca) {
      printf("RESULT: FAIL (extensions missing)\n");
      return 1;
   }
   GLint mov, mtoc, minv;
   glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &mov);
   glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS_EXT, &mtoc);
   glGetIntegerv(GL_MAX_GEOMETRY_SHADER_INVOCATIONS_EXT, &minv);
   printf("limits: max output vertices %d, total output components %d, invocations %d\n",
          mov, mtoc, minv);
   CHECK(mov >= 256 && mtoc >= 1024 && minv >= 32, "limits below the EXT_geometry_shader minimums");

   glGenTextures(1, &tex2d);
   glBindTexture(GL_TEXTURE_2D, tex2d);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
   glGenFramebuffers(1, &fbo2d);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo2d);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex2d, 0);

   case_points();
   PFNFBTEX fbtex = (PFNFBTEX)eglGetProcAddress("glFramebufferTextureEXT");
   CHECK(fbtex != NULL, "glFramebufferTextureEXT missing");
   if (fbtex)
      case_layered(fbtex);

   uint8_t px[W * W * 4];
   /* triangles_adjacency: 6 vertices, adjacent ones are 1, 3, 5. */
   {
      const float ids[] = { 1, 2, 3, 4, 5, 6 };
      const int sel[] = { 1, 3, 5 };
      case_adj("tri_adj", "triangles_adjacency", 6, GL_TRIANGLES_ADJACENCY_EXT, ids, 6, sel, 3, 1);
      glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, px);
      const int want[1][4] = { { c16(2), c16(4), c16(6), 255 } };
      if (!expect_quad_colors("tri_adj", px, 1, want))
         printf("PASS: tri_adj (adjacent vertices 1,3,5 delivered)\n");
   }
   /* lines_adjacency: 2 primitives of 4 vertices; colour = ids 0..3. */
   {
      const float ids[] = { 1, 2, 3, 4, 9, 10, 11, 12 };
      const int sel[] = { 0, 1, 2, 3 };
      case_adj("line_adj", "lines_adjacency", 4, GL_LINES_ADJACENCY_EXT, ids, 8, sel, 4, 2);
      glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, px);
      const int want[2][4] = { { c16(1), c16(2), c16(3), c16(4) },
                               { c16(9), c16(10), c16(11), c16(12) } };
      if (!expect_quad_colors("line_adj", px, 2, want))
         printf("PASS: line_adj (2 primitives x 4 vertices)\n");
   }
   /* triangle_strip_adjacency, 8 vertices = 2 triangles (EXT_geometry_shader
    * table "Triangles generated by triangle strips with adjacency", n = 2):
    * tri 0 = vertices 0, 2, 4; tri 1 (last, odd) = 4, 2, 6. Colour = the
    * ids of gl_in[0], gl_in[2], gl_in[4]. */
   {
      const float ids[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
      const int sel[] = { 0, 2, 4 };
      case_adj("strip_adj", "triangles_adjacency", 6, GL_TRIANGLE_STRIP_ADJACENCY_EXT, ids, 8, sel, 3, 2);
      glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, px);
      /* tri 0: 0,2,4 -> ids 1,3,5; tri 1 (last, odd): 2i+2=4, 2i=2, 2i+4=6 -> ids 5,3,7 */
      const int want[2][4] = { { c16(1), c16(3), c16(5), 255 },
                               { c16(5), c16(3), c16(7), 255 } };
      if (!expect_quad_colors("strip_adj", px, 2, want))
         printf("PASS: strip_adj (2 triangles, vertex order per the strip table)\n");
   }
   case_cube_array();

   GLenum err = glGetError();
   CHECK(err == GL_NO_ERROR, "GL error 0x%x", err);
   if (dump)
      fclose(dump);
   printf("RESULT: %s (%d)\n", failures ? "FAIL" : "PASS", failures);
   return failures != 0;
}
