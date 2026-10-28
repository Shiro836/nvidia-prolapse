/* SPDX-License-Identifier: MIT
 *
 * Shader features of SurfaceFlinger's RenderEngine (Skia GL backend,
 * AOSP 13) on the nvrm gallium driver, compared against llvmpipe.
 *
 *   gl-skia-shaders <vendor-substring> <outdir> [<refdir>] [case-substring]
 *
 * Each case is a GLSL ES program written the way Skia's GrGLSLProgramBuilder
 * emits it (sk_RTAdjust vertex transform, sk_FragCoord with u_skRTFlip,
 * mediump/highp mixes, samplerExternalOES for gralloc buffers, Skia's
 * integer dither, derivative-based AA with discard, mip sharpening bias,
 * RenderEngine's linear-effect transfer functions). It is drawn as a
 * full-screen triangle into a 64x64 RGBA8 FBO and read back to
 * <outdir>/<case>.rgba. With <refdir> (written by the same program on
 * llvmpipe) every case is compared with a per-channel tolerance (external_size_lod
 * with its analytic result instead: llvmpipe is wrong there); exit 0 only
 * if every case compiled, linked, drew without a GL error and matched.
 * Without <refdir> it only writes the images (reference run).
 *
 * Reference (host system Mesa, llvmpipe):
 *   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
 *   LIBGL_ALWAYS_SOFTWARE=1 ./gl-skia-shaders-sys llvmpipe ref
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 64
#define TOL 3

static PFNEGLCREATEIMAGEKHRPROC create_image;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target;
static EGLDisplay dpy;
static EGLContext ctx;
static GLuint tex2d, tex_mip, tex_ext;

/* Skia's vertex stage: device-space position through sk_RTAdjust, local
 * coordinates through a 3x3 matrix (GrGLSLGeometryProcessor). */
static const char *vs_es3 =
   "#version 300 es\n"
   "precision mediump float;\n"
   "precision mediump sampler2D;\n"
   "uniform highp vec4 sk_RTAdjust;\n"
   "uniform highp mat3 umatrix_S1_c0;\n"
   "in highp vec2 position;\n"
   "out highp vec2 vTransformedCoords_3_S0;\n"
   "out mediump vec4 vcolor_S0;\n"
   "void main() {\n"
   "  vcolor_S0 = vec4(position.x / 64.0, position.y / 64.0, 0.5, 1.0);\n"
   "  vTransformedCoords_3_S0 = (umatrix_S1_c0 * vec3(position, 1.0)).xy;\n"
   "  gl_Position = vec4(position, 0.0, 1.0);\n"
   "  gl_Position = vec4(gl_Position.xy * sk_RTAdjust.xz + gl_Position.ww * sk_RTAdjust.yw, 0.0, gl_Position.w);\n"
   "}\n";

static const char *vs_es1 =
   "#version 100\n"
   "precision mediump float;\n"
   "uniform highp vec4 sk_RTAdjust;\n"
   "uniform highp mat3 umatrix_S1_c0;\n"
   "attribute highp vec2 position;\n"
   "varying highp vec2 vTransformedCoords_3_S0;\n"
   "varying mediump vec4 vcolor_S0;\n"
   "void main() {\n"
   "  vcolor_S0 = vec4(position.x / 64.0, position.y / 64.0, 0.5, 1.0);\n"
   "  vTransformedCoords_3_S0 = (umatrix_S1_c0 * vec3(position, 1.0)).xy;\n"
   "  gl_Position = vec4(position.x * sk_RTAdjust.x + sk_RTAdjust.y, position.y * sk_RTAdjust.z + sk_RTAdjust.w, 0.0, 1.0);\n"
   "}\n";

#define FS_ES3_HEAD \
   "#version 300 es\n" \
   "precision mediump float;\n" \
   "precision mediump sampler2D;\n" \
   "uniform highp vec2 u_skRTFlip;\n" \
   "in highp vec2 vTransformedCoords_3_S0;\n" \
   "in mediump vec4 vcolor_S0;\n" \
   "out mediump vec4 sk_FragColor;\n"

static const struct skcase {
   const char *name;
   bool es1;
   int texture; /* 0 none, 1 2D, 2 2D mipmapped, 3 external */
   const char *fs;
   bool analytic; /* compare with expected_size_fetch(), not the reference */
} cases[] = {
   { "color", false, 0, FS_ES3_HEAD
     "void main() { sk_FragColor = vcolor_S0; }\n", false },
   { "texture2d", false, 1, FS_ES3_HEAD
     "uniform sampler2D uTextureSampler_0_S0;\n"
     "void main() { sk_FragColor = texture(uTextureSampler_0_S0, vTransformedCoords_3_S0) * vcolor_S0.w; }\n", false },
   /* RenderEngine samples gralloc buffers imported as EGLImages through
    * GL_TEXTURE_EXTERNAL_OES (SkiaGLRenderEngine::mapExternalTextureBuffer). */
   { "external_essl3", false, 3,
     "#version 300 es\n"
     "#extension GL_OES_EGL_image_external_essl3 : require\n"
     "precision mediump float;\n"
     "uniform highp vec2 u_skRTFlip;\n"
     "uniform mediump samplerExternalOES uTextureSampler_0_S0;\n"
     "in highp vec2 vTransformedCoords_3_S0;\n"
     "in mediump vec4 vcolor_S0;\n"
     "out mediump vec4 sk_FragColor;\n"
     "void main() {\n"
     "  mediump vec4 _0 = texture(uTextureSampler_0_S0, vTransformedCoords_3_S0);\n"
     "  sk_FragColor = vec4(_0.xyz, 1.0) * vcolor_S0.w;\n"
     "}\n", false },
   { "external_essl1", true, 3,
     "#version 100\n"
     "#extension GL_OES_EGL_image_external : require\n"
     "precision mediump float;\n"
     "uniform mediump samplerExternalOES uTextureSampler_0_S0;\n"
     "varying highp vec2 vTransformedCoords_3_S0;\n"
     "varying mediump vec4 vcolor_S0;\n"
     "void main() { gl_FragColor = texture2D(uTextureSampler_0_S0, vTransformedCoords_3_S0); }\n", false },
   { "external_size_lod", false, 3,
     "#version 300 es\n"
     "#extension GL_OES_EGL_image_external_essl3 : require\n"
     "precision highp float;\n"
     "uniform samplerExternalOES s;\n"
     "in highp vec2 vTransformedCoords_3_S0;\n"
     "out vec4 o;\n"
     "void main() {\n"
     "  ivec2 sz = textureSize(s, 0);\n"
     "  o = texelFetch(s, ivec2(gl_FragCoord.xy) % sz, 0) * 0.5 + vec4(vec2(sz) / 256.0, 0.0, 0.5);\n"
     "}\n",
     /* llvmpipe (Mesa 26.2.4) returns textureSize (64, 0) and a zero texel
      * for an external sampler; checked against the defined result. */
     true },
   /* sk_FragCoord with the render-target flip, Skia's integer dither
    * (GrDitherEffect / SkSL dither: bit interleave of x, y). */
   { "dither_fragcoord", false, 1, FS_ES3_HEAD
     "uniform mediump float urange_S1;\n"
     "uniform sampler2D uTextureSampler_0_S0;\n"
     "void main() {\n"
     "  highp vec4 sk_FragCoord = vec4(gl_FragCoord.x, u_skRTFlip.x + u_skRTFlip.y * gl_FragCoord.y, gl_FragCoord.z, gl_FragCoord.w);\n"
     "  mediump vec4 c = texture(uTextureSampler_0_S0, vTransformedCoords_3_S0);\n"
     "  mediump float value;\n"
     "  {\n"
     "    uint x = uint(sk_FragCoord.x);\n"
     "    uint y = uint(sk_FragCoord.y) ^ x;\n"
     "    uint m = (((((y & 1u) << 5 | (x & 1u) << 4) | (y & 2u) << 2) | (x & 2u) << 1) | (y & 4u) >> 1) | (x & 4u) >> 2;\n"
     "    value = float(m) / 64.0 - 0.4921875;\n"
     "  }\n"
     "  sk_FragColor = vec4(clamp(c.xyz + value * urange_S1, 0.0, c.w), c.w);\n"
     "}\n", false },
   /* Analytic AA edges (GrAAConvexPathRenderer / circle effects): derivatives,
    * inversesqrt, discard of fully outside fragments. */
   { "derivatives_discard", false, 0, FS_ES3_HEAD
     "void main() {\n"
     "  highp vec2 p = vTransformedCoords_3_S0 - vec2(0.5);\n"
     "  highp float d = dot(p, p) - 0.16;\n"
     "  highp vec2 g = vec2(dFdx(d), dFdy(d));\n"
     "  highp float dist = d * inversesqrt(max(dot(g, g), 1e-8));\n"
     "  mediump float cov = clamp(0.5 - dist, 0.0, 1.0);\n"
     "  if (cov <= 0.0) discard;\n"
     "  sk_FragColor = vcolor_S0 * cov + vec4(fwidth(d) * 8.0, 0.0, 0.0, 0.0);\n"
     "}\n", false },
   /* GrGLSLShaderBuilder sharpens mipmapped sampling with a -0.475 bias. */
   { "mip_bias", false, 2, FS_ES3_HEAD
     "uniform sampler2D uTextureSampler_0_S0;\n"
     "void main() { sk_FragColor = texture(uTextureSampler_0_S0, vTransformedCoords_3_S0 * 3.0, -0.475); }\n", false },
   /* RenderEngine LinearEffect (libs/renderengine/skia/filters/LinearEffect.cpp,
    * shaders/shaders.cpp): sRGB/PQ transfer functions, tone map, mat3/mat4. */
   { "linear_effect", false, 1, FS_ES3_HEAD
     "uniform sampler2D uTextureSampler_0_S0;\n"
     "uniform highp mat4 in_rgbToXyz;\n"
     "uniform highp mat3 in_xyzToRgb;\n"
     "uniform highp float in_libtonemap_inputMaxLuminance;\n"
     "highp vec3 EOTF_sRGB(highp vec3 c) {\n"
     "  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));\n"
     "}\n"
     "highp vec3 EOTF_PQ(highp vec3 c) {\n"
     "  const highp float m1 = 0.1593017578125, m2 = 78.84375;\n"
     "  const highp float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;\n"
     "  highp vec3 tmp = pow(clamp(c, 0.0, 1.0), 1.0 / vec3(m2));\n"
     "  tmp = max(tmp - c1, 0.0) / (c2 - c3 * tmp);\n"
     "  return pow(tmp, 1.0 / vec3(m1));\n"
     "}\n"
     "highp vec3 OETF_sRGB(highp vec3 c) {\n"
     "  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));\n"
     "}\n"
     "void main() {\n"
     "  highp vec4 c = texture(uTextureSampler_0_S0, vTransformedCoords_3_S0);\n"
     "  highp vec3 lin = EOTF_sRGB(c.rgb) + EOTF_PQ(c.bgr) * 0.01;\n"
     "  highp vec3 xyz = (in_rgbToXyz * vec4(lin, 1.0)).xyz;\n"
     "  highp float y = xyz.y * in_libtonemap_inputMaxLuminance;\n"
     "  xyz *= (y / (1.0 + y)) / max(xyz.y, 1e-6) / in_libtonemap_inputMaxLuminance;\n"
     "  highp vec3 rgb = clamp(in_xyzToRgb * xyz, 0.0, 1.0);\n"
     "  sk_FragColor = vec4(OETF_sRGB(rgb), c.a);\n"
     "}\n", false },
   /* Kawase/gaussian blur passes: loops of offset samples, ES3 integer loop. */
   { "blur_loop", false, 1, FS_ES3_HEAD
     "uniform sampler2D uTextureSampler_0_S0;\n"
     "uniform highp vec2 in_blurOffset;\n"
     "void main() {\n"
     "  mediump vec4 sum = vec4(0.0);\n"
     "  for (int i = -2; i <= 2; i++) {\n"
     "    highp vec2 o = float(i) * in_blurOffset;\n"
     "    sum += texture(uTextureSampler_0_S0, vTransformedCoords_3_S0 + o) * (i == 0 ? 0.4 : 0.15);\n"
     "  }\n"
     "  sk_FragColor = sum;\n"
     "}\n", false },
   /* Rounded-rect clip (GrRRectEffect): mediump vectors, half-precision
    * style math, max/min/length, saturate. */
   { "rrect_mediump", false, 0, FS_ES3_HEAD
     "uniform mediump vec4 uinnerRect_S1;\n"
     "uniform mediump vec2 uradiusPlusHalf_S1;\n"
     "void main() {\n"
     "  highp vec4 sk_FragCoord = vec4(gl_FragCoord.x, u_skRTFlip.x + u_skRTFlip.y * gl_FragCoord.y, gl_FragCoord.z, gl_FragCoord.w);\n"
     "  mediump vec2 dxy0 = uinnerRect_S1.xy - sk_FragCoord.xy;\n"
     "  mediump vec2 dxy1 = sk_FragCoord.xy - uinnerRect_S1.zw;\n"
     "  mediump vec2 dxy = max(max(dxy0, dxy1), 0.0);\n"
     "  mediump float alpha = clamp(uradiusPlusHalf_S1.x - length(dxy), 0.0, 1.0);\n"
     "  sk_FragColor = vcolor_S0 * alpha;\n"
     "}\n", false },
};

static void
texture_pattern(uint8_t *px)
{
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         uint8_t *p = px + 4 * (y * W + x);
         p[0] = x * 4;
         p[1] = y * 4;
         p[2] = ((x / 8 + y / 8) & 1) ? 230 : 20;
         p[3] = 255;
      }
}

/* external_size_lod: texel * 0.5 + (64/256, 64/256, 0, 0.5), unorm8. */
static void
expected_size_fetch(uint8_t *out)
{
   static uint8_t px[W * H * 4];
   texture_pattern(px);
   const float add[4] = { 0.25f, 0.25f, 0.0f, 0.5f };
   for (int i = 0; i < W * H * 4; i++) {
      float v = px[i] / 255.0f * 0.5f + add[i % 4];
      out[i] = (uint8_t)(fminf(v, 1.0f) * 255.0f + 0.5f);
   }
}

static GLuint
compile(GLenum type, const char *src, const char *name)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok = 0;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("FAIL %s: %s compile: %s\n", name, type == GL_VERTEX_SHADER ? "VS" : "FS", log);
      return 0;
   }
   return s;
}

static void
make_textures(void)
{
   static uint8_t px[W * H * 4];
   texture_pattern(px);
   glGenTextures(1, &tex2d);
   glBindTexture(GL_TEXTURE_2D, tex2d);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glGenTextures(1, &tex_mip);
   glBindTexture(GL_TEXTURE_2D, tex_mip);
   glTexStorage2D(GL_TEXTURE_2D, 7, GL_RGBA8, W, H);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   glGenerateMipmap(GL_TEXTURE_2D);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

   /* External: an EGLImage of a separate 2D texture with the same pixels. */
   GLuint src;
   glGenTextures(1, &src);
   glBindTexture(GL_TEXTURE_2D, src);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glFinish();
   EGLImageKHR img = create_image(dpy, ctx, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(uintptr_t)src, NULL);
   if (img == EGL_NO_IMAGE_KHR) {
      printf("FAIL: EGLImage from texture (0x%x)\n", eglGetError());
      exit(1);
   }
   glGenTextures(1, &tex_ext);
   glBindTexture(GL_TEXTURE_EXTERNAL_OES, tex_ext);
   image_target(GL_TEXTURE_EXTERNAL_OES, img);
   glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   GLenum e = glGetError();
   if (e) {
      printf("FAIL: external texture setup GL error 0x%x\n", e);
      exit(1);
   }
}

static void
set_uniforms(GLuint prog)
{
   GLint l;
   if ((l = glGetUniformLocation(prog, "sk_RTAdjust")) >= 0)
      glUniform4f(l, 2.0f / W, -1.0f, 2.0f / H, -1.0f);
   if ((l = glGetUniformLocation(prog, "u_skRTFlip")) >= 0)
      glUniform2f(l, (float)H, -1.0f);
   if ((l = glGetUniformLocation(prog, "umatrix_S1_c0")) >= 0) {
      const float m[9] = { 1.0f / W, 0, 0, 0, 1.0f / H, 0, 0, 0, 1 };
      glUniformMatrix3fv(l, 1, GL_FALSE, m);
   }
   if ((l = glGetUniformLocation(prog, "urange_S1")) >= 0)
      glUniform1f(l, 1.0f / 32.0f);
   if ((l = glGetUniformLocation(prog, "in_rgbToXyz")) >= 0) {
      const float m[16] = { 0.4124f, 0.2126f, 0.0193f, 0, 0.3576f, 0.7152f, 0.1192f, 0,
                            0.1805f, 0.0722f, 0.9505f, 0, 0, 0, 0, 1 };
      glUniformMatrix4fv(l, 1, GL_FALSE, m);
   }
   if ((l = glGetUniformLocation(prog, "in_xyzToRgb")) >= 0) {
      const float m[9] = { 3.2406f, -0.9689f, 0.0557f, -1.5372f, 1.8758f, -0.2040f,
                           -0.4986f, 0.0415f, 1.0570f };
      glUniformMatrix3fv(l, 1, GL_FALSE, m);
   }
   if ((l = glGetUniformLocation(prog, "in_libtonemap_inputMaxLuminance")) >= 0)
      glUniform1f(l, 4.0f);
   if ((l = glGetUniformLocation(prog, "in_blurOffset")) >= 0)
      glUniform2f(l, 2.0f / W, 1.0f / H);
   if ((l = glGetUniformLocation(prog, "uinnerRect_S1")) >= 0)
      glUniform4f(l, 16.5f, 16.5f, 47.5f, 47.5f);
   if ((l = glGetUniformLocation(prog, "uradiusPlusHalf_S1")) >= 0)
      glUniform2f(l, 8.5f, 1.0f / 8.5f);
   if ((l = glGetUniformLocation(prog, "uTextureSampler_0_S0")) >= 0)
      glUniform1i(l, 0);
   if ((l = glGetUniformLocation(prog, "s")) >= 0)
      glUniform1i(l, 0);
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "usage: %s <vendor-substring> <outdir> [<refdir>] [case-substring]\n", argv[0]);
      return 2;
   }
   const char *outdir = argv[2], *refdir = argc > 3 && strcmp(argv[3], "-") ? argv[3] : NULL;
   const char *only = argc > 4 ? argv[4] : NULL;
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) {
      printf("FAIL: eglInitialize 0x%x\n", eglGetError());
      return 1;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint ca[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
   ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
   if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: make current\n");
      return 1;
   }
   const char *vendor = (const char *)glGetString(GL_VENDOR);
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("GL_VENDOR '%s' GL_RENDERER '%s' GL_VERSION '%s'\n", vendor, renderer,
          (const char *)glGetString(GL_VERSION));
   if (!strstr(vendor, argv[1]) && !strstr(renderer, argv[1])) {
      printf("FAIL: not on %s\n", argv[1]);
      return 1;
   }
   const char *exts = (const char *)glGetString(GL_EXTENSIONS);
   static const char *need[] = { "GL_OES_EGL_image_external", "GL_OES_EGL_image_external_essl3" };
   for (unsigned i = 0; i < 2; i++)
      printf("%s: %s\n", need[i], strstr(exts, need[i]) ? "exposed" : "NOT exposed");
   printf("GL_EXT_shader_framebuffer_fetch: %s\n",
          strstr(exts, "GL_EXT_shader_framebuffer_fetch") ? "exposed" : "not exposed");
   create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
   make_textures();

   GLuint dst, fbo, vbo;
   glGenTextures(1, &dst);
   glBindTexture(GL_TEXTURE_2D, dst);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);
   const float tri[6] = { 0, 0, 2 * W, 0, 0, 2 * H };
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
   glViewport(0, 0, W, H);

   int failures = 0, ran = 0;
   for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
      const struct skcase *k = &cases[c];
      if (only && !strstr(k->name, only))
         continue;
      ran++;
      GLuint vs = compile(GL_VERTEX_SHADER, k->es1 ? vs_es1 : vs_es3, k->name);
      GLuint fs = compile(GL_FRAGMENT_SHADER, k->fs, k->name);
      if (!vs || !fs) {
         failures++;
         continue;
      }
      GLuint prog = glCreateProgram();
      glAttachShader(prog, vs);
      glAttachShader(prog, fs);
      glBindAttribLocation(prog, 0, "position");
      glLinkProgram(prog);
      GLint ok = 0;
      glGetProgramiv(prog, GL_LINK_STATUS, &ok);
      if (!ok) {
         char log[2048];
         glGetProgramInfoLog(prog, sizeof(log), NULL, log);
         printf("FAIL %s: link: %s\n", k->name, log);
         failures++;
         continue;
      }
      glUseProgram(prog);
      set_uniforms(prog);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, k->texture == 2 ? tex_mip : tex2d);
      if (k->texture == 3)
         glBindTexture(GL_TEXTURE_EXTERNAL_OES, tex_ext);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glEnableVertexAttribArray(0);
      glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      static uint8_t px[W * H * 4], ref[W * H * 4];
      glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
      GLenum e = glGetError();
      if (e) {
         printf("FAIL %s: GL error 0x%x\n", k->name, e);
         failures++;
         continue;
      }
      char path[512];
      snprintf(path, sizeof(path), "%s/%s.rgba", outdir, k->name);
      FILE *f = fopen(path, "wb");
      if (!f || fwrite(px, sizeof(px), 1, f) != 1) {
         printf("FAIL %s: write %s\n", k->name, path);
         return 1;
      }
      fclose(f);
      if (!refdir) { /* reference run: write only */
         printf("drawn %s\n", k->name);
         continue;
      }
      if (k->analytic) {
         expected_size_fetch(ref);
      } else {
         snprintf(path, sizeof(path), "%s/%s.rgba", refdir, k->name);
         f = fopen(path, "rb");
         if (!f || fread(ref, sizeof(ref), 1, f) != 1) {
            printf("FAIL %s: no reference %s\n", k->name, path);
            failures++;
            continue;
         }
         fclose(f);
      }
      int bad = 0, maxd = 0;
      for (int i = 0; i < W * H; i++) {
         int d = 0;
         for (int ch = 0; ch < 4; ch++) {
            int v = abs(px[4 * i + ch] - ref[4 * i + ch]);
            d = v > d ? v : d;
         }
         maxd = d > maxd ? d : maxd;
         bad += d > TOL;
      }
      printf("%s %s: %d px beyond %d LSB (max diff %d)\n", bad ? "FAIL" : "PASS", k->name, bad, TOL, maxd);
      failures += bad != 0;
      glDeleteProgram(prog);
   }
   printf("%s: %d cases, %d failed\n", failures ? "SKIA SHADERS FAIL" : "SKIA SHADERS PASS", ran, failures);
   return failures != 0;
}
