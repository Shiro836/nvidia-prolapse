/* SPDX-License-Identifier: MIT
 *
 * D1: rendering into a LINEAR colour buffer with a depth buffer bound, and
 * into LINEAR buffers whose pitch is not a 128 B multiple. NVK never
 * programs a LAYOUT_PITCH colour target in those cases (nvk_rendering_linear,
 * nvk_cmd_draw.c:935-963) and renders through a block-linear shadow; nvrm
 * does the same (nvrm_context.c nvrm_fb_shadow_*).
 *
 *   gl-linear-zs ref <outdir>      reference: texture FBOs (any GL driver)
 *   gl-linear-zs dmabuf <outdir>   LINEAR dma-bufs (a host-visible exportable
 *                                  Vulkan buffer, our NVK) imported with
 *                                  EGL_EXT_image_dma_buf_import, LINEAR modifier
 *
 * Scenes, flat colours only (exact): {64x64 pitch 256, 40x40 pitch 192} x
 * {depth-tested with a DEPTH_COMPONENT24 renderbuffer, no depth}. Each scene
 * is drawn in two batches (glFinish between them) so the second batch has to
 * reload the shadow from the linear buffer. Writes <outdir>/<scene>.rgba
 * (glReadPixels); in dmabuf mode also checks the linear memory itself
 * (CPU map of the Vulkan buffer) against the readback. Exit 0 = all ok.
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <drm_fourcc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static EGLDisplay dpy;
static PFNEGLCREATEIMAGEKHRPROC create_image;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target;

static struct {
   VkInstance inst;
   VkPhysicalDevice pd;
   VkDevice dev;
   PFN_vkGetMemoryFdKHR get_memory_fd;
} vk;

static bool
vk_init(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_2 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   if (vkCreateInstance(&ici, NULL, &vk.inst) != VK_SUCCESS)
      return false;
   uint32_t n = 8;
   VkPhysicalDevice pds[8];
   vkEnumeratePhysicalDevices(vk.inst, &n, pds);
   for (uint32_t i = 0; i < n && !vk.pd; i++) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(pds[i], &p);
      if (p.vendorID == 0x10de)
         vk.pd = pds[i];
   }
   if (!vk.pd)
      return false;
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueCount = 1, .pQueuePriorities = &prio };
   const char *exts[] = { "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf" };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 2, .ppEnabledExtensionNames = exts };
   if (vkCreateDevice(vk.pd, &dci, NULL, &vk.dev) != VK_SUCCESS)
      return false;
   vk.get_memory_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk.dev, "vkGetMemoryFdKHR");
   return true;
}

/* A host-visible exportable buffer: linear memory by construction. */
static bool
vk_linear_dmabuf(VkDeviceSize size, int *fd, void **map)
{
   VkExternalMemoryBufferCreateInfo ebci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ebci,
                              .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   VkBuffer buf;
   if (vkCreateBuffer(vk.dev, &bci, NULL, &buf) != VK_SUCCESS)
      return false;
   VkMemoryRequirements r;
   vkGetBufferMemoryRequirements(vk.dev, buf, &r);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(vk.pd, &mp);
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   uint32_t type = UINT32_MAX;
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
      if ((r.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         type = i;
   if (type == UINT32_MAX)
      return false;
   VkExportMemoryAllocateInfo exp = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &exp,
                               .allocationSize = r.size, .memoryTypeIndex = type };
   VkDeviceMemory mem;
   if (vkAllocateMemory(vk.dev, &ai, NULL, &mem) != VK_SUCCESS ||
       vkBindBufferMemory(vk.dev, buf, mem, 0) != VK_SUCCESS ||
       vkMapMemory(vk.dev, mem, 0, size, 0, map) != VK_SUCCESS)
      return false;
   memset(*map, 0x5a, size); /* garbage the shadow copy-in must carry */
   VkMemoryGetFdInfoKHR gfi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   return vk.get_memory_fd(vk.dev, &gfi, fd) == VK_SUCCESS;
}

static GLuint prog;
static GLint u_color;

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

static void
quad(float x0, float y0, float x1, float y1, float z, float r, float g, float b)
{
   const float v[] = { x0, y0, z, x1, y0, z, x0, y1, z, x1, y1, z };
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, v);
   glEnableVertexAttribArray(0);
   glUniform4f(u_color, r, g, b, 1.0f);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

/* Two batches: background + far quad, glFinish, then a near quad, a quad
 * behind it (hidden when depth-tested) and a clear-scissored strip. */
static void
scene(int w, int h, bool depth)
{
   glViewport(0, 0, w, h);
   glClearColor(0.0f, 0.0f, 0.25f, 1.0f);
   glClearDepthf(1.0f);
   glClear(GL_COLOR_BUFFER_BIT | (depth ? GL_DEPTH_BUFFER_BIT : 0));
   if (depth) {
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
   } else {
      glDisable(GL_DEPTH_TEST);
   }
   quad(-0.75f, -0.75f, 0.25f, 0.25f, 0.5f, 1, 0, 0);
   glFinish();
   quad(-0.25f, -0.25f, 0.75f, 0.75f, 0.0f, 0, 1, 0);
   quad(-0.5f, -0.5f, 0.5f, 0.5f, 0.75f, 1, 1, 0); /* behind both when depth-tested */
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, h - 4, w, 4);
   glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_DEPTH_TEST);
}

static void
write_file(const char *dir, const char *name, const void *px, size_t size)
{
   char path[512];
   snprintf(path, sizeof(path), "%s/%s.rgba", dir, name);
   FILE *f = fopen(path, "wb");
   if (!f || fwrite(px, 1, size, f) != size) {
      printf("FAIL: write %s\n", path);
      failures++;
   }
   if (f)
      fclose(f);
}

int
main(int argc, char **argv)
{
   if (argc != 3 || (strcmp(argv[1], "ref") && strcmp(argv[1], "dmabuf"))) {
      fprintf(stderr, "usage: %s ref|dmabuf <outdir>\n", argv[0]);
      return 2;
   }
   const bool dmabuf = !strcmp(argv[1], "dmabuf");
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_dpy =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   dpy = get_dpy(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (!eglInitialize(dpy, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API)) {
      printf("FAIL: EGL init\n");
      return 1;
   }
   const EGLint cattr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, cattr);
   if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: context\n");
      return 1;
   }
   printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));
   create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
      eglGetProcAddress("glEGLImageTargetTexture2DOES");
   if (dmabuf && !vk_init()) {
      printf("FAIL: Vulkan init (VK_DRIVER_FILES = our NVK ICD)\n");
      return 1;
   }

   prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER,
      "#version 300 es\nlayout(location=0) in vec3 p;\n"
      "void main() { gl_Position = vec4(p.xy, p.z * 2.0 - 1.0, 1.0); }\n"));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER,
      "#version 300 es\nprecision mediump float;\nuniform vec4 c;\nout vec4 o;\n"
      "void main() { o = c; }\n"));
   glLinkProgram(prog);
   glUseProgram(prog);
   u_color = glGetUniformLocation(prog, "c");

   static const struct { int w, h; uint32_t pitch; } sizes[] = {
      { 64, 64, 256 }, /* 128 B aligned: shadow only because of depth */
      { 40, 40, 192 }, /* pitch % 128 != 0: shadow always */
   };
   for (unsigned s = 0; s < 2; s++) {
      for (int d = 0; d < 2; d++) {
         const int w = sizes[s].w, h = sizes[s].h;
         const uint32_t pitch = sizes[s].pitch;
         char name[64];
         snprintf(name, sizeof(name), "%dx%d-p%u-%s", w, h, pitch, d ? "depth" : "nodepth");
         GLuint tex, fbo, rb;
         glGenTextures(1, &tex);
         glBindTexture(GL_TEXTURE_2D, tex);
         void *map = NULL;
         if (dmabuf) {
            int fd = -1;
            if (!vk_linear_dmabuf((VkDeviceSize)pitch * h, &fd, &map)) {
               printf("FAIL: %s: exportable host-visible Vulkan buffer\n", name);
               failures++;
               continue;
            }
            const EGLAttrib attrs[] = {
               EGL_WIDTH, w, EGL_HEIGHT, h, EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ABGR8888,
               EGL_DMA_BUF_PLANE0_FD_EXT, fd, EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
               EGL_DMA_BUF_PLANE0_PITCH_EXT, pitch,
               EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLAttrib)(DRM_FORMAT_MOD_LINEAR & 0xffffffff),
               EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLAttrib)(DRM_FORMAT_MOD_LINEAR >> 32),
               EGL_NONE };
            PFNEGLCREATEIMAGEPROC ci = (PFNEGLCREATEIMAGEPROC)eglGetProcAddress("eglCreateImage");
            EGLImage img = ci(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
            if (img == EGL_NO_IMAGE) {
               printf("FAIL: %s: LINEAR dma-buf import 0x%x\n", name, eglGetError());
               failures++;
               continue;
            }
            image_target(GL_TEXTURE_2D, img);
         } else {
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
         }
         glGenFramebuffers(1, &fbo);
         glBindFramebuffer(GL_FRAMEBUFFER, fbo);
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
         if (d) {
            glGenRenderbuffers(1, &rb);
            glBindRenderbuffer(GL_RENDERBUFFER, rb);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
         }
         if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            printf("FAIL: %s: framebuffer incomplete\n", name);
            failures++;
            continue;
         }
         scene(w, h, d);
         uint8_t *px = malloc((size_t)w * h * 4);
         glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
         CHECK(glGetError() == GL_NO_ERROR, "%s: no GL error", name);
         write_file(argv[2], name, px, (size_t)w * h * 4);
         if (dmabuf) {
            /* The dma-buf itself (after glFinish) must hold the pixels. */
            glFinish();
            int bad = 0;
            for (int y = 0; y < h; y++)
               bad += memcmp((uint8_t *)map + (size_t)y * pitch, px + (size_t)y * w * 4, w * 4) != 0;
            CHECK(bad == 0, "%s: linear dma-buf memory == readback (%d bad rows)", name, bad);
         }
         free(px);
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
         glDeleteFramebuffers(1, &fbo);
         glDeleteTextures(1, &tex);
      }
   }
   printf("%s\n", failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
