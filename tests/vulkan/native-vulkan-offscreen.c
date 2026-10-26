#define _GNU_SOURCE
/* SPDX-License-Identifier: MIT
 * Independent public Vulkan graphics test: fullscreen triangle -> optimal tiled
 * RGBA8 image -> GPU image-to-buffer copy -> pixel/guard verification.
 * Usage: probe /absolute/ICD.so vertex.spv fragment.spv
 */
#define VK_NO_PROTOTYPES
#include "native-driver-identity.h"
#include <vulkan/vulkan.h>
#include "native-vulkan-runtime.h"
#include <dlfcn.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INSTANCE_API(X) X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
 X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceQueueFamilyProperties) X(CreateDevice) X(GetDeviceProcAddr)
#define DEVICE_API(X) X(DestroyDevice) X(GetDeviceQueue) X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements) \
 X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindImageMemory) \
 X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(CreateImageView) X(DestroyImageView) X(CreateRenderPass) \
 X(DestroyRenderPass) X(CreateFramebuffer) X(DestroyFramebuffer) X(CreateShaderModule) X(DestroyShaderModule) \
 X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateGraphicsPipelines) X(DestroyPipeline) \
 X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(ResetCommandBuffer) \
 X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdBeginRenderPass) X(CmdEndRenderPass) X(CmdBindPipeline) \
 X(CmdPushConstants) X(CmdDraw) X(CmdCopyImageToBuffer) X(CmdPipelineBarrier) X(CreateFence) X(DestroyFence) \
 X(ResetFences) X(WaitForFences) X(QueueSubmit) X(DeviceWaitIdle)
#define DECLARE(name) static PFN_vk##name p##name;
INSTANCE_API(DECLARE)
DEVICE_API(DECLARE)
#undef DECLARE

static void fail(const char *message) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
static void result(VkResult status, const char *operation, int line) {
    if (status) { fprintf(stderr, "FAIL: line %d %s returned %d\n", line, operation, status); exit(1); }
}
#define CHECK(call) result((call), #call, __LINE__)
static unsigned fd_count(void) {
    DIR *dir=opendir("/proc/self/fd"); if (!dir) fail("fd directory");
    unsigned count=0; struct dirent *entry;
    while ((entry=readdir(dir))) if (entry->d_name[0]!='.') count++;
    closedir(dir); return count;
}
static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory_properties;
static VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags) {
    for (uint32_t i=0;i<memory_properties.memoryTypeCount;i++) {
        if ((requirements.memoryTypeBits&(1u<<i)) &&
            (memory_properties.memoryTypes[i].propertyFlags&flags)==flags) {
            VkMemoryAllocateInfo info={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize=requirements.size,.memoryTypeIndex=i};
            VkDeviceMemory memory; CHECK(pAllocateMemory(device,&info,NULL,&memory)); return memory;
        }
    }
    fail("required Vulkan memory type unavailable"); return VK_NULL_HANDLE;
}
static VkShaderModule load_shader(const char *path) {
    FILE *file=fopen(path,"rb"); if (!file || fseek(file,0,SEEK_END)) fail("open shader");
    long size=ftell(file); if (size<20 || size%4 || fseek(file,0,SEEK_SET)) fail("shader size");
    uint32_t *code=malloc((size_t)size); if (!code || fread(code,1,(size_t)size,file)!=(size_t)size) fail("read shader");
    fclose(file);
    VkShaderModuleCreateInfo info={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=(size_t)size,.pCode=code};
    VkShaderModule shader; CHECK(pCreateShaderModule(device,&info,NULL,&shader)); free(code); return shader;
}
static void check_maps(enum nv_expect_driver expect) {
    char why[4200];
    if (nv_expect_driver_check(expect,why,sizeof(why))) fail(why);
}

int main(int argc,char **argv) {
    /* The fd checks count the whole process. Mesa's shader disk cache writes each entry on its own nice-19
     * "disk$" threads (src/util/disk_cache.c:90-93, src/util/u_queue.c:258) and holds a <hash>.tmp descriptor
     * from open to close (src/util/disk_cache_os.c:883-972, disk_cache_write_item_to_disk), unordered with
     * those counts. The explicit disable also covers a cache at an explicit path (disk_cache_os.c:1182).
     * Lines: Mesa 26.2.5 with this repository's Mesa patches applied. Set before any Mesa library loads. */
    if (setenv("MESA_SHADER_CACHE_DISABLE", "true", 1) != 0) {
        perror("setenv MESA_SHADER_CACHE_DISABLE");
        return 2;
    }
    /* ICD.so vert.spv frag.spv [--stock-control | --expect-driver nvidia-prolapse|glibc-shim] */
    int stock_control=argc==5 && !strcmp(argv[4],"--stock-control");
    enum nv_expect_driver expect=NV_EXPECT_PROLAPSE;
    int expect_given=argc==6 && !strcmp(argv[4],"--expect-driver");
    if (expect_given && nv_expect_driver_parse(argv[5],&expect)) { fprintf(stderr,"--expect-driver is nvidia-prolapse or glibc-shim\n"); return 2; }
    if (argc!=4 && !stock_control && !expect_given) { fprintf(stderr,"usage: %s ICD.so offscreen.vert.spv offscreen.frag.spv [--stock-control | --expect-driver nvidia-prolapse|glibc-shim]\n",argv[0]); return 2; }
    nv_probe_android_runtime_init();
    /* NVP_OFFSCREEN_MODE (fault bisection; unset = full, the gate itself):
       full    clear + fullscreen draw + copy, checker pattern checked
       clear   render pass with loadOp CLEAR only (no pipeline bind/draw), clear colour checked
       noclear loadOp DONT_CARE + draw (no CLEAR_SURFACE), checker pattern checked
       empty   empty command buffer (3D context init + submit only), no readback
       copy    layout transition + copy-engine image->buffer copy only (no 3D work), guards checked */
    const char *mode_env=getenv("NVP_OFFSCREEN_MODE"); const char *mode=mode_env&&*mode_env?mode_env:"full";
    int do_clear=1,do_draw=1,do_pass=1,do_copy_only=0;
    if (!strcmp(mode,"clear")) do_draw=0; else if (!strcmp(mode,"noclear")) do_clear=0;
    else if (!strcmp(mode,"empty")) do_pass=do_draw=do_clear=0;
    else if (!strcmp(mode,"copy")) { do_pass=do_draw=do_clear=0; do_copy_only=1; }
    else if (strcmp(mode,"full")) fail("unknown NVP_OFFSCREEN_MODE (full|clear|noclear|empty|copy)");
    if (mode_env) printf("mode: %s\n",mode);
    unsigned before=fd_count();
    void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if (!library) fail(dlerror());
    PFN_vkGetInstanceProcAddr gipa=(PFN_vkGetInstanceProcAddr)dlsym(library,"vk_icdGetInstanceProcAddr");
    if (!gipa) gipa=(PFN_vkGetInstanceProcAddr)dlsym(library,"vkGetInstanceProcAddr");
    if (!gipa) fail("vk_icdGetInstanceProcAddr missing");
    PFN_vkCreateInstance create=(PFN_vkCreateInstance)gipa(VK_NULL_HANDLE,"vkCreateInstance");
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="source-native offscreen probe",.apiVersion=VK_API_VERSION_1_0};
    VkInstanceCreateInfo instance_info={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    VkInstance instance; CHECK(create(&instance_info,NULL,&instance));
#define LOAD_INSTANCE(name) p##name=(PFN_vk##name)gipa(instance,"vk"#name); if (!p##name) fail("missing vk"#name);
    INSTANCE_API(LOAD_INSTANCE)
#undef LOAD_INSTANCE
    uint32_t count=0; CHECK(pEnumeratePhysicalDevices(instance,&count,NULL)); if (!count) fail("no Vulkan GPU");
    VkPhysicalDevice *physical=calloc(count,sizeof(*physical)); if (!physical) fail("physical list allocation");
    CHECK(pEnumeratePhysicalDevices(instance,&count,physical));
    VkPhysicalDevice selected=VK_NULL_HANDLE; VkPhysicalDeviceProperties props;
    for (uint32_t i=0;i<count;i++) { pGetPhysicalDeviceProperties(physical[i],&props); if (props.vendorID==0x10de) { selected=physical[i]; break; } }
    free(physical); if (!selected) fail("no NVIDIA-vendor device");
    printf("GPU: %s vendor=%04x device=%04x\n",props.deviceName,props.vendorID,props.deviceID);
    pGetPhysicalDeviceMemoryProperties(selected,&memory_properties);
    uint32_t queue_count=0; pGetPhysicalDeviceQueueFamilyProperties(selected,&queue_count,NULL);
    VkQueueFamilyProperties *queues=calloc(queue_count,sizeof(*queues)); if (!queues) fail("queue allocation");
    pGetPhysicalDeviceQueueFamilyProperties(selected,&queue_count,queues);
    uint32_t family=UINT32_MAX;
    for (uint32_t i=0;i<queue_count;i++) if (queues[i].queueCount && (queues[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)) { family=i; break; }
    free(queues); if (family==UINT32_MAX) fail("no graphics queue");
    float priority=1;
    VkDeviceQueueCreateInfo queue_info={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkDeviceCreateInfo device_info={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&queue_info};
    CHECK(pCreateDevice(selected,&device_info,NULL,&device));
#define LOAD_DEVICE(name) p##name=(PFN_vk##name)pGetDeviceProcAddr(device,"vk"#name); if (!p##name) fail("missing vk"#name);
    DEVICE_API(LOAD_DEVICE)
#undef LOAD_DEVICE
    VkQueue queue; pGetDeviceQueue(device,family,0,&queue);
    enum { WIDTH=37, HEIGHT=19, PIXELS=WIDTH*HEIGHT, GUARD=64, PAYLOAD=PIXELS*4, BUFFER_SIZE=PAYLOAD+2*GUARD };
    VkImageCreateInfo image_info={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,
        .format=VK_FORMAT_R8G8B8A8_UNORM,.extent={WIDTH,HEIGHT,1},.mipLevels=1,.arrayLayers=1,
        .samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,
        .usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkImage image; CHECK(pCreateImage(device,&image_info,NULL,&image));
    VkMemoryRequirements image_requirements; pGetImageMemoryRequirements(device,image,&image_requirements);
    VkDeviceMemory image_memory=allocate(image_requirements,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(pBindImageMemory(device,image,image_memory,0));
    VkBufferCreateInfo buffer_info={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=BUFFER_SIZE,
        .usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer; CHECK(pCreateBuffer(device,&buffer_info,NULL,&buffer));
    VkMemoryRequirements buffer_requirements; pGetBufferMemoryRequirements(device,buffer,&buffer_requirements);
    VkDeviceMemory buffer_memory=allocate(buffer_requirements,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(pBindBufferMemory(device,buffer,buffer_memory,0));
    uint8_t *mapped; CHECK(pMapMemory(device,buffer_memory,0,BUFFER_SIZE,0,(void**)&mapped));
    VkImageViewCreateInfo view_info={.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=image,
        .viewType=VK_IMAGE_VIEW_TYPE_2D,.format=image_info.format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    VkImageView view; CHECK(pCreateImageView(device,&view_info,NULL,&view));
    VkAttachmentDescription attachment={.format=image_info.format,.samples=VK_SAMPLE_COUNT_1_BIT,
        .loadOp=do_clear?VK_ATTACHMENT_LOAD_OP_CLEAR:VK_ATTACHMENT_LOAD_OP_DONT_CARE,.storeOp=VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
    VkAttachmentReference reference={0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass={.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,.colorAttachmentCount=1,.pColorAttachments=&reference};
    VkSubpassDependency dependencies[2]={{.srcSubpass=VK_SUBPASS_EXTERNAL,.dstSubpass=0,
        .srcStageMask=VK_PIPELINE_STAGE_TRANSFER_BIT,.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT,.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
      {.srcSubpass=0,.dstSubpass=VK_SUBPASS_EXTERNAL,
        .srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,.dstStageMask=VK_PIPELINE_STAGE_TRANSFER_BIT,
        .srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT}};
    VkRenderPassCreateInfo pass_info={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,.attachmentCount=1,.pAttachments=&attachment,
        .subpassCount=1,.pSubpasses=&subpass,.dependencyCount=2,.pDependencies=dependencies};
    VkRenderPass pass; CHECK(pCreateRenderPass(device,&pass_info,NULL,&pass));
    VkFramebufferCreateInfo fb_info={.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,.renderPass=pass,.attachmentCount=1,.pAttachments=&view,.width=WIDTH,.height=HEIGHT,.layers=1};
    VkFramebuffer framebuffer; CHECK(pCreateFramebuffer(device,&fb_info,NULL,&framebuffer));
    VkShaderModule vertex=load_shader(argv[2]),fragment=load_shader(argv[3]);
    VkPushConstantRange push={VK_SHADER_STAGE_FRAGMENT_BIT,0,4};
    VkPipelineLayoutCreateInfo layout_info={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
    VkPipelineLayout layout; CHECK(pCreatePipelineLayout(device,&layout_info,NULL,&layout));
    VkPipelineShaderStageCreateInfo stages[2]={
        {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vertex,.pName="main"},
        {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fragment,.pName="main"}};
    VkPipelineVertexInputStateCreateInfo vertex_input={.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly={.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport viewport={0,0,WIDTH,HEIGHT,0,1}; VkRect2D scissor={{0,0},{WIDTH,HEIGHT}};
    VkPipelineViewportStateCreateInfo viewport_info={.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.pViewports=&viewport,.scissorCount=1,.pScissors=&scissor};
    VkPipelineRasterizationStateCreateInfo raster={.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1};
    VkPipelineMultisampleStateCreateInfo multisample={.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState blend_attachment={.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo blend={.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&blend_attachment};
    VkGraphicsPipelineCreateInfo pipeline_info={.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.stageCount=2,.pStages=stages,
        .pVertexInputState=&vertex_input,.pInputAssemblyState=&assembly,.pViewportState=&viewport_info,.pRasterizationState=&raster,
        .pMultisampleState=&multisample,.pColorBlendState=&blend,.layout=layout,.renderPass=pass,.subpass=0};
    VkPipeline pipeline; CHECK(pCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&pipeline_info,NULL,&pipeline));
    VkCommandPoolCreateInfo pool_info={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=family};
    VkCommandPool pool; CHECK(pCreateCommandPool(device,&pool_info,NULL,&pool));
    VkCommandBufferAllocateInfo allocate_info={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer command; CHECK(pAllocateCommandBuffers(device,&allocate_info,&command));
    VkFenceCreateInfo fence_info={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CHECK(pCreateFence(device,&fence_info,NULL,&fence));
    for (uint32_t phase=0;phase<3;phase++) {
        memset(mapped,0xcd,BUFFER_SIZE); CHECK(pResetCommandBuffer(command,0));
        VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(pBeginCommandBuffer(command,&begin));
        VkClearValue clear={.color={{0,0,1,0}}};
        VkRenderPassBeginInfo render={.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,.renderPass=pass,.framebuffer=framebuffer,
            .renderArea={{0,0},{WIDTH,HEIGHT}},.clearValueCount=1,.pClearValues=&clear};
        if (do_copy_only) {
            VkImageMemoryBarrier to_src={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=0,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=image,
                .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
            pCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&to_src);
            VkBufferImageCopy copy={.bufferOffset=GUARD,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={WIDTH,HEIGHT,1}};
            pCmdCopyImageToBuffer(command,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
        }
        if (do_pass) {
        pCmdBeginRenderPass(command,&render,VK_SUBPASS_CONTENTS_INLINE);
        if (do_draw) { pCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
        pCmdPushConstants(command,layout,VK_SHADER_STAGE_FRAGMENT_BIT,0,4,&phase); pCmdDraw(command,3,1,0,0); }
        pCmdEndRenderPass(command);
        VkBufferImageCopy copy={.bufferOffset=GUARD,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={WIDTH,HEIGHT,1}};
        pCmdCopyImageToBuffer(command,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
        VkBufferMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.buffer=buffer,.offset=0,.size=VK_WHOLE_SIZE};
        pCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,NULL,1,&barrier,0,NULL);
        }
        CHECK(pEndCommandBuffer(command)); CHECK(pResetFences(device,1,&fence));
        VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&command};
        CHECK(pQueueSubmit(queue,1,&submit,fence)); CHECK(pWaitForFences(device,1,&fence,VK_TRUE,UINT64_C(10000000000)));
        if (do_copy_only) {
            for (unsigned i=0;i<GUARD;i++) if (mapped[i]!=0xcd || mapped[GUARD+PAYLOAD+i]!=0xcd) fail("readback guard modified");
            printf("phase %u: copy-only submit completed\n",phase); continue;
        }
        if (!do_pass) { printf("phase %u: empty submit completed\n",phase); continue; }
        for (unsigned i=0;i<GUARD;i++) if (mapped[i]!=0xcd || mapped[GUARD+PAYLOAD+i]!=0xcd) fail("readback guard modified");
        for (unsigned y=0;y<HEIGHT;y++) for (unsigned x=0;x<WIDTH;x++) {
            uint8_t expected[4]={((x+phase)&1)*255,((y+phase)&1)*255,64,255};
            if (!do_draw) { expected[0]=0; expected[1]=0; expected[2]=255; expected[3]=0; }
            uint8_t *actual=mapped+GUARD+(y*WIDTH+x)*4;
            if (memcmp(actual,expected,4)) { fprintf(stderr,"pixel(%u,%u) phase%u got %u,%u,%u,%u expected %u,%u,%u,%u\n",x,y,phase,actual[0],actual[1],actual[2],actual[3],expected[0],expected[1],expected[2],expected[3]); fail("pixel mismatch"); }
        }
        printf("phase %u: %u pixels + 128 guard bytes correct\n",phase,PIXELS);
    }
    CHECK(pDeviceWaitIdle(device)); if (!stock_control) check_maps(expect);
    pDestroyFence(device,fence,NULL); pDestroyCommandPool(device,pool,NULL); pDestroyPipeline(device,pipeline,NULL);
    pDestroyPipelineLayout(device,layout,NULL); pDestroyShaderModule(device,fragment,NULL); pDestroyShaderModule(device,vertex,NULL);
    pDestroyFramebuffer(device,framebuffer,NULL); pDestroyRenderPass(device,pass,NULL); pDestroyImageView(device,view,NULL);
    pUnmapMemory(device,buffer_memory); pDestroyBuffer(device,buffer,NULL); pFreeMemory(device,buffer_memory,NULL);
    pDestroyImage(device,image,NULL); pFreeMemory(device,image_memory,NULL); pDestroyDevice(device,NULL); pDestroyInstance(instance,NULL);
    nv_probe_android_logger_close();
    dlclose(library);
    unsigned after=fd_count();
    const char *identity=stock_control?"explicit stock control":expect==NV_EXPECT_GLIBC_SHIM?"NVIDIA's driver through the glibc-compat shims":"no NVIDIA userspace/shim";
    if (stock_control || nv_expect_driver_fds_checked(expect)) { if (after!=before) fail("file descriptor leak"); }
    else if (after!=before) printf("fd %u -> %u: held by the driver, which stays loaded (glibc-shim)\n",before,after);
    printf("PASS: %u rendered pixels; fd %u -> %u; %s\n",PIXELS*3,before,after,identity);
    return 0;
}
