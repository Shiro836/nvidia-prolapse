/* vk-first-wait: does a sync_fd exported by NVIDIA's Vulkan ICD cover the work submitted before it, when the queue's
 * FIRST external-sync operation was a GPU wait on an imported sync_fd? Vulkan analogue of
 * nvidia-prolapse/egl/test/fence-first-wait.c (the EGL defect, FENCE-CONTRACT.md §3.3).
 *
 * Contract: a sync_fd exported from a semaphore/fence signal operation signals when that operation does, i.e. after
 * every command submitted before it in submission order has completed (Vulkan 1.3 §7.4.1 "signal operation order";
 * VK_KHR_external_semaphore_fd / VK_KHR_external_fence_fd: SYNC_FD export = the payload's signal).
 *
 * Per round (a NEW THREAD with a NEW VkDevice and queue(s) each round):
 *   consumer thread: create device C (the mode's queue layout), a slow command buffer (N x vkCmdFillBuffer over
 *                    256 MiB) for the target queue's family, the mode's "start" operations; report ready
 *   main thread:     producer device P: fill work + exportable fence -> sync_fd Fa (still pending when handed over)
 *   consumer thread: the mode's sequence on the target queue: import Fa and GPU-wait it, submit the slow work, export
 *                    a sync_fd Fr; then poll(Fr) until signalled, then vkQueueWaitIdle.
 * A correct Fr signals no earlier than the work, so vkQueueWaitIdle right after it returns at once. A round counts as
 * early when vkQueueWaitIdle still waited more than 2 ms after Fr had signalled, or, with --timestamps, when the GPU
 * work ended more than 0.1 ms after poll() had seen Fr signalled.
 *
 * Sequences (what the consumer does on the target queue):
 *   sem-export        import Fa -> semaphore (TEMPORARY); submit {wait it; work; signal exportable semaphore};
 *                     vkGetSemaphoreFdKHR(SYNC_FD)
 *   fence-export      import Fa -> semaphore; submit {wait it; work} + exportable fence; vkGetFenceFdKHR(SYNC_FD)
 *   hal-seq           the sequence our HAL performs (hal/src/anb.c): vkAcquireImageANDROID(Fa, semaphore, fence)
 *                     [= import into semaphore + dup into fence, both TEMPORARY]; the app's submit {wait semaphore;
 *                     work; signal render-done semaphore}; vkQueueSignalReleaseImageANDROID(render-done) [= submit
 *                     {wait render-done} + pooled exportable fence; vkGetFenceFdKHR]. With --hal the HAL's own entry
 *                     points are called; with --icd the same ICD calls are made here.
 *   hal-seq-m1first   hal-seq preceded by one whole frame whose acquire fence is -1 (a never-used swapchain slot:
 *                     BufferSlot.h:183), i.e. the order a fresh swapchain sees
 *   fence-import-cpu  import Fa -> fence (TEMPORARY); vkWaitForFences; submit work + exportable fence; export
 *   no-import         no import: submit work + exportable fence; export (first export on a fresh queue)
 * Start operations ("warm", done by the consumer before Fa exists): one empty vkQueueSubmit with an exportable fence
 * + vkGetFenceFdKHR (what the HAL does at vkCreateDevice, hal/AHB-CONTRACT.md §8), or with --sem-start-style modes an
 * empty submit signalling an exportable semaphore + vkGetSemaphoreFdKHR.
 *
 * Modes: the six sequences above on a one-queue device, plus (see MODES[] for the exact list)
 *   <seq>-warm        sem-export / fence-export / hal-seq after one start operation on the same queue
 *   mq-*              multi-queue layouts: is the defect's state per queue, per family or per device?
 *   warm-qidle, warm-didle  start, then vkQueueWaitIdle / vkDeviceWaitIdle, then fence-export: does idling reset it?
 *   warm-thread       start in another thread than the one that later runs fence-export
 *   warm-q2           start on the queue returned by vkGetDeviceQueue2(flags 0); fence-export on vkGetDeviceQueue's
 *   sem-warm          start = empty submit signalling an exportable semaphore + vkGetSemaphoreFdKHR
 *
 *   cc -O1 -o vk-first-wait vk-first-wait.c -ldl -lpthread
 *   vk-first-wait (--icd /usr/lib/libEGL_nvidia.so.0 | --hal out/host/vulkan.nvidia.so) [--mode M] [--rounds N]
 *                 [--fills N] [--a-fills N] [--timestamps] [--no-fence-ext] [--no-anb]
 *   vk-first-wait (--icd ... | --hal ...) --bench-create N [--bench-layout g1|g2|g16|all]
 * --timestamps: GPU timestamps around the work, mapped to CLOCK_MONOTONIC with VK_KHR_calibrated_timestamps; each round
 *               then also reports when the work really ended relative to the moment poll() saw Fr signalled.
 * --no-fence-ext: the consumer device does not enable VK_KHR_external_fence_fd (semaphore-only sequences); with --hal
 *               and --no-anb this exercises the HAL's semaphore start path.
 * --bench-create N: N x (vkCreateDevice + vkGetDeviceQueue for every queue of the layout), median/min/max in ms.
 * Exit status: number of early rounds (100 on a setup failure). */
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>
#include <vulkan/vk_android_native_buffer.h>

#define DIE(...) do { printf("FATAL: " __VA_ARGS__); printf("\n"); exit(100); } while (0)
#define MUST(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) DIE("%s -> %d", #x, r_); } while (0)

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkCreateInstance CreateInstance;
static int via_hal, no_fence_ext, no_anb;
static VkInstance inst;
static VkPhysicalDevice pd;
static uint32_t memtype = UINT32_MAX;
static uint32_t fam_g = UINT32_MAX, fam_c = UINT32_MAX, fam_t = UINT32_MAX; /* graphics, compute-only, transfer-only */
static uint32_t cnt_g, cnt_c, cnt_t;
static int fills = 30, a_fills = 400, use_ts;
static float ts_period; /* ns per timestamp tick */

enum seq { SEQ_SEM_EXPORT, SEQ_FENCE_EXPORT, SEQ_HAL, SEQ_HAL_M1, SEQ_FENCE_IMPORT_CPU, SEQ_NO_IMPORT };
enum layout { L_G1, L_G2, L_GC, L_ALL, L_G16 };
enum start { S_NONE, S_FENCE, S_SEM };
enum after { A_NONE, A_QIDLE, A_DIDLE };
struct mode {
    const char *name;
    enum seq seq;
    enum layout layout;
    char tfam; /* target queue: 'g' or 'c' */
    uint32_t tidx;
    const char *warm; /* queues that get a start operation: "", "g0", "g1", "all" */
    enum start how;
    enum after after;
    int warm_thread, warm_q2;
    const char *what;
};
static const struct mode MODES[] = {
    {"sem-export", SEQ_SEM_EXPORT, L_G1, 'g', 0, "", S_NONE, A_NONE, 0, 0, "fresh queue: import-wait; work; export semaphore"},
    {"fence-export", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "", S_NONE, A_NONE, 0, 0, "fresh queue: import-wait; work; export fence"},
    {"hal-seq", SEQ_HAL, L_G1, 'g', 0, "", S_NONE, A_NONE, 0, 0, "fresh queue: acquire(Fa); render; signal-release"},
    {"hal-seq-m1first", SEQ_HAL_M1, L_G1, 'g', 0, "", S_NONE, A_NONE, 0, 0, "frame 1 acquire(-1), then hal-seq"},
    {"fence-import-cpu", SEQ_FENCE_IMPORT_CPU, L_G1, 'g', 0, "", S_NONE, A_NONE, 0, 0, "CPU wait on imported fence; work; export"},
    {"no-import", SEQ_NO_IMPORT, L_G1, 'g', 0, "", S_NONE, A_NONE, 0, 0, "no import; work; export"},
    {"sem-export-warm", SEQ_SEM_EXPORT, L_G1, 'g', 0, "g0", S_FENCE, A_NONE, 0, 0, "start; then sem-export"},
    {"fence-export-warm", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "g0", S_FENCE, A_NONE, 0, 0, "start; then fence-export"},
    {"hal-seq-warm", SEQ_HAL, L_G1, 'g', 0, "g0", S_FENCE, A_NONE, 0, 0, "start; then hal-seq"},
    /* is the state per queue, per family or per device? (task 1) */
    {"mq-fresh-g1", SEQ_FENCE_EXPORT, L_G2, 'g', 1, "", S_NONE, A_NONE, 0, 0, "graphics x2, nothing started; fence-export on g1"},
    {"mq-warm-g0-use-g1", SEQ_FENCE_EXPORT, L_G2, 'g', 1, "g0", S_FENCE, A_NONE, 0, 0, "graphics x2; start g0; fence-export on g1"},
    {"mq-warm-g1-use-g1", SEQ_FENCE_EXPORT, L_G2, 'g', 1, "g1", S_FENCE, A_NONE, 0, 0, "graphics x2; start g1; fence-export on g1"},
    {"mq-warm-g0-use-g15", SEQ_FENCE_EXPORT, L_G16, 'g', 15, "g0", S_FENCE, A_NONE, 0, 0, "graphics x16; start g0; fence-export on g15"},
    {"mq-fresh-c0", SEQ_FENCE_EXPORT, L_GC, 'c', 0, "", S_NONE, A_NONE, 0, 0, "graphics + compute family; nothing started; fence-export on c0"},
    {"mq-warm-g0-use-c0", SEQ_FENCE_EXPORT, L_GC, 'c', 0, "g0", S_FENCE, A_NONE, 0, 0, "graphics + compute; start g0; fence-export on c0"},
    {"mq-warm-all-use-c0", SEQ_FENCE_EXPORT, L_ALL, 'c', 0, "all", S_FENCE, A_NONE, 0, 0, "graphics x2 + compute + transfer; start all; fence-export on c0"},
    {"mq-warm-all-use-g1", SEQ_FENCE_EXPORT, L_ALL, 'g', 1, "all", S_FENCE, A_NONE, 0, 0, "graphics x2 + compute + transfer; start all; fence-export on g1"},
    /* does idling, or another thread, or the vkGetDeviceQueue2 handle change anything? (task 1) */
    {"warm-qidle", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "g0", S_FENCE, A_QIDLE, 0, 0, "start; vkQueueWaitIdle; fence-export"},
    {"warm-didle", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "g0", S_FENCE, A_DIDLE, 0, 0, "start; vkDeviceWaitIdle; fence-export"},
    {"warm-thread", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "g0", S_FENCE, A_NONE, 1, 0, "start in another thread; fence-export in this one"},
    {"warm-q2", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "g0", S_FENCE, A_NONE, 0, 1, "start on vkGetDeviceQueue2(flags 0)'s handle; fence-export on vkGetDeviceQueue's"},
    {"sem-warm", SEQ_FENCE_EXPORT, L_G1, 'g', 0, "g0", S_SEM, A_NONE, 0, 0, "start = empty submit + semaphore export; fence-export"},
    {"sem-warm-sem-export", SEQ_SEM_EXPORT, L_G1, 'g', 0, "g0", S_SEM, A_NONE, 0, 0, "start = semaphore export; sem-export"},
};
static const struct mode *M;

#define DEVFN(X) X(GetDeviceQueue) X(DestroyDevice) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) \
    X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(CreateCommandPool) X(DestroyCommandPool) \
    X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdFillBuffer) X(QueueSubmit) \
    X(QueueWaitIdle) X(DeviceWaitIdle) X(CreateFence) X(DestroyFence) X(WaitForFences) X(CreateSemaphore) \
    X(DestroySemaphore) X(GetSemaphoreFdKHR) X(ImportSemaphoreFdKHR) X(GetDeviceQueue2)
#define FENCEFN(X) X(GetFenceFdKHR) X(ImportFenceFdKHR)
#define TSFN(X) X(CreateQueryPool) X(DestroyQueryPool) X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults) \
    X(GetCalibratedTimestampsKHR)
#define MAXF 4
struct dev {
    VkDevice d;
    uint32_t nfam, fam[MAXF], cnt[MAXF];
    VkQueue qs[MAXF][16];
    VkQueue q; /* the target queue */
    VkCommandPool pool;
    VkCommandBuffer work;
    VkBuffer buf;
    VkDeviceMemory mem;
#define F(n) PFN_vk##n n;
    DEVFN(F) FENCEFN(F) TSFN(F)
#undef F
    VkQueryPool qp;
    PFN_vkAcquireImageANDROID AcquireImageANDROID;
    PFN_vkQueueSignalReleaseImageANDROID QueueSignalReleaseImageANDROID;
};

static void layout_of(enum layout l, struct dev *v) {
    v->nfam = 0;
#define ADD(f, c) do { if ((f) == UINT32_MAX) DIE("layout needs a queue family this GPU lacks"); \
                       v->fam[v->nfam] = (f); v->cnt[v->nfam++] = (c); } while (0)
    switch (l) {
    case L_G1: ADD(fam_g, 1); break;
    case L_G2: ADD(fam_g, 2); break;
    case L_GC: ADD(fam_g, 1); ADD(fam_c, 1); break;
    case L_ALL: ADD(fam_g, 2); ADD(fam_c, 1); ADD(fam_t, 1); break;
    case L_G16: if (cnt_g < 16) DIE("graphics family has %u queues, layout needs 16", cnt_g); ADD(fam_g, 16); break;
    }
#undef ADD
}
static VkQueue queue_of(struct dev *v, char fam, uint32_t idx) {
    uint32_t f = fam == 'g' ? fam_g : fam == 'c' ? fam_c : fam_t;
    for (uint32_t i = 0; i < v->nfam; i++) if (v->fam[i] == f && idx < v->cnt[i]) return v->qs[i][idx];
    DIE("queue %c%u not in the layout", fam, idx);
}

/* Creates the device with the layout's queues and fetches every queue. nfill < 0: no work command buffer
 * (vkCreateDevice benchmark). */
static void dev_create(struct dev *v, int nfill, enum layout l, char tfam, uint32_t tidx) {
    PFN_vkCreateDevice CreateDevice = (PFN_vkCreateDevice)gipa(inst, "vkCreateDevice");
    PFN_vkGetDeviceProcAddr gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    static const float prio[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    layout_of(l, v);
    VkDeviceQueueCreateInfo qci[MAXF];
    for (uint32_t i = 0; i < v->nfam; i++)
        qci[i] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = v->fam[i],
                                           .queueCount = v->cnt[i], .pQueuePriorities = prio};
    const char *exts[4];
    uint32_t ne = 0;
    exts[ne++] = "VK_KHR_external_semaphore_fd";
    if (!no_fence_ext) exts[ne++] = "VK_KHR_external_fence_fd";
    if (via_hal && !no_anb) exts[ne++] = VK_ANDROID_NATIVE_BUFFER_EXTENSION_NAME;
    if (use_ts && nfill >= 0) exts[ne++] = "VK_KHR_calibrated_timestamps";
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = v->nfam,
                              .pQueueCreateInfos = qci, .enabledExtensionCount = ne, .ppEnabledExtensionNames = exts};
    MUST(CreateDevice(pd, &dci, NULL, &v->d));
#define F(n) if (!(v->n = (PFN_vk##n)gdpa(v->d, "vk" #n))) DIE("vk" #n);
    DEVFN(F)
    if (!no_fence_ext) { FENCEFN(F) }
    if (use_ts && nfill >= 0) { TSFN(F) }
#undef F
    for (uint32_t i = 0; i < v->nfam; i++)
        for (uint32_t k = 0; k < v->cnt[i]; k++) v->GetDeviceQueue(v->d, v->fam[i], k, &v->qs[i][k]);
    if (nfill < 0) return;
    v->q = queue_of(v, tfam, tidx);
    uint32_t tf = tfam == 'g' ? fam_g : fam_c;
    if (use_ts) {
        VkQueryPoolCreateInfo qpci = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                      .queryCount = 2};
        MUST(v->CreateQueryPool(v->d, &qpci, NULL, &v->qp));
    }
    if (via_hal && !no_anb) {
        v->AcquireImageANDROID = (PFN_vkAcquireImageANDROID)gdpa(v->d, "vkAcquireImageANDROID");
        v->QueueSignalReleaseImageANDROID =
            (PFN_vkQueueSignalReleaseImageANDROID)gdpa(v->d, "vkQueueSignalReleaseImageANDROID");
        if (!v->AcquireImageANDROID || !v->QueueSignalReleaseImageANDROID) DIE("HAL lacks the ANB entry points");
    }
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 256u << 20,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    MUST(v->CreateBuffer(v->d, &bci, NULL, &v->buf));
    VkMemoryRequirements mr;
    v->GetBufferMemoryRequirements(v->d, v->buf, &mr);
    if (!(mr.memoryTypeBits & (1u << memtype))) DIE("buffer cannot live in memory type %u", memtype);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                .memoryTypeIndex = memtype};
    MUST(v->AllocateMemory(v->d, &mai, NULL, &v->mem));
    MUST(v->BindBufferMemory(v->d, v->buf, v->mem, 0));
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = tf};
    MUST(v->CreateCommandPool(v->d, &cpci, NULL, &v->pool));
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = v->pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    MUST(v->AllocateCommandBuffers(v->d, &cbai, &v->work));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    MUST(v->BeginCommandBuffer(v->work, &bi));
    if (use_ts) {
        v->CmdResetQueryPool(v->work, v->qp, 0, 2);
        v->CmdWriteTimestamp(v->work, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, v->qp, 0);
    }
    for (int i = 0; i < nfill; i++) v->CmdFillBuffer(v->work, v->buf, 0, VK_WHOLE_SIZE, (uint32_t)i);
    if (use_ts) v->CmdWriteTimestamp(v->work, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, v->qp, 1);
    MUST(v->EndCommandBuffer(v->work));
}
static void dev_destroy(struct dev *v) {
    v->DeviceWaitIdle(v->d);
    if (use_ts) v->DestroyQueryPool(v->d, v->qp, NULL);
    v->DestroyCommandPool(v->d, v->pool, NULL);
    v->DestroyBuffer(v->d, v->buf, NULL);
    v->FreeMemory(v->d, v->mem, NULL);
    v->DestroyDevice(v->d, NULL);
}
static VkFence export_fence(struct dev *v) {
    VkExportFenceCreateInfo e = {.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
                                 .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = no_fence_ext ? NULL : &e};
    VkFence f;
    MUST(v->CreateFence(v->d, &fci, NULL, &f));
    return f;
}
static VkSemaphore semaphore(struct dev *v, int exportable) {
    VkExportSemaphoreCreateInfo e = {.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                                     .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = exportable ? &e : NULL};
    VkSemaphore s;
    MUST(v->CreateSemaphore(v->d, &sci, NULL, &s));
    return s;
}
static int fence_fd(struct dev *v, VkFence f) {
    VkFenceGetFdInfoKHR gi = {.sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR, .fence = f,
                              .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
    int fd = -1;
    MUST(v->GetFenceFdKHR(v->d, &gi, &fd));
    return fd;
}
static int sem_fd(struct dev *v, VkSemaphore s) {
    VkSemaphoreGetFdInfoKHR gi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = s,
                                  .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    int fd = -1;
    MUST(v->GetSemaphoreFdKHR(v->d, &gi, &fd));
    return fd;
}
static void import_sem(struct dev *v, VkSemaphore s, int fd) {
    VkImportSemaphoreFdInfoKHR ii = {.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, .semaphore = s,
                                     .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                     .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, .fd = fd};
    MUST(v->ImportSemaphoreFdKHR(v->d, &ii));
}
static void import_fence(struct dev *v, VkFence f, int fd) {
    VkImportFenceFdInfoKHR ii = {.sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR, .fence = f,
                                 .flags = VK_FENCE_IMPORT_TEMPORARY_BIT,
                                 .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT, .fd = fd};
    MUST(v->ImportFenceFdKHR(v->d, &ii));
}
static void submit_q(struct dev *v, VkQueue q, VkSemaphore wait, VkCommandBuffer cb, VkSemaphore signal, VkFence f) {
    VkPipelineStageFlags st = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = wait ? 1 : 0,
                       .pWaitSemaphores = &wait, .pWaitDstStageMask = &st, .commandBufferCount = cb ? 1 : 0,
                       .pCommandBuffers = &cb, .signalSemaphoreCount = signal ? 1 : 0, .pSignalSemaphores = &signal};
    MUST(v->QueueSubmit(q, 1, &si, f));
}
static void submit(struct dev *v, VkSemaphore wait, VkCommandBuffer cb, VkSemaphore signal, VkFence f) {
    submit_q(v, v->q, wait, cb, signal, f);
}
/* One start operation on q: an empty submit with an exportable fence (or semaphore) and its sync_fd export. The fd
 * is waited for, so the fence/semaphore can be destroyed (no pending use). */
static void start_queue(struct dev *v, VkQueue q, enum start how) {
    int fd;
    if (how == S_SEM) {
        VkSemaphore s = semaphore(v, 1);
        submit_q(v, q, VK_NULL_HANDLE, VK_NULL_HANDLE, s, VK_NULL_HANDLE);
        fd = sem_fd(v, s);
        struct pollfd p = {.fd = fd, .events = POLLIN};
        if (poll(&p, 1, 5000) != 1) DIE("start (semaphore) did not signal");
        v->DestroySemaphore(v->d, s, NULL);
    } else {
        VkFence f = export_fence(v);
        submit_q(v, q, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, f);
        fd = fence_fd(v, f);
        struct pollfd p = {.fd = fd, .events = POLLIN};
        if (poll(&p, 1, 5000) != 1) DIE("start (fence) did not signal");
        v->DestroyFence(v->d, f, NULL);
    }
    close(fd);
}
static void start_set(struct dev *v) {
    if (!*M->warm) return;
    if (!strcmp(M->warm, "all")) {
        for (uint32_t i = 0; i < v->nfam; i++)
            for (uint32_t k = 0; k < v->cnt[i]; k++) start_queue(v, v->qs[i][k], M->how);
        return;
    }
    VkQueue q = queue_of(v, M->warm[0], (uint32_t)atoi(M->warm + 1));
    if (M->warm_q2) {
        VkDeviceQueueInfo2 qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2, .queueFamilyIndex = fam_g,
                                 .queueIndex = (uint32_t)atoi(M->warm + 1)};
        VkQueue q2 = VK_NULL_HANDLE;
        v->GetDeviceQueue2(v->d, &qi, &q2);
        printf("  vkGetDeviceQueue2(flags 0) handle %p, vkGetDeviceQueue handle %p (%s)\n", (void *)q2, (void *)q,
               q2 == q ? "same" : "DIFFERENT");
        q = q2;
    }
    start_queue(v, q, M->how);
}
struct warm_arg { struct dev *v; };
static void *warm_thread(void *a) { start_set(((struct warm_arg *)a)->v); return NULL; }

/* hand-off (see fence-first-wait.c): the consumer is fully set up before Fa exists */
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static int ready, posted, fa = -1;

struct result { int at_creation; double t_sig, t_idle, end_after_sig, work_ms, cal_dev_ms; };

static void *consumer(void *arg) {
    struct result *r = arg;
    struct dev c;
    dev_create(&c, fills, M->layout, M->tfam, M->tidx);
    VkSemaphore s1 = semaphore(&c, 0), s2 = semaphore(&c, 0), sx = semaphore(&c, 1);
    VkFence f1 = export_fence(&c), fx = export_fence(&c), fw = export_fence(&c);
    if (M->warm_thread) {
        pthread_t t;
        struct warm_arg wa = {&c};
        pthread_create(&t, NULL, warm_thread, &wa);
        pthread_join(t, NULL);
    } else {
        start_set(&c);
    }
    if (M->after == A_QIDLE) MUST(c.QueueWaitIdle(c.q));
    if (M->after == A_DIDLE) MUST(c.DeviceWaitIdle(c.d));
    pthread_mutex_lock(&mu);
    ready = 1;
    pthread_cond_broadcast(&cv);
    while (!posted) pthread_cond_wait(&cv, &mu);
    pthread_mutex_unlock(&mu);

    int fr = -1;
    double t0 = 0;
    switch (M->seq) {
    case SEQ_SEM_EXPORT:
        import_sem(&c, s1, fa);
        submit(&c, s1, c.work, sx, VK_NULL_HANDLE);
        fr = sem_fd(&c, sx);
        t0 = now_ms();
        break;
    case SEQ_FENCE_EXPORT:
        import_sem(&c, s1, fa);
        submit(&c, s1, c.work, VK_NULL_HANDLE, fx);
        fr = fence_fd(&c, fx);
        t0 = now_ms();
        break;
    case SEQ_HAL_M1:
    case SEQ_HAL:
        if (M->seq == SEQ_HAL_M1) { /* frame 1 on a never-used slot: acquire fence -1, render, release */
            int f1fd = -1;
            if (via_hal) {
                MUST(c.AcquireImageANDROID(c.d, VK_NULL_HANDLE, -1, s1, f1));
            } else {
                import_sem(&c, s1, -1);
                import_fence(&c, f1, -1);
            }
            submit(&c, s1, c.work, s2, VK_NULL_HANDLE);
            if (via_hal) {
                MUST(c.QueueSignalReleaseImageANDROID(c.q, 1, &s2, VK_NULL_HANDLE, &f1fd));
            } else {
                submit(&c, s2, VK_NULL_HANDLE, VK_NULL_HANDLE, fw);
                f1fd = fence_fd(&c, fw);
            }
            close(f1fd);
            MUST(c.QueueWaitIdle(c.q));
        }
        if (via_hal) {
            MUST(c.AcquireImageANDROID(c.d, VK_NULL_HANDLE, fa, s1, f1)); /* the HAL ignores the image */
        } else { /* what the HAL does (anb.c nvhal_AcquireImageANDROID) */
            int d2 = fcntl(fa, F_DUPFD_CLOEXEC, 0);
            import_sem(&c, s1, fa);
            import_fence(&c, f1, d2);
        }
        submit(&c, s1, c.work, s2, VK_NULL_HANDLE); /* the app's rendering, signalling render-done */
        if (via_hal) {
            MUST(c.QueueSignalReleaseImageANDROID(c.q, 1, &s2, VK_NULL_HANDLE, &fr));
        } else { /* what the HAL does (anb.c nvhal_QueueSignalReleaseImageANDROID) */
            submit(&c, s2, VK_NULL_HANDLE, VK_NULL_HANDLE, fx);
            fr = fence_fd(&c, fx);
        }
        t0 = now_ms();
        break;
    case SEQ_FENCE_IMPORT_CPU:
        import_fence(&c, f1, fa);
        MUST(c.WaitForFences(c.d, 1, &f1, VK_TRUE, 5000000000ull));
        submit(&c, VK_NULL_HANDLE, c.work, VK_NULL_HANDLE, fx);
        fr = fence_fd(&c, fx);
        t0 = now_ms();
        break;
    case SEQ_NO_IMPORT:
        close(fa);
        submit(&c, VK_NULL_HANDLE, c.work, VK_NULL_HANDLE, fx);
        fr = fence_fd(&c, fx);
        t0 = now_ms();
        break;
    }
    if (fr < 0) DIE("no sync_fd exported");
    struct pollfd q = {.fd = fr, .events = POLLIN};
    r->at_creation = poll(&q, 1, 0) == 1;
    if (poll(&q, 1, 5000) != 1) DIE("Fr did not signal in 5 s");
    double sig_abs = now_ms();
    r->t_sig = sig_abs - t0;
    MUST(c.QueueWaitIdle(c.q));
    r->t_idle = now_ms() - t0;
    if (use_ts) { /* when did the work really end, on CLOCK_MONOTONIC, relative to the moment poll() saw Fr signalled */
        uint64_t qr[2];
        MUST(c.GetQueryPoolResults(c.d, c.qp, 0, 2, sizeof qr, qr, sizeof qr[0],
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        VkCalibratedTimestampInfoKHR ci[2] = {
            {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
            {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
        uint64_t ct[2], maxdev = 0;
        MUST(c.GetCalibratedTimestampsKHR(c.d, 2, ci, ct, &maxdev));
        double end_mono_ms = (double)ct[1] / 1e6 - ((double)(int64_t)(ct[0] - qr[1]) * ts_period) / 1e6;
        r->end_after_sig = end_mono_ms - sig_abs;
        r->work_ms = (double)(int64_t)(qr[1] - qr[0]) * ts_period / 1e6;
        r->cal_dev_ms = (double)maxdev / 1e6;
    }
    close(fr);
    c.DestroySemaphore(c.d, s1, NULL);
    c.DestroySemaphore(c.d, s2, NULL);
    c.DestroySemaphore(c.d, sx, NULL);
    c.DestroyFence(c.d, f1, NULL);
    c.DestroyFence(c.d, fx, NULL);
    c.DestroyFence(c.d, fw, NULL);
    dev_destroy(&c);
    return NULL;
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* vkCreateDevice + vkGetDeviceQueue (every queue of the layout) latency, N runs; then vkDestroyDevice (not timed) */
static void bench_create(int n, enum layout l) {
    double *t = calloc((size_t)n, sizeof *t);
    for (int i = 0; i < n; i++) {
        struct dev v;
        memset(&v, 0, sizeof v);
        double t0 = now_ms();
        dev_create(&v, -1, l, 'g', 0);
        t[i] = now_ms() - t0;
        v.DestroyDevice(v.d, NULL);
    }
    qsort(t, (size_t)n, sizeof *t, cmp_d);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += t[i];
    struct dev v;
    layout_of(l, &v);
    uint32_t nq = 0;
    for (uint32_t i = 0; i < v.nfam; i++) nq += v.cnt[i];
    printf("bench-create: %d x (vkCreateDevice + vkGetDeviceQueue x%u) via %s: median %.3f ms, mean %.3f, min %.3f, "
           "p90 %.3f, max %.3f\n", n, nq, via_hal ? "HAL" : "ICD", t[n / 2], sum / n, t[0], t[(n * 9) / 10], t[n - 1]);
    free(t);
}

int main(int argc, char **argv) {
    const char *icd = NULL, *hal = NULL, *mode = "sem-export", *blayout = "g1";
    int rounds = 20, bench = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--icd") && i + 1 < argc) icd = argv[++i];
        else if (!strcmp(argv[i], "--hal") && i + 1 < argc) hal = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fills") && i + 1 < argc) fills = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--a-fills") && i + 1 < argc) a_fills = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timestamps")) use_ts = 1;
        else if (!strcmp(argv[i], "--no-fence-ext")) no_fence_ext = 1;
        else if (!strcmp(argv[i], "--no-anb")) no_anb = 1;
        else if (!strcmp(argv[i], "--bench-create") && i + 1 < argc) bench = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench-layout") && i + 1 < argc) blayout = argv[++i];
        else if (!strcmp(argv[i], "--list-modes")) {
            for (size_t k = 0; k < sizeof MODES / sizeof *MODES; k++) printf("%-22s %s\n", MODES[k].name, MODES[k].what);
            return 0;
        } else DIE("usage: vk-first-wait (--icd ICD.so | --hal vulkan.nvidia.so) [--mode M] [--rounds N] [--fills N] "
                   "[--a-fills N] [--timestamps] [--no-fence-ext] [--no-anb] [--bench-create N [--bench-layout L]] "
                   "[--list-modes]");
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    for (size_t k = 0; k < sizeof MODES / sizeof *MODES; k++) if (!strcmp(MODES[k].name, mode)) M = &MODES[k];
    if (!M) DIE("unknown mode %s (--list-modes)", mode);
    if (no_fence_ext && (M->seq != SEQ_SEM_EXPORT || M->how == S_FENCE))
        DIE("--no-fence-ext needs a semaphore-only mode (sem-export, sem-warm-sem-export)");
    if (!!icd == !!hal) DIE("give exactly one of --icd, --hal");
    if (icd) {
        void *so = dlopen(icd, RTLD_NOW | RTLD_LOCAL);
        if (!so) DIE("dlopen %s: %s", icd, dlerror());
        VkResult (*neg)(uint32_t *) = (VkResult(*)(uint32_t *))dlsym(so, "vk_icdNegotiateLoaderICDInterfaceVersion");
        uint32_t ver = 5;
        if (neg && neg(&ver) != VK_SUCCESS) DIE("ICD interface negotiation");
        if (!(gipa = (PFN_vkGetInstanceProcAddr)dlsym(so, "vk_icdGetInstanceProcAddr"))) DIE("no vk_icdGetInstanceProcAddr");
    } else {
        via_hal = 1;
        void *so = dlopen(hal, RTLD_NOW | RTLD_LOCAL);
        if (!so) DIE("dlopen %s: %s", hal, dlerror());
        hwvulkan_module_t *hmi = dlsym(so, HAL_MODULE_INFO_SYM_AS_STR);
        if (!hmi || strcmp(hmi->common.id, HWVULKAN_HARDWARE_MODULE_ID)) DIE("no HMI");
        hmi->common.dso = so;
        hwvulkan_device_t *h = NULL;
        if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, (hw_device_t **)&h) || !h) DIE("open vk0");
        gipa = h->GetInstanceProcAddr;
    }
    if ((M->seq == SEQ_HAL || M->seq == SEQ_HAL_M1) && via_hal && no_anb) DIE("hal-seq modes need VK_ANDROID_native_buffer");
    CreateInstance = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vk-first-wait",
                            .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    MUST(CreateInstance(&ici, NULL, &inst));
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties GetProps = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetQF =
        (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkGetPhysicalDeviceMemoryProperties GetMP =
        (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(inst, "vkGetPhysicalDeviceMemoryProperties");
    uint32_t n = 0;
    MUST(EnumeratePhysicalDevices(inst, &n, NULL));
    VkPhysicalDevice pds[8];
    if (n > 8) n = 8;
    MUST(EnumeratePhysicalDevices(inst, &n, pds));
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < n && !pd; i++) {
        GetProps(pds[i], &props);
        if (props.vendorID == 0x10de) pd = pds[i]; /* by vendor ID, never by node path */
    }
    if (!pd) DIE("no NVIDIA (0x10de) physical device");
    ts_period = props.limits.timestampPeriod;
    VkQueueFamilyProperties qf[16];
    uint32_t nq = 16;
    GetQF(pd, &nq, qf);
    for (uint32_t i = 0; i < nq; i++) {
        VkQueueFlags fl = qf[i].queueFlags;
        if (fam_g == UINT32_MAX && (fl & VK_QUEUE_GRAPHICS_BIT)) fam_g = i, cnt_g = qf[i].queueCount;
        else if (fam_c == UINT32_MAX && (fl & VK_QUEUE_COMPUTE_BIT) && !(fl & VK_QUEUE_GRAPHICS_BIT)) fam_c = i, cnt_c = qf[i].queueCount;
        else if (fam_t == UINT32_MAX && fl == (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT)) fam_t = i, cnt_t = qf[i].queueCount;
    }
    VkPhysicalDeviceMemoryProperties mp;
    GetMP(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount && memtype == UINT32_MAX; i++)
        if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) memtype = i;
    if (bench) {
        enum layout l = L_G1;
        if (!strcmp(blayout, "g2")) l = L_G2;
        else if (!strcmp(blayout, "g16")) l = L_G16;
        else if (!strcmp(blayout, "all")) l = L_ALL;
        else if (strcmp(blayout, "g1")) DIE("--bench-layout g1|g2|g16|all");
        bench_create(bench, l);
        return 0;
    }
    printf("vk-first-wait: %s via %s, driver %u.%u (NVIDIA encoding), mode %s (%s), %d rounds, %d fills (consumer), "
           "%d fills (producer); families graphics %u x%u, compute %u x%u, transfer %u x%u\n", props.deviceName,
           icd ? icd : hal, props.driverVersion >> 22, (props.driverVersion >> 14) & 0xff, M->name, M->what, rounds, fills,
           a_fills, fam_g, cnt_g, fam_c, cnt_c, fam_t, cnt_t);

    struct dev p;
    int save_nf = no_fence_ext;
    no_fence_ext = 0; /* the producer always exports a fence */
    dev_create(&p, a_fills, L_G1, 'g', 0);
    no_fence_ext = save_nf;
    int bad = 0;
    for (int r = 0; r < rounds; r++) {
        struct result res = {0};
        ready = posted = 0;
        pthread_t t;
        pthread_create(&t, NULL, consumer, &res);
        pthread_mutex_lock(&mu);
        while (!ready) pthread_cond_wait(&cv, &mu);
        pthread_mutex_unlock(&mu);
        VkExportFenceCreateInfo e = {.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
                                     .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
        VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = &e};
        VkFence pf;
        MUST(p.CreateFence(p.d, &fci, NULL, &pf));
        submit(&p, VK_NULL_HANDLE, p.work, VK_NULL_HANDLE, pf);
        fa = fence_fd(&p, pf);
        struct pollfd qa = {.fd = fa, .events = POLLIN};
        int fa_pending = poll(&qa, 1, 0) == 0;
        pthread_mutex_lock(&mu);
        posted = 1;
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mu);
        pthread_join(t, NULL);
        MUST(p.QueueWaitIdle(p.q));
        p.DestroyFence(p.d, pf, NULL);
        double gap = res.t_idle - res.t_sig;
        int early = use_ts ? res.end_after_sig > 0.1 : gap > 2.0;
        bad += early;
        char tsb[160] = "";
        if (use_ts)
            snprintf(tsb, sizeof tsb, "; GPU work (%.2f ms) ended %.3f ms %s poll() saw Fr signalled (calibration +-%.3f ms)",
                     res.work_ms, res.end_after_sig > 0 ? res.end_after_sig : -res.end_after_sig,
                     res.end_after_sig > 0 ? "AFTER" : "before", res.cal_dev_ms);
        printf("round %d: Fa %s when posted; Fr %s at export, signalled at %.2f ms, queue idle at %.2f ms (gap %.2f)%s%s\n",
               r, fa_pending ? "pending" : "signalled", res.at_creation ? "SIGNALLED" : "pending", res.t_sig, res.t_idle,
               gap, tsb, early ? "  <-- EARLY FENCE" : "");
    }
    printf("vk-first-wait: mode %s via %s: %d of %d rounds had an exported sync_fd signalled before the work it covers\n",
           M->name, icd ? "ICD" : "HAL", bad, rounds);
    dev_destroy(&p);
    return bad;
}
