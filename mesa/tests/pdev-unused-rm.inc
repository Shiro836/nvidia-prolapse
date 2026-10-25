/* Generated exact prototypes from nv_rm.h; unreachable test tripwires. */
int32_t nv_rm_context_create(struct NvRmDevice* device, uint32_t engines, uint64_t* context) { assert(!"unexpected nv_rm_context_create"); abort(); }
int32_t nv_rm_context_create_shared(struct NvRmDevice* device, uint32_t engines,
                            uint64_t share, uint64_t* context) { assert(!"unexpected nv_rm_context_create_shared"); abort(); }
int32_t nv_rm_exec(struct NvRmDevice* device,
           uint64_t context,
           const struct NvRmExec* pushes,
           uint32_t count) { assert(!"unexpected nv_rm_exec"); abort(); }
int32_t nv_rm_submit_waits(struct NvRmDevice* device, uint64_t context,
                   const struct NvRmExec* pushes, uint32_t count,
                   int drm_fd, const int* wait_fds, uint32_t wait_count,
                   uint64_t* token) { assert(!"unexpected nv_rm_submit_waits"); abort(); }
int32_t nv_rm_submit(struct NvRmDevice* device, uint64_t context,
             const struct NvRmExec* pushes, uint32_t count, uint64_t* token) { assert(!"unexpected nv_rm_submit"); abort(); }
int32_t nv_rm_poll(struct NvRmDevice* device, uint64_t context, uint64_t token,
           uint32_t* complete) { assert(!"unexpected nv_rm_poll"); abort(); }
int32_t nv_rm_context_fence(struct NvRmDevice* device, uint64_t context, uint64_t token,
                    int drm_fd, int* sync_fd) { assert(!"unexpected nv_rm_context_fence"); abort(); }
int32_t nv_rm_check_status(struct NvRmDevice* device) { assert(!"unexpected nv_rm_check_status"); abort(); }
int32_t nv_rm_timestamp(struct NvRmDevice* device, uint64_t* timestamp) { assert(!"unexpected nv_rm_timestamp"); abort(); }
int32_t nv_rm_export(struct NvRmDevice* device, uint64_t memory, int drm_fd, int* dma_buf_fd) { assert(!"unexpected nv_rm_export"); abort(); }
