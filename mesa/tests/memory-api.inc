/* Fake memory bridge calls are enabled only by the cleanup regression. */
static struct memory_api_state {
   bool active, publish_loss, defer_release;
   int32_t map_status, unmap_status, free_status, alloc_status, unbind_status,
      va_free_status;
   uintptr_t address;
   unsigned maps, unmaps, frees, allocs, imports, binds, unbinds, va_frees;
   /* Last nv_rm_va_free: mappings handed over and the deferred release. */
   uint32_t va_count;
   uint64_t va_ids[4], va_address, va_size;
   NvRmVaRelease release;
   void *release_data;
} memory_api;
int32_t
nv_rm_bind(struct NvRmDevice *device, uint64_t memory, uint64_t offset,
           uint64_t size, uint64_t address, uint32_t kind, uint64_t *mapping)
{
   assert(memory_api.active && device == (struct NvRmDevice *)(uintptr_t)42);
   assert(memory == 55 && offset == 0 && size == 4096 && address >= NV_RM_VA_START);
   memory_api.binds++;
   *mapping = 66;
   return VK_SUCCESS;
}
int32_t
nv_rm_unbind(struct NvRmDevice *device, uint64_t mapping)
{
   assert(memory_api.active && device == (struct NvRmDevice *)(uintptr_t)42);
   assert(mapping == 66);
   memory_api.unbinds++;
   return memory_api.unbind_status;
}
/* Mimics the backend contract: the range goes back through the callback,
 * immediately or (defer_release) when the test runs it as "retirement". */
int32_t
nv_rm_va_free(struct NvRmDevice *device, const uint64_t *mappings, uint32_t count,
              uint64_t address, uint64_t size, NvRmVaRelease release, void *data)
{
   assert(memory_api.active && device == (struct NvRmDevice *)(uintptr_t)42);
   assert(release && data && address >= NV_RM_VA_START && size);
   assert(count <= ARRAY_SIZE(memory_api.va_ids) && (count == 0 || mappings));
   memory_api.va_frees++;
   memory_api.va_count = count;
   for (uint32_t i = 0; i < count; i++)
      memory_api.va_ids[i] = mappings[i];
   memory_api.va_address = address;
   memory_api.va_size = size;
   if (memory_api.va_free_status)
      return memory_api.va_free_status;
   if (memory_api.defer_release && count) {
      memory_api.release = release;
      memory_api.release_data = data;
   } else {
      release(data, address, size);
   }
   return VK_SUCCESS;
}
static void
memory_run_deferred_release(void)
{
   assert(memory_api.release);
   NvRmVaRelease release = memory_api.release;
   memory_api.release = NULL;
   release(memory_api.release_data, memory_api.va_address, memory_api.va_size);
}
static void
memory_check(struct NvRmDevice *device, uint64_t memory)
{
   assert(memory_api.active);
   assert(device == (struct NvRmDevice *)(uintptr_t)42 && memory == 55);
}
int32_t
nv_rm_map(struct NvRmDevice *device, uint64_t memory, uintptr_t *address)
{
   memory_check(device, memory);
   memory_api.maps++;
   if (memory_api.publish_loss)
      p_atomic_set(&fake.rm_lost, 1);
   *address = memory_api.address;
   return memory_api.map_status;
}
int32_t
nv_rm_unmap(struct NvRmDevice *device, uint64_t memory)
{
   memory_check(device, memory);
   memory_api.unmaps++;
   return memory_api.unmap_status;
}
int32_t
nv_rm_free(struct NvRmDevice *device, uint64_t memory)
{
   memory_check(device, memory);
   memory_api.frees++;
   return memory_api.free_status;
}
int32_t
nv_rm_alloc(struct NvRmDevice *device, uint64_t size, uint64_t alignment,
            uint32_t placement, uint32_t kind, uint32_t tile,
            struct NvRmAllocation *out)
{
   memory_check(device, 55);
   memory_api.allocs++;
   *out = (struct NvRmAllocation){ .id = 55, .size = size };
   return memory_api.alloc_status;
}
int32_t
nv_rm_import(struct NvRmDevice *device, int drm_fd, int fd,
             struct NvRmAllocation *out)
{
   memory_check(device, 55);
   assert(drm_fd == 99 && fd == 22);
   memory_api.imports++;
   *out = (struct NvRmAllocation){ .id = 55, .size = 4096 };
   return memory_api.alloc_status;
}
