/* These are the actual public Mesa entrypoints from pinned vk_fence.c and
 * vk_semaphore.c, linked into this offline harness with assertions enabled. */
enum vk_device_timeline_mode test_native_timeline_mode(struct vk_physical_device *);

static void test_sync_properties(struct vk_physical_device *physical,
                                 const struct vk_sync_type *const *types)
{
   vk_object_base_init(NULL, &physical->base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   physical->supported_sync_types = types;
   assert(types[0] && types[1] && !types[2]);
   assert((types[0]->features & (VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE)) ==
          (VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE));
   assert(types[0]->import_opaque_fd && types[0]->export_opaque_fd);
   assert(!types[0]->import_sync_file && !types[0]->export_sync_file);
   assert(types[1]->features & VK_SYNC_FEATURE_BINARY);
   assert(!(types[1]->features & VK_SYNC_FEATURE_TIMELINE));
   assert(!types[1]->import_opaque_fd && !types[1]->export_opaque_fd);
   assert(types[1]->import_sync_file && types[1]->export_sync_file);
   assert(test_native_timeline_mode(physical) == VK_DEVICE_TIMELINE_MODE_NATIVE);
   VkPhysicalDevice handle = vk_physical_device_to_handle(physical);
   const VkExternalFenceHandleTypeFlagBits fence_types[] = {
      VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT,
      VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(fence_types); i++) {
      VkPhysicalDeviceExternalFenceInfo query = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
         .handleType = fence_types[i],
      };
      VkExternalFenceProperties result = {
         .sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES,
      };
      vk_common_GetPhysicalDeviceExternalFenceProperties(handle, &query, &result);
      assert(result.externalFenceFeatures ==
         (VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT));
      assert(result.compatibleHandleTypes == fence_types[i]);
      assert(result.exportFromImportedHandleTypes == fence_types[i]);
   }
   const VkExternalSemaphoreHandleTypeFlagBits semaphore_types[] = {
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
   };
   for (unsigned timeline = 0; timeline < 2; timeline++) {
      for (unsigned i = 0; i < ARRAY_SIZE(semaphore_types); i++) {
         VkSemaphoreTypeCreateInfo type = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY,
         };
         VkPhysicalDeviceExternalSemaphoreInfo query = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            .pNext = &type, .handleType = semaphore_types[i],
         };
         VkExternalSemaphoreProperties result = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
         };
         vk_common_GetPhysicalDeviceExternalSemaphoreProperties(handle, &query, &result);
         if (timeline && i == 1) {
            assert(result.externalSemaphoreFeatures == 0 && result.compatibleHandleTypes == 0 &&
                   result.exportFromImportedHandleTypes == 0);
         } else {
            assert(result.externalSemaphoreFeatures ==
               (VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT));
            assert(result.compatibleHandleTypes == semaphore_types[i]);
            assert(result.exportFromImportedHandleTypes == semaphore_types[i]);
         }
      }
   }
   vk_object_base_finish(&physical->base);
   puts("PASS: actual Mesa fence/semaphore properties expose same-type-only FD masks; one native timeline type");
}
