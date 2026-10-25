//! Vulkan/ICD function lookup and loader-private negotiation; graphics absent.
use super::*;

/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdNegotiateLoaderICDInterfaceVersion(p: *mut u32) -> vk::Result { unsafe {
    if p.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *p = (*p).min(5);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetInstanceProcAddr(
    h: vk::Instance,
    p: *const c_char,
) -> vk::PFN_vkVoidFunction { unsafe {
    vkGetInstanceProcAddr(h, p)
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetPhysicalDeviceProcAddr(
    h: vk::Instance,
    p: *const c_char,
) -> vk::PFN_vkVoidFunction { unsafe {
    if p.is_null() {
        return None;
    }
    let name = CStr::from_ptr(p).to_bytes();
    if !name.starts_with(b"vkGetPhysicalDevice")
        && !matches!(
            name,
            b"vkCreateDevice"
                | b"vkEnumerateDeviceExtensionProperties"
                | b"vkEnumerateDeviceLayerProperties"
        )
    {
        return None;
    }
    vkGetInstanceProcAddr(h, p)
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetDeviceProcAddr(
    h: vk::Device,
    p: *const c_char,
) -> vk::PFN_vkVoidFunction { unsafe {
    if h == vk::Device::null() || p.is_null() {
        return None;
    }
    let name = CStr::from_ptr(p).to_bytes();
    if name.starts_with(b"vkGetPhysicalDevice")
        || name.starts_with(b"vkEnumerate")
        || matches!(
            name,
            b"vkGetInstanceProcAddr"
                | b"vkCreateInstance"
                | b"vkDestroyInstance"
                | b"vkCreateDevice"
        )
    {
        return None;
    }
    lookup(p, false)
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetInstanceProcAddr(
    h: vk::Instance,
    p: *const c_char,
) -> vk::PFN_vkVoidFunction { unsafe {
    lookup(p, h == vk::Instance::null())
}}
unsafe fn lookup(p: *const c_char, global: bool) -> vk::PFN_vkVoidFunction { unsafe {
    if p.is_null() {
        return None;
    }
    let name = CStr::from_ptr(p).to_bytes();
    macro_rules! entry{($($f:ident),*$(,)?)=>{$(if name==stringify!($f).as_bytes(){return Some(std::mem::transmute::<*const (),unsafe extern "system" fn()>($f as *const ()));})*};}
    entry!(
        vkGetInstanceProcAddr,
        vkCreateInstance,
        vkEnumerateInstanceVersion,
        vkEnumerateInstanceExtensionProperties,
        vkEnumerateInstanceLayerProperties
    );
    if global {
        return None;
    }
    entry!(
        vkGetDeviceProcAddr,
        vkDestroyInstance,
        vkEnumeratePhysicalDevices,
        vkGetPhysicalDeviceProperties,
        vkGetPhysicalDeviceFeatures,
        vkGetPhysicalDeviceMemoryProperties,
        vkGetPhysicalDeviceQueueFamilyProperties,
        vkGetPhysicalDeviceFormatProperties,
        vkGetPhysicalDeviceImageFormatProperties,
        vkGetPhysicalDeviceSparseImageFormatProperties,
        vkEnumerateDeviceExtensionProperties,
        vkEnumerateDeviceLayerProperties,
        vkCreateDevice,
        vkDestroyDevice,
        vkGetDeviceQueue,
        vkDeviceWaitIdle,
        vkQueueWaitIdle,
        vkAllocateMemory,
        vkFreeMemory,
        vkMapMemory,
        vkUnmapMemory,
        vkFlushMappedMemoryRanges,
        vkInvalidateMappedMemoryRanges,
        vkGetDeviceMemoryCommitment,
        vkCreateBuffer,
        vkDestroyBuffer,
        vkGetBufferMemoryRequirements,
        vkBindBufferMemory,
        vkCreateShaderModule,
        vkDestroyShaderModule,
        vkCreateDescriptorSetLayout,
        vkDestroyDescriptorSetLayout,
        vkCreatePipelineLayout,
        vkDestroyPipelineLayout,
        vkCreateComputePipelines,
        vkDestroyPipeline,
        vkCreatePipelineCache,
        vkDestroyPipelineCache,
        vkGetPipelineCacheData,
        vkMergePipelineCaches,
        vkCreateDescriptorPool,
        vkDestroyDescriptorPool,
        vkResetDescriptorPool,
        vkAllocateDescriptorSets,
        vkFreeDescriptorSets,
        vkUpdateDescriptorSets,
        vkCreateCommandPool,
        vkDestroyCommandPool,
        vkResetCommandPool,
        vkAllocateCommandBuffers,
        vkFreeCommandBuffers,
        vkBeginCommandBuffer,
        vkEndCommandBuffer,
        vkResetCommandBuffer,
        vkCmdBindPipeline,
        vkCmdBindDescriptorSets,
        vkCmdPushConstants,
        vkCmdDispatch,
        vkCmdCopyBuffer,
        vkCmdFillBuffer,
        vkCmdUpdateBuffer,
        vkCmdPipelineBarrier,
        vkQueueSubmit,
        vkCreateFence,
        vkDestroyFence,
        vkGetFenceStatus,
        vkResetFences,
        vkWaitForFences,
        vkCreateSemaphore,
        vkDestroySemaphore
    );
    None
}}
