//! Instance/device discovery and bounded compute-only capabilities.
use super::*;

/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceVersion(p: *mut u32) -> vk::Result {
    *p = vk::API_VERSION_1_0;
    vk::Result::SUCCESS
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceExtensionProperties(
    _layer: *const c_char,
    count: *mut u32,
    out: *mut vk::ExtensionProperties,
) -> vk::Result {
    enumerate(count, out, &[])
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceLayerProperties(
    count: *mut u32,
    out: *mut vk::LayerProperties,
) -> vk::Result {
    enumerate(count, out, &[])
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkCreateInstance(
    info: *const vk::InstanceCreateInfo,
    alloc: *const vk::AllocationCallbacks,
    out: *mut vk::Instance,
) -> vk::Result {
    *out = vk::Instance::null();
    let i = &*info;
    if i.enabled_extension_count != 0 {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    }
    if i.enabled_layer_count != 0 {
        return vk::Result::ERROR_LAYER_NOT_PRESENT;
    }
    // Loader-private create-info chains are consumed by the loader and may remain
    // on the ICD call; no public instance extensions are enabled by this driver.
    if !alloc.is_null() || !loader_chain(i.p_next, 47) || !i.flags.is_empty() {
        return BAD;
    }
    if !i.p_application_info.is_null() {
        let v = (*i.p_application_info).api_version;
        if !(*i.p_application_info).p_next.is_null() {
            return BAD;
        }
        if v != 0 && (vk::api_version_major(v) != 1 || vk::api_version_minor(v) != 0) {
            return vk::Result::ERROR_INCOMPATIBLE_DRIVER;
        }
    }
    let b = match Backend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => return failure("create instance", e),
    };
    let mut s = state();
    let h = dispatch();
    let p = dispatch();
    s.instances.insert(
        h,
        Instance {
            backend: b,
            physical: p,
        },
    );
    s.physical.insert(p, h);
    *out = vk::Instance::from_raw(h);
    vk::Result::SUCCESS
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkDestroyInstance(
    h: vk::Instance,
    _a: *const vk::AllocationCallbacks,
) {
    let mut s = state();
    if let Some(i) = s.instances.remove(&h.as_raw()) {
        s.physical.remove(&i.physical);
        drop_dispatch(i.physical);
        drop_dispatch(h.as_raw());
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkEnumeratePhysicalDevices(
    h: vk::Instance,
    n: *mut u32,
    p: *mut vk::PhysicalDevice,
) -> vk::Result {
    let s = state();
    let Some(i) = s.instances.get(&h.as_raw()) else {
        return BAD;
    };
    enumerate(n, p, &[vk::PhysicalDevice::from_raw(i.physical)])
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceProperties(
    h: vk::PhysicalDevice,
    p: *mut vk::PhysicalDeviceProperties,
) {
    let s = state();
    *p = vk::PhysicalDeviceProperties::default();
    let Some(i) = s.physical.get(&h.as_raw()).and_then(|i| s.instances.get(i)) else {
        return;
    };
    let info = i.backend.info();
    let o = &mut *p;
    o.api_version = vk::API_VERSION_1_0;
    o.driver_version = 1;
    o.pipeline_cache_uuid = CACHE_UUID;
    o.vendor_id = info.vendor_id;
    o.device_id = info.device_id;
    o.device_type = vk::PhysicalDeviceType::DISCRETE_GPU;
    let name = format!("Native Rust experimental compute ({})", info.name);
    for (d, b) in o.device_name.iter_mut().zip(name.bytes().take(255)) {
        *d = b as c_char
    }
    let l = &mut o.limits;
    l.max_memory_allocation_count = 4096;
    l.max_bound_descriptor_sets = 1;
    l.max_per_stage_descriptor_storage_buffers = 8;
    l.max_descriptor_set_storage_buffers = 8;
    l.max_per_stage_resources = 8;
    l.max_storage_buffer_range = 1 << 27;
    l.max_push_constants_size = 128;
    l.max_compute_shared_memory_size = 32768;
    l.max_compute_work_group_count = [65535; 3];
    l.max_compute_work_group_invocations = 1024;
    l.max_compute_work_group_size = [1024, 1024, 64];
    l.min_memory_map_alignment = 4096;
    l.min_storage_buffer_offset_alignment = 16;
    l.non_coherent_atom_size = 64;
    l.buffer_image_granularity = 4096;
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFeatures(
    _h: vk::PhysicalDevice,
    p: *mut vk::PhysicalDeviceFeatures,
) {
    *p = vk::PhysicalDeviceFeatures::default()
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceMemoryProperties(
    h: vk::PhysicalDevice,
    p: *mut vk::PhysicalDeviceMemoryProperties,
) {
    *p = vk::PhysicalDeviceMemoryProperties::default();
    let s = state();
    let Some(i) = s.physical.get(&h.as_raw()).and_then(|i| s.instances.get(i)) else {
        return;
    };
    (*p).memory_heap_count = 1;
    (*p).memory_heaps[0] = vk::MemoryHeap {
        size: i.backend.info().memory_size,
        flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
    };
    (*p).memory_type_count = 1;
    (*p).memory_types[0] = vk::MemoryType {
        property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        heap_index: 0,
    };
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceQueueFamilyProperties(
    _h: vk::PhysicalDevice,
    n: *mut u32,
    p: *mut vk::QueueFamilyProperties,
) {
    let q = vk::QueueFamilyProperties {
        queue_flags: vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
        queue_count: 1,
        timestamp_valid_bits: 0,
        min_image_transfer_granularity: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    let _ = enumerate(n, p, &[q]);
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFormatProperties(
    _h: vk::PhysicalDevice,
    _f: vk::Format,
    p: *mut vk::FormatProperties,
) {
    *p = vk::FormatProperties::default()
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceImageFormatProperties(
    _h: vk::PhysicalDevice,
    _f: vk::Format,
    _t: vk::ImageType,
    _tiling: vk::ImageTiling,
    _usage: vk::ImageUsageFlags,
    _flags: vk::ImageCreateFlags,
    _p: *mut vk::ImageFormatProperties,
) -> vk::Result {
    vk::Result::ERROR_FORMAT_NOT_SUPPORTED
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSparseImageFormatProperties(
    _h: vk::PhysicalDevice,
    _f: vk::Format,
    _t: vk::ImageType,
    _samples: vk::SampleCountFlags,
    _u: vk::ImageUsageFlags,
    _tiling: vk::ImageTiling,
    n: *mut u32,
    _p: *mut vk::SparseImageFormatProperties,
) {
    *n = 0
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkEnumerateDeviceExtensionProperties(
    _h: vk::PhysicalDevice,
    _layer: *const c_char,
    n: *mut u32,
    p: *mut vk::ExtensionProperties,
) -> vk::Result {
    enumerate(n, p, &[])
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkEnumerateDeviceLayerProperties(
    _h: vk::PhysicalDevice,
    n: *mut u32,
    p: *mut vk::LayerProperties,
) -> vk::Result {
    enumerate(n, p, &[])
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkCreateDevice(
    h: vk::PhysicalDevice,
    p: *const vk::DeviceCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::Device,
) -> vk::Result {
    *o = vk::Device::null();
    let p = &*p;
    if p.enabled_extension_count != 0 {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    }
    if !a.is_null()
        || !loader_chain(p.p_next, 48)
        || !p.flags.is_empty()
        || p.queue_create_info_count != 1
    {
        return BAD;
    }
    if !p.p_enabled_features.is_null() {
        let fs = std::slice::from_raw_parts(
            p.p_enabled_features as *const u32,
            std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4,
        );
        if fs.iter().any(|&v| v != 0) {
            return BAD;
        }
    }
    let q = &*p.p_queue_create_infos;
    if q.queue_family_index != 0 || q.queue_count != 1 || !q.flags.is_empty() || !q.p_next.is_null()
    {
        return BAD;
    }
    let mut s = state();
    let Some(b) = s
        .physical
        .get(&h.as_raw())
        .and_then(|i| s.instances.get(i))
        .map(|i| i.backend.clone())
    else {
        return BAD;
    };
    let h = dispatch();
    let q = dispatch();
    s.devices.insert(
        h,
        Device {
            backend: b,
            queue: q,
        },
    );
    s.queues.insert(q, h);
    *o = vk::Device::from_raw(h);
    vk::Result::SUCCESS
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkDestroyDevice(h: vk::Device, _a: *const vk::AllocationCallbacks) {
    let mut s = state();
    if let Some(d) = s.devices.remove(&h.as_raw()) {
        s.queues.remove(&d.queue);
        drop_dispatch(d.queue);
        drop_dispatch(h.as_raw());
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceQueue(h: vk::Device, f: u32, i: u32, o: *mut vk::Queue) {
    *o = vk::Queue::null();
    if f == 0 && i == 0 {
        if let Some(d) = state().devices.get(&h.as_raw()) {
            *o = vk::Queue::from_raw(d.queue)
        }
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkDeviceWaitIdle(h: vk::Device) -> vk::Result {
    let s = state();
    let Some(b) = backend(&s, h.as_raw()) else {
        return BAD;
    };
    match b.idle() {
        Ok(()) => vk::Result::SUCCESS,
        Err(e) => failure("idle", e),
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[no_mangle]
pub unsafe extern "system" fn vkQueueWaitIdle(h: vk::Queue) -> vk::Result {
    let d = state().queues.get(&h.as_raw()).copied();
    match d {
        Some(d) => vkDeviceWaitIdle(vk::Device::from_raw(d)),
        None => BAD,
    }
}
