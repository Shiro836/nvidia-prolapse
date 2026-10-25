//! Source SPIR-V compilation, owned layout metadata and buffer descriptors.
use super::*;

/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateShaderModule(
    _h: vk::Device,
    p: *const vk::ShaderModuleCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::ShaderModule,
) -> vk::Result { unsafe {
    *o = vk::ShaderModule::null();
    let p = &*p;
    if !a.is_null()
        || !p.p_next.is_null()
        || !p.flags.is_empty()
        || p.code_size < 20
        || !p.code_size.is_multiple_of(4)
        || p.p_code.is_null()
    {
        return BAD;
    }
    let words = std::slice::from_raw_parts(p.p_code, p.code_size / 4);
    if words[0] != 0x07230203 {
        return vk::Result::ERROR_INVALID_SHADER_NV;
    }
    let mut s = state();
    let id = s.id();
    s.shaders.insert(id, words.to_vec());
    *o = vk::ShaderModule::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyShaderModule(
    _h: vk::Device,
    m: vk::ShaderModule,
    _a: *const vk::AllocationCallbacks,
) {
    state().shaders.remove(&m.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateDescriptorSetLayout(
    _h: vk::Device,
    p: *const vk::DescriptorSetLayoutCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::DescriptorSetLayout,
) -> vk::Result { unsafe {
    *o = vk::DescriptorSetLayout::null();
    let p = &*p;
    if !a.is_null() || !p.p_next.is_null() || !p.flags.is_empty() {
        return BAD;
    }
    let mut bindings = HashMap::new();
    for b in slice(p.p_bindings, p.binding_count) {
        if b.binding >= 8
            || b.descriptor_type != vk::DescriptorType::STORAGE_BUFFER
            || b.descriptor_count != 1
            || b.stage_flags != vk::ShaderStageFlags::COMPUTE
            || !b.p_immutable_samplers.is_null()
            || bindings.insert(b.binding, 1).is_some()
        {
            return BAD;
        }
    }
    let mut s = state();
    let id = s.id();
    s.layouts.insert(id, Layout { bindings });
    *o = vk::DescriptorSetLayout::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyDescriptorSetLayout(
    _h: vk::Device,
    l: vk::DescriptorSetLayout,
    _a: *const vk::AllocationCallbacks,
) {
    state().layouts.remove(&l.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreatePipelineLayout(
    _h: vk::Device,
    p: *const vk::PipelineLayoutCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::PipelineLayout,
) -> vk::Result { unsafe {
    *o = vk::PipelineLayout::null();
    let p = &*p;
    if !a.is_null() || !p.p_next.is_null() || !p.flags.is_empty() || p.set_layout_count > 1 {
        return BAD;
    }
    let mut push = 0;
    for r in slice(p.p_push_constant_ranges, p.push_constant_range_count) {
        if r.stage_flags != vk::ShaderStageFlags::COMPUTE
            || r.offset + r.size > 128
            || !r.offset.is_multiple_of(4)
            || !r.size.is_multiple_of(4)
        {
            return BAD;
        }
        push = push.max(r.offset + r.size)
    }
    let mut s = state();
    let set = if p.set_layout_count == 1 {
        let id = (*p.p_set_layouts).as_raw();
        if !s.layouts.contains_key(&id) {
            return BAD;
        }
        Some(s.layouts[&id].clone())
    } else {
        None
    };
    let id = s.id();
    s.pipeline_layouts.insert(id, PipelineLayout { set, push });
    *o = vk::PipelineLayout::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyPipelineLayout(
    _h: vk::Device,
    l: vk::PipelineLayout,
    _a: *const vk::AllocationCallbacks,
) {
    state().pipeline_layouts.remove(&l.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateComputePipelines(
    h: vk::Device,
    _cache: vk::PipelineCache,
    n: u32,
    p: *const vk::ComputePipelineCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::Pipeline,
) -> vk::Result { unsafe {
    let mut s = state();
    for j in 0..n as usize {
        *o.add(j) = vk::Pipeline::null()
    }
    if !a.is_null() {
        return BAD;
    }
    for (j, p) in slice(p, n).iter().enumerate() {
        if !p.p_next.is_null()
            || !p.flags.is_empty()
            || !p.stage.p_next.is_null()
            || !p.stage.flags.is_empty()
            || p.stage.stage != vk::ShaderStageFlags::COMPUTE
            || !p.stage.p_specialization_info.is_null()
            || p.stage.p_name.is_null()
            || !s.pipeline_layouts.contains_key(&p.layout.as_raw())
        {
            return BAD;
        }
        let Some(words) = s.shaders.get(&p.stage.module.as_raw()) else {
            return BAD;
        };
        let shader = match compiler::compile(words, CStr::from_ptr(p.stage.p_name)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("native Vulkan shader compile: {e}");
                return vk::Result::ERROR_INVALID_SHADER_NV;
            }
        };
        let layout = s.pipeline_layouts[&p.layout.as_raw()].clone();
        let id = s.id();
        s.pipelines.insert(
            id,
            Pipeline {
                device: h.as_raw(),
                layout,
                shader,
            },
        );
        *o.add(j) = vk::Pipeline::from_raw(id)
    }
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyPipeline(
    _h: vk::Device,
    p: vk::Pipeline,
    _a: *const vk::AllocationCallbacks,
) {
    state().pipelines.remove(&p.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreatePipelineCache(
    h: vk::Device,
    p: *const vk::PipelineCacheCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::PipelineCache,
) -> vk::Result { unsafe {
    *o = vk::PipelineCache::null();
    let p = &*p;
    if !a.is_null() || !p.p_next.is_null() || !p.flags.is_empty() || p.initial_data_size != 0 {
        return BAD;
    }
    let mut s = state();
    let id = s.id();
    s.caches.insert(id, h.as_raw());
    *o = vk::PipelineCache::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyPipelineCache(
    _h: vk::Device,
    c: vk::PipelineCache,
    _a: *const vk::AllocationCallbacks,
) {
    state().caches.remove(&c.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetPipelineCacheData(
    h: vk::Device,
    c: vk::PipelineCache,
    n: *mut usize,
    p: *mut c_void,
) -> vk::Result { unsafe {
    let s = state();
    if s.caches.get(&c.as_raw()) != Some(&h.as_raw()) {
        return BAD;
    }
    let Some(b) = backend(&s, h.as_raw()) else {
        return BAD;
    };
    // An empty cache is still serialized with the required version-one header.
    let mut bytes = [0u8; 32];
    for (j, word) in [32u32, 1, b.info().vendor_id, b.info().device_id]
        .iter()
        .enumerate()
    {
        bytes[j * 4..j * 4 + 4].copy_from_slice(&word.to_le_bytes())
    }
    bytes[16..].copy_from_slice(&CACHE_UUID);
    if p.is_null() {
        *n = bytes.len();
        return vk::Result::SUCCESS;
    }
    let count = (*n).min(bytes.len());
    ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast(), count);
    *n = count;
    if count < bytes.len() {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkMergePipelineCaches(
    _h: vk::Device,
    _dst: vk::PipelineCache,
    _n: u32,
    _p: *const vk::PipelineCache,
) -> vk::Result {
    BAD
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateDescriptorPool(
    h: vk::Device,
    p: *const vk::DescriptorPoolCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::DescriptorPool,
) -> vk::Result { unsafe {
    *o = vk::DescriptorPool::null();
    let p = &*p;
    if !a.is_null()
        || !p.p_next.is_null()
        || !(p.flags & !vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET).is_empty()
        || p.max_sets == 0
        || slice(p.p_pool_sizes, p.pool_size_count)
            .iter()
            .any(|p| p.ty != vk::DescriptorType::STORAGE_BUFFER)
    {
        return BAD;
    }
    let mut s = state();
    let id = s.id();
    s.descriptor_pools.insert(id, (h.as_raw(), p.max_sets));
    *o = vk::DescriptorPool::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyDescriptorPool(
    _h: vk::Device,
    p: vk::DescriptorPool,
    _a: *const vk::AllocationCallbacks,
) {
    let mut s = state();
    s.sets.retain(|_, v| v.pool != p.as_raw());
    s.descriptor_pools.remove(&p.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkResetDescriptorPool(
    _h: vk::Device,
    p: vk::DescriptorPool,
    flags: vk::DescriptorPoolResetFlags,
) -> vk::Result {
    if !flags.is_empty() {
        return BAD;
    }
    state().sets.retain(|_, v| v.pool != p.as_raw());
    vk::Result::SUCCESS
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkAllocateDescriptorSets(
    h: vk::Device,
    p: *const vk::DescriptorSetAllocateInfo,
    o: *mut vk::DescriptorSet,
) -> vk::Result { unsafe {
    let p = &*p;
    let mut s = state();
    let Some(&(dev, max)) = s.descriptor_pools.get(&p.descriptor_pool.as_raw()) else {
        return BAD;
    };
    if !p.p_next.is_null()
        || dev != h.as_raw()
        || s.sets
            .values()
            .filter(|v| v.pool == p.descriptor_pool.as_raw())
            .count()
            + p.descriptor_set_count as usize
            > max as usize
    {
        return vk::Result::ERROR_OUT_OF_POOL_MEMORY;
    }
    let layouts = slice(p.p_set_layouts, p.descriptor_set_count);
    if layouts.iter().any(|l| !s.layouts.contains_key(&l.as_raw())) {
        return BAD;
    }
    for (j, l) in layouts.iter().enumerate() {
        let layout = s.layouts[&l.as_raw()].clone();
        let id = s.id();
        s.sets.insert(
            id,
            DescriptorSet {
                device: dev,
                pool: p.descriptor_pool.as_raw(),
                layout,
                buffers: HashMap::new(),
            },
        );
        *o.add(j) = vk::DescriptorSet::from_raw(id)
    }
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkFreeDescriptorSets(
    _h: vk::Device,
    _pool: vk::DescriptorPool,
    n: u32,
    p: *const vk::DescriptorSet,
) -> vk::Result { unsafe {
    let mut s = state();
    for d in slice(p, n) {
        s.sets.remove(&d.as_raw());
    }
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkUpdateDescriptorSets(
    h: vk::Device,
    n: u32,
    p: *const vk::WriteDescriptorSet,
    cn: u32,
    cp: *const vk::CopyDescriptorSet,
) { unsafe {
    let mut s = state();
    for w in slice(p, n) {
        let valid = s
            .sets
            .get(&w.dst_set.as_raw())
            .filter(|d| d.device == h.as_raw())
            .map(|d| &d.layout)
            .is_some_and(|l| l.bindings.contains_key(&w.dst_binding));
        if !valid
            || !w.p_next.is_null()
            || w.descriptor_type != vk::DescriptorType::STORAGE_BUFFER
            || w.descriptor_count != 1
            || w.dst_array_element != 0
            || w.p_buffer_info.is_null()
        {
            eprintln!("native Vulkan: rejected unsupported descriptor write");
            if let Some(d) = s.sets.get_mut(&w.dst_set.as_raw()) {
                d.buffers.clear()
            }
            continue;
        }
        let b = &*w.p_buffer_info;
        if let Some(d) = s.sets.get_mut(&w.dst_set.as_raw()) {
            d.buffers
                .insert(w.dst_binding, (b.buffer.as_raw(), b.offset, b.range));
        }
    }
    for c in slice(cp, cn) {
        let value = if c.descriptor_count == 1
            && c.src_array_element == 0
            && c.dst_array_element == 0
            && c.p_next.is_null()
        {
            s.sets
                .get(&c.src_set.as_raw())
                .and_then(|d| d.buffers.get(&c.src_binding))
                .copied()
        } else {
            None
        };
        if let Some(d) = s.sets.get_mut(&c.dst_set.as_raw()) {
            if let Some(v) = value {
                d.buffers.insert(c.dst_binding, v);
            } else {
                d.buffers.clear();
                eprintln!("native Vulkan: rejected unsupported descriptor copy");
            }
        }
    }
}}
