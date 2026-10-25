//! Host-coherent VRAM allocations, buffer suballocation and explicit mapping.
use super::*;

/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkAllocateMemory(
    h: vk::Device,
    p: *const vk::MemoryAllocateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::DeviceMemory,
) -> vk::Result {
    unsafe {
        *o = vk::DeviceMemory::null();
        let p = &*p;
        if !a.is_null() || !p.p_next.is_null() || p.memory_type_index != 0 || p.allocation_size == 0
        {
            return BAD;
        }
        let mut s = state();
        let Some(b) = backend(&s, h.as_raw()) else {
            return BAD;
        };
        match b.allocate(p.allocation_size as usize) {
            Ok(m) => {
                let id = s.id();
                s.memories.insert(
                    id,
                    Memory {
                        device: h.as_raw(),
                        id: m.id,
                        size: m.size as u64,
                        cpu: m.cpu_address,
                        mapped: false,
                    },
                );
                *o = vk::DeviceMemory::from_raw(id);
                vk::Result::SUCCESS
            }
            Err(e) => failure("allocate memory", e),
        }
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkFreeMemory(
    h: vk::Device,
    m: vk::DeviceMemory,
    _a: *const vk::AllocationCallbacks,
) {
    let mut s = state();
    if let Some(m) = s.memories.remove(&m.as_raw()) {
        if let Some(b) = backend(&s, h.as_raw()) {
            if let Err(e) = b.free(m.id) {
                let _ = failure("free memory", e);
            }
        }
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkMapMemory(
    h: vk::Device,
    m: vk::DeviceMemory,
    offset: u64,
    size: u64,
    flags: vk::MemoryMapFlags,
    o: *mut *mut c_void,
) -> vk::Result {
    unsafe {
        *o = ptr::null_mut();
        let mut s = state();
        let Some(m) = s.memories.get_mut(&m.as_raw()) else {
            return BAD;
        };
        let size = if size == vk::WHOLE_SIZE {
            m.size.saturating_sub(offset)
        } else {
            size
        };
        if m.device != h.as_raw()
            || m.mapped
            || !flags.is_empty()
            || size == 0
            || offset.checked_add(size).is_none_or(|n| n > m.size)
        {
            return vk::Result::ERROR_MEMORY_MAP_FAILED;
        }
        m.mapped = true;
        *o = (m.cpu + offset as usize) as *mut c_void;
        vk::Result::SUCCESS
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkUnmapMemory(_h: vk::Device, m: vk::DeviceMemory) {
    if let Some(m) = state().memories.get_mut(&m.as_raw()) {
        m.mapped = false;
    }
}
unsafe fn mapped_ranges(h: vk::Device, n: u32, p: *const vk::MappedMemoryRange) -> vk::Result {
    unsafe {
        let s = state();
        for r in slice(p, n) {
            let Some(m) = s.memories.get(&r.memory.as_raw()) else {
                return BAD;
            };
            let size = if r.size == vk::WHOLE_SIZE {
                m.size.saturating_sub(r.offset)
            } else {
                r.size
            };
            if !r.p_next.is_null()
                || m.device != h.as_raw()
                || !m.mapped
                || r.offset.checked_add(size).is_none_or(|v| v > m.size)
            {
                return BAD;
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        vk::Result::SUCCESS
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkFlushMappedMemoryRanges(
    h: vk::Device,
    n: u32,
    p: *const vk::MappedMemoryRange,
) -> vk::Result {
    unsafe { mapped_ranges(h, n, p) }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkInvalidateMappedMemoryRanges(
    h: vk::Device,
    n: u32,
    p: *const vk::MappedMemoryRange,
) -> vk::Result {
    unsafe { mapped_ranges(h, n, p) }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetDeviceMemoryCommitment(
    _h: vk::Device,
    m: vk::DeviceMemory,
    p: *mut u64,
) {
    unsafe { *p = state().memories.get(&m.as_raw()).map_or(0, |m| m.size) }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateBuffer(
    h: vk::Device,
    p: *const vk::BufferCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::Buffer,
) -> vk::Result {
    unsafe {
        *o = vk::Buffer::null();
        let p = &*p;
        let allowed = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        if !a.is_null()
            || !p.p_next.is_null()
            || !p.flags.is_empty()
            || p.size == 0
            || p.size > (1 << 27)
            || !(p.usage & !allowed).is_empty()
            || p.sharing_mode != vk::SharingMode::EXCLUSIVE
        {
            return BAD;
        }
        let mut s = state();
        if !s.devices.contains_key(&h.as_raw()) {
            return BAD;
        }
        let id = s.id();
        s.buffers.insert(
            id,
            Buffer {
                device: h.as_raw(),
                size: p.size,
                binding: None,
            },
        );
        *o = vk::Buffer::from_raw(id);
        vk::Result::SUCCESS
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyBuffer(
    _h: vk::Device,
    b: vk::Buffer,
    _a: *const vk::AllocationCallbacks,
) {
    state().buffers.remove(&b.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetBufferMemoryRequirements(
    _h: vk::Device,
    b: vk::Buffer,
    p: *mut vk::MemoryRequirements,
) {
    unsafe {
        *p = vk::MemoryRequirements::default();
        if let Some(b) = state().buffers.get(&b.as_raw()) {
            *p = vk::MemoryRequirements {
                size: b.size.next_multiple_of(4096),
                alignment: 4096,
                memory_type_bits: 1,
            };
        }
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkBindBufferMemory(
    h: vk::Device,
    b: vk::Buffer,
    m: vk::DeviceMemory,
    offset: u64,
) -> vk::Result {
    let mut s = state();
    let Some(mem) = s.memories.get(&m.as_raw()) else {
        return BAD;
    };
    let size = mem.size;
    if mem.device != h.as_raw() {
        return BAD;
    }
    let Some(b) = s.buffers.get_mut(&b.as_raw()) else {
        return BAD;
    };
    if b.device != h.as_raw()
        || b.binding.is_some()
        || !offset.is_multiple_of(4096)
        || offset.checked_add(b.size).is_none_or(|v| v > size)
    {
        return BAD;
    }
    b.binding = Some((m.as_raw(), offset));
    vk::Result::SUCCESS
}
