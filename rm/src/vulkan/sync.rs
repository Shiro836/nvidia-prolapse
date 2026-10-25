//! Fence polling and binary semaphores for the synchronous single queue.
use super::*;

/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateFence(
    h: vk::Device,
    p: *const vk::FenceCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::Fence,
) -> vk::Result { unsafe {
    *o = vk::Fence::null();
    let p = &*p;
    if !a.is_null()
        || !p.p_next.is_null()
        || !(p.flags & !vk::FenceCreateFlags::SIGNALED).is_empty()
    {
        return BAD;
    }
    let mut s = state();
    let id = s.id();
    s.fences.insert(
        id,
        (h.as_raw(), p.flags.contains(vk::FenceCreateFlags::SIGNALED)),
    );
    *o = vk::Fence::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyFence(
    _h: vk::Device,
    f: vk::Fence,
    _a: *const vk::AllocationCallbacks,
) {
    state().fences.remove(&f.as_raw());
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetFenceStatus(h: vk::Device, f: vk::Fence) -> vk::Result {
    match state().fences.get(&f.as_raw()) {
        Some(&(dev, true)) if dev == h.as_raw() => vk::Result::SUCCESS,
        Some(&(dev, false)) if dev == h.as_raw() => vk::Result::NOT_READY,
        _ => BAD,
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkResetFences(
    h: vk::Device,
    n: u32,
    p: *const vk::Fence,
) -> vk::Result { unsafe {
    let mut s = state();
    for f in slice(p, n) {
        let Some((dev, signal)) = s.fences.get_mut(&f.as_raw()) else {
            return BAD;
        };
        if *dev != h.as_raw() {
            return BAD;
        }
        *signal = false;
    }
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkWaitForFences(
    h: vk::Device,
    n: u32,
    p: *const vk::Fence,
    all: vk::Bool32,
    timeout: u64,
) -> vk::Result { unsafe {
    let start = std::time::Instant::now();
    loop {
        {
            let s = state();
            let mut signaled = 0;
            for f in slice(p, n) {
                let Some(&(dev, signal)) = s.fences.get(&f.as_raw()) else {
                    return BAD;
                };
                if dev != h.as_raw() {
                    return BAD;
                }
                signaled += u32::from(signal)
            }
            if (all != 0 && signaled == n) || (all == 0 && signaled > 0) {
                return vk::Result::SUCCESS;
            }
        }
        if start.elapsed().as_nanos() >= timeout as u128 {
            return vk::Result::TIMEOUT;
        }
        std::thread::sleep(std::time::Duration::from_nanos(timeout.min(100_000)));
    }
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateSemaphore(
    h: vk::Device,
    p: *const vk::SemaphoreCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::Semaphore,
) -> vk::Result { unsafe {
    *o = vk::Semaphore::null();
    let p = &*p;
    if !a.is_null() || !p.p_next.is_null() || !p.flags.is_empty() {
        return BAD;
    }
    let mut s = state();
    let id = s.id();
    s.semaphores.insert(id, (h.as_raw(), false));
    *o = vk::Semaphore::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroySemaphore(
    _h: vk::Device,
    p: vk::Semaphore,
    _a: *const vk::AllocationCallbacks,
) {
    state().semaphores.remove(&p.as_raw());
}
