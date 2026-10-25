//! Command recording snapshots push constants; descriptor contents resolve at submit.
//! Synchronous GPU completion implements queue ordering and memory barriers.
use super::*;

/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCreateCommandPool(
    h: vk::Device,
    p: *const vk::CommandPoolCreateInfo,
    a: *const vk::AllocationCallbacks,
    o: *mut vk::CommandPool,
) -> vk::Result { unsafe {
    *o = vk::CommandPool::null();
    let p = &*p;
    let allowed =
        vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
    if !a.is_null()
        || !p.p_next.is_null()
        || p.queue_family_index != 0
        || !(p.flags & !allowed).is_empty()
    {
        return BAD;
    }
    let mut s = state();
    let id = s.id();
    s.pools.insert(id, Pool { device: h.as_raw() });
    *o = vk::CommandPool::from_raw(id);
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkDestroyCommandPool(
    _h: vk::Device,
    p: vk::CommandPool,
    _a: *const vk::AllocationCallbacks,
) { unsafe {
    let mut s = state();
    s.commands.retain(|h, c| {
        if c.pool == p.as_raw() {
            drop_dispatch(*h);
            false
        } else {
            true
        }
    });
    s.pools.remove(&p.as_raw());
}}
fn reset(c: &mut Commands) {
    c.recording = false;
    c.executable = false;
    c.error = false;
    c.pipeline = None;
    c.set = None;
    c.push = vec![0; 128];
    c.commands.clear();
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkResetCommandPool(
    _h: vk::Device,
    p: vk::CommandPool,
    _flags: vk::CommandPoolResetFlags,
) -> vk::Result {
    for c in state()
        .commands
        .values_mut()
        .filter(|c| c.pool == p.as_raw())
    {
        reset(c)
    }
    vk::Result::SUCCESS
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkAllocateCommandBuffers(
    h: vk::Device,
    p: *const vk::CommandBufferAllocateInfo,
    o: *mut vk::CommandBuffer,
) -> vk::Result { unsafe {
    let p = &*p;
    let mut s = state();
    if !p.p_next.is_null()
        || p.level != vk::CommandBufferLevel::PRIMARY
        || s.pools
            .get(&p.command_pool.as_raw())
            .is_none_or(|v| v.device != h.as_raw())
    {
        return BAD;
    }
    for j in 0..p.command_buffer_count as usize {
        let id = dispatch();
        s.commands.insert(
            id,
            Commands {
                device: h.as_raw(),
                pool: p.command_pool.as_raw(),
                recording: false,
                executable: false,
                error: false,
                pipeline: None,
                set: None,
                push: vec![0; 128],
                commands: Vec::new(),
            },
        );
        *o.add(j) = vk::CommandBuffer::from_raw(id)
    }
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkFreeCommandBuffers(
    _h: vk::Device,
    _pool: vk::CommandPool,
    n: u32,
    p: *const vk::CommandBuffer,
) { unsafe {
    let mut s = state();
    for c in slice(p, n) {
        if s.commands.remove(&c.as_raw()).is_some() {
            drop_dispatch(c.as_raw())
        }
    }
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkBeginCommandBuffer(
    h: vk::CommandBuffer,
    p: *const vk::CommandBufferBeginInfo,
) -> vk::Result { unsafe {
    let p = &*p;
    let mut s = state();
    let Some(c) = s.commands.get_mut(&h.as_raw()) else {
        return BAD;
    };
    if c.recording
        || !p.p_next.is_null()
        || !p.p_inheritance_info.is_null()
        || !(p.flags
            & !(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                | vk::CommandBufferUsageFlags::SIMULTANEOUS_USE))
            .is_empty()
    {
        return BAD;
    }
    reset(c);
    c.recording = true;
    vk::Result::SUCCESS
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkEndCommandBuffer(h: vk::CommandBuffer) -> vk::Result {
    let mut s = state();
    let Some(c) = s.commands.get_mut(&h.as_raw()) else {
        return BAD;
    };
    if !c.recording {
        return BAD;
    }
    c.recording = false;
    c.executable = !c.error;
    if c.error {
        BAD
    } else {
        vk::Result::SUCCESS
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkResetCommandBuffer(
    h: vk::CommandBuffer,
    _flags: vk::CommandBufferResetFlags,
) -> vk::Result {
    let mut s = state();
    let Some(c) = s.commands.get_mut(&h.as_raw()) else {
        return BAD;
    };
    reset(c);
    vk::Result::SUCCESS
}
fn record(h: vk::CommandBuffer, f: impl FnOnce(&mut Commands)) {
    if let Some(c) = state().commands.get_mut(&h.as_raw()) {
        if !c.recording {
            c.error = true
        } else {
            f(c)
        }
    }
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdBindPipeline(
    h: vk::CommandBuffer,
    b: vk::PipelineBindPoint,
    p: vk::Pipeline,
) {
    record(h, |c| {
        if b != vk::PipelineBindPoint::COMPUTE {
            c.error = true
        } else {
            c.pipeline = Some(p.as_raw())
        }
    })
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdBindDescriptorSets(
    h: vk::CommandBuffer,
    b: vk::PipelineBindPoint,
    _l: vk::PipelineLayout,
    first: u32,
    n: u32,
    p: *const vk::DescriptorSet,
    dn: u32,
    _dp: *const u32,
) { unsafe {
    record(h, |c| {
        if b != vk::PipelineBindPoint::COMPUTE || first != 0 || n != 1 || dn != 0 {
            c.error = true
        } else {
            c.set = Some((*p).as_raw())
        }
    })
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdPushConstants(
    h: vk::CommandBuffer,
    l: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    p: *const c_void,
) { unsafe {
    let max = state()
        .pipeline_layouts
        .get(&l.as_raw())
        .map_or(0, |l| l.push);
    record(h, |c| {
        if stages != vk::ShaderStageFlags::COMPUTE
            || offset.checked_add(size).is_none_or(|n| n > max)
            || !offset.is_multiple_of(4)
            || !size.is_multiple_of(4)
        {
            c.error = true
        } else {
            c.push[offset as usize..(offset + size) as usize]
                .copy_from_slice(std::slice::from_raw_parts(p as *const u8, size as usize))
        }
    })
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdDispatch(h: vk::CommandBuffer, x: u32, y: u32, z: u32) {
    record(h, |c| {
        if [x, y, z].iter().any(|&v| v > 65535) {
            c.error = true;
            return;
        }
        if let Some(p) = c.pipeline {
            c.commands.push(Command::Dispatch {
                pipeline: p,
                set: c.set,
                push: c.push.clone(),
                groups: [x, y, z],
            })
        } else {
            c.error = true
        }
    })
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdCopyBuffer(
    h: vk::CommandBuffer,
    src: vk::Buffer,
    dst: vk::Buffer,
    n: u32,
    p: *const vk::BufferCopy,
) { unsafe {
    record(h, |c| {
        c.commands.push(Command::Copy(
            src.as_raw(),
            dst.as_raw(),
            slice(p, n).to_vec(),
        ))
    })
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdFillBuffer(
    h: vk::CommandBuffer,
    b: vk::Buffer,
    offset: u64,
    size: u64,
    data: u32,
) {
    record(h, |c| {
        c.commands
            .push(Command::Fill(b.as_raw(), offset, size, data))
    })
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdUpdateBuffer(
    h: vk::CommandBuffer,
    b: vk::Buffer,
    offset: u64,
    size: u64,
    p: *const c_void,
) { unsafe {
    record(h, |c| {
        if size > 65536 || !size.is_multiple_of(4) || !offset.is_multiple_of(4) {
            c.error = true
        } else {
            c.commands.push(Command::Update(
                b.as_raw(),
                offset,
                std::slice::from_raw_parts(p as *const u8, size as usize).to_vec(),
            ))
        }
    })
}}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkCmdPipelineBarrier(
    h: vk::CommandBuffer,
    _src: vk::PipelineStageFlags,
    _dst: vk::PipelineStageFlags,
    flags: vk::DependencyFlags,
    mn: u32,
    mp: *const vk::MemoryBarrier,
    bn: u32,
    bp: *const vk::BufferMemoryBarrier,
    in_: u32,
    _ip: *const vk::ImageMemoryBarrier,
) { unsafe {
    record(h, |c| {
        if in_ != 0
            || !(flags & !vk::DependencyFlags::BY_REGION).is_empty()
            || slice(mp, mn).iter().any(|v| !v.p_next.is_null())
            || slice(bp, bn).iter().any(|v| {
                !v.p_next.is_null()
                    || !matches!(
                        (v.src_queue_family_index, v.dst_queue_family_index),
                        (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED) | (0, 0)
                    )
            })
        {
            c.error = true
        }
    })
}}
fn execute(s: &State, dev: u64, b: &Backend, cmd: &Command) -> Result<(), vk::Result> {
    match cmd {
        Command::Copy(src, dst, regions) => {
            for r in regions {
                let (si, so, n) = range(s, dev, *src, r.src_offset, r.size)?;
                let (di, do_, _) = range(s, dev, *dst, r.dst_offset, r.size)?;
                b.copy(si, so, di, do_, n).map_err(|e| failure("copy", e))?;
            }
        }
        Command::Dispatch {
            pipeline,
            set,
            push,
            groups,
        } => {
            let p = s.pipelines.get(pipeline).ok_or(BAD)?;
            if p.device != dev {
                return Err(BAD);
            }
            let l = &p.layout;
            let mut bindings = Vec::new();
            if let Some(want) = &l.set {
                let d = s.sets.get(&set.ok_or(BAD)?).ok_or(BAD)?;
                if d.device != dev {
                    return Err(BAD);
                }
                let got = &d.layout;
                if want.bindings != got.bindings {
                    return Err(BAD);
                }
                for binding in want.bindings.keys() {
                    let (buf, off, len) = d.buffers.get(binding).ok_or(BAD)?;
                    let (id, offset, len) = range(s, dev, *buf, *off, *len)?;
                    bindings.push((*binding, id, offset, len));
                }
            }
            if groups.contains(&0) {
                return Ok(());
            }
            b.dispatch(&p.shader, &bindings, &push[..l.push as usize], *groups)
                .map_err(|e| failure("dispatch", e))?;
        }
        // CPU stores are a valid implementation of transfer commands for persistently
        // mapped, host-coherent allocations. Queue execution is synchronous; ordering
        // is maintained against every preceding and following GPU command.
        Command::Fill(buf, offset, len, value) => {
            let (id, off, n) = range(s, dev, *buf, *offset, *len)?;
            if !off.is_multiple_of(4) || !n.is_multiple_of(4) {
                return Err(BAD);
            }
            let m = s
                .memories
                .values()
                .find(|m| m.id == id && m.device == dev)
                .ok_or(BAD)?;
            for j in (0..n as usize).step_by(4) {
                unsafe { ptr::write_volatile((m.cpu + off as usize + j) as *mut u32, *value) }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
        Command::Update(buf, offset, data) => {
            let (id, off, _) = range(s, dev, *buf, *offset, data.len() as u64)?;
            let m = s
                .memories
                .values()
                .find(|m| m.id == id && m.device == dev)
                .ok_or(BAD)?;
            for (j, &v) in data.iter().enumerate() {
                unsafe { ptr::write_volatile((m.cpu + off as usize + j) as *mut u8, v) }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }
    Ok(())
}
/// # Safety
/// The caller must satisfy this Vulkan command's pointer, handle lifetime and
/// external synchronization requirements from the Vulkan specification.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkQueueSubmit(
    q: vk::Queue,
    n: u32,
    p: *const vk::SubmitInfo,
    f: vk::Fence,
) -> vk::Result { unsafe {
    let mut s = state();
    let Some(&d) = s.queues.get(&q.as_raw()) else {
        return BAD;
    };
    let Some(b) = backend(&s, d) else { return BAD };
    if f != vk::Fence::null()
        && !s
            .fences
            .get(&f.as_raw())
            .is_some_and(|&(dev, signal)| dev == d && !signal)
    {
        return BAD;
    }
    for sub in slice(p, n) {
        if !sub.p_next.is_null() {
            return BAD;
        }
        for sem in slice(sub.p_wait_semaphores, sub.wait_semaphore_count) {
            let Some((dev, signaled)) = s.semaphores.get_mut(&sem.as_raw()) else {
                return BAD;
            };
            if *dev != d || !*signaled {
                eprintln!(
                    "native Vulkan: synchronous queue cannot wait on an unsignaled semaphore"
                );
                return BAD;
            }
            *signaled = false;
        }
        for h in slice(sub.p_command_buffers, sub.command_buffer_count) {
            let Some(c) = s.commands.get(&h.as_raw()) else {
                return BAD;
            };
            if c.device != d || !c.executable || c.recording || c.error {
                return BAD;
            }
            for command in &c.commands {
                if let Err(e) = execute(&s, d, &b, command) {
                    return e;
                }
            }
        }
        for sem in slice(sub.p_signal_semaphores, sub.signal_semaphore_count) {
            let Some((dev, signaled)) = s.semaphores.get_mut(&sem.as_raw()) else {
                return BAD;
            };
            if *dev != d || *signaled {
                return BAD;
            }
            *signaled = true;
        }
    }
    if f != vk::Fence::null() {
        s.fences.get_mut(&f.as_raw()).unwrap().1 = true
    }
    vk::Result::SUCCESS
}}
