//! Experimental compute-only Vulkan front end over the source-built RM backend.
//! This is not a conformant Vulkan implementation: images, graphics, WSI and
//! extensions are deliberately absent. Every accepted command executes real GPU
//! work; a recording error is sticky and prevents successful submission.
//! ABI types: pinned ash/Vulkan-Headers. Dispatch header: vk_icd.h ICD_LOADER_MAGIC.
#![allow(non_snake_case, unsafe_op_in_unsafe_fn)]
use crate::{
    backend::Backend,
    compiler::{self, Shader},
};
use ash::vk::{self, Handle};
use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr},
    ptr,
    sync::{Arc, Mutex, OnceLock},
};
const CACHE_UUID: [u8; 16] = *b"NVRUST-COMPUTE01";
const BAD: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
#[repr(C)]
struct Dispatch {
    loader: usize,
}
fn dispatch() -> u64 {
    Box::into_raw(Box::new(Dispatch { loader: 0x01CDC0DE })) as u64
}
unsafe fn drop_dispatch(h: u64) {
    if h != 0 {
        drop(Box::from_raw(h as *mut Dispatch));
    }
}
struct Instance {
    backend: Arc<Backend>,
    physical: u64,
}
struct Device {
    backend: Arc<Backend>,
    queue: u64,
}
struct Memory {
    device: u64,
    id: u64,
    size: u64,
    cpu: usize,
    mapped: bool,
}
#[derive(Clone)]
struct Buffer {
    device: u64,
    size: u64,
    binding: Option<(u64, u64)>,
}
#[derive(Clone)]
struct Layout {
    bindings: HashMap<u32, u32>,
}
#[derive(Clone)]
struct PipelineLayout {
    set: Option<Layout>,
    push: u32,
}
#[derive(Clone)]
struct DescriptorSet {
    device: u64,
    pool: u64,
    layout: Layout,
    buffers: HashMap<u32, (u64, u64, u64)>,
}
#[derive(Clone)]
struct Pipeline {
    device: u64,
    layout: PipelineLayout,
    shader: Shader,
}
#[derive(Clone)]
enum Command {
    Copy(u64, u64, Vec<vk::BufferCopy>),
    Dispatch {
        pipeline: u64,
        set: Option<u64>,
        push: Vec<u8>,
        groups: [u32; 3],
    },
    Fill(u64, u64, u64, u32),
    Update(u64, u64, Vec<u8>),
}
struct Commands {
    device: u64,
    pool: u64,
    recording: bool,
    executable: bool,
    error: bool,
    pipeline: Option<u64>,
    set: Option<u64>,
    push: Vec<u8>,
    commands: Vec<Command>,
}
struct Pool {
    device: u64,
}
#[derive(Default)]
struct State {
    next: u64,
    instances: HashMap<u64, Instance>,
    physical: HashMap<u64, u64>,
    devices: HashMap<u64, Device>,
    queues: HashMap<u64, u64>,
    memories: HashMap<u64, Memory>,
    buffers: HashMap<u64, Buffer>,
    shaders: HashMap<u64, Vec<u32>>,
    layouts: HashMap<u64, Layout>,
    pipeline_layouts: HashMap<u64, PipelineLayout>,
    pipelines: HashMap<u64, Pipeline>,
    descriptor_pools: HashMap<u64, (u64, u32)>,
    sets: HashMap<u64, DescriptorSet>,
    pools: HashMap<u64, Pool>,
    commands: HashMap<u64, Commands>,
    fences: HashMap<u64, (u64, bool)>,
    semaphores: HashMap<u64, (u64, bool)>,
    caches: HashMap<u64, u64>,
}
impl State {
    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }
}
fn state() -> std::sync::MutexGuard<'static, State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(State::default()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}
fn failure(label: &str, e: impl std::fmt::Display) -> vk::Result {
    eprintln!("native Vulkan {label}: {e}");
    vk::Result::ERROR_DEVICE_LOST
}
unsafe fn slice<'a, T>(p: *const T, n: u32) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n as usize)
    }
}
unsafe fn enumerate<T: Copy>(count: *mut u32, out: *mut T, values: &[T]) -> vk::Result {
    if count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    if out.is_null() {
        *count = values.len() as u32;
        return vk::Result::SUCCESS;
    }
    let n = (*count as usize).min(values.len());
    ptr::copy_nonoverlapping(values.as_ptr(), out, n);
    *count = n as u32;
    if n < values.len() {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}
unsafe fn loader_chain(mut p: *const c_void, allowed: i32) -> bool {
    let mut count = 0;
    while !p.is_null() {
        let h = &*(p as *const vk::BaseInStructure);
        if h.s_type.as_raw() != allowed || count >= 16 {
            return false;
        }
        p = h.p_next.cast();
        count += 1;
    }
    true
}
fn backend(s: &State, device: u64) -> Option<Arc<Backend>> {
    s.devices.get(&device).map(|d| d.backend.clone())
}
fn range(
    s: &State,
    dev: u64,
    b: u64,
    offset: u64,
    size: u64,
) -> Result<(u64, u64, u64), vk::Result> {
    let b = s.buffers.get(&b).ok_or(BAD)?;
    if b.device != dev {
        return Err(BAD);
    }
    let (m, base) = b.binding.ok_or(BAD)?;
    let m = s.memories.get(&m).ok_or(BAD)?;
    let size = if size == vk::WHOLE_SIZE {
        b.size.checked_sub(offset).ok_or(BAD)?
    } else {
        size
    };
    if offset.checked_add(size).is_none_or(|n| n > b.size)
        || base
            .checked_add(offset)
            .and_then(|v| v.checked_add(size))
            .is_none_or(|n| n > m.size)
    {
        return Err(BAD);
    }
    Ok((m.id, base + offset, size))
}

mod commands;
mod dispatch;
mod instance;
mod memory;
mod pipeline;
mod sync;
pub use commands::*;
pub use dispatch::*;
pub use instance::*;
pub use memory::*;
pub use pipeline::*;
pub use sync::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_instance_requests_fail_before_opening_gpu() {
        unsafe {
            let mut handle = vk::Instance::null();
            let extensions = [c"VK_UNKNOWN_native_test".as_ptr()];
            let info = vk::InstanceCreateInfo::default().enabled_extension_names(&extensions);
            assert_eq!(
                vkCreateInstance(&info, ptr::null(), &mut handle),
                vk::Result::ERROR_EXTENSION_NOT_PRESENT
            );
            assert_eq!(handle, vk::Instance::null());
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let info = vk::InstanceCreateInfo::default().application_info(&app);
            assert_eq!(
                vkCreateInstance(&info, ptr::null(), &mut handle),
                vk::Result::ERROR_INCOMPATIBLE_DRIVER
            );
            let extension = vk::BaseInStructure::default();
            let info = vk::InstanceCreateInfo {
                p_next: (&extension as *const vk::BaseInStructure).cast(),
                ..Default::default()
            };
            assert_eq!(vkCreateInstance(&info, ptr::null(), &mut handle), BAD);
            let allocator = vk::AllocationCallbacks::default();
            assert_eq!(
                vkCreateInstance(&vk::InstanceCreateInfo::default(), &allocator, &mut handle),
                BAD
            );
        }
    }

    #[test]
    fn unsupported_device_features_and_chains_are_rejected() {
        unsafe {
            let mut handle = vk::Device::null();
            let extensions = [c"VK_UNKNOWN_native_test".as_ptr()];
            let info = vk::DeviceCreateInfo::default().enabled_extension_names(&extensions);
            assert_eq!(
                vkCreateDevice(vk::PhysicalDevice::null(), &info, ptr::null(), &mut handle),
                vk::Result::ERROR_EXTENSION_NOT_PRESENT
            );
            let features = vk::PhysicalDeviceFeatures::default().geometry_shader(true);
            let queues = [vk::DeviceQueueCreateInfo::default()];
            let info = vk::DeviceCreateInfo::default()
                .enabled_features(&features)
                .queue_create_infos(&queues);
            assert_eq!(
                vkCreateDevice(vk::PhysicalDevice::null(), &info, ptr::null(), &mut handle),
                BAD
            );
            let extension = vk::BaseInStructure::default();
            let info = vk::DeviceCreateInfo {
                p_next: (&extension as *const vk::BaseInStructure).cast(),
                ..Default::default()
            };
            assert_eq!(
                vkCreateDevice(vk::PhysicalDevice::null(), &info, ptr::null(), &mut handle),
                BAD
            );
        }
    }

    #[test]
    fn dispatch_tables_do_not_expose_graphics_or_wrong_scope() {
        unsafe {
            assert!(
                vkGetInstanceProcAddr(vk::Instance::null(), c"vkCreateInstance".as_ptr()).is_some()
            );
            assert!(
                vkGetInstanceProcAddr(vk::Instance::null(), c"vkCmdDispatch".as_ptr()).is_none()
            );
            assert!(
                vkGetDeviceProcAddr(vk::Device::from_raw(1), c"vkCmdDispatch".as_ptr()).is_some()
            );
            assert!(
                vkGetDeviceProcAddr(vk::Device::from_raw(1), c"vkCreateDevice".as_ptr()).is_none()
            );
            assert!(vkGetDeviceProcAddr(vk::Device::from_raw(1), c"vkCmdDraw".as_ptr()).is_none());
            assert!(vk_icdGetPhysicalDeviceProcAddr(
                vk::Instance::from_raw(1),
                c"vkCmdDispatch".as_ptr()
            )
            .is_none());
        }
    }

    #[test]
    fn buffer_suballocation_ranges_and_overflow_are_checked() {
        let mut s = State::default();
        s.memories.insert(
            1,
            Memory {
                device: 17,
                id: 101,
                size: 16384,
                cpu: 0,
                mapped: false,
            },
        );
        s.buffers.insert(
            2,
            Buffer {
                device: 17,
                size: 8192,
                binding: Some((1, 4096)),
            },
        );
        assert_eq!(range(&s, 17, 2, 64, 1024), Ok((101, 4160, 1024)));
        assert_eq!(
            range(&s, 17, 2, 4096, vk::WHOLE_SIZE),
            Ok((101, 8192, 4096))
        );
        assert_eq!(range(&s, 18, 2, 0, 1), Err(BAD));
        assert_eq!(range(&s, 17, 2, u64::MAX, 2), Err(BAD));
        assert_eq!(range(&s, 17, 2, 8193, vk::WHOLE_SIZE), Err(BAD));
    }

    #[test]
    fn push_constants_are_snapshotted_and_recording_errors_stick() {
        let id = dispatch();
        let (layout, pipeline) = {
            let mut s = state();
            let layout = s.id();
            let pipeline = s.id();
            s.pipeline_layouts.insert(
                layout,
                PipelineLayout {
                    set: None,
                    push: 16,
                },
            );
            s.commands.insert(
                id,
                Commands {
                    device: 1,
                    pool: 0,
                    recording: false,
                    executable: false,
                    error: false,
                    pipeline: None,
                    set: None,
                    push: vec![0; 128],
                    commands: vec![],
                },
            );
            (layout, pipeline)
        };
        unsafe {
            let cmd = vk::CommandBuffer::from_raw(id);
            assert_eq!(
                vkBeginCommandBuffer(cmd, &vk::CommandBufferBeginInfo::default()),
                vk::Result::SUCCESS
            );
            vkCmdBindPipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                vk::Pipeline::from_raw(pipeline),
            );
            for v in [11u32, 27] {
                vkCmdPushConstants(
                    cmd,
                    vk::PipelineLayout::from_raw(layout),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    4,
                    (&v as *const u32).cast(),
                );
                vkCmdDispatch(cmd, 1, 1, 1);
            }
            assert_eq!(vkEndCommandBuffer(cmd), vk::Result::SUCCESS);
            {
                let s = state();
                let c = &s.commands[&id];
                for (command, expected) in c.commands.iter().zip([11u32, 27]) {
                    match command {
                        Command::Dispatch { push, .. } => {
                            assert_eq!(&push[..4], &expected.to_ne_bytes())
                        }
                        _ => panic!("wrong command"),
                    }
                }
            }
            assert_eq!(
                vkBeginCommandBuffer(cmd, &vk::CommandBufferBeginInfo::default()),
                vk::Result::SUCCESS
            );
            vkCmdBindPipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                vk::Pipeline::from_raw(pipeline),
            );
            vkCmdBindPipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                vk::Pipeline::from_raw(pipeline),
            );
            assert_eq!(vkEndCommandBuffer(cmd), BAD);
            let mut s = state();
            s.commands.remove(&id);
            s.pipeline_layouts.remove(&layout);
            drop_dispatch(id);
        }
    }

    #[test]
    fn fence_wait_releases_registry_for_signaling_thread() {
        let id = {
            let mut s = state();
            let id = s.id();
            s.fences.insert(id, (9, false));
            id
        };
        unsafe {
            let f = vk::Fence::from_raw(id);
            let d = vk::Device::from_raw(9);
            assert_eq!(vkGetFenceStatus(d, f), vk::Result::NOT_READY);
            assert_eq!(vkWaitForFences(d, 1, &f, vk::TRUE, 0), vk::Result::TIMEOUT);
            let thread = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(1));
                state().fences.get_mut(&id).unwrap().1 = true;
            });
            assert_eq!(
                vkWaitForFences(d, 1, &f, vk::TRUE, 1_000_000_000),
                vk::Result::SUCCESS
            );
            thread.join().unwrap();
            assert_eq!(vkResetFences(d, 1, &f), vk::Result::SUCCESS);
            assert_eq!(vkWaitForFences(d, 1, &f, vk::TRUE, 0), vk::Result::TIMEOUT);
            state().fences.remove(&id);
        }
    }
}
