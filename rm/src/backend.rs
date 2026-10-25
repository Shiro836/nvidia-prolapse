//! Thread-safe Vulkan-facing API. One worker owns all thread-confined RM/GPU
//! objects. Callers exchange IDs and mapped addresses, never Rust references.
//! Submission is synchronous for now: success means the GPU release completed.
use crate::diag::diag;
use crate::{
    compiler::Shader,
    compute::ComputeJob,
    gpu::{self, AddressSpace, GpuMapping},
    rm::{Client, Device, Error, Mapping, Memory, Result},
};
use std::{collections::HashMap, rc::Rc, sync::mpsc, thread};

#[derive(Clone, Debug)]
pub struct Info {
    pub vendor_id: u32,
    pub device_id: u32,
    pub name: String,
    pub memory_size: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct Allocation {
    pub id: u64,
    pub address: u64,
    pub cpu_address: usize,
    pub size: usize,
}
struct Buffer {
    gpu: GpuMapping,
    _cpu: Mapping,
    memory: Rc<Memory>,
    requested_size: usize,
}
struct Context {
    buffers: HashMap<u64, Buffer>,
    space: AddressSpace,
    device: Rc<Device>,
    next_id: u64,
    allocated: u64,
    lost: bool,
    info: Info,
}
type Job = Box<dyn FnOnce(&mut Context) + Send>;
pub struct Backend {
    sender: Option<mpsc::Sender<Job>>,
    worker: Option<thread::JoinHandle<()>>,
    info: Info,
}
impl Backend {
    pub fn new() -> Result<Self> {
        let (sender, receiver) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("nv-native-rm".into())
            .spawn(move || {
                let mut context = match Context::new() {
                    Ok(context) => context,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if ready_tx.send(Ok(context.info.clone())).is_err() {
                    return;
                }
                while let Ok(job) = receiver.recv() {
                    job(&mut context);
                }
            })
            .map_err(|e| Error::Io("start native RM worker", e))?;
        match ready_rx.recv() {
            Ok(Ok(info)) => Ok(Self {
                sender: Some(sender),
                worker: Some(worker),
                info,
            }),
            result => {
                drop(sender);
                let _ = worker.join();
                match result {
                    Ok(Err(error)) => Err(error),
                    _ => Err(Error::Contract(
                        "native worker failed during startup".into(),
                    )),
                }
            }
        }
    }
    pub fn info(&self) -> &Info {
        &self.info
    }
    fn call<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Context) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Contract("native backend stopped".into()))?
            .send(Box::new(move |context| {
                let _ = tx.send(operation(context));
            }))
            .map_err(|_| Error::Contract("native GPU worker exited".into()))?;
        rx.recv()
            .map_err(|_| Error::Contract("native GPU worker failed during operation".into()))?
    }
    pub fn allocate(&self, size: usize) -> Result<Allocation> {
        self.call(move |context| context.allocate(size))
    }
    pub fn free(&self, id: u64) -> Result<()> {
        self.call(move |context| {
            let buffer = context
                .buffers
                .remove(&id)
                .ok_or_else(|| Error::Contract("free of unknown allocation".into()))?;
            context.allocated -= buffer.memory.size as u64;
            drop(buffer);
            Ok(())
        })
    }
    pub fn copy(
        &self,
        src: u64,
        src_offset: u64,
        dst: u64,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        self.call(move |context| context.copy(src, src_offset, dst, dst_offset, size))
    }
    pub fn dispatch(
        &self,
        shader: &Shader,
        bindings: &[(u32, u64, u64, u64)],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let shader = shader.clone();
        let bindings = bindings.to_vec();
        let push = push.to_vec();
        self.call(move |context| context.dispatch(&shader, &bindings, &push, groups))
    }
    pub fn idle(&self) -> Result<()> {
        self.call(|context| context.live())
    }
}
impl Drop for Backend {
    fn drop(&mut self) {
        self.sender.take(); // disconnect, then drain queued work before destruction
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                diag!("native RM worker panicked during cleanup");
            }
        }
    }
}
impl Context {
    fn new() -> Result<Self> {
        let client = Client::open()?;
        let card = client
            .cards()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Contract("no NVIDIA GPU".into()))?;
        let device = client.device(card)?;
        if device.sm_version()? != 0xa04 {
            return Err(Error::Contract(
                "experimental native Vulkan backend requires validated sm_120 GPU".into(),
            ));
        }
        let info = Info {
            vendor_id: card.pci.vendor as u32,
            device_id: card.pci.device as u32,
            name: "Experimental Rust RM compute (sm_120)".into(),
            // A deliberately bounded prototype heap, not a claim to expose all VRAM.
            memory_size: card.framebuffer_size.min(512 * 1024 * 1024),
        };
        if info.memory_size == 0 {
            return Err(Error::Contract(
                "RM returned zero framebuffer aperture".into(),
            ));
        }
        let space = AddressSpace::new(&device)?;
        Ok(Self {
            buffers: HashMap::new(),
            space,
            device,
            next_id: 1,
            allocated: 0,
            lost: false,
            info,
        })
    }
    fn live(&self) -> Result<()> {
        if self.lost {
            Err(Error::Contract(
                "native device lost after failed submission; recreate device".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn allocate(&mut self, size: usize) -> Result<Allocation> {
        self.live()?;
        let rounded = size
            .checked_add(4095)
            .map(|n| n & !4095)
            .ok_or_else(|| Error::Contract("allocation size overflow".into()))?;
        if size == 0
            || self
                .allocated
                .checked_add(rounded as u64)
                .is_none_or(|n| n > self.info.memory_size)
        {
            return Err(Error::Contract(
                "allocation exceeds bounded native heap".into(),
            ));
        }
        let memory = self.device.allocate_vram(size)?;
        let gpu = self.space.map(&memory)?;
        let cpu = memory.map()?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| Error::Contract("allocation ID exhaustion".into()))?;
        let result = Allocation {
            id,
            address: gpu.address(),
            cpu_address: cpu.cpu_address(),
            size,
        };
        self.allocated += memory.size as u64;
        self.buffers.insert(
            id,
            Buffer {
                gpu,
                _cpu: cpu,
                memory,
                requested_size: size,
            },
        );
        Ok(result)
    }
    fn address(&self, id: u64, offset: u64, size: u64) -> Result<u64> {
        let buffer = self
            .buffers
            .get(&id)
            .ok_or_else(|| Error::Contract("unknown allocation ID".into()))?;
        if size == 0
            || offset
                .checked_add(size)
                .is_none_or(|end| end > buffer.requested_size as u64)
        {
            return Err(Error::Contract(
                "buffer range exceeds allocated memory".into(),
            ));
        }
        buffer
            .gpu
            .address()
            .checked_add(offset)
            .ok_or_else(|| Error::Contract("GPU address overflow".into()))
    }
    fn completion(&self) -> Result<(GpuMapping, Mapping)> {
        let memory = self.device.allocate_vram(4096)?;
        let gpu = self.space.map(&memory)?;
        let mut cpu = memory.map()?;
        cpu.write_u32(0, 0)?;
        cpu.write_u32(4, 0)?;
        Ok((gpu, cpu))
    }
    fn copy(
        &mut self,
        src: u64,
        src_offset: u64,
        dst: u64,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        self.live()?;
        if size == 0 || size > u32::MAX as u64 {
            return Err(Error::Contract("copy size outside supported range".into()));
        }
        let source = self.address(src, src_offset, size)?;
        let destination = self.address(dst, dst_offset, size)?;
        if src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size {
            return Err(Error::Contract("overlapping GPU copy ranges".into()));
        }
        let (complete_gpu, complete_cpu) = self.completion()?;
        let mut words = Vec::new();
        gpu::method(&mut words, 4, 0, &[0xcab5]);
        let source = gpu::hi_lo(source);
        let destination = gpu::hi_lo(destination);
        gpu::method(
            &mut words,
            4,
            0x400,
            &[source[0], source[1], destination[0], destination[1]],
        );
        gpu::method(&mut words, 4, 0x418, &[size as u32]);
        gpu::method(&mut words, 4, 0x300, &[2 | (1 << 7) | (1 << 8)]);
        let release = gpu::hi_lo(complete_gpu.address());
        gpu::method(&mut words, 4, 0x240, &[release[0], release[1], 1]);
        gpu::method(&mut words, 4, 0x300, &[(1 << 2) | (1 << 3)]);
        let result = gpu::submit_once(&self.device, &self.space, 0xcab5, &complete_cpu, &words);
        if result.is_err() {
            self.lost = true;
        }
        result
    }
    fn dispatch(
        &mut self,
        shader: &Shader,
        bindings: &[(u32, u64, u64, u64)],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        self.live()?;
        if push.len() > 128 || bindings.len() > 8 {
            return Err(Error::Contract(
                "descriptor or push-constant limit exceeded".into(),
            ));
        }
        if groups.contains(&0) {
            return Ok(());
        }
        let mut constants = vec![0u8; 1152];
        let mut seen = 0u32;
        for &(binding, id, offset, range) in bindings {
            if binding >= 8 || seen & (1 << binding) != 0 || range > u32::MAX as u64 {
                return Err(Error::Contract(
                    "invalid/duplicate storage binding or range".into(),
                ));
            }
            let address = self.address(id, offset, range)?;
            seen |= 1 << binding;
            let slot = binding as usize * 16;
            constants[slot..slot + 8].copy_from_slice(&address.to_le_bytes());
            constants[slot + 8..slot + 12].copy_from_slice(&(range as u32).to_le_bytes());
        }
        constants[1024..1024 + push.len()].copy_from_slice(push);
        let (complete_gpu, complete_cpu) = self.completion()?;
        let job = ComputeJob::compiled(
            &self.device,
            &self.space,
            complete_gpu.address(),
            shader,
            &constants,
            groups,
        )?;
        let mut words = Vec::new();
        gpu::method(&mut words, 1, 0, &[0xcec0]);
        gpu::method(
            &mut words,
            1,
            0x7b0,
            &gpu::hi_lo(crate::compute::LOCAL_WINDOW),
        );
        gpu::method(
            &mut words,
            1,
            0x2a0,
            &gpu::hi_lo(crate::compute::SHARED_WINDOW),
        );
        gpu::method(&mut words, 1, 0x2b4, &[(job.address >> 8) as u32]);
        gpu::method(&mut words, 1, 0x2c0, &[9]);
        let result = gpu::submit_once(&self.device, &self.space, 0xcec0, &complete_cpu, &words);
        if result.is_err() {
            self.lost = true;
        }
        result
    }
}
