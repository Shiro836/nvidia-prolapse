// SPDX-License-Identifier: MIT
//! Error type of the RM backend: the failure of an ioctl, an RM status, a broken contract,
//! a lost device or a bounded wait that ran out, with the operation that failed.
use crate::units::{ByteSize, ClassId, FenceValue, GpuVa};
use std::{io, time::Duration};

/// Why an RM backend operation failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A syscall or ioctl named by the first field failed.
    #[error("{0}: {1}")]
    Io(&'static str, #[source] io::Error),
    /// RM returned a non-zero `NV_STATUS` (open-gpu-kernel-modules `nvstatuscodes.h`).
    #[error("{operation}: RM status 0x{status:08x}")]
    Rm {
        /// The RM call that failed.
        operation: &'static str,
        /// Its raw status.
        status: u32,
    },
    /// The kernel, RM or the caller broke an interface contract.
    #[error("{0}")]
    Contract(String),
    /// The device or channel is unusable; the object must be destroyed.
    #[error("{0}")]
    DeviceLost(String),
    /// A bounded wait on the GPU, the kernel or another process ran out.
    #[error(transparent)]
    Wait(#[from] WaitError),
    /// A synchronization operation failed (payloads, the semaphore slot pool, the foreign
    /// watcher, CPU waiters).
    #[error(transparent)]
    Sync(#[from] SyncError),
    /// A GPU channel, its queue or its channel group failed.
    #[error(transparent)]
    Channel(#[from] ChannelError),
    /// Allocating, CPU-mapping or GPU-mapping memory failed.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

impl Error {
    /// The device or channel is unusable and the object must be destroyed: what the C ABI
    /// reports as `VK_ERROR_DEVICE_LOST`.
    pub fn is_device_lost(&self) -> bool {
        match self {
            Self::DeviceLost(_) => true,
            Self::Sync(error) => error.is_device_lost(),
            Self::Channel(error) => error.is_device_lost(),
            Self::Memory(error) => error.is_device_lost(),
            Self::Io(..) | Self::Rm { .. } | Self::Contract(_) | Self::Wait(_) => false,
        }
    }
}

/// Why a synchronization operation failed: the subsystem of payloads (binary, timeline,
/// imported `sync_file`, cross-process shared), their semaphore slot pool, the foreign
/// watcher thread and the CPU waiters.
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    /// A thread panicked while holding the named object's lock (the FFI guard marked the
    /// device lost); state behind a poisoned lock is never used again (RUST.md §8).
    #[error("{object} lock poisoned: a thread panicked while holding it")]
    Poisoned {
        /// The object whose lock is poisoned.
        object: &'static str,
    },
    /// The payload or watcher belongs to a device that was closed or lost.
    #[error("{0}: the device was closed or lost")]
    Closed(&'static str),
    /// An imported `sync_file` reported a producer error (a negative status, linux
    /// `sync_file.h` `sync_file_info.status`).
    #[error("imported sync_file producer error {status}")]
    ProducerError {
        /// The status the kernel reported.
        status: i32,
    },
    /// The producer of a cross-process shared payload marked it failed.
    #[error("shared synchronization producer failed")]
    SharedProducerFailed,
    /// The operation does not apply to this payload.
    #[error("{operation}: {reason}")]
    Invalid {
        /// The operation.
        operation: &'static str,
        /// Why it does not apply.
        reason: &'static str,
    },
    /// A binary payload's generation counter reached its limit.
    #[error("{operation}: binary payload generation exhausted")]
    GenerationExhausted {
        /// The operation that needed a new generation.
        operation: &'static str,
    },
}

impl SyncError {
    /// Whether this failure makes the device unusable (`Error::is_device_lost`).
    pub fn is_device_lost(&self) -> bool {
        match self {
            Self::Poisoned { .. }
            | Self::Closed(_)
            | Self::ProducerError { .. }
            | Self::SharedProducerFailed => true,
            Self::Invalid { .. } | Self::GenerationExhausted { .. } => false,
        }
    }
}

/// Why a GPU channel operation failed: RM objects of the channel and its engines, the
/// channel group (TSG) and its subcontexts, the GPFIFO ring and its doorbell, the host
/// methods a submission adds, and the queue's completion timeline.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    /// RM refused to allocate an object of `class` (`NV_ESC_RM_ALLOC`, `nvos.h` `NVOS21`/`NVOS64`).
    #[error("allocate RM class {class:#x}: {source}")]
    Allocate {
        /// The RM class.
        class: ClassId,
        /// The failed allocation.
        #[source]
        source: Box<Error>,
    },
    /// RM answered a class-list query (`NV2080_CTRL_CMD_GPU_GET_ENGINE_CLASSLIST`,
    /// `ctrl2080gpu.h`) outside its contract.
    #[error("RM engine class list: {reason}")]
    ClassList {
        /// What was wrong with the answer.
        reason: &'static str,
    },
    /// The device lacks a class the operation needs.
    #[error("{operation} requires RM class {class:#x}, absent from device")]
    MissingClass {
        /// The operation.
        operation: &'static str,
        /// The missing class.
        class: ClassId,
    },
    /// The channel's host class has no method family this backend encodes (published
    /// `cl{a06f,a16f,b06f,c06f,c36f}.h` and later).
    #[error("unimplemented host method class {class:#x}")]
    HostClass {
        /// The host (GPFIFO) class.
        class: ClassId,
    },
    /// The GPU generation or its host class lacks a feature the operation needs.
    #[error("{operation}: {reason}")]
    Unsupported {
        /// The operation.
        operation: &'static str,
        /// The missing feature.
        reason: &'static str,
    },
    /// The engine mask names no engine, or engines this backend does not create.
    #[error("unsupported GPU engine mask {mask:#x}")]
    EngineMask {
        /// The requested mask.
        mask: u32,
    },
    /// A channel joining another queue's channel group asked for engines other than compute.
    #[error(
        "a context in a shared TSG supports only the compute engine (mask 0x10), got {mask:#x}"
    )]
    SharedGroupEngine {
        /// The requested mask.
        mask: u32,
    },
    /// The channel group cannot take the object.
    #[error("TSG {reason}")]
    Group {
        /// Why.
        reason: &'static str,
    },
    /// A buffer the backend places itself would overlap the VA range Mesa allocates from.
    #[error("{what} collides with Mesa VA heap")]
    VaHeapCollision {
        /// The internal buffer.
        what: &'static str,
    },
    /// A host semaphore address is not 16-byte aligned or not below 2^40.
    #[error("unaligned or out-of-range host semaphore address {address:#x}")]
    SemaphoreAddress {
        /// The address.
        address: u64,
    },
    /// The host methods of a completion do not fit the reserved command slot.
    #[error("host methods of {words} words exceed the reserved command slot")]
    HostMethodsTooLarge {
        /// Their length in words.
        words: usize,
    },
    /// A batch has no GPFIFO entry or more than the ring, waits or signals can hold.
    #[error(
        "{operation}: invalid/incomplete GPFIFO batch ({entries} entries, {waits} waits, {signals} signals)"
    )]
    Batch {
        /// The operation.
        operation: &'static str,
        /// GPFIFO entries (pushes, plus the backend's own entries).
        entries: usize,
        /// Semaphore acquires.
        waits: usize,
        /// Semaphore releases.
        signals: usize,
    },
    /// A push is outside the GPFIFO entry encoding (`GP_ENTRY0`/`GP_ENTRY1`, `clc96f.h`: 4-byte aligned
    /// address below 2^40, length below 2^21 words, only the documented flags).
    #[error(
        "GPFIFO push outside hardware limits: address {address:#x} size {size:#x} flags {flags:#x}"
    )]
    Push {
        /// The push address.
        address: u64,
        /// Its size in bytes.
        size: u32,
        /// Its flags.
        flags: u32,
    },
    /// The 64-bit completion sequence would wrap.
    #[error("GPU completion sequence exhausted")]
    SequenceExhausted,
    /// A 32-bit semaphore payload cannot distinguish the next value from the oldest pending
    /// one.
    #[error("sem32 submission exceeds half range (next {token}, retired {retired})")]
    Sem32Range {
        /// The value the submission would signal.
        token: FenceValue,
        /// The newest retired value.
        retired: FenceValue,
    },
    /// Submissions on this channel were stopped (device loss).
    #[error("cannot submit to a stopped RM channel")]
    Stopped,
    /// The queue's channel was marked lost (its error notifier reported an RC error, or its
    /// completion broke the timeline).
    #[error("{operation}: persistent RM channel is lost")]
    Lost {
        /// The operation.
        operation: &'static str,
    },
    /// The same lost channel, reported by the channel's own loss scan, retirement and fence
    /// export. Their callers report the loss themselves; this status keeps the
    /// `VK_ERROR_INITIALIZATION_FAILED` these paths returned.
    #[error("{operation}: persistent RM channel is lost")]
    LostChannel {
        /// The operation.
        operation: &'static str,
    },
    /// RM's error notifier for the channel (an `NvNotification`, `nvos.h`) reports an RC
    /// error: `info32` holds the error type, `info16` and the time stamp what RM's error path
    /// filled in.
    #[error(
        "GPU channel error {error} at PUT={put} sequence={sequence}; device lost \
         (host class {host_class:#x}, notifier info16={info16:#06x} status={status:#06x} timeStamp={timestamp:#x})"
    )]
    Rc {
        /// `info32`: the RC error type.
        error: u32,
        /// The ring's PUT index.
        put: usize,
        /// The newest issued completion value.
        sequence: u64,
        /// The host (GPFIFO) class.
        host_class: u32,
        /// `info16`.
        info16: u32,
        /// `status`.
        status: u32,
        /// `timeStamp`.
        timestamp: u64,
    },
    /// A completion token the channel never issued.
    #[error("{operation}: completion token {token} was not issued (newest {issued})")]
    Token {
        /// The operation.
        operation: &'static str,
        /// The token.
        token: FenceValue,
        /// The newest issued token.
        issued: FenceValue,
    },
    /// The channel's completion values contradict its issued timeline.
    #[error("{reason}")]
    Timeline {
        /// The contradiction.
        reason: &'static str,
    },
    /// `GP_PUT` read back differently from what was written (BAR0 USERD without a
    /// write-through doorbell).
    #[error("GP_PUT read back {visible} after writing {put}; channel state unknown")]
    PutReadback {
        /// The value read.
        visible: u32,
        /// The value written.
        put: u32,
    },
    /// A modern host class without its doorbell mapping.
    #[error("missing modern doorbell")]
    Doorbell,
    /// A backend self-check read a different result from the GPU.
    #[error("GPU result mismatch word {word}: {actual:#x} != {expected:#x}")]
    ResultMismatch {
        /// The word index.
        word: usize,
        /// The value read.
        actual: u32,
        /// The value expected.
        expected: u32,
    },
}

impl ChannelError {
    /// Whether this failure makes the device unusable (`Error::is_device_lost`).
    pub fn is_device_lost(&self) -> bool {
        match self {
            Self::Lost { .. } => true,
            Self::Allocate { .. }
            | Self::ClassList { .. }
            | Self::MissingClass { .. }
            | Self::HostClass { .. }
            | Self::Unsupported { .. }
            | Self::EngineMask { .. }
            | Self::SharedGroupEngine { .. }
            | Self::Group { .. }
            | Self::VaHeapCollision { .. }
            | Self::SemaphoreAddress { .. }
            | Self::HostMethodsTooLarge { .. }
            | Self::Batch { .. }
            | Self::Push { .. }
            | Self::SequenceExhausted
            | Self::Sem32Range { .. }
            | Self::Stopped
            | Self::LostChannel { .. }
            | Self::Rc { .. }
            | Self::Token { .. }
            | Self::Timeline { .. }
            | Self::PutReadback { .. }
            | Self::Doorbell
            | Self::ResultMismatch { .. } => false,
        }
    }
}

/// Why a memory operation failed: RM allocations, their CPU mappings and the BAR budget those
/// mappings are charged against, and GPU virtual mappings.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// The request is outside what the operation accepts.
    #[error("{operation}: {reason}")]
    Invalid {
        /// The operation.
        operation: &'static str,
        /// Why the request is refused.
        reason: &'static str,
    },
    /// Block-linear memory asked for a kind other than the generic uncompressed one.
    #[error("block-linear memory kind {kind:#x} is not the uncompressed generic kind {generic:#x}")]
    BlockLinearKind {
        /// The requested kind.
        kind: u32,
        /// The device's generic kind.
        generic: u32,
    },
    /// RM granted a pool block with other attributes or size than requested (the block is
    /// freed again; `free` is that free's failure, if any).
    #[error(
        "RM granted pool block attr {attr:#010x} size {size:#x} (requested attr {requested_attr:#010x}: VIDMEM, \
         CONTIGUOUS, PAGE_SIZE HUGE, COMPR NONE, size {requested_size:#x}){}",
        freeing(free.as_deref())
    )]
    PoolBlockGrant {
        /// Granted `attr`.
        attr: u32,
        /// Granted size.
        size: u64,
        /// Requested `attr`.
        requested_attr: u32,
        /// Requested size.
        requested_size: u64,
        /// The failed free of the granted block.
        free: Option<Box<Error>>,
    },
    /// RM granted block-linear memory with another kind or layout than requested (freed again
    /// as for `PoolBlockGrant`).
    #[error(
        "RM granted block-linear memory {size:#x} kind {format:#x} attr {attr:#010x} (requested kind {kind:#x}, \
         FORMAT BLOCK_LINEAR, COMPR NONE){}",
        freeing(free.as_deref())
    )]
    BlockLinearGrant {
        /// The allocation size.
        size: u64,
        /// Granted kind (`format`).
        format: u32,
        /// Granted `attr`.
        attr: u32,
        /// Requested kind.
        kind: u32,
        /// The failed free of the granted memory.
        free: Option<Box<Error>>,
    },
    /// A CPU mapping would exceed the BAR aperture measured at device open.
    #[error(
        "BAR mapping budget exhausted: {used}+{requested} exceeds measured aperture {limit}; reduce live CPU maps or enable a larger BAR"
    )]
    BarPressure {
        /// Bytes mapped now.
        used: u64,
        /// Bytes this mapping asks for.
        requested: u64,
        /// The measured aperture.
        limit: u64,
    },
    /// The BAR budget's accounting does not add up.
    #[error("BAR mapping budget: {reason}")]
    BarAccounting {
        /// What does not add up.
        reason: &'static str,
    },
    /// An earlier CPU unmap failed halfway; the device's mappings are no longer trusted.
    #[error("CPU mapping cleanup previously failed")]
    CleanupFailed,
    /// Unmapping an idle CPU mapping to make room for a new one failed.
    #[error("CPU map pressure eviction failed: {0}")]
    EvictionFailed(#[source] Box<Error>),
    /// A CPU mapping failed and undoing its partial setup failed too.
    #[error("{error}; CPU map rollback failed: {cleanup}")]
    RollbackFailed {
        /// The mapping failure.
        error: Box<Error>,
        /// The rollback failure.
        cleanup: Box<Error>,
    },
    /// RM or `mmap` returned a CPU mapping that cannot be used.
    #[error("{reason}")]
    CpuMapping {
        /// What is wrong with it.
        reason: &'static str,
    },
    /// RM's CPU mapping of a USERD or register window is larger than the address space.
    #[error("USERD size {length:#x} exceeds CPU address range")]
    UserdLength {
        /// The length RM returned.
        length: u64,
    },
    /// A CPU mapping, or a reference to one, is used after it was released.
    #[error("{reason}")]
    NotLive {
        /// Which object.
        reason: &'static str,
    },
    /// An access is unaligned or outside the CPU mapping.
    #[error(
        "unaligned/out-of-bounds {access} at offset {offset:#x} ({len} bytes) of a {size:#x}-byte mapping"
    )]
    OutOfBounds {
        /// The access.
        access: &'static str,
        /// Its offset.
        offset: usize,
        /// Its length.
        len: usize,
        /// The mapping's size.
        size: usize,
    },
    /// A CPU round trip through the mapping read back a different value.
    #[error("VRAM CPU round trip failed at word {word}: {actual:#x} != {expected:#x}")]
    PatternMismatch {
        /// The word index.
        word: usize,
        /// The value read.
        actual: u32,
        /// The value written.
        expected: u32,
    },
    /// A GPU mapping request is outside the memory, the 40-bit VA space or the 8-bit kind.
    #[error(
        "GPU mapping exceeds memory/VA/kind limits: offset {offset:#x} length {length:#x} of {allocation_size:#x} bytes"
    )]
    GpuMappingLimits {
        /// Offset into the memory.
        offset: u64,
        /// Mapped length.
        length: ByteSize,
        /// The memory's size.
        allocation_size: ByteSize,
    },
    /// RM returned a GPU VA that is unaligned, beyond 2^40 or not the requested one.
    #[error("RM returned invalid or displaced GPU VA {address:#x}")]
    GpuVa {
        /// The returned address.
        address: GpuVa,
    },
    /// Before Turing RM chose a PTE kind for a typed VA other than the requested one.
    #[error(
        "RM chose PTE kind {chosen:#x} for a typed VA that requested kind {requested:#x} \
         (driver without kindOverride, pre-Turing); refusing the mapping"
    )]
    KindMismatch {
        /// The kind RM chose.
        chosen: u32,
        /// The requested kind.
        requested: u32,
    },
}

impl MemoryError {
    /// Whether this failure makes the device unusable (`Error::is_device_lost`).
    pub fn is_device_lost(&self) -> bool {
        match self {
            Self::CleanupFailed | Self::EvictionFailed(_) | Self::RollbackFailed { .. } => true,
            Self::Invalid { .. }
            | Self::BlockLinearKind { .. }
            | Self::PoolBlockGrant { .. }
            | Self::BlockLinearGrant { .. }
            | Self::BarPressure { .. }
            | Self::BarAccounting { .. }
            | Self::CpuMapping { .. }
            | Self::UserdLength { .. }
            | Self::NotLive { .. }
            | Self::OutOfBounds { .. }
            | Self::PatternMismatch { .. }
            | Self::GpuMappingLimits { .. }
            | Self::GpuVa { .. }
            | Self::KindMismatch { .. } => false,
        }
    }
}

/// The suffix of a grant error whose cleanup free failed too.
fn freeing(free: Option<&Error>) -> String {
    free.map_or_else(String::new, |error| {
        format!("; freeing it failed too: {error}")
    })
}

/// A wait that RUST.md §7 bounds, and why it ended without the awaited state.
#[derive(Debug, thiserror::Error)]
pub enum WaitError {
    /// The awaited GPU work or event did not complete within `limit`.
    #[error("{operation}: {object} did not complete within {limit:?} (waited {elapsed:?})")]
    Timeout {
        /// The wait (the operation that issued the work).
        operation: &'static str,
        /// The awaited object and the state it was left in (channel, value, `GP_GET`).
        object: String,
        /// The bound.
        limit: Duration,
        /// Time from the start of the wait to the expiry check.
        elapsed: Duration,
    },
}

/// `Result` with the backend's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
