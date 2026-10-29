# Native synchronization ownership (caller-side, candidate F2)

The backend implements Mesa's `vk_sync` contract without an ordinary-path
worker. Submission, status, reset, host signal, CPU waits, export/import,
retirement and loss checks run on the calling thread, as stock NVIDIA 615 does
for every per-frame synchronization and submission operation (behavioural
guidance only; this code is written from public RM/Linux/Mesa interfaces).
The single optional thread is the lazily started foreign-fence watcher.

## Storage: the mapped word is the only truth

* `src/slots.rs`: one per-device pool of 16-byte semaphore slots in RM system
  memory (CPU-cached, GPU-uncached, `rm.rs` allocation attributes), allocated
  in 64 KiB pages that are CPU- and GPU-mapped once for the device lifetime.
  Ordinary payloads are one slot: binary 0/1, or a timeline value. There is no
  CPU shadow counter, publication job or point tree for local payloads.
* Lifetime: a slot carries a reference count. The CPU owner (payload, gate
  registration, queue retirement word) holds one; every accepted submission
  that encodes an acquire/release on the slot holds one until its private
  retirement word passes (`gpu.rs` `HeldSlots`, released on the submitting
  caller like stock's end-of-submit temporary retirement). A slot is recycled
  only at zero and is zeroed before reuse. Queue teardown releases held slots
  only after its channel object is freed (field order).
* Timelines: Volta+ hosts use one 64-bit word (64-bit SEM_EXECUTE, strict GEQ
  acquire; CPU reads with a real 64-bit atomic, cmpxchg8b/SSE on i686).
  Pre-Volta hosts (32-bit SEMAPHORE[A-D] only) use a split view: GPU low word at
  +0, 64-bit shadow at +8 raised (CAS-max) before any GPU release can execute,
  reconstructed low-then-shadow within a 2^31 window; out-of-window steps and
  waits are named errors (`slots.rs` tests cover wrap and rejection).

## Operations

| Operation | Behaviour |
|---|---|
| Binary GPU signal | release 1 to the current slot (WFI, after the work's visibility membar); submission recorded as the payload's pending signal |
| Binary GPU wait | acquire == 1 on the current slot; the slot reference moves to the batch and the payload is rebound to a fresh zeroed slot (unsignaled, no pending signal). No GPU reset write is needed: the consumed slot is never reused before the batch retires |
| CPU reset | sole owner: write 0 in place (stock same-slot reset); otherwise rebind to a fresh slot and leave the captured slot untouched |
| CPU binary signal | same rule, writes 1 |
| Status / WAIT_PENDING | load the mapped word; pending = signalled or a submitted signal (binary) / max submitted value (timeline) |
| Timeline wait | acquire >= value on the payload slot; a host signal is a CAS-max store the GPU re-polls. Mesa only submits it once the point is pending (submitted signal or host value): wait-before-signal is held in the queue's submit thread, never in a GPU ring |
| Host timeline signal | CAS-max of the mapped word, then a Dekker-checked local wake; no RM ioctl for local waiters |
| Foreign SYNC_FD import | owned dup + terminal-status cache. At submit a terminal success needs no gate, an error rejects the submission (never success), a pending fence gets a gate slot acquired by the GPU and released only by the watcher on success |
| SYNC_FD export | dup of a foreign payload; -1 for a signalled slot; the queue SEMSURF fence for a pending submitted signal; for a signal reserved by a submit-thread submission, the export queue's SEMSURF fence after a GPU acquire of the reserved slot (kernel 5 s cap -> -110 never affects ordinary completion); an internal export queue gated by the watcher for imported shared payloads. An unsignaled native payload with no submitted signal is a named error (VUID-VkSemaphoreGetFdInfoKHR-handleType-03254 / VkFenceGetFdInfoKHR-handleType-01454): a GPU wait on it would be orphaned by Mesa's post-export reset (`vk_semaphore.c:700`, `vk_fence.c:468`) and the fence could never signal |
| OPAQUE_FD | unchanged private sealed-memfd format (`sync_memory.rs`). Converting a local payload transfers its slot as a publication source; local GPU points are published by caller paths and the watcher, followed by one HOST broadcast |
| Move (Mesa submit-thread submissions only) | dst takes src's payload and pending state, src gets a fresh slot; shared sources keep E's generation semantics |

Mesa context (pinned 0b97dcdd): upstream selects IMMEDIATE submit for native
timelines (`vk_device.c:195-198`); `rm-integration.patch` (`nvk_device.c`)
selects THREADED_ON_DEMAND instead (see "Wait-before-signal and ring
pressure" below). Threaded-capable mode enables the post-submit binary reset
on caller-side submits (`vk_queue.c` IMMEDIATE branch) and the move into a
temporary for submit-thread submits (THREADED branch). Neither touches a slot
captured by in-flight work: a GPU binary wait already rebound the payload to a
fresh slot it owns alone, so the reset is an in-place 0 store; the move
transfers the payload (`take_for_move`). Copy-export resets
(`vk_fence.c:420-469`, `vk_semaphore.c:655-705`) follow the reset rule above,
and SYNC_FD export first waits for a pending signal (WAIT_PENDING,
`vk_semaphore.c:676`, `vk_fence.c:444`), which binary signal reservation
(below) makes immediate for submit-thread submissions.

## Wait-before-signal and ring pressure

A GPU ring is finite (`gpu.rs` QUEUE_ENTRIES). A submit that finds it full
CPU-waits for the queue's oldest batch on the submitting thread
(the nvkmd backend's `submit`; stock's GPFIFO slot advance waits the same way,
stock GPFIFO submit at ELF b7f0f0). Formerly native timeline waits
went straight to the GPU, so one thread could deadlock itself: submit to QA
waiting on S >= 1 (not signalled, no signal submitted), keep submitting to QA
until the ring is full, and block in the ring-full wait for that first batch,
while S could only be signalled by the same thread later
(`tools/native-vulkan-ring-wbs.c` reproduces it: hang with 1100 submits).

Invariant now: **the backend never receives a wait whose signal has not been
submitted or host-signalled.** `vk_queue_submit` (patched) runs a zero-timeout
VK_SYNC_WAIT_PENDING check on the submission's waits; this backend answers it
from the payload (`syncobj.rs` `ready`: timeline `pending_max` = largest
submitted GPU signal or host value; binary = signalled or a submitted signal;
a sync_file exists, so it is pending). A not-pending wait starts/uses the
queue's submit thread, which blocks in WAIT_PENDING (woken by submission and
host signal) before calling the driver; later submits to that queue queue
behind it in order. Proof the ring-full wait terminates: every batch in every
ring was submitted with all waits already satisfied by signal operations
submitted strictly earlier (or host values already written), so batches form
a DAG in submission order; the earliest unfinished batch has all its
dependencies finished and its ring predecessors finished, so the GPU makes
progress and the oldest batch of any ring retires. Exceptions are imported
sync_files and shared (OPAQUE_FD) payloads whose producer is outside this
device; they are bounded by their producers and keep the existing gate/loss
policy.

Binary signal reservation: a submission on the submit thread performs its
signals later, but Vulkan counts them as submitted when vkQueueSubmit returns.
Without care, a binary wait behind it (Mesa's THREADED branch waits
WAIT_PENDING with no timeout before moving the payload) or a fence/semaphore
SYNC_FD export (Mesa waits WAIT_PENDING first) blocked until the thread ran,
i.e. until the deferred submission's own timeline dependency was signalled,
possibly by the blocked thread itself (vast 2026-10-14: opaque probe consumer
vkQueueSubmit and sync-fd probe vkGetFenceFdKHR each blocked 6.25 s). Fence
export there is valid usage (VUID-VkFenceGetFdInfoKHR-handleType-01454 has no
"depends on" clause). The patched `vk_queue_reserve_binary_signals` binds each
binary signal of a deferred submission at vkQueueSubmit to the payload it
targets through the new optional `vk_sync_type::reserve_signal`
(`nv_rm_sync_reserve`, `syncobj.rs` `reserve_into`): the deferred batch
signals a temporary that holds the source's current slot (native) or a fixed
shared generation published as PENDING at once (OPAQUE shared); the source
reports a pending signal (`RESERVED`). Waits, moves and resets of the source
after that never redirect the deferred signal. SYNC_FD export of a reserved
native payload submits one GPU acquire of that slot on the internal export
queue (no CPU wait; ring pressure there is a named capacity error) and returns
its SEMSURF fence; the 5 s kernel cap then yields -110 like any long pending
export.

Fast path: a queue never pays the thread when every wait is pending. Upstream
THREADED_ON_DEMAND keeps a queue threaded forever once started; the patch
returns submissions to the caller whenever the thread's list is empty (checked
under the queue mutex; the thread deletes a submission only after
`driver_submit` returned, and only the externally synchronized caller adds
one), so ordering is preserved without routing ordinary work through the
thread. The added cost per submit with waits is one zero-timeout WAIT_PENDING
query (no syscall) and, for permanent binary waits, Mesa's post-submit reset.

## Submission

`nv_rm_submit_batch` plans every wait/signal under each payload's lock without
mutating it, encodes waits, work and signals into one GPFIFO segment (acquire
prefix entry, pushes, tail entry ending with the immutable NON_STALL_INTERRUPT
suffix) and rings one doorbell. Ring pressure returns VK_NOT_READY with the
oldest token and commits nothing (hardware test `ring_full_leaves_payloads_
untouched`). Payload state changes are applied only after acceptance.
Per-batch tail slots are 512 bytes (<= 10 waits / 7 signals inline); larger
batches use retained overflow segments released at retirement.

Doorbell ordering (Volta+): GPFIFO entries and GP_PUT live in VRAM written
through BAR1. Candidate F2 rang the doorbell through the BAR0 usermode page
with no flush and lost kicks: with idle gaps between submits the host acted on
the doorbell before the GP_PUT write was visible, a batch stayed unexecuted for
> 2.1 s and retired 22 us after the next doorbell
(`device::reclaim_tests::every_submitted_batch_runs_without_a_later_doorbell`,
`build/fence-tracing/deferred-reclaim/kick-test-*`). An idle queue's last batch
is then stranded forever — the SystemUI/Launcher hangs, where the drained batch
had no foreign dependency. Fix: on Hopper+ the usermode page is allocated with
NV_HOPPER_USERMODE_A_PARAMS {bBar1Mapping = 1} (as public UVM does,
`nv_gpu_ops.c:5646-5659`; stock 615 also rings through such a BAR1 page,
runtime-observed in `build/fence-tracing/deferred-reclaim/stock-userd/`), so
the doorbell follows GP_PUT on the same BAR1 posted-write path: fence, GP_PUT,
fence, doorbell, no read. Volta..Ada only have the BAR0 page: GP_PUT is read
back through BAR1 (UVM's flush, `uvm_turing_host.c:224-229`) and checked
before the doorbell. Pre-Volta keeps its USERD read-back.

## CPU waits: leader/follower on one device HOST descriptor

`src/waiters.rs`. Waiters register (Dekker counter), re-check, then either
claim leadership and poll the device HOST + RC OS-event descriptors plus a
private eventfd, or sleep on a condition variable as followers. Public
`nvidia_poll` clears the dataless flag it reports, so only the leader polls;
whenever it leaves poll it bumps a generation and wakes all followers, and the
next waiter claims leadership. An interrupt that arrives while nobody polls stays
latched in the kernel flag. CPU-side changes (host signal, watcher-observed
foreign/shared completion, loss) use `notify()`: SeqCst fence + registration
load, so no wake is lost and no syscall happens when nobody waits. Submission
wakes only WAIT_PENDING waiters. Deadlines are the caller's only; the leader
returns at least once per second to run the loss scan. There is no sleep-poll
loop. Public RM routes the HOST non-stall interrupt subdevice-wide
(`event_notification.c:452-466`, `intr.c:1191`), so a blocked leader also wakes
on other clients' interrupts and simply re-checks (stock behaves the same).

## Loss

Sticky atomic flag. Per call: flag load + 1 s throttle compare; the heavy scan
(every channel's VRAM error notifier) runs at most once per second, or forced by
RC readiness seen by a polling leader, or on failed submission. Mesa's loss
callback (`nv_rm_mark_lost`) only sets the flag and wakes waiters/watcher.
Every C entry point converts a Rust panic into published loss and
VK_ERROR_DEVICE_LOST instead of unwinding into C.

## Foreign watcher (`nvk-rm-foreign`)

`src/watcher.rs`. Started on the first pending foreign dependency or shared
publication; otherwise never created. Blocks in poll without timeout on its
eventfd, retained foreign sync_files and (only while shared work exists) its own
independent HOST listener. Divergence from stock's
DRM_IOCTL_NVIDIA_SEMSURF_FENCE_WAIT, whose public implementation hides setup
failures, advances on producer errors and cannot cancel foreign callbacks at
context destruction (`nvidia-drm-fence.c:1639-1750, 774-784`): the watcher
decodes the authoritative sync_file status, releases a gate only on success,
publishes loss on error, and is cancelled and joined at teardown.

## Resource reclamation: frees never wait for the GPU

`rm-core/src/reclaim.rs`, `rm/src/device.rs` `reclaim_*`. GPU work can legitimately be gated on
a foreign sync_file whose producer is the thread that is freeing memory
(Android: the release fence of a buffer needs the app's next frame). The former
`nv_rm_free`/`nv_rm_unbind` drained every queue's last token first and so
deadlocked SystemUI's RenderThread in `vkGetMemoryFdPropertiesKHR` until kernel
timeouts (`build/fence-tracing/caller-release/FINDINGS.md`). Now:

* `nv_rm_free` evicts the CPU VMA (not GPU-visible) and drops the id at once;
  `nv_rm_va_free` takes the VA's mapping ids. Both record, under Inner (the
  lock every submit holds), each queue's last token that has not retired. With
  no such token the RM unmap/free and the C VA release run immediately;
  otherwise they form one FIFO group reclaimed after every recorded token
  retired: RM unmaps first, then the RM memory objects (a mapping also holds its
  memory alive), then the C callback returns the range to Mesa's heap. The
  range is therefore never reusable while still mapped or reachable.
* Groups are reclaimed amortised on the caller: at every submit, alloc, import,
  map and bind (under Inner), and after status/poll/wait calls when a lock-free
  pending hint is set and Inner is free (`try_lock`). Storage is three retained
  deques: no allocation per free in steady state.
* A lost queue never counts as retired: its groups wait for close, which
  deschedules and frees every channel first and then reclaims everything and
  runs all outstanding C releases before `nv_rm_close` returns. A failed
  deferred unmap publishes loss and keeps the range reserved (never released).
* `nv_rm_unbind` is immediate with no wait. It only serves application-ordered
  binds: non-sparse NVK binds target fresh VA (`nvk_buffer.c:255`,
  `nvk_image.c:1493`, `nvk_mem_arena.c:91` at a new offset), Vulkan forbids
  binding a resource twice (VUID-VkBindBufferMemoryInfo-buffer-07459,
  VUID-VkBindImageMemoryInfo-image-07460) and freeing memory still used by
  submitted commands (VUID-vkFreeMemory-memory-00677). Sparse queue binds, the
  only path that rebinds a live range, are not advertised (`has_sparse` unset:
  `nvk_physical_device.c:1388,1507`), and the bind context CPU-waits its
  semaphores before any VA change (the backend's context `wait`). Driver-internal
  frees of possibly in-flight memory (the old SLM area: `nvk_device.c:118`,
  `nvk_queue.c:65`) go through the backend's `va_free` -> `nv_rm_va_free`, i.e. deferred.
* Remaining synchronous waits are all explicit or bounded: caller waits
  (`nv_rm_wait`, `nv_rm_sync_wait_many`, Mesa's own `nvkmd_ctx_sync` used by
  `NVK_DEBUG=push_sync` and upload-queue syncs), the ring-pressure retry on the
  caller's own oldest batch (the backend's `submit`, like any GPFIFO producer; it
  terminates because no received wait depends on unsubmitted work, see
  "Wait-before-signal and ring pressure"),
  `nv_rm_context_destroy` (own queue, 5 s bound, then loss) and close (5 s).

Submission commit re-checks that each consumed/signalled payload still names
the slot the plan encoded; a mismatch (only possible when a host operation on
the same semaphore/fence races its submit call, an external-synchronization
violation) is logged, because the GPU then waits on or releases a slot the
payload no longer names.

Growth of the deferred list is bounded by GPU progress: a group is reclaimed
at the first backend call after its snapshot retired, so it holds only memory
freed while that work was outstanding (kernel drivers keep freed BOs alive
behind unsignalled fences the same way). Stock 615 frees application memory
immediately (Vulkan forbids freeing in-use memory) and defers only internal
objects; the adapter cannot tell the two apart at `nvkmd_mem` free, so it
defers both — a superset that never waits.

`vkGetMemoryFdPropertiesKHR` (the AHB-properties call on every HWUI dequeue)
imports the dma-buf through `nvkmd_dev_import_dma_buf`
(`nvk_device_memory.c:74`) only to read `mem->flags` (`:103`) and then unrefs
it (`:111`); NVK's nvkmd import contract always returns a VA-bound memory
(the backend's `dev_import_dma_buf`, and NVK reads `mem->va->addr` directly, e.g.
`nvk_buffer.c:261`), so skipping the temporary bind needs an NVK change. With
deferred reclamation that bind/unbind never waits; a flags-only import op is a
possible later optimisation.

Semaphore encodings (`gpu.rs` `host_semaphore`): acquires use
ACQUIRE_SWITCH_TSG (bit 12) like stock, so a parked channel yields its
timeslice and the host re-polls it without a doorbell (fixture
`cpu_released_acquire_runs_without_a_doorbell_after_idle`). Binary acquires
use circular GEQ 1 on a 0/1 word, equal to ACQUIRE == 1 for the only values a
slot can hold; consumption detaches the slot instead of a GPU release of 0.
Every release carries RELEASE_WFI (bit 20), so no release can publish before
the batch's prior work completed.

## Diagnostics

The first loss publication logs its source location (`Core::mark_lost` is
`#[track_caller]`); a panic hook installed at device open logs every panic's
location and message before the C-boundary guard turns it into device loss.

`src/diag.rs`: every backend diagnostic goes to logcat on Android
(`__android_log_write`, priority ERROR, tag `nvk-rm`; liblog is already in
NVK's own dependency closure) and to stderr on GNU builds.

## Threads, ownership and teardown

All Rc/RefCell RM owners live in one `Inner` behind a mutex (`rm/src/device.rs`
`Confined` documents the Send argument: nothing Rc escapes; the mutex orders
every owner). Lock order: Inner -> payload state -> {shared local, slot free
list, watcher state, waiter condvar}, and Inner -> the C VA-heap lock (deferred
release callback; C never holds that lock across a backend call); nothing
blocks on GPU progress or a foreign fence while holding Inner. Close drains for
at most 5 s, publishes loss and deschedules on failure, stops and joins the
watcher, closes the pool, frees every channel, reclaims every deferred group
(running the C VA releases), then drops mappings, pool pages, listeners,
address space and device. No code
runs after `nv_rm_close` returns. A stuck kernel ioctl remains outside any
userspace wall-clock bound.

## Tests

CPU (GNU32/GNU64): slot lifetime and concurrency, split-view wrap/rejection,
reset/detach/move/import/foreign replacement, shared publication and failure,
Dekker stress with forced parking, device-event leader/follower wake, loss
wake, panic guard. Hardware (ignored, `device::hardware_tests`): slot recycling
over 3000 frames, ring-full transactionality, exporter close publication,
bounded owned-gate teardown, two-waiter wake latency (GPU fence and host
timeline, both waiters and follower lag < 1 ms p99). `device::reclaim_tests`:
free/unbind while foreign-gated returns < 50 ms and reclaims after retirement;
GPU writes/reads freed memory after the free call; concurrent
alloc/bind/free + host/foreign-gated submits + waiters (NV_RM_STRESS_SECONDS);
watcher gate release with four contended waiters per round. Runtime fixtures:
`tools/native-vulkan-{async,fence-design}.c` on host/Android64/Android32.
