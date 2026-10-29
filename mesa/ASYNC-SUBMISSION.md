> **Superseded threading (2026-10-13):** the RM worker described below was removed.
> Submission, waits and retirement now run on caller threads; the five-second
> batch watchdog and kernel WAIT paths are gone. Current contract:
> [SYNCHRONIZATION.md](SYNCHRONIZATION.md). Kernel limits quoted below still apply.

# Async transport contract and measured kernel limits

**2026-10-13 timeout correction (newer than the boost investigation):** the
five-second batch watchdog is an unimplemented correctness repair, not a valid
NVIDIA invariant. Display-active5090 reports CUDA timeout=1/RM watchdog flags6
but completes6.25s and9.5s kernels. Disassembly confirms the attribute tests RUNNING,
not a duration. Ordinary Mesa signals currently use capped DRM fences too;
WAIT can hide installation failure and kernel foreign-wait cleanup cannot be
universally bounded.


The nvkmd backend queues borrowed sync-file dependencies and push buffers with
`nv_rm_submit_waits`. Success returns a native 64-bit completion token before GPU
completion. `VK_NOT_READY` consumes nothing and returns the oldest outstanding
token for outside-worker backpressure. The worker never sleeps for GPU progress.

Each batch owns an immutable acquire prefix and release tail until its private
retirement word advances. A native RM semaphore surface supplies real pending
DRM sync files; its release is followed by an interrupt. A separate final release
after the interrupt proves all tail words were consumed before slot reuse. One
GPFIFO entry stays unused to distinguish a full ring from an empty ring.
Dependency milestones cannot retire command storage. (Superseded: free and VA
teardown no longer drain; they defer the RM free/unmap and the VA return until
the tokens outstanding at free time retire — SYNCHRONIZATION.md, "Resource
reclamation". Context destruction waits its own queue, bounded by 5 s.)

`nv_rm_check_status` checks channel notifiers and retained input/output sync-file
errors, including a signaled fence carrying a producer error. Such errors become
`VK_ERROR_DEVICE_LOST`, and all channels are disabled before object teardown.
Normal close drains and joins the worker: there is no detached code that could
outlive `dlclose`. CPU mappings close their temporary fd after successful mmap;
Linux's VMA owns the file reference and Rust retains the RM allocation.

The matching 615.71.09 open kernel module imposes limits that userspace cannot
remove. In `kernel-open/nvidia-drm/nvidia-drm-fence.c`, line43 defines a 5000ms
semaphore-fence maximum; lines1447–1449 clamp zero and larger requested timeouts
to that maximum. Direct syncobj attachment also uses that limit. The transport
therefore checks a five-second batch watchdog even for batches with no exported
Vulkan signals. This does not implement universally unbounded Vulkan dependency
waits. `UINT64_MAX` at the local wait API cannot override the kernel boundary.

The same kernel's imported-fence wait callback advances its post value even for
an errored producer fence, and its wait ioctl returns success on some internal
failure paths (lines1639–1750). Retained sync-file status detects producer errors;
the batch watchdog detects an accepted wait that never advances. Mesa's status
hook must run after synchronization so errors are not mistaken for successful
GPU work merely because a DRM syncobj became signaled.

Finally, fence-context teardown waits for every imported foreign fence callback
(lines774–784); there is no userspace cancellation operation. An arbitrary
foreign fence that never signals can therefore block kernel context cleanup
after device loss. The driver disables channels and preserves resources rather
than freeing GPU-visible memory early or leaving an executing detached worker.
Removing this limit requires a kernel API change with correct cancellation and
reference ownership. The deterministic failure fixture always releases its gate.
These constraints are explicit; passing the async regression is not a claim of
complete Vulkan conformance or universal foreign-fence recovery.
