# Experimental native compute interface

This folder translates a deliberately limited Vulkan interface into the real
Rust RM backend. It does not implement a conformant Vulkan driver or Android
presentation. No existing NVIDIA userspace DSO participates in submission.

Read in this order:

1. `mod.rs`: owned object metadata, dispatchable object headers, global registry,
   buffer-range arithmetic and focused non-GPU checks.
2. `instance.rs`: discover the real GPU and report the bounded prototype heap;
   expose one compute/transfer queue, no optional core features or extensions.
3. `memory.rs`: allocate host-coherent VRAM through the backend, bind buffers at
   explicit allocation offsets and map the real memory into the caller.
4. `pipeline.rs`: retain shader words, compile SPIR-V through Mesa NIR/NAK at
   pipeline creation, copy layout metadata and maintain storage-buffer bindings.
5. `commands.rs`: record commands; snapshot push constants at each dispatch;
   resolve descriptor contents when submitting; execute dispatch/copy synchronously.
6. `sync.rs`: signal fences only after execution succeeds, release the registry
   while waiting, and track binary semaphores on the single synchronous queue.
7. `dispatch.rs`: expose only implemented function pointers and the Khronos ICD
   loader interface. Unknown functions return null.

The registry mutex serializes frontend operations. The backend owns a separate
worker thread because its RM and GPU objects intentionally remain thread-bound.
Mapped GPU allocation addresses cross the boundary as integers, never borrowed
Rust references. Vulkan's normal pointer-lifetime and external-synchronization
requirements apply to every exported unsafe C function.

Pipeline layouts own copies of descriptor-layout metadata; descriptor sets own
copies too. A pipeline owns its pipeline-layout metadata. Destroying the original
layout handles therefore cannot invalidate information retained by those objects.
Commands retain descriptor-set identity, not prematurely copied descriptor data;
recorded push constants are copied immediately, as Vulkan requires.

A command-recording error is sticky through `vkEndCommandBuffer` and submission.
Unsupported extensions, features, public pNext structures, allocator callbacks,
graphics binding, image barriers, secondary command buffers, descriptor arrays,
dynamic descriptors, multiple descriptor sets and shader specialization fail
explicitly. Valid loader-private create-info structures are permitted.

GPU dispatch and copy return only after their hardware completion semaphore.
Consequently the implemented buffer memory barriers need no deferred action.
Fill and update commands use ordered CPU stores into the host-coherent mapping;
this is an explicit prototype implementation, not a claim of GPU acceleration.
There are no image, rendering, WSI, external-memory or sync-file entrypoints.

The source shader compiler accepts the documented storage-buffer and push-constant
layout supplied by the compiler bridge. This frontend never recognizes a shader
fixture or substitutes a canned binary. Shader compilation errors are returned
as errors with the compiler diagnostic on stderr.

Measured host path: the independent public probe passed both direct GIPA and the
Khronos loader (`libvulkan.so.1.4.363`). Two separately compiled SPIR-V shaders,
six invocation counts (1, 63, 64, 65, 257, 1024), 4,422 generated/transformed/copied
words, full allocation guards, descriptor rebinding, nonzero binding/descriptor
offsets, fence reset, and descriptor count 4→4 passed. This is a compute gate,
not Vulkan conformance or a working Waydroid replacement.
