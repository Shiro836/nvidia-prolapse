# Compatibility implementation — source contracts and measured tuples

**2026-10-13 event-driven fence implementation; local correctness gates pass.**
Ordinary completion, exported fence errors and channel faults remain distinct.
One owner waits on public RM HOST/RC events, jobs and retained producer FDs;
no periodic maintenance tick or ordinary-work age watchdog remains. Work and
final signals share a submission. Ownership, cancellation, notification ordering
and selective watch sets are documented in `mesa/SYNCHRONIZATION.md`.
Current source passes local5090/open615 host/Android64/Android32 async plus16
core/HAL gates, long ordinary/opaque waits and bounded owned-gate teardown.
95 CPU tests and six links/37 exports pass. No deployment or
all-driver runtime promotion. Earlier rental records retain their source hashes.


2026-10-13: automatic performance policy removed after stock-policy investigation; prior hardware bundles retain their exact historical scope. The T1 Vast.ai sweep exposed portability defects; the repair pass now has targeted runtime evidence under `build/vast-results/20261012-fixes/`. Match every result to its recorded bundle hash, GPU, exact driver, flavour and CPU ABI. Source coverage and successful builds do not establish support on untested tuples. The active Waydroid installation is not changed by these tests.

## Exact ABI and capability decisions

`src/compat.rs` and generated `compat_data.rs` select an exact release from independently compiled public C declarations, preserving branch backports. The union contains335 recognized versions;225 have complete client ABI evidence.110 older versions and every unknown version fail before opening a device. Of the225,137 have the fixed semaphore-fence interface used here; the other88 require a different synchronization implementation. Regenerate with `tools/generate-compat.py`; inputs/provenance stay private.

`src/capabilities.rs` intersects explicit known classes with the RM class list and checks the queried ISA against the supported3D/compute pair. RM ISA is not converted arithmetically to SM: GB20x0xa04/0xc00 meansSM120, GA1000x802 meansSM80, and TU116/TU1170x703 meansSM75 while preserving the real chipset identity. Copy engines are selected through the queried graphics partner list and matching per-engine class list, then allocated with version1 and the discovered engine ordinal; an absent or ambiguous partner fails explicitly. Maxwell/Pascal use pre-Volta submission; Volta and newer use a selected USERMODE class. Unknown families fail by name. Host module flavour, exact PCI/subsystem membership, platform and execution remain independent requirements; consult `build/gpu-coverage/matrix.md` from the repository root.

## Source changes and invariants

| Area | Enforced invariant | Evidence / checks |
|---|---|---|
| Version / H18 | Exact version and public profile, strict kernel version handshake; no nearest-version or relaxed-version assumption | Every live public request is packed against generated C metadata; old status and nonmonotonic channel layout tests |
| Wire / H8,H10,H15–H18 | Zero-initialized target fields, no reads of implicit Rust padding; field relocation and output decoding; absent nonzero fields rejected | Full canonical Rust member offsets/widths compared with public C fields; exact fixed DRM/private layouts required |
| Classes / H1–H4 | No arbitrary class suffix or unsupported frontend; finite method-family intersection and paired ISA/engine IDs | Public headers/pinned Mesa plus negative/known-device unit cases |
| Memory / H11–H12 | CPU mappings own both VMA and RM leases; munmap precedes NVOS34 with the original 64-bit RM cookie/flags; post-map failures roll back; memory and BAR credit survive failed cleanup | Fault/retry tests, exact-release C layout probes, 40 local sequential mappings, remote 256MiB BAR stress on all three tested ABIs |
| Kinds / H13–H14 | Explicit kind0 is PITCH, separate from inherited physical kind; older drivers get typed virtual reservations, newer get kindOverride | Independent public-header mapping flags; separate objects preserve alias reservations; checked VA bounds |
| Unmap | Failed RM unmap retains the mapping and VA reservation and loses the device; no successful reuse report | Explicit fallible unmap before removal; C binding retained on failure; source review |
| Submission / H5–H9 | Pre-Volta SYS GPFIFO + borrowed RM USERD/GPPut; TSG owns VA, legacy contexts inherited. Modern client USERD + queried doorbell/token | Public RM TSG/USERD source and independently generated host packet oracle |
| Host methods / H6–H7 | Endianness, payload width, circular32 acquire, release WFI, Maxwell C/D versus old A/B membar, command slot bounds | Public C-generated packet vectors and invalid-input tests |
| Sem32 / H10 | Cached system-memory atomic maximum published first; low-before-maximum reconstruction; independent native64/monitored caps; half-range bound and drain across epoch | Public sem_surf.c, closed550/580 scouting, rollover/origin/layout tests |
| Retirement | Intermediate external-wait values never reclaim command storage; a final private release before an immutable IRQ suffix owns ring/tail retirement | Seeded epoch transition and4096 ring-reuse test; checked credits/sequence exhaustion |
| Old notification / H10 | Known no-BIND versions use constructor callback registration; BIND/UNBIND never called just because a header has the ID | Exact profile predicates, inspected closed550 stub, public550 constructor |
| External buffers / H17 | Separate DRM file namespaces, borrowed input fds, owned output fds, exact NVKMS formats; foreign exporter eligibility checked | Private/public C layout guards and field assertions; no CPU-copy fallback |
| CPU ABI | Correct ioctl request type and off64_t libc entrypoints; no raw variadic syscall assumptions; page rounding checked independently of GPU pages | Five Android target compiles; NDK C wire census; LP32 symbol/LLVM inspection |
| CPU ordering | Device-inclusive barriers, posted USERD readback, no atomics on MMIO; cached aligned AtomicU64 only for CPU tracker | Target instruction inspection; actual cache/device coherency remains a hardware gate |
| Probe / H19 | Experimental standalone compiler/compute path remains explicitly GB202-specific, separate from NVK | Default feature remains disabled for production bridge; no compatibility claim from that probe |
| Synchronization domains | Ordinary memory-backed events; lazy DRM sync-file export; sealed CPU-only opaque metadata; same-type FD compatibility; immutable captured generations | Native sync unit/FD lifetime tests, >6s Vulkan separation/opaque tests, real Mesa preflight with absent syncobj caps; new notification protocol requires tuple-specific retests |
| OS notifications | Blocking cached HOST/RC FD pair, one poll consumer, listeners before channel use, listener free before OS-event free/close, failed cleanup retains owners | Public exact-release event layouts; GNU32/64 lifecycle tests; selected RM pair smoke; five Android poll/eventfd ABI probes |
| Wait ownership | Caller-side leader/follower on the device RM event; mapped slot is the predicate; software loss wakes every waiter without RM calls; finite API timeout ends only its caller | Lost-wakeup/loss/latency tests and actual Mesa runtime hook harness |
| Watch selection | RC always; HOST for runnable queue heads or shared aliases; foreign gates retain producer FDs; retirement precedes shared publication and parked predicates cannot retire again | Exact queue/token helper, shared mappings, dispatch fairness and reused descriptor tests; >6s two-device opaque runtime |
| Submit batching | Commands/waits/final signals share a batch; waits first chunk, signals last; ring-full publishes nothing; accepted prefix is not replayed | Actual Mesa chunk/retry/failure harness;1030 marker retirement test on all three host CPU ABIs |
| Device loss | Observe channel errors, failed dependencies and transport loss; caller deadline returns timeout; own exported timeout never loses producer or retires GPU storage | Real Mesa loss/drain harness, autonomous maintenance test, imported-error GPU fixture; ordinary and exported timeout separation |
| Performance policy | Queue correctness does not require automatic maximum-performance votes; submission, retirement and deschedule issue none. Explicit diagnostics remain separate | Local rebuilt compute4422/graphics2109 pass with zero observed boost controls, not production authority |
| Presentation / H20–H21 | Pinned Mesa NIL supplies per-architecture layouts; unimplemented sparse/format/frontend paths reject explicitly | Local host/Android64/Android32 core and HAL gates pass; real-guest presentation and other GPU/driver tuples remain separate |

## Boundaries that source changes cannot turn into a passed test

- The public RC index for C76F is not established; that family refuses by name rather than borrowing another family's event index. Closed-core and SR-IOV VF error-delivery coverage is not certified by the tested bare-metal GSP path.
- Pre-SEMSURF legacy fence APIs can report early success on timeout, wrap or teardown. CP9 is not enabled; a safe kernel/API contract is still missing. Pre515 RM inner layouts and390 external-memory transport remain unresolved CP13 work.
- Hopper/datacenter Blackwell and pre-Maxwell frontend families remain explicitly gated by the current pinned NVK/compiler contract. Integrated/coherent GPU discovery and allocation are not supplied by pretending an iGPU is a PCIe discrete device. POWER has no established native Android target here. GNU32 ARM is rejected until its exact architecture/ABI is selected; Android ARMv7 is compiled.
- Imported foreign sync-files retain owned descriptors and feed userspace-serviced GPU gates; no production NVIDIA DRM WAIT callback is installed. A five-second teardown wait deschedules channels before freeing backing. An unresponsive kernel ioctl can still exceed that wall-clock deadline; external fence errors are not completion or proof of rendered data.
- CPU pages4/16/64KiB are accounted for, but kernel compat behavior, non-x86 device cacheability, untested old-driver aliases and pre-Turing devices still need hardware gates. Sampled small-BAR, Maxwell/Pascal copy, TU11x graphics and bionic x86 paths now have tuple-specific evidence; this does not validate non-x86 hardware or complete Android images. FullCTS/soak are separate.

Offline verification commands: `cargo test --offline --no-default-features --lib` (hardware fixtures remain ignored), `cargo clippy --offline --no-default-features --lib -- -D warnings`, public C ABI/packet oracles and `build/gpu-coverage/cpu-backend-check.py`. Linked Mesa builds use `mesa/build.py`. Local and rented-host tests are authorized for the repair pass; deployment and full Waydroid/VM testing are separate work. Final event implementation counts and hashes are in `build/fence-tracing/event-cpu-links-gated/` and `build/fence-tracing/event-final-runtime/`; historical compatibility reports remain in `build/gpu-coverage/implementation-status.json`.
