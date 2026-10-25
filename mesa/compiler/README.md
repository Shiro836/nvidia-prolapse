# Native SPIR-V compiler

`nv_nak.c` converts a supplied SPIR-V compute entrypoint into NIR, lowers the
native driver's buffer/push-constant ABI, and invokes Mesa's Rust NAK compiler
for SM120. It compiles the actual supplied program in-process. There is no
fixed-shader matching, PTX compilation, CUDA/NVRTC, or proprietary userspace
dependency. The adapter is C because Mesa exposes NIR passes through its C ABI;
NAK itself and the native GPU backend are Rust.

## Build

From the repository root:

```sh
python mesa/compiler/build.py --target both
```

On a machine missing local compiler tools or Mesa wrap dependencies, use
`--download-deps` once. This builds bindgen-cli **0.72.1** and cbindgen **0.29.2**
under `build/nak-tools`, and permits Meson to download dependencies using the
hashes in the pinned Mesa wraps. Normal rebuilds use `--wrap-mode=nodownload`.

Prerequisites: existing complete `build/mesa-git` git objects, Meson (tested 1.12.1),
Ninja, C/C++ compiler, Rust (tested 1.95.0) with `x86_64-linux-android` target,
Python Mako, the repository's NDK r29 and source-built libdrm headers/build.
Install the Rust target with `rustup target add x86_64-linux-android` if absent.
The existing Mesa checkout was expanded with `git sparse-checkout disable`.
The script creates and verifies an isolated source worktree and never alters
the original Mesa checkout or the installed Waydroid stack.

Source is pinned to Mesa **ddaf85e353442d9d7f192fdcf1823e652e3f2ee3** (mesa-26.2.5). The checked-in
`mesa-compiler-only.patch` makes the isolated worktree build NAK without the NVK
driver and its OpenCL-generated helpers. It also selects C11 thread definitions
compatible with the current glibc headers, with size/alignment assertions for
the host mutex adapter. The compiler uses stderr/environment diagnostics under
bionic: Android logcat/property/atrace integration is intentionally excluded
from this library, with no fake liblog/libcutils implementation.

Artifacts:

```
build/nak-host/src/nouveau/native-bridge/libnv_native_nak.so
build/nak-android/src/nouveau/native-bridge/libnv_native_nak.so
build/nak-{host,android}/src/nouveau/native-bridge/nv-nak-check
```

`NV_NATIVE_COMPILER_PATH` points the Rust driver at the desired library.
The Android shared library has only `libm.so`, `libdl.so`, and `libc.so` as
DT_NEEDED entries. NDK libc++ and the Rust standard runtime are statically linked from the installed
toolchains; this work did not rebuild those toolchains. No host glibc library is linked into the Android artifact.
The host build can also link the distribution's open-source SPIRV-Tools when
Mesa detects it; the bionic build does not need SPIRV-Tools.

## ABI

The complete, 48-byte x86_64 C/Rust result structure and ownership rules are in
`nv_nak.h`. Compile calls must be serialized by the caller. Successful result
code belongs to the bridge until `nv_nak_free`; Rust copies it before freeing.
Errors produce a NUL-terminated diagnostic and no result allocation.

GPU constant buffer 0 is 1152 bytes:

| Offset | Content |
|---|---|
| `binding * 16`, bindings 0..7 | `u64 GPU address, u32 byte length, u32 reserved` |
| 1024..1151 | Vulkan push constants |

Currently supported resource scope: descriptor set 0, storage buffers, one
descriptor per binding, push constants up to 128 bytes. Descriptor GPU addresses
already include `VkDescriptorBufferInfo.offset`. General integer/float program
instructions, control flow, workgroup identifiers, shared memory and barriers
go through Mesa/NAK; this is not a claim that every such combination has been
hardware tested. The runtime checks shader resource requirements separately.
Current dispatch has no base-workgroup offset, so base system values lower to
zero. `NumWorkgroups`, images/samplers, UBOs, descriptor arrays, specialization
constants, other stages and other shader models are unsupported. Buffer length
records are reserved for later bounds lowering; robustBufferAccess is not
implemented by this bridge. Valid Vulkan programs must keep accesses in bounds.

## Verification

`check.c` independently checks result structure size and offsets in both host
and Android builds, compiles supplied SPIR-V files, prints code/resource
metadata, exercises free/reset and rejects a malformed instruction stream.
For example, after generating the public probe's shaders:

```sh
build/nak-host/src/nouveau/native-bridge/nv-nak-check \
  build/native-vulkan-probe/generate.spv build/native-vulkan-probe/transform.spv
```

Run the Android check with the extracted guest runtime linker and absolute
executable path, setting `LD_LIBRARY_PATH` to the runtime's `lib64` plus the
Android bridge directory. Both supplied shaders compiled under actual bionic:
generate: 272 code bytes, transform: 400; 19 GPRs each, local/shared/barriers 0,
workgroup 64x1x1. Parent driver verification additionally executes these programs
on the GPU through Vulkan. Compiler-only checks do not establish rendering,
Android app sandbox compatibility, or Vulkan conformance.

## Source anchors and licensing

- Mesa `src/vulkan/runtime/vk_nir.c:157`: SPIR-V entrypoint preparation,
  initializers, return lowering and inlining.
- Mesa `src/nouveau/vulkan/nvk_shader.c:38,360,425`: shared memory layout,
  compute system values and explicit buffer-address lowering.
- Mesa `src/nouveau/vulkan/nvk_nir_lower_descriptors.c:714,807`: descriptor
  resolution and push constants loaded through `nir_ldc_nv`.
- Mesa `src/nouveau/compiler/nak/api.rs:413,484`: postprocessing, Rust code
  generation, and reported compiler failures.
- Mesa `src/nouveau/compiler/nak.h:132`: shader metadata and code ownership.

These paths are relative to the pinned `build/mesa-nak-source` worktree.
The bridge is original MIT-licensed adapter code informed by the cited Mesa
interfaces. Mesa files retain their original copyright and license notices;
Mesa's `docs/license.rst` describes the upstream licenses. The local patch
does not replace those notices.
