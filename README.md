# nvidia-prolapse

An open driver stack for NVIDIA GPUs in Android containers on Linux (Waydroid and similar).
Android apps get Vulkan from Mesa's NVK and NAK, running on a Rust backend that talks to the
host's NVIDIA kernel module directly. GLES comes from ANGLE on top of that Vulkan driver.
No proprietary NVIDIA userspace library runs inside the container.

This repository is being assembled from the [waydroid-nvidia](https://github.com/Shiro836/waydroid-nvidia)
project, where the stack runs today, and rewritten in Rust as it moves. Our own C and C++ get
ported to Rust. Third-party code (Mesa, ANGLE) and the C++ interfaces Android forces on us stay as
they are.

## Components

| Directory | What it is |
|---|---|
| `rm/` | Rust backend that talks to NVIDIA's kernel module (RM ioctls), plus the NVDEC video core |
| `mesa/` | Mesa NVK and NAK at a pinned release, our patch stack, and the C adapter (`nvkmd_rm.c`) between NVK and `rm/` |
| `hal/` | Android Vulkan HAL (`vulkan.nvidia.so`) over NVK: native and hardware buffers, ASTC/ETC2 emulation |
| `gralloc/` | gralloc backend for minigbm (`libgbm_mesa_wrapper.so`) and the minigbm patches it needs |
| `media/` | Codec2 hardware video decoder (AVC, HEVC, VP9) over the NVDEC core in `rm/` |
| `libdrm/` | libdrm built for bionic from the pinned upstream release |
| `android/` | Pinned Android build inputs: NDK, AOSP headers, link stubs from the target image |
| `tests/` | Vulkan tests that run the built driver on a real GPU |
| `tools/` | Build helpers (pinned downloads, compiler launcher, Rust build records) |

`./build.sh` builds everything for the host and for Android x86_64 and i686.

Contributors: [RUST.md](RUST.md) is the standard every crate follows.

Licence: MIT (see [LICENSE](LICENSE)). Third-party code keeps its upstream licence.
