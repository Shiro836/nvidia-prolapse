# nvidia-prolapse

An open driver stack for NVIDIA GPUs in Android containers on Linux (Waydroid and similar).
Android apps get Vulkan from Mesa's NVK and NAK, running on a Rust backend that talks to the
host's NVIDIA kernel module directly. GLES comes from ANGLE on top of that Vulkan driver.
No proprietary NVIDIA userspace library runs inside the container.

This repository is being assembled from the [waydroid-nvidia](https://github.com/Shiro836/waydroid-nvidia)
project, where the stack runs today, and rewritten in Rust as it moves. Our own C and C++ get
ported to Rust. Third-party code (Mesa, ANGLE) and the C++ interfaces Android forces on us stay as
they are.

Contributors: [RUST.md](RUST.md) is the standard every crate follows.

Licence: MIT (see [LICENSE](LICENSE)). Third-party code keeps its upstream licence.
