#!/usr/bin/env python3
"""Build the Vulkan test programs that tests/vulkan/run.py runs, for the host and both Android ABIs.

  build/native-vulkan-probe/      probe-<target> (compute + copy), interop-<target> (dma-buf + sync_fd between
                                  two devices), generate.spv, transform.spv
  build/native-vulkan-offscreen/  probe-<target> (triangle -> tiled image -> readback), vertex.spv, fragment.spv
  hal/out/host/                   the HAL's host test programs (hal/Makefile); the Android ones come from
                                  hal/build-nvk-tests.py

target = host (cc) | android (x86_64) | android32 (i686), NDK r29 API 33 (android/provision.sh). No GPU is used.
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NDK = ROOT / "build/android-work/ndk/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin"
COMPILERS = {"host": "cc", "android": str(NDK / "x86_64-linux-android33-clang"),
             "android32": str(NDK / "i686-linux-android33-clang")}
WARN = ["-std=c11", "-Wall", "-Wextra", "-Werror"]


def run(*command, **kwargs):
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def main():
    core, graphics = ROOT / "build/native-vulkan-probe", ROOT / "build/native-vulkan-offscreen"
    core.mkdir(parents=True, exist_ok=True)
    graphics.mkdir(parents=True, exist_ok=True)
    shaders = HERE / "shaders"
    for name in ("generate", "transform"):
        run("glslangValidator", "-V", "--target-env", "vulkan1.0", shaders / f"{name}.comp", "-o", core / f"{name}.spv")
        run("spirv-val", "--target-env", "vulkan1.0", core / f"{name}.spv")
    for stage, name in (("vert", "vertex"), ("frag", "fragment")):
        run("glslangValidator", "-V", shaders / f"offscreen.{stage}", "-o", graphics / f"{name}.spv")
        run("spirv-val", graphics / f"{name}.spv")
    for target, cc in COMPILERS.items():
        run(cc, *WARN, HERE / "native-vulkan-probe.c", "-ldl", "-o", core / f"probe-{target}")
        run(cc, *WARN, HERE / "native-vulkan-interop.c", "-ldl", "-o", core / f"interop-{target}")
        run(cc, *WARN, HERE / "native-vulkan-offscreen.c", "-ldl", "-o", graphics / f"probe-{target}")
    run("make", "-C", ROOT / "hal", "out/host/anb-roundtrip", "out/host/astc-roundtrip", "out/host/etc2-roundtrip",
        "out/host/ahb-roundtrip", "out/host/ahb-yuv")
    print(f"Built the Vulkan test programs into {core}, {graphics} and {ROOT / 'hal/out/host'}")


if __name__ == "__main__":
    main()
