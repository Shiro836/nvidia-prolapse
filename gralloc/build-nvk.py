#!/usr/bin/env python3
"""Build the existing gralloc wrapper over source NVK into isolated outputs."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gralloc"
NDK = Path("/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", choices=("host", "android", "android32", "all"), default="all")
args = parser.parse_args()
common = ["-std=c11", "-O2", "-g", f"-ffile-prefix-map={SOURCE}=gralloc",
          "-fPIC", "-fvisibility=hidden", "-Wall", "-Wextra", "-Werror", "-Iinclude",
          "-DNVG_SOURCE_RM", '-DNVG_ICD_SONAME="libvulkan_nouveau.so"',
          "-shared", "-Wl,-soname,libgbm_mesa_wrapper.so", "-Wl,-z,defs", "-Wl,-z,nodelete"]
for target, compiler, flags in [
    ("host", "cc", ["-I/usr/include/libdrm", "-ldl", "-lpthread"]),
    ("android", NDK / "x86_64-linux-android33-clang", ["-D_FILE_OFFSET_BITS=64", "-llog", "-ldl"]),
    ("android32", NDK / "i686-linux-android33-clang", ["-D_FILE_OFFSET_BITS=64", "-llog", "-ldl"]),
]:
    if args.target not in (target, "all"):
        continue
    output = ROOT / "build/source-native-integration" / target / "libgbm_mesa_wrapper.so"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(compiler), *common, "nvidia_gbm_wrapper.c", *flags, "-o", str(output)],
                   cwd=SOURCE, check=True)
    print(output, flush=True)
