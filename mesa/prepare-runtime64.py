#!/usr/bin/env python3
"""Extract the image's unmodified 64-bit bionic (linker64, libc, libdl, libm) into build/bionic-runtime64.

The x86_64 host tests (tests/vulkan/run.py --target android, mesa/verify-artifacts.py) start their Android
programs as `build/bionic-runtime64/linker64 <program>` with build/bionic-runtime64/lib64 first on
LD_LIBRARY_PATH. Only bionic itself comes from here: in the image, /system/lib64/{libc,libdl,libm}.so are
symlinks into the runtime APEX, so the copies under build/android-work/deps/image-rootfs cannot stand in.
Every other platform library is read from that image-rootfs. ld-android.so, which libc and libdl need, is
provided by linker64 itself (bionic's linker registers it as a built-in library).
"""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from android_image import extract  # noqa: E402

APEX = "/system/apex/com.android.runtime"
LIBRARIES = ("libc.so", "libdl.so", "libm.so")
BUILT_IN = {"ld-android.so"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", type=Path, default=Path("/var/lib/waydroid/images/system.img"))
args = parser.parse_args()
out = ROOT / "build/bionic-runtime64"
(out / "lib64").mkdir(parents=True, exist_ok=True)
files = [(f"{APEX}/bin/linker64", out / "linker64")]
files += [(f"{APEX}/lib64/bionic/{name}", out / "lib64" / name) for name in LIBRARIES]
for source, destination in files:
    extract(args.image, source, destination, "x86_64")
    print(f"{hashlib.sha256(destination.read_bytes()).hexdigest()}  {source}")
(out / "linker64").chmod(0o755)
for name in LIBRARIES:
    dynamic = subprocess.check_output(["readelf", "-d", str(out / "lib64" / name)], text=True)
    missing = set(re.findall(r"NEEDED.*\[(.*?)\]", dynamic)) - set(LIBRARIES) - BUILT_IN
    if missing:
        raise SystemExit(f"{name} needs {sorted(missing)}, which build/bionic-runtime64 does not hold")
