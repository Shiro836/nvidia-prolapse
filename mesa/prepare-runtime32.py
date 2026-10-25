#!/usr/bin/env python3
"""Extract an unmodified Android platform dependency closure (read-only image)."""
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", type=Path, default=Path("/var/lib/waydroid/images/system.img"))
args = parser.parse_args()
out = ROOT / "build/nvk-android32/runtime"
out.mkdir(parents=True, exist_ok=True)
# libz.so: nvrm's shader cache (Mesa disk_cache, -Dzlib=enabled on Android).
pending = ["libcutils.so", "libhardware.so", "liblog.so", "libsync.so", "libnativewindow.so", "libz.so", "linker"]
seen = set()
while pending:
    name = pending.pop()
    if name in seen:
        continue
    seen.add(name)
    if "/" in name or name in (".", ".."):
        raise SystemExit(f"Invalid ELF dependency name: {name}")
    if name == "linker":
        source = "/system/apex/com.android.runtime/bin/linker"
    elif name in ("libc.so", "libdl.so", "libm.so", "libdl_android.so"):
        source = "/system/apex/com.android.runtime/lib/bionic/" + name
    else:
        source = "/system/lib/" + name
    destination = out / name
    temporary = out / (name + ".extract")
    temporary.unlink(missing_ok=True)
    result = subprocess.run(["debugfs", "-R", f"dump {source} {temporary}", str(args.image)],
                            capture_output=True, text=True, check=True)
    if not temporary.exists():
        raise SystemExit(f"Image lacks {source}: {result.stderr}")
    header = subprocess.check_output(["readelf", "-h", str(temporary)], text=True)
    if "ELF32" not in header or "Intel 80386" not in header:
        raise SystemExit(f"Expected x86 ELF32: {source}")
    temporary.replace(destination)
    dynamic = subprocess.check_output(["readelf", "-d", str(destination)], text=True)
    pending += re.findall(r"NEEDED.*\[(.*?)\]", dynamic)
    print(source)
(out / "linker").chmod(0o755)
drm = out / "libdrm.so.2"
drm.unlink(missing_ok=True)
drm.symlink_to(ROOT / "build/libdrm/build-android33-x86/libdrm.so")
