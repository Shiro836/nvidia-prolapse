#!/usr/bin/env python3
"""Inspect actual source NVK artifacts; no GPU enumeration or command execution."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# The pin lives in one place, mesa/build.py's `PIN = "<sha>"` line (build.sh reads it the same way).
pins = re.findall(r'^PIN = "([0-9a-f]{40})"', (HERE / "build.py").read_text(), re.MULTILINE)
if len(pins) != 1:
    raise SystemExit(f"mesa/build.py: expected exactly one PIN = \"<40-hex sha>\" line, found {len(pins)}")
report = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "mesa_pin": pins[0],
          "note": "Artifact and instance lifecycle checks only; GPU gates are separate", "checks": []}
for target in ("host", "android", "android32"):
    build = ROOT / f"build/nvk-{target}"
    library = build / "src/nouveau/vulkan/libvulkan_nouveau.so"
    command = [str(build / "inspect"), str(library)]
    environment = dict(os.environ)
    if target == "android":
        runtime = ROOT / "build/bionic-runtime64"
        environment["LD_LIBRARY_PATH"] = ":".join(str(path) for path in
            (runtime / "lib64", build / "runtime", ROOT / "build/android-work/deps/image-rootfs/system/lib64"))
        command.insert(0, str(runtime / "linker64"))
    elif target == "android32":
        environment["LD_LIBRARY_PATH"] = str(build / "runtime")
        command = ["unshare", "--user", "--map-root-user", "--pid", "--fork",
                   str(build / "runtime/linker"), *command]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
    assert "[libnv_bionic_native.so]" in dynamic
    if target != "host":
        assert "GLIBC_" not in subprocess.check_output(["readelf", "-V", str(library)], text=True)
    report["checks"].append({"target": target, "command": command,
                             "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH"),
                             "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                             "sha256": hashlib.sha256(library.read_bytes()).hexdigest(), "dynamic": dynamic})
    print(target + ": " + result.stdout.strip(), flush=True)
for name in ("rm-integration.patch", "build-compat.patch"):
    report[name + "_sha256"] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
(ROOT / "build/nvk-build-verification.json").write_text(json.dumps(report, indent=2) + "\n")
