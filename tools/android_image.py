#!/usr/bin/env python3
"""Read unmodified files out of an Android ext4 image (system.img) without mounting it.

`extract()` copies one file with debugfs (read-only on the image, no root needed) and checks its ELF class
and machine with readelf before it replaces the destination, so a wrong-ABI or missing file fails loudly
instead of leaving a partial runtime behind.
"""
from pathlib import Path
import subprocess

ABIS = {"x86": ("ELF32", "Intel 80386"), "x86_64": ("ELF64", "Advanced Micro Devices X86-64")}


def extract(image, source, destination, abi):
    """Dump `source` (an absolute path inside `image`) to `destination`; raise SystemExit when it is absent
    or not an ELF file of `abi` (a key of ABIS)."""
    elf_class, machine = ABIS[abi]
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".extract")
    temporary.unlink(missing_ok=True)
    result = subprocess.run(["debugfs", "-R", f"dump {source} {temporary}", str(image)],
                            capture_output=True, text=True, check=True)
    if not temporary.exists():
        raise SystemExit(f"Image lacks {source}: {result.stderr}")
    header = subprocess.check_output(["readelf", "-h", str(temporary)], text=True)
    if elf_class not in header or machine not in header:
        raise SystemExit(f"Expected {abi} {elf_class}: {source}")
    temporary.replace(destination)
