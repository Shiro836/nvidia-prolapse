#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate mesa/nv_rm.h from rm-abi/ (the RM backend's C ABI) with cbindgen.

    mesa/gen-rm-header.py           write mesa/nv_rm.h
    mesa/gen-rm-header.py --check   exit 1 when the committed header differs

cbindgen is the one Mesa's Rust build pins (build/nak-tools, installed by
mesa/compiler/build.py --tools-only; CBINDGEN=<path> overrides the location); the
version is checked so that every checkout generates the same bytes.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "rm-abi"
HEADER = ROOT / "mesa/nv_rm.h"
CBINDGEN_VERSION = "0.29.2"  # mesa/compiler/build.py


def generate():
    tool = Path(os.environ.get("CBINDGEN", ROOT / "build/nak-tools/bin/cbindgen"))
    if not tool.exists():
        sys.exit(f"gen-rm-header: no {tool}; run mesa/compiler/build.py --tools-only --download-deps")
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout.split()
    if version != ["cbindgen", CBINDGEN_VERSION]:
        sys.exit(f"gen-rm-header: need cbindgen {CBINDGEN_VERSION}, found {' '.join(version)}")
    run = subprocess.run([tool, "--quiet", "--config", str(CRATE / "cbindgen.toml"), str(CRATE)],
                         capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"gen-rm-header: cbindgen failed:\n{run.stderr}")
    return run.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not HEADER.exists() or HEADER.read_text() != text:
            sys.exit("gen-rm-header: mesa/nv_rm.h is stale; run mesa/gen-rm-header.py")
        return 0
    HEADER.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
