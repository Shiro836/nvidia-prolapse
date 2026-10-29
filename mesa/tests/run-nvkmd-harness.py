#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline test of NVK's nvkmd backend: nvkmd-harness, linked against the host NVK build.

    mesa/tests/run-nvkmd-harness.py                    every suite, call log compared with the golden one
    mesa/tests/run-nvkmd-harness.py pdev loss          only these suites
    mesa/tests/run-nvkmd-harness.py --update           rewrite the golden log (an intended change)

The harness binary links the objects of build/nvk-host (`mesa/build.py --target host`) exactly as
libvulkan_nouveau.so links them, with libc's open/fstat/close/calloc/free, libdrm's version and syncobj
calls, Mesa's diagnostic sinks and Mesa's clock replaced by the harness (--wrap) and the RM entries
defined by it.
Each suite checks its outcomes, and its call log (every RM, render node and libdrm call, with arguments
and results) must equal the suite's part of nvkmd-harness.calls byte for byte. That file is the log of
the C backend the Rust one was ported from, on the same suites; it changes only with a deliberate
behaviour change, in the commit that makes it.
No device file is opened and no GPU is touched.
"""
import argparse
import difflib
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/nvkmd-harness"
BUILD = ROOT / "build/nvk-host"
GOLDEN = Path(__file__).resolve().parent / "nvkmd-harness.calls"
# What the harness answers in place of libc, libdrm, Mesa's diagnostics and Mesa's clock.
WRAPS = ["open64", "close", "fstat64", "calloc", "free", "drmGetVersion", "drmFreeVersion", "drmGetCap",
         "drmSyncobjCreate", "drmSyncobjWait", "drmSyncobjDestroy", "drmSyncobjTimelineWait",
         "__vk_errorf", "__vk_errorv", "__vk_log_impl", "os_time_get_nano"]
# Link arguments of libvulkan_nouveau.so that make a shared library or bring in the real RM.
DROPPED = {"-shared", "-Wl,--whole-archive", "-Wl,--no-whole-archive", "-lnv_bionic_native", "-Wl,-Bsymbolic"}


def link_args(build):
    """libvulkan_nouveau.so's link line in `build`, turned into arguments for an executable."""
    command = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands",
                                       "src/nouveau/vulkan/libvulkan_nouveau.so"], text=True).splitlines()[-1]
    args = shlex.split(command)[1:]
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "-o":
            skip = True
            continue
        if arg in DROPPED or arg.startswith(("-Wl,-soname,", "-Wl,--version-script", "-Wl,-z,defs")):
            continue
        if not arg.startswith("-") and not Path(arg).is_absolute():
            arg = str(build / arg)
        out.append(arg)
    return out + [f"-Wl,--wrap={name}" for name in WRAPS]


def build_harness():
    if not (BUILD / "build.ninja").exists():
        sys.exit(f"run-nvkmd-harness: {BUILD} is not configured; run mesa/build.py --target host")
    OUT.mkdir(parents=True, exist_ok=True)
    args_file = OUT / "link-args.txt"
    args_file.write_text("\n".join(link_args(BUILD)) + "\n")
    env = dict(os.environ, NVKMD_HARNESS_LINK=str(args_file), CARGO_TARGET_DIR=str(OUT / "target"))
    subprocess.run(["cargo", "build", "--locked", "--offline", "-p", "nvkmd-harness", "--features", "mesa"],
                   cwd=ROOT, env=env, check=True)
    return OUT / "target/debug/nvkmd-harness"


def split_suites(log):
    suites, name = {}, None
    for line in log.splitlines():
        if line.startswith("== "):
            name = line[3:]
            suites[name] = []
        suites[name].append(line)
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true",
                        help="write the log of every suite to nvkmd-harness.calls instead of comparing")
    parser.add_argument("suites", nargs="*", help="suite names (default: all)")
    args = parser.parse_args()
    if args.update and args.suites:
        parser.error("--update records every suite")
    binary = build_harness()
    run = subprocess.run([str(binary), *args.suites], capture_output=True, text=True, timeout=120)
    (OUT / "calls.log").write_text(run.stdout)
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        sys.exit(f"nvkmd harness: FAIL (exit {run.returncode}); calls so far in {OUT / 'calls.log'}")
    logs = split_suites(run.stdout)
    print(f"nvkmd harness: PASS {' '.join(logs)}")
    if args.update:
        GOLDEN.write_text(run.stdout)
        print(f"nvkmd harness call log written to {GOLDEN}")
        return 0
    golden = split_suites(GOLDEN.read_text())
    failed = False
    for suite, lines in logs.items():
        if lines == golden.get(suite):
            print(f"nvkmd harness call log {suite}: identical to the golden log ({len(lines)} lines)")
            continue
        failed = True
        print(f"nvkmd harness call log {suite}: differs from the golden log")
        sys.stdout.writelines(difflib.unified_diff([line + "\n" for line in golden.get(suite, [])],
                                                   [line + "\n" for line in lines],
                                                   "golden", "harness", lineterm="\n", n=2))
    if failed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
