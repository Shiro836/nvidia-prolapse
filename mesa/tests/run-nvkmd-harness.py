#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline differential test of the nvkmd backends: nvkmd-harness, linked once against each backend.

    mesa/tests/run-nvkmd-harness.py                    both backends, compare their call logs
    mesa/tests/run-nvkmd-harness.py --backend c        one backend, its own checks only
    mesa/tests/run-nvkmd-harness.py pdev loss          only these suites

The harness binary links the objects of the host NVK build of each backend (build/nvk-host for the C
backend, build/nvk-host-rust for the Rust one: `mesa/build.py --target host [--nvkmd rust]`), exactly as
libvulkan_nouveau.so links them, with libc's open/fstat/close/calloc/free, libdrm's version and syncobj
calls, Mesa's diagnostic sinks and Mesa's clock replaced by the harness (--wrap) and the RM entries
defined by it.
Each suite checks its outcomes on every backend; with both backends the call logs (every RM, render
node and libdrm call, with arguments and results) must be byte-identical, suite by suite.
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
BUILDS = {"c": ROOT / "build/nvk-host", "rust": ROOT / "build/nvk-host-rust"}
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


def build_harness(backend):
    build = BUILDS[backend]
    if not (build / "build.ninja").exists():
        sys.exit(f"run-nvkmd-harness: {build} is not configured; run mesa/build.py --target host"
                 + (" --nvkmd rust" if backend == "rust" else ""))
    work = OUT / backend
    work.mkdir(parents=True, exist_ok=True)
    args_file = work / "link-args.txt"
    args_file.write_text("\n".join(link_args(build)) + "\n")
    env = dict(os.environ, NVKMD_HARNESS_LINK=str(args_file), CARGO_TARGET_DIR=str(work / "target"))
    subprocess.run(["cargo", "build", "--locked", "--offline", "-p", "nvkmd-harness", "--features", "mesa"],
                   cwd=ROOT, env=env, check=True)
    return work / "target/debug/nvkmd-harness"


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
    parser.add_argument("--backend", choices=["c", "rust", "both"], default="both")
    parser.add_argument("suites", nargs="*", help="suite names (default: all)")
    args = parser.parse_args()
    backends = ["c", "rust"] if args.backend == "both" else [args.backend]
    logs = {}
    for backend in backends:
        binary = build_harness(backend)
        run = subprocess.run([str(binary), *args.suites], capture_output=True, text=True, timeout=120)
        (OUT / backend / "calls.log").write_text(run.stdout)
        if run.returncode != 0:
            sys.stderr.write(run.stderr)
            sys.exit(f"nvkmd harness ({backend} backend): FAIL (exit {run.returncode}); "
                     f"calls so far in {OUT / backend / 'calls.log'}")
        logs[backend] = split_suites(run.stdout)
        print(f"nvkmd harness ({backend} backend): PASS {' '.join(logs[backend])}")
    if len(backends) == 2:
        failed = False
        for suite in logs["c"]:
            if logs["c"][suite] == logs["rust"].get(suite):
                print(f"nvkmd harness call log {suite}: C == Rust ({len(logs['c'][suite])} lines)")
                continue
            failed = True
            print(f"nvkmd harness call log {suite}: C != Rust")
            logs["c"][suite] = [line + "\n" for line in logs["c"][suite]]
            logs["rust"][suite] = [line + "\n" for line in logs["rust"].get(suite, [])]
            sys.stdout.writelines(difflib.unified_diff(logs["c"][suite], logs["rust"].get(suite, []),
                                                       "c", "rust", lineterm="\n", n=2))
        if failed:
            sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
