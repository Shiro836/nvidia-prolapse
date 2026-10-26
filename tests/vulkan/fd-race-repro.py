#!/usr/bin/env python3
"""Reproduce the load-dependent fd-count and wake-latency failures.

Runs the host anb-roundtrip (or the RM two-waiter unit test) repeatedly while a
stress-ng CPU load runs, and counts failures per check. Build first
(./build.sh && tests/vulkan/build.py). The stress-ng process is started here and
stopped by its own pid; nothing else is signalled.

modes (anb-roundtrip):
  cache    a fresh MESA_SHADER_CACHE_DIR per run: Mesa's disk$ writer threads
           (nice 19) hold <hash>.tmp descriptors while the test counts fds
  nocache  MESA_SHADER_CACHE_DISABLE=true: the control for `cache`
  watcher  warm cache; the nvk-rm-foreign thread is switched to SCHED_IDLE as
           soon as it exists, which widens the window in which it still owns an
           imported sync_file duplicate after the client's wait returned
mode waiters: the rm two-waiter p99 latency unit test (no GPU)
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as regression  # noqa: E402  (tests/vulkan/run.py)

ROOT = regression.ROOT
CHECKS = {
    "fds-around-C": "[FAIL] fds around C",
    "acquire-fd-count": "AcquireImageANDROID(sem+fence) + ",
}


def watcher_tid(pid):
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            if (task / "comm").read_text().strip() == "nvk-rm-foreign":
                return int(task.name)
        except OSError:
            pass
    return None


def run_anb(mode, cache_root, rust_profile):
    (name, command), = [c for c in regression.cases("host", "hal") if c[0] == "anb"]
    env, argv = regression.launch("host", command, rust_profile)
    env = {**os.environ, **env}
    if mode == "nocache":
        env["MESA_SHADER_CACHE_DISABLE"] = "true"
    else:
        cache = cache_root / ("cold" if mode == "cache" else "warm")
        if mode == "cache":
            shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(exist_ok=True)
        env["MESA_SHADER_CACHE_DIR"] = str(cache)
    with tempfile.TemporaryFile("w+") as log:
        child = subprocess.Popen(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if mode == "watcher":
            while child.poll() is None:
                tid = watcher_tid(child.pid)
                if tid:
                    subprocess.run(["chrt", "--idle", "-p", "0", str(tid)], check=False)
                    break
                time.sleep(0.0005)
        rc = child.wait()
        log.seek(0)
        text = log.read()
    failed = [k for k, marker in CHECKS.items()
              if any(marker in line and "[FAIL]" in line for line in text.splitlines())]
    other = [line for line in text.splitlines()
             if "[FAIL]" in line and not any(m in line for m in CHECKS.values())]
    return rc, failed, other


def run_waiters(rust_profile):
    argv = ["cargo", "test", "-q", "-p", "nv-bionic-native", "--lib",
            *(["--release"] if rust_profile == "release" else []),
            "waiters::tests::two_waiters_on_one_value_both_wake_promptly", "--", "--exact"]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    p99 = [line for line in result.stdout.splitlines() + result.stderr.splitlines() if "p99" in line]
    return result.returncode, p99


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("cache", "nocache", "watcher", "waiters"))
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--load", type=int, default=2 * os.cpu_count(),
                        help="stress-ng --cpu workers (0 = none); default 2 x CPUs")
    parser.add_argument("--rust-profile", choices=("dev", "release"), default="release")
    args = parser.parse_args()
    if args.mode == "waiters":  # build before the load starts
        subprocess.run(["cargo", "test", "-q", "--no-run", "-p", "nv-bionic-native", "--lib",
                        *(["--release"] if args.rust_profile == "release" else [])], cwd=ROOT, check=True)
    stress = None
    if args.load:
        stress = subprocess.Popen(["stress-ng", "--cpu", str(args.load), "--timeout", "3600s"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    counts = {}
    try:
        with tempfile.TemporaryDirectory(prefix="fd-race-cache-") as cache_root:
            for i in range(1, args.runs + 1):
                if args.mode == "waiters":
                    rc, p99 = run_waiters(args.rust_profile)
                    if rc:
                        counts["p99>=1ms"] = counts.get("p99>=1ms", 0) + 1
                        print(f"run {i}: {' '.join(p99)}", flush=True)
                    continue
                rc, failed, other = run_anb(args.mode, Path(cache_root), args.rust_profile)
                for key in failed + (["other"] if other else []):
                    counts[key] = counts.get(key, 0) + 1
                if rc or failed or other:
                    print(f"run {i}: rc={rc} {failed} {other[:2]}", flush=True)
    finally:
        if stress:
            stress.terminate()
            stress.wait()
    print(f"{args.mode}: {args.runs} runs, load {args.load}: "
          + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no failures"))
    return 1 if counts else 0


if __name__ == "__main__":
    sys.exit(main())
