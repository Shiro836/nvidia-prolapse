#!/usr/bin/env python3
"""Check that a test program's RM clients are freed in its own process when it exits.

RM frees a client when the last reference to its control file (/dev/nvidiactl) is dropped, in
whichever task drops it. If that is another process, RM's automatic unmap of the client's CPU
mappings fails there and the kernel logs one `rs_client.c` assertion per mapping. This script
makes another process hold the control file across the program's exit: it starts one
tests/vulkan program (run.py's command and environment), stops it once it holds --vmas
/dev/nvidia* mappings, copies every /dev/nvidiactl descriptor of it with pidfd_getfd, lets it
run to a normal exit, then closes the copies and counts the kernel log's `rs_client.c` lines
in that window. A program killed by a signal runs no exit handler: --kill shows that case, in
which the kernel logs the assertions for the NVIDIA driver as well.

    tests/vulkan/exit-holder-repro.py --target android --case graphics --runs 5
    tests/vulkan/exit-holder-repro.py --target android --lib <dir> -- <program> <args>

A program that frees its devices before it exits shows nothing either way; GL programs that
exit without eglTerminate are the usual case. With `-- program args` the program runs under
run.py's environment for --target, with every --lib directory after the driver's.

RM's automatic unmap checks the calling process's user, so the holder must run as another
user than the program: run this script as root with --as-user <name> (the program then runs as
that user, as a root /proc reader or service would hold an app's files). A holder of the same
user frees the client without assertions.

Needs: the build run.py checks, `journalctl -k` readable, pidfd_getfd (Linux 5.6+; root, or a
child with yama ptrace_scope <= 1). Exit status 1 when any run logs an assertion.
"""
import argparse
import ctypes
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as regression  # noqa: E402  (tests/vulkan/run.py)

libc = ctypes.CDLL(None, use_errno=True)
SYS_PIDFD_OPEN, SYS_PIDFD_GETFD = 434, 438


def descendants(pid):
    """pid and every process below it."""
    found, queue = [], [pid]
    while queue:
        current = queue.pop()
        found.append(current)
        for task in Path(f"/proc/{current}/task").glob("*"):
            try:
                queue += [int(c) for c in (task / "children").read_text().split()]
            except OSError:
                pass
    return found


def nvidia_vmas(pid):
    try:
        return sum("/dev/nvidia" in line for line in Path(f"/proc/{pid}/maps").read_text().splitlines())
    except OSError:
        return 0


def ctl_fds(pid):
    fds = []
    for entry in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if os.readlink(entry) == "/dev/nvidiactl":
                fds.append(int(entry.name))
        except OSError:
            pass
    return sorted(fds)


def asserts_since(start):
    log = subprocess.run(["journalctl", "-k", "--no-pager", "-o", "short-unix", "--since", f"@{start}"],
                         capture_output=True, text=True, check=True).stdout
    return sum("rs_client.c" in line for line in log.splitlines())


def one_run(env, argv, vmas, kill, timeout):
    # The journal window starts at a whole second after every earlier run's lines.
    time.sleep(1.0)
    start = math.ceil(time.time())
    time.sleep(start - time.time())
    top = subprocess.Popen(argv, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           cwd=regression.ROOT)
    target = None
    deadline = time.monotonic() + timeout
    while target is None:
        if time.monotonic() > deadline or top.poll() is not None:
            top.kill()
            top.wait()
            raise RuntimeError(f"the program never held {vmas} /dev/nvidia* mappings")
        target = next((p for p in descendants(top.pid) if nvidia_vmas(p) >= vmas), None)
    os.kill(target, signal.SIGSTOP)
    held = []
    try:
        pidfd = libc.syscall(SYS_PIDFD_OPEN, target, 0)
        if pidfd < 0:
            raise OSError(ctypes.get_errno(), "pidfd_open")
        for fd in ctl_fds(target):
            copy = libc.syscall(SYS_PIDFD_GETFD, pidfd, fd, 0)
            if copy < 0:
                raise OSError(ctypes.get_errno(), f"pidfd_getfd {fd}")
            held.append(copy)
        os.close(pidfd)
        os.kill(target, signal.SIGKILL if kill else signal.SIGCONT)
        status = top.wait(timeout=timeout)
        time.sleep(0.3)
    finally:
        for fd in held:
            os.close(fd)
    time.sleep(0.5)
    return status, len(held), asserts_since(start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=("host", "android", "android32"), default="android")
    parser.add_argument("--case", default="graphics", help="a run.py case name")
    parser.add_argument("--rust-profile", default="release")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--vmas", type=int, default=8, help="stop the program at this many /dev/nvidia* mappings")
    parser.add_argument("--kill", action="store_true", help="SIGKILL instead of a normal exit")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--as-user", help="run the program as this user (the script runs as root)")
    parser.add_argument("--lib", type=Path, action="append", default=[], help="extra library directory")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- program and arguments")
    args = parser.parse_args()
    regression.profiles.verify_metadata(regression.ROOT, args.target, args.rust_profile)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command:
        args.case = Path(command[0]).name
    else:
        command = dict(regression.cases(args.target, "all")).get(args.case)
        if command is None:
            parser.error(f"no case {args.case} for {args.target}")
    env, argv = regression.launch(args.target, command, args.rust_profile)
    paths = env["LD_LIBRARY_PATH"].split(":")
    env["LD_LIBRARY_PATH"] = ":".join([paths[0], *map(str, args.lib), *paths[1:]])
    if args.as_user:
        argv = ["setpriv", f"--reuid={args.as_user}", f"--regid={args.as_user}", "--init-groups",
                "env", *(f"{k}={v}" for k, v in env.items()), *argv]
    failed = 0
    for index in range(args.runs):
        status, held, count = one_run(env, argv, args.vmas, args.kill, args.timeout)
        print(f"run {index + 1}: exit {status}, held {held} control fds, rs_client.c lines {count}", flush=True)
        failed += count > 0
    print(f"exit-holder {args.target}/{args.case} {'SIGKILL' if args.kill else 'normal exit'}: "
          f"{failed}/{args.runs} runs with assertions")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
