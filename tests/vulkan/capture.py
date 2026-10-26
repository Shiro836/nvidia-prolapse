#!/usr/bin/env python3
"""Record what the driver sends to the kernel and the GPU during the tests/vulkan programs, and diff two
such recordings.

  tests/vulkan/capture.py record [--target all|host|android|android32] [--rust-profile release]
                                 --out build/capture/<name>
  tests/vulkan/capture.py diff build/capture/<a> build/capture/<b>
  tests/vulkan/capture.py normalise build/capture/<name>     (re-derive the text files from the raw ones)

`record` runs every program run.py would run (same commands, environment and build-identity checks),
serially, each one under
  * strace -ff -e trace=ioctl: the ioctl sequence per thread (raw logs: <case>.strace/<tid>);
  * the Rust RM's own command-stream capture (rm capture: NV_RM_CAPTURE, push mode): every submission's
    pushbuffer bytes, contexts, fence exports and texture/sampler pool contents (raw: <case>/cap-*.bin).
It then writes <case>.ioctl and <case>.push, normalised text in which every field that differs between two
runs of the same build is masked (see normalise_* below), and exits non-zero when a program fails.

`diff` compares the normalised files of two recordings case by case and exits 0 only when they are
identical. The raw files stay next to them for a closer look (rm/tools/nvcap*.py decode the captures).

Both recordings must come from the same GPU and driver: pushbuffer contents depend on the GPU class, and
the ioctl sequence on the kernel module version.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

profiles = run.profiles
ROOT = run.ROOT
MAGIC = b"NVCAP01\0"
# Cases whose pushbuffers cannot be captured, with the reason; their .push file says so and only the
# ioctl sequence is compared. The rm capture names its file cap-<pid>-<pci bus/device>-<n>.bin, so two
# devices on one GPU in one process (interop opens two) create and write the same file.
NO_PUSH = {"interop": "two devices of one GPU in one process write the same rm capture file"}


# Request codes whose count depends on GPU timing (dropped from the ioctl sequence):
#   SYNC_IOC_FILE_INFO = _IOWR('>', 4, struct sync_file_info) (linux/sync_file.h), a sync_file status
#     poll; how often it runs, and whether it runs at all on a waiting thread, depends on when the GPU
#     finishes.
#   DRM_IOCTL_NVIDIA_SEMSURF_FENCE_CREATE = DRM_IOWR(DRM_COMMAND_BASE + 0x15, 24 bytes)
#     (open-gpu-kernel-modules kernel-open/nvidia-drm/nvidia-drm-ioctl.h): a fence export creates a
#     sync_file only when the exported point has not completed yet; a completed one needs none.
TIMING_DEPENDENT = {0xc0383e04: "SYNC_IOC_FILE_INFO", 0xc0186455: "DRM_IOCTL_NVIDIA_SEMSURF_FENCE_CREATE"}
IOCTL = re.compile(r"ioctl\((\d+)(?:<([^>]*)>)?, (0x[0-9a-f]+|\d+), (.*)\) += (.*)$")


def normalise_ioctl(traces):
    """strace -ff -X raw -y output files (<case>.strace/<tid>) -> one block per thread, threads in tid order.

    Masked: thread ids (renumbered T0, T1, ... in creation order: the kernel assigns them), fd numbers
    (replaced by the path strace -y resolved; DRM render node numbers and anon-inode ids removed), and the
    argument (a user-space pointer). Dropped: the TIMING_DEPENDENT requests. Kept: the order of every
    other ioctl per thread, its request code (direction, type, number, size), target file and result.
    """
    out = []
    for index, trace in enumerate(sorted(traces, key=lambda path: int(path.name))):
        calls = []
        for line in trace.read_text().splitlines():
            if not line.startswith("ioctl("):
                continue
            match = IOCTL.match(line)
            if not match:
                raise SystemExit(f"{trace}: unparsed strace line: {line}")
            _, target, request, _, result = match.groups()
            if int(request, 0) in TIMING_DEPENDENT:
                continue
            target = re.sub(r"^/dev/dri/renderD\d+$", "/dev/dri/renderD*", target or "?")
            target = re.sub(r"^anon_inode:\[?([a-z_-]+)\]?.*$", r"\1", target)
            calls.append(f"ioctl({target}, {int(request, 0):#010x}) = {result}")
        if calls:
            out.append(f"== T{index}")
            out += calls
    return "\n".join(out) + "\n"


def records(path):
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        raise SystemExit(f"{path}: not an rm capture (magic {data[:8]!r})")
    offset = len(MAGIC)
    while offset + 8 <= len(data):
        tag, length = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + length > len(data):
            raise SystemExit(f"{path}: record at {offset - 8} runs past the end")
        yield tag, data[offset:offset + length]
        offset += length
    if offset != len(data):
        raise SystemExit(f"{path}: {len(data) - offset} trailing bytes")


# NVK uploads draw.sample_locations (struct nvk_root_descriptor_table, nvk_cmd_buffer.h: byte 28 of root
# table 0, one byte per sample, NVK_MAX_SAMPLES = 8) as a whole, but writes only the entries of the samples
# in use; the rest is whatever the command buffer's heap memory held (zero on glibc, garbage on bionic).
# Those bytes (samples 1-7) are masked in every 1INC SET_ROOT_TABLE_SELECTOR + LOAD_ROOT_TABLE upload
# (3D subchannel 0, methods 0x0504/0x0508 from TURING_A, clc597.h) that covers them.
SAMPLE_LOCATIONS = range(29, 36)
SET_ROOT_TABLE_SELECTOR = 0x0504


def masked_root_bytes(values):
    """Byte masks per word for one push: {word index: [byte positions]} for root-table uploads."""
    masks = {}
    i = 0
    while i < len(values):
        header = values[i]
        op, count = header >> 29, (header >> 16) & 0x1FFF
        subchannel, method = (header >> 13) & 7, (header & 0x1FFF) << 2
        if op == 4 or header == 0:  # IMMD (data in the header) or a NOP word
            i += 1
            continue
        if op not in (1, 3, 5):
            break  # not a method header: stop decoding, mask nothing further
        if op == 5 and subchannel == 0 and method == SET_ROOT_TABLE_SELECTOR and count >= 2 \
                and i + 1 < len(values):
            selector = values[i + 1]
            table, offset = selector & 7, (selector >> 8) & 0xFF
            if table == 0:
                for k in range(count - 1):
                    word = i + 2 + k
                    hit = [b for b in range(4) if offset + 4 * k + b in SAMPLE_LOCATIONS]
                    if hit and word < len(values):
                        masks[word] = hit
        i += 1 + count
    return masks


def words(payload, offset, count, masks=None):
    values = struct.unpack_from(f"<{count}I", payload, offset)
    masks = masks(values) if masks else {}
    text = []
    for index, value in enumerate(values):
        digits = list(f"{value:08x}")
        for byte in masks.get(index, ()):
            digits[6 - 2 * byte:8 - 2 * byte] = "??"
        text.append("".join(digits))
    return [" ".join(text[i:i + 8]) for i in range(0, count, 8)]


def normalise_capture(path):
    """One rm capture file -> text, one line per record, pushbuffer words 8 per line.

    Dropped: TIME records (GPU start/end timestamps of a submission, written whenever the driver reads
    them back, so their position depends on GPU timing). Masked: CPU and GPU timestamps (START cpu/gpu ns,
    SUBMIT/MARK/END cpu ns), the pid in START, context ids (the address of the context object in the
    driver's heap), renumbered C0, C1, ... by first appearance, the id of a fence-export MARK (kinds 2 and
    3, sync-file and opaque fd: the sync object id nvkmd passes to nv_rm_sync_export_fd, a heap address),
    renumbered F0, F1, ... the same way, and the unwritten NVK sample-location bytes (SAMPLE_LOCATIONS).
    Kept: every other pushbuffer byte, push GPU address, size and flags, submission tokens, wait/signal
    counts, context engines and class, fence-export kinds, pool addresses and contents.
    """
    contexts = {}
    exports = {}

    def ctx(value):
        return contexts.setdefault(value, f"C{len(contexts)}")

    def export(value):
        return exports.setdefault(value, f"F{len(exports)}")

    out = []
    for tag, p in records(path):
        if tag == 1:
            out.append(f"START push={struct.unpack_from('<I', p, 20)[0]}")
        elif tag == 2:
            _, c, token, waits, foreign, signals, pushes = struct.unpack_from("<QQQIIII", p)
            out.append(f"SUBMIT {ctx(c)} token={token} waits={waits} foreign={foreign} signals={signals} "
                       f"pushes={pushes}")
            offset = 40
            for _ in range(pushes):
                address, size, flags, nbytes = struct.unpack_from("<QIII", p, offset)
                offset += 20
                out.append(f"  push {address:#x} size={size} flags={flags:#x} bytes={nbytes}")
                out += ["    " + line for line in words(p, offset, nbytes // 4, masked_root_bytes)]
                offset += nbytes
        elif tag == 3:
            pass
        elif tag == 4:
            _, c, token, kind, _ = struct.unpack_from("<QQQII", p)
            out.append(f"MARK {ctx(c)} token={export(token) if kind in (2, 3) else token} kind={kind}")
        elif tag == 5:
            out.append(f"END reason={struct.unpack_from('<QI', p)[1]}")
        elif tag == 6:
            c, engines, klass = struct.unpack_from("<QII", p)
            out.append(f"CONTEXT {ctx(c)} engines={engines:#x} class={klass:#x}")
        elif tag == 7:
            token, address, subchannel, kind, entries, nbytes = struct.unpack_from("<QQIIII", p)
            out.append(f"POOL token={token} {address:#x} subchannel={subchannel} kind={kind} "
                       f"entries={entries} bytes={nbytes}")
            out += ["    " + line for line in words(p, 32, nbytes // 4)]
        else:
            raise SystemExit(f"{path}: unknown record tag {tag}")
    return "\n".join(out) + "\n"


def normalise(out, case):
    """(Re)write <case>.ioctl and <case>.push from the raw files of one recorded case."""
    traces = out / f"{case}.strace"
    (out / f"{case}.ioctl").write_text(normalise_ioctl(list(traces.iterdir())))
    name = case.split("-", 1)[1]
    caps = sorted((out / case).glob("cap-*.bin"))
    if name in NO_PUSH:
        text = f"not captured: {NO_PUSH[name]}\n"
    elif not caps:
        raise SystemExit(f"{case}: the rm capture wrote no file into {out / case}")
    else:
        text = "".join(f"=== capture {i}\n" + normalise_capture(c) for i, c in enumerate(caps))
    (out / f"{case}.push").write_text(text)


def record_target(out, target, args):
    identity = profiles.verify_metadata(ROOT, target, args.rust_profile)
    failed = []
    for name, command in run.cases(target, args.suite):
        env, argv = run.launch(target, command, args.rust_profile)
        case = f"{target}-{name}"
        capture = out / case
        capture.mkdir()
        if name not in NO_PUSH:
            # The capture starts at the first submit after <dir>/go exists and removes it when it ends.
            (capture / "go").write_text("push\n")
            env = {**env, "NV_RM_CAPTURE": str(capture), "NV_RM_CAPTURE_MODE": "push",
                   "NV_RM_CAPTURE_FRAMES": "1000000"}
        traces = out / f"{case}.strace"
        traces.mkdir()
        strace = ["strace", "-ff", "-qq", "-y", "-X", "raw", "-e", "trace=ioctl", "-o", str(traces / "t"),
                  "--"]
        with (out / f"{case}.log").open("w") as log:
            result = subprocess.run(strace + argv, env={**os.environ, **env}, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT)
        (capture / "go").unlink(missing_ok=True)
        print(f"[{case}] exit={result.returncode}", flush=True)
        if result.returncode:
            failed.append(case)
        for trace in traces.iterdir():
            trace.rename(traces / trace.name.removeprefix("t."))
        normalise(out, case)
    if profiles.verify_metadata(ROOT, target, args.rust_profile) != identity:
        raise SystemExit(f"{target}: Rust build metadata changed during the recording")
    return identity, failed


def record(args):
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; record into a new directory")
    out.mkdir(parents=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    targets = ("host", "android", "android32") if args.target == "all" else (args.target,)
    builds, failed = {}, []
    for target in targets:
        builds[target], target_failed = record_target(out, target, args)
        failed += target_failed
    (out / "recording.json").write_text(json.dumps({"checkout": head, "rust_profile": args.rust_profile,
                                                     "rust_builds": builds}, indent=2) + "\n")
    if failed:
        raise SystemExit(f"failed: {' '.join(failed)}")
    print(f"recorded into {out}")
    return 0


def renormalise(args):
    for traces in sorted(args.dir.glob("*.strace")):
        normalise(args.dir, traces.name.removesuffix(".strace"))
    print(f"normalised {args.dir}")
    return 0


def diff(args):
    names = sorted({p.name for d in (args.a, args.b) for p in d.iterdir() if p.suffix in (".ioctl", ".push")})
    differing = 0
    for name in names:
        a, b = args.a / name, args.b / name
        if not a.exists() or not b.exists():
            print(f"{name}: only in {args.a if a.exists() else args.b}")
            differing += 1
            continue
        lines = list(difflib.unified_diff(a.read_text().splitlines(), b.read_text().splitlines(),
                                          str(a), str(b), lineterm="", n=2))
        if lines:
            differing += 1
            print("\n".join(lines[:args.max_lines]))
            if len(lines) > args.max_lines:
                print(f"... {len(lines) - args.max_lines} more diff lines")
        else:
            print(f"{name}: identical")
    print(f"{differing} of {len(names)} files differ")
    return 1 if differing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--target", choices=("host", "android", "android32", "all"), default="all")
    rec.add_argument("--rust-profile", choices=profiles.PROFILES, default="release")
    rec.add_argument("--suite", choices=("core", "hal", "all"), default="all")
    rec.add_argument("--out", type=Path, required=True)
    dif = sub.add_parser("diff")
    dif.add_argument("a", type=Path)
    dif.add_argument("b", type=Path)
    dif.add_argument("--max-lines", type=int, default=200)
    nor = sub.add_parser("normalise", help="rewrite the .ioctl/.push files of a recording from its raw files")
    nor.add_argument("dir", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        if shutil.which("strace") is None:
            raise SystemExit("strace is required (host package strace)")
        return record(args)
    return diff(args) if args.command == "diff" else renormalise(args)


if __name__ == "__main__":
    sys.exit(main())
