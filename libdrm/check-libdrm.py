#!/usr/bin/env python3
"""Artifact check for the bionic libdrm.so.2 (run with `python3 -I`); exits non-zero on any violation.

Reads the real ELF with llvm-readelf:

  - ELF64 x86-64 (--arch x86: ELF32 i386), ET_DYN; DT_SONAME is exactly libdrm.so.2
  - DT_NEEDED is a subset of the bionic set (--allow-needed); no glibc SONAME; no TEXTREL
  - ABI shape against the host's own libdrm (--host-libdrm): the exported drm* sets are
    compared; when the host library is the pinned version, any difference is an error,
    otherwise the difference is printed
"""
import argparse
import os
import re
import subprocess
import sys

GLIBC_SONAMES = re.compile(r"^(libc\.so\.6|libm\.so\.6|libdl\.so\.2|libpthread\.so\.0|librt\.so\.1|ld-linux.*)$")


def readelf(tool, args, path):
    return subprocess.run([tool] + args + [path], check=True, capture_output=True, text=True).stdout


def dyn_exports(tool, so):
    """{name: (type, bind, vis)} of the defined dynamic symbols (version suffix stripped)."""
    out = {}
    for line in readelf(tool, ["--dyn-syms", "-W"], so).splitlines():
        m = re.match(r"^\s*(\d+): ([0-9a-f]+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s*(\S*)$", line)
        if not m or m.group(1) == "0" or m.group(7) == "UND":
            continue
        name = m.group(8).split("@")[0]
        out[name] = (m.group(4), m.group(5), m.group(6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readelf", required=True)
    ap.add_argument("--allow-needed", required=True)
    ap.add_argument("--arch", default="x86_64", choices=("x86_64", "x86"))
    ap.add_argument("--host-libdrm", default="/usr/lib/libdrm.so.2")
    ap.add_argument("--pinned-version", required=True, help="e.g. 2.4.134")
    ap.add_argument("so")
    a = ap.parse_args()
    R, so = a.readelf, a.so
    errors = []

    hdr = readelf(R, ["-h"], so)
    want_class, want_machine = (("ELF64", "Advanced Micro Devices X86-64") if a.arch == "x86_64"
                                else ("ELF32", "Intel 80386"))
    for key, want in (("Class", want_class), ("Type", "DYN"), ("Machine", want_machine)):
        m = re.search(rf"^\s*{key}:\s+(.*)$", hdr, re.M)
        if not m or not m.group(1).startswith(want):
            errors.append(f"ELF {key} is {m.group(1) if m else '?'}, expected {want}")

    dyn = readelf(R, ["-d"], so)
    soname = re.findall(r"\(SONAME\)\s+Library soname: \[(.*)\]", dyn)
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.*)\]", dyn)
    if soname != ["libdrm.so.2"]:
        errors.append(f"DT_SONAME {soname}, expected ['libdrm.so.2']")
    allowed = set(a.allow_needed.split(","))
    for n in needed:
        if GLIBC_SONAMES.match(n):
            errors.append(f"DT_NEEDED {n} is a glibc SONAME")
        elif n not in allowed:
            errors.append(f"DT_NEEDED {n} not in the bionic set {sorted(allowed)}")
    if "TEXTREL" in dyn:
        errors.append("has TEXTREL")
    print(f"check-libdrm: SONAME {soname} NEEDED {needed}")

    have = dyn_exports(R, so)

    host_real = os.path.realpath(a.host_libdrm)
    if not os.path.isfile(host_real):
        errors.append(f"host libdrm {a.host_libdrm} not found (needed for the ABI-shape comparison)")
    else:
        # upstream names the non-Android file libdrm.so.2.<patch>.0 (meson.build 'version')
        m = re.search(r"libdrm\.so\.2\.(\d+)\.0$", host_real)
        host_ver = f"2.4.{m.group(1)}" if m else "unknown"
        host = dyn_exports(R, host_real)
        hd = {n for n in host if n.startswith("drm")}
        od = {n for n in have if n.startswith("drm")}
        only_host, only_ours = sorted(hd - od), sorted(od - hd)
        print(f"check-libdrm: ABI shape vs host {host_real} (libdrm {host_ver}): host exports {len(hd)} drm*, "
              f"ours {len(od)}; host-only: {' '.join(only_host) or 'none'}; ours-only: {' '.join(only_ours) or 'none'}")
        if host_ver == a.pinned_version and (only_host or only_ours):
            errors.append(f"same version {host_ver} as the host but different drm* export sets")
        elif host_ver != a.pinned_version:
            print(f"check-libdrm: NOTE host libdrm is {host_ver}, pin is {a.pinned_version}; the diff above is informational")

    if errors:
        for e in errors:
            print(f"check-libdrm: FAIL {so}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"check-libdrm: OK {so}")


if __name__ == "__main__":
    main()
