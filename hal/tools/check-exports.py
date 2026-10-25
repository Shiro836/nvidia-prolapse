#!/usr/bin/env python3
"""Artifact check for vulkan.nvidia.so (run with `python3 -I`); exits non-zero on any violation.
Reads the real ELF with llvm-readelf.

  - DT_SONAME is exactly vulkan.nvidia.so; DT_NEEDED is a subset of --allow-needed
  - the only defined dynamic symbol is HMI: OBJECT, GLOBAL, DEFAULT, sizeof(hwvulkan_module_t)
    (= 248 on LP64: 6 pointer-size fields + tag/versions + reserved[25]), in a writable section (the Android loader writes HMI.common.dso,
    frameworks/native vulkan/libvulkan/driver.cpp:209)
  - no TEXTREL, no executable stack
"""
import argparse
import re
import subprocess
import sys


def readelf(tool, args, path):
    return subprocess.run([tool] + args + [path], check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readelf", required=True)
    ap.add_argument("--allow-needed", required=True)
    ap.add_argument("--hmi-size", type=int, default=248,
                    help="sizeof(hwvulkan_module_t): 248 on LP64, 128 on LP32 (hw_module_t reserved[25] of "
                         "pointer-size words)")
    ap.add_argument("so")
    a = ap.parse_args()
    errors = []
    R, so = a.readelf, a.so

    dyn = readelf(R, ["-d"], so)
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.*)\]", dyn)
    soname = re.findall(r"\(SONAME\)\s+Library soname: \[(.*)\]", dyn)
    if soname != ["vulkan.nvidia.so"]:
        errors.append(f"SONAME {soname}, expected ['vulkan.nvidia.so']")
    allowed = set(a.allow_needed.split(","))
    for n in needed:
        if n not in allowed:
            errors.append(f"DT_NEEDED {n} not in the allowed set {sorted(allowed)}")
    if "TEXTREL" in dyn:
        errors.append("has TEXTREL")

    sections = {}
    for line in readelf(R, ["-S", "-W"], so).splitlines():
        m = re.match(r"^\s*\[\s*(\d+)\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+[0-9a-f]+\s+[0-9a-f]+\s+(\S*)", line)
        if m:
            sections[m.group(1)] = (m.group(2), m.group(3))

    defined = []
    for line in readelf(R, ["--dyn-syms", "-W"], so).splitlines():
        m = re.match(r"^\s*(\d+): ([0-9a-f]+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s*(\S*)$", line)
        if not m or m.group(1) == "0":
            continue
        _, _, size, typ, bind, vis, ndx, name = m.groups()
        if ndx != "UND":
            defined.append((name, int(size), typ, bind, vis, ndx))
    names = [d[0] for d in defined]
    if names != ["HMI"]:
        errors.append(f"defined dynamic symbols {names}, expected exactly ['HMI']")
    for name, size, typ, bind, vis, ndx in defined:
        if name != "HMI":
            continue
        if (typ, bind, vis, size) != ("OBJECT", "GLOBAL", "DEFAULT", a.hmi_size):
            errors.append(f"HMI is {typ} {bind} {vis} size {size}, expected OBJECT GLOBAL DEFAULT {a.hmi_size}")
        sec = sections.get(ndx)
        if not sec or "W" not in sec[1]:
            errors.append(f"HMI lives in section {sec}, which is not writable")

    ph = readelf(R, ["-l", "-W"], so)
    m = re.search(r"GNU_STACK\s+.*\s(RW?E?)\s+0x", ph)
    if m and "E" in m.group(1):
        errors.append("executable stack")

    if errors:
        for e in errors:
            print(f"check-exports: {so}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"check-exports: {so}: OK (exports only HMI [OBJECT {a.hmi_size} B, writable section {sections[defined[0][5]][0]}], "
          f"NEEDED {needed})")


if __name__ == "__main__":
    main()
