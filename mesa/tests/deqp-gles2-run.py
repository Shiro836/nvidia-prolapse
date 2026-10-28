#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Crash-resilient dEQP-GLES2 runner: runs a case list in batches, a crash or
timeout marks the running case Crash/Timeout and resumes after it.
usage: deqp-gles2-run.py <deqp-gles2 binary> <caselist> <outdir> [env-mode]
env-mode: nvrm (default) | llvmpipe | nvidia"""
import os, re, subprocess, sys, collections, json

exe, caselist, outdir = sys.argv[1:4]
mode = sys.argv[4] if len(sys.argv) > 4 else "nvrm"
os.makedirs(outdir, exist_ok=True)
cases = [l.strip() for l in open(caselist) if l.strip()]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../build/nvrm-host" + os.environ.get("NVRM_GL_DIR_SUFFIX", ""))
env = dict(os.environ)
if mode == "nvrm":
    # libGLESv2: src/mesa/glapi/es2api from Mesa 26 on, src/mapi/es2api on
    # older pins (both listed so baselines built on 25.2 still run).
    env["LD_LIBRARY_PATH"] = f"{root}/src/egl:{root}/src/mesa/glapi/es2api:{root}/src/mapi/es2api"
    env["LIBGL_DRIVERS_PATH"] = f"{root}/src/gallium/targets/dri"
elif mode == "llvmpipe":
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
elif mode == "nvidia":
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
args = ["--deqp-surface-type=pbuffer", "--deqp-gl-config-name=" + os.environ.get("DEQP_CONFIG", "rgba8888d0s0ms0"),
        "--deqp-surface-width=256", "--deqp-surface-height=256",
        "--deqp-log-images=disable", "--deqp-log-shader-sources=disable",
        "--deqp-watchdog=enable"]
results = {}
first_err = {}
batch = 0
# Re-queue every case without a result. dEQP runs a list in its own order, not
# the list's, so a crash can leave unrun cases before the one that died.
while any(c not in results for c in cases):
    rest = [c for c in cases if c not in results]
    lf = os.path.join(outdir, f"batch{batch}.txt")
    with open(lf, "w") as f:
        f.write("\n".join(rest) + "\n")
    cwd = os.path.dirname(exe)
    try:
        p = subprocess.run([exe, f"--deqp-caselist-file={lf}", "--deqp-log-filename=/dev/null"] + args,
                           cwd=cwd, env=env, capture_output=True, text=True, errors="replace", timeout=600)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = "timeout"
    cur = None
    for line in out.splitlines():
        m = re.match(r"Test case '(.*)'\.\.", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s+(Pass|Fail|NotSupported|QualityWarning|CompatibilityWarning|InternalError|Crash|Timeout|ResourceError)\b(.*)", line)
        if m and cur:
            results[cur] = m.group(1)
            if m.group(1) not in ("Pass", "NotSupported"):
                first_err.setdefault(cur, m.group(2).strip())
            cur = None
    done = sum(1 for c in rest if c in results)
    if cur is not None:  # died inside cur
        results[cur] = "Crash" if rc != "timeout" else "Timeout"
        msg = [l for l in err.splitlines() if "nvrm:" in l or "abort" in l.lower() or "Assert" in l]
        first_err[cur] = (msg[-1] if msg else f"rc={rc}")[:300]
    if cur is None and done == 0:
        # nothing ran: mark the first one and move on
        results[rest[0]] = "Crash"
        first_err[rest[0]] = ("no output: " + err[-300:]).replace("\n", " | ")
    batch += 1
cnt = collections.Counter(results.values())
json.dump({"counts": cnt, "results": results, "errors": first_err},
          open(os.path.join(outdir, "results.json"), "w"), indent=1)
print(mode, len(cases), dict(cnt))
