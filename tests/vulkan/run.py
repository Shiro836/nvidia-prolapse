#!/usr/bin/env python3
"""Print or run the existing source-native Vulkan regression commands serially.

Default is a checked dry run: no process is loaded and no GPU work is submitted.
Build the programs first (tests/vulkan/build.py). The android (x86_64) target runs the
programs under the image's unmodified bionic linker64/libc/libdl/libm in build/bionic-runtime64
(mesa/prepare-runtime64.py), android32 under build/nvk-android32/runtime (mesa/prepare-runtime32.py);
build.sh extracts both. --nvkmd selects NVK's kernel-driver backend: the C one (build/nvk-<target>) or the
Rust one (build/nvk-<target>-rust, mesa/build.py --nvkmd rust).
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import native_rust_profile as profiles

ROOT = Path(__file__).resolve().parents[2]


def cases(target, suite, nvkmd="c"):
    """Return existing probe invocations; no shell parsing or implicit fallback."""
    icd = profiles.nvk_dir(ROOT, target, nvkmd) / "src/nouveau/vulkan/libvulkan_nouveau.so"
    integration = ROOT / "build/source-native-integration" / target
    hal = integration / "vulkan.nvidia.so"
    core = ROOT / "build/native-vulkan-probe"
    graphics = ROOT / "build/native-vulkan-offscreen"
    result = []
    if suite in ("core", "all"):
        result += [
            ("compute", [core / f"probe-{target}", "--icd", "--driver", icd,
                         "--shader", core / "generate.spv", "--transform",
                         core / "transform.spv", "--negative"]),
            ("graphics", [graphics / f"probe-{target}", icd,
                          graphics / "vertex.spv", graphics / "fragment.spv"]),
            ("interop", [core / f"interop-{target}", "--driver", icd]),
        ]
        if target == "host":  # built for the host only (tests/vulkan/build.py)
            result.append(("copy-indirect", [core / "copy-indirect-host", "--driver", icd,
                                             "--transform", core / "transform.spv"]))
    if suite in ("hal", "all"):
        tests = ROOT / "hal/out/host" if target == "host" else integration
        # The strict default requires an actually pending release fd. Never
        # carry --allow-synchronous-submit into an asynchronous regression gate.
        result += [("anb", [tests / "anb-roundtrip", hal, "--bounded"]),
                   ("astc", [tests / "astc-roundtrip", hal]),
                   ("etc2", [tests / "etc2-roundtrip", hal])]
        if target == "host":
            result += [("ahb", [tests / "ahb-roundtrip", hal]),
                       ("ahb-yuv", [tests / "ahb-yuv", hal])]
        # AHB's standalone host fixture exports a fake AHardwareBuffer API.
        # Android AHB must instead be checked against real gralloc in the guest.
    return result


def launch(target, command, rust_profile="dev", nvkmd="c"):
    env = {"DISPLAY": "", "WAYLAND_DISPLAY": ""}
    nvk = profiles.nvk_dir(ROOT, target, nvkmd)
    # The bionic runtime is prepared once per target, in the C backend's build dir.
    nvk_runtime = profiles.nvk_dir(ROOT, target) / "runtime"
    rust = profiles.rust_library(ROOT, target, rust_profile).parent
    # The selected Rust directory precedes every other library directory, so a
    # stale colocated bridge cannot shadow the verified profile.
    paths = [rust, nvk / "src/nouveau/vulkan"]
    if target == "android":
        runtime = ROOT / "build/bionic-runtime64"
        paths += [runtime / "lib64",
                  nvk_runtime, ROOT / "build/android-work/deps/image-rootfs/system/lib64"]
        command = [runtime / "linker64", *command]
    elif target == "android32":
        paths += [nvk_runtime]
        # Android i686's pthread TID field cannot represent this host's large
        # PIDs. A new user/PID namespace gives the test a small TID, no sudo.
        command = ["unshare", "--user", "--map-root-user", "--pid", "--fork",
                   nvk_runtime / "linker", *command]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, paths))
    return env, list(map(str, command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("host", "android", "android32", "all"), default="all")
    parser.add_argument("--rust-profile", choices=profiles.PROFILES, default="dev")
    parser.add_argument("--suite", choices=("core", "hal", "all"), default="all")
    parser.add_argument("--nvkmd", choices=profiles.BACKENDS, default="c",
                        help="NVK's kernel-driver backend: nvkmd_rm.c or the Rust nvkmd-rm (mesa/build.py --nvkmd)")
    parser.add_argument("--run", action="store_true", help="run serial GPU tests after coordinating the GPU slot")
    parser.add_argument("--logs", type=Path, default=ROOT / "build/native-vulkan-regression")
    args = parser.parse_args()
    targets = ("host", "android", "android32") if args.target == "all" else (args.target,)
    plan = []
    builds = {}
    for target in targets:
        try:
            builds[target] = profiles.verify_metadata(ROOT, target, args.rust_profile, args.nvkmd)
        except RuntimeError as error:
            parser.error(str(error))
        for name, command in cases(target, args.suite, args.nvkmd):
            for item in command:
                if isinstance(item, Path) and not item.is_file():
                    parser.error(f"missing artifact: {item}")
            env, argv = launch(target, command, args.rust_profile, args.nvkmd)
            for item in argv:
                if item.startswith("/") and not Path(item).is_file():
                    parser.error(f"missing launch artifact: {item}")
            plan.append((target, name, env, argv))
    if args.run:
        args.logs.mkdir(parents=True, exist_ok=True)
        import json
        (args.logs / "rust-builds.json").write_text(json.dumps(builds, indent=2) + "\n")
    print(f"Rust profile: {args.rust_profile}; nvkmd backend: {args.nvkmd}; "
          "successful build identities verified", flush=True)
    for target, name, env, argv in plan:
        printed = shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *argv])
        print(f"[{target}/{name}]\n{printed}", flush=True)
        if not args.run:
            continue
        log = args.logs / f"{target}-{name}.log"
        with log.open("w") as output:
            output.write(printed + "\n")
            output.flush()
            result = subprocess.run(argv, env={**os.environ, **env}, cwd=ROOT,
                                    stdout=output, stderr=subprocess.STDOUT)
        print(f"exit={result.returncode} log={log}", flush=True)
        if result.returncode:
            print(log.read_text()[-12000:], file=sys.stderr)
            return result.returncode
    for target, expected in builds.items():
        if profiles.verify_metadata(ROOT, target, args.rust_profile, args.nvkmd) != expected:
            parser.error(f"{target}: Rust build metadata changed during regression gate")
    print(f"{'Completed' if args.run else 'Prepared'} {len(plan)} tests; "
          "Android AHB and presentation require separate real-guest gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
