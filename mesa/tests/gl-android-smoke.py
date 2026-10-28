#!/usr/bin/env python3
"""Host smoke test of the Android nvrm build: gl-g0b-triangle --es3 (surfaceless ES 3.0, clear +
triangle + indexed quad + readback) through libEGL_nvrm/libGLESv2_nvrm, run by the bionic linker
on the host GPU like tests/vulkan/run.py. Flat scene must be byte-identical to
llvmpipe (host Mesa via glvnd); the gradient scene is reported (<=1 LSB interpolation differences
are expected and do not fail). Missing artifacts fail loudly.

Capability gate (G2): gl-dmabuf.c (built without Vulkan) prints the EGL extension string and must
pass on each ABI: EGL_EXT_image_dma_buf_import(_modifiers), EGL_MESA_image_dma_buf_export,
EGL_ANDROID_native_fence_sync and EGL_KHR_wait_sync present, a texture exported as a dma-buf and
re-imported reads/samples/writes back exactly, and a native fence fd from eglDupNativeFenceFDANDROID
signals and re-imports for eglWaitSyncKHR. The Rust backend is NVK's (profiles.rust_library), the
same file nvrm-build.json records.

RenderEngine shader gate: gl-skia-shaders.c (SurfaceFlinger's Skia GL shader features: samplerExternalOES
in ESSL1/ESSL3 incl. textureSize/texelFetch, sk_FragCoord flip, integer dither, derivatives + discard,
mip bias, LinearEffect transfer functions, blur loop, mediump rrect) must print SKIA SHADERS PASS on each
ABI against llvmpipe references (one case against its analytic result: llvmpipe is wrong there).

YUV gate: gl-yuv.c (NV12/P010 dma-bufs from our NVK, gralloc 0004 layout, 854x480 / 1920x1080 with 1088
rows / 640x362, raw memory and VK multi-planar images, BT601/709/2020 limited/full hints) imported as
EGLImages, sampled through samplerExternalOES and compared with a CPU reference (<= 2 LSB), plus the
refusals; must print GL YUV PASS on each ABI. NVK's ICD of the same ABI (build/nvk-<target>) is
dlopen'ed directly, no Vulkan loader.

  [NVRM_GL_DIR_SUFFIX=-x] mesa/tests/gl-android-smoke.py [--target android|android32|both] OUT
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SFX = os.environ.get("NVRM_GL_DIR_SUFFIX", "")
NDK = ROOT / "build/android-work/ndk/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64"
TRIPLES = {"android": "x86_64-linux-android", "android32": "i686-linux-android"}
sys.path.insert(0, str(ROOT / "tools"))
import native_rust_profile as profiles  # noqa: E402


def must(path):
    if not Path(path).exists():
        raise SystemExit(f"missing {path}")
    return Path(path)


def launch(target, argv):
    build = ROOT / f"build/nvrm-{target}{SFX}"
    # nvrm links NVK's Rust backend (build.py build_gl_android): run with that file.
    rust = profiles.rust_library(ROOT, target, "release")
    paths = [must(rust).parent, build / "src/egl", build / "src/mesa/glapi/es2api",
             build / "src/gallium/targets/dri"]
    if target == "android":
        # the image's own 64-bit bionic (mesa/prepare-runtime64.py)
        runtime = ROOT / "build/bionic-runtime64"
        paths += [runtime / "lib64", ROOT / "build/nvk-android/runtime",
                  ROOT / "build/android-work/deps/image-rootfs/system/lib64"]
        command = [must(runtime / "linker64"), *argv]
    else:
        paths += [ROOT / "build/nvk-android32/runtime"]
        # i686 bionic's pthread TID field cannot hold this host's PIDs: own PID namespace.
        command = ["unshare", "--user", "--map-root-user", "--pid", "--fork",
                   must(ROOT / "build/nvk-android32/runtime/linker"), *argv]
    return {"LD_LIBRARY_PATH": ":".join(map(str, paths))}, list(map(str, command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("android", "android32", "both"), default="both")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = must(HERE / "gl-g0b-triangle.c")
    host = out / "gl-g0b-triangle-sys"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-lEGL", "-lGLESv2", "-o", host],
                   check=True)
    ref_env = {**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1",
               "__EGL_VENDOR_LIBRARY_FILENAMES": str(must("/usr/share/glvnd/egl_vendor.d/50_mesa.json"))}
    for mode, name in ((["--flat"], "flat"), ([], "grad")):
        subprocess.run([host, "--es3", *mode, "llvmpipe", out / f"ref-{name}.rgba"], env=ref_env, check=True)
    skia = must(HERE / "gl-skia-shaders.c")
    skia_host = out / "gl-skia-shaders-sys"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", skia, "-lEGL", "-lGLESv2", "-lm",
                    "-o", skia_host], check=True)
    (out / "skia-ref").mkdir(exist_ok=True)
    # Reference run only writes images (its analytic case is not judged here).
    subprocess.run([skia_host, "llvmpipe", out / "skia-ref"], env=ref_env, check=True,
                   stdout=subprocess.DEVNULL)
    failed = False
    for target in (("android", "android32") if args.target == "both" else (args.target,)):
        build = ROOT / f"build/nvrm-{target}{SFX}"
        exe = out / f"gl-g0b-triangle-{target}"
        subprocess.run([must(NDK / f"bin/{TRIPLES[target]}33-clang"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        source, f"-L{must(build / 'src/egl/libEGL_nvrm.so').parent}",
                        f"-L{must(build / 'src/mesa/glapi/es2api/libGLESv2_nvrm.so').parent}",
                        "-lEGL_nvrm", "-lGLESv2_nvrm", "-o", exe], check=True)
        for mode, name in ((["--flat"], "flat"), ([], "grad")):
            env, argv = launch(target, [exe, "--es3", *mode, "nvrm", out / f"out-{name}-{target}.rgba",
                                        out / f"ref-{name}.rgba"])
            result = subprocess.run(argv, env=env, capture_output=True, text=True)
            log = "".join(l for l in (result.stdout + result.stderr).splitlines(True)
                          if "linker configuration" not in l)
            (out / f"{target}-{name}.log").write_text(log)
            print(f"[{target}/{name}] exit={result.returncode}\n{log}", flush=True)
            if name == "flat" and result.returncode:
                failed = True
            if name == "grad":
                import re
                maxd = re.search(r"max channel difference (\d+)", log)
                if ("OpenGL ES 3." not in log or "coverage differences 0," not in log
                        or not maxd or int(maxd.group(1)) > 1):
                    failed = True
        # G2 capability gate: dma-buf import/export + native fence round trip.
        cap = out / f"gl-dmabuf-{target}"
        subprocess.run([must(NDK / f"bin/{TRIPLES[target]}33-clang"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-DNVRM_NO_VULKAN", f"-I{must(ROOT / f'build/mesa-gl-source{SFX}/include')}",
                        f"-I{ROOT / f'build/mesa-gl-source{SFX}/include/drm-uapi'}",
                        must(HERE / "gl-dmabuf.c"), f"-L{build / 'src/egl'}", f"-L{build / 'src/mesa/glapi/es2api'}",
                        "-lEGL_nvrm", "-lGLESv2_nvrm", "-o", cap], check=True)
        env, argv = launch(target, [cap, "nvrm"])
        result = subprocess.run(argv, env=env, capture_output=True, text=True)
        log = "".join(l for l in (result.stdout + result.stderr).splitlines(True)
                      if "linker configuration" not in l)
        (out / f"{target}-capability.log").write_text(log)
        print(f"[{target}/capability] exit={result.returncode}\n{log}", flush=True)
        if result.returncode or "GL DMABUF PASS" not in log:
            failed = True
        # RenderEngine (Skia) shader features.
        sk = out / f"gl-skia-shaders-{target}"
        subprocess.run([must(NDK / f"bin/{TRIPLES[target]}33-clang"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        skia, f"-L{build / 'src/egl'}", f"-L{build / 'src/mesa/glapi/es2api'}",
                        "-lEGL_nvrm", "-lGLESv2_nvrm", "-lm", "-o", sk], check=True)
        (out / f"skia-{target}").mkdir(exist_ok=True)
        env, argv = launch(target, [sk, "nvrm", out / f"skia-{target}", out / "skia-ref"])
        result = subprocess.run(argv, env=env, capture_output=True, text=True)
        log = "".join(l for l in (result.stdout + result.stderr).splitlines(True)
                      if "linker configuration" not in l)
        (out / f"{target}-skia.log").write_text(log)
        print(f"[{target}/skia-shaders] exit={result.returncode}\n{log}", flush=True)
        if result.returncode or "SKIA SHADERS PASS" not in log:
            failed = True
        # YUV EGLImage import + samplerExternalOES (video layers in SurfaceFlinger).
        yuv = out / f"gl-yuv-{target}"
        subprocess.run([must(NDK / f"bin/{TRIPLES[target]}33-clang"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        f"-I{must(ROOT / f'build/mesa-gl-source{SFX}/include')}",
                        f"-I{ROOT / f'build/mesa-gl-source{SFX}/include/drm-uapi'}",
                        must(HERE / "gl-yuv.c"), f"-L{build / 'src/egl'}", f"-L{build / 'src/mesa/glapi/es2api'}",
                        "-lEGL_nvrm", "-lGLESv2_nvrm", "-lm", "-ldl", "-o", yuv], check=True)
        env, argv = launch(target, [yuv, "nvrm"])
        env["NVRM_YUV_VK_ICD"] = str(must(ROOT / f"build/nvk-{target}/src/nouveau/vulkan/libvulkan_nouveau.so"))
        result = subprocess.run(argv, env=env, capture_output=True, text=True)
        log = "".join(l for l in (result.stdout + result.stderr).splitlines(True)
                      if "linker configuration" not in l)
        (out / f"{target}-yuv.log").write_text(log)
        print(f"[{target}/yuv] exit={result.returncode}\n{log}", flush=True)
        if result.returncode or "GL YUV PASS" not in log:
            failed = True
    print("ANDROID SMOKE FAIL" if failed else "ANDROID SMOKE PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
