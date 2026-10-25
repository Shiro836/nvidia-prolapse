"""Static libelf for the Android cross builds of Mesa >= 26 (NVK requires it:
meson.build "NVK requires libelf"; src/nouveau/cubin parses CUDA cubins for
VK_NVX_binary_import).

Same source and recipe as upstream Mesa's Android CI
(.gitlab-ci/container/debian/android_build.sh): libelf-0.8.13, md5
4136d7b4c04df68b686570afa26988ac, autoreconf (enables __LIBELF64), configure
--host=x86_64-linux-gnu --disable-nls --disable-shared with the NDK clang of the
target ABI. Pinned here by sha256 as well. Output:
build/libelf-android/<triple>/{lib/libelf.a,include,lib/pkgconfig/libelf.pc}.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fetch import fetch  # noqa: E402  (mirrors, retries, verified before kept)

VERSION = "libelf-0.8.13"
# the URL Mesa's Android CI uses first, then the same tarball (same md5 + sha256) from the BLFS archive
URLS = (f"https://fossies.org/linux/misc/old/{VERSION}.tar.gz",
        f"https://ftp.osuosl.org/pub/blfs/conglomeration/libelf/{VERSION}.tar.gz")
MD5 = "4136d7b4c04df68b686570afa26988ac"      # upstream Mesa CI
SHA256 = "591a9b4ec81c1f2042a97aa60564e0cb79d041c52faa7416acb38bc95bd2c76d"


def _run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def build(root, ndk_bin, triple, download):
    """Build (once) and return the pkg-config file of libelf for `triple`
    (x86_64-linux-android or i686-linux-android), API 33. Serialized per triple with a lock file: NAK and NVK
    builds of one ABI may ask for it at the same time (packaging/native/build-native.sh)."""
    import fcntl
    base = Path(root) / "build/libelf-android"
    base.mkdir(parents=True, exist_ok=True)
    with open(base / f".lock-{triple}", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build(base, ndk_bin, triple, download)


def _build(base, ndk_bin, triple, download):
    prefix = base / triple
    pc = prefix / "lib/pkgconfig/libelf.pc"
    archive = base / "downloads" / f"{VERSION}.tar.gz"
    if not archive.exists():
        if not download:
            raise SystemExit(f"Missing {archive}; use --download-deps (Android NVK needs libelf, Mesa >= 26)")
        fetch(URLS, archive, SHA256, MD5)
    data = archive.read_bytes()
    if hashlib.md5(data).hexdigest() != MD5 or hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit(f"Checksum mismatch: {archive}")
    stamp = prefix / f".built-{SHA256[:16]}"
    if stamp.exists() and pc.exists() and (prefix / "lib/libelf.a").exists():
        return pc
    src = base / f"src-{triple}"
    shutil.rmtree(src, ignore_errors=True)
    shutil.rmtree(prefix, ignore_errors=True)
    src.mkdir(parents=True)
    _run(["tar", "-xf", archive, "-C", src, "--strip-components=1"])
    _run(["autoreconf"], cwd=src, stdout=subprocess.DEVNULL)
    env = dict(os.environ)
    env.update(CC=str(Path(ndk_bin) / f"{triple}33-clang"),
               CXX=str(Path(ndk_bin) / f"{triple}33-clang++"),
               AR=str(Path(ndk_bin) / "llvm-ar"), RANLIB=str(Path(ndk_bin) / "llvm-ranlib"))
    env.pop("PKG_CONFIG_PATH", None)
    _run(["./configure", "--host=x86_64-linux-gnu", "--disable-nls", "--disable-shared",
          f"--prefix={prefix}"], cwd=src, env=env, stdout=subprocess.DEVNULL)
    _run(["make", "-j8", "install"], cwd=src, env=env, stdout=subprocess.DEVNULL)
    sys_elf = (prefix / "include/libelf/sys_elf.h").read_text()
    if "#define __LIBELF64 1" not in sys_elf:
        raise SystemExit(f"{VERSION} for {triple} was configured without __LIBELF64")
    stamp.write_text(SHA256 + "\n")
    return pc


def install_pc(pc, pkg_dir):
    """Expose libelf.pc in a cross build's pkg-config directory."""
    Path(pkg_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(pc, Path(pkg_dir) / "libelf.pc")
