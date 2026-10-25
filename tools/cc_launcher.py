"""Compiler launcher (ccache) for the native stack's Meson builds.

NATIVE_CC_LAUNCHER (packaging/native/build-native.sh sets "ccache" inside its build container, cache in
build/ccache) is put in front of every C/C++ compiler: host builds through CC/CXX, Android builds through the
cross file's [binaries]. Unset or empty: the compilers run directly, as before.

Meson keeps the compiler command it detected at the first `meson setup`; `--reconfigure` does not detect it again
(checked with meson 1.12.1). setup_command() therefore drops the old configuration of a build dir whose launcher
differs from the current one, so that setup runs fresh in place: the build dir's other files stay (build/nvk-android32
also holds the 32-bit runtime), and ninja rebuilds whatever the new command lines change.

Rust is not wrapped: Meson ignores a compiler cache in its Rust entry, and the cargo builds keep their persistent
target dirs in build/.
"""
import os
from pathlib import Path

MARK = "meson-private/native-cc-launcher"


def launcher():
    return os.environ.get("NATIVE_CC_LAUNCHER", "").split()


def host_env(env):
    """CC/CXX for a native (host) Meson build."""
    env.update(CC=" ".join(launcher() + ["cc"]), CXX=" ".join(launcher() + ["c++"]))
    return env


def cross_binary(path):
    """A cross file [binaries] value for compiler PATH, e.g. ['ccache', '/ndk/bin/clang']."""
    return "[" + ", ".join(f"'{part}'" for part in launcher() + [str(path)]) + "]"


def setup_command(build, source):
    """`meson setup BUILD SOURCE`, with --reconfigure when BUILD is configured with the current launcher."""
    build = Path(build)
    core = build / "meson-private/coredata.dat"
    mark = build / MARK
    if core.exists() and (mark.read_text() if mark.exists() else "") != " ".join(launcher()):
        core.unlink()
    return ["meson", "setup", build, source] + (["--reconfigure"] if core.exists() else [])


def record(build):
    """After a successful setup: remember which launcher BUILD's configuration uses."""
    (Path(build) / MARK).write_text(" ".join(launcher()))
