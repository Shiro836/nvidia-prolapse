#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unit tests of mesa/build.py's nvrm (--gl) source preparation, without git, Mesa or a GPU:
python3 -m pytest -q mesa/test_build_gl.py"""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("nvrm_build", HERE / "build.py")
build = importlib.util.module_from_spec(SPEC)
sys.modules["nvrm_build"] = build
SPEC.loader.exec_module(build)


def prepare(monkeypatch, tmp_path):
    """Runs prepare_gl_source with every external effect replaced; returns the commands it ran."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("bindgen", "cbindgen"):
        (tools / tool).touch()
    source = tmp_path / "mesa-gl-source"
    (source / "subprojects").mkdir(parents=True)
    commands = []
    monkeypatch.setattr(build, "TOOLS", tools)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "GL_SOURCE", source)
    monkeypatch.setattr(build, "run", lambda command, **kwargs: commands.append([str(x) for x in command]))
    monkeypatch.setattr(build.subprocess, "check_output", lambda *args, **kwargs: build.PIN + "\n")
    monkeypatch.setattr(build, "host_dependencies", lambda download: None)
    monkeypatch.setattr(build, "gl_env", dict)

    class Args:
        skip_prepare = False
        download_deps = False

    build.prepare_gl_source(Args())
    return commands


def test_every_applied_patch_is_a_stamp_input(monkeypatch, tmp_path):
    """--skip-prepare trusts the stamp: a patch the preparation applies but the stamp does not hash
    could change without the GL worktree being prepared again."""
    applied = [Path(c[-1]) for c in prepare(monkeypatch, tmp_path) if c[3:4] == ["apply"]]
    assert applied, "the preparation applied no patch"
    inputs = set(build.GL_PREP_INPUTS)
    missing = [p.name for p in applied if p not in inputs]
    assert not missing, f"applied but not in the preparation stamp: {missing}"


def test_stamp_written_and_checked(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    build.check_prepared(tmp_path / "mesa-gl-source", build.GL_PREP_INPUTS, "GL", "--gl ")
