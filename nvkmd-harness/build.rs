// SPDX-License-Identifier: MIT
//! Passes the link arguments of the Mesa build and backend under test
//! (`NVKMD_HARNESS_LINK`: a file with one linker argument per line, written by
//! `mesa/tests/run-nvkmd-harness.py`) to the harness binary.
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-env-changed=NVKMD_HARNESS_LINK");
    let Some(path) = env::var_os("NVKMD_HARNESS_LINK") else {
        return;
    };
    println!("cargo:rerun-if-changed={}", path.display());
    let args = fs::read_to_string(&path).expect("NVKMD_HARNESS_LINK names a readable file");
    for arg in args.lines().filter(|arg| !arg.is_empty()) {
        println!("cargo:rustc-link-arg-bins={arg}");
    }
}
