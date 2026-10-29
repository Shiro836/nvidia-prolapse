// SPDX-License-Identifier: MIT
//! Offline harness of NVK's nvkmd backend (`nvkmd-rm`, linked in with Mesa's
//! objects). It drives the backend through Mesa's public nvkmd interface
//! against a fake RM, render node and allocator, checks each scenario's
//! outcome, and writes every call the backend made to the fakes to stdout.
//! `mesa/tests/run-nvkmd-harness.py` builds it and compares the log byte for
//! byte with the golden one.
//!
//! Usage: `nvkmd-harness [suite...]` (default: every suite). No device file
//! is opened and no GPU is touched.
use std::io::Write as _;
use std::process::ExitCode;

mod ctx;
mod fake;
mod fake_ctx;
mod fake_sync;
mod memory;
mod mesa;
mod runtime;
mod suites;
mod sync;

fn main() -> ExitCode {
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    for name in &wanted {
        if !suites::SUITES.iter().any(|(suite, _)| suite == name) {
            let names: Vec<&str> = suites::SUITES.iter().map(|(suite, _)| *suite).collect();
            let mut stderr = std::io::stderr().lock();
            let report = writeln!(
                stderr,
                "nvkmd-harness: unknown suite {name}; suites: {}",
                names.join(" ")
            );
            return if report.is_ok() {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            };
        }
    }
    fake::init();
    let mut stdout = std::io::stdout().lock();
    for (name, run) in suites::SUITES {
        if !wanted.is_empty() && !wanted.iter().any(|wanted| wanted == name) {
            continue;
        }
        fake::note(format!("== {name}"));
        run();
        fake::note(format!("PASS {name}"));
        for line in fake::take_log() {
            if writeln!(stdout, "{line}").is_err() {
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
