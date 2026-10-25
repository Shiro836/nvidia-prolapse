//! Explicit, bounded diagnostic only. Never called by the production backend.
use nv_bionic_native::{perf_diagnostic::PerformanceVote, rm};
use std::io::Write;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: nv-native-perf-boost PCI_BDF HOLD_MILLISECONDS(1..10000)".into());
    }
    let hold: u64 = args[2].parse()?;
    if !(1..=10000).contains(&hold) {
        return Err("hold duration must be 1..10000 milliseconds".into());
    }
    let client = rm::Client::open()?;
    let selected: Vec<_> = client
        .cards()?
        .into_iter()
        .filter(|c| {
            format!(
                "{:04x}:{:02x}:{:02x}.{:x}",
                c.pci.domain, c.pci.bus, c.pci.slot, c.pci.function
            )
            .eq_ignore_ascii_case(&args[1])
        })
        .collect();
    if selected.len() != 1 {
        return Err("requested PCI BDF must match exactly one NVIDIA RM card".into());
    }
    let device = client.device(selected[0])?;
    let mut vote = PerformanceVote::acquire(device)?;
    println!(
        "READY pci={} version={} flags=2 duration_seconds=10 hold_ms={hold}",
        args[1], client.version
    );
    std::io::stdout().flush()?;
    std::thread::sleep(std::time::Duration::from_millis(hold));
    vote.clear()?;
    println!("CLEARED pci={} flags=0 duration_seconds=0", args[1]);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("nv-native-perf-boost: {e}");
        std::process::exit(1);
    }
}
