// SPDX-License-Identifier: MIT
// Debt ledger (RUST.md "Existing code"): every workspace lint this crate still violates.
#![expect(
    clippy::print_stderr,
    clippy::print_stdout,
    missing_docs,
    reason = "pre-standard code; cleanup tracked"
)]
use nv_bionic_native::{
    abi,
    rm::{Client, Error},
};

fn run() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 1
        || args
            .first()
            .is_some_and(|a| !["--abi", "--gpu-map", "--copy", "--compute"].contains(&a.as_str()))
    {
        return Err(Error::Contract(
            "usage: nv-native-check [--abi|--gpu-map|--copy|--compute]".into(),
        ));
    }
    if std::env::args().nth(1).as_deref() == Some("--abi") {
        for (name, size) in abi::layouts() {
            println!("{name}\t{size}");
        }
        return Ok(());
    }
    let client = Client::open()?;
    println!(
        "native RM backend; running driver {}; libc={}",
        client.version,
        if cfg!(target_os = "android") {
            "bionic"
        } else {
            "host libc"
        }
    );
    let card = client
        .cards()?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Contract("no NVIDIA card returned by RM".into()))?;
    println!(
        "GPU {:04x}:{:02x}:{:02x}.{} vendor={:04x} device={:04x} RM-ID={:#x} minor={}",
        card.pci.domain,
        card.pci.bus,
        card.pci.slot,
        card.pci.function,
        card.pci.vendor,
        card.pci.device,
        card.gpu_id,
        card.minor
    );
    let device = client.device(card)?;
    let classes = device.classes()?;
    println!(
        "RM device/subdevice created; {} supported classes",
        classes.len()
    );
    let memory = device.allocate_vram(65536)?;
    if std::env::args().nth(1).as_deref() == Some("--gpu-map") {
        let space = nv_bionic_native::gpu::AddressSpace::new(&device)?;
        let gpu = space.map(&memory)?;
        println!(
            "RM-managed GPU VA allocation/map succeeded: {:#x}",
            gpu.address()
        );
    }
    if std::env::args().nth(1).as_deref() == Some("--copy") {
        let words = nv_bionic_native::gpu::copy_check(&device)?;
        println!("GPU copy engine: {words} words copied correctly by hardware");
    }
    if std::env::args().nth(1).as_deref() == Some("--compute") {
        let words = nv_bionic_native::gpu::compute_check(&device)?;
        println!("GPU compute shader: {words} arithmetic results correct");
    }
    let mut mapping = memory.map()?;
    let words = mapping.check_pattern()?;
    println!("VRAM allocation + CPU mmap round trip: {words} words correct");
    let maps = std::fs::read_to_string("/proc/self/maps")
        .map_err(|e| Error::Io("read own loaded libraries", e))?;
    for line in maps
        .lines()
        .filter(|l| l.contains("libc.so") || l.contains("linker64"))
    {
        println!("runtime-map: {line}");
    }
    if maps.lines().any(|l| {
        l.contains("libnvidia-") || l.contains("libEGL_nvidia") || l.contains("libnvglibc")
    }) {
        return Err(Error::Contract(
            "unexpected proprietary userspace driver or glibc shim in process maps".into(),
        ));
    }
    drop(mapping);
    drop(memory);
    drop(device);
    drop(client);
    println!(
        "PASS: requested native RM checks; no NVIDIA userspace blobs/shim loaded. Full Vulkan rendering remains unimplemented."
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("nv-native-check: {error}");
        std::process::exit(1);
    }
}
