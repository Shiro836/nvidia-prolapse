// SPDX-License-Identifier: MIT
//! The scenarios. Each drives the backend only through Mesa's nvkmd interface
//! ([`crate::mesa`]) against the fake world ([`crate::fake`]), checks what
//! Vulkan prescribes for the outcome, and leaves the call log for the
//! cross-backend comparison.
use nvkmd_sys::{
    NVKMD_SYS_VK_DEVICE_CHECK_STATUS, NVKMD_SYS_VK_DEVICE_NOTIFY_LOST,
    NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA, VK_ERROR_DEVICE_LOST, VK_ERROR_INCOMPATIBLE_DRIVER,
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_OUT_OF_HOST_MEMORY, VK_SYNC_FEATURE_BINARY,
    VK_SYNC_FEATURE_TIMELINE, VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL, VkResult, vk_sync_type,
};

use crate::fake::{self, Version, live_allocations, note, reset, state};
use crate::mesa::{self, Pdev, VkDevice};

/// A suite: its name on the command line and in the log, and its body.
pub(crate) type Suite = (&'static str, fn());

/// Every suite, in run order.
pub(crate) const SUITES: &[Suite] = &[
    ("pdev", pdev),
    ("info", info),
    ("loss", loss),
    ("memory-api", crate::memory::api),
    ("memory-cleanup", crate::memory::cleanup),
    ("sync-api", crate::sync::api),
    ("wait-loss", crate::sync::wait_loss),
    ("submit-fusion", crate::ctx::submit_fusion),
    ("sync-properties", crate::runtime::sync_properties),
    ("queue-loss", crate::runtime::queue_loss),
    ("mem-stream", crate::runtime::mem_stream),
];

fn refused(case: &str, expected: VkResult) {
    let result = mesa::try_create_pdev().map(Pdev::destroy);
    note(format!("{case}: probe = {result:?}"));
    assert_eq!(result, Err(expected), "{case}");
    assert_eq!(
        live_allocations(),
        0,
        "{case}: a refused probe frees what it allocated"
    );
    let state = state();
    assert_eq!(
        state.closes,
        state.opens - state.failed_opens,
        "{case}: the render node is closed"
    );
}

/// `vk_sync_type` as a log line: size, features and which callbacks are set.
fn describe_sync_type(index: usize, sync: &vk_sync_type) -> String {
    let slots = [
        ("init", sync.init.is_some()),
        ("finish", sync.finish.is_some()),
        ("signal", sync.signal.is_some()),
        ("signal_many", sync.signal_many.is_some()),
        ("get_value", sync.get_value.is_some()),
        ("reset", sync.reset.is_some()),
        ("move", sync.move_.is_some()),
        ("reserve_signal", sync.reserve_signal.is_some()),
        ("wait", sync.wait.is_some()),
        ("wait_many", sync.wait_many.is_some()),
        ("import_opaque_fd", sync.import_opaque_fd.is_some()),
        ("export_opaque_fd", sync.export_opaque_fd.is_some()),
        ("import_win32_handle", sync.import_win32_handle.is_some()),
        ("export_win32_handle", sync.export_win32_handle.is_some()),
        (
            "set_win32_export_params",
            sync.set_win32_export_params.is_some(),
        ),
        ("import_sync_file", sync.import_sync_file.is_some()),
        ("export_sync_file", sync.export_sync_file.is_some()),
    ];
    let set: Vec<&str> = slots
        .iter()
        .filter(|(_, set)| *set)
        .map(|(name, _)| *name)
        .collect();
    format!(
        "sync_type[{index}] size={} features={:#x} callbacks={}",
        sync.size,
        sync.features,
        set.join(",")
    )
}

/// Physical-device probing and KMD device creation: every refusal and failure
/// path of `nvkmd_try_create_pdev_for_drm` and `create_dev`, then an accepted
/// device whose filled-in fields and sync types go to the log.
fn pdev() {
    probe_refusals();
    reset();
    let pdev = accepted_pdev();
    kmd_devices(&pdev);
    pdev.destroy();
    assert_eq!(live_allocations(), 0, "devices free what they allocated");
    let state = state();
    assert_eq!(
        state.opens - state.failed_opens,
        state.closes,
        "every render node open is closed"
    );
    assert_eq!(state.rm_opens, state.rm_closes, "every RM device is closed");
}

/// Every way `nvkmd_try_create_pdev_for_drm` refuses or fails.
fn probe_refusals() {
    reset();
    state().open_error = true;
    refused("render node open fails", VK_ERROR_INCOMPATIBLE_DRIVER);
    for version in [Version::Missing, Version::Nouveau, Version::Unnamed] {
        reset();
        state().version = version;
        refused(
            &format!("DRM driver {version:?}"),
            VK_ERROR_INCOMPATIBLE_DRIVER,
        );
    }
    reset();
    fake::fail_next_calloc();
    refused("device allocation fails", VK_ERROR_OUT_OF_HOST_MEMORY);
    reset();
    state().stat_error = true;
    refused("render node stat fails", VK_ERROR_INITIALIZATION_FAILED);
    assert!(
        state().diagnostic.contains("stat failed"),
        "the stat failure is named"
    );
    assert_eq!(
        state().rm_opens,
        0,
        "no RM device without the render node's identity"
    );
    reset();
    state().rm_error = true;
    refused("RM open fails", VK_ERROR_INITIALIZATION_FAILED);
    assert!(
        state().diagnostic.contains("query device"),
        "the RM failure is named"
    );
    assert_eq!(state().rm_closes, 0, "no RM device to close");
    reset();
    state().rm_error = true;
    state().rm_partial = true;
    refused(
        "RM open fails after creating its device",
        VK_ERROR_INITIALIZATION_FAILED,
    );
    assert_eq!(state().rm_closes, 1, "the half-made RM device is closed");
}

/// An accepted device: its fields and sync types go to the log.
fn accepted_pdev() -> Pdev {
    let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");
    note(fake::describe_pdev(pdev.get()));
    {
        let state = state();
        assert_eq!(
            (state.rm_opens, state.rm_closes),
            (1, 1),
            "probing closes its RM device"
        );
    }
    let types = pdev.sync_types();
    assert_eq!(
        types.len(),
        2,
        "a timeline/opaque-fd type and a binary sync_file type"
    );
    let features = types[0].features;
    assert_eq!(
        features
            & (VK_SYNC_FEATURE_BINARY
                | VK_SYNC_FEATURE_TIMELINE
                | VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL),
        VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE | VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL,
        "the first type is a wait-before-signal timeline"
    );
    assert_eq!(
        types[1].features & VK_SYNC_FEATURE_TIMELINE,
        0,
        "the sync_file type is binary"
    );
    for (index, sync) in types.iter().enumerate() {
        note(describe_sync_type(index, sync));
    }
    pdev
}

/// `create_dev` during probing (no logical device): its failure paths, then
/// a device that opens and is destroyed.
fn kmd_devices(pdev: &Pdev) {
    fake::fail_next_calloc();
    let result = pdev.create_dev(None).map(mesa::Dev::destroy);
    fake::stop_failing_calloc();
    note(format!("device allocation fails: create_dev = {result:?}"));
    assert_eq!(result, Err(VK_ERROR_OUT_OF_HOST_MEMORY));
    state().open_error = true;
    let result = pdev.create_dev(None).map(mesa::Dev::destroy);
    note(format!("render node open fails: create_dev = {result:?}"));
    assert_eq!(result, Err(VK_ERROR_INITIALIZATION_FAILED));
    state().open_error = false;
    state().rm_error = true;
    state().rm_partial = true;
    let result = pdev.create_dev(None).map(mesa::Dev::destroy);
    note(format!(
        "RM open fails after creating its device: create_dev = {result:?}"
    ));
    assert_eq!(result, Err(VK_ERROR_INITIALIZATION_FAILED));
    state().rm_error = false;
    state().rm_partial = false;
    let dev = pdev.create_dev(None).expect("the KMD device opens");
    note("create_dev = Ok");
    dev.destroy();
}

/// A change to what the fake RM reports.
type Report = fn(&mut nv_rm_abi::NvRmInfo);

/// RM values wider than the `nv_device_info` field they are reported in:
/// the device is refused as one this driver cannot describe (NVK's answer to
/// devices it cannot drive, `nvk_physical_device.c`), never reported with a
/// truncated value.
fn info() {
    let cases: [(&str, Report); 3] = [
        ("3D class wider than 16 bits", |info| {
            info.cls_eng3d = 0x1_c797;
        }),
        ("PCI domain wider than 16 bits", |info| {
            info.pci_domain = 0x1_0000;
        }),
        ("SM version wider than 8 bits", |info| {
            info.sm = 0x100;
        }),
    ];
    for (case, report) in cases {
        reset();
        state().report = report;
        refused(case, VK_ERROR_INCOMPATIBLE_DRIVER);
        assert!(
            state().diagnostic.contains("not representable"),
            "{case}: the refusal is named"
        );
        let state = state();
        assert_eq!(
            state.rm_opens, state.rm_closes,
            "{case}: the RM device is closed"
        );
    }
}

/// Loss publication between Mesa and the RM: a logical device's loss reaches
/// `nv_rm_mark_lost` once, after Mesa counted it; destroying the KMD device
/// clears the hooks before the RM device closes, so later loss stays in Mesa.
fn loss() {
    reset();
    let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");
    let vk = VkDevice::new();
    let dev = pdev.create_dev(Some(&vk)).expect("the KMD device opens");
    for (member, what) in [
        (NVKMD_SYS_VK_DEVICE_NOTIFY_LOST, "notify_lost"),
        (NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA, "notify_lost_data"),
        (NVKMD_SYS_VK_DEVICE_CHECK_STATUS, "check_status"),
    ] {
        assert!(vk.has(member), "a logical device gets {what}");
    }
    state().loss_device = vk.as_ptr().expose_provenance();

    let result = vk.set_lost(c"offline device failure");
    note(format!("vk_device_set_lost = {result}"));
    assert_eq!(result, VK_ERROR_DEVICE_LOST);
    assert_eq!(state().loss_calls, 1, "the loss reaches the RM");
    assert!(state().rm_lost);
    let result = vk.set_lost(c"repeat");
    note(format!("vk_device_set_lost again = {result}"));
    assert_eq!(result, VK_ERROR_DEVICE_LOST);
    assert_eq!(
        state().loss_calls,
        1,
        "a repeated loss is not published again"
    );

    dev.destroy();
    vk.clear_lost();
    let result = vk.set_lost(c"after backend detach");
    note(format!("vk_device_set_lost after destroy = {result}"));
    assert_eq!(result, VK_ERROR_DEVICE_LOST);
    assert!(vk.is_lost());
    assert_eq!(state().loss_calls, 1, "a destroyed backend is not notified");
    state().loss_device = 0;
    drop(vk);
    pdev.destroy();
    assert_eq!(live_allocations(), 0, "devices free what they allocated");
    let state = state();
    assert_eq!(
        state.opens, state.closes,
        "every render node open is closed"
    );
    assert_eq!(state.rm_opens, state.rm_closes, "every RM device is closed");
}
