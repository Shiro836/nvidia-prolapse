// SPDX-License-Identifier: MIT
//! Mesa's runtime over the backend's sync types and loss hooks: the external
//! fence and semaphore property queries an application sees
//! (`vk_fence.c`, `vk_semaphore.c`) and a queue's loss (`vk_queue.c`
//! `_vk_queue_set_lost`), and NVK's memory stream over the backend's timeline
//! type (`nvk_mem_stream.c`). These run Mesa's own code against stand-in
//! physical devices and queues ([`crate::mesa`]); the backend is reached only
//! through the types it lists and the hooks it installs.
use std::ffi::c_void;

use nvkmd_sys::{
    VK_ERROR_DEVICE_LOST, VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT,
    VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT, VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT,
    VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT, VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT,
    VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, VK_SEMAPHORE_TYPE_BINARY,
    VK_SEMAPHORE_TYPE_TIMELINE, VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES,
    VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
    VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, VK_SYNC_FEATURE_TIMELINE,
    VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL, VkExternalFenceProperties, VkExternalSemaphoreProperties,
    VkPhysicalDeviceExternalFenceInfo, VkPhysicalDeviceExternalSemaphoreInfo,
    VkSemaphoreTypeCreateInfo,
};

use nvkmd_sys::{
    VK_ERROR_INITIALIZATION_FAILED, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_SUCCESS, vk_sync_type,
};

use crate::fake::{self, live_allocations, note, reset, state};
use crate::mesa::{self, MemStream, NvkPhysical, PhysicalDevice, Queue, SyncProvider, VkDevice};

/// What Mesa answers for one external handle type: features, compatible
/// types, types exportable from an import.
type Answer = (u32, u32, u32);

fn show((features, compatible, exportable): Answer) -> String {
    format!(
        "features {features:#x}, compatible {compatible:#x}, export from import {exportable:#x}"
    )
}

fn fence(physical: &PhysicalDevice, handle: u32) -> Answer {
    let info = VkPhysicalDeviceExternalFenceInfo {
        sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
        pNext: std::ptr::null(),
        handleType: handle,
    };
    let mut result = VkExternalFenceProperties {
        sType: VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES,
        pNext: std::ptr::null_mut(),
        exportFromImportedHandleTypes: 0,
        compatibleHandleTypes: 0,
        externalFenceFeatures: 0,
    };
    physical.fence_properties(&info, &mut result);
    (
        result.externalFenceFeatures,
        result.compatibleHandleTypes,
        result.exportFromImportedHandleTypes,
    )
}

fn semaphore(physical: &PhysicalDevice, timeline: bool, handle: u32) -> Answer {
    let kind = VkSemaphoreTypeCreateInfo {
        sType: VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        pNext: std::ptr::null(),
        semaphoreType: if timeline {
            VK_SEMAPHORE_TYPE_TIMELINE
        } else {
            VK_SEMAPHORE_TYPE_BINARY
        },
        initialValue: 0,
    };
    let info = VkPhysicalDeviceExternalSemaphoreInfo {
        sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
        pNext: std::ptr::from_ref(&kind).cast::<c_void>(),
        handleType: handle,
    };
    let mut result = VkExternalSemaphoreProperties {
        sType: VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        pNext: std::ptr::null_mut(),
        exportFromImportedHandleTypes: 0,
        compatibleHandleTypes: 0,
        externalSemaphoreFeatures: 0,
    };
    physical.semaphore_properties(&info, &mut result);
    (
        result.externalSemaphoreFeatures,
        result.compatibleHandleTypes,
        result.exportFromImportedHandleTypes,
    )
}

/// The external fence and semaphore properties Mesa derives from the
/// backend's sync types, and the timeline mode those types select.
pub(crate) fn sync_properties() {
    reset();
    let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");

    // vk_device.c get_timeline_mode: one type with VK_SYNC_FEATURE_TIMELINE
    // (Mesa asserts there is at most one) that also waits before signal is
    // VK_DEVICE_TIMELINE_MODE_NATIVE, the mode the submit path relies on.
    let timelines: Vec<u32> = pdev
        .sync_types()
        .iter()
        .map(|kind| kind.features)
        .filter(|features| features & VK_SYNC_FEATURE_TIMELINE != 0)
        .collect();
    note(format!("timeline sync types: {}", timelines.len()));
    assert_eq!(timelines.len(), 1, "exactly one timeline sync type");
    assert!(
        timelines
            .iter()
            .all(|features| features & VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL != 0),
        "the timeline type waits before signal (native timeline mode)"
    );

    let physical = PhysicalDevice::new(&pdev);
    let both_fence =
        VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT;
    for (handle, name) in [
        (VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT, "opaque fd"),
        (VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT, "sync fd"),
    ] {
        let answer = fence(&physical, handle);
        note(format!("fence {name}: {}", show(answer)));
        assert_eq!(
            answer,
            (both_fence, handle, handle),
            "fence {name}: same type only"
        );
    }
    let both_semaphore =
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT;
    for timeline in [false, true] {
        let kind = if timeline { "timeline" } else { "binary" };
        for (handle, name) in [
            (VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, "opaque fd"),
            (VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, "sync fd"),
        ] {
            let answer = semaphore(&physical, timeline, handle);
            note(format!("{kind} semaphore {name}: {}", show(answer)));
            // Sync files have copy semantics and no timeline payload: Mesa offers
            // them for binary semaphores only (vk_semaphore.c).
            let expected = if timeline && handle == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT {
                (0, 0, 0)
            } else {
                (both_semaphore, handle, handle)
            };
            assert_eq!(answer, expected, "{kind} semaphore {name}");
        }
    }
    drop(physical);
    pdev.destroy();
    assert_eq!(live_allocations(), 0, "devices free what they allocated");
}

/// A queue's loss: Mesa counts it on the device and calls the backend's hook
/// while the queue's submit mutex is held (`vk_queue.c` `_vk_queue_set_lost`),
/// once per queue; the hook must neither block nor take that mutex.
pub(crate) fn queue_loss() {
    reset();
    let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");
    let vk = VkDevice::new();
    let dev = pdev.create_dev(Some(&vk)).expect("the KMD device opens");
    state().loss_device = vk.as_ptr().expose_provenance();

    let queues = [Queue::new(&vk), Queue::new(&vk)];
    for (index, queue) in queues.iter().enumerate() {
        let held = queue.lock_submit();
        let result = queue.set_lost(c"offline submit failure");
        note(format!("queue {index}: vk_queue_set_lost = {result}"));
        assert_eq!(result, VK_ERROR_DEVICE_LOST);
        let result = queue.set_lost(c"repeat");
        note(format!("queue {index}: vk_queue_set_lost again = {result}"));
        assert_eq!(result, VK_ERROR_DEVICE_LOST);
        drop(held);
        assert!(queue.is_lost(), "the queue records its loss");
    }
    note(format!(
        "device loss count {}, RM loss publications {}",
        vk.lost_count(),
        state().loss_calls
    ));
    assert_eq!(
        vk.lost_count(),
        2,
        "each queue's loss counts once on the device"
    );
    assert_eq!(
        state().loss_calls,
        2,
        "each queue's loss reaches the RM once"
    );
    assert!(state().rm_lost);

    drop(queues);
    dev.destroy();
    state().loss_device = 0;
    drop(vk);
    pdev.destroy();
    assert_eq!(live_allocations(), 0, "devices free what they allocated");
}

/// NVK's memory stream over the backend's sync types (`nvk_mem_stream.c`
/// `nvk_mem_stream_init`): it refuses a physical device without an operational
/// timeline type before `vk_sync_create` could call through a NULL callback,
/// passes an RM failure to create its timeline through, and leaves no sync
/// object behind on any failure. Before that, the logical device's DRM sync
/// provider (`vk_device_set_drm_fd`): an allocation failure leaves none, and a
/// render node without timeline syncobjs gives one without timeline waits.
pub(crate) fn mem_stream() {
    reset();
    state().timeline_cap = Some(0);
    let pdev = mesa::try_create_pdev().expect("the fake NVIDIA device is accepted");
    let vk = VkDevice::new();
    let dev = pdev.create_dev(Some(&vk)).expect("the KMD device opens");
    vk.attach(&dev);
    state().loss_device = vk.as_ptr().expose_provenance();

    fake::fail_next_calloc();
    let provider = SyncProvider::new(fake::SYNC_PROVIDER_FD);
    fake::stop_failing_calloc();
    note(format!(
        "sync provider, allocation fails: {}",
        if provider.is_some() {
            "created"
        } else {
            "none"
        }
    ));
    assert!(provider.is_none(), "a failed allocation gives no provider");
    let provider = SyncProvider::new(fake::SYNC_PROVIDER_FD).expect("the provider allocates");
    note(format!(
        "sync provider: timeline wait {}",
        provider.has_timeline_wait()
    ));
    assert!(
        !provider.has_timeline_wait(),
        "no timeline waits without the render node's timeline syncobjs"
    );

    let physical = NvkPhysical::new();
    vk.set_runtime(&physical, &provider);
    let timeline = *pdev.sync_types()[0];
    assert!(
        timeline.features & VK_SYNC_FEATURE_TIMELINE != 0,
        "the backend lists its timeline type first (NVK uses sync_types[0])"
    );
    let zeroed = mesa::zeroed_sync_type();
    let binary = vk_sync_type {
        features: timeline.features & !VK_SYNC_FEATURE_TIMELINE,
        ..timeline
    };
    let without_init = vk_sync_type {
        init: None,
        ..timeline
    };
    let without_finish = vk_sync_type {
        finish: None,
        ..timeline
    };
    for (case, kind) in [
        ("zeroed type", &zeroed),
        ("type without timeline", &binary),
        ("timeline type without init", &without_init),
        ("timeline type without finish", &without_finish),
    ] {
        let types = [std::ptr::from_ref(kind), std::ptr::null()];
        physical.set_sync_types(types.as_ptr());
        refused_stream(&vk, case);
    }
    physical.set_sync_types(std::ptr::null());
    refused_stream(&vk, "no sync types");

    physical.set_sync_types(pdev.get().sync_types);
    state().sync.create_status = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    let mut stream = MemStream::new();
    let result = stream.init(&vk);
    state().sync.create_status = 0;
    note(format!(
        "RM timeline creation fails: nvk_mem_stream_init = {result}, sync {}",
        stream.has_sync()
    ));
    assert_eq!(
        result, VK_ERROR_OUT_OF_DEVICE_MEMORY,
        "the RM status passes through"
    );
    assert!(!stream.has_sync(), "a failed init leaves no sync object");

    let mut stream = MemStream::new();
    let result = stream.init(&vk);
    note(format!(
        "operational timeline type: nvk_mem_stream_init = {result}, sync {}",
        stream.has_sync()
    ));
    assert_eq!(result, VK_SUCCESS);
    assert!(stream.has_sync(), "the stream holds its timeline");
    stream.finish(&vk);

    dev.destroy();
    state().loss_device = 0;
    drop(vk);
    drop(physical);
    drop(provider);
    pdev.destroy();
    assert_eq!(live_allocations(), 0, "devices free what they allocated");
}

/// `nvk_mem_stream_init` refuses the stand-in physical device's sync types
/// with `VK_ERROR_INITIALIZATION_FAILED` and a diagnostic, and no sync object.
fn refused_stream(vk: &VkDevice, case: &str) {
    state().diagnostic.clear();
    let mut stream = MemStream::new();
    let result = stream.init(vk);
    let diagnostic = std::mem::take(&mut state().diagnostic);
    note(format!(
        "{case}: nvk_mem_stream_init = {result}, sync {}, diagnostic {:?}",
        stream.has_sync(),
        diagnostic.trim_end()
    ));
    assert_eq!(result, VK_ERROR_INITIALIZATION_FAILED, "{case}");
    assert!(!stream.has_sync(), "{case}: no sync object");
    assert!(!diagnostic.is_empty(), "{case}: Mesa reports why");
}
