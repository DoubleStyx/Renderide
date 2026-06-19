//! wgpu-hal Vulkan adapter / device assembly and final packaging into [`XrWgpuHandles`].

use std::sync::Arc;

use ash::vk;
use openxr as xr;
use wgpu::hal;
use wgpu::hal::api::Vulkan as HalVulkan;
use wgpu::wgt;

use crate::frontend::input::HeadsetMetadata;

use super::super::input::OpenxrInput;
use super::super::session::XrSessionState;
use super::types::{XrBootstrapError, XrWgpuHandles};
use super::version::format_vk_api_version;
use super::vulkan::OpenxrAshVkInstance;

/// `wgpu`-hal Vulkan instance plus exposed adapter, validated physical device properties, graphics queue index.
pub(super) struct WgpuHalVkChain {
    pub(super) wgpu_vk_instance: hal::vulkan::Instance,
    pub(super) wgpu_exposed: hal::ExposedAdapter<HalVulkan>,
    pub(super) vk_device_properties: vk::PhysicalDeviceProperties,
    pub(super) queue_family_index: u32,
}

/// Resolves device properties, the graphics queue family, and the wgpu-hal Vulkan instance/adapter.
pub(super) fn build_wgpu_hal_and_queue_family(
    ash_vk: OpenxrAshVkInstance,
) -> Result<WgpuHalVkChain, XrBootstrapError> {
    let OpenxrAshVkInstance {
        vk_entry,
        vk_instance,
        vk_target_version,
        vk_physical_device,
        extensions,
        flags,
    } = ash_vk;

    // SAFETY: `vk_physical_device` was enumerated from `vk_instance` above, so the call targets
    // a valid physical device owned by that instance.
    let vk_device_properties =
        unsafe { vk_instance.get_physical_device_properties(vk_physical_device) };
    if vk_device_properties.api_version < vk_target_version {
        return Err(XrBootstrapError::Message(format!(
            "Vulkan physical device does not support API version {} (need at least {}).",
            format_vk_api_version(vk_device_properties.api_version),
            format_vk_api_version(vk_target_version)
        )));
    }

    // SAFETY: as above -- valid instance/device pair.
    let queue_family_index =
        unsafe { vk_instance.get_physical_device_queue_family_properties(vk_physical_device) }
            .into_iter()
            .enumerate()
            .find_map(|(i, info)| {
                info.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    .then_some(i as u32)
            })
            .ok_or_else(|| XrBootstrapError::Message("No Vulkan graphics queue family.".into()))?;

    let wgpu_vk_instance = create_wgpu_hal_vulkan_instance(
        vk_entry,
        vk_instance,
        vk_target_version,
        extensions,
        flags,
    )?;

    let wgpu_exposed = wgpu_vk_instance
        .expose_adapter(vk_physical_device)
        .ok_or_else(|| XrBootstrapError::Wgpu("expose_adapter returned None".into()))?;

    Ok(WgpuHalVkChain {
        wgpu_vk_instance,
        wgpu_exposed,
        vk_device_properties,
        queue_family_index,
    })
}

fn create_wgpu_hal_vulkan_instance(
    vk_entry: ash::Entry,
    vk_instance: ash::Instance,
    vk_target_version: u32,
    extensions: Vec<&'static std::ffi::CStr>,
    flags: wgt::InstanceFlags,
) -> Result<hal::vulkan::Instance, XrBootstrapError> {
    // SAFETY: `vk_entry`/`vk_instance` are a live, matched pair created through OpenXR; the
    // extensions list is the same one passed to `vk_instance` creation, preserving wgpu-hal's
    // required invariants for `Instance::from_raw`.
    unsafe {
        hal::vulkan::Instance::from_raw(
            vk_entry,
            vk_instance,
            vk_target_version,
            0,
            None,
            extensions,
            flags,
            wgt::MemoryBudgetThresholds::default(),
            false,
            None,
        )
        .map_err(|e| XrBootstrapError::Vulkan(format!("hal Instance::from_raw: {e}")))
    }
}

/// wgpu-hal + OpenXR session packaging into [`XrWgpuHandles`].
pub(super) struct WgpuHalOpenXrAssembly {
    pub(super) wgpu_vk_instance: hal::vulkan::Instance,
    pub(super) wgpu_exposed: hal::ExposedAdapter<HalVulkan>,
    pub(super) vk_device: ash::Device,
    pub(super) enabled_device_extensions: Vec<&'static std::ffi::CStr>,
    pub(super) wgpu_features: wgt::Features,
    pub(super) queue_family_index: u32,
    pub(super) xr_session: XrSessionState,
    pub(super) xr_system_id: xr::SystemId,
    pub(super) headset_metadata: HeadsetMetadata,
    pub(super) openxr_input: Option<OpenxrInput>,
    pub(super) composition_layer_depth_enabled: bool,
}

/// Wraps Ash device and wgpu-hal adapter in [`wgpu::Instance`] / [`wgpu::Device`] / [`XrWgpuHandles`].
pub(super) fn wgpu_from_hal_openxr_chain(
    assembly: WgpuHalOpenXrAssembly,
) -> Result<XrWgpuHandles, XrBootstrapError> {
    let mut limits = assembly.wgpu_exposed.capabilities.limits.clone();
    limits.max_multiview_view_count = limits.max_multiview_view_count.max(2);
    let memory_hints = wgpu::MemoryHints::default();

    let wgpu_open_device = open_wgpu_hal_device_from_ash(
        &assembly.wgpu_exposed,
        assembly.vk_device,
        assembly.enabled_device_extensions.as_slice(),
        assembly.wgpu_features,
        &limits,
        &memory_hints,
        assembly.queue_family_index,
    )?;

    let wgpu_instance = wgpu_instance_from_hal(assembly.wgpu_vk_instance);
    let wgpu_adapter = wgpu_adapter_from_hal(&wgpu_instance, assembly.wgpu_exposed);

    let device_desc = wgpu::DeviceDescriptor {
        label: Some("renderide-openxr"),
        required_features: assembly.wgpu_features,
        required_limits: limits,
        memory_hints,
        experimental_features: Default::default(),
        trace: Default::default(),
    };

    let (wgpu_device, wgpu_queue) =
        wgpu_device_from_hal(&wgpu_adapter, wgpu_open_device, &device_desc)?;

    Ok(XrWgpuHandles {
        wgpu_instance,
        wgpu_adapter,
        device: Arc::new(wgpu_device),
        queue: Arc::new(wgpu_queue),
        xr_session: assembly.xr_session,
        xr_system_id: assembly.xr_system_id,
        headset_metadata: assembly.headset_metadata,
        openxr_input: assembly.openxr_input,
        composition_layer_depth_enabled: assembly.composition_layer_depth_enabled,
    })
}

fn open_wgpu_hal_device_from_ash(
    exposed: &hal::ExposedAdapter<HalVulkan>,
    vk_device: ash::Device,
    enabled_device_extensions: &[&'static std::ffi::CStr],
    wgpu_features: wgt::Features,
    limits: &wgt::Limits,
    memory_hints: &wgpu::MemoryHints,
    queue_family_index: u32,
) -> Result<hal::OpenDevice<HalVulkan>, XrBootstrapError> {
    // SAFETY: `vk_device` was created through `exposed.adapter` with exactly the
    // features/extensions passed here; the queue family and index match `vkCreateDevice`.
    unsafe {
        exposed
            .adapter
            .device_from_raw(
                vk_device,
                None,
                enabled_device_extensions,
                wgpu_features,
                limits,
                memory_hints,
                queue_family_index,
                0,
            )
            .map_err(|e| XrBootstrapError::Wgpu(format!("device_from_raw: {e}")))
    }
}

fn wgpu_instance_from_hal(hal_instance: hal::vulkan::Instance) -> wgpu::Instance {
    // SAFETY: `hal_instance` was built from the live OpenXR-created Vulkan instance; ownership
    // transfers into the wgpu `Instance`.
    unsafe { wgpu::Instance::from_hal::<HalVulkan>(hal_instance) }
}

fn wgpu_adapter_from_hal(
    instance: &wgpu::Instance,
    exposed: hal::ExposedAdapter<HalVulkan>,
) -> wgpu::Adapter {
    // SAFETY: `exposed` was enumerated from the same Vulkan instance now held by `instance`.
    unsafe { instance.create_adapter_from_hal(exposed) }
}

fn wgpu_device_from_hal(
    adapter: &wgpu::Adapter,
    open_device: hal::OpenDevice<HalVulkan>,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue), XrBootstrapError> {
    // SAFETY: `open_device` was opened from `adapter`'s underlying hal adapter, and `desc` uses
    // the same features/limits passed to `device_from_raw`.
    unsafe { adapter.create_device_from_hal(open_device, desc) }
        .map_err(|e| XrBootstrapError::Wgpu(format!("create_device_from_hal: {e}")))
}
