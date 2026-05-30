//! Public `init_wgpu_openxr` orchestrator: load the OpenXR loader, negotiate Vulkan through
//! [`XR_KHR_vulkan_enable2`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_KHR_vulkan_enable2),
//! and wrap the result as wgpu plus an OpenXR session.

use super::super::debug_utils::OpenxrDebugUtilsMessenger;
use super::instance::{
    OpenxrInstanceBundle, create_openxr_instance, load_xr_entry,
    probe_head_set_and_vulkan_requirements,
};
use super::session_init::{OpenXrSessionBootstrapDescriptor, openxr_session_state_and_input};
use super::types::{XrBootstrapError, XrWgpuHandles};
use super::vulkan::{
    VulkanOpenXrDeviceCreateDescriptor, create_openxr_vulkan_instance,
    create_vulkan_logical_device_openxr, vk_physical_device_name,
};
use super::wgpu_assembly::{
    WgpuHalOpenXrAssembly, WgpuHalVkChain, build_wgpu_hal_and_queue_family,
    wgpu_from_hal_openxr_chain,
};

/// Builds a Vulkan instance through OpenXR and wraps it as wgpu [`wgpu::Instance`] / [`wgpu::Device`].
///
/// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env overrides,
/// matching [`crate::gpu::instance_flags_for_gpu_init`] and desktop [`crate::gpu::GpuContext::new`].
///
/// `power_preference` is logged for diagnostic context: the OpenXR runtime selects the Vulkan
/// physical device via `xrGetVulkanGraphicsDeviceKHR`, so the renderer cannot override it. When the
/// runtime's choice does not match the user's configured preference, the log line printed below
/// makes the mismatch visible without requiring a Vulkan-layer trace.
pub fn init_wgpu_openxr(
    gpu_validation_layers: bool,
    power_preference: wgpu::PowerPreference,
) -> Result<XrWgpuHandles, XrBootstrapError> {
    // Runtimes often log with printf/stderr; ensure stdio forwarding (idempotent; usually already done in `run`).
    crate::native_stdio::ensure_stdio_forwarded_to_logger();

    let xr_entry = load_xr_entry()
        .map_err(|e| XrBootstrapError::Message(format!("OpenXR loader not found: {e}")))?;

    let OpenxrInstanceBundle {
        xr_instance,
        profile_gates,
        composition_layer_depth_enabled,
    } = create_openxr_instance(xr_entry)?;

    let openxr_debug_messenger = OpenxrDebugUtilsMessenger::try_create(&xr_instance);

    let (xr_system_id, environment_blend_mode, reqs) =
        probe_head_set_and_vulkan_requirements(&xr_instance)?;
    logger::info!(
        "OpenXR system: id={:?} environment_blend_mode={:?} vulkan_min={} vulkan_max={}",
        xr_system_id,
        environment_blend_mode,
        reqs.min_api_version_supported,
        reqs.max_api_version_supported,
    );
    let ash_vk =
        create_openxr_vulkan_instance(&xr_instance, xr_system_id, gpu_validation_layers, &reqs)?;
    let vk_physical_device = ash_vk.vk_physical_device;
    let vk_instance = ash_vk.vk_instance.clone();

    let WgpuHalVkChain {
        wgpu_vk_instance,
        wgpu_exposed,
        vk_device_properties,
        queue_family_index,
    } = build_wgpu_hal_and_queue_family(ash_vk)?;

    {
        let device_name = vk_physical_device_name(&vk_device_properties);
        logger::info!(
            "OpenXR-selected VkPhysicalDevice: {} type={:?} vendor=0x{:04x} device=0x{:04x} \
             (user power_preference={:?}; OpenXR runtime picks the device, not wgpu)",
            device_name,
            wgpu_exposed.info.device_type,
            vk_device_properties.vendor_id,
            vk_device_properties.device_id,
            power_preference,
        );
    };

    let (wgpu_features, enabled_device_extensions, vk_device) =
        create_vulkan_logical_device_openxr(VulkanOpenXrDeviceCreateDescriptor {
            xr_instance: &xr_instance,
            xr_system_id,
            vk_instance: &vk_instance,
            vk_physical_device,
            queue_family_index,
            wgpu_exposed: &wgpu_exposed,
            vk_device_properties: &vk_device_properties,
        })?;

    let (xr_session, openxr_input) =
        openxr_session_state_and_input(OpenXrSessionBootstrapDescriptor {
            xr_instance,
            openxr_debug_messenger,
            environment_blend_mode,
            xr_system_id,
            vk_instance: &vk_instance,
            vk_physical_device,
            vk_device: &vk_device,
            queue_family_index,
            profile_gates,
        })?;

    wgpu_from_hal_openxr_chain(WgpuHalOpenXrAssembly {
        wgpu_vk_instance,
        wgpu_exposed,
        vk_device,
        enabled_device_extensions,
        wgpu_features,
        queue_family_index,
        xr_session,
        xr_system_id,
        openxr_input,
        composition_layer_depth_enabled,
    })
}
