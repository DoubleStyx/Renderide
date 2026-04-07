//! Create a Vulkan instance and device via OpenXR `KHR_vulkan_enable2`, then wrap with wgpu.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use ash::vk::{self, Handle};
use openxr as xr;
use thiserror::Error;
use wgpu::hal;
use wgpu::hal::api::Vulkan as HalVulkan;

use wgpu::wgt;

use super::input::OpenxrInput;

/// WGPU + OpenXR objects produced by [`init_wgpu_openxr`].
pub struct XrWgpuHandles {
    /// WGPU instance (Vulkan backend).
    pub wgpu_instance: wgpu::Instance,
    /// Adapter for the XR-selected physical device.
    pub wgpu_adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<Mutex<wgpu::Queue>>,
    /// OpenXR session, frame stream, and reference space.
    pub xr_session: super::session::XrSessionState,
    /// Active system (HMD) id.
    pub xr_system_id: xr::SystemId,
    /// Controller actions and spaces; `None` if action creation or Touch bindings failed.
    pub openxr_input: Option<OpenxrInput>,
}

/// Bootstrap failure (missing runtime, Vulkan, or extension).
#[derive(Debug, Error)]
pub enum XrBootstrapError {
    /// User-visible message for logs.
    #[error("{0}")]
    Message(String),
    /// OpenXR API error.
    #[error("OpenXR: {0}")]
    OpenXr(#[from] xr::sys::Result),
    /// Vulkan / ash error.
    #[error("Vulkan: {0}")]
    Vulkan(String),
    /// WGPU could not use the XR device.
    #[error("wgpu: {0}")]
    Wgpu(String),
}

impl From<vk::Result> for XrBootstrapError {
    fn from(e: vk::Result) -> Self {
        Self::Vulkan(format!("{e:?}"))
    }
}

/// Builds a Vulkan instance through OpenXR and wraps it as wgpu [`wgpu::Instance`] / [`wgpu::Device`].
pub fn init_wgpu_openxr() -> Result<XrWgpuHandles, XrBootstrapError> {
    let xr_entry = unsafe { xr::Entry::load() }
        .map_err(|e| XrBootstrapError::Message(format!("OpenXR loader not found: {e}")))?;
    let available_extensions = xr_entry
        .enumerate_extensions()
        .map_err(|e| XrBootstrapError::Message(format!("enumerate_extensions: {e}")))?;
    if !available_extensions.khr_vulkan_enable2 {
        return Err(XrBootstrapError::Message(
            "OpenXR runtime does not expose XR_KHR_vulkan_enable2 (need Vulkan rendering).".into(),
        ));
    }

    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;
    #[cfg(target_os = "android")]
    {
        enabled_extensions.khr_android_create_instance = true;
    }

    let xr_instance = xr_entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "Renderide",
            application_version: 1,
            engine_name: "Renderide",
            engine_version: 1,
            api_version: xr::Version::new(1, 0, 0),
        },
        &enabled_extensions,
        &[],
    )?;

    let xr_system_id = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    let environment_blend_mode = xr_instance.enumerate_environment_blend_modes(
        xr_system_id,
        xr::ViewConfigurationType::PRIMARY_STEREO,
    )?[0];

    let vk_target_version = vk::make_api_version(0, 1, 1, 0);
    let vk_target_version_xr = xr::Version::new(1, 1, 0);
    let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(xr_system_id)?;
    if vk_target_version_xr < reqs.min_api_version_supported
        || vk_target_version_xr.major() > reqs.max_api_version_supported.major()
    {
        return Err(XrBootstrapError::Message(format!(
            "OpenXR requires Vulkan between {} and {}.0.0",
            reqs.min_api_version_supported,
            reqs.max_api_version_supported.major() + 1
        )));
    }

    let vk_entry =
        unsafe { ash::Entry::load() }.map_err(|e| XrBootstrapError::Vulkan(e.to_string()))?;

    let instance_api_version = match unsafe { vk_entry.try_enumerate_instance_version() } {
        Ok(Some(v)) => v,
        Ok(None) => vk::API_VERSION_1_0,
        Err(e) => {
            return Err(XrBootstrapError::Vulkan(format!(
                "try_enumerate_instance_version: {e}"
            )))
        }
    };

    let flags = wgt::InstanceFlags::empty();
    let extensions =
        hal::vulkan::Instance::desired_extensions(&vk_entry, instance_api_version, flags)
            .map_err(|e| XrBootstrapError::Vulkan(format!("desired_extensions: {e}")))?;

    let app_name = std::ffi::CString::new("Renderide")
        .map_err(|_| XrBootstrapError::Message("app name".into()))?;
    let vk_app_info = vk::ApplicationInfo::default()
        .application_name(app_name.as_c_str())
        .application_version(1)
        .engine_name(app_name.as_c_str())
        .engine_version(1)
        .api_version(vk_target_version);

    let extensions_cstr: Vec<_> = extensions.iter().map(|s| s.as_ptr()).collect();
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&vk_app_info)
        .enabled_extension_names(&extensions_cstr);

    let vk_instance = unsafe {
        let raw = xr_instance
            .create_vulkan_instance(
                xr_system_id,
                // OpenXR expects `PFN_vkVoidFunction`-compatible getInstanceProcAddr; ash’s type differs by ABI.
                #[allow(clippy::missing_transmute_annotations)]
                std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                &create_info as *const _ as *const _,
            )?
            .map_err(vk::Result::from_raw)?;
        let handle = raw as usize as u64;
        ash::Instance::load(vk_entry.static_fn(), vk::Instance::from_raw(handle))
    };

    let vk_physical_device = vk::PhysicalDevice::from_raw(unsafe {
        xr_instance
            .vulkan_graphics_device(xr_system_id, vk_instance.handle().as_raw() as *const c_void)?
    } as usize as u64);

    let vk_device_properties =
        unsafe { vk_instance.get_physical_device_properties(vk_physical_device) };
    if vk_device_properties.api_version < vk_target_version {
        return Err(XrBootstrapError::Message(
            "Vulkan physical device does not support API version 1.1.".into(),
        ));
    }

    let queue_family_index =
        unsafe { vk_instance.get_physical_device_queue_family_properties(vk_physical_device) }
            .into_iter()
            .enumerate()
            .find_map(|(i, info)| {
                if info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .ok_or_else(|| XrBootstrapError::Message("No Vulkan graphics queue family.".into()))?;

    let wgpu_vk_instance = unsafe {
        hal::vulkan::Instance::from_raw(
            vk_entry.clone(),
            vk_instance.clone(),
            vk::API_VERSION_1_3,
            0,
            None,
            extensions,
            flags,
            wgt::MemoryBudgetThresholds::default(),
            false,
            None,
        )
        .map_err(|e| XrBootstrapError::Vulkan(format!("hal Instance::from_raw: {e}")))?
    };

    let wgpu_exposed = wgpu_vk_instance
        .expose_adapter(vk_physical_device)
        .ok_or_else(|| XrBootstrapError::Wgpu("expose_adapter returned None".into()))?;

    let compression = wgt::Features::TEXTURE_COMPRESSION_BC
        | wgt::Features::TEXTURE_COMPRESSION_ETC2
        | wgt::Features::TEXTURE_COMPRESSION_ASTC;
    let wgpu_features = wgt::Features::MULTIVIEW | (wgpu_exposed.features & compression);

    let enabled_device_extensions = wgpu_exposed
        .adapter
        .required_device_extensions(wgpu_features);

    let mut enabled_phd_features = wgpu_exposed
        .adapter
        .physical_device_features(&enabled_device_extensions, wgpu_features);

    let family_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0f32]);
    let str_pointers: Vec<_> = enabled_device_extensions
        .iter()
        .map(|e| e.as_ptr())
        .collect();
    let pre_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&family_info))
        .enabled_extension_names(&str_pointers);
    let device_create_info = enabled_phd_features.add_to_device_create(pre_info);

    let vk_device = unsafe {
        let raw = xr_instance
            .create_vulkan_device(
                xr_system_id,
                #[allow(clippy::missing_transmute_annotations)]
                std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                vk_physical_device.as_raw() as *const c_void,
                &device_create_info as *const _ as *const _,
            )?
            .map_err(vk::Result::from_raw)?;
        ash::Device::load(
            vk_instance.fp_v1_0(),
            vk::Device::from_raw(raw as usize as u64),
        )
    };

    let (session, frame_wait, frame_stream) = unsafe {
        xr_instance.create_session::<xr::Vulkan>(
            xr_system_id,
            &xr::vulkan::SessionCreateInfo {
                instance: vk_instance.handle().as_raw() as _,
                physical_device: vk_physical_device.as_raw() as _,
                device: vk_device.handle().as_raw() as _,
                queue_family_index,
                queue_index: 0,
            },
        )
    }
    .map_err(XrBootstrapError::OpenXr)?;
    let stage: xr::Space = session
        .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
        .map_err(XrBootstrapError::OpenXr)?;
    let openxr_input = match OpenxrInput::new(&xr_instance, &session) {
        Ok(i) => Some(i),
        Err(e) => {
            logger::warn!("OpenXR controller input unavailable (continuing without actions): {e}");
            None
        }
    };
    let xr_session = super::session::XrSessionState::new(
        xr_instance,
        environment_blend_mode,
        session,
        frame_wait,
        frame_stream,
        stage,
    );

    let mut limits = wgpu_exposed.capabilities.limits.clone();
    // The OpenXR path renders to 2-layer array targets. Passing `Limits::default()` sets
    // `max_multiview_view_count` to 0, so wgpu-core rejects the pass with "Multiview view count
    // limit violated". Use the adapter's reported limits and require at least two views for stereo.
    limits.max_multiview_view_count = limits.max_multiview_view_count.max(2);
    let memory_hints = wgpu::MemoryHints::default();

    let wgpu_open_device = unsafe {
        wgpu_exposed.adapter.device_from_raw(
            vk_device,
            None,
            &enabled_device_extensions,
            wgpu_features,
            &limits,
            &memory_hints,
            queue_family_index,
            0,
        )
    }
    .map_err(|e| XrBootstrapError::Wgpu(format!("device_from_raw: {e}")))?;

    let wgpu_instance = unsafe { wgpu::Instance::from_hal::<HalVulkan>(wgpu_vk_instance) };
    let wgpu_adapter = unsafe { wgpu_instance.create_adapter_from_hal(wgpu_exposed) };

    let device_desc = wgpu::DeviceDescriptor {
        label: Some("renderide-openxr"),
        required_features: wgpu_features,
        required_limits: limits,
        memory_hints,
        experimental_features: Default::default(),
        trace: Default::default(),
    };

    let (wgpu_device, wgpu_queue) =
        unsafe { wgpu_adapter.create_device_from_hal(wgpu_open_device, &device_desc) }
            .map_err(|e| XrBootstrapError::Wgpu(format!("create_device_from_hal: {e}")))?;

    Ok(XrWgpuHandles {
        wgpu_instance,
        wgpu_adapter,
        device: Arc::new(wgpu_device),
        queue: Arc::new(Mutex::new(wgpu_queue)),
        xr_session,
        xr_system_id,
        openxr_input,
    })
}
