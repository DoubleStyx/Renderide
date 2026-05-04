//! Create a Vulkan instance and device via OpenXR `KHR_vulkan_enable2`, then wrap with wgpu.
//!
//! Vulkan validation layers follow the same rules as desktop GPU init: [`crate::config::DebugSettings::gpu_validation_layers`]
//! in `config.toml` (and `RENDERIDE_GPU_VALIDATION`), plus `WGPU_*` env overrides via [`crate::gpu::instance_flags_for_gpu_init`].

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use ash::khr::timeline_semaphore as khr_timeline_semaphore;
use ash::vk::{self, Handle};
use openxr as xr;
use wgpu::hal;
use wgpu::hal::api::Vulkan as HalVulkan;

use wgpu::wgt;

use super::input::{ManifestError, OpenxrInput, ProfileExtensionGates, load_manifest};
pub use types::{XrBootstrapError, XrWgpuHandles};
use version::{choose_vulkan_api_version_for_wgpu, format_vk_api_version};

mod types;
mod version;

/// Cached `vkGetInstanceProcAddr` function pointer captured from the active [`ash::Entry`].
/// Installed by [`create_openxr_vulkan_instance`] before calling
/// [`xr::Instance::create_vulkan_instance`]; consumed by [`vk_get_instance_proc_addr_shim`].
static CACHED_VK_GET_INSTANCE_PROC_ADDR: OnceLock<vk::PFN_vkGetInstanceProcAddr> = OnceLock::new();

/// OpenXR-shaped wrapper around the cached `vkGetInstanceProcAddr`.
///
/// OpenXR's `xrCreateVulkanInstanceKHR` expects a function pointer typed
/// `unsafe extern "system" fn(*const c_void, *const c_char) -> Option<...>`, while ash exposes the
/// same loader entry typed `unsafe extern "system" fn(vk::Instance, *const c_char) -> Option<...>`.
/// `vk::Instance` is a `#[repr(transparent)]` pointer-sized handle so the calling convention is
/// identical at the ABI level, but transmuting between fn-pointer types is not language-blessed.
/// This shim instead receives the OpenXR-typed call and forwards through the cached, properly
/// typed ash entry -- no fn-pointer transmute is required.
unsafe extern "system" fn vk_get_instance_proc_addr_shim(
    instance: *const c_void,
    name: *const std::os::raw::c_char,
) -> Option<unsafe extern "system" fn()> {
    let real = CACHED_VK_GET_INSTANCE_PROC_ADDR.get().copied()?;
    let handle = vk::Instance::from_raw(instance as usize as u64);
    // SAFETY: `real` is the live `vkGetInstanceProcAddr` pointer captured from a successfully
    // loaded `ash::Entry`; OpenXR forwards the original `instance` handle and `name` pointer
    // verbatim, so the contract for `vkGetInstanceProcAddr` is upheld.
    unsafe { real(handle, name) }
}

/// Converts the fixed-size NUL-padded `device_name` from [`vk::PhysicalDeviceProperties`] to a
/// printable [`String`] without invoking unsafe `CStr` parsing.
fn vk_physical_device_name(props: &vk::PhysicalDeviceProperties) -> String {
    let bytes: Vec<u8> = props
        .device_name
        .iter()
        .take_while(|b| **b != 0)
        .map(|b| *b as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Fails fast if neither [`vkWaitSemaphores`] (Vulkan 1.2 core) nor [`vkWaitSemaphoresKHR`] is
/// exported for the logical device. WGPU uses one of these for timeline-semaphore fences.
fn verify_device_has_wait_semaphores(
    vk_instance: &ash::Instance,
    device: vk::Device,
) -> Result<(), XrBootstrapError> {
    // SAFETY: `vk_instance` is a valid Vulkan instance and `device` is a `VkDevice` it created;
    // the C-string literal has static lifetime and is NUL-terminated.
    let addr_core = unsafe {
        (vk_instance.fp_v1_0().get_device_proc_addr)(device, c"vkWaitSemaphores".as_ptr())
    };
    // SAFETY: same preconditions as above.
    let addr_khr = unsafe {
        (vk_instance.fp_v1_0().get_device_proc_addr)(device, c"vkWaitSemaphoresKHR".as_ptr())
    };
    if addr_core.is_none() && addr_khr.is_none() {
        return Err(XrBootstrapError::Vulkan(
            "Vulkan device missing vkWaitSemaphores and vkWaitSemaphoresKHR; timeline semaphore support is required for wgpu."
                .into(),
        ));
    }
    Ok(())
}

/// Loads the OpenXR API entry: tries [`super::openxr_loader_paths::openxr_loader_candidate_paths`]
/// with [`xr::Entry::load_from`], then falls back to [`xr::Entry::load`] (default library search).
fn load_xr_entry() -> Result<xr::Entry, xr::LoadError> {
    let paths = super::openxr_loader_paths::openxr_loader_candidate_paths();
    for path in paths {
        // SAFETY: `xr::Entry::load_from` dynamically loads the OpenXR loader from `path`; the
        // crate requires callers to guarantee the library at that path is a valid, ABI-compatible
        // OpenXR loader. The candidate paths come from platform-known install locations.
        match unsafe { xr::Entry::load_from(&path) } {
            Ok(entry) => {
                logger::debug!("OpenXR loader loaded from {}", path.display());
                return Ok(entry);
            }
            Err(e) => {
                logger::trace!("OpenXR loader not loaded from {}: {e}", path.display());
            }
        }
    }
    // SAFETY: `xr::Entry::load()` uses the default dynamic-linker search for the OpenXR loader;
    // relies on the platform's standard library search path to resolve a valid loader.
    match unsafe { xr::Entry::load() } {
        Ok(entry) => {
            logger::debug!("OpenXR loader loaded via default library search");
            Ok(entry)
        }
        Err(e) => Err(e),
    }
}

/// Result of [`create_openxr_instance`] for [`init_wgpu_openxr`].
struct OpenxrInstanceBundle {
    xr_instance: xr::Instance,
    profile_gates: ProfileExtensionGates,
}

/// Loads extension flags, validates `XR_KHR_vulkan_enable2`, and creates the OpenXR [`xr::Instance`].
///
/// Every controller-related extension the runtime advertises is enabled so the corresponding
/// vendor interaction profile can be suggested. The resulting [`ProfileExtensionGates`] tells
/// [`super::input::OpenxrInput`] which profile binding tables to attempt.
fn create_openxr_instance(xr_entry: xr::Entry) -> Result<OpenxrInstanceBundle, XrBootstrapError> {
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
    enabled_extensions.khr_generic_controller = available_extensions.khr_generic_controller;
    enabled_extensions.bd_controller_interaction = available_extensions.bd_controller_interaction;
    enabled_extensions.ext_hp_mixed_reality_controller =
        available_extensions.ext_hp_mixed_reality_controller;
    enabled_extensions.ext_samsung_odyssey_controller =
        available_extensions.ext_samsung_odyssey_controller;
    enabled_extensions.htc_vive_cosmos_controller_interaction =
        available_extensions.htc_vive_cosmos_controller_interaction;
    enabled_extensions.htc_vive_focus3_controller_interaction =
        available_extensions.htc_vive_focus3_controller_interaction;
    enabled_extensions.fb_touch_controller_pro = available_extensions.fb_touch_controller_pro;
    enabled_extensions.meta_touch_controller_plus = available_extensions.meta_touch_controller_plus;
    if available_extensions.ext_debug_utils {
        enabled_extensions.ext_debug_utils = true;
    }
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

    let profile_gates = ProfileExtensionGates {
        khr_generic_controller: enabled_extensions.khr_generic_controller,
        bd_controller: enabled_extensions.bd_controller_interaction,
        ext_hp_mixed_reality_controller: enabled_extensions.ext_hp_mixed_reality_controller,
        ext_samsung_odyssey_controller: enabled_extensions.ext_samsung_odyssey_controller,
        htc_vive_cosmos_controller_interaction: enabled_extensions
            .htc_vive_cosmos_controller_interaction,
        htc_vive_focus3_controller_interaction: enabled_extensions
            .htc_vive_focus3_controller_interaction,
        fb_touch_controller_pro: enabled_extensions.fb_touch_controller_pro,
        meta_touch_controller_plus: enabled_extensions.meta_touch_controller_plus,
    };

    Ok(OpenxrInstanceBundle {
        xr_instance,
        profile_gates,
    })
}

type VulkanGraphicsRequirements = <xr::Vulkan as xr::Graphics>::Requirements;

/// HMD system, blend mode, and Vulkan requirements from OpenXR.
fn probe_head_set_and_vulkan_requirements(
    xr_instance: &xr::Instance,
) -> Result<
    (
        xr::SystemId,
        xr::EnvironmentBlendMode,
        VulkanGraphicsRequirements,
    ),
    XrBootstrapError,
> {
    let xr_system_id = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    let environment_blend_mode = xr_instance.enumerate_environment_blend_modes(
        xr_system_id,
        xr::ViewConfigurationType::PRIMARY_STEREO,
    )?[0];
    let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(xr_system_id)?;
    Ok((xr_system_id, environment_blend_mode, reqs))
}

/// Ash entry, Vulkan instance created via OpenXR, physical device, and wgpu-hal instance flags.
struct OpenxrAshVkInstance {
    vk_entry: ash::Entry,
    vk_instance: ash::Instance,
    vk_target_version: u32,
    vk_physical_device: vk::PhysicalDevice,
    extensions: Vec<&'static std::ffi::CStr>, // Must match `hal::vulkan::Instance::desired_extensions`.
    flags: wgt::InstanceFlags,
}

/// Creates [`ash::Instance`] and resolves the OpenXR-chosen physical device.
fn create_openxr_vulkan_instance(
    xr_instance: &xr::Instance,
    xr_system_id: xr::SystemId,
    gpu_validation_layers: bool,
    reqs: &VulkanGraphicsRequirements,
) -> Result<OpenxrAshVkInstance, XrBootstrapError> {
    // SAFETY: `ash::Entry::load()` dynamically loads the platform's Vulkan loader. Relies on the
    // dynamic linker's standard search path to locate a compatible `libvulkan`.
    let vk_entry =
        unsafe { ash::Entry::load() }.map_err(|e| XrBootstrapError::Vulkan(e.to_string()))?;

    // SAFETY: `vk_entry` was just successfully loaded; calling
    // `vkEnumerateInstanceVersion` via its static function table is sound.
    let instance_api_version = match unsafe { vk_entry.try_enumerate_instance_version() } {
        Ok(Some(v)) => v,
        Ok(None) => vk::API_VERSION_1_0,
        Err(e) => {
            return Err(XrBootstrapError::Vulkan(format!(
                "try_enumerate_instance_version: {e}"
            )));
        }
    };

    let vk_target_version = choose_vulkan_api_version_for_wgpu(instance_api_version, reqs)?;

    let flags = crate::gpu::instance_flags_for_gpu_init(gpu_validation_layers);
    let extensions =
        hal::vulkan::Instance::desired_extensions(&vk_entry, instance_api_version, flags)
            .map_err(|e| XrBootstrapError::Vulkan(format!("desired_extensions: {e}")))?;

    let app_name = c"Renderide";
    let vk_app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(1)
        .engine_name(app_name)
        .engine_version(1)
        .api_version(vk_target_version);

    let extensions_cstr: Vec<_> = extensions.iter().map(|s| s.as_ptr()).collect();
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&vk_app_info)
        .enabled_extension_names(&extensions_cstr);

    // Install the loader entry pointer behind the OpenXR-shaped shim; subsequent OpenXR calls
    // forward through `vk_get_instance_proc_addr_shim` without any fn-pointer transmute.
    let _ = CACHED_VK_GET_INSTANCE_PROC_ADDR.set(vk_entry.static_fn().get_instance_proc_addr);
    // SAFETY: `xr_instance` and `xr_system_id` are valid; `create_info` is fully initialised
    // above and borrowed for the call's duration only. The shim above receives the OpenXR
    // arguments and forwards to the live `vkGetInstanceProcAddr` captured from `vk_entry`. The
    // resulting raw `VkInstance` handle is wrapped via `ash::Instance::load` so it shares the
    // same loader function table.
    let vk_instance = unsafe {
        let raw = xr_instance
            .create_vulkan_instance(
                xr_system_id,
                vk_get_instance_proc_addr_shim,
                core::ptr::from_ref(&create_info).cast(),
            )?
            .map_err(vk::Result::from_raw)?;
        let handle = raw as usize as u64;
        ash::Instance::load(vk_entry.static_fn(), vk::Instance::from_raw(handle))
    };

    // SAFETY: `xr_instance` is a valid OpenXR instance; `vk_instance`'s handle is the VkInstance
    // just created from it, so it is the correct argument for `xrGetVulkanGraphicsDeviceKHR`.
    let vk_physical_device = vk::PhysicalDevice::from_raw(unsafe {
        xr_instance
            .vulkan_graphics_device(xr_system_id, vk_instance.handle().as_raw() as *const c_void)?
    } as usize as u64);

    Ok(OpenxrAshVkInstance {
        vk_entry,
        vk_instance,
        vk_target_version,
        vk_physical_device,
        extensions,
        flags,
    })
}

/// `wgpu`-hal Vulkan instance plus exposed adapter, validated physical device properties, graphics queue index.
struct WgpuHalVkChain {
    wgpu_vk_instance: hal::vulkan::Instance,
    wgpu_exposed: hal::ExposedAdapter<HalVulkan>,
    vk_device_properties: vk::PhysicalDeviceProperties,
    queue_family_index: u32,
}

fn build_wgpu_hal_and_queue_family(
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

    // SAFETY: `vk_entry`/`vk_instance` are a live, matched pair just constructed above; the
    // extensions list is the same one passed to `vk_instance` creation, preserving wgpu-hal's
    // required invariants for `Instance::from_raw`.
    let wgpu_vk_instance = unsafe {
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
        .map_err(|e| XrBootstrapError::Vulkan(format!("hal Instance::from_raw: {e}")))?
    };

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

/// Vulkan device creation inputs for OpenXR `create_vulkan_device` + wgpu-hal negotiation.
struct VulkanOpenXrDeviceCreateDescriptor<'a> {
    xr_instance: &'a xr::Instance,
    xr_system_id: xr::SystemId,
    vk_entry: &'a ash::Entry,
    vk_instance: &'a ash::Instance,
    vk_physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    wgpu_exposed: &'a hal::ExposedAdapter<HalVulkan>,
    vk_device_properties: &'a vk::PhysicalDeviceProperties,
}

/// Creates the Vulkan logical device through OpenXR using wgpu-hal feature negotiation.
fn create_vulkan_logical_device_openxr(
    desc: VulkanOpenXrDeviceCreateDescriptor<'_>,
) -> Result<(wgt::Features, Vec<&'static std::ffi::CStr>, ash::Device), XrBootstrapError> {
    let compression = wgt::Features::TEXTURE_COMPRESSION_BC
        | wgt::Features::TEXTURE_COMPRESSION_ETC2
        | wgt::Features::TEXTURE_COMPRESSION_ASTC;
    let optional_float32_filterable = wgt::Features::FLOAT32_FILTERABLE;
    let optional_depth32_stencil8 = wgt::Features::DEPTH32FLOAT_STENCIL8;
    let timestamp = wgt::Features::TIMESTAMP_QUERY | wgt::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    // TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES: unlock hardware-reported MSAA sample counts (device
    // exposes the real tiers instead of the WebGPU baseline).
    // MULTISAMPLE_ARRAY: required for multisampled 2D array color/depth textures used by the stereo
    // (single-pass multiview) MSAA path. Absence is silently handled: the stereo path falls back to
    // `sample_count = 1` in [`crate::gpu::GpuContext::set_swapchain_msaa_requested_stereo`].
    let adapter_format_features = wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let multisample_array = wgt::Features::MULTISAMPLE_ARRAY;
    let wgpu_features = wgt::Features::MULTIVIEW
        | (desc.wgpu_exposed.features
            & (compression
                | optional_float32_filterable
                | optional_depth32_stencil8
                | adapter_format_features
                | multisample_array
                | timestamp));

    let mut enabled_device_extensions = desc
        .wgpu_exposed
        .adapter
        .required_device_extensions(wgpu_features);

    if desc.vk_device_properties.api_version >= vk::API_VERSION_1_2
        && desc
            .wgpu_exposed
            .adapter
            .physical_device_capabilities()
            .supports_extension(khr_timeline_semaphore::NAME)
        && !enabled_device_extensions
            .iter()
            .copied()
            .any(|e| e == khr_timeline_semaphore::NAME)
    {
        enabled_device_extensions.push(khr_timeline_semaphore::NAME);
    }

    let mut enabled_phd_features = desc
        .wgpu_exposed
        .adapter
        .physical_device_features(&enabled_device_extensions, wgpu_features);

    let family_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(desc.queue_family_index)
        .queue_priorities(&[1.0f32]);
    let str_pointers: Vec<_> = enabled_device_extensions
        .iter()
        .map(|e| e.as_ptr())
        .collect();
    let pre_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&family_info))
        .enabled_extension_names(&str_pointers);
    let device_create_info = enabled_phd_features.add_to_device_create(pre_info);

    // SAFETY: see `create_openxr_vulkan_instance` -- the same ABI-compatible pointer transmute.
    // `vk_physical_device` was obtained from the matching `xr_instance`/`xr_system_id` pair;
    // `device_create_info` references data that outlives the call. `ash::Device::load` ties the
    // new `VkDevice` to `desc.vk_instance`'s function table.
    let vk_device = unsafe {
        let raw = desc
            .xr_instance
            .create_vulkan_device(
                desc.xr_system_id,
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        vk::Instance,
                        *const i8,
                    )
                        -> Option<unsafe extern "system" fn()>,
                    unsafe extern "system" fn(
                        *const c_void,
                        *const i8,
                    )
                        -> Option<unsafe extern "system" fn()>,
                >(desc.vk_entry.static_fn().get_instance_proc_addr),
                desc.vk_physical_device.as_raw() as *const c_void,
                core::ptr::from_ref(&device_create_info).cast(),
            )?
            .map_err(vk::Result::from_raw)?;
        let device_handle = vk::Device::from_raw(raw as usize as u64);
        verify_device_has_wait_semaphores(desc.vk_instance, device_handle)?;
        ash::Device::load(desc.vk_instance.fp_v1_0(), device_handle)
    };

    Ok((wgpu_features, enabled_device_extensions, vk_device))
}

/// OpenXR session, reference space, optional controller actions, and [`super::session::XrSessionState`].
struct OpenXrSessionBootstrapDescriptor<'a> {
    xr_instance: xr::Instance,
    openxr_debug_messenger: Option<super::debug_utils::OpenxrDebugUtilsMessenger>,
    environment_blend_mode: xr::EnvironmentBlendMode,
    xr_system_id: xr::SystemId,
    vk_instance: &'a ash::Instance,
    vk_physical_device: vk::PhysicalDevice,
    vk_device: &'a ash::Device,
    queue_family_index: u32,
    profile_gates: ProfileExtensionGates,
}

fn openxr_session_state_and_input(
    desc: OpenXrSessionBootstrapDescriptor<'_>,
) -> Result<(super::session::XrSessionState, Option<OpenxrInput>), XrBootstrapError> {
    // SAFETY: `desc.vk_instance`/`vk_physical_device`/`vk_device` form a matched OpenXR-negotiated
    // Vulkan chain from `create_vulkan_logical_device_openxr`; `queue_family_index` was chosen
    // above to be a graphics-capable family on the selected device.
    let (session, frame_wait, frame_stream) = unsafe {
        desc.xr_instance.create_session::<xr::Vulkan>(
            desc.xr_system_id,
            &xr::vulkan::SessionCreateInfo {
                instance: desc.vk_instance.handle().as_raw() as xr::sys::platform::VkInstance,
                physical_device: desc.vk_physical_device.as_raw()
                    as xr::sys::platform::VkPhysicalDevice,
                device: desc.vk_device.handle().as_raw() as xr::sys::platform::VkDevice,
                queue_family_index: desc.queue_family_index,
                queue_index: 0,
            },
        )
    }
    .map_err(XrBootstrapError::OpenXr)?;
    let stage: xr::Space = session
        .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
        .map_err(XrBootstrapError::OpenXr)?;

    let openxr_input = match load_manifest() {
        Ok((manifest, location)) => {
            logger::info!(
                "Loaded OpenXR action manifest from {} ({} profile(s))",
                location.root.display(),
                manifest.profiles.len()
            );
            match OpenxrInput::new(&desc.xr_instance, &session, &desc.profile_gates, &manifest) {
                Ok(i) => Some(i),
                Err(e) => {
                    logger::warn!(
                        "OpenXR controller input unavailable (continuing without actions): {e}"
                    );
                    None
                }
            }
        }
        Err(ManifestError::ActionsManifestMissing { ref searched }) => {
            logger::warn!(
                "OpenXR action manifest not found; searched: {}",
                searched.join(", ")
            );
            None
        }
        Err(e) => {
            logger::warn!("OpenXR action manifest load failed: {e}");
            None
        }
    };
    let xr_session =
        super::session::XrSessionState::new(super::session::XrSessionStateDescriptor {
            xr_instance: desc.xr_instance,
            openxr_debug_messenger: desc.openxr_debug_messenger,
            environment_blend_mode: desc.environment_blend_mode,
            session,
            frame_wait,
            frame_stream,
            stage,
        });
    Ok((xr_session, openxr_input))
}

/// wgpu-hal + OpenXR session packaging into [`XrWgpuHandles`].
struct WgpuHalOpenXrAssembly {
    wgpu_vk_instance: hal::vulkan::Instance,
    wgpu_exposed: hal::ExposedAdapter<HalVulkan>,
    vk_device: ash::Device,
    enabled_device_extensions: Vec<&'static std::ffi::CStr>,
    wgpu_features: wgt::Features,
    queue_family_index: u32,
    xr_session: super::session::XrSessionState,
    xr_system_id: xr::SystemId,
    openxr_input: Option<OpenxrInput>,
}

/// Wraps Ash device and wgpu-hal adapter in [`wgpu::Instance`] / [`wgpu::Device`] / [`XrWgpuHandles`].
fn wgpu_from_hal_openxr_chain(
    assembly: WgpuHalOpenXrAssembly,
) -> Result<XrWgpuHandles, XrBootstrapError> {
    let mut limits = assembly.wgpu_exposed.capabilities.limits.clone();
    limits.max_multiview_view_count = limits.max_multiview_view_count.max(2);
    let memory_hints = wgpu::MemoryHints::default();

    // SAFETY: `assembly.vk_device` was created through the wgpu-hal adapter described by
    // `assembly.wgpu_exposed` with exactly the features/extensions passed here; the queue family
    // and index were those used during `vkCreateDevice`.
    let wgpu_open_device = unsafe {
        assembly.wgpu_exposed.adapter.device_from_raw(
            assembly.vk_device,
            None,
            assembly.enabled_device_extensions.as_slice(),
            assembly.wgpu_features,
            &limits,
            &memory_hints,
            assembly.queue_family_index,
            0,
        )
    }
    .map_err(|e| XrBootstrapError::Wgpu(format!("device_from_raw: {e}")))?;

    // SAFETY: `assembly.wgpu_vk_instance` is a valid `hal::vulkan::Instance` just built from a
    // live `ash::Entry`/`ash::Instance` pair; ownership transfers into the wgpu `Instance`.
    let wgpu_instance = unsafe { wgpu::Instance::from_hal::<HalVulkan>(assembly.wgpu_vk_instance) };
    // SAFETY: `assembly.wgpu_exposed` was enumerated from the same Vulkan instance now held by
    // `wgpu_instance`, so the exposed adapter is coherent with it.
    let wgpu_adapter = unsafe { wgpu_instance.create_adapter_from_hal(assembly.wgpu_exposed) };

    let device_desc = wgpu::DeviceDescriptor {
        label: Some("renderide-openxr"),
        required_features: assembly.wgpu_features,
        required_limits: limits,
        memory_hints,
        experimental_features: Default::default(),
        trace: Default::default(),
    };

    // SAFETY: `wgpu_open_device` was opened from `wgpu_adapter`'s underlying hal adapter above;
    // `device_desc` uses the same features/limits passed to `device_from_raw`.
    let (wgpu_device, wgpu_queue) =
        unsafe { wgpu_adapter.create_device_from_hal(wgpu_open_device, &device_desc) }
            .map_err(|e| XrBootstrapError::Wgpu(format!("create_device_from_hal: {e}")))?;

    Ok(XrWgpuHandles {
        wgpu_instance,
        wgpu_adapter,
        device: Arc::new(wgpu_device),
        queue: Arc::new(wgpu_queue),
        xr_session: assembly.xr_session,
        xr_system_id: assembly.xr_system_id,
        openxr_input: assembly.openxr_input,
    })
}

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
    } = create_openxr_instance(xr_entry)?;

    let openxr_debug_messenger =
        super::debug_utils::OpenxrDebugUtilsMessenger::try_create(&xr_instance);

    let (xr_system_id, environment_blend_mode, reqs) =
        probe_head_set_and_vulkan_requirements(&xr_instance)?;
    let ash_vk =
        create_openxr_vulkan_instance(&xr_instance, xr_system_id, gpu_validation_layers, &reqs)?;
    let vk_physical_device = ash_vk.vk_physical_device;
    let vk_entry = ash_vk.vk_entry.clone();
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
            vk_entry: &vk_entry,
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
    })
}
