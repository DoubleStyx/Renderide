//! Bootstrap result and error types.

use std::sync::Arc;

use ash::vk;
use openxr as xr;
use thiserror::Error;

use crate::frontend::input::HeadsetMetadata;
use crate::xr::input::OpenxrInput;
use crate::xr::session::XrSessionState;

/// WGPU + OpenXR objects produced by [`super::init_wgpu_openxr`].
pub struct XrWgpuHandles {
    /// WGPU instance (Vulkan backend).
    pub wgpu_instance: wgpu::Instance,
    /// Adapter for the XR-selected physical device.
    pub wgpu_adapter: wgpu::Adapter,
    /// WGPU device shared with the desktop path (XR + window mirror).
    pub device: Arc<wgpu::Device>,
    /// Default queue for submits (wgpu::Queue is internally synchronized).
    pub queue: Arc<wgpu::Queue>,
    /// OpenXR session, frame stream, and reference space.
    pub xr_session: XrSessionState,
    /// Active system (HMD) id.
    pub xr_system_id: xr::SystemId,
    /// Host-facing OpenXR headset metadata forwarded through lock-step VR input.
    pub(crate) headset_metadata: HeadsetMetadata,
    /// Controller actions and spaces; `None` if action creation or Touch bindings failed.
    pub openxr_input: Option<OpenxrInput>,
    /// Whether `XR_KHR_composition_layer_depth` was enabled for this instance.
    pub composition_layer_depth_enabled: bool,
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
