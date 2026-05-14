//! Device creation and uncaptured-error wiring.
//!
//! Wraps [`wgpu::Adapter::request_device`] with the renderer's required-features set and
//! the non-panicking uncaptured error handler. Also hosts the helper that builds
//! [`crate::profiling::GpuProfilerHandle`] with path-specific fallback logging.

use std::sync::Arc;

use super::super::context::GpuError;
use super::super::instance_setup::required_limits_for_adapter;
use super::super::sync::device_health::GpuDeviceHealth;
use super::super::sync::mapped_buffer_health::{
    GpuMappedBufferHealth, validation_mentions_mapped_buffer_invalidation,
};

/// Asynchronously requests a device from `adapter` for `required_features`.
///
/// Installs the renderer's non-panicking uncaptured error handler before returning so all
/// GPU paths (windowed, headless, OpenXR) get the same protection.
pub(crate) async fn request_device_for_adapter(
    adapter: &wgpu::Adapter,
    required_features: wgpu::Features,
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    device_health: Arc<GpuDeviceHealth>,
) -> Result<(Arc<wgpu::Device>, wgpu::Queue), GpuError> {
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("renderide-skeleton"),
            required_features,
            required_limits: required_limits_for_adapter(adapter),
            ..Default::default()
        })
        .await
        .map_err(|e| GpuError::Device(format!("{e:?}")))?;
    install_uncaptured_error_handler(&device, mapped_buffer_health, device_health);
    Ok((Arc::new(device), queue))
}

/// Installs a non-panicking uncaptured error handler on `device` so stray wgpu validation
/// errors (for example, a [`wgpu::Device::create_view`] on a texture left invalid by a
/// device-lost event) are logged instead of terminating the process via wgpu's default
/// panicking handler. Callers that pass an externally built device (OpenXR bootstrap) must
/// invoke this explicitly so that path gets the same protection as the owned-device paths.
pub(crate) fn install_uncaptured_error_handler(
    device: &wgpu::Device,
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    device_health: Arc<GpuDeviceHealth>,
) {
    let lost_health = Arc::clone(&mapped_buffer_health);
    device.set_device_lost_callback(move |reason, message| {
        logger::error!("wgpu device lost: reason={reason:?} message={message}");
        lost_health.mark_invalid("wgpu device lost");
        device_health.mark_lost(format!("{reason:?}: {message}"));
    });

    device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| match err {
        wgpu::Error::OutOfMemory { source } => {
            logger::error!("wgpu out-of-memory error: {source}");
            mapped_buffer_health.mark_invalid("wgpu out of memory");
        }
        wgpu::Error::Validation {
            description,
            source,
        } => {
            let source_text = source.to_string();
            if validation_mentions_mapped_buffer_invalidation(&description, &source_text) {
                mapped_buffer_health.mark_invalid("wgpu mapped buffer validation error");
            }
            logger::error!("wgpu validation error: {description} ({source})");
        }
        wgpu::Error::Internal {
            description,
            source,
        } => {
            logger::error!("wgpu internal error: {description} ({source})");
            mapped_buffer_health.mark_invalid("wgpu internal error");
        }
    }));
}

/// Attempts to create the Tracy GPU profiler and logs a path-specific fallback when unavailable.
pub(crate) fn try_gpu_profiler(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    unavailable_message: &str,
) -> Option<crate::profiling::GpuProfilerHandle> {
    let gpu_profiler = crate::profiling::GpuProfilerHandle::try_new(adapter, device, queue);
    if cfg!(feature = "tracy") && gpu_profiler.is_none() {
        logger::warn!("{unavailable_message}");
    }
    gpu_profiler
}
