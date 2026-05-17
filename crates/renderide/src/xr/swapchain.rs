//! OpenXR stereo swapchain images imported into wgpu for the acquired frame.
//!
//! These images are always created with `sample_count = 1` and act as the **resolve** target for
//! the final renderer-owned HMD color copy. Each acquired OpenXR-owned `VkImage` is wrapped in
//! wgpu only for the current acquire/release interval so wgpu's Vulkan layout tracker cannot carry
//! stale state across compositor ownership handoffs.

use std::sync::Arc;

use ash::vk::{self, Handle};
use openxr as xr;
use parking_lot::Mutex;
use thiserror::Error;
use wgpu::TextureUses;
use wgpu::hal::api::Vulkan as HalVulkan;
use wgpu::hal::{self, MemoryFlags};

use super::XrWgpuHandles;

/// Two array layers (left / right) for `PRIMARY_STEREO`.
pub const XR_VIEW_COUNT: u32 = 2;

/// Color format matching [`XR_VK_FORMAT`] and wgpu import.
pub const XR_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Vulkan format passed to OpenXR (`VK_FORMAT_R8G8B8A8_SRGB`).
pub const XR_VK_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// Swapchain creation or wgpu import failure.
#[derive(Debug, Error)]
pub enum XrSwapchainError {
    /// OpenXR API error.
    #[error("OpenXR: {0}")]
    OpenXr(#[from] xr::sys::Result),
    /// No view configuration from the runtime.
    #[error("no PRIMARY_STEREO view configuration")]
    NoViewConfiguration,
    /// Device is not Vulkan / hal interop unavailable.
    #[error("wgpu device is not Vulkan or as_hal failed")]
    NotVulkanHal,
    /// OpenXR returned a swapchain image index outside the enumerated image table.
    #[error("OpenXR swapchain image index {index} out of range for {image_count} image(s)")]
    ImageIndexOutOfRange {
        /// Image index returned by `xrAcquireSwapchainImage`.
        index: u32,
        /// Number of images returned by `xrEnumerateSwapchainImages`.
        image_count: u32,
    },
}

/// OpenXR swapchain plus raw Vulkan image handles enumerated from the runtime.
pub struct XrStereoSwapchain {
    images: Vec<vk::Image>,
    /// Per-eye rectangle size in pixels.
    pub resolution: (u32, u32),
    /// Runtime swapchain handle (acquire / release / composition). Behind a [`Mutex`]
    /// so the driver thread can release the image and reference the swapchain in the
    /// projection layer for `xrEndFrame` while the main thread retains shared ownership
    /// across ticks. The runtime is the sole owner of the underlying `VkImage`s and frees
    /// them on `xrDestroySwapchain`.
    pub handle: Arc<Mutex<xr::Swapchain<xr::Vulkan>>>,
}

/// Per-frame wgpu wrapper for an acquired OpenXR swapchain image.
///
/// The wrapper owns the imported wgpu texture and its two-layer array view. It must stay alive
/// until all command buffers referencing the acquired image have been submitted to the GPU queue.
/// The OpenXR runtime owns the underlying `VkImage`; dropping this value only releases wgpu's
/// tracking wrapper.
pub struct XrAcquiredSwapchainImage {
    texture: wgpu::Texture,
    array_view: wgpu::TextureView,
}

impl XrAcquiredSwapchainImage {
    /// Two-layer color target view used by the final owned-HMD-color to OpenXR copy.
    pub fn array_view(&self) -> &wgpu::TextureView {
        &self.array_view
    }

    /// Consumes the acquired-image wrapper, leaving the imported wgpu texture alive.
    pub fn into_texture(self) -> wgpu::Texture {
        self.texture
    }
}

impl XrStereoSwapchain {
    /// Creates an OpenXR swapchain and records the runtime-owned Vulkan images.
    pub fn new(handles: &XrWgpuHandles) -> Result<Self, XrSwapchainError> {
        let session = handles.xr_session.xr_vulkan_session();
        let xr_instance = handles.xr_session.xr_instance();
        let system_id = handles.xr_system_id;
        let views = xr_instance.enumerate_view_configuration_views(
            system_id,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let v0 = views.first().ok_or(XrSwapchainError::NoViewConfiguration)?;
        let resolution = (
            v0.recommended_image_rect_width,
            v0.recommended_image_rect_height,
        );
        logger::info!(
            "OpenXR view configuration: views={} recommended={}x{} max={}x{}",
            views.len(),
            v0.recommended_image_rect_width,
            v0.recommended_image_rect_height,
            v0.max_image_rect_width,
            v0.max_image_rect_height,
        );

        let handle = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr_swapchain_usage_flags(),
            format: XR_VK_FORMAT.as_raw() as u32,
            sample_count: 1,
            width: resolution.0,
            height: resolution.1,
            face_count: 1,
            array_size: XR_VIEW_COUNT,
            mip_count: 1,
        })?;

        let images = handle.enumerate_images()?;
        logger::info!(
            "OpenXR swapchain images: count={} format={:?} resolution={}x{} array_layers={}",
            images.len(),
            XR_COLOR_FORMAT,
            resolution.0,
            resolution.1,
            XR_VIEW_COUNT,
        );
        let images = images.into_iter().map(vk::Image::from_raw).collect();

        Ok(Self {
            images,
            resolution,
            handle: Arc::new(Mutex::new(handle)),
        })
    }

    /// Number of runtime-owned images in the swapchain.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Imports the acquired swapchain image into wgpu for the current frame.
    pub fn import_acquired_image(
        &self,
        device: &wgpu::Device,
        image_index: usize,
    ) -> Result<XrAcquiredSwapchainImage, XrSwapchainError> {
        let Some(vk_image) = self.images.get(image_index).copied() else {
            return Err(XrSwapchainError::ImageIndexOutOfRange {
                index: u32_saturating_from_usize(image_index),
                image_count: u32_saturating_from_usize(self.images.len()),
            });
        };

        // SAFETY: `XrWgpuHandles` is produced by XR bootstrap from the same Vulkan device used to
        // create the OpenXR session, and `self.images` came from the session's swapchain.
        let hal_device =
            unsafe { device.as_hal::<HalVulkan>() }.ok_or(XrSwapchainError::NotVulkanHal)?;
        Ok(import_openxr_swapchain_image(
            device,
            &hal_device,
            vk_image,
            self.resolution,
        ))
    }
}

fn import_openxr_swapchain_image(
    device: &wgpu::Device,
    hal_device: &<HalVulkan as hal::Api>::Device,
    vk_image: vk::Image,
    resolution: (u32, u32),
) -> XrAcquiredSwapchainImage {
    let hal_desc = xr_swapchain_hal_descriptor(resolution);
    // Hand wgpu a no-op drop callback so its `destroy_texture` sees
    // `texture.drop_guard.is_some()` and skips `vkDestroyImage`. The OpenXR runtime is the sole
    // owner of `vk_image` and frees it on `xrDestroySwapchain`; calling `vkDestroyImage` from
    // wgpu would double-free during shutdown. A no-op closure is correct because callers import
    // only images enumerated from a live OpenXR swapchain and keep that swapchain alive until the
    // per-frame wrapper has been dropped.
    let drop_callback: hal::DropCallback = Box::new(|| {});
    // SAFETY: `vk_image` was returned by `xrEnumerateSwapchainImages` on a swapchain created from
    // the OpenXR session inside `XrWgpuHandles`; that session and `hal_device` come from the same
    // bootstrap-created Vulkan device. The descriptor mirrors the swapchain create info. The
    // non-null `drop_callback` signals to wgpu-hal that the `VkImage` is externally owned (the
    // OpenXR runtime); wgpu must not call `vkDestroyImage` on it. The runtime keeps the image
    // valid until `xrDestroySwapchain`, and callers keep the swapchain alive while the returned
    // wgpu wrapper exists.
    let hal_tex = unsafe {
        hal_device.texture_from_raw(
            vk_image,
            &hal_desc,
            Some(drop_callback),
            hal::vulkan::TextureMemory::External,
        )
    };
    let wgpu_desc = xr_swapchain_wgpu_descriptor(&hal_desc);
    // SAFETY: `hal_tex` was imported from the Vulkan device backing `device`, and `wgpu_desc`
    // matches the HAL descriptor used for the import.
    let texture = unsafe { device.create_texture_from_hal::<HalVulkan>(hal_tex, &wgpu_desc) };
    let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("xr_swapchain_array"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(XR_VIEW_COUNT),
        ..Default::default()
    });
    crate::profiling::note_resource_churn!(TextureView, "xr::swapchain_array_view");
    XrAcquiredSwapchainImage {
        texture,
        array_view,
    }
}

fn xr_swapchain_usage_flags() -> xr::SwapchainUsageFlags {
    xr::SwapchainUsageFlags::COLOR_ATTACHMENT
}

fn u32_saturating_from_usize(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}

fn xr_swapchain_hal_descriptor(resolution: (u32, u32)) -> hal::TextureDescriptor<'static> {
    hal::TextureDescriptor {
        label: Some("xr_swapchain"),
        size: wgpu::Extent3d {
            width: resolution.0,
            height: resolution.1,
            depth_or_array_layers: XR_VIEW_COUNT,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: XR_COLOR_FORMAT,
        usage: TextureUses::COLOR_TARGET,
        memory_flags: MemoryFlags::empty(),
        view_formats: Vec::new(),
    }
}

fn xr_swapchain_wgpu_descriptor(
    hal_desc: &hal::TextureDescriptor<'_>,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("xr_swapchain"),
        size: hal_desc.size,
        mip_level_count: hal_desc.mip_level_count,
        sample_count: hal_desc.sample_count,
        dimension: hal_desc.dimension,
        format: XR_COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    }
}

/// Two-layer depth target for multiview (`D2Array`, [`XR_VIEW_COUNT`] layers).
///
/// Returns [`None`] when `limits` cannot accommodate the requested extent or
/// [`XR_VIEW_COUNT`] array layers; callers fall back to skipping stereo depth allocation.
pub fn create_stereo_depth_texture(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    extent: (u32, u32),
) -> Option<(wgpu::Texture, wgpu::TextureView)> {
    let w = extent.0.max(1);
    let h = extent.1.max(1);
    if !limits.texture_2d_fits(w, h) {
        logger::warn!(
            "xr stereo depth: extent {w}x{h} exceeds max_texture_dimension_2d={}; skipping",
            limits.max_texture_dimension_2d()
        );
        return None;
    }
    if !limits.array_layers_fit(XR_VIEW_COUNT) {
        logger::warn!(
            "xr stereo depth: requires {XR_VIEW_COUNT} array layers but max_texture_array_layers={}; skipping",
            limits.max_texture_array_layers()
        );
        return None;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("xr_stereo_depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: XR_VIEW_COUNT,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::gpu::main_forward_depth_stencil_format(device.features()),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("xr_stereo_depth_array"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(XR_VIEW_COUNT),
        ..Default::default()
    });
    crate::profiling::note_resource_churn!(TextureView, "xr::stereo_depth_array_view");
    Some((tex, view))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapchain_create_usage_matches_import_descriptors() {
        let xr_usage = xr_swapchain_usage_flags();
        assert_eq!(xr_usage, xr::SwapchainUsageFlags::COLOR_ATTACHMENT);

        let hal_desc = xr_swapchain_hal_descriptor((2496, 2688));
        assert_eq!(hal_desc.usage, TextureUses::COLOR_TARGET);
        assert!(!hal_desc.usage.contains(TextureUses::RESOURCE));
        assert!(!hal_desc.usage.contains(TextureUses::COPY_SRC));
        assert!(!hal_desc.usage.contains(TextureUses::COPY_DST));

        let wgpu_desc = xr_swapchain_wgpu_descriptor(&hal_desc);
        assert_eq!(wgpu_desc.usage, wgpu::TextureUsages::RENDER_ATTACHMENT);
        assert!(
            !wgpu_desc
                .usage
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        assert!(!wgpu_desc.usage.contains(wgpu::TextureUsages::COPY_SRC));
        assert!(!wgpu_desc.usage.contains(wgpu::TextureUsages::COPY_DST));
    }
}
