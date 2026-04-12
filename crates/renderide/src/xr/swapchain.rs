//! OpenXR stereo swapchain images imported as wgpu [`wgpu::Texture`] / array [`wgpu::TextureView`].

use ash::vk::{self, Handle};
use openxr as xr;
use thiserror::Error;
use wgpu::hal::api::Vulkan as HalVulkan;
use wgpu::hal::{self, MemoryFlags};
use wgpu::TextureUses;

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
    /// `create_texture_from_hal` failed validation.
    #[error("wgpu: {0}")]
    Wgpu(String),
}

/// OpenXR swapchain plus one wgpu texture + D2Array view per swapchain image.
pub struct XrStereoSwapchain {
    /// Runtime swapchain handle (acquire / release / composition).
    pub handle: xr::Swapchain<xr::Vulkan>,
    /// Per-eye rectangle size in pixels.
    pub resolution: (u32, u32),
    /// One entry per swapchain buffer index.
    pub wgpu_buffers: Vec<(wgpu::Texture, wgpu::TextureView)>,
}

impl XrStereoSwapchain {
    /// Creates an OpenXR swapchain and imports each Vulkan image into wgpu.
    ///
    /// # Safety
    ///
    /// `device` must be the same Vulkan `VkDevice` used to create the OpenXR session.
    pub unsafe fn new(
        session: &xr::Session<xr::Vulkan>,
        xr_instance: &xr::Instance,
        system_id: xr::SystemId,
        device: &wgpu::Device,
    ) -> Result<Self, XrSwapchainError> {
        let views = xr_instance.enumerate_view_configuration_views(
            system_id,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let v0 = views.first().ok_or(XrSwapchainError::NoViewConfiguration)?;
        let resolution = (
            v0.recommended_image_rect_width,
            v0.recommended_image_rect_height,
        );

        let handle = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED,
            format: XR_VK_FORMAT.as_raw() as u32,
            sample_count: 1,
            width: resolution.0,
            height: resolution.1,
            face_count: 1,
            array_size: XR_VIEW_COUNT,
            mip_count: 1,
        })?;

        let images = handle.enumerate_images()?;
        let hal_device = device
            .as_hal::<HalVulkan>()
            .ok_or(XrSwapchainError::NotVulkanHal)?;

        let mut wgpu_buffers = Vec::with_capacity(images.len());

        for vk_handle in images {
            let vk_image = vk::Image::from_raw(vk_handle);
            let hal_desc = hal::TextureDescriptor {
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
                usage: TextureUses::COLOR_TARGET | TextureUses::COPY_DST | TextureUses::RESOURCE,
                memory_flags: MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            let hal_tex = unsafe {
                hal_device.texture_from_raw(
                    vk_image,
                    &hal_desc,
                    None,
                    hal::vulkan::TextureMemory::External,
                )
            };
            let wgpu_desc = wgpu::TextureDescriptor {
                label: Some("xr_swapchain"),
                size: hal_desc.size,
                mip_level_count: hal_desc.mip_level_count,
                sample_count: hal_desc.sample_count,
                dimension: hal_desc.dimension,
                format: XR_COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            let texture =
                unsafe { device.create_texture_from_hal::<HalVulkan>(hal_tex, &wgpu_desc) };
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("xr_swapchain_array"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                array_layer_count: Some(XR_VIEW_COUNT),
                ..Default::default()
            });
            wgpu_buffers.push((texture, view));
        }

        Ok(Self {
            handle,
            resolution,
            wgpu_buffers,
        })
    }

    /// `wgpu` color array view for a swapchain image index from [`xr::Swapchain::acquire_image`].
    pub fn color_view_for_image(&self, image_index: usize) -> Option<&wgpu::TextureView> {
        self.wgpu_buffers.get(image_index).map(|(_, v)| v)
    }

    /// Backing [`wgpu::Texture`] for a swapchain image index (array texture, two layers).
    pub fn color_texture_for_image(&self, image_index: usize) -> Option<&wgpu::Texture> {
        self.wgpu_buffers.get(image_index).map(|(t, _)| t)
    }

    /// Single-eye [`wgpu::TextureView`] (`D2`) for sampling one layer of the acquired swapchain image.
    ///
    /// `layer` must be `<` [`XR_VIEW_COUNT`] (0 = left, 1 = right).
    pub fn color_layer_view_for_image(
        &self,
        image_index: usize,
        layer: u32,
    ) -> Option<wgpu::TextureView> {
        let (tex, _) = self.wgpu_buffers.get(image_index)?;
        if layer >= XR_VIEW_COUNT {
            return None;
        }
        Some(tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("xr_swapchain_layer"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        }))
    }
}

/// Two-layer depth target for multiview (`D2Array`, [`XR_VIEW_COUNT`] layers).
pub fn create_stereo_depth_texture(
    device: &wgpu::Device,
    extent: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let w = extent.0.max(1);
    let h = extent.1.max(1);
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
        format: wgpu::TextureFormat::Depth32Float,
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
    (tex, view)
}
