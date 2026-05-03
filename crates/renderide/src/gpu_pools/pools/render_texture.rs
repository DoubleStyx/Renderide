//! GPU render targets for host [`crate::shared::SetRenderTextureFormat`] (Unity `RenderTexture` assets).
//!
//! Color textures use `RENDER_ATTACHMENT | TEXTURE_BINDING` so the same asset can be sampled from
//! materials after the offscreen pass. Depth buffers are separate textures when `depth > 0`; depth
//! also includes `COPY_SRC` so [`crate::backend::frame_gpu::FrameGpuResources::copy_scene_depth_snapshot`]
//! can copy scene depth for intersection / frame bindings (same as main `renderide-depth`).
//!
//! ### Orientation
//!
//! Render textures are sampled through the same material UV path as host-uploaded textures.
//! The renderer does not rewrite material `_ST` values based on texture asset kind; tiling and
//! offset remain authored material data, and shader sampling helpers apply the renderer-wide
//! texture-origin convention uniformly.
//!
//! ### Wrap policy
//!
//! Render-texture sampler state intentionally mirrors the host's U/V wrap modes. Camera-preview
//! targets that must avoid opposite-edge bleed should be created with
//! [`crate::shared::TextureWrapMode::Clamp`] by the host; the renderer does not infer clamp from
//! "is written by a camera" because repeat is valid authorable behavior for sampled render
//! textures.

use std::sync::Arc;

use crate::assets::texture::estimate_gpu_texture_bytes;
use crate::gpu::GpuLimits;
use crate::gpu_pools::GpuResource;
use crate::gpu_pools::VramResourceKind;
use crate::gpu_pools::resource_pool::{
    GpuResourcePool, UntrackedAccess, impl_resident_pool_facade,
};
use crate::gpu_pools::sampler_state::SamplerState;
use crate::shared::SetRenderTextureFormat;

/// Host render texture mirrored as a wgpu color target + optional depth.
#[derive(Debug)]
pub struct GpuRenderTexture {
    /// Host render-texture asset id.
    pub asset_id: i32,
    /// Color target (`Rgba16Float`); sampleable after offscreen draws.
    pub color_texture: Arc<wgpu::Texture>,
    /// Default view over the full color mip.
    pub color_view: Arc<wgpu::TextureView>,
    /// Optional depth texture (always allocated for scene draws in [`Self::new_from_format`]).
    pub depth_texture: Option<Arc<wgpu::Texture>>,
    /// View over `depth_texture` when present.
    pub depth_view: Option<Arc<wgpu::TextureView>>,
    /// wgpu format of `color_texture`.
    pub wgpu_color_format: wgpu::TextureFormat,
    /// Pixel width of the render target.
    pub width: u32,
    /// Pixel height of the render target.
    pub height: u32,
    /// Estimated VRAM for color + depth.
    pub resident_bytes: u64,
    /// Sampler state mirrored from host format for material binds.
    pub sampler: SamplerState,
}

impl GpuResource for GpuRenderTexture {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

impl GpuRenderTexture {
    /// Creates GPU storage for a host [`SetRenderTextureFormat`].
    ///
    /// Color format: **`Rgba16Float`** when `hdr_color` (Unity `ARGBHalf` / HDR parity), else
    /// **`Rgba8Unorm`** for lower VRAM on typical LDR targets. Depth is always [`Depth32Float`]
    /// (or the device-preferred depth/stencil chosen by [`crate::gpu::main_forward_depth_stencil_format`]).
    /// Size is clamped per edge via [`GpuLimits::clamp_render_texture_edge`].
    pub fn new_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetRenderTextureFormat,
        hdr_color: bool,
    ) -> Option<Self> {
        let w = limits.clamp_render_texture_edge(fmt.size.x);
        let h = limits.clamp_render_texture_edge(fmt.size.y);
        if w == 0 || h == 0 {
            return None;
        }
        let max_dim = limits.max_texture_dimension_2d();
        if w > max_dim || h > max_dim {
            logger::warn!(
                "render texture {}: size {}x{} exceeds max_texture_dimension_2d ({max_dim})",
                fmt.asset_id,
                w,
                h
            );
            return None;
        }

        let wgpu_color_format = if hdr_color {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let color_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("RenderTexture {}", fmt.asset_id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let color_view =
            Arc::new(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // Host `depth` is Unity depth-stencil bits; when zero the asset may still be used as a full
        // scene target -- we always allocate a depth attachment so the forward pass can run.
        // `TEXTURE_BINDING` is required so Hi-Z build can bind the depth view for mip0 (`hi_z_mip0_d_bg`).
        let depth_format = crate::gpu::main_forward_depth_stencil_format(device.features());
        let dt = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("RenderTextureDepth {}", fmt.asset_id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let dv = Arc::new(dt.create_view(&wgpu::TextureViewDescriptor::default()));
        let depth_texture = Some(dt);
        let depth_view = Some(dv);

        let color_bytes = estimate_gpu_texture_bytes(wgpu_color_format, w, h, 1);
        let depth_bytes = estimate_gpu_texture_bytes(depth_format, w, h, 1);
        let resident_bytes = color_bytes.saturating_add(depth_bytes);

        let sampler = SamplerState::from_render_texture_format(fmt);

        Some(Self {
            asset_id: fmt.asset_id,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            wgpu_color_format,
            width: w,
            height: h,
            resident_bytes,
            sampler,
        })
    }

    /// `true` when the color target exists and can be sampled (always after successful creation).
    #[inline]
    pub fn is_sampleable(&self) -> bool {
        true
    }
}

/// Pool of [`GpuRenderTexture`] entries keyed by host asset id.
#[derive(Debug)]
pub struct RenderTexturePool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuRenderTexture, UntrackedAccess>,
}

impl_resident_pool_facade!(
    RenderTexturePool,
    GpuRenderTexture,
    VramResourceKind::Texture,
);

#[cfg(test)]
mod tests {
    //! Unit tests for host-driven render-texture sampler metadata.

    use crate::gpu_pools::sampler_state::SamplerState;
    use crate::shared::{SetRenderTextureFormat, TextureFilterMode, TextureWrapMode};
    use glam::IVec2;

    /// Builds a format row with the supplied wrap modes.
    fn render_texture_format(
        wrap_u: TextureWrapMode,
        wrap_v: TextureWrapMode,
    ) -> SetRenderTextureFormat {
        SetRenderTextureFormat {
            asset_id: 42,
            size: IVec2::new(128, 64),
            depth: 24,
            filter_mode: TextureFilterMode::Bilinear,
            aniso_level: 8,
            wrap_u,
            wrap_v,
        }
    }

    /// Render textures must preserve the host's U/V wrap modes instead of renderer-forcing clamp.
    #[test]
    fn sampler_state_preserves_host_wrap_modes() {
        let fmt = render_texture_format(TextureWrapMode::Mirror, TextureWrapMode::Clamp);
        let sampler = SamplerState::from_render_texture_format(&fmt);

        assert_eq!(sampler.wrap_u, TextureWrapMode::Mirror);
        assert_eq!(sampler.wrap_v, TextureWrapMode::Clamp);
    }

    /// Explicit repeat stays repeat so authorable render-texture tiling keeps working.
    #[test]
    fn sampler_state_preserves_explicit_repeat() {
        let fmt = render_texture_format(TextureWrapMode::Repeat, TextureWrapMode::Repeat);
        let sampler = SamplerState::from_render_texture_format(&fmt);

        assert_eq!(sampler.wrap_u, TextureWrapMode::Repeat);
        assert_eq!(sampler.wrap_v, TextureWrapMode::Repeat);
    }

    /// Negative anisotropy values from the host are clamped before sampler creation.
    #[test]
    fn sampler_state_clamps_negative_anisotropy() {
        let mut fmt = render_texture_format(TextureWrapMode::Clamp, TextureWrapMode::Clamp);
        fmt.aniso_level = -4;

        let sampler = SamplerState::from_render_texture_format(&fmt);

        assert_eq!(sampler.aniso_level, 0);
    }
}
