//! GPU-resident [`SetTexture3DFormat`](crate::shared::SetTexture3DFormat) pool ([`GpuTexture3d`]) with VRAM accounting.

use std::sync::Arc;

use crate::assets::texture::{estimate_gpu_texture3d_bytes, resolve_texture3d_wgpu_format};
use crate::gpu::GpuLimits;
use crate::shared::{
    ColorProfile, SetTexture3DFormat, SetTexture3DProperties, TextureFilterMode, TextureFormat,
    TextureWrapMode,
};

use super::GpuResource;
use super::budget::TextureResidencyMeta;
use super::resource_pool::{GpuResourcePool, TexturePoolAccess, impl_texture_pool_facade};
use super::texture_allocation::{
    SampledTextureAllocation, TextureViewInit, create_sampled_copy_dst_texture,
};

/// Sampler-related fields mirrored from [`SetTexture3DProperties`](crate::shared::SetTexture3DProperties).
#[derive(Clone, Debug)]
pub struct Texture3dSamplerState {
    /// Min/mag filter from host.
    pub filter_mode: TextureFilterMode,
    /// Anisotropic filtering level (host units).
    pub aniso_level: i32,
    /// U address mode.
    pub wrap_u: TextureWrapMode,
    /// V address mode.
    pub wrap_v: TextureWrapMode,
    /// W address mode (depth).
    pub wrap_w: TextureWrapMode,
    /// Mip bias applied when sampling.
    pub mipmap_bias: f32,
}

impl Default for Texture3dSamplerState {
    fn default() -> Self {
        Self {
            filter_mode: TextureFilterMode::default(),
            aniso_level: 1,
            wrap_u: TextureWrapMode::default(),
            wrap_v: TextureWrapMode::default(),
            wrap_w: TextureWrapMode::default(),
            mipmap_bias: 0.0,
        }
    }
}

impl Texture3dSamplerState {
    /// Copies fields from host properties.
    pub fn from_props(props: Option<&SetTexture3DProperties>) -> Self {
        let Some(p) = props else {
            return Self::default();
        };
        Self {
            filter_mode: p.filter_mode,
            aniso_level: p.aniso_level,
            wrap_u: p.wrap_u,
            wrap_v: p.wrap_v,
            wrap_w: p.wrap_w,
            mipmap_bias: 0.0,
        }
    }
}

/// GPU Texture3D: mips live only in [`wgpu::Texture`].
#[derive(Debug)]
pub struct GpuTexture3d {
    /// Host Texture3D asset id.
    pub asset_id: i32,
    /// GPU texture storage (all mips allocated; uploads fill subsets).
    pub texture: Arc<wgpu::Texture>,
    /// Default full-mip view for binding (`TextureViewDimension::D3`).
    pub view: Arc<wgpu::TextureView>,
    /// Resolved wgpu format for `texture`.
    pub wgpu_format: wgpu::TextureFormat,
    /// Host [`TextureFormat`] enum (compression / layout family).
    pub host_format: TextureFormat,
    /// Linear vs sRGB sampling policy from host.
    pub color_profile: ColorProfile,
    /// Texture width in texels (mip0).
    pub width: u32,
    /// Texture height in texels (mip0).
    pub height: u32,
    /// Texture depth in texels (mip0).
    pub depth: u32,
    /// Mip chain length allocated on GPU.
    pub mip_levels_total: u32,
    /// Mips with authored texels uploaded so far.
    pub mip_levels_resident: u32,
    /// Estimated VRAM for allocated mips.
    pub resident_bytes: u64,
    /// Sampler fields for material bind groups.
    pub sampler: Texture3dSamplerState,
    /// Streaming / eviction hints from host properties.
    pub residency: TextureResidencyMeta,
}

impl GpuTexture3d {
    /// Allocates GPU storage for `fmt` (empty mips; data arrives via upload path).
    ///
    /// Returns [`None`] when any dimension is zero, or when any edge exceeds
    /// [`GpuLimits::max_texture_dimension_3d`].
    pub fn new_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetTexture3DFormat,
        props: Option<&SetTexture3DProperties>,
    ) -> Option<Self> {
        let w = fmt.width.max(0) as u32;
        let h = fmt.height.max(0) as u32;
        let d = fmt.depth.max(0) as u32;
        if w == 0 || h == 0 || d == 0 {
            return None;
        }
        let max_dim = limits.max_texture_dimension_3d();
        if w > max_dim || h > max_dim || d > max_dim {
            logger::warn!(
                "texture3d {}: format size {}×{}×{} exceeds max_texture_dimension_3d ({max_dim}); GPU texture not created",
                fmt.asset_id,
                w,
                h,
                d
            );
            return None;
        }
        let mips = fmt.mipmap_count.max(1) as u32;
        let wgpu_format = resolve_texture3d_wgpu_format(device, fmt);
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: d,
        };
        let texture_label = format!("Texture3D {}", fmt.asset_id);
        let view_label = format!("Texture3D {} view", fmt.asset_id);
        let (texture, view) = create_sampled_copy_dst_texture(
            device,
            SampledTextureAllocation {
                label: &texture_label,
                size,
                mip_level_count: mips,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu_format,
                view: TextureViewInit {
                    label: Some(&view_label),
                    dimension: Some(wgpu::TextureViewDimension::D3),
                },
            },
        );
        let resident_bytes = estimate_gpu_texture3d_bytes(wgpu_format, w, h, d, mips);
        let sampler = Texture3dSamplerState::from_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_texture3d_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture,
            view,
            wgpu_format,
            host_format: fmt.format,
            color_profile: fmt.profile,
            width: w,
            height: h,
            depth: d,
            mip_levels_total: mips,
            mip_levels_resident: 0,
            resident_bytes,
            sampler,
            residency,
        })
    }

    /// Updates sampler fields and residency hints from host properties.
    pub fn apply_properties(&mut self, p: &SetTexture3DProperties) {
        self.sampler = Texture3dSamplerState::from_props(Some(p));
        self.residency = TextureResidencyMeta::from_texture3d_props(p);
    }
}

impl GpuResource for GpuTexture3d {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

/// Resident Texture3D table; pairs with [`super::TexturePool`] under one renderer.
pub struct Texture3dPool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuTexture3d, TexturePoolAccess>,
}

impl_texture_pool_facade!(Texture3dPool, GpuTexture3d);
