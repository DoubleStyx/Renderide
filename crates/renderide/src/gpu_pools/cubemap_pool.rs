//! GPU-resident [`SetCubemapFormat`](crate::shared::SetCubemapFormat) pool ([`GpuCubemap`]) with VRAM accounting.

use std::sync::Arc;

use crate::assets::texture::{estimate_gpu_cubemap_bytes, resolve_cubemap_wgpu_format};
use crate::gpu::GpuLimits;
use crate::shared::{
    ColorProfile, SetCubemapFormat, SetCubemapProperties, TextureFilterMode, TextureFormat,
    TextureWrapMode,
};

use super::GpuResource;
use super::budget::TextureResidencyMeta;
use super::resource_pool::{GpuResourcePool, TexturePoolAccess, impl_texture_pool_facade};
use super::texture_allocation::{
    SampledTextureAllocation, TextureViewInit, create_sampled_copy_dst_texture,
};

/// Sampler-related fields mirrored from [`SetCubemapProperties`](crate::shared::SetCubemapProperties).
#[derive(Clone, Debug)]
pub struct CubemapSamplerState {
    /// Min/mag filter from host.
    pub filter_mode: TextureFilterMode,
    /// Anisotropic filtering level (host units).
    pub aniso_level: i32,
    /// Mip bias applied when sampling.
    pub mipmap_bias: f32,
    /// Default U address mode (repeat; host cubemap properties do not carry wrap).
    pub wrap_u: TextureWrapMode,
    /// Default V address mode (repeat).
    pub wrap_v: TextureWrapMode,
}

impl Default for CubemapSamplerState {
    fn default() -> Self {
        Self {
            filter_mode: TextureFilterMode::default(),
            aniso_level: 1,
            mipmap_bias: 0.0,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Repeat,
        }
    }
}

impl CubemapSamplerState {
    /// Copies fields from host properties.
    pub fn from_props(props: Option<&SetCubemapProperties>) -> Self {
        let Some(p) = props else {
            return Self::default();
        };
        Self {
            filter_mode: p.filter_mode,
            aniso_level: p.aniso_level,
            mipmap_bias: p.mipmap_bias,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Repeat,
        }
    }
}

/// GPU cubemap: six faces in one array texture (`TextureViewDimension::Cube`).
#[derive(Debug)]
pub struct GpuCubemap {
    /// Host cubemap asset id.
    pub asset_id: i32,
    /// GPU texture storage (all mips allocated; uploads fill subsets).
    pub texture: Arc<wgpu::Texture>,
    /// Default full-mip cube view for binding.
    pub view: Arc<wgpu::TextureView>,
    /// Resolved wgpu format for `texture`.
    pub wgpu_format: wgpu::TextureFormat,
    /// Host [`TextureFormat`] enum (compression / layout family).
    pub host_format: TextureFormat,
    /// Linear vs sRGB sampling policy from host.
    pub color_profile: ColorProfile,
    /// Face size in texels (mip0).
    pub size: u32,
    /// Mip chain length allocated on GPU.
    pub mip_levels_total: u32,
    /// Mips with authored texels uploaded so far.
    pub mip_levels_resident: u32,
    /// Whether native compressed face bytes were left in host V orientation and need sampling compensation.
    pub storage_v_inverted: bool,
    /// Estimated VRAM for allocated mips.
    pub resident_bytes: u64,
    /// Sampler fields for material bind groups.
    pub sampler: CubemapSamplerState,
    /// Streaming / eviction hints from host properties.
    pub residency: TextureResidencyMeta,
}

impl GpuCubemap {
    /// Allocates GPU storage for `fmt` (empty mips; data arrives via upload path).
    ///
    /// Returns [`None`] when `size` is zero, when the edge exceeds `max_texture_dimension_2d`, or
    /// when `max_texture_array_layers` is below six (cubemap faces).
    pub fn new_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetCubemapFormat,
        props: Option<&SetCubemapProperties>,
    ) -> Option<Self> {
        let s = fmt.size.max(0) as u32;
        if s == 0 {
            return None;
        }
        let max_dim = limits.max_texture_dimension_2d();
        if s > max_dim {
            logger::warn!(
                "cubemap {}: face size {} exceeds max_texture_dimension_2d ({max_dim}); GPU texture not created",
                fmt.asset_id,
                s
            );
            return None;
        }
        if !limits.cubemap_fits_texture_array_layers() {
            let max_layers = limits.max_texture_array_layers();
            logger::warn!(
                "cubemap {}: max_texture_array_layers ({max_layers}) < {}; GPU texture not created",
                fmt.asset_id,
                crate::gpu::CUBEMAP_ARRAY_LAYERS
            );
            return None;
        }
        let mips = fmt.mipmap_count.max(1) as u32;
        let wgpu_format = resolve_cubemap_wgpu_format(device, fmt);
        let size = wgpu::Extent3d {
            width: s,
            height: s,
            depth_or_array_layers: 6,
        };
        let texture_label = format!("Cubemap {}", fmt.asset_id);
        let view_label = format!("Cubemap {} cube view", fmt.asset_id);
        let (texture, view) = create_sampled_copy_dst_texture(
            device,
            SampledTextureAllocation {
                label: &texture_label,
                size,
                mip_level_count: mips,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                view: TextureViewInit {
                    label: Some(&view_label),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                },
            },
        );
        let resident_bytes = estimate_gpu_cubemap_bytes(wgpu_format, s, mips);
        let sampler = CubemapSamplerState::from_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_cubemap_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture,
            view,
            wgpu_format,
            host_format: fmt.format,
            color_profile: fmt.profile,
            size: s,
            mip_levels_total: mips,
            mip_levels_resident: 0,
            storage_v_inverted: false,
            resident_bytes,
            sampler,
            residency,
        })
    }

    /// Updates sampler fields and residency hints from host properties.
    pub fn apply_properties(&mut self, p: &SetCubemapProperties) {
        self.sampler = CubemapSamplerState::from_props(Some(p));
        self.residency = TextureResidencyMeta::from_cubemap_props(p);
    }
}

impl GpuResource for GpuCubemap {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

/// Resident cubemap table.
pub struct CubemapPool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuCubemap, TexturePoolAccess>,
}

impl_texture_pool_facade!(CubemapPool, GpuCubemap);
