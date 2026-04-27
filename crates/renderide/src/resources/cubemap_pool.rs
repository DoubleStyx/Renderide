//! GPU-resident [`SetCubemapFormat`](crate::shared::SetCubemapFormat) pool ([`GpuCubemap`]) with VRAM accounting.

use std::collections::HashMap;
use std::sync::Arc;

use crate::assets::texture::{estimate_gpu_cubemap_bytes, resolve_cubemap_wgpu_format};
use crate::gpu::GpuLimits;
use crate::shared::{
    ColorProfile, SetCubemapFormat, SetCubemapProperties, TextureFilterMode, TextureFormat,
    TextureWrapMode,
};

use super::budget::{TextureResidencyMeta, VramAccounting, VramResourceKind};
use super::{GpuResource, StreamingPolicy};

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
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Cubemap {}", fmt.asset_id)),
            size,
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("Cubemap {} cube view", fmt.asset_id)),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let resident_bytes = estimate_gpu_cubemap_bytes(wgpu_format, s, mips);
        let sampler = CubemapSamplerState::from_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_cubemap_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture: Arc::new(texture),
            view: Arc::new(view),
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
    textures: HashMap<i32, GpuCubemap>,
    accounting: VramAccounting,
    streaming: Box<dyn StreamingPolicy>,
}

impl CubemapPool {
    /// Creates an empty pool with the given streaming policy.
    pub fn new(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self {
            textures: HashMap::new(),
            accounting: VramAccounting::default(),
            streaming,
        }
    }

    /// Default pool with [`crate::resources::NoopStreamingPolicy`].
    pub fn default_pool() -> Self {
        Self::new(Box::new(super::NoopStreamingPolicy))
    }

    /// VRAM accounting for resident textures.
    pub fn accounting(&self) -> &VramAccounting {
        &self.accounting
    }

    /// Mutable VRAM totals (insert/remove update accounting).
    pub fn accounting_mut(&mut self) -> &mut VramAccounting {
        &mut self.accounting
    }

    /// Streaming policy for mip eviction suggestions.
    pub fn streaming_mut(&mut self) -> &mut dyn StreamingPolicy {
        self.streaming.as_mut()
    }

    /// Inserts or replaces a cubemap. Returns `true` if a previous entry was replaced.
    pub fn insert_texture(&mut self, tex: GpuCubemap) -> bool {
        let id = tex.asset_id;
        let existed_before = self.textures.contains_key(&id);
        let bytes = tex.resident_bytes;
        if let Some(old) = self.textures.insert(id, tex) {
            self.accounting
                .on_resident_removed(VramResourceKind::Texture, old.resident_bytes);
        }
        self.accounting
            .on_resident_added(VramResourceKind::Texture, bytes);
        self.streaming.note_texture_access(id);
        existed_before
    }

    /// Removes a cubemap by host id; returns `true` if it was present.
    pub fn remove_texture(&mut self, asset_id: i32) -> bool {
        if let Some(old) = self.textures.remove(&asset_id) {
            self.accounting
                .on_resident_removed(VramResourceKind::Texture, old.resident_bytes);
            return true;
        }
        false
    }

    /// Borrows a resident cubemap by host asset id.
    #[inline]
    pub fn get_texture(&self, asset_id: i32) -> Option<&GpuCubemap> {
        self.textures.get(&asset_id)
    }

    /// Mutably borrows a resident cubemap (mip uploads, property changes).
    #[inline]
    pub fn get_texture_mut(&mut self, asset_id: i32) -> Option<&mut GpuCubemap> {
        self.textures.get_mut(&asset_id)
    }

    /// Full map for iteration and HUD stats.
    #[inline]
    pub fn textures(&self) -> &HashMap<i32, GpuCubemap> {
        &self.textures
    }

    /// Number of resident cubemap entries in the pool.
    #[inline]
    pub fn resident_texture_count(&self) -> usize {
        self.textures.len()
    }
}
