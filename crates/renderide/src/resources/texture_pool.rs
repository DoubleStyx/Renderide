//! GPU-resident Texture2D pool ([`GpuTexture2d`]) with VRAM accounting.

use hashbrown::HashMap;
use std::sync::Arc;

use crate::assets::texture::{estimate_gpu_texture_bytes, resolve_texture2d_wgpu_format};
use crate::gpu::GpuLimits;
use crate::shared::{
    ColorProfile, SetTexture2DFormat, SetTexture2DProperties, TextureFilterMode, TextureFormat,
    TextureWrapMode,
};

use super::budget::{TextureResidencyMeta, VramAccounting, VramResourceKind};
use super::{GpuResource, StreamingPolicy};

/// Sampler-related fields mirrored from [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties) for future bind groups.
#[derive(Clone, Debug)]
pub struct Texture2dSamplerState {
    /// Min/mag filter from host.
    pub filter_mode: TextureFilterMode,
    /// Anisotropic filtering level (host units).
    pub aniso_level: i32,
    /// U address mode.
    pub wrap_u: TextureWrapMode,
    /// V address mode.
    pub wrap_v: TextureWrapMode,
    /// Mip bias applied when sampling.
    pub mipmap_bias: f32,
}

impl Default for Texture2dSamplerState {
    fn default() -> Self {
        Self {
            filter_mode: TextureFilterMode::default(),
            aniso_level: 1,
            wrap_u: TextureWrapMode::default(),
            wrap_v: TextureWrapMode::default(),
            mipmap_bias: 0.0,
        }
    }
}

impl Texture2dSamplerState {
    /// Copies fields from host properties.
    pub fn from_props(props: Option<&SetTexture2DProperties>) -> Self {
        let Some(p) = props else {
            return Self::default();
        };
        Self {
            filter_mode: p.filter_mode,
            aniso_level: p.aniso_level,
            wrap_u: p.wrap_u,
            wrap_v: p.wrap_v,
            mipmap_bias: p.mipmap_bias,
        }
    }
}

/// GPU Texture2D: no CPU mip storage; mips live only in [`wgpu::Texture`].
///
/// **`mip_levels_resident`** tracks how many mips currently hold uploaded or synthesized texels. A future
/// streaming pass may reduce resident mips under [`crate::resources::StreamingPolicy`] (evict fine
/// mips, re-upload from SHM or transcode). Prefer **recreating** the `wgpu::Texture` with a lower
/// `mip_level_count` over sparse partial images until wgpu exposes true sparse textures.
#[derive(Debug)]
pub struct GpuTexture2d {
    /// Host Texture2D asset id.
    pub asset_id: i32,
    /// GPU texture storage (all mips allocated; uploads fill subsets).
    pub texture: Arc<wgpu::Texture>,
    /// Default full-mip view for binding.
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
    /// Mip chain length allocated on GPU.
    pub mip_levels_total: u32,
    /// Contiguous mips with uploaded or synthesized texels available for sampling.
    pub mip_levels_resident: u32,
    /// Uploaded mip-level bitset; [`Self::mip_levels_resident`] is the contiguous prefix from mip 0.
    resident_mip_mask: u64,
    /// Estimated VRAM for allocated mips.
    pub resident_bytes: u64,
    /// Sampler fields for future bind groups.
    pub sampler: Texture2dSamplerState,
    /// Streaming / eviction hints from host properties.
    pub residency: TextureResidencyMeta,
}

impl GpuTexture2d {
    /// Allocates GPU storage for `fmt` (empty mips; data arrives via [`crate::assets::texture::write_texture2d_mips`]).
    ///
    /// Returns [`None`] when width or height is zero, or when either edge exceeds
    /// [`GpuLimits::max_texture_dimension_2d`] (avoids wgpu validation panic).
    pub fn new_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetTexture2DFormat,
        props: Option<&SetTexture2DProperties>,
    ) -> Option<Self> {
        let w = fmt.width.max(0) as u32;
        let h = fmt.height.max(0) as u32;
        if w == 0 || h == 0 {
            return None;
        }
        let max_dim = limits.max_texture_dimension_2d();
        if w > max_dim || h > max_dim {
            logger::warn!(
                "texture {}: format size {}×{} exceeds max_texture_dimension_2d ({max_dim}); GPU texture not created",
                fmt.asset_id,
                w,
                h
            );
            return None;
        }
        let mips = fmt.mipmap_count.max(1) as u32;
        let wgpu_format = resolve_texture2d_wgpu_format(device, fmt);
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Texture2D {}", fmt.asset_id)),
            size,
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let resident_bytes = estimate_gpu_texture_bytes(wgpu_format, w, h, mips);
        let sampler = Texture2dSamplerState::from_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_texture_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture: Arc::new(texture),
            view: Arc::new(view),
            wgpu_format,
            host_format: fmt.format,
            color_profile: fmt.profile,
            width: w,
            height: h,
            mip_levels_total: mips,
            mip_levels_resident: 0,
            resident_mip_mask: 0,
            resident_bytes,
            sampler,
            residency,
        })
    }

    /// Marks uploaded mip levels and updates the contiguous resident prefix used for sampler LOD clamps.
    pub fn mark_mips_resident(&mut self, start_mip: u32, uploaded_mips: u32) {
        self.mip_levels_resident = mark_resident_mip_mask(
            &mut self.resident_mip_mask,
            self.mip_levels_total,
            start_mip,
            uploaded_mips,
        );
    }

    /// Updates sampler fields and residency hints from host properties.
    pub fn apply_properties(&mut self, p: &SetTexture2DProperties) {
        self.sampler = Texture2dSamplerState::from_props(Some(p));
        self.residency = TextureResidencyMeta::from_texture_props(p);
    }
}

impl GpuResource for GpuTexture2d {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

fn mark_resident_mip_mask(
    resident_mip_mask: &mut u64,
    mip_levels_total: u32,
    start_mip: u32,
    uploaded_mips: u32,
) -> u32 {
    if uploaded_mips == 0 || start_mip >= mip_levels_total {
        return resident_prefix_len(*resident_mip_mask, mip_levels_total);
    }

    let end = start_mip
        .saturating_add(uploaded_mips)
        .min(mip_levels_total)
        .min(64);
    for mip in start_mip.min(64)..end {
        *resident_mip_mask |= 1u64 << mip;
    }

    resident_prefix_len(*resident_mip_mask, mip_levels_total)
}

fn resident_prefix_len(resident_mip_mask: u64, mip_levels_total: u32) -> u32 {
    let mut contiguous = 0u32;
    while contiguous < mip_levels_total.min(64) && (resident_mip_mask & (1u64 << contiguous)) != 0 {
        contiguous += 1;
    }
    contiguous
}

/// Resident Texture2D table; pairs with [`super::MeshPool`] under one renderer.
pub struct TexturePool {
    textures: HashMap<i32, GpuTexture2d>,
    accounting: VramAccounting,
    streaming: Box<dyn StreamingPolicy>,
}

impl TexturePool {
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

    /// Inserts or replaces a texture. Returns `true` if a previous entry was replaced.
    pub fn insert_texture(&mut self, tex: GpuTexture2d) -> bool {
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

    /// Removes a texture by host id; returns `true` if it was present.
    pub fn remove_texture(&mut self, asset_id: i32) -> bool {
        if let Some(old) = self.textures.remove(&asset_id) {
            self.accounting
                .on_resident_removed(VramResourceKind::Texture, old.resident_bytes);
            return true;
        }
        false
    }

    /// Borrows a resident texture by host asset id.
    #[inline]
    pub fn get_texture(&self, asset_id: i32) -> Option<&GpuTexture2d> {
        self.textures.get(&asset_id)
    }

    /// Mutably borrows a resident texture (mip uploads, property changes).
    #[inline]
    pub fn get_texture_mut(&mut self, asset_id: i32) -> Option<&mut GpuTexture2d> {
        self.textures.get_mut(&asset_id)
    }

    /// Full map for iteration and HUD stats.
    #[inline]
    pub fn textures(&self) -> &HashMap<i32, GpuTexture2d> {
        &self.textures
    }

    /// Number of resident Texture2D entries in the pool.
    #[inline]
    pub fn resident_texture_count(&self) -> usize {
        self.textures.len()
    }
}

#[cfg(test)]
mod tests {
    use super::mark_resident_mip_mask;

    #[test]
    fn resident_prefix_waits_for_lower_mip_gap() {
        let mut mask = 0;
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 3, 2), 0);
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 0, 3), 5);
    }

    #[test]
    fn resident_prefix_clamps_to_total_mips() {
        let mut mask = 0;
        assert_eq!(mark_resident_mip_mask(&mut mask, 4, 0, 10), 4);
    }
}
