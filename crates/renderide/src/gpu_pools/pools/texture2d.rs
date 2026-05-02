//! GPU-resident Texture2D pool ([`GpuTexture2d`]) with VRAM accounting.

use std::sync::Arc;

use crate::assets::texture::{estimate_gpu_texture_bytes, resolve_texture2d_wgpu_format};
use crate::gpu::GpuLimits;
use crate::shared::{ColorProfile, SetTexture2DFormat, SetTexture2DProperties, TextureFormat};

use crate::gpu_pools::GpuResource;
use crate::gpu_pools::budget::TextureResidencyMeta;
use crate::gpu_pools::resource_pool::{
    GpuResourcePool, StreamingAccess, impl_streaming_pool_facade,
};
use crate::gpu_pools::sampler_state::SamplerState;
use crate::gpu_pools::texture_allocation::{
    SampledTextureAllocation, TextureViewInit, create_sampled_copy_dst_texture,
};

/// GPU Texture2D: no CPU mip storage; mips live only in [`wgpu::Texture`].
///
/// **`mip_levels_resident`** tracks how many mips currently hold uploaded or synthesized texels. A future
/// streaming pass may reduce resident mips under [`crate::gpu_pools::StreamingPolicy`] (evict fine
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
    /// Whether native compressed bytes were left in host V orientation and need sampling compensation.
    pub storage_v_inverted: bool,
    /// Uploaded mip-level bitset; [`Self::mip_levels_resident`] is the contiguous prefix from mip 0.
    resident_mip_mask: u64,
    /// Estimated VRAM for allocated mips.
    pub resident_bytes: u64,
    /// Sampler fields for material bind groups.
    pub sampler: SamplerState,
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
                "texture {}: format size {}x{} exceeds max_texture_dimension_2d ({max_dim}); GPU texture not created",
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
        let label = format!("Texture2D {}", fmt.asset_id);
        let (texture, view) = create_sampled_copy_dst_texture(
            device,
            SampledTextureAllocation {
                label: &label,
                size,
                mip_level_count: mips,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                view: TextureViewInit {
                    label: None,
                    dimension: None,
                },
            },
        );
        let resident_bytes = estimate_gpu_texture_bytes(wgpu_format, w, h, mips);
        let sampler = SamplerState::from_texture2d_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_host_props)
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
            mip_levels_total: mips,
            mip_levels_resident: 0,
            storage_v_inverted: false,
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
        self.sampler = SamplerState::from_texture2d_props(Some(p));
        self.residency = TextureResidencyMeta::from_host_props(p);
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
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuTexture2d, StreamingAccess>,
}

impl_streaming_pool_facade!(
    TexturePool,
    GpuTexture2d,
    StreamingAccess::texture,
    StreamingAccess::texture_noop,
);

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
