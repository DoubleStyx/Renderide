//! GPU-resident Texture2D pool ([`GpuTexture2d`]) with VRAM accounting.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::assets::texture::{
    estimate_gpu_texture_bytes, host_texture_mip_count, legal_texture2d_mip_level_count,
    resolve_texture2d_wgpu_format,
};
use crate::gpu::GpuLimits;
use crate::shared::{ColorProfile, SetTexture2DFormat, SetTexture2DProperties, TextureFormat};

use crate::gpu_pools::budget::TextureResidencyMeta;
use crate::gpu_pools::impl_gpu_resource;
use crate::gpu_pools::resource_pool::{
    GpuResourcePool, StreamingAccess, impl_streaming_pool_facade,
};
use crate::gpu_pools::sampler_state::SamplerState;
use crate::gpu_pools::texture_allocation::{
    SampledTextureAllocation, TextureViewInit, clamp_texture_mip_count,
    create_sampled_copy_dst_texture, validate_texture_extent,
};

static NEXT_TEXTURE2D_VIEW_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Texture2dAllocationDesc {
    width: u32,
    height: u32,
    mip_levels_total: u32,
    wgpu_format: wgpu::TextureFormat,
}

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
    /// Monotonic identifier for the current texture view allocation.
    pub view_generation: u64,
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
    /// Monotonic generation bumped whenever this texture's GPU texel contents are uploaded.
    pub content_generation: u64,
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
        let desc = Self::allocation_desc_from_format(device, limits, fmt)?;
        let size = wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: 1,
        };
        let label = format!("Texture2D {}", fmt.asset_id);
        let (texture, view) = create_sampled_copy_dst_texture(
            device,
            SampledTextureAllocation {
                label: &label,
                size,
                mip_level_count: desc.mip_levels_total,
                dimension: wgpu::TextureDimension::D2,
                format: desc.wgpu_format,
                view: TextureViewInit {
                    label: None,
                    dimension: None,
                },
            },
        );
        let resident_bytes = estimate_gpu_texture_bytes(
            desc.wgpu_format,
            desc.width,
            desc.height,
            desc.mip_levels_total,
        );
        let sampler = SamplerState::from_texture2d_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture,
            view,
            view_generation: NEXT_TEXTURE2D_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed),
            wgpu_format: desc.wgpu_format,
            host_format: fmt.format,
            color_profile: fmt.profile,
            width: desc.width,
            height: desc.height,
            mip_levels_total: desc.mip_levels_total,
            mip_levels_resident: 0,
            content_generation: 0,
            storage_v_inverted: false,
            resident_mip_mask: 0,
            resident_bytes,
            sampler,
            residency,
        })
    }

    /// Returns `true` when `fmt` resolves to this texture's current GPU allocation shape.
    pub(crate) fn allocation_matches_format(
        &self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetTexture2DFormat,
    ) -> bool {
        Self::allocation_desc_from_format(device, limits, fmt)
            .is_some_and(|desc| self.allocation_matches_desc(desc))
    }

    /// Updates format metadata without changing the GPU allocation or resident mip state.
    pub(crate) fn apply_format_metadata(
        &mut self,
        fmt: &SetTexture2DFormat,
        props: Option<&SetTexture2DProperties>,
    ) {
        self.host_format = fmt.format;
        self.color_profile = fmt.profile;
        self.sampler = SamplerState::from_texture2d_props(props);
        self.residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
    }

    fn allocation_desc_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetTexture2DFormat,
    ) -> Option<Texture2dAllocationDesc> {
        let wgpu_format = resolve_texture2d_wgpu_format(device, fmt);
        texture2d_allocation_desc(limits, fmt, wgpu_format)
    }

    fn allocation_matches_desc(&self, desc: Texture2dAllocationDesc) -> bool {
        self.width == desc.width
            && self.height == desc.height
            && self.mip_levels_total == desc.mip_levels_total
            && self.wgpu_format == desc.wgpu_format
    }

    /// Marks uploaded mip levels and updates the contiguous resident prefix used for sampler LOD clamps.
    pub fn mark_mips_resident(&mut self, start_mip: u32, uploaded_mips: u32) {
        if uploaded_mips == 0 {
            return;
        }
        self.mip_levels_resident = mark_resident_mip_mask(
            &mut self.resident_mip_mask,
            self.mip_levels_total,
            start_mip,
            uploaded_mips,
        );
    }

    /// Marks that a completed upload changed this texture's GPU contents.
    pub fn mark_content_uploaded(&mut self) {
        self.content_generation = self.content_generation.wrapping_add(1).max(1);
    }

    /// Updates sampler fields and residency hints from host properties.
    pub fn apply_properties(&mut self, p: &SetTexture2DProperties) {
        self.sampler = SamplerState::from_texture2d_props(Some(p));
        self.residency = TextureResidencyMeta::from_host_props(p);
    }
}

impl_gpu_resource!(GpuTexture2d);

fn texture2d_allocation_desc(
    limits: &GpuLimits,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
) -> Option<Texture2dAllocationDesc> {
    let w = fmt.width.max(0) as u32;
    let h = fmt.height.max(0) as u32;
    if w == 0 || h == 0 {
        return None;
    }
    let max_dim = limits.max_texture_dimension_2d();
    if !validate_texture_extent(
        fmt.asset_id,
        "texture",
        "format size",
        &format_args!("{w}x{h}"),
        &[w, h],
        max_dim,
        "max_texture_dimension_2d",
    ) {
        return None;
    }
    let requested_mips = host_texture_mip_count(fmt.mipmap_count);
    let legal_mips = legal_texture2d_mip_level_count(w, h);
    let mip_levels_total = clamp_texture_mip_count(
        fmt.asset_id,
        "texture",
        &format_args!("{w}x{h}"),
        requested_mips,
        legal_mips,
    );
    Some(Texture2dAllocationDesc {
        width: w,
        height: h,
        mip_levels_total,
        wgpu_format,
    })
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

impl TexturePool {
    /// Iterates resident textures for diagnostics.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &GpuTexture2d> {
        self.inner.resources().values()
    }

    /// Number of resident Texture2D entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use hashbrown::HashMap;

    use crate::gpu::GpuLimits;
    use crate::shared::{ColorProfile, SetTexture2DFormat, TextureFormat};

    use super::{
        NEXT_TEXTURE2D_VIEW_GENERATION, mark_resident_mip_mask, texture2d_allocation_desc,
    };

    fn test_limits(max_texture_dimension_2d: u32) -> GpuLimits {
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    fn format(
        width: i32,
        height: i32,
        mipmap_count: i32,
        format: TextureFormat,
        profile: ColorProfile,
    ) -> SetTexture2DFormat {
        SetTexture2DFormat {
            asset_id: 7,
            width,
            height,
            mipmap_count,
            format,
            profile,
        }
    }

    #[test]
    fn texture_view_generation_is_unique() {
        let first = NEXT_TEXTURE2D_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed);
        let second = NEXT_TEXTURE2D_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed);
        assert_ne!(first, second);
    }

    #[test]
    fn allocation_desc_reuses_same_gpu_shape() {
        let limits = test_limits(4096);
        let base = texture2d_allocation_desc(
            &limits,
            &format(64, 32, 4, TextureFormat::RGBA32, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("base allocation");
        let same_storage = texture2d_allocation_desc(
            &limits,
            &format(64, 32, 4, TextureFormat::RGB24, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("same allocation");

        assert_eq!(base, same_storage);
    }

    #[test]
    fn allocation_desc_changes_for_size_mips_or_storage() {
        let limits = test_limits(4096);
        let base = texture2d_allocation_desc(
            &limits,
            &format(64, 32, 4, TextureFormat::RGBA32, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("base allocation");

        assert_ne!(
            base,
            texture2d_allocation_desc(
                &limits,
                &format(128, 32, 4, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("different width")
        );
        assert_ne!(
            base,
            texture2d_allocation_desc(
                &limits,
                &format(64, 32, 2, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("different mips")
        );
        assert_ne!(
            base,
            texture2d_allocation_desc(
                &limits,
                &format(64, 32, 4, TextureFormat::RGBA32, ColorProfile::SRGB),
                wgpu::TextureFormat::Rgba8UnormSrgb,
            )
            .expect("different storage")
        );
    }

    #[test]
    fn allocation_desc_rejects_invalid_size() {
        let limits = test_limits(64);

        assert!(
            texture2d_allocation_desc(
                &limits,
                &format(0, 32, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
        assert!(
            texture2d_allocation_desc(
                &limits,
                &format(128, 32, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
    }

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
