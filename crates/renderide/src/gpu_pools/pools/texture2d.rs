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
const MAX_GPU_TEXTURE_ALLOCATION_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Texture2dAllocationDesc {
    width: u32,
    height: u32,
    mip_levels_total: u32,
    wgpu_format: wgpu::TextureFormat,
}

/// GPU Texture2D: no CPU mip storage; mips live only in [`wgpu::Texture`].
///
/// **`mip_levels_resident`** tracks the size of the contiguous mip range currently exposed through
/// [`Self::view`]. During tail-first streaming that view initially starts at a coarse mip and grows
/// toward mip 0 as finer levels arrive. A future streaming pass may reduce resident mips under
/// [`crate::gpu_pools::StreamingPolicy`] (evict fine mips, re-upload from SHM or transcode).
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
    /// Number of contiguous uploaded or synthesized mips exposed by [`Self::view`].
    pub mip_levels_resident: u32,
    /// Monotonic generation bumped whenever this texture's GPU texel contents are uploaded.
    pub content_generation: u64,
    /// Whether native compressed bytes were left in host V orientation and need sampling compensation.
    pub storage_v_inverted: bool,
    /// First texture mip represented as LOD 0 by [`Self::view`].
    resident_mip_base: u32,
    /// Uploaded mip-level bitset used to select a fully initialized contiguous view.
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
        let resident_bytes = estimate_gpu_texture_bytes(
            desc.wgpu_format,
            desc.width,
            desc.height,
            desc.mip_levels_total,
        );
        if resident_bytes > MAX_GPU_TEXTURE_ALLOCATION_BYTES {
            logger::warn!(
                "Texture2D {} rejected: estimated resident bytes {} exceed cap {}",
                fmt.asset_id,
                resident_bytes,
                MAX_GPU_TEXTURE_ALLOCATION_BYTES
            );
            return None;
        }
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
            resident_mip_base: 0,
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

    /// Marks uploaded mip levels and clamps the binding view to a contiguous initialized range.
    ///
    /// Mips outside the selected range are allocated but may hold no texels yet. Rebasing the view
    /// to a coarse tail mip makes a texture usable before its much larger mip 0 has decoded, while
    /// still preventing minified samples from reading uninitialized levels.
    pub fn mark_mips_resident(&mut self, start_mip: u32, uploaded_mips: u32) {
        if uploaded_mips == 0 {
            return;
        }
        let (resident_base, resident_count) = mark_resident_mip_mask(
            &mut self.resident_mip_mask,
            self.mip_levels_total,
            start_mip,
            uploaded_mips,
        );
        self.apply_resident_mip_range(resident_base, resident_count);
    }

    /// Starts a new full-chain stream at its first completed physical mip.
    ///
    /// The previous upload remains sampleable while the first new mip decodes. Once that write
    /// completes, clearing the old mask prevents stale fine mips from being exposed alongside the
    /// new coarse tail. Later mips in the same upload use [`Self::mark_mips_resident`].
    pub fn begin_mip_stream(&mut self, start_mip: u32, uploaded_mips: u32) {
        if uploaded_mips == 0 {
            return;
        }
        let (resident_base, resident_count) = begin_resident_mip_mask(
            &mut self.resident_mip_mask,
            self.mip_levels_total,
            start_mip,
            uploaded_mips,
        );
        self.apply_resident_mip_range(resident_base, resident_count);
    }

    fn apply_resident_mip_range(&mut self, resident_base: u32, resident_count: u32) {
        if resident_base == self.resident_mip_base && resident_count == self.mip_levels_resident {
            return;
        }
        self.resident_mip_base = resident_base;
        self.mip_levels_resident = resident_count;
        self.refresh_resident_view();
    }

    /// Returns whether one physical texture mip has initialized texels.
    pub fn mip_is_resident(&self, mip_level: u32) -> bool {
        mip_level < self.mip_levels_total.min(64)
            && (self.resident_mip_mask & (1u64 << mip_level)) != 0
    }

    /// Rebuilds the binding view over the resident mip range and bumps the view generation.
    ///
    /// Material bind signatures already hash the view generation and the resident count, so the
    /// replacement is picked up by the same rebuild that residency changes trigger today.
    fn refresh_resident_view(&mut self) {
        let mip_level_count = self.mip_levels_resident.min(self.mip_levels_total);
        if mip_level_count == 0 {
            return;
        }
        let mip_end = self.resident_mip_base.saturating_add(mip_level_count);
        let label = format!(
            "Texture2D {} mips {}..{mip_end}",
            self.asset_id, self.resident_mip_base
        );
        self.view = Arc::new(self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&label),
            base_mip_level: self.resident_mip_base,
            mip_level_count: Some(mip_level_count),
            ..Default::default()
        }));
        crate::profiling::note_resource_churn!(TextureView, "gpu_pools::texture2d_resident_view");
        self.view_generation = NEXT_TEXTURE2D_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed);
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
) -> (u32, u32) {
    if uploaded_mips == 0 || start_mip >= mip_levels_total {
        return resident_mip_range(*resident_mip_mask, mip_levels_total);
    }

    let end = start_mip
        .saturating_add(uploaded_mips)
        .min(mip_levels_total)
        .min(64);
    for mip in start_mip.min(64)..end {
        *resident_mip_mask |= 1u64 << mip;
    }

    resident_mip_range(*resident_mip_mask, mip_levels_total)
}

fn begin_resident_mip_mask(
    resident_mip_mask: &mut u64,
    mip_levels_total: u32,
    start_mip: u32,
    uploaded_mips: u32,
) -> (u32, u32) {
    *resident_mip_mask = 0;
    mark_resident_mip_mask(
        resident_mip_mask,
        mip_levels_total,
        start_mip,
        uploaded_mips,
    )
}

/// Chooses the longest initialized mip run, preferring the finer base when runs tie.
fn resident_mip_range(resident_mip_mask: u64, mip_levels_total: u32) -> (u32, u32) {
    let limit = mip_levels_total.min(64);
    let mut best_base = 0u32;
    let mut best_count = 0u32;
    let mut mip = 0u32;
    while mip < limit {
        if resident_mip_mask & (1u64 << mip) == 0 {
            mip += 1;
            continue;
        }
        let base = mip;
        while mip < limit && resident_mip_mask & (1u64 << mip) != 0 {
            mip += 1;
        }
        let count = mip - base;
        if count > best_count {
            best_base = base;
            best_count = count;
        }
    }
    (best_base, best_count)
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
        NEXT_TEXTURE2D_VIEW_GENERATION, begin_resident_mip_mask, mark_resident_mip_mask,
        texture2d_allocation_desc,
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
    fn coarse_tail_is_immediately_exposed_and_grows_toward_mip_zero() {
        let mut mask = 0;
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 5, 1), (5, 1));
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 4, 1), (4, 2));
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 0, 4), (0, 6));
    }

    #[test]
    fn resident_range_clamps_to_total_mips() {
        let mut mask = 0;
        assert_eq!(mark_resident_mip_mask(&mut mask, 4, 0, 10), (0, 4));
    }

    #[test]
    fn resident_range_prefers_finer_run_when_lengths_tie() {
        let mut mask = 0;
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 4, 2), (4, 2));
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 0, 2), (0, 2));
    }

    #[test]
    fn second_tail_first_upload_drops_old_fine_mips_at_first_completion() {
        let mut mask = 0;
        assert_eq!(begin_resident_mip_mask(&mut mask, 6, 0, 6), (0, 6));

        // Keep the old complete upload visible until the first physical mip of upload two lands,
        // then atomically begin the new stream from its coarse tail.
        assert_eq!(begin_resident_mip_mask(&mut mask, 6, 5, 1), (5, 1));
        assert_eq!(mask, 1 << 5);
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 4, 1), (4, 2));
    }

    #[test]
    fn nonzero_start_chain_preserves_earlier_residency() {
        let mut mask = 0;
        assert_eq!(begin_resident_mip_mask(&mut mask, 6, 0, 3), (0, 3));

        // A complete suffix update starting at mip 3 is not a full replacement. Its tail-first
        // writes accumulate alongside the valid lower mips until the suffix becomes contiguous.
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 5, 1), (0, 3));
        assert_eq!(mask & 0b111, 0b111);
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 4, 1), (0, 3));
        assert_eq!(mark_resident_mip_mask(&mut mask, 6, 3, 1), (0, 6));
    }
}
