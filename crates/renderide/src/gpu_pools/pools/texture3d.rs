//! GPU-resident [`SetTexture3DFormat`](crate::shared::SetTexture3DFormat) pool ([`GpuTexture3d`]) with VRAM accounting.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::assets::texture::{
    estimate_gpu_texture3d_bytes, host_texture_mip_count, legal_texture3d_mip_level_count,
    resolve_texture3d_wgpu_format,
};
use crate::gpu::GpuLimits;
use crate::shared::{SetTexture3DFormat, SetTexture3DProperties};

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

static NEXT_TEXTURE3D_VIEW_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Texture3dAllocationDesc {
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_levels_total: u32,
    wgpu_format: wgpu::TextureFormat,
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
    /// Monotonic identifier for the current bindable view.
    pub view_generation: u64,
    /// Resolved wgpu format for `texture`.
    pub wgpu_format: wgpu::TextureFormat,
    /// Mip chain length allocated on GPU.
    pub mip_levels_total: u32,
    /// Mips with authored texels uploaded so far.
    pub mip_levels_resident: u32,
    /// Estimated VRAM for allocated mips.
    pub resident_bytes: u64,
    /// Sampler fields for material bind groups.
    pub sampler: SamplerState,
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
        let desc = Self::allocation_desc_from_format(device, limits, fmt)?;
        let size = wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: desc.depth_or_array_layers,
        };
        let texture_label = format!("Texture3D {}", fmt.asset_id);
        let view_label = format!("Texture3D {} view", fmt.asset_id);
        let (texture, view) = create_sampled_copy_dst_texture(
            device,
            SampledTextureAllocation {
                label: &texture_label,
                size,
                mip_level_count: desc.mip_levels_total,
                dimension: wgpu::TextureDimension::D3,
                format: desc.wgpu_format,
                view: TextureViewInit {
                    label: Some(&view_label),
                    dimension: Some(wgpu::TextureViewDimension::D3),
                },
            },
        );
        let resident_bytes = estimate_gpu_texture3d_bytes(
            desc.wgpu_format,
            desc.width,
            desc.height,
            desc.depth_or_array_layers,
            desc.mip_levels_total,
        );
        let sampler = SamplerState::from_texture3d_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture,
            view,
            view_generation: NEXT_TEXTURE3D_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed),
            wgpu_format: desc.wgpu_format,
            mip_levels_total: desc.mip_levels_total,
            mip_levels_resident: 0,
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
        fmt: &SetTexture3DFormat,
    ) -> bool {
        Self::allocation_desc_from_format(device, limits, fmt)
            .is_some_and(|desc| self.allocation_matches_desc(desc))
    }

    /// Updates format metadata without changing the GPU allocation or resident mip state.
    pub(crate) fn apply_format_metadata(
        &mut self,
        _fmt: &SetTexture3DFormat,
        props: Option<&SetTexture3DProperties>,
    ) {
        self.sampler = SamplerState::from_texture3d_props(props);
        self.residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
    }

    fn allocation_desc_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetTexture3DFormat,
    ) -> Option<Texture3dAllocationDesc> {
        let wgpu_format = resolve_texture3d_wgpu_format(device, fmt);
        texture3d_allocation_desc(limits, fmt, wgpu_format)
    }

    fn allocation_matches_desc(&self, desc: Texture3dAllocationDesc) -> bool {
        let size = self.texture.size();
        size.width == desc.width
            && size.height == desc.height
            && size.depth_or_array_layers == desc.depth_or_array_layers
            && self.mip_levels_total == desc.mip_levels_total
            && self.wgpu_format == desc.wgpu_format
    }

    /// Updates sampler fields and residency hints from host properties.
    pub fn apply_properties(&mut self, p: &SetTexture3DProperties) {
        self.sampler = SamplerState::from_texture3d_props(Some(p));
        self.residency = TextureResidencyMeta::from_host_props(p);
    }
}

impl_gpu_resource!(GpuTexture3d);

fn texture3d_allocation_desc(
    limits: &GpuLimits,
    fmt: &SetTexture3DFormat,
    wgpu_format: wgpu::TextureFormat,
) -> Option<Texture3dAllocationDesc> {
    let w = fmt.width.max(0) as u32;
    let h = fmt.height.max(0) as u32;
    let d = fmt.depth.max(0) as u32;
    if w == 0 || h == 0 || d == 0 {
        return None;
    }
    let max_dim = limits.max_texture_dimension_3d();
    if !validate_texture_extent(
        fmt.asset_id,
        "texture3d",
        "format size",
        &format_args!("{w}x{h}x{d}"),
        &[w, h, d],
        max_dim,
        "max_texture_dimension_3d",
    ) {
        return None;
    }
    let requested_mips = host_texture_mip_count(fmt.mipmap_count);
    let legal_mips = legal_texture3d_mip_level_count(w, h, d);
    let mip_levels_total = clamp_texture_mip_count(
        fmt.asset_id,
        "texture3d",
        &format_args!("{w}x{h}x{d}"),
        requested_mips,
        legal_mips,
    );
    Some(Texture3dAllocationDesc {
        width: w,
        height: h,
        depth_or_array_layers: d,
        mip_levels_total,
        wgpu_format,
    })
}

/// Resident Texture3D table; pairs with [`super::TexturePool`] under one renderer.
pub struct Texture3dPool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuTexture3d, StreamingAccess>,
}

impl_streaming_pool_facade!(
    Texture3dPool,
    GpuTexture3d,
    StreamingAccess::texture,
    StreamingAccess::texture_noop,
);

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use crate::gpu::GpuLimits;
    use crate::shared::{ColorProfile, SetTexture3DFormat, TextureFormat};

    use super::texture3d_allocation_desc;

    fn test_limits(max_texture_dimension_3d: u32) -> GpuLimits {
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_3d,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    fn format(
        width: i32,
        height: i32,
        depth: i32,
        mipmap_count: i32,
        format: TextureFormat,
        profile: ColorProfile,
    ) -> SetTexture3DFormat {
        SetTexture3DFormat {
            asset_id: 9,
            width,
            height,
            depth,
            mipmap_count,
            format,
            profile,
        }
    }

    #[test]
    fn allocation_desc_reuses_same_gpu_shape() {
        let limits = test_limits(512);
        let base = texture3d_allocation_desc(
            &limits,
            &format(32, 16, 8, 4, TextureFormat::RGBA32, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("base allocation");
        let same_storage = texture3d_allocation_desc(
            &limits,
            &format(32, 16, 8, 4, TextureFormat::RGB24, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("same allocation");

        assert_eq!(base, same_storage);
    }

    #[test]
    fn allocation_desc_changes_for_depth_mips_or_storage() {
        let limits = test_limits(512);
        let base = texture3d_allocation_desc(
            &limits,
            &format(32, 16, 8, 4, TextureFormat::RGBA32, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("base allocation");

        assert_ne!(
            base,
            texture3d_allocation_desc(
                &limits,
                &format(32, 16, 16, 4, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("different depth")
        );
        assert_ne!(
            base,
            texture3d_allocation_desc(
                &limits,
                &format(32, 16, 8, 2, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("different mips")
        );
        assert_ne!(
            base,
            texture3d_allocation_desc(
                &limits,
                &format(32, 16, 8, 4, TextureFormat::RGBA32, ColorProfile::SRGB),
                wgpu::TextureFormat::Rgba8UnormSrgb,
            )
            .expect("different storage")
        );
    }

    #[test]
    fn allocation_desc_rejects_invalid_size() {
        let limits = test_limits(64);

        assert!(
            texture3d_allocation_desc(
                &limits,
                &format(32, 0, 8, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
        assert!(
            texture3d_allocation_desc(
                &limits,
                &format(32, 16, 128, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
    }
}
