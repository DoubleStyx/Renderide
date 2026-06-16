//! GPU-resident [`SetCubemapFormat`](crate::shared::SetCubemapFormat) pool ([`GpuCubemap`]) with VRAM accounting.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::assets::texture::{
    estimate_gpu_cubemap_bytes, host_texture_mip_count, legal_texture2d_mip_level_count,
    resolve_cubemap_wgpu_format,
};
use crate::gpu::GpuLimits;
use crate::shared::{SetCubemapFormat, SetCubemapProperties, TextureFormat};

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

static NEXT_CUBEMAP_ALLOCATION_GENERATION: AtomicU64 = AtomicU64::new(1);
const MAX_GPU_TEXTURE_ALLOCATION_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// One full-mip 2D texture view for each cubemap face.
type CubemapFaceViews = [Arc<wgpu::TextureView>; crate::gpu::CUBEMAP_ARRAY_LAYERS as usize];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CubemapAllocationDesc {
    size: u32,
    mip_levels_total: u32,
    wgpu_format: wgpu::TextureFormat,
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
    /// Full-mip 2D-array view for manual cross-face filtering.
    pub array_view: Arc<wgpu::TextureView>,
    /// Full-mip 2D views for each cubemap face.
    pub face_views: CubemapFaceViews,
    /// Resolved wgpu format for `texture`.
    pub wgpu_format: wgpu::TextureFormat,
    /// Host [`TextureFormat`] enum (compression / layout family).
    pub host_format: TextureFormat,
    /// Face size in texels (mip0).
    pub size: u32,
    /// Mip chain length allocated on GPU.
    pub mip_levels_total: u32,
    /// Mips with authored texels uploaded so far.
    pub mip_levels_resident: u32,
    /// Monotonic generation bumped whenever this cubemap's GPU texel contents are uploaded.
    pub content_generation: u64,
    /// Monotonic identifier for the current GPU allocation.
    pub allocation_generation: u64,
    /// Whether face data needs shader-side storage-orientation compensation.
    pub storage_v_inverted: bool,
    /// Estimated VRAM for allocated mips.
    pub resident_bytes: u64,
    /// Sampler fields for material bind groups.
    pub sampler: SamplerState,
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
        let desc = Self::allocation_desc_from_format(device, limits, fmt)?;
        let size = wgpu::Extent3d {
            width: desc.size,
            height: desc.size,
            depth_or_array_layers: 6,
        };
        let resident_bytes =
            estimate_gpu_cubemap_bytes(desc.wgpu_format, desc.size, desc.mip_levels_total);
        if resident_bytes > MAX_GPU_TEXTURE_ALLOCATION_BYTES {
            logger::warn!(
                "Cubemap {} rejected: estimated resident bytes {} exceed cap {}",
                fmt.asset_id,
                resident_bytes,
                MAX_GPU_TEXTURE_ALLOCATION_BYTES
            );
            return None;
        }
        let texture_label = format!("Cubemap {}", fmt.asset_id);
        let view_label = format!("Cubemap {} cube view", fmt.asset_id);
        let (texture, view) = create_sampled_copy_dst_texture(
            device,
            SampledTextureAllocation {
                label: &texture_label,
                size,
                mip_level_count: desc.mip_levels_total,
                dimension: wgpu::TextureDimension::D2,
                format: desc.wgpu_format,
                view: TextureViewInit {
                    label: Some(&view_label),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                },
            },
        );
        let array_view = create_cubemap_array_view(
            texture.as_ref(),
            fmt.asset_id,
            desc.wgpu_format,
            desc.mip_levels_total,
        );
        let face_views = create_cubemap_face_views(
            texture.as_ref(),
            fmt.asset_id,
            desc.wgpu_format,
            desc.mip_levels_total,
        );
        let sampler = SamplerState::from_cubemap_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture,
            view,
            array_view,
            face_views,
            wgpu_format: desc.wgpu_format,
            host_format: fmt.format,
            size: desc.size,
            mip_levels_total: desc.mip_levels_total,
            mip_levels_resident: 0,
            content_generation: 0,
            allocation_generation: NEXT_CUBEMAP_ALLOCATION_GENERATION
                .fetch_add(1, Ordering::Relaxed),
            storage_v_inverted: false,
            resident_bytes,
            sampler,
            residency,
        })
    }

    /// Returns `true` when `fmt` resolves to this cubemap's current GPU allocation shape.
    pub(crate) fn allocation_matches_format(
        &self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetCubemapFormat,
    ) -> bool {
        Self::allocation_desc_from_format(device, limits, fmt)
            .is_some_and(|desc| self.allocation_matches_desc(desc))
    }

    /// Updates format metadata without changing the GPU allocation or resident mip state.
    pub(crate) fn apply_format_metadata(
        &mut self,
        fmt: &SetCubemapFormat,
        props: Option<&SetCubemapProperties>,
    ) {
        self.host_format = fmt.format;
        self.sampler = SamplerState::from_cubemap_props(props);
        self.residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
    }

    fn allocation_desc_from_format(
        device: &wgpu::Device,
        limits: &GpuLimits,
        fmt: &SetCubemapFormat,
    ) -> Option<CubemapAllocationDesc> {
        let wgpu_format = resolve_cubemap_wgpu_format(device, fmt);
        cubemap_allocation_desc(limits, fmt, wgpu_format)
    }

    fn allocation_matches_desc(&self, desc: CubemapAllocationDesc) -> bool {
        self.size == desc.size
            && self.mip_levels_total == desc.mip_levels_total
            && self.wgpu_format == desc.wgpu_format
    }

    /// Marks that at least one face/mip upload changed this cubemap's GPU contents.
    pub fn mark_content_uploaded(&mut self) {
        self.content_generation = self.content_generation.wrapping_add(1).max(1);
    }

    /// Updates sampler fields and residency hints from host properties.
    pub fn apply_properties(&mut self, p: &SetCubemapProperties) {
        self.sampler = SamplerState::from_cubemap_props(Some(p));
        self.residency = TextureResidencyMeta::from_host_props(p);
    }
}

fn cubemap_allocation_desc(
    limits: &GpuLimits,
    fmt: &SetCubemapFormat,
    wgpu_format: wgpu::TextureFormat,
) -> Option<CubemapAllocationDesc> {
    let s = fmt.size.max(0) as u32;
    if s == 0 {
        return None;
    }
    let max_dim = limits.max_texture_dimension_2d();
    if !validate_texture_extent(
        fmt.asset_id,
        "cubemap",
        "face size",
        &s,
        &[s],
        max_dim,
        "max_texture_dimension_2d",
    ) {
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
    let requested_mips = host_texture_mip_count(fmt.mipmap_count);
    let legal_mips = legal_texture2d_mip_level_count(s, s);
    let mip_levels_total = clamp_texture_mip_count(
        fmt.asset_id,
        "cubemap",
        &format_args!("face size {s}"),
        requested_mips,
        legal_mips,
    );
    Some(CubemapAllocationDesc {
        size: s,
        mip_levels_total,
        wgpu_format,
    })
}

/// Creates the full cubemap array view used by shaders that sample faces manually.
fn create_cubemap_array_view(
    texture: &wgpu::Texture,
    asset_id: i32,
    format: wgpu::TextureFormat,
    mip_count: u32,
) -> Arc<wgpu::TextureView> {
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("Cubemap {asset_id} array view")),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_count),
        base_array_layer: 0,
        array_layer_count: Some(crate::gpu::CUBEMAP_ARRAY_LAYERS),
    }));
    crate::profiling::note_resource_churn!(TextureView, "cubemap::array_view");
    view
}

/// Creates one full-mip 2D sampled view for each cubemap face.
fn create_cubemap_face_views(
    texture: &wgpu::Texture,
    asset_id: i32,
    format: wgpu::TextureFormat,
    mip_count: u32,
) -> CubemapFaceViews {
    let views = std::array::from_fn(|face| {
        Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("Cubemap {asset_id} face {face} view")),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_count),
            base_array_layer: face as u32,
            array_layer_count: Some(1),
        }))
    });
    crate::profiling::note_resource_churn!(TextureView, "cubemap::face_views");
    views
}

impl_gpu_resource!(GpuCubemap);

/// Resident cubemap table.
pub struct CubemapPool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuCubemap, StreamingAccess>,
}

impl_streaming_pool_facade!(
    CubemapPool,
    GpuCubemap,
    StreamingAccess::texture,
    StreamingAccess::texture_noop,
);

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use crate::gpu::GpuLimits;
    use crate::shared::{ColorProfile, SetCubemapFormat, TextureFormat};

    use super::cubemap_allocation_desc;

    fn test_limits(max_texture_dimension_2d: u32, max_texture_array_layers: u32) -> GpuLimits {
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d,
                max_texture_array_layers,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    fn format(
        size: i32,
        mipmap_count: i32,
        format: TextureFormat,
        profile: ColorProfile,
    ) -> SetCubemapFormat {
        SetCubemapFormat {
            asset_id: 11,
            size,
            mipmap_count,
            format,
            profile,
        }
    }

    #[test]
    fn allocation_desc_reuses_same_gpu_shape() {
        let limits = test_limits(4096, crate::gpu::CUBEMAP_ARRAY_LAYERS);
        let base = cubemap_allocation_desc(
            &limits,
            &format(64, 4, TextureFormat::RGBA32, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("base allocation");
        let same_storage = cubemap_allocation_desc(
            &limits,
            &format(64, 4, TextureFormat::RGB24, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("same allocation");

        assert_eq!(base, same_storage);
    }

    #[test]
    fn allocation_desc_changes_for_size_mips_or_storage() {
        let limits = test_limits(4096, crate::gpu::CUBEMAP_ARRAY_LAYERS);
        let base = cubemap_allocation_desc(
            &limits,
            &format(64, 4, TextureFormat::RGBA32, ColorProfile::Linear),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("base allocation");

        assert_ne!(
            base,
            cubemap_allocation_desc(
                &limits,
                &format(128, 4, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("different size")
        );
        assert_ne!(
            base,
            cubemap_allocation_desc(
                &limits,
                &format(64, 2, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("different mips")
        );
        assert_ne!(
            base,
            cubemap_allocation_desc(
                &limits,
                &format(64, 4, TextureFormat::RGBA32, ColorProfile::SRGB),
                wgpu::TextureFormat::Rgba8UnormSrgb,
            )
            .expect("different storage")
        );
    }

    #[test]
    fn allocation_desc_rejects_invalid_size_or_array_layers() {
        let valid_limits = test_limits(64, crate::gpu::CUBEMAP_ARRAY_LAYERS);
        let short_array_limits = test_limits(64, crate::gpu::CUBEMAP_ARRAY_LAYERS - 1);

        assert!(
            cubemap_allocation_desc(
                &valid_limits,
                &format(0, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
        assert!(
            cubemap_allocation_desc(
                &valid_limits,
                &format(128, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
        assert!(
            cubemap_allocation_desc(
                &short_array_limits,
                &format(64, 1, TextureFormat::RGBA32, ColorProfile::Linear),
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_none()
        );
    }
}
