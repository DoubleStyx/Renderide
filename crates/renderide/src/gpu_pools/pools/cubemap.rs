//! GPU-resident [`SetCubemapFormat`](crate::shared::SetCubemapFormat) pool ([`GpuCubemap`]) with VRAM accounting.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::assets::texture::{
    estimate_gpu_cubemap_bytes, host_texture_mip_count, legal_texture2d_mip_level_count,
    resolve_cubemap_wgpu_format,
};
use crate::gpu::GpuLimits;
use crate::shared::{SetCubemapFormat, SetCubemapProperties};

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
    /// Resolved wgpu format for `texture`.
    pub wgpu_format: wgpu::TextureFormat,
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
        let mips = clamp_texture_mip_count(
            fmt.asset_id,
            "cubemap",
            &format_args!("face size {s}"),
            requested_mips,
            legal_mips,
        );
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
        let array_view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("Cubemap {} array view", fmt.asset_id)),
            format: Some(wgpu_format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mips),
            base_array_layer: 0,
            array_layer_count: Some(6),
        }));
        crate::profiling::note_resource_churn!(TextureView, "cubemap::array_view");
        let resident_bytes = estimate_gpu_cubemap_bytes(wgpu_format, s, mips);
        let sampler = SamplerState::from_cubemap_props(props);
        let residency = props
            .map(TextureResidencyMeta::from_host_props)
            .unwrap_or_default();
        Some(Self {
            asset_id: fmt.asset_id,
            texture,
            view,
            array_view,
            wgpu_format,
            size: s,
            mip_levels_total: mips,
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
