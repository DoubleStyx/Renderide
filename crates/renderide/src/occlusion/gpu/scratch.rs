//! Reusable per-pyramid GPU scratch (staging rings, uniforms) and bind-group cache.

use crate::hi_z_cpu::pyramid::{mip_dimensions, mip_levels_for_extent};

use super::readback_ring::HIZ_STAGING_RING;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Maximum number of mip levels retained in each Hi-Z pyramid.
pub(crate) const HIZ_MAX_MIPS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct LayerUniform {
    layer: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct DownsampleUniform {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

/// Transient GPU resources reused while extent and mip count stay stable.
pub(crate) struct HiZGpuScratch {
    /// Pyramid base extent `(width, height)` in texels.
    pub extent: (u32, u32),
    /// Total mip count (mip0 through `mip_levels - 1`).
    pub mip_levels: u32,
    /// Triple-buffered staging for async readback.
    pub staging_desktop: [wgpu::Buffer; HIZ_STAGING_RING],
    /// Triple-buffered staging for the stereo-right pyramid.
    pub staging_r: Option<[wgpu::Buffer; HIZ_STAGING_RING]>,
    /// Immutable per-layer uniforms used by stereo mip0 dispatches.
    pub layer_uniforms: Option<[wgpu::Buffer; 2]>,
    /// Immutable per-mip uniforms used by downsample dispatches.
    pub downsample_uniforms: Vec<wgpu::Buffer>,
    /// Cached bind groups for this scratch's pipelines. Invalidated alongside the scratch itself
    /// (i.e. when `extent` / `mip_levels` / stereo layout changes trigger a fresh allocation).
    pub bind_groups: HiZBindGroupCache,
}

impl HiZGpuScratch {
    pub(crate) fn new(
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        extent: (u32, u32),
        stereo: bool,
    ) -> Option<Self> {
        let (bw, bh) = extent;
        if bw == 0 || bh == 0 {
            return None;
        }
        if !limits.texture_2d_fits(bw, bh) {
            logger::warn!(
                "hi_z scratch: pyramid extent {bw}x{bh} exceeds max_texture_dimension_2d={}; skipping",
                limits.max_texture_dimension_2d()
            );
            return None;
        }
        let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
        if mip_levels == 0 {
            return None;
        }
        let staging_size = staging_size_pyramid(bw, bh, mip_levels);
        if !limits.buffer_size_fits(staging_size) {
            logger::warn!(
                "hi_z scratch: staging size {staging_size} exceeds max_buffer_size={}; skipping",
                limits.max_buffer_size()
            );
            return None;
        }

        let staging_desktop = make_staging_ring(device, staging_size, "hi_z_staging_desktop");
        let staging_r = stereo.then(|| make_staging_ring(device, staging_size, "hi_z_staging_r"));

        let layer_uniforms = stereo.then(|| make_layer_uniforms(device));
        let downsample_uniforms = make_downsample_uniforms(device, (bw, bh), mip_levels);

        let bind_groups = HiZBindGroupCache::with_shape(mip_levels, stereo);
        Some(Self {
            extent: (bw, bh),
            mip_levels,
            staging_desktop,
            staging_r,
            layer_uniforms,
            downsample_uniforms,
            bind_groups,
        })
    }

    /// Returns the staging ring for the optional stereo-right pyramid, when configured.
    pub(crate) fn staging_right(&self) -> Option<&[wgpu::Buffer; HIZ_STAGING_RING]> {
        self.staging_r.as_ref()
    }

    /// Returns true when this scratch was allocated with a stereo-right staging ring.
    pub(crate) fn is_stereo(&self) -> bool {
        self.staging_r.is_some()
    }
}

/// Number of history targets retained by the Hi-Z bind-group cache.
///
/// Texture history alternates between exactly two halves. Keeping both avoids rebuilding every
/// bind group on every frame while still bounding retention if a target is reallocated without a
/// scratch shape change.
const HIZ_BIND_GROUP_CACHE_BANKS: usize = 2;

/// Cached Hi-Z encode bind groups whose bindings are stable for one destination target.
///
/// Texture history ping-pongs between two allocations, so the cache retains one lazily populated
/// bank per half. The source depth view is deliberately not part of the bank identity: render-graph
/// resolution can create a fresh descriptor-equivalent view handle for the same depth texture
/// every frame. A source-texture change invalidates only mip0, while the seven destination-only
/// downsample bind groups remain reusable. Recreating [`HiZGpuScratch`] still invalidates every
/// bank when the extent, mip count, or stereo layout changes.
pub(crate) struct HiZBindGroupCache {
    banks: Vec<HiZBindGroupCacheBank>,
    mip_levels: u32,
    stereo: bool,
}

/// Bind groups for one exact source-depth / destination-pyramid combination.
struct HiZBindGroupCacheBank {
    mip0_depth_texture: wgpu::Texture,
    pyramid_left_mip0_view: wgpu::TextureView,
    pyramid_right_mip0_view: Option<wgpu::TextureView>,
    mip0_desktop: Option<wgpu::BindGroup>,
    mip0_stereo: [Option<wgpu::BindGroup>; 2],
    downsample_desktop: Vec<Option<wgpu::BindGroup>>,
    downsample_right: Vec<Option<wgpu::BindGroup>>,
}

impl HiZBindGroupCache {
    /// Creates an empty two-target cache sized for `mip_levels` transitions.
    fn with_shape(mip_levels: u32, stereo: bool) -> Self {
        Self {
            banks: Vec::with_capacity(HIZ_BIND_GROUP_CACHE_BANKS),
            mip_levels,
            stereo,
        }
    }

    /// Selects the bank for the current ping-pong destination and updates its source depth.
    ///
    /// A hit is promoted to the MRU position. A third distinct target replaces only the LRU bank,
    /// which handles reallocations without retaining stale texture views indefinitely.
    pub(crate) fn select_target(
        &mut self,
        depth_texture: &wgpu::Texture,
        left_mip0_view: &wgpu::TextureView,
        right_mip0_view: Option<&wgpu::TextureView>,
    ) {
        let mip_levels = self.mip_levels;
        let stereo = self.stereo;
        select_or_insert_mru(
            &mut self.banks,
            HIZ_BIND_GROUP_CACHE_BANKS,
            |bank| bank.matches_pyramid(left_mip0_view, right_mip0_view),
            || {
                HiZBindGroupCacheBank::with_target(
                    mip_levels,
                    stereo,
                    depth_texture,
                    left_mip0_view,
                    right_mip0_view,
                )
            },
        );
        // The render graph may recreate the source TextureView handle every frame even while the
        // underlying depth texture stays stable. Only mip0 samples source depth; downsample passes
        // read and write the selected history pyramid exclusively.
        if let Some(bank) = self.banks.first_mut() {
            bank.select_depth_texture(depth_texture);
        }
    }

    /// Returns a clone of the cached mip0 desktop bind group, building it via `build` on miss.
    pub(crate) fn mip0_desktop_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        build: F,
    ) -> wgpu::BindGroup {
        let Some(bank) = self.banks.first_mut() else {
            return build();
        };
        bank.mip0_desktop.get_or_insert_with(build).clone()
    }

    /// Returns a clone of the cached mip0 stereo bind group for `layer`, building via `build` on miss.
    pub(crate) fn mip0_stereo_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        layer: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = (layer as usize).min(1);
        let Some(bank) = self.banks.first_mut() else {
            return build();
        };
        bank.mip0_stereo[idx].get_or_insert_with(build).clone()
    }

    /// Returns a clone of the desktop downsample bind group at `mip`, building via `build` on miss.
    pub(crate) fn downsample_desktop_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        mip: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = mip as usize;
        let Some(bank) = self.banks.first_mut() else {
            return build();
        };
        let Some(cached) = bank.downsample_desktop.get_mut(idx) else {
            return build();
        };
        cached.get_or_insert_with(build).clone()
    }

    /// Returns a clone of the stereo-right downsample bind group at `mip`, building via `build` on miss.
    pub(crate) fn downsample_right_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        mip: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = mip as usize;
        let Some(bank) = self.banks.first_mut() else {
            return build();
        };
        let Some(cached) = bank.downsample_right.get_mut(idx) else {
            return build();
        };
        cached.get_or_insert_with(build).clone()
    }
}

impl HiZBindGroupCacheBank {
    fn with_target(
        mip_levels: u32,
        stereo: bool,
        depth_texture: &wgpu::Texture,
        left_mip0_view: &wgpu::TextureView,
        right_mip0_view: Option<&wgpu::TextureView>,
    ) -> Self {
        let n = (mip_levels.saturating_sub(1)) as usize;
        Self {
            mip0_depth_texture: depth_texture.clone(),
            pyramid_left_mip0_view: left_mip0_view.clone(),
            pyramid_right_mip0_view: right_mip0_view.cloned(),
            mip0_desktop: None,
            mip0_stereo: [None, None],
            downsample_desktop: vec![None; n],
            downsample_right: if stereo { vec![None; n] } else { Vec::new() },
        }
    }

    fn matches_pyramid(
        &self,
        left_mip0_view: &wgpu::TextureView,
        right_mip0_view: Option<&wgpu::TextureView>,
    ) -> bool {
        &self.pyramid_left_mip0_view == left_mip0_view
            && self.pyramid_right_mip0_view.as_ref() == right_mip0_view
    }

    /// Invalidates only source-dependent mip0 bindings when the depth allocation changes.
    fn select_depth_texture(&mut self, depth_texture: &wgpu::Texture) {
        select_mip0_source_texture(
            &mut self.mip0_depth_texture,
            depth_texture.clone(),
            &mut self.mip0_desktop,
            &mut self.mip0_stereo,
        );
    }
}

/// Selects the underlying source texture identity and clears only bindings that sample mip0 on a
/// change. Kept generic so the invalidation contract can be tested without constructing GPU
/// objects; production instantiates it with [`wgpu::Texture`] and [`wgpu::BindGroup`].
fn select_mip0_source_texture<Source, Cached>(
    current_source: &mut Source,
    next_source: Source,
    mip0_desktop: &mut Option<Cached>,
    mip0_stereo: &mut [Option<Cached>; 2],
) -> bool
where
    Source: PartialEq,
{
    if current_source == &next_source {
        return false;
    }
    *current_source = next_source;
    *mip0_desktop = None;
    *mip0_stereo = [None, None];
    true
}

/// Promotes a matching entry to index zero or inserts a fresh MRU entry, evicting the LRU entry
/// once `capacity` is reached.
fn select_or_insert_mru<T>(
    entries: &mut Vec<T>,
    capacity: usize,
    mut matches: impl FnMut(&T) -> bool,
    build: impl FnOnce() -> T,
) {
    debug_assert!(capacity > 0);
    debug_assert!(entries.len() <= capacity);
    if let Some(index) = entries.iter().position(&mut matches) {
        entries.swap(0, index);
        return;
    }

    let entry = build();
    if entries.len() < capacity {
        entries.push(entry);
        let index = entries.len() - 1;
        entries.swap(0, index);
    } else {
        entries[capacity - 1] = entry;
        entries.swap(0, capacity - 1);
    }
}

fn staging_size_pyramid(base_w: u32, base_h: u32, mip_levels: u32) -> u64 {
    let mut total = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((0, 0));
        let row_pitch = u64::from(wgpu::util::align_to(
            w * 4,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        ));
        total += row_pitch * u64::from(h);
    }
    total
}

fn make_layer_uniforms(device: &wgpu::Device) -> [wgpu::Buffer; 2] {
    std::array::from_fn(|layer| {
        let payload = layer_uniform_for_layer(layer as u32);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(match layer {
                0 => "hi_z_layer_uniform_0",
                _ => "hi_z_layer_uniform_1",
            }),
            contents: bytemuck::bytes_of(&payload),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        crate::profiling::note_resource_churn!(Buffer, "occlusion::hi_z_layer_uniform");
        buffer
    })
}

fn layer_uniform_for_layer(layer: u32) -> LayerUniform {
    LayerUniform {
        layer,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

fn make_downsample_uniforms(
    device: &wgpu::Device,
    extent: (u32, u32),
    mip_levels: u32,
) -> Vec<wgpu::Buffer> {
    (0..mip_levels.saturating_sub(1))
        .filter_map(|mip| {
            let payload = downsample_uniform_for_mip(extent, mip)?;
            let label = format!("hi_z_downsample_uniform_{mip}");
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label.as_str()),
                contents: bytemuck::bytes_of(&payload),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            crate::profiling::note_resource_churn!(Buffer, "occlusion::hi_z_downsample_uniform");
            Some(buffer)
        })
        .collect()
}

fn downsample_uniform_for_mip(extent: (u32, u32), mip: u32) -> Option<DownsampleUniform> {
    let (base_w, base_h) = extent;
    let (src_w, src_h) = mip_dimensions(base_w, base_h, mip)?;
    let (dst_w, dst_h) = mip_dimensions(base_w, base_h, mip + 1)?;
    Some(DownsampleUniform {
        src_w,
        src_h,
        dst_w,
        dst_h,
    })
}

fn make_staging_ring(
    device: &wgpu::Device,
    staging_size: u64,
    label_prefix: &str,
) -> [wgpu::Buffer; HIZ_STAGING_RING] {
    std::array::from_fn(|i| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}_{i}")),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "occlusion::hi_z_staging_ring");
        buffer
    })
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::{
        LayerUniform, downsample_uniform_for_mip, layer_uniform_for_layer,
        select_mip0_source_texture, select_or_insert_mru,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FakeTexture(u32);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FakeView {
        texture: FakeTexture,
        serial: u32,
    }

    fn select_value(entries: &mut Vec<u32>, value: u32) {
        select_or_insert_mru(entries, 2, |entry| *entry == value, || value);
    }

    #[test]
    fn bounded_mru_cache_retains_both_ping_pong_targets() {
        let mut entries = Vec::new();

        select_value(&mut entries, 10);
        select_value(&mut entries, 20);
        assert_eq!(entries, [20, 10]);

        select_value(&mut entries, 10);
        assert_eq!(entries, [10, 20]);

        select_value(&mut entries, 30);
        assert_eq!(entries, [30, 10]);
    }

    #[test]
    fn descriptor_equivalent_view_churn_reuses_mip0_bindings() {
        let first_view = FakeView {
            texture: FakeTexture(7),
            serial: 1,
        };
        let next_view = FakeView {
            texture: FakeTexture(7),
            serial: 2,
        };
        assert_ne!(
            first_view, next_view,
            "the graph supplied a fresh view handle"
        );

        let mut source_texture = first_view.texture;
        let mut desktop = Some(10);
        let mut stereo = [Some(20), Some(30)];

        assert!(!select_mip0_source_texture(
            &mut source_texture,
            next_view.texture,
            &mut desktop,
            &mut stereo,
        ));
        assert_eq!(desktop, Some(10));
        assert_eq!(stereo, [Some(20), Some(30)]);
    }

    #[test]
    fn depth_texture_reallocation_invalidates_all_mip0_bindings_only() {
        let mut source_texture = FakeTexture(7);
        let mut desktop = Some(10);
        let mut stereo = [Some(20), Some(30)];
        let downsample = [Some(40), Some(50)];

        assert!(select_mip0_source_texture(
            &mut source_texture,
            FakeTexture(8),
            &mut desktop,
            &mut stereo,
        ));
        assert_eq!(source_texture, FakeTexture(8));
        assert_eq!(desktop, None);
        assert_eq!(stereo, [None, None]);
        assert_eq!(downsample, [Some(40), Some(50)]);
    }

    #[test]
    fn layer_uniform_payloads_select_expected_layers() {
        let left = layer_uniform_for_layer(0);
        let right = layer_uniform_for_layer(1);

        assert_eq!(left.layer, 0);
        assert_eq!(right.layer, 1);
        assert_eq!(size_of::<LayerUniform>(), 16);
    }

    #[test]
    fn downsample_uniform_payloads_match_mip_dimensions() {
        let mip0 = downsample_uniform_for_mip((9, 5), 0).unwrap();
        let mip1 = downsample_uniform_for_mip((9, 5), 1).unwrap();

        assert_eq!(
            (mip0.src_w, mip0.src_h, mip0.dst_w, mip0.dst_h),
            (9, 5, 4, 2)
        );
        assert_eq!(
            (mip1.src_w, mip1.src_h, mip1.dst_w, mip1.dst_h),
            (4, 2, 2, 1)
        );
    }
}
