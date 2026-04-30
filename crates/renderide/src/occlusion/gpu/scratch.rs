//! Reusable per-pyramid GPU scratch (staging rings, uniforms) and bind-group cache.

use crate::occlusion::cpu::pyramid::{mip_dimensions, mip_levels_for_extent};

use super::readback_ring::HIZ_STAGING_RING;

/// Maximum number of mip levels retained in each Hi-Z pyramid.
pub(crate) const HIZ_MAX_MIPS: u32 = 8;

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
    /// Uniform buffer carrying the current target array layer for stereo mip0 dispatches.
    pub layer_uniform: wgpu::Buffer,
    /// Uniform buffer carrying src/dst extents for the active downsample dispatch.
    pub downsample_uniform: wgpu::Buffer,
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

        let layer_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_layer_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let downsample_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_downsample_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_groups = HiZBindGroupCache::with_shape(mip_levels, stereo);
        Some(Self {
            extent: (bw, bh),
            mip_levels,
            staging_desktop,
            staging_r,
            layer_uniform,
            downsample_uniform,
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

/// Cached Hi-Z encode bind groups whose bindings are stable for the lifetime of a
/// [`HiZGpuScratch`]. Built lazily on first use so the cache is both cheap to initialise and
/// self-invalidating: recreating the scratch wipes every slot.
pub(crate) struct HiZBindGroupCache {
    /// `depth_view` last bound into the mip0 slots. Mip0 bindings are rebuilt when this changes
    /// (e.g. depth target reallocation between frames).
    mip0_depth_view: Option<wgpu::TextureView>,
    /// Mip0 view last used as the desktop/left pyramid output.
    pyramid_left_mip0_view: Option<wgpu::TextureView>,
    /// Mip0 view last used as the stereo-right pyramid output.
    pyramid_right_mip0_view: Option<wgpu::TextureView>,
    /// Mip0 bind group for desktop (non-stereo) dispatches.
    mip0_desktop: Option<wgpu::BindGroup>,
    /// Mip0 bind groups for stereo dispatches, indexed by array layer (`[layer0, layer1]`).
    mip0_stereo: [Option<wgpu::BindGroup>; 2],
    /// Downsample bind groups for the desktop / stereo-left pyramid, one per mip transition.
    downsample_desktop: Vec<Option<wgpu::BindGroup>>,
    /// Downsample bind groups for the stereo-right pyramid, one per mip transition.
    downsample_right: Vec<Option<wgpu::BindGroup>>,
}

impl HiZBindGroupCache {
    /// Creates an empty cache sized for `mip_levels` transitions; allocates a right-eye slot set
    /// only when `stereo` is true.
    fn with_shape(mip_levels: u32, stereo: bool) -> Self {
        let n = (mip_levels.saturating_sub(1)) as usize;
        Self {
            mip0_depth_view: None,
            pyramid_left_mip0_view: None,
            pyramid_right_mip0_view: None,
            mip0_desktop: None,
            mip0_stereo: [None, None],
            downsample_desktop: vec![None; n],
            downsample_right: if stereo { vec![None; n] } else { Vec::new() },
        }
    }

    /// Drops the mip0 slots whenever the caller-provided `depth_view` differs from the one used
    /// to build the cached entries.
    pub(crate) fn invalidate_mip0_if_depth_changed(&mut self, depth_view: &wgpu::TextureView) {
        if self.mip0_depth_view.as_ref() != Some(depth_view) {
            self.mip0_depth_view = Some(depth_view.clone());
            self.mip0_desktop = None;
            self.mip0_stereo = [None, None];
        }
    }

    /// Drops bind groups that reference the destination pyramid when the ping-pong half changes.
    pub(crate) fn invalidate_pyramid_if_target_changed(
        &mut self,
        left_mip0_view: &wgpu::TextureView,
        right_mip0_view: Option<&wgpu::TextureView>,
    ) {
        if self.pyramid_left_mip0_view.as_ref() == Some(left_mip0_view)
            && self.pyramid_right_mip0_view.as_ref() == right_mip0_view
        {
            return;
        }
        self.pyramid_left_mip0_view = Some(left_mip0_view.clone());
        self.pyramid_right_mip0_view = right_mip0_view.cloned();
        self.mip0_desktop = None;
        self.mip0_stereo = [None, None];
        for slot in &mut self.downsample_desktop {
            *slot = None;
        }
        for slot in &mut self.downsample_right {
            *slot = None;
        }
    }

    /// Returns a clone of the cached mip0 desktop bind group, building it via `build` on miss.
    pub(crate) fn mip0_desktop_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        build: F,
    ) -> wgpu::BindGroup {
        self.mip0_desktop.get_or_insert_with(build).clone()
    }

    /// Returns a clone of the cached mip0 stereo bind group for `layer`, building via `build` on miss.
    pub(crate) fn mip0_stereo_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        layer: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = (layer as usize).min(1);
        self.mip0_stereo[idx].get_or_insert_with(build).clone()
    }

    /// Returns a clone of the desktop downsample bind group at `mip`, building via `build` on miss.
    pub(crate) fn downsample_desktop_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        mip: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = mip as usize;
        self.downsample_desktop[idx]
            .get_or_insert_with(build)
            .clone()
    }

    /// Returns a clone of the stereo-right downsample bind group at `mip`, building via `build` on miss.
    pub(crate) fn downsample_right_or_build<F: FnOnce() -> wgpu::BindGroup>(
        &mut self,
        mip: u32,
        build: F,
    ) -> wgpu::BindGroup {
        let idx = mip as usize;
        self.downsample_right[idx].get_or_insert_with(build).clone()
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

fn make_staging_ring(
    device: &wgpu::Device,
    staging_size: u64,
    label_prefix: &str,
) -> [wgpu::Buffer; HIZ_STAGING_RING] {
    std::array::from_fn(|i| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}_{i}")),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    })
}
