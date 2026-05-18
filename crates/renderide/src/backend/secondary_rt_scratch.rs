//! Backend-owned scratch render targets for partial secondary render-texture camera viewports.

use std::sync::Arc;

use hashbrown::HashMap;

/// Maximum distinct scratch target layouts retained by the cache.
const SECONDARY_RT_SCRATCH_CACHE_LIMIT: usize = 16;

/// Layout key for reusable partial-viewport render targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SecondaryRtScratchKey {
    /// Scratch color/depth width in pixels.
    width: u32,
    /// Scratch color/depth height in pixels.
    height: u32,
    /// Color attachment format.
    color_format: wgpu::TextureFormat,
    /// Depth attachment format.
    depth_format: wgpu::TextureFormat,
}

/// Color/depth handles used as the graph target for a partial render texture viewport.
#[derive(Clone)]
pub(crate) struct SecondaryRtScratchTargets {
    /// Color texture rendered by the graph and copied into the host render texture.
    pub(crate) color_texture: Arc<wgpu::Texture>,
    /// Color view rendered by graph raster passes.
    pub(crate) color_view: Arc<wgpu::TextureView>,
    /// Depth texture used by graph raster and depth-copy passes.
    pub(crate) depth_texture: Arc<wgpu::Texture>,
    /// Depth view used by graph raster and compute passes.
    pub(crate) depth_view: Arc<wgpu::TextureView>,
}

/// Small cache of scratch targets for partial secondary camera viewport renders.
#[derive(Default)]
pub(crate) struct SecondaryRtScratchCache {
    /// Reusable scratch targets keyed by extent and attachment formats.
    targets: HashMap<SecondaryRtScratchKey, SecondaryRtScratchTargets>,
}

impl SecondaryRtScratchCache {
    /// Creates an empty scratch target cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns a scratch target matching `extent_px` and formats, creating it on a cache miss.
    pub(crate) fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        extent_px: (u32, u32),
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Option<SecondaryRtScratchTargets> {
        if extent_px.0 == 0 || extent_px.1 == 0 {
            return None;
        }
        let key = SecondaryRtScratchKey {
            width: extent_px.0,
            height: extent_px.1,
            color_format,
            depth_format,
        };
        if let Some(targets) = self.targets.get(&key) {
            return Some(targets.clone());
        }
        self.evict_one_entry_if_full();
        let targets = create_scratch_targets(device, key);
        self.targets.insert(key, targets.clone());
        Some(targets)
    }

    /// Removes one cached layout before inserting a new one when the cache is full.
    fn evict_one_entry_if_full(&mut self) {
        if self.targets.len() < SECONDARY_RT_SCRATCH_CACHE_LIMIT {
            return;
        }
        if let Some(key) = self.targets.keys().next().copied() {
            self.targets.remove(&key);
        }
    }
}

/// Allocates a color/depth scratch target for one partial viewport layout.
fn create_scratch_targets(
    device: &wgpu::Device,
    key: SecondaryRtScratchKey,
) -> SecondaryRtScratchTargets {
    let size = wgpu::Extent3d {
        width: key.width,
        height: key.height,
        depth_or_array_layers: 1,
    };
    let color_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("renderide-secondary-rt-viewport-color"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: key.color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    let color_view = Arc::new(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));
    crate::profiling::note_resource_churn!(TextureView, "backend::secondary_rt_scratch_color_view");

    let depth_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("renderide-secondary-rt-viewport-depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: key.depth_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    let depth_view = Arc::new(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
    crate::profiling::note_resource_churn!(TextureView, "backend::secondary_rt_scratch_depth_view");

    SecondaryRtScratchTargets {
        color_texture,
        color_view,
        depth_texture,
        depth_view,
    }
}
