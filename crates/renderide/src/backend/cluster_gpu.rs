//! Clustered forward lighting: GPU buffers for per-cluster light lists and compute-only uniforms.
//!
//! [`ClusterBufferCache`] recreates storage when the viewport, Z slice count, or stereo layer count
//! changes. Tile size and per-tile caps match the clustered light compute shader and PBS fragment sampling.

use std::mem::size_of;

/// Screen tile size in pixels (DOOM-style cluster grid XY).
pub const TILE_SIZE: u32 = 16;
/// Exponential depth slice count (view-space Z bins).
pub const CLUSTER_COUNT_Z: u32 = 24;
/// Maximum lights assigned to a single cluster (buffer index order).
pub const MAX_LIGHTS_PER_TILE: u32 = 32;
/// Uniform buffer size for clustered light compute `ClusterParams` (WGSL layout + tail padding).
pub const CLUSTER_PARAMS_UNIFORM_SIZE: u64 = 256;

/// References to GPU buffers shared by the clustered light compute pass and raster `@group(0)`.
pub struct ClusterBufferRefs<'a> {
    /// One `u32` count per cluster (compute writes; fragment reads plain `u32`; one thread per cluster).
    pub cluster_light_counts: &'a wgpu::Buffer,
    /// Flattened `cluster_id * MAX_LIGHTS_PER_TILE + slot` light indices.
    pub cluster_light_indices: &'a wgpu::Buffer,
    /// Uniform block for compute only (`ClusterParams` in WGSL).
    pub params_buffer: &'a wgpu::Buffer,
}

/// Caches cluster buffers; bumps [`Self::version`] when storage is recreated.
pub struct ClusterBufferCache {
    cluster_light_counts: Option<wgpu::Buffer>,
    cluster_light_indices: Option<wgpu::Buffer>,
    params_buffer: Option<wgpu::Buffer>,
    /// `(viewport, cluster_count_z, stereo_cluster_layers)`.
    cached_viewport: ((u32, u32), u32, u32),
    /// Incremented when buffers are recreated (bind group invalidation).
    pub version: u64,
}

impl ClusterBufferCache {
    /// Empty cache; [`Self::ensure_buffers`] allocates on first use.
    pub fn new() -> Self {
        Self {
            cluster_light_counts: None,
            cluster_light_indices: None,
            params_buffer: None,
            cached_viewport: ((0, 0), 0, 0),
            version: 0,
        }
    }

    /// Ensures buffers exist for `viewport`, `cluster_count_z`, and packed stereo layers (`1` or `2`).
    pub fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        viewport: (u32, u32),
        cluster_count_z: u32,
        stereo_cluster_layers: u32,
    ) -> Option<ClusterBufferRefs<'_>> {
        let (width, height) = viewport;
        if width == 0 || height == 0 {
            return None;
        }
        let layers = stereo_cluster_layers.clamp(1, 2);
        let cluster_count_x = width.div_ceil(TILE_SIZE);
        let cluster_count_y = height.div_ceil(TILE_SIZE);
        let cluster_count = (cluster_count_x * cluster_count_y * cluster_count_z * layers) as usize;
        let cache_key = (viewport, cluster_count_z, layers);
        if self.cluster_light_counts.is_none() || self.cached_viewport != cache_key {
            self.version = self.version.wrapping_add(1);
            self.cached_viewport = cache_key;

            self.cluster_light_counts = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_light_counts"),
                size: (cluster_count * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cluster_light_indices = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_light_indices"),
                size: (cluster_count * MAX_LIGHTS_PER_TILE as usize * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.params_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_params_uniform"),
                size: CLUSTER_PARAMS_UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        Some(ClusterBufferRefs {
            cluster_light_counts: self.cluster_light_counts.as_ref()?,
            cluster_light_indices: self.cluster_light_indices.as_ref()?,
            params_buffer: self.params_buffer.as_ref()?,
        })
    }

    /// Returns buffers when they match the last successful [`Self::ensure_buffers`] key.
    pub fn get_buffers(
        &self,
        viewport: (u32, u32),
        cluster_count_z: u32,
        stereo_cluster_layers: u32,
    ) -> Option<ClusterBufferRefs<'_>> {
        let layers = stereo_cluster_layers.clamp(1, 2);
        let cache_key = (viewport, cluster_count_z, layers);
        if self.cached_viewport != cache_key {
            return None;
        }
        Some(ClusterBufferRefs {
            cluster_light_counts: self.cluster_light_counts.as_ref()?,
            cluster_light_indices: self.cluster_light_indices.as_ref()?,
            params_buffer: self.params_buffer.as_ref()?,
        })
    }
}

impl Default for ClusterBufferCache {
    fn default() -> Self {
        Self::new()
    }
}
