//! Cluster buffer cache for the clustered light compute pass.
//!
//! Caches cluster_light_counts, cluster_light_indices, and params buffers.
//! Recreates only when viewport (tile_count_x * tile_count_y) changes.

use std::mem::size_of;

/// Tile size in pixels (matches clustered_light pass).
const TILE_SIZE: u32 = 16;
/// Maximum lights per tile (matches clustered_light pass).
const MAX_LIGHTS_PER_TILE: u32 = 32;
/// Size of ClusterParams uniform (3 mat4 + 2 f32 + 4 u32 + 2 f32 = 224 bytes).
const CLUSTER_PARAMS_SIZE: u64 = 224;

/// References to cluster buffers for the compute pass.
pub struct ClusterBufferRefs<'a> {
    /// Cluster light counts (atomic u32 per tile).
    pub cluster_light_counts: &'a wgpu::Buffer,
    /// Cluster light indices (u32 per light per tile).
    pub cluster_light_indices: &'a wgpu::Buffer,
    /// Cluster params uniform.
    pub params_buffer: &'a wgpu::Buffer,
}

/// Cache for cluster buffers. Recreates only when viewport changes.
pub struct ClusterBufferCache {
    cluster_light_counts: Option<wgpu::Buffer>,
    cluster_light_indices: Option<wgpu::Buffer>,
    params_buffer: Option<wgpu::Buffer>,
    cached_viewport: (u32, u32),
    /// Incremented when buffers are recreated. Used for invalidating PBR bind group cache.
    pub version: u64,
}

impl ClusterBufferCache {
    /// Creates a new empty cache.
    pub fn new() -> Self {
        Self {
            cluster_light_counts: None,
            cluster_light_indices: None,
            params_buffer: None,
            cached_viewport: (0, 0),
            version: 0,
        }
    }

    /// Ensures buffers exist for the given viewport. Recreates only when viewport changes.
    /// Returns references to the buffers, or None if viewport is zero-sized.
    pub fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        viewport: (u32, u32),
    ) -> Option<ClusterBufferRefs<'_>> {
        let (width, height) = viewport;
        if width == 0 || height == 0 {
            return None;
        }
        if self.cluster_light_counts.is_none() || self.cached_viewport != viewport {
            self.version = self.version.wrapping_add(1);
            let cluster_count_x = width.div_ceil(TILE_SIZE);
            let cluster_count_y = height.div_ceil(TILE_SIZE);
            let cluster_count = (cluster_count_x * cluster_count_y) as usize;

            self.cluster_light_counts = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster light counts"),
                size: (cluster_count * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cluster_light_indices = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster light indices"),
                size: (cluster_count * MAX_LIGHTS_PER_TILE as usize * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.params_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster params uniform"),
                size: CLUSTER_PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cached_viewport = viewport;
        }
        Some(ClusterBufferRefs {
            cluster_light_counts: self.cluster_light_counts.as_ref()?,
            cluster_light_indices: self.cluster_light_indices.as_ref()?,
            params_buffer: self.params_buffer.as_ref()?,
        })
    }

    /// Returns references to cluster buffers if they exist and match the viewport.
    pub fn get_buffers(&self, viewport: (u32, u32)) -> Option<ClusterBufferRefs<'_>> {
        if self.cached_viewport != viewport {
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
