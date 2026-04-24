//! Clustered forward lighting: shared GPU buffers for per-cluster light lists.
//!
//! [`ClusterBufferCache`] tracks a grow-only high-water-mark over `cluster_light_counts` and
//! `cluster_light_indices` — the only two buffers shared across views. The per-view compute
//! params uniform (`ClusterParams`) is **not** managed here; it lives in
//! `PerViewFrameState::cluster_params_buffer` to avoid a CPU-side write race during parallel
//! per-view recording (see `frame_resource_manager.rs`).

use std::mem::size_of;

use crate::gpu::GpuLimits;

/// Screen tile size in pixels (DOOM-style cluster grid XY). Filament-like coarse grid to keep
/// `total_clusters` small (4× fewer than a 16-pixel tile). Keep in sync with `TILE_SIZE` in
/// `shaders/source/modules/pbs_cluster.wgsl`.
pub const TILE_SIZE: u32 = 32;
/// Exponential depth slice count (view-space Z bins). Filament-like shallow slice count.
pub const CLUSTER_COUNT_Z: u32 = 16;
/// Maximum lights assigned to a single cluster (buffer index order). Keep in sync with
/// `MAX_LIGHTS_PER_TILE` in `shaders/source/modules/pbs_cluster.wgsl` and
/// `shaders/source/compute/clustered_light.wgsl`. Bumped from 32 to reduce far-cluster overflow
/// that produced dark "splotches" in scenes with many lights.
///
/// Indices are packed 2-per-`u32` in `cluster_light_indices` (low 16 bits = even slot, high 16
/// bits = odd slot); the cluster's own compute thread is the sole writer, so no atomics are
/// required. `MAX_LIGHTS_PER_TILE` must therefore be even — enforced by the assert below.
pub const MAX_LIGHTS_PER_TILE: u32 = 64;

const _: () = assert!(MAX_LIGHTS_PER_TILE.is_multiple_of(2));
/// Uniform buffer size for clustered light compute `ClusterParams` (WGSL layout + tail padding).
pub const CLUSTER_PARAMS_UNIFORM_SIZE: u64 = 256;

/// References to the two storage buffers shared across all views.
///
/// `params_buffer` is intentionally **not** included: it is per-view and lives in
/// `PerViewFrameState::cluster_params_buffer` to prevent a CPU write-order race during
/// parallel rayon recording (multiple views share one `FrameUploadBatch`, last write wins).
#[derive(Clone, Copy)]
pub struct ClusterBufferRefs<'a> {
    /// One `u32` count per cluster (compute writes; fragment reads plain `u32`; one thread per cluster).
    pub cluster_light_counts: &'a wgpu::Buffer,
    /// Packed light indices: 2 × `u16` indices per `u32` slot. Slot `k` within cluster `c` lives
    /// at `u32` word `c * (MAX_LIGHTS_PER_TILE / 2) + (k >> 1)`, bits `(k & 1) * 16 ..+16`.
    pub cluster_light_indices: &'a wgpu::Buffer,
}

/// Shared cluster-buffer cache; grow-only high-water-mark across every active view so all views
/// can reference the same underlying storage. Bumps [`Self::version`] only when the underlying
/// buffers are reallocated (shrink is never performed), which drives bind-group invalidation for
/// every per-view consumer.
///
/// Correctness across views relies on WebGPU's in-order execution guarantee within a single
/// [`wgpu::Queue::submit`]: a view's clustered-light compute pass writes its lists, then its
/// raster reads them, all within one command buffer; the next view's command buffer (submitted
/// in order) overwrites only after the previous view's reads retire. This is why there is no
/// per-view offset — each view uses range `[0..view_cluster_count)` in turn.
///
/// When `stereo` is true, the counts and indices buffers are allocated at **2x** size so the
/// compute pass can write eye-0 clusters at `[0..N)` and eye-1 at `[N..2N)`.
pub struct ClusterBufferCache {
    cluster_light_counts: Option<wgpu::Buffer>,
    cluster_light_indices: Option<wgpu::Buffer>,
    /// High-water-mark configuration currently provisioned. Grow-only; never shrunk.
    cached_key: ClusterCacheKey,
    /// Incremented each time the buffers are reallocated to satisfy a larger request.
    pub version: u64,
}

/// Grow-only cache key: tracks the max viewport, Z slice count, and stereo flag observed so far.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct ClusterCacheKey {
    viewport: (u32, u32),
    cluster_count_z: u32,
    stereo: bool,
}

impl ClusterBufferCache {
    /// Empty cache; [`Self::ensure_buffers`] allocates on first use.
    pub fn new() -> Self {
        Self {
            cluster_light_counts: None,
            cluster_light_indices: None,
            cached_key: ClusterCacheKey::default(),
            version: 0,
        }
    }

    /// Ensures buffers fit `viewport`, `cluster_count_z`, and `stereo`. **Grow-only**: if the
    /// current allocation already covers the request (equal or larger on every axis, with a
    /// one-way `mono → stereo` transition), the existing buffers are reused and [`Self::version`]
    /// is not bumped. When a reallocation is required, the new high-water-mark replaces the
    /// previous allocation and `version` bumps to invalidate bind-group caches.
    ///
    /// When `stereo` is true, count and index buffers are doubled to hold per-eye cluster data.
    ///
    /// Returns [`None`] when cluster storage would exceed [`GpuLimits`] storage/buffer caps.
    pub fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        viewport: (u32, u32),
        cluster_count_z: u32,
        stereo: bool,
    ) -> Option<ClusterBufferRefs<'_>> {
        let (width, height) = viewport;
        if width == 0 || height == 0 {
            return None;
        }
        let new_max_w = width.max(self.cached_key.viewport.0);
        let new_max_h = height.max(self.cached_key.viewport.1);
        let new_max_z = cluster_count_z.max(self.cached_key.cluster_count_z);
        let new_max_stereo = stereo || self.cached_key.stereo;
        let cluster_count_x = new_max_w.div_ceil(TILE_SIZE);
        let cluster_count_y = new_max_h.div_ceil(TILE_SIZE);
        let clusters_per_eye = (cluster_count_x * cluster_count_y * new_max_z) as usize;
        let eye_multiplier = if new_max_stereo { 2 } else { 1 };
        let total_clusters = clusters_per_eye * eye_multiplier;
        let counts_bytes = (total_clusters * size_of::<u32>()) as u64;
        let indices_bytes =
            (total_clusters * (MAX_LIGHTS_PER_TILE as usize / 2) * size_of::<u32>()) as u64;
        let max_bind = limits.max_storage_buffer_binding_size();
        let max_buf = limits.max_buffer_size();
        if counts_bytes > max_bind
            || indices_bytes > max_bind
            || counts_bytes > max_buf
            || indices_bytes > max_buf
        {
            logger::warn!(
                "cluster buffers: max viewport {:?} stereo={} would need counts={} indices={} bytes; exceeds max_storage_buffer_binding_size ({}) or max_buffer_size ({})",
                (new_max_w, new_max_h),
                new_max_stereo,
                counts_bytes,
                indices_bytes,
                max_bind,
                max_buf
            );
            return None;
        }
        let new_key = ClusterCacheKey {
            viewport: (new_max_w, new_max_h),
            cluster_count_z: new_max_z,
            stereo: new_max_stereo,
        };
        if self.cluster_light_counts.is_none() || self.cached_key != new_key {
            self.version = self.version.wrapping_add(1);
            self.cached_key = new_key;

            self.cluster_light_counts = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_light_counts"),
                size: counts_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cluster_light_indices = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_light_indices"),
                size: indices_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.current_refs()
    }

    /// Returns refs to the currently-provisioned storage buffers, or [`None`] if not yet
    /// allocated. All views share these; see [`ClusterBufferCache`] for the ordering argument
    /// that makes GPU-side sharing safe. `params_buffer` is excluded — it is per-view.
    pub fn current_refs(&self) -> Option<ClusterBufferRefs<'_>> {
        Some(ClusterBufferRefs {
            cluster_light_counts: self.cluster_light_counts.as_ref()?,
            cluster_light_indices: self.cluster_light_indices.as_ref()?,
        })
    }
}

impl Default for ClusterBufferCache {
    fn default() -> Self {
        Self::new()
    }
}
