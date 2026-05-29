//! Counters reported by [`super::TransientPool`] for diagnostics and HUD readout.

use crate::gpu_resource::CacheStats;

/// Pool statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransientPoolMetrics {
    /// Texture-pool cache counters.
    pub texture_cache: CacheStats,
    /// Transient texture-view cache counters.
    pub texture_view_cache: CacheStats,
    /// Compatible texture views retained by transient texture slots.
    pub texture_view_cache_entries: usize,
    /// Texture reuse hits.
    pub texture_hits: usize,
    /// Texture allocation misses.
    pub texture_misses: usize,
    /// Buffer-pool cache counters.
    pub buffer_cache: CacheStats,
    /// Buffer reuse hits.
    pub buffer_hits: usize,
    /// Buffer allocation misses.
    pub buffer_misses: usize,
    /// Pool texture slots that currently hold GPU [`wgpu::Texture`] handles (after GC drops dead entries).
    pub retained_textures: usize,
    /// Pool buffer slots that currently hold GPU [`wgpu::Buffer`] handles.
    pub retained_buffers: usize,
}
