//! Cached `wgpu::BindGroup`s keyed by the buffer identities they reference.
//!
//! Skinning and blendshape compute dispatches reuse the same bind groups across many draws when
//! their underlying buffers are stable. Keys capture both the scratch generation (so bind groups
//! created against a now-grown scratch buffer are dropped) and the per-input buffer identities.

use std::sync::Arc;

use hashbrown::HashMap;

/// Cache key for a blendshape scatter bind group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendshapeBindGroupKey {
    /// Scratch buffer generation used by the params binding.
    pub scratch_generation: u64,
    /// Stable identity of the mesh sparse-delta buffer.
    pub sparse_buffer: u64,
    /// Stable identity of the output stream buffer.
    pub output_buffer: u64,
}

/// Cache key for a skinning bind group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkinningBindGroupKey {
    /// Scratch buffer generation used by palette, params, and dummy bindings.
    pub scratch_generation: u64,
    /// Stable identity of the source position buffer.
    pub src_positions: u64,
    /// Stable identity of the bone-index buffer.
    pub bone_indices: u64,
    /// Stable identity of the bone-weight buffer.
    pub bone_weights: u64,
    /// Stable identity of the raw influence offset buffer.
    pub bone_influence_offsets: u64,
    /// Stable identity of the raw influence buffer.
    pub bone_influences: u64,
    /// Stable identity of the destination position buffer.
    pub dst_positions: u64,
    /// Stable identity of the source normal buffer.
    pub src_normals: u64,
    /// Stable identity of the destination normal buffer.
    pub dst_normals: u64,
    /// Stable identity of the source tangent buffer or the dummy tangent buffer.
    pub src_tangents: u64,
    /// Stable identity of the destination tangent buffer or the dummy tangent buffer.
    pub dst_tangents: u64,
}

/// Returns a process-local identity for a `wgpu::Buffer` handle.
#[inline]
pub fn buffer_identity(buffer: &wgpu::Buffer) -> u64 {
    let ptr: *const wgpu::Buffer = buffer;
    let ptr = ptr as usize as u64;
    ptr ^ buffer.size().rotate_left(17)
}

/// Caches reusable bind groups for the blendshape scatter and skinning compute dispatches.
#[derive(Default)]
pub(super) struct BindGroupCaches {
    blendshape: HashMap<BlendshapeBindGroupKey, Arc<wgpu::BindGroup>>,
    skinning: HashMap<SkinningBindGroupKey, Arc<wgpu::BindGroup>>,
}

impl BindGroupCaches {
    #[inline]
    pub(super) fn blendshape(&self, key: BlendshapeBindGroupKey) -> Option<Arc<wgpu::BindGroup>> {
        self.blendshape.get(&key).cloned()
    }

    #[inline]
    pub(super) fn insert_blendshape(
        &mut self,
        key: BlendshapeBindGroupKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
        self.blendshape.insert(key, bind_group);
    }

    #[inline]
    pub(super) fn skinning(&self, key: SkinningBindGroupKey) -> Option<Arc<wgpu::BindGroup>> {
        self.skinning.get(&key).cloned()
    }

    #[inline]
    pub(super) fn insert_skinning(
        &mut self,
        key: SkinningBindGroupKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
        self.skinning.insert(key, bind_group);
    }

    /// Invalidates every cached bind group. Called when a scratch buffer is replaced so stale bind
    /// groups referencing the freed buffer are not reused.
    pub(super) fn clear_on_grow(&mut self) {
        self.blendshape.clear();
        self.skinning.clear();
    }
}
