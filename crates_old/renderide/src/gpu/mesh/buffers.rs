//! Cached wgpu vertex and index buffers for a mesh asset, plus submesh draw range helpers.

use std::sync::Arc;

// MeshPipeline removed — see gpu::pipeline::PipelineManager and concrete pipelines.

/// Merges contiguous `(index_start, index_count)` tuples for fewer draw calls; otherwise returns
/// a copy of `submeshes`.
fn merge_contiguous_submesh_ranges(submeshes: &[(u32, u32)]) -> Vec<(u32, u32)> {
    if submeshes.len() <= 1 {
        return submeshes.to_vec();
    }
    let contiguous = submeshes.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0);
    if contiguous {
        let first = submeshes[0].0;
        let total_count: u32 = submeshes.iter().map(|(_, c)| c).sum();
        vec![(first, total_count)]
    } else {
        submeshes.to_vec()
    }
}

/// Cached wgpu buffers for a mesh asset.
#[derive(Clone)]
pub struct GpuMeshBuffers {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    /// Interleaved position + normal + UV0 when the mesh has UV0 (for [`crate::gpu::PipelineVariant::PbrHostAlbedo`]).
    pub vertex_buffer_pos_normal_uv: Option<Arc<wgpu::Buffer>>,
    pub vertex_buffer_uv: Option<Arc<wgpu::Buffer>>,
    /// Canvas / UI layout: position, uv0, color, aux (tangent or text normal data).
    pub vertex_buffer_ui: Option<Arc<wgpu::Buffer>>,
    pub vertex_buffer_skinned: Option<Arc<wgpu::Buffer>>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub submeshes: Vec<(u32, u32)>,
    pub index_format: wgpu::IndexFormat,
    pub has_uvs: bool,
    /// Storage buffer for blendshape offsets (num_blendshapes × num_vertices × [`BlendshapeOffset`](crate::gpu::mesh::BlendshapeOffset)).
    /// Always `Some` for skinned meshes; uses a dummy 36-byte buffer when the mesh has no blendshapes.
    pub blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    /// Number of blendshape slots. Zero when mesh has no blendshapes (shader loop runs 0 times).
    pub num_blendshapes: u32,
}

impl GpuMeshBuffers {
    /// Returns references to the position+normal vertex and index buffers for normal-debug draws.
    #[inline(always)]
    pub fn normal_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (self.vertex_buffer.as_ref(), self.index_buffer.as_ref())
    }

    /// Returns position+normal+UV and index buffers when [`Self::vertex_buffer_pos_normal_uv`] exists.
    #[inline(always)]
    pub fn pos_normal_uv_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        Some((
            self.vertex_buffer_pos_normal_uv.as_ref()?.as_ref(),
            self.index_buffer.as_ref(),
        ))
    }

    /// Returns references to the vertex and index buffers for UV/overlay draws.
    ///
    /// Prefers UV vertex buffer when present. Caches Arc deref to reduce overhead in hot paths.
    #[inline(always)]
    pub fn uv_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        let vb = self
            .vertex_buffer_uv
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(self.vertex_buffer.as_ref());
        let ib = self.index_buffer.as_ref();
        (vb, ib)
    }

    /// Returns UI canvas vertex and index buffers when [`Self::vertex_buffer_ui`] was built.
    pub fn ui_canvas_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        let vb = self.vertex_buffer_ui.as_ref()?.as_ref();
        Some((vb, self.index_buffer.as_ref()))
    }

    /// Returns references to the skinned vertex and index buffers.
    ///
    /// Returns `None` when skinned vertex buffer is not available.
    #[inline(always)]
    pub fn skinned_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        let vb = self.vertex_buffer_skinned.as_ref()?.as_ref();
        let ib = self.index_buffer.as_ref();
        Some((vb, ib))
    }

    /// Returns draw ranges `(index_start, index_count)` for indexed drawing.
    ///
    /// When submeshes are contiguous in the index buffer, merges them into a single range
    /// to reduce draw calls. Otherwise returns per-submesh ranges.
    pub fn draw_ranges(&self) -> Vec<(u32, u32)> {
        merge_contiguous_submesh_ranges(&self.submeshes)
    }

    /// Per-submesh index ranges as uploaded (no contiguous merge). For multi-material draws.
    pub fn draw_ranges_per_submesh(&self) -> Vec<(u32, u32)> {
        self.submeshes.clone()
    }

    /// Indexed draw ranges for this mesh instance.
    ///
    /// When `index_range_override` is `Some` and `count > 0`, returns that single range only.
    /// Otherwise returns [`Self::draw_ranges`].
    pub fn effective_draw_ranges(
        &self,
        index_range_override: Option<(u32, u32)>,
    ) -> Vec<(u32, u32)> {
        if let Some((start, count)) = index_range_override {
            if count > 0 {
                vec![(start, count)]
            } else {
                vec![]
            }
        } else {
            self.draw_ranges()
        }
    }
}

#[cfg(test)]
mod draw_range_tests {
    use super::merge_contiguous_submesh_ranges;

    #[test]
    fn merge_contiguous_submesh_ranges_combines() {
        let merged = merge_contiguous_submesh_ranges(&[(0, 6), (6, 6)]);
        assert_eq!(merged, vec![(0, 12)]);
    }

    #[test]
    fn merge_contiguous_submesh_ranges_preserves_gap() {
        let merged = merge_contiguous_submesh_ranges(&[(0, 6), (10, 6)]);
        assert_eq!(merged, vec![(0, 6), (10, 6)]);
    }
}
