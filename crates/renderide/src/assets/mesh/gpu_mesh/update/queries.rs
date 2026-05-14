//! Per-stream readiness and deform-support queries on [`GpuMesh`].

use super::super::super::layout::blendshape_deform_is_active;
use super::super::GpuMesh;

impl GpuMesh {
    /// `true` when [`Self::positions_buffer`] and [`Self::normals_buffer`] exist for the debug mesh path.
    pub fn debug_streams_ready(&self) -> bool {
        self.positions_buffer.is_some() && self.normals_buffer.is_some()
    }

    /// `true` when this mesh has the tangent and UV1-UV3 streams required by extended embedded shaders.
    pub fn extended_vertex_streams_ready(&self) -> bool {
        self.tangent_buffer.is_some()
            && self.uv1_buffer.is_some()
            && self.uv2_buffer.is_some()
            && self.uv3_buffer.is_some()
    }

    /// `true` when this mesh has the standalone UV1 stream required by compact UV1 shaders.
    pub fn uv1_vertex_stream_ready(&self) -> bool {
        self.uv1_buffer.is_some()
    }

    /// `true` when this mesh has the standalone tangent stream required by compact shaders.
    pub fn tangent_vertex_stream_ready(&self) -> bool {
        self.tangent_buffer.is_some()
    }

    /// `true` when this mesh has the raw tangent payload stream required by UI payload shaders.
    pub fn raw_tangent_vertex_stream_ready(&self) -> bool {
        self.raw_tangent_buffer.is_some()
    }

    /// `true` when this mesh has the standalone UV2 stream required by compact shaders.
    pub fn uv2_vertex_stream_ready(&self) -> bool {
        self.uv2_buffer.is_some()
    }

    /// `true` when this mesh has the standalone UV3 stream required by compact shaders.
    pub fn uv3_vertex_stream_ready(&self) -> bool {
        self.uv3_buffer.is_some()
    }

    /// `true` when this mesh has the packed UV0-UV7 stream required by wide UV shaders.
    pub fn wide_uv_vertex_stream_ready(&self) -> bool {
        self.wide_uv_buffer.is_some()
    }

    /// Returns whether this mesh has every GPU stream needed to produce world-space skinned output.
    pub fn supports_world_space_skin_deform(&self, bone_transform_indices: Option<&[i32]>) -> bool {
        bone_transform_indices.is_some()
            && self.has_skeleton
            && self.normals_buffer.is_some()
            && self.bone_indices_buffer.is_some()
            && self.bone_weights_vec4_buffer.is_some()
            && !self.skinning_bind_matrices.is_empty()
    }

    /// Returns whether the mesh has valid sparse blendshape data and at least one active shape.
    pub fn supports_active_blendshape_deform(&self, blend_weights: &[f32]) -> bool {
        let has_supported_channel = self.blendshape_has_position_deltas
            || (self.blendshape_has_normal_deltas && self.normals_buffer.is_some())
            || (self.blendshape_has_tangent_deltas && self.tangent_buffer.is_some());
        blendshape_deform_is_active(
            self.num_blendshapes,
            &self.blendshape_shape_frame_spans,
            &self.blendshape_frame_ranges,
            blend_weights,
        ) && self.blendshape_sparse_buffer.is_some()
            && has_supported_channel
    }

    /// Returns whether active blendshape tangent deltas can affect tangent-space shading.
    ///
    /// Unlike [`Self::supports_active_blendshape_deform`], this does not require the tangent stream
    /// to already exist. Draw collection uses it to request lazy tangent-stream pre-warm before the
    /// frame-global deform pass records.
    pub fn supports_active_tangent_blendshape_deform(&self, blend_weights: &[f32]) -> bool {
        self.blendshape_has_tangent_deltas
            && blendshape_deform_is_active(
                self.num_blendshapes,
                &self.blendshape_shape_frame_spans,
                &self.blendshape_frame_ranges,
                blend_weights,
            )
            && self.blendshape_sparse_buffer.is_some()
    }
}
