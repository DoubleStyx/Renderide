//! Lazy extended vertex streams and in-place mesh buffer updates ([`super::GpuMesh::write_in_place`]).

use std::sync::Arc;

use glam::Mat4;

use crate::materials::RasterPrimitiveTopology;
use crate::shared::{MeshUploadData, MeshUploadHintFlag};

use super::super::gpu_mesh_hints::{
    derived_streams_compatible_for_in_place, mesh_upload_hint_any_selective,
    mesh_upload_hint_touches_vertex_streams, validated_submesh_ranges,
    validated_submesh_topologies, wgpu_index_format,
};
use super::super::layout::{
    MeshBufferLayout, blendshape_deform_is_active, compute_index_count, compute_vertex_stride,
    extract_bind_poses,
};
use super::super::upload_impl::{
    ExtendedVertexUploadSource, upload_default_extended_vertex_streams,
    upload_default_uv1_vertex_stream, upload_extended_vertex_streams, upload_uv1_vertex_stream,
};
use super::{
    BoneBufferWriteHints, ExtendedVertexStreamSource, GpuMesh, MeshInPlaceWriteContext,
    blendshape_and_deform_buffers_match_for_in_place, compatible_for_in_place_real_skeleton,
    extended_vertex_stream_bytes, extended_vertex_stream_source_from_raw,
    write_in_place_blendshape_buffer, write_in_place_bone_buffers, write_in_place_index_buffer,
    write_in_place_vertex_and_derived_streams,
};

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

    /// Creates tangent / UV1-3 streams the first time an embedded shader needs them.
    pub(crate) fn ensure_extended_vertex_streams(&mut self, device: &wgpu::Device) -> bool {
        profiling::scope!("asset::mesh_ensure_extended_vertex_streams");
        if self.extended_vertex_streams_ready() {
            return true;
        }

        let old_bytes = extended_vertex_stream_bytes(self);
        let vc_usize = self.vertex_count as usize;
        let (tangent_buffer, uv1_buffer, uv2_buffer, uv3_buffer) =
            if let Some(source) = self.extended_vertex_stream_source.as_ref() {
                upload_extended_vertex_streams(
                    device,
                    self.asset_id,
                    ExtendedVertexUploadSource {
                        vertex_slice: source.vertex_bytes.as_ref(),
                        index_slice: source.index_bytes.as_ref(),
                        vertex_count: vc_usize,
                        vertex_stride: self.vertex_stride as usize,
                        vertex_attributes: source.vertex_attributes.as_ref(),
                        index_format: source.index_format,
                        submeshes: source.submeshes.as_ref(),
                    },
                )
            } else {
                upload_default_extended_vertex_streams(device, self.asset_id, vc_usize)
            };

        if tangent_buffer.is_none()
            || uv1_buffer.is_none()
            || uv2_buffer.is_none()
            || uv3_buffer.is_none()
        {
            return false;
        }

        //perf xlinka: pay the 40 bytes/vertex only for meshes that hit extended shaders.
        self.tangent_buffer = tangent_buffer;
        self.uv1_buffer = uv1_buffer;
        self.uv2_buffer = uv2_buffer;
        self.uv3_buffer = uv3_buffer;
        let new_bytes = extended_vertex_stream_bytes(self);
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        self.extended_vertex_stream_source = None;
        true
    }

    /// Creates a UV1 stream the first time an embedded shader needs it without other extended streams.
    pub(crate) fn ensure_uv1_vertex_stream(&mut self, device: &wgpu::Device) -> bool {
        profiling::scope!("asset::mesh_ensure_uv1_vertex_stream");
        if self.uv1_vertex_stream_ready() {
            return true;
        }

        let old_bytes = self.uv1_buffer.as_ref().map_or(0, |buffer| buffer.size());
        let vc_usize = self.vertex_count as usize;
        let uv1_buffer = if let Some(source) = self.extended_vertex_stream_source.as_ref() {
            upload_uv1_vertex_stream(
                device,
                self.asset_id,
                source.vertex_bytes.as_ref(),
                vc_usize,
                self.vertex_stride as usize,
                source.vertex_attributes.as_ref(),
            )
        } else {
            upload_default_uv1_vertex_stream(device, self.asset_id, vc_usize)
        };

        let Some(uv1_buffer) = uv1_buffer else {
            return false;
        };
        self.uv1_buffer = Some(uv1_buffer);
        let new_bytes = self.uv1_buffer.as_ref().map_or(0, |buffer| buffer.size());
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(old_bytes)
            .saturating_add(new_bytes);
        true
    }

    /// Whether `data`/`layout` match this mesh's buffer sizes and optional derived streams so we can
    /// [`Self::write_in_place`] instead of allocating new buffers.
    pub(crate) fn compatible_for_in_place_update(
        &self,
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        raw: &[u8],
    ) -> bool {
        profiling::scope!("asset::mesh_check_in_place_update");
        if raw.len() < layout.total_buffer_length {
            return false;
        }
        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();
        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let index_count = compute_index_count(&data.submeshes);
        let index_count_u32 = index_count.max(0) as u32;
        if self.vertex_stride != vertex_stride
            || self.vertex_count != data.vertex_count.max(0) as u32
            || self.index_count != index_count_u32
            || self.index_format != wgpu_index_format(data.index_buffer_format)
        {
            return false;
        }
        if self.vertex_buffer.size() != layout.vertex_size as u64
            || self.index_buffer.size() != layout.index_buffer_length as u64
        {
            return false;
        }

        let vc_usize = data.vertex_count.max(0) as usize;
        let vertex_stride_us = vertex_stride as usize;
        let vertex_slice = &raw[..layout.vertex_size];

        let needs_bone_buffers = data.bone_count > 0;

        let no_gpu_bones = self.bone_counts_buffer.is_none()
            && self.bone_indices_buffer.is_none()
            && self.bone_weights_vec4_buffer.is_none()
            && self.bind_poses_buffer.is_none();
        let no_gpu_blend = self.blendshape_sparse_buffer.is_none()
            && self.num_blendshapes == 0
            && self.blendshape_frame_ranges.is_empty()
            && self.blendshape_shape_frame_spans.is_empty();

        let data_static = data.bone_count == 0 && !use_blendshapes;
        let gpu_static =
            !self.has_skeleton && self.num_blendshapes == 0 && no_gpu_bones && no_gpu_blend;

        if data_static && gpu_static {
            return derived_streams_compatible_for_in_place(
                self,
                vertex_slice,
                data,
                vc_usize,
                vertex_stride_us,
            );
        }

        if self.has_skeleton != (data.bone_count > 0) {
            return false;
        }

        if !blendshape_and_deform_buffers_match_for_in_place(
            self,
            data,
            layout,
            raw,
            use_blendshapes,
        ) {
            return false;
        }

        if !needs_bone_buffers {
            if self.bone_counts_buffer.is_some()
                || self.bind_poses_buffer.is_some()
                || self.bone_indices_buffer.is_some()
                || self.bone_weights_vec4_buffer.is_some()
            {
                return false;
            }
            return derived_streams_compatible_for_in_place(
                self,
                vertex_slice,
                data,
                vc_usize,
                vertex_stride_us,
            );
        }

        if data.bone_count > 0 {
            return compatible_for_in_place_real_skeleton(
                self,
                data,
                layout,
                raw,
                vc_usize,
                vertex_stride_us,
                vertex_slice,
            );
        }

        false
    }

    /// Overwrites vertex, index, and optional bone/blendshape/derived stream data using
    /// [`wgpu::Queue::write_buffer`], honoring [`MeshUploadHintFlag`] when set (otherwise full writes).
    pub(crate) fn write_in_place(
        &self,
        queue: &wgpu::Queue,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        hint: MeshUploadHintFlag,
    ) -> Option<GpuMesh> {
        profiling::scope!("asset::mesh_write_in_place");
        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let vc_usize = data.vertex_count.max(0) as usize;
        let vertex_stride_us = vertex_stride as usize;

        let deform = classify_in_place_deform_streams(data);
        let flags = decode_in_place_write_flags(hint);

        let (want_submeshes, want_submesh_topologies) = {
            profiling::scope!("asset::mesh_write_in_place::validate_submeshes");
            (
                validated_submesh_ranges(&data.submeshes, self.index_count),
                validated_submesh_topologies(&data.submeshes, self.index_count),
            )
        };

        let write_context = MeshInPlaceWriteContext {
            mesh: self,
            queue,
            raw,
            layout,
            data,
            vertex_count: vc_usize,
            vertex_stride: vertex_stride_us,
        };

        {
            profiling::scope!("asset::mesh_write_in_place::write_vertex_and_derived");
            write_in_place_vertex_and_derived_streams(
                &write_context,
                flags.write_vertex,
                flags.write_index,
            );
        }
        {
            profiling::scope!("asset::mesh_write_in_place::write_index");
            write_in_place_index_buffer(self, queue, raw, layout, flags.write_index);
        }
        {
            profiling::scope!("asset::mesh_write_in_place::write_bones");
            write_in_place_bone_buffers(
                &write_context,
                BoneBufferWriteHints {
                    needs_bone_buffers: deform.needs_bone_buffers,
                    full: flags.full,
                    write_bone_weights: flags.write_bone_weights,
                    write_bind_poses: flags.write_bind_poses,
                },
            )?;
        }
        {
            profiling::scope!("asset::mesh_write_in_place::write_blendshapes");
            write_in_place_blendshape_buffer(self, queue, raw, layout, data, flags.write_blend)?;
        }

        let skinning = updated_in_place_skinning_matrices(self, raw, data, layout, flags);

        let extended_vertex_stream_source = {
            profiling::scope!("asset::mesh_write_in_place::update_extended_stream_source");
            updated_extended_vertex_stream_source(
                self,
                raw,
                data,
                layout,
                flags.write_vertex,
                flags.write_index,
            )
        };

        Some(rebuild_mesh_after_in_place_write(
            self,
            data,
            want_submeshes,
            want_submesh_topologies,
            skinning,
            extended_vertex_stream_source,
        ))
    }
}

#[derive(Clone, Copy)]
struct InPlaceDeformStreams {
    needs_bone_buffers: bool,
}

#[derive(Clone, Copy)]
struct InPlaceWriteFlags {
    full: bool,
    write_vertex: bool,
    write_index: bool,
    write_bone_weights: bool,
    write_bind_poses: bool,
    write_blend: bool,
}

fn classify_in_place_deform_streams(data: &MeshUploadData) -> InPlaceDeformStreams {
    profiling::scope!("asset::mesh_write_in_place::classify_deform_streams");
    InPlaceDeformStreams {
        needs_bone_buffers: data.bone_count > 0,
    }
}

fn decode_in_place_write_flags(hint: MeshUploadHintFlag) -> InPlaceWriteFlags {
    profiling::scope!("asset::mesh_write_in_place::decode_hints");
    let full = !mesh_upload_hint_any_selective(hint);
    InPlaceWriteFlags {
        full,
        write_vertex: full || mesh_upload_hint_touches_vertex_streams(hint),
        write_index: full || hint.geometry(),
        write_bone_weights: full || hint.bone_weights(),
        write_bind_poses: full || hint.bind_poses(),
        write_blend: full || hint.blendshapes(),
    }
}

fn updated_in_place_skinning_matrices(
    mesh: &GpuMesh,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    flags: InPlaceWriteFlags,
) -> Vec<Mat4> {
    profiling::scope!("asset::mesh_write_in_place::update_skinning_matrices");
    let mut skinning = mesh.skinning_bind_matrices.clone();
    if data.bone_count > 0 && (flags.full || flags.write_bind_poses) {
        let bp_raw =
            &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
        if let Some(arr) = extract_bind_poses(bp_raw, data.bone_count as usize) {
            skinning = arr.iter().map(Mat4::from_cols_array_2d).collect();
        }
    }
    skinning
}

fn rebuild_mesh_after_in_place_write(
    mesh: &GpuMesh,
    data: &MeshUploadData,
    submeshes: Vec<(u32, u32)>,
    submesh_topologies: Vec<RasterPrimitiveTopology>,
    skinning: Vec<Mat4>,
    extended_vertex_stream_source: Option<ExtendedVertexStreamSource>,
) -> GpuMesh {
    profiling::scope!("asset::mesh_write_in_place::rebuild_metadata");
    GpuMesh {
        asset_id: mesh.asset_id,
        vertex_buffer: Arc::clone(&mesh.vertex_buffer),
        index_buffer: Arc::clone(&mesh.index_buffer),
        index_format: mesh.index_format,
        index_count: mesh.index_count,
        submeshes,
        submesh_topologies,
        vertex_count: mesh.vertex_count,
        vertex_stride: mesh.vertex_stride,
        bounds: data.bounds,
        bone_counts_buffer: mesh.bone_counts_buffer.clone(),
        bone_indices_buffer: mesh.bone_indices_buffer.clone(),
        bone_weights_vec4_buffer: mesh.bone_weights_vec4_buffer.clone(),
        bind_poses_buffer: mesh.bind_poses_buffer.clone(),
        blendshape_sparse_buffer: mesh.blendshape_sparse_buffer.clone(),
        blendshape_frame_ranges: mesh.blendshape_frame_ranges.clone(),
        blendshape_shape_frame_spans: mesh.blendshape_shape_frame_spans.clone(),
        num_blendshapes: mesh.num_blendshapes,
        blendshape_has_position_deltas: mesh.blendshape_has_position_deltas,
        blendshape_has_normal_deltas: mesh.blendshape_has_normal_deltas,
        blendshape_has_tangent_deltas: mesh.blendshape_has_tangent_deltas,
        positions_buffer: mesh.positions_buffer.clone(),
        normals_buffer: mesh.normals_buffer.clone(),
        uv0_buffer: mesh.uv0_buffer.clone(),
        color_buffer: mesh.color_buffer.clone(),
        tangent_buffer: mesh.tangent_buffer.clone(),
        uv1_buffer: mesh.uv1_buffer.clone(),
        uv2_buffer: mesh.uv2_buffer.clone(),
        uv3_buffer: mesh.uv3_buffer.clone(),
        extended_vertex_stream_source,
        has_skeleton: mesh.has_skeleton,
        skinning_bind_matrices: skinning,
        resident_bytes: mesh.resident_bytes,
    }
}

fn updated_extended_vertex_stream_source(
    mesh: &GpuMesh,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    write_vertex: bool,
    write_index: bool,
) -> Option<ExtendedVertexStreamSource> {
    if !write_vertex && !write_index {
        return mesh.extended_vertex_stream_source.clone();
    }
    if mesh.extended_vertex_streams_ready() {
        return None;
    }
    extended_vertex_stream_source_from_raw(raw, data, layout)
}
