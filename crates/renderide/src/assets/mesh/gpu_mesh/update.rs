//! Lazy extended vertex streams and in-place mesh buffer updates ([`super::GpuMesh::write_in_place`]).

use std::sync::Arc;

use glam::Mat4;

use crate::shared::{MeshUploadData, MeshUploadHintFlag};

use super::super::gpu_mesh_hints::{
    derived_streams_compatible_for_in_place, mesh_upload_hint_any_selective,
    mesh_upload_hint_touches_vertex_streams, validated_submesh_ranges, wgpu_index_format,
};
use super::super::layout::{
    blendshape_deform_is_active, compute_index_count, compute_vertex_stride, extract_bind_poses,
    synthetic_bone_data_for_blendshape_only, MeshBufferLayout,
};
use super::super::upload_impl::{
    upload_default_extended_vertex_streams, upload_extended_vertex_streams,
};
use super::{
    blendshape_and_deform_buffers_match_for_in_place, compatible_for_in_place_real_skeleton,
    compatible_for_in_place_synthetic_blendshape_skeleton, extended_vertex_stream_bytes,
    extended_vertex_stream_source_from_raw, write_in_place_blendshape_buffer,
    write_in_place_bone_buffers, write_in_place_index_buffer,
    write_in_place_vertex_and_derived_streams, GpuMesh, MeshInPlaceWriteContext,
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
        blendshape_deform_is_active(
            self.num_blendshapes,
            &self.blendshape_shape_frame_spans,
            &self.blendshape_frame_ranges,
            blend_weights,
        ) && self.blendshape_sparse_buffer.is_some()
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
                    source.vertex_bytes.as_ref(),
                    vc_usize,
                    self.vertex_stride as usize,
                    source.vertex_attributes.as_ref(),
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

        let needs_bone_buffers = data.bone_count > 0 || (use_blendshapes && data.vertex_count > 0);

        let no_gpu_bones = self.bone_counts_buffer.is_none()
            && self.bone_indices_buffer.is_none()
            && self.bone_weights_vec4_buffer.is_none()
            && self.bind_poses_buffer.is_none();
        let no_gpu_blend = self.blendshape_sparse_buffer.is_none()
            && self.blendshape_shape_descriptor_buffer.is_none()
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

        if use_blendshapes && data.vertex_count > 0 {
            return compatible_for_in_place_synthetic_blendshape_skeleton(
                self,
                data,
                vertex_slice,
                vc_usize,
                vertex_stride_us,
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

        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();
        let needs_bone_buffers = data.bone_count > 0 || (use_blendshapes && data.vertex_count > 0);
        let synthetic_bones = data.bone_count == 0 && use_blendshapes && data.vertex_count > 0;

        let full = !mesh_upload_hint_any_selective(hint);
        let write_vertex = full || mesh_upload_hint_touches_vertex_streams(hint);
        let write_ib = full || hint.geometry();
        let write_bone_weights = full || hint.bone_weights();
        let write_bind_poses = full || hint.bind_poses();
        let write_blend = full || hint.blendshapes();

        let want_submeshes = validated_submesh_ranges(&data.submeshes, self.index_count);

        write_in_place_vertex_and_derived_streams(
            &MeshInPlaceWriteContext {
                mesh: self,
                queue,
                raw,
                layout,
                data,
                vertex_count: vc_usize,
                vertex_stride: vertex_stride_us,
            },
            write_vertex,
        );
        write_in_place_index_buffer(self, queue, raw, layout, write_ib);
        write_in_place_bone_buffers(
            &MeshInPlaceWriteContext {
                mesh: self,
                queue,
                raw,
                layout,
                data,
                vertex_count: vc_usize,
                vertex_stride: vertex_stride_us,
            },
            needs_bone_buffers,
            synthetic_bones,
            full,
            write_bone_weights,
            write_bind_poses,
        )?;
        write_in_place_blendshape_buffer(self, queue, raw, layout, data, write_blend)?;

        let mut skinning = self.skinning_bind_matrices.clone();
        if data.bone_count > 0 && (full || write_bind_poses) {
            let bp_raw =
                &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
            if let Some(arr) = extract_bind_poses(bp_raw, data.bone_count as usize) {
                skinning = arr.iter().map(Mat4::from_cols_array_2d).collect();
            }
        } else if synthetic_bones && (full || write_bone_weights || write_bind_poses) {
            let (bind_poses_arr, _, _) = synthetic_bone_data_for_blendshape_only(data.vertex_count);
            skinning = bind_poses_arr
                .iter()
                .map(Mat4::from_cols_array_2d)
                .collect();
        }

        let has_extended_gpu_streams = self.extended_vertex_streams_ready();
        let extended_vertex_stream_source = if write_vertex && !has_extended_gpu_streams {
            extended_vertex_stream_source_from_raw(raw, data, layout)
        } else if write_vertex {
            None
        } else {
            self.extended_vertex_stream_source.clone()
        };

        Some(Self {
            asset_id: self.asset_id,
            vertex_buffer: Arc::clone(&self.vertex_buffer),
            index_buffer: Arc::clone(&self.index_buffer),
            index_format: self.index_format,
            index_count: self.index_count,
            submeshes: want_submeshes,
            vertex_count: self.vertex_count,
            vertex_stride: self.vertex_stride,
            bounds: data.bounds,
            bone_counts_buffer: self.bone_counts_buffer.clone(),
            bone_indices_buffer: self.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: self.bone_weights_vec4_buffer.clone(),
            bind_poses_buffer: self.bind_poses_buffer.clone(),
            blendshape_sparse_buffer: self.blendshape_sparse_buffer.clone(),
            blendshape_shape_descriptor_buffer: self.blendshape_shape_descriptor_buffer.clone(),
            blendshape_frame_ranges: self.blendshape_frame_ranges.clone(),
            blendshape_shape_frame_spans: self.blendshape_shape_frame_spans.clone(),
            num_blendshapes: self.num_blendshapes,
            positions_buffer: self.positions_buffer.clone(),
            normals_buffer: self.normals_buffer.clone(),
            uv0_buffer: self.uv0_buffer.clone(),
            color_buffer: self.color_buffer.clone(),
            tangent_buffer: self.tangent_buffer.clone(),
            uv1_buffer: self.uv1_buffer.clone(),
            uv2_buffer: self.uv2_buffer.clone(),
            uv3_buffer: self.uv3_buffer.clone(),
            extended_vertex_stream_source,
            has_skeleton: self.has_skeleton,
            skinning_bind_matrices: skinning,
            resident_bytes: self.resident_bytes,
        })
    }
}
