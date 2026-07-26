//! [`GpuMesh::compatible_for_in_place_update`] and [`GpuMesh::write_in_place`] plus their free helpers.

use std::sync::Arc;

use glam::Mat4;

use crate::render_contract::RasterPrimitiveTopology;
use crate::shared::{
    MeshUploadData, MeshUploadHintFlag, SubmeshBufferDescriptor, VertexAttributeDescriptor,
};

use super::super::super::layout::{
    MeshBufferLayout, compute_index_count, compute_vertex_stride, extract_bind_poses,
};
use super::super::hints::{
    derived_streams_compatible_for_in_place, mesh_upload_hint_any_selective,
    mesh_upload_hint_touches_vertex_streams, validated_submesh_ranges,
    validated_submesh_topologies, wgpu_index_format,
};
use super::super::upload::{MeshGpuUploadContext, queue_init_buffer_size_matches};
use super::super::{
    ExtendedVertexStreamSource, GpuMesh, MeshDerivedStreamDemand, MeshDerivedStreamMask,
    MeshGeometryStorage, MeshInPlaceUploadPlan, extended_vertex_stream_source_from_raw,
    geometry_storage_after_index_validation, rebuildable_derived_stream_mask,
    shared_static_indices_are_valid,
};
use super::in_place_buffers::{
    BoneBufferWriteHints, MeshInPlaceWriteContext,
    blendshape_and_deform_buffers_match_for_in_place, compatible_for_in_place_real_skeleton,
    write_in_place_blendshape_buffer, write_in_place_bone_buffers, write_in_place_index_buffer,
    write_in_place_vertex_and_derived_streams,
};

impl GpuMesh {
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
        let Some(vertex_buffer) = self.vertex_buffer.as_ref() else {
            return false;
        };
        let Some(index_buffer) = self.index_buffer.as_ref() else {
            return false;
        };
        if !queue_init_buffer_size_matches(vertex_buffer.size(), layout.vertex_size)
            || !queue_init_buffer_size_matches(index_buffer.size(), layout.index_buffer_length)
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
            && self.bone_influence_offsets_buffer.is_none()
            && self.bone_influences_buffer.is_none()
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
                || self.bone_influence_offsets_buffer.is_some()
                || self.bone_influences_buffer.is_some()
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

    /// Builds an in-place upload plan after validating allocation/layout compatibility.
    ///
    /// Exact retained-byte matches allow later stages to omit unchanged vertex/index-derived CPU
    /// work and GPU writes. `None` requires a full upload.
    pub(crate) fn plan_in_place_upload(
        &self,
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        raw: &[u8],
        demand: MeshDerivedStreamDemand,
    ) -> Option<MeshInPlaceUploadPlan> {
        if !self.compatible_for_in_place_update(data, layout, raw) {
            return None;
        }
        let Some(source) = self.extended_vertex_stream_source.as_ref() else {
            return Some(MeshInPlaceUploadPlan::default());
        };
        let plan = retained_geometry_match(source, raw, data, layout)
            .restricted_for_resident_state(
                self.geometry_storage,
                self.dedicated_derived_stream_mask(),
                self.derived_stream_state.dirty_mask,
                self.tangent_fallback_mode,
                demand,
            );
        #[cfg(feature = "tracy")]
        {
            if plan.vertex_unchanged {
                profiling::scope!("asset::mesh_in_place_vertex_write_elided");
            }
            if plan.index_unchanged {
                profiling::scope!("asset::mesh_in_place_index_write_elided");
            }
        }
        Some(plan)
    }

    /// Overwrites vertex, index, and optional bone/blendshape/derived stream data through
    /// the mesh upload sink, honoring [`MeshUploadHintFlag`] when set (otherwise full writes).
    pub(crate) fn write_in_place(
        &self,
        upload_ctx: MeshGpuUploadContext<'_>,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        hint: MeshUploadHintFlag,
        plan: MeshInPlaceUploadPlan,
    ) -> Option<GpuMesh> {
        profiling::scope!("asset::mesh_write_in_place");
        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let vc_usize = data.vertex_count.max(0) as usize;
        let vertex_stride_us = vertex_stride as usize;

        let deform = classify_in_place_deform_streams(data);
        let flags = decode_in_place_write_flags(hint).without_unchanged_geometry(plan);

        let (want_submeshes, want_submesh_topologies) = {
            profiling::scope!("asset::mesh_write_in_place::validate_submeshes");
            (
                validated_submesh_ranges(&data.submeshes, self.index_count),
                validated_submesh_topologies(&data.submeshes, self.index_count),
            )
        };

        let write_context = MeshInPlaceWriteContext {
            mesh: self,
            upload_sink: upload_ctx.upload_sink,
            raw,
            layout,
            data,
            vertex_count: vc_usize,
            vertex_stride: vertex_stride_us,
            demand_mask: upload_ctx.derived_stream_demand.mask,
            prepared_derived_streams: upload_ctx.prepared_derived_streams,
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
            write_in_place_index_buffer(
                self,
                upload_ctx.upload_sink,
                raw,
                layout,
                flags.write_index,
            );
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
            write_in_place_blendshape_buffer(
                self,
                upload_ctx.upload_sink,
                raw,
                layout,
                data,
                flags.write_blend,
            )?;
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
        let derived_stream_state = updated_derived_stream_state(
            self,
            upload_ctx.derived_stream_demand,
            extended_vertex_stream_source.as_ref(),
            flags.write_vertex,
            flags.write_index,
        );
        let indices_valid = !self.uses_shared_static_geometry()
            || shared_static_indices_are_valid(raw, data, layout, self.index_count);
        let geometry_storage =
            geometry_storage_after_index_validation(self.geometry_storage, indices_valid);

        Some(rebuild_mesh_after_in_place_write(
            self,
            data,
            want_submeshes,
            want_submesh_topologies,
            skinning,
            extended_vertex_stream_source,
            derived_stream_state,
            geometry_storage,
        ))
    }
}

impl MeshInPlaceUploadPlan {
    /// Restricts derived-stream preparation to geometry sources that actually changed.
    pub(crate) fn changed_derived_stream_demand(
        self,
        demand: MeshDerivedStreamDemand,
    ) -> MeshDerivedStreamDemand {
        let mut changed = MeshDerivedStreamMask::EMPTY;
        if !self.vertex_unchanged {
            changed |= MeshDerivedStreamMask::VERTEX_DERIVED;
        }
        if !self.index_unchanged {
            changed |= MeshDerivedStreamMask::INDEX_DERIVED;
        }
        MeshDerivedStreamDemand {
            mask: demand.mask.intersection(changed),
            tangent_fallback_mode: demand.tangent_fallback_mode,
        }
    }

    fn restricted_for_resident_state(
        mut self,
        geometry_storage: MeshGeometryStorage,
        dedicated_streams: MeshDerivedStreamMask,
        dirty_streams: MeshDerivedStreamMask,
        resident_tangent_fallback: crate::render_contract::EmbeddedTangentFallbackMode,
        demand: MeshDerivedStreamDemand,
    ) -> Self {
        if geometry_storage != MeshGeometryStorage::Dedicated {
            return Self::default();
        }
        let clean_streams = dedicated_streams.without(dirty_streams);
        let mut required = demand.mask.without(clean_streams);
        if demand.mask.contains(MeshDerivedStreamMask::TANGENT)
            && demand.tangent_fallback_mode > resident_tangent_fallback
        {
            required |= MeshDerivedStreamMask::TANGENT;
        }
        if required.intersects(MeshDerivedStreamMask::VERTEX_DERIVED) {
            self.vertex_unchanged = false;
        }
        if required.intersects(MeshDerivedStreamMask::INDEX_DERIVED) {
            self.index_unchanged = false;
        }
        self
    }
}

fn retained_geometry_match(
    source: &ExtendedVertexStreamSource,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
) -> MeshInPlaceUploadPlan {
    profiling::scope!("asset::mesh_retained_geometry_match");
    let vertex_unchanged =
        vertex_attributes_match(source.vertex_attributes.as_ref(), &data.vertex_attributes)
            && raw
                .get(..layout.vertex_size)
                .is_some_and(|vertex| vertex == source.vertex_bytes.as_ref());

    let index_end = layout
        .index_buffer_start
        .checked_add(layout.index_buffer_length);
    let index_unchanged = source.index_format == data.index_buffer_format
        && submesh_geometry_matches(source.submeshes.as_ref(), &data.submeshes)
        && index_end
            .and_then(|end| raw.get(layout.index_buffer_start..end))
            .is_some_and(|index| index == source.index_bytes.as_ref());

    MeshInPlaceUploadPlan {
        vertex_unchanged,
        index_unchanged,
    }
}

fn vertex_attributes_match(
    previous: &[VertexAttributeDescriptor],
    current: &[VertexAttributeDescriptor],
) -> bool {
    previous.len() == current.len()
        && previous.iter().zip(current).all(|(previous, current)| {
            previous.attribute == current.attribute
                && previous.format == current.format
                && previous.dimensions == current.dimensions
        })
}

fn submesh_geometry_matches(
    previous: &[SubmeshBufferDescriptor],
    current: &[SubmeshBufferDescriptor],
) -> bool {
    previous.len() == current.len()
        && previous.iter().zip(current).all(|(previous, current)| {
            previous.topology == current.topology
                && previous.index_start == current.index_start
                && previous.index_count == current.index_count
        })
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

impl InPlaceWriteFlags {
    fn without_unchanged_geometry(mut self, plan: MeshInPlaceUploadPlan) -> Self {
        if plan.vertex_unchanged {
            self.write_vertex = false;
        }
        if plan.index_unchanged {
            self.write_index = false;
        }
        self
    }
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
        write_vertex: full || hint.geometry() || mesh_upload_hint_touches_vertex_streams(hint),
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
    derived_stream_state: super::super::MeshDerivedStreamState,
    geometry_storage: MeshGeometryStorage,
) -> GpuMesh {
    profiling::scope!("asset::mesh_write_in_place::rebuild_metadata");
    GpuMesh {
        asset_id: mesh.asset_id,
        dynamic_geometry: mesh.dynamic_geometry || data.upload_hint.flags.dynamic(),
        geometry_storage,
        shared_static_resident_streams: if geometry_storage == MeshGeometryStorage::SharedStatic {
            mesh.shared_static_resident_streams
        } else {
            MeshDerivedStreamMask::EMPTY
        },
        vertex_buffer: mesh.vertex_buffer.clone(),
        index_buffer: mesh.index_buffer.as_ref().map(Arc::clone),
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
        bone_influence_offsets_buffer: mesh.bone_influence_offsets_buffer.clone(),
        bone_influences_buffer: mesh.bone_influences_buffer.clone(),
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
        raw_tangent_buffer: mesh.raw_tangent_buffer.clone(),
        tangent_fallback_mode: mesh.tangent_fallback_mode,
        uv1_buffer: mesh.uv1_buffer.clone(),
        uv2_buffer: mesh.uv2_buffer.clone(),
        uv3_buffer: mesh.uv3_buffer.clone(),
        wide_low_uv_buffer: mesh.wide_low_uv_buffer.clone(),
        wide_high_uv_buffer: mesh.wide_high_uv_buffer.clone(),
        derived_stream_state,
        extended_vertex_stream_source,
        has_skeleton: mesh.has_skeleton,
        skinning_bind_matrices: skinning,
        resident_bytes: mesh.resident_bytes,
    }
}

fn updated_derived_stream_state(
    mesh: &GpuMesh,
    demand: MeshDerivedStreamDemand,
    source: Option<&ExtendedVertexStreamSource>,
    write_vertex: bool,
    write_index: bool,
) -> super::super::MeshDerivedStreamState {
    let mut changed = MeshDerivedStreamMask::EMPTY;
    if write_vertex {
        changed |= MeshDerivedStreamMask::VERTEX_DERIVED;
    }
    if write_index {
        changed |= MeshDerivedStreamMask::INDEX_DERIVED;
    }
    // In-place rewrites require writable per-mesh buffers. Arena-only residency describes the
    // prior immutable copy and must never make an updated local stream look present.
    let available = mesh.dedicated_derived_stream_mask();
    let rebuildable = rebuildable_derived_stream_mask(source, available);
    mesh.derived_stream_state
        .after_in_place_update(demand, changed, rebuildable)
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
    let source = extended_vertex_stream_source_from_raw(raw, data, layout)?;
    mesh.should_keep_extended_vertex_stream_source(&source)
        .then_some(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_contract::EmbeddedTangentFallbackMode;
    use crate::shared::{
        IndexBufferFormat, MeshUploadHint, SubmeshTopology, VertexAttributeFormat,
        VertexAttributeType,
    };

    fn geometry_fixture() -> (Vec<u8>, MeshUploadData, MeshBufferLayout) {
        let vertex_size = 12;
        let index_buffer_length = 6;
        let mut raw = vec![0u8; vertex_size + index_buffer_length];
        raw[vertex_size..].copy_from_slice(&[0, 0, 0, 0, 0, 0]);
        let data = MeshUploadData {
            asset_id: 17,
            buffer: crate::shared::buffer::SharedMemoryBufferDescriptor {
                length: raw.len() as i32,
                ..Default::default()
            },
            vertex_count: 1,
            vertex_attributes: vec![VertexAttributeDescriptor {
                attribute: VertexAttributeType::Position,
                format: VertexAttributeFormat::Float32,
                dimensions: 3,
            }],
            submeshes: vec![SubmeshBufferDescriptor {
                topology: SubmeshTopology::Triangles,
                index_start: 0,
                index_count: 3,
                ..Default::default()
            }],
            index_buffer_format: IndexBufferFormat::UInt16,
            upload_hint: MeshUploadHint::default(),
            ..Default::default()
        };
        let layout = MeshBufferLayout {
            vertex_size,
            index_buffer_start: vertex_size,
            index_buffer_length,
            bone_counts_start: vertex_size + index_buffer_length,
            bone_counts_length: 0,
            bone_weights_start: vertex_size + index_buffer_length,
            bone_weights_length: 0,
            bind_poses_start: vertex_size + index_buffer_length,
            bind_poses_length: 0,
            blendshape_data_start: vertex_size + index_buffer_length,
            blendshape_data_length: 0,
            total_buffer_length: raw.len(),
        };
        (raw, data, layout)
    }

    #[test]
    fn geometry_hint_rewrites_vertex_and_index_buffers_in_place() {
        let flags = decode_in_place_write_flags(MeshUploadHintFlag(MeshUploadHintFlag::GEOMETRY));
        assert!(flags.write_vertex);
        assert!(flags.write_index);
        assert!(!flags.full);
    }

    #[test]
    fn exact_retained_geometry_match_detects_vertex_and_index_changes_independently() {
        let (raw, data, layout) = geometry_fixture();
        let source = extended_vertex_stream_source_from_raw(&raw, &data, &layout).expect("source");

        let identical = retained_geometry_match(&source, &raw, &data, &layout);
        assert!(identical.vertex_unchanged);
        assert!(identical.index_unchanged);

        let mut changed_vertex = raw.clone();
        changed_vertex[0] ^= 1;
        let vertex_match = retained_geometry_match(&source, &changed_vertex, &data, &layout);
        assert!(!vertex_match.vertex_unchanged);
        assert!(vertex_match.index_unchanged);

        let mut changed_index = raw.clone();
        changed_index[layout.index_buffer_start] ^= 1;
        let index_match = retained_geometry_match(&source, &changed_index, &data, &layout);
        assert!(index_match.vertex_unchanged);
        assert!(!index_match.index_unchanged);
    }

    #[test]
    fn retained_geometry_match_rejects_changed_interpretation_metadata() {
        let (raw, data, layout) = geometry_fixture();
        let source = extended_vertex_stream_source_from_raw(&raw, &data, &layout).expect("source");

        let mut changed_attributes = data.clone();
        changed_attributes.vertex_attributes[0].attribute = VertexAttributeType::Normal;
        let attribute_match = retained_geometry_match(&source, &raw, &changed_attributes, &layout);
        assert!(!attribute_match.vertex_unchanged);

        let mut changed_submeshes = data.clone();
        changed_submeshes.submeshes[0].topology = SubmeshTopology::Points;
        let submesh_match = retained_geometry_match(&source, &raw, &changed_submeshes, &layout);
        assert!(!submesh_match.index_unchanged);
    }

    #[test]
    fn unchanged_geometry_removes_only_its_derived_preparation_work() {
        let demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::VERTEX_DERIVED,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::GenerateMissing,
        };
        let all_unchanged = MeshInPlaceUploadPlan {
            vertex_unchanged: true,
            index_unchanged: true,
        }
        .changed_derived_stream_demand(demand);
        assert!(all_unchanged.mask.is_empty());

        let index_changed = MeshInPlaceUploadPlan {
            vertex_unchanged: true,
            index_unchanged: false,
        }
        .changed_derived_stream_demand(demand);
        assert_eq!(index_changed.mask, MeshDerivedStreamMask::TANGENT);

        let vertex_changed = MeshInPlaceUploadPlan {
            vertex_unchanged: false,
            index_unchanged: true,
        }
        .changed_derived_stream_demand(demand);
        assert_eq!(vertex_changed.mask, demand.mask);
    }

    #[test]
    fn missing_or_dirty_demand_disables_affected_geometry_elision() {
        let demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::DRAWABLE_PRIMARY | MeshDerivedStreamMask::TANGENT,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::GenerateMissing,
        };
        let plan = MeshInPlaceUploadPlan {
            vertex_unchanged: true,
            index_unchanged: true,
        }
        .restricted_for_resident_state(
            MeshGeometryStorage::Dedicated,
            MeshDerivedStreamMask::DRAWABLE_PRIMARY,
            MeshDerivedStreamMask::EMPTY,
            EmbeddedTangentFallbackMode::PreserveHostOrDefault,
            demand,
        );

        assert!(!plan.vertex_unchanged);
        assert!(!plan.index_unchanged);

        let dirty_color_demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::COLOR,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::PreserveHostOrDefault,
        };
        let dirty_color = MeshInPlaceUploadPlan {
            vertex_unchanged: true,
            index_unchanged: true,
        }
        .restricted_for_resident_state(
            MeshGeometryStorage::Dedicated,
            MeshDerivedStreamMask::COLOR,
            MeshDerivedStreamMask::COLOR,
            EmbeddedTangentFallbackMode::PreserveHostOrDefault,
            dirty_color_demand,
        );
        assert!(!dirty_color.vertex_unchanged);
        assert!(dirty_color.index_unchanged);
    }

    #[test]
    fn shared_static_geometry_never_elides_dedicated_upload_writes() {
        let demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::DRAWABLE_PRIMARY,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::PreserveHostOrDefault,
        };
        let plan = MeshInPlaceUploadPlan {
            vertex_unchanged: true,
            index_unchanged: true,
        }
        .restricted_for_resident_state(
            MeshGeometryStorage::SharedStatic,
            demand.mask,
            MeshDerivedStreamMask::EMPTY,
            EmbeddedTangentFallbackMode::PreserveHostOrDefault,
            demand,
        );

        assert_eq!(plan, MeshInPlaceUploadPlan::default());
        assert_eq!(plan.changed_derived_stream_demand(demand), demand);
    }

    #[test]
    fn full_upload_keeps_non_geometry_writes_when_geometry_is_unchanged() {
        let flags = decode_in_place_write_flags(MeshUploadHintFlag(0)).without_unchanged_geometry(
            MeshInPlaceUploadPlan {
                vertex_unchanged: true,
                index_unchanged: true,
            },
        );

        assert!(flags.full);
        assert!(!flags.write_vertex);
        assert!(!flags.write_index);
        assert!(flags.write_bone_weights);
        assert!(flags.write_bind_poses);
        assert!(flags.write_blend);
    }
}
