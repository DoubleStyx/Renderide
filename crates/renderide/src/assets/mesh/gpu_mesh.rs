//! GPU-resident mesh buffers; an optional CPU vertex copy is kept only until lazy extended streams build.

pub(in crate::assets::mesh) mod attribute_reader;
mod demand;
mod fingerprint;
mod hints;
mod tangent_generation;
mod update;
mod upload;
mod validation;

pub(crate) use demand::{MeshDerivedStreamDemand, MeshDerivedStreamMask, MeshDerivedStreamState};
pub(crate) use fingerprint::mesh_upload_input_fingerprint;
pub(crate) use upload::{
    MeshBufferUploadSink, MeshGpuUploadContext, PreparedDerivedStreams,
    allocate_generated_point_mesh, prepare_derived_stream_bytes,
    try_upload_generated_mesh_from_parts,
};
pub(crate) use validation::{compute_and_validate_mesh_layout, try_upload_mesh_from_raw};

use std::fmt;
use std::sync::Arc;

use crate::shared::{
    IndexBufferFormat, MeshUploadData, RenderBoundingBox, SubmeshBufferDescriptor, SubmeshTopology,
    VertexAttributeDescriptor, VertexAttributeType,
};
use glam::Mat4;

use super::layout::{
    BlendshapeFrameRange, BlendshapeFrameSpan, MeshBufferLayout, UV_VERTEX_ATTRIBUTE_TYPES,
    compute_vertex_stride,
};

use upload::{
    create_core_vertex_index_buffers, extract_derived_vertex_streams,
    resident_bytes_for_mesh_upload, upload_blendshape_buffer, upload_bone_and_skin_buffers,
    validate_mesh_upload_layout,
};

use hints::{validated_submesh_ranges, validated_submesh_topologies};

use crate::render_contract::{EmbeddedTangentFallbackMode, RasterPrimitiveTopology};

const EMPTY_MESH_PLACEHOLDER_BYTES: u64 = 4;

/// Storage used for immutable raster geometry.
///
/// Draw paths must use this value instead of inferring storage from upload metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MeshGeometryStorage {
    /// Frequently rewritten, generated, skinned, or otherwise dedicated geometry.
    Dedicated,
    /// Rigid immutable geometry addressed by asset id in the shared static geometry store.
    SharedStatic,
}

#[inline]
fn geometry_storage_for_upload(
    data: &MeshUploadData,
    uses_blendshape_geometry: bool,
    shared_store_attached: bool,
) -> MeshGeometryStorage {
    if shared_store_attached
        && !data.upload_hint.flags.dynamic()
        && data.bone_count <= 0
        && !uses_blendshape_geometry
    {
        MeshGeometryStorage::SharedStatic
    } else {
        MeshGeometryStorage::Dedicated
    }
}

#[inline]
const fn geometry_storage_after_index_validation(
    requested: MeshGeometryStorage,
    indices_valid: bool,
) -> MeshGeometryStorage {
    if matches!(requested, MeshGeometryStorage::SharedStatic) && !indices_valid {
        MeshGeometryStorage::Dedicated
    } else {
        requested
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MeshIndexRangeError {
    max_index: Option<u32>,
    needed_bytes: usize,
    available_bytes: usize,
}

impl MeshIndexRangeError {
    fn payload_too_short(needed_bytes: usize, available_bytes: usize) -> Self {
        Self {
            max_index: None,
            needed_bytes,
            available_bytes,
        }
    }

    fn vertex_out_of_range(max_index: u32) -> Self {
        Self {
            max_index: Some(max_index),
            needed_bytes: 0,
            available_bytes: 0,
        }
    }
}

/// Validates indices before a mesh enters the shared arena.
///
/// An out-of-range index can address a neighboring allocation when the full arena is bound.
fn validate_mesh_index_range(
    index_bytes: &[u8],
    index_format: IndexBufferFormat,
    index_count: u32,
    vertex_count: u32,
) -> Result<Option<u32>, MeshIndexRangeError> {
    let index_size = match index_format {
        IndexBufferFormat::UInt16 => 2usize,
        IndexBufferFormat::UInt32 => 4usize,
    };
    let needed_bytes = usize::try_from(index_count)
        .ok()
        .and_then(|count| count.checked_mul(index_size))
        .ok_or_else(|| MeshIndexRangeError::payload_too_short(usize::MAX, index_bytes.len()))?;
    let payload = index_bytes
        .get(..needed_bytes)
        .ok_or_else(|| MeshIndexRangeError::payload_too_short(needed_bytes, index_bytes.len()))?;
    let max_index = match index_format {
        IndexBufferFormat::UInt16 => payload
            .chunks_exact(2)
            .map(|bytes| u32::from(u16::from_le_bytes([bytes[0], bytes[1]])))
            .max(),
        IndexBufferFormat::UInt32 => payload
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .max(),
    };
    if max_index.is_some_and(|index| index >= vertex_count) {
        return Err(MeshIndexRangeError::vertex_out_of_range(
            max_index.unwrap_or(0),
        ));
    }
    Ok(max_index)
}

/// Rejects invalid shared-arena indices and logs once per asset.
fn shared_static_indices_are_valid(
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    index_count: u32,
) -> bool {
    let index_end = layout
        .index_buffer_start
        .checked_add(layout.index_buffer_length);
    let index_bytes = index_end.and_then(|end| raw.get(layout.index_buffer_start..end));
    let result = index_bytes.map_or_else(
        || {
            Err(MeshIndexRangeError::payload_too_short(
                layout.index_buffer_length,
                raw.len().saturating_sub(layout.index_buffer_start),
            ))
        },
        |bytes| {
            validate_mesh_index_range(
                bytes,
                data.index_buffer_format,
                index_count,
                data.vertex_count.max(0) as u32,
            )
        },
    );
    let Err(error) = result else {
        return true;
    };

    static INVALID_SHARED_STATIC_INDEX_LOG: std::sync::LazyLock<
        crate::diagnostics::log_once::KeyedLogOnce<i32>,
    > = std::sync::LazyLock::new(crate::diagnostics::log_once::KeyedLogOnce::new);
    if INVALID_SHARED_STATIC_INDEX_LOG.should_log(data.asset_id) {
        match error.max_index {
            Some(max_index) => logger::warn!(
                "mesh {} excluded from shared geometry arena: max index {} is outside vertex_count {} \
                 (declared indices={}); using dedicated direct rendering",
                data.asset_id,
                max_index,
                data.vertex_count.max(0),
                index_count
            ),
            None => logger::warn!(
                "mesh {} excluded from shared geometry arena: index payload is truncated \
                 (need {} bytes for {} indices, have {}); using dedicated direct rendering",
                data.asset_id,
                error.needed_bytes,
                index_count,
                error.available_bytes
            ),
        }
    }
    false
}

#[derive(Clone)]
pub(super) struct ExtendedVertexStreamSource {
    vertex_bytes: Arc<[u8]>,
    index_bytes: Arc<[u8]>,
    vertex_attributes: Arc<[VertexAttributeDescriptor]>,
    index_format: IndexBufferFormat,
    submeshes: Arc<[SubmeshBufferDescriptor]>,
    can_generate_missing_tangents: bool,
    has_primary_payload: bool,
    has_uv0_payload: bool,
    has_color_payload: bool,
    has_compact_extended_payload: bool,
    has_low_uv_payload: bool,
    has_wide_low_uv_payload: bool,
    has_wide_high_uv_payload: bool,
}

/// Reusable in-place upload decision made after validating the resident mesh and exact retained
/// geometry bytes.
///
/// A `true` field is backed by byte-for-byte comparison with the CPU source already retained for
/// lazy derived-stream rebuilding. No additional mesh payload is retained for this optimization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MeshInPlaceUploadPlan {
    vertex_unchanged: bool,
    index_unchanged: bool,
}

impl fmt::Debug for ExtendedVertexStreamSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtendedVertexStreamSource")
            .field("vertex_bytes_len", &self.vertex_bytes.len())
            .field("index_bytes_len", &self.index_bytes.len())
            .field("vertex_attributes_len", &self.vertex_attributes.len())
            .field("index_format", &self.index_format)
            .field("submeshes_len", &self.submeshes.len())
            .field(
                "can_generate_missing_tangents",
                &self.can_generate_missing_tangents,
            )
            .field("has_primary_payload", &self.has_primary_payload)
            .field("has_uv0_payload", &self.has_uv0_payload)
            .field("has_color_payload", &self.has_color_payload)
            .field(
                "has_compact_extended_payload",
                &self.has_compact_extended_payload,
            )
            .field("has_low_uv_payload", &self.has_low_uv_payload)
            .field("has_wide_low_uv_payload", &self.has_wide_low_uv_payload)
            .field("has_wide_high_uv_payload", &self.has_wide_high_uv_payload)
            .finish()
    }
}

/// Resident mesh on GPU: no CPU geometry retained.
///
/// **Vertex groups** in Renderite are expressed through per-vertex bone influence streams
/// (`bone_counts` + bone weight tail) when the host provides skeleton data.
#[derive(Debug, Clone)]
pub struct GpuMesh {
    /// Host mesh asset id (`MeshUploadData.asset_id`).
    pub asset_id: i32,
    /// Whether the host marks this mesh as frequently rewritten.
    pub(crate) dynamic_geometry: bool,
    /// Authoritative raster-geometry storage route.
    pub(crate) geometry_storage: MeshGeometryStorage,
    /// Derived streams resident in the shared static geometry store.
    pub(crate) shared_static_resident_streams: MeshDerivedStreamMask,
    /// Host-interleaved vertices retained for dynamic or generated meshes.
    pub vertex_buffer: Option<Arc<wgpu::Buffer>>,
    /// Dedicated GPU index buffer, released after shared-static residency.
    pub index_buffer: Option<Arc<wgpu::Buffer>>,
    /// Element size for `index_buffer` (`Uint16` vs `Uint32`).
    pub index_format: wgpu::IndexFormat,
    /// Total index elements across all submeshes.
    pub index_count: u32,
    /// Per-submesh `(first_index, index_count)` in elements of `index_format`.
    ///
    /// Aligned row-for-row with [`Self::submesh_topologies`].
    pub submeshes: Vec<(u32, u32)>,
    /// Per-submesh primitive topology, aligned row-for-row with [`Self::submeshes`].
    ///
    /// Sourced from [`crate::shared::SubmeshBufferDescriptor::topology`] at upload time.
    /// The synthesized fallback range used when the host sends no submeshes (or every submesh
    /// fails validation) defaults to [`RasterPrimitiveTopology::TriangleList`].
    pub submesh_topologies: Vec<RasterPrimitiveTopology>,
    /// Vertex count from the host upload (used for deform and draw ranges).
    pub vertex_count: u32,
    /// Byte stride of one interleaved vertex in `vertex_buffer`.
    pub vertex_stride: u32,
    /// Axis-aligned bounds in mesh space (from host).
    pub bounds: RenderBoundingBox,
    /// Optional 1 byte per vertex for skinned meshes.
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex joint indices as `vec4<u32>` (16 bytes / vertex) for skinning compute.
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex bone weights as `vec4<f32>` for skinning compute.
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex prefix offsets into [`Self::bone_influences_buffer`] for unlimited skinning.
    pub bone_influence_offsets_buffer: Option<Arc<wgpu::Buffer>>,
    /// Raw `(u32 bone_index, f32 weight)` influence entries for unlimited skinning.
    pub bone_influences_buffer: Option<Arc<wgpu::Buffer>>,
    /// Column-major `float4x4` bind poses (64 bytes per bone).
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    /// Sparse packed blendshape delta words for all shapes.
    pub blendshape_sparse_buffer: Option<Arc<wgpu::Buffer>>,
    /// CPU copy of each sparse frame range for scatter dispatch.
    pub blendshape_frame_ranges: Vec<BlendshapeFrameRange>,
    /// Per-shape spans into [`Self::blendshape_frame_ranges`].
    pub blendshape_shape_frame_spans: Vec<BlendshapeFrameSpan>,
    /// Number of logical blendshape slots (`max(blendshape_index)+1`).
    pub num_blendshapes: u32,
    /// Whether uploaded blendshape rows include nonzero position deltas.
    pub blendshape_has_position_deltas: bool,
    /// Whether uploaded blendshape rows include nonzero normal deltas.
    pub blendshape_has_normal_deltas: bool,
    /// Whether uploaded blendshape rows include nonzero tangent deltas.
    pub blendshape_has_tangent_deltas: bool,
    /// Decomposed position stream (`vec4<f32>` per vertex) for compute + debug raster.
    pub positions_buffer: Option<Arc<wgpu::Buffer>>,
    /// Bind-pose normal stream (`vec4<f32>` per vertex; xyz used). Skinning writes deformed normals
    /// to the GPU skin cache arena; see [`crate::mesh_deform::GpuSkinCache`].
    pub normals_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec2<f32>` UV0 stream (`8` bytes/vertex) for embedded raster materials; zeros when uv0 is absent.
    pub uv0_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec4<f32>` color stream for UI/text embedded materials; defaults to opaque white when absent.
    pub color_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec4<f32>` tangent stream for shaders using extended vertex inputs.
    pub tangent_buffer: Option<Arc<wgpu::Buffer>>,
    /// Raw `vec4<f32>` tangent payload for UI shaders that use the tangent semantic as data.
    pub raw_tangent_buffer: Option<Arc<wgpu::Buffer>>,
    /// Tangent fallback policy used for the current tangent stream.
    pub tangent_fallback_mode: EmbeddedTangentFallbackMode,
    /// `vec2<f32>` UV1 stream for shaders using extended vertex inputs.
    pub uv1_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec2<f32>` UV2 stream for shaders using extended vertex inputs.
    pub uv2_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec2<f32>` UV3 stream for shaders using extended vertex inputs.
    pub uv3_buffer: Option<Arc<wgpu::Buffer>>,
    /// Packed UV0-UV3 stream for shaders using 3D/4D low UV inputs.
    pub wide_low_uv_buffer: Option<Arc<wgpu::Buffer>>,
    /// Packed UV4-UV7 stream for shaders using high UV inputs.
    pub wide_high_uv_buffer: Option<Arc<wgpu::Buffer>>,
    /// Demand and dirty-state for lazily maintained derived streams.
    pub(crate) derived_stream_state: MeshDerivedStreamState,
    /// CPU vertex source kept only until lazy extended streams are created.
    extended_vertex_stream_source: Option<ExtendedVertexStreamSource>,
    /// True when the host uploaded a real skeleton (`bone_count > 0`).
    pub has_skeleton: bool,
    /// Unity [`Mesh.bindposes`](https://docs.unity3d.com/ScriptReference/Mesh-bindposes.html):
    /// inverse bind matrices (mesh space -> bone bind space). Per-frame palette is
    /// `world_bone * skinning_bind_matrices[i]`.
    pub skinning_bind_matrices: Vec<Mat4>,
    /// Approximate VRAM (bytes), used by [`crate::gpu_pools::VramAccounting`].
    pub resident_bytes: u64,
}

fn has_compact_extended_payload(attrs: &[VertexAttributeDescriptor]) -> bool {
    attrs.iter().any(|a| {
        matches!(
            a.attribute,
            VertexAttributeType::Tangent
                | VertexAttributeType::UV1
                | VertexAttributeType::UV2
                | VertexAttributeType::UV3
        )
    })
}

fn is_uv_attribute_in_range(
    attr: VertexAttributeDescriptor,
    range: std::ops::Range<usize>,
    min_dimensions: i32,
) -> bool {
    UV_VERTEX_ATTRIBUTE_TYPES[range]
        .iter()
        .any(|uv| (attr.attribute as i16) == (*uv as i16))
        && attr.dimensions >= min_dimensions
}

fn has_low_uv_payload(attrs: &[VertexAttributeDescriptor]) -> bool {
    attrs
        .iter()
        .any(|attr| is_uv_attribute_in_range(*attr, 0..4, 2))
}

fn has_wide_low_uv_payload(attrs: &[VertexAttributeDescriptor]) -> bool {
    attrs
        .iter()
        .any(|attr| is_uv_attribute_in_range(*attr, 0..4, 3))
}

fn has_wide_high_uv_payload(attrs: &[VertexAttributeDescriptor]) -> bool {
    attrs
        .iter()
        .any(|attr| is_uv_attribute_in_range(*attr, 4..8, 2))
}

fn has_supported_vertex_attribute(
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    min_dimensions: i32,
) -> bool {
    attrs
        .iter()
        .any(|attr| (attr.attribute as i16) == (target as i16) && attr.dimensions >= min_dimensions)
}

fn can_generate_missing_tangents(data: &MeshUploadData, layout: &MeshBufferLayout) -> bool {
    data.vertex_count > 0
        && layout.index_buffer_length > 0
        && data.submeshes.iter().any(|submesh| {
            submesh.topology == SubmeshTopology::Triangles && submesh.index_count >= 3
        })
        && has_supported_vertex_attribute(&data.vertex_attributes, VertexAttributeType::Position, 3)
        && has_supported_vertex_attribute(&data.vertex_attributes, VertexAttributeType::Normal, 3)
        && has_supported_vertex_attribute(&data.vertex_attributes, VertexAttributeType::UV0, 2)
}

pub(super) fn extended_vertex_stream_source_from_raw(
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
) -> Option<ExtendedVertexStreamSource> {
    let can_generate_missing_tangents = can_generate_missing_tangents(data, layout);
    let has_primary_payload =
        has_supported_vertex_attribute(&data.vertex_attributes, VertexAttributeType::Position, 3)
            && has_supported_vertex_attribute(
                &data.vertex_attributes,
                VertexAttributeType::Normal,
                3,
            );
    let has_uv0_payload =
        has_supported_vertex_attribute(&data.vertex_attributes, VertexAttributeType::UV0, 2);
    let has_color_payload =
        has_supported_vertex_attribute(&data.vertex_attributes, VertexAttributeType::Color, 4);
    let has_compact_extended_payload =
        has_compact_extended_payload(&data.vertex_attributes) || can_generate_missing_tangents;
    let has_low_uv_payload = has_low_uv_payload(&data.vertex_attributes);
    let has_wide_low_uv_payload = has_wide_low_uv_payload(&data.vertex_attributes);
    let has_wide_high_uv_payload = has_wide_high_uv_payload(&data.vertex_attributes);
    if data.vertex_count <= 0
        || (data.vertex_attributes.is_empty() && !can_generate_missing_tangents)
    {
        return None;
    }
    let vertex_bytes = raw.get(..layout.vertex_size)?.to_vec();
    let index_end = layout
        .index_buffer_start
        .checked_add(layout.index_buffer_length)?;
    let index_bytes = raw.get(layout.index_buffer_start..index_end)?.to_vec();
    Some(ExtendedVertexStreamSource {
        vertex_bytes: Arc::from(vertex_bytes),
        index_bytes: Arc::from(index_bytes),
        vertex_attributes: Arc::from(data.vertex_attributes.clone()),
        index_format: data.index_buffer_format,
        submeshes: Arc::from(data.submeshes.clone()),
        can_generate_missing_tangents,
        has_primary_payload,
        has_uv0_payload,
        has_color_payload,
        has_compact_extended_payload,
        has_low_uv_payload,
        has_wide_low_uv_payload,
        has_wide_high_uv_payload,
    })
}

pub(super) fn extended_vertex_stream_bytes(mesh: &GpuMesh) -> u64 {
    [
        mesh.tangent_buffer.as_ref(),
        mesh.raw_tangent_buffer.as_ref(),
        mesh.uv1_buffer.as_ref(),
        mesh.uv2_buffer.as_ref(),
        mesh.uv3_buffer.as_ref(),
        mesh.wide_low_uv_buffer.as_ref(),
        mesh.wide_high_uv_buffer.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|b| b.size())
    .sum()
}

/// Takes `buffer` when `present`, returning the bytes it held (for VRAM accounting) or 0.
fn take_stream_bytes(present: bool, buffer: &mut Option<Arc<wgpu::Buffer>>) -> u64 {
    if present {
        buffer.take().map_or(0, |dropped| dropped.size())
    } else {
        0
    }
}

fn rebuildable_derived_stream_mask(
    source: Option<&ExtendedVertexStreamSource>,
    available_mask: MeshDerivedStreamMask,
) -> MeshDerivedStreamMask {
    let mut mask = available_mask;
    if let Some(source) = source {
        if source.has_primary_payload {
            mask |= MeshDerivedStreamMask::POSITION | MeshDerivedStreamMask::NORMAL;
        }
        if source.has_uv0_payload {
            mask |= MeshDerivedStreamMask::UV0;
        }
        if source.has_color_payload {
            mask |= MeshDerivedStreamMask::COLOR;
        }
        if source.has_compact_extended_payload {
            mask |= MeshDerivedStreamMask::TANGENT
                | MeshDerivedStreamMask::RAW_TANGENT
                | MeshDerivedStreamMask::UV1
                | MeshDerivedStreamMask::UV2
                | MeshDerivedStreamMask::UV3;
        }
        if source.has_low_uv_payload || source.has_wide_low_uv_payload {
            mask |= MeshDerivedStreamMask::WIDE_UV_LOW;
        }
        if source.has_wide_high_uv_payload {
            mask |= MeshDerivedStreamMask::WIDE_UV_HIGH;
        }
        if source.can_generate_missing_tangents {
            mask |= MeshDerivedStreamMask::TANGENT;
        }
    }
    mask
}

impl GpuMesh {
    /// Whether raster passes must resolve this mesh through the shared static geometry store.
    #[inline]
    pub(crate) const fn uses_shared_static_geometry(&self) -> bool {
        matches!(self.geometry_storage, MeshGeometryStorage::SharedStatic)
    }

    /// Whether this mesh's position and index data are authoritative in the shared static store.
    #[inline]
    pub(crate) const fn has_shared_static_core_residency(&self) -> bool {
        self.uses_shared_static_geometry()
            && self
                .shared_static_resident_streams
                .contains(MeshDerivedStreamMask::POSITION)
    }

    /// Whether raster/cull consumers can resolve both position and index data through either route.
    #[inline]
    pub(crate) fn has_raster_core_residency(&self) -> bool {
        (self.positions_buffer.is_some() && self.index_buffer.is_some())
            || self.has_shared_static_core_residency()
    }

    /// Dedicated bytes that duplicate streams already authoritative in the shared static store.
    #[cfg(feature = "tracy")]
    pub(crate) fn shared_static_duplicate_source_bytes(&self) -> u64 {
        if !self.has_shared_static_core_residency() {
            return 0;
        }
        let resident = self.shared_static_resident_streams;
        let mut bytes = self.index_buffer.as_ref().map_or(0, |buffer| buffer.size());
        for (stream, buffer) in [
            (
                MeshDerivedStreamMask::POSITION,
                self.positions_buffer.as_ref(),
            ),
            (MeshDerivedStreamMask::NORMAL, self.normals_buffer.as_ref()),
            (MeshDerivedStreamMask::UV0, self.uv0_buffer.as_ref()),
            (MeshDerivedStreamMask::COLOR, self.color_buffer.as_ref()),
            (MeshDerivedStreamMask::TANGENT, self.tangent_buffer.as_ref()),
            (
                MeshDerivedStreamMask::RAW_TANGENT,
                self.raw_tangent_buffer.as_ref(),
            ),
            (MeshDerivedStreamMask::UV1, self.uv1_buffer.as_ref()),
            (MeshDerivedStreamMask::UV2, self.uv2_buffer.as_ref()),
            (MeshDerivedStreamMask::UV3, self.uv3_buffer.as_ref()),
            (
                MeshDerivedStreamMask::WIDE_UV_LOW,
                self.wide_low_uv_buffer.as_ref(),
            ),
            (
                MeshDerivedStreamMask::WIDE_UV_HIGH,
                self.wide_high_uv_buffer.as_ref(),
            ),
        ] {
            if resident.contains(stream) {
                bytes = bytes.saturating_add(buffer.map_or(0, |buffer| buffer.size()));
            }
        }
        bytes
    }

    /// Returns streams backed by this mesh's dedicated GPU buffers.
    pub(crate) fn dedicated_derived_stream_mask(&self) -> MeshDerivedStreamMask {
        let mut mask = MeshDerivedStreamMask::EMPTY;
        if self.positions_buffer.is_some() {
            mask |= MeshDerivedStreamMask::POSITION;
        }
        if self.normals_buffer.is_some() {
            mask |= MeshDerivedStreamMask::NORMAL;
        }
        if self.uv0_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV0;
        }
        if self.color_buffer.is_some() {
            mask |= MeshDerivedStreamMask::COLOR;
        }
        if self.tangent_buffer.is_some() {
            mask |= MeshDerivedStreamMask::TANGENT;
        }
        if self.raw_tangent_buffer.is_some() {
            mask |= MeshDerivedStreamMask::RAW_TANGENT;
        }
        if self.uv1_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV1;
        }
        if self.uv2_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV2;
        }
        if self.uv3_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV3;
        }
        if self.wide_low_uv_buffer.is_some() {
            mask |= MeshDerivedStreamMask::WIDE_UV_LOW;
        }
        if self.wide_high_uv_buffer.is_some() {
            mask |= MeshDerivedStreamMask::WIDE_UV_HIGH;
        }
        mask
    }

    /// Returns streams resident either in dedicated GPU buffers or the shared static store.
    pub(crate) fn available_derived_stream_mask(&self) -> MeshDerivedStreamMask {
        self.dedicated_derived_stream_mask() | self.shared_static_resident_streams
    }

    /// Returns whether all requested streams are current and resident in either storage route.
    #[inline]
    pub(super) fn derived_streams_ready(&self, mask: MeshDerivedStreamMask) -> bool {
        self.derived_stream_state
            .streams_ready(self.available_derived_stream_mask().contains(mask), mask)
    }

    /// Releases dedicated copies of shared-static streams and returns their byte count.
    pub(crate) fn release_shared_static_geometry_sources(
        &mut self,
        resident_streams: MeshDerivedStreamMask,
    ) -> u64 {
        if !self.uses_shared_static_geometry() {
            return 0;
        }
        self.shared_static_resident_streams |= resident_streams;
        let mut released = 0u64;
        released += take_stream_bytes(
            self.has_shared_static_core_residency(),
            &mut self.index_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::POSITION),
            &mut self.positions_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::NORMAL),
            &mut self.normals_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::UV0),
            &mut self.uv0_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::COLOR),
            &mut self.color_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::TANGENT),
            &mut self.tangent_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::RAW_TANGENT),
            &mut self.raw_tangent_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::UV1),
            &mut self.uv1_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::UV2),
            &mut self.uv2_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::UV3),
            &mut self.uv3_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::WIDE_UV_LOW),
            &mut self.wide_low_uv_buffer,
        );
        released += take_stream_bytes(
            resident_streams.contains(MeshDerivedStreamMask::WIDE_UV_HIGH),
            &mut self.wide_high_uv_buffer,
        );
        if released != 0 {
            self.resident_bytes = self.resident_bytes.saturating_sub(released);
        }
        released
    }

    /// Builds a device-free mesh for draw-preparation mutation tests.
    #[cfg(test)]
    pub(crate) fn test_draw_prep_mesh(asset_id: i32) -> Self {
        Self {
            asset_id,
            dynamic_geometry: false,
            geometry_storage: MeshGeometryStorage::Dedicated,
            shared_static_resident_streams: MeshDerivedStreamMask::EMPTY,
            vertex_buffer: None,
            index_buffer: None,
            index_format: wgpu::IndexFormat::Uint32,
            index_count: 3,
            submeshes: vec![(0, 3)],
            submesh_topologies: vec![RasterPrimitiveTopology::TriangleList],
            vertex_count: 3,
            vertex_stride: 12,
            bounds: RenderBoundingBox::default(),
            bone_counts_buffer: None,
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bone_influence_offsets_buffer: None,
            bone_influences_buffer: None,
            bind_poses_buffer: None,
            blendshape_sparse_buffer: None,
            blendshape_frame_ranges: Vec::new(),
            blendshape_shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            blendshape_has_position_deltas: false,
            blendshape_has_normal_deltas: false,
            blendshape_has_tangent_deltas: false,
            positions_buffer: None,
            normals_buffer: None,
            uv0_buffer: None,
            color_buffer: None,
            tangent_buffer: None,
            raw_tangent_buffer: None,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
            uv1_buffer: None,
            uv2_buffer: None,
            uv3_buffer: None,
            wide_low_uv_buffer: None,
            wide_high_uv_buffer: None,
            derived_stream_state: MeshDerivedStreamState::default(),
            extended_vertex_stream_source: None,
            has_skeleton: false,
            skinning_bind_matrices: Vec::new(),
            resident_bytes: 0,
        }
    }

    /// Creates a resident mesh entry for a host upload with no geometry payload.
    pub fn empty(device: &wgpu::Device, data: &MeshUploadData) -> Self {
        profiling::scope!("asset::mesh_empty_gpu_upload");
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("mesh {} empty indices", data.asset_id)),
            size: EMPTY_MESH_PLACEHOLDER_BYTES,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_empty_indices");
        let resident_bytes = index_buffer.size();

        Self {
            asset_id: data.asset_id,
            dynamic_geometry: data.upload_hint.flags.dynamic(),
            geometry_storage: MeshGeometryStorage::Dedicated,
            shared_static_resident_streams: MeshDerivedStreamMask::EMPTY,
            vertex_buffer: None,
            index_buffer: Some(Arc::new(index_buffer)),
            index_format: match data.index_buffer_format {
                IndexBufferFormat::UInt16 => wgpu::IndexFormat::Uint16,
                IndexBufferFormat::UInt32 => wgpu::IndexFormat::Uint32,
            },
            index_count: 0,
            submeshes: Vec::new(),
            submesh_topologies: Vec::new(),
            vertex_count: 0,
            vertex_stride: compute_vertex_stride(&data.vertex_attributes).max(1) as u32,
            bounds: data.bounds,
            bone_counts_buffer: None,
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bone_influence_offsets_buffer: None,
            bone_influences_buffer: None,
            bind_poses_buffer: None,
            blendshape_sparse_buffer: None,
            blendshape_frame_ranges: Vec::new(),
            blendshape_shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            blendshape_has_position_deltas: false,
            blendshape_has_normal_deltas: false,
            blendshape_has_tangent_deltas: false,
            positions_buffer: None,
            normals_buffer: None,
            uv0_buffer: None,
            color_buffer: None,
            tangent_buffer: None,
            raw_tangent_buffer: None,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
            uv1_buffer: None,
            uv2_buffer: None,
            uv3_buffer: None,
            wide_low_uv_buffer: None,
            wide_high_uv_buffer: None,
            derived_stream_state: MeshDerivedStreamState::default(),
            extended_vertex_stream_source: None,
            has_skeleton: false,
            skinning_bind_matrices: Vec::new(),
            resident_bytes,
        }
    }

    /// Uploads mesh data from a raw byte slice covering at least `layout.total_buffer_length`.
    ///
    /// `raw` must be the mapping for `data.buffer` only for the duration of this call.
    pub fn upload(
        ctx: MeshGpuUploadContext<'_>,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
    ) -> Option<Self> {
        profiling::scope!("asset::mesh_full_gpu_upload");
        let max_buf = ctx.gpu_limits.max_buffer_size();
        {
            profiling::scope!("asset::mesh_validate_upload_layout");
            if !validate_mesh_upload_layout(raw, data, layout, ctx.gpu_limits) {
                return None;
            }
        }

        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();

        let core = create_core_vertex_index_buffers(ctx, raw, data, layout)?;
        let requested_geometry_storage =
            geometry_storage_for_upload(data, use_blendshapes, ctx.static_geometry_store.is_some());
        let indices_valid = requested_geometry_storage != MeshGeometryStorage::SharedStatic
            || shared_static_indices_are_valid(raw, data, layout, core.index_count_u32);
        let geometry_storage =
            geometry_storage_after_index_validation(requested_geometry_storage, indices_valid);
        let vc_usize = data.vertex_count.max(0) as usize;

        let derived = extract_derived_vertex_streams(ctx, raw, data, layout, &core)?;
        let derived_available_mask = derived.available_mask();
        let extended_vertex_stream_source = {
            profiling::scope!("asset::mesh_capture_extended_stream_source");
            extended_vertex_stream_source_from_raw(raw, data, layout)
        };
        let derived_rebuildable_mask = rebuildable_derived_stream_mask(
            extended_vertex_stream_source.as_ref(),
            derived_available_mask,
        );
        let derived_stream_state = MeshDerivedStreamState::after_full_upload(
            ctx.derived_stream_demand,
            derived_available_mask,
            derived_rebuildable_mask,
        );

        let bone_skin = upload_bone_and_skin_buffers(ctx, raw, data, layout, vc_usize)?;

        let blend_up = upload_blendshape_buffer(ctx, raw, data, layout, use_blendshapes, max_buf)?;
        let num_blendshapes = blend_up.num_blendshapes;

        let (submeshes, submesh_topologies) = {
            profiling::scope!("asset::mesh_validate_submesh_ranges");
            (
                validated_submesh_ranges(&data.submeshes, core.index_count_u32),
                validated_submesh_topologies(&data.submeshes, core.index_count_u32),
            )
        };

        let resident_bytes = {
            profiling::scope!("asset::mesh_resident_byte_count");
            resident_bytes_for_mesh_upload(
                core.vb.as_ref(),
                &core.ib,
                &derived,
                &bone_skin,
                blend_up.sparse_buffer.as_ref(),
            )
        };

        Some(Self {
            asset_id: data.asset_id,
            dynamic_geometry: data.upload_hint.flags.dynamic(),
            geometry_storage,
            shared_static_resident_streams: MeshDerivedStreamMask::EMPTY,
            vertex_buffer: core.vb.map(Arc::new),
            index_buffer: Some(Arc::new(core.ib)),
            index_format: core.index_format,
            index_count: core.index_count_u32,
            submeshes,
            submesh_topologies,
            vertex_count: data.vertex_count.max(0) as u32,
            vertex_stride: core.vertex_stride,
            bounds: data.bounds,
            bone_counts_buffer: bone_skin.bone_counts_buffer,
            bone_indices_buffer: bone_skin.bone_indices_buffer,
            bone_weights_vec4_buffer: bone_skin.bone_weights_vec4_buffer,
            bone_influence_offsets_buffer: bone_skin.bone_influence_offsets_buffer,
            bone_influences_buffer: bone_skin.bone_influences_buffer,
            bind_poses_buffer: bone_skin.bind_poses_buffer,
            blendshape_sparse_buffer: blend_up.sparse_buffer,
            blendshape_frame_ranges: blend_up.frame_ranges,
            blendshape_shape_frame_spans: blend_up.shape_frame_spans,
            num_blendshapes,
            blendshape_has_position_deltas: blend_up.has_position_deltas,
            blendshape_has_normal_deltas: blend_up.has_normal_deltas,
            blendshape_has_tangent_deltas: blend_up.has_tangent_deltas,
            positions_buffer: derived.positions_buffer,
            normals_buffer: derived.normals_buffer,
            uv0_buffer: derived.uv0_buffer,
            color_buffer: derived.color_buffer,
            tangent_buffer: derived.tangent_buffer,
            raw_tangent_buffer: derived.raw_tangent_buffer,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
            uv1_buffer: derived.uv1_buffer,
            uv2_buffer: derived.uv2_buffer,
            uv3_buffer: derived.uv3_buffer,
            wide_low_uv_buffer: derived.wide_low_uv_buffer,
            wide_high_uv_buffer: derived.wide_high_uv_buffer,
            derived_stream_state,
            extended_vertex_stream_source,
            has_skeleton: data.bone_count > 0,
            skinning_bind_matrices: bone_skin.skinning_bind_matrices,
            resident_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::layout::{compute_mesh_buffer_layout, index_bytes_per_element};
    use super::*;
    use crate::shared::{MeshUploadHint, MeshUploadHintFlag, VertexAttributeFormat};

    fn uv_attr(attribute: VertexAttributeType, dimensions: i32) -> VertexAttributeDescriptor {
        VertexAttributeDescriptor {
            attribute,
            format: VertexAttributeFormat::Float32,
            dimensions,
        }
    }

    fn source_for_attrs(
        attrs: Vec<VertexAttributeDescriptor>,
    ) -> Option<ExtendedVertexStreamSource> {
        let layout = compute_mesh_buffer_layout(
            compute_vertex_stride(&attrs),
            1,
            0,
            index_bytes_per_element(IndexBufferFormat::UInt16),
            0,
            0,
            None,
        )
        .expect("layout");
        let raw = vec![0u8; layout.total_buffer_length];
        let data = MeshUploadData {
            vertex_count: 1,
            index_buffer_format: IndexBufferFormat::UInt16,
            vertex_attributes: attrs,
            ..Default::default()
        };

        extended_vertex_stream_source_from_raw(&raw, &data, &layout)
    }

    #[test]
    fn extended_source_captures_uv7_for_lazy_wide_uv_upload() {
        let source =
            source_for_attrs(vec![uv_attr(VertexAttributeType::UV7, 2)]).expect("wide uv source");

        assert!(!source.has_compact_extended_payload);
        assert!(!source.has_wide_low_uv_payload);
        assert!(source.has_wide_high_uv_payload);
    }

    #[test]
    fn extended_source_captures_vec4_uv0_for_lazy_wide_uv_upload() {
        let source =
            source_for_attrs(vec![uv_attr(VertexAttributeType::UV0, 4)]).expect("wide uv source");

        assert!(!source.has_compact_extended_payload);
        assert!(source.has_low_uv_payload);
        assert!(source.has_wide_low_uv_payload);
        assert!(!source.has_wide_high_uv_payload);
    }

    #[test]
    fn extended_source_captures_plain_vec2_uv0_for_lazy_uv0_upload() {
        let source =
            source_for_attrs(vec![uv_attr(VertexAttributeType::UV0, 2)]).expect("uv0 source");

        assert!(source.has_uv0_payload);
        assert!(source.has_low_uv_payload);
        assert!(!source.has_wide_low_uv_payload);
        assert!(!source.has_wide_high_uv_payload);
    }

    #[test]
    fn shared_arena_index_validation_accepts_in_range_u16_and_u32_payloads() {
        let u16_bytes = [0u16, 2, 1]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let u32_bytes = [3u32, 0, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();

        assert_eq!(
            validate_mesh_index_range(&u16_bytes, IndexBufferFormat::UInt16, 3, 3),
            Ok(Some(2))
        );
        assert_eq!(
            validate_mesh_index_range(&u32_bytes, IndexBufferFormat::UInt32, 3, 4),
            Ok(Some(3))
        );
    }

    #[test]
    fn shared_arena_index_validation_rejects_neighbor_fetches_for_both_widths() {
        let u16_bytes = [0u16, 7, 1]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let u32_bytes = [0u32, 1, 70_000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();

        assert_eq!(
            validate_mesh_index_range(&u16_bytes, IndexBufferFormat::UInt16, 3, 3),
            Err(MeshIndexRangeError::vertex_out_of_range(7))
        );
        assert_eq!(
            validate_mesh_index_range(&u32_bytes, IndexBufferFormat::UInt32, 3, 3),
            Err(MeshIndexRangeError::vertex_out_of_range(70_000))
        );
    }

    #[test]
    fn shared_arena_index_validation_reads_exact_declared_count_and_rejects_truncation() {
        let bytes_with_ignored_tail = [0u16, 1, u16::MAX]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();

        assert_eq!(
            validate_mesh_index_range(&bytes_with_ignored_tail, IndexBufferFormat::UInt16, 2, 2,),
            Ok(Some(1))
        );
        assert_eq!(
            validate_mesh_index_range(
                &bytes_with_ignored_tail[..3],
                IndexBufferFormat::UInt16,
                2,
                2
            ),
            Err(MeshIndexRangeError::payload_too_short(4, 3))
        );
    }

    #[test]
    fn invalid_indices_force_only_shared_static_requests_to_dedicated_storage() {
        assert_eq!(
            geometry_storage_after_index_validation(MeshGeometryStorage::SharedStatic, false),
            MeshGeometryStorage::Dedicated
        );
        assert_eq!(
            geometry_storage_after_index_validation(MeshGeometryStorage::SharedStatic, true),
            MeshGeometryStorage::SharedStatic
        );
        assert_eq!(
            geometry_storage_after_index_validation(MeshGeometryStorage::Dedicated, false),
            MeshGeometryStorage::Dedicated
        );
    }

    #[test]
    fn rigid_immutable_upload_selects_shared_static_storage() {
        let data = MeshUploadData::default();
        assert_eq!(
            geometry_storage_for_upload(&data, false, true),
            MeshGeometryStorage::SharedStatic
        );
    }

    #[test]
    fn dynamic_skin_and_blendshape_uploads_stay_dedicated() {
        let dynamic = MeshUploadData {
            upload_hint: MeshUploadHint {
                flags: MeshUploadHintFlag(MeshUploadHintFlag::DYNAMIC),
            },
            ..Default::default()
        };
        assert_eq!(
            geometry_storage_for_upload(&dynamic, false, true),
            MeshGeometryStorage::Dedicated
        );

        let skinned = MeshUploadData {
            bone_count: 1,
            ..Default::default()
        };
        assert_eq!(
            geometry_storage_for_upload(&skinned, false, true),
            MeshGeometryStorage::Dedicated
        );
        assert_eq!(
            geometry_storage_for_upload(&MeshUploadData::default(), true, true),
            MeshGeometryStorage::Dedicated
        );
        assert_eq!(
            geometry_storage_for_upload(&MeshUploadData::default(), false, false),
            MeshGeometryStorage::Dedicated
        );
    }
}
