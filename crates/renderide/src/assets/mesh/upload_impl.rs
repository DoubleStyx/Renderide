//! Helpers for [`super::GpuMesh::upload`](GpuMesh::upload); keeps the `impl` readable.

use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::gpu::GpuLimits;
use crate::mesh_deform::{
    BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES, blendshape_sparse_buffers_fit_device,
};
use crate::shared::{
    IndexBufferFormat, MeshUploadData, SubmeshBufferDescriptor, VertexAttributeDescriptor,
    VertexAttributeType,
};

use super::gpu_mesh_hints::wgpu_index_format;
use super::layout::{
    BlendshapeFrameRange, BlendshapeFrameSpan, MeshBufferLayout, color_float4_stream_bytes,
    compute_index_count, compute_vertex_stride, extract_bind_poses, extract_blendshape_offsets,
    extract_float3_position_normal_as_vec4_streams, split_bone_weights_tail_for_gpu,
    synthetic_bone_data_for_blendshape_only, uv0_float2_stream_bytes, vertex_float2_stream_bytes,
};
use super::tangent_generation::tangent_stream_bytes;

/// Tangent plus UV1-UV3 optional vertex buffers from extended stream upload.
type ExtendedVertexStreams = (
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
);

/// CPU-side mesh source required to build lazy tangent and UV1-UV3 vertex streams.
pub(super) struct ExtendedVertexUploadSource<'a> {
    /// Interleaved vertex bytes from the host mesh payload.
    pub vertex_slice: &'a [u8],
    /// Index bytes from the host mesh payload.
    pub index_slice: &'a [u8],
    /// Number of vertices in `vertex_slice`.
    pub vertex_count: usize,
    /// Byte stride of one interleaved vertex.
    pub vertex_stride: usize,
    /// Host vertex attribute descriptors, in interleaved order.
    pub vertex_attributes: &'a [VertexAttributeDescriptor],
    /// Host index-buffer format.
    pub index_format: IndexBufferFormat,
    /// Host submesh descriptors.
    pub submeshes: &'a [SubmeshBufferDescriptor],
}

/// Interleaved VB, IB, and layout-derived scalars after validation.
pub(super) struct CoreBuffers {
    pub vb: wgpu::Buffer,
    pub ib: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub vertex_stride: u32,
    pub vertex_stride_us: usize,
    pub index_count_u32: u32,
}

/// Aggregated bone/skin GPU state and skinning matrices.
pub(super) struct BoneSkinUpload {
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    pub skinning_bind_matrices: Vec<Mat4>,
}

/// Position/normal streams, UV0, and vertex color.
pub(super) struct DerivedStreams {
    pub positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub normals_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv0_buffer: Option<Arc<wgpu::Buffer>>,
    pub color_buffer: Option<Arc<wgpu::Buffer>>,
    pub tangent_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv1_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv2_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv3_buffer: Option<Arc<wgpu::Buffer>>,
}

/// Validates raw length and device buffer-size limits, including per-derived-stream sizes
/// that would otherwise reach `device.create_buffer_init` and trigger a fatal panic in
/// `wgpu`'s `get_mapped_range` when the underlying buffer creation fails validation.
///
/// Returns `false` when the upload must abort.
pub(super) fn validate_mesh_upload_layout(
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    gpu_limits: &GpuLimits,
) -> bool {
    if raw.len() < layout.total_buffer_length {
        logger::error!(
            "mesh {}: raw too short (need {}, got {})",
            data.asset_id,
            layout.total_buffer_length,
            raw.len()
        );
        return false;
    }

    let max_buf = gpu_limits.max_buffer_size();
    let max_storage = gpu_limits.max_storage_buffer_binding_size();
    let vc = data.vertex_count.max(0) as u64;

    // (label, size in bytes, requires storage-binding limit)
    //
    // Derived per-vertex storage streams (positions, normals, bone_indices, bone_weights_vec4)
    // are all `vc * 16` and bound to STORAGE bindings, so a single entry covers all of them.
    // Likewise the largest VERTEX-only derived stream (color, tangent) is `vc * 16`.
    let checks: [(&str, u64, bool); 7] = [
        ("interleaved vertices", layout.vertex_size as u64, false),
        ("indices", layout.index_buffer_length as u64, false),
        (
            "total mesh layout",
            layout.total_buffer_length as u64,
            false,
        ),
        ("derived per-vertex storage stream", vc * 16, true),
        ("derived per-vertex vertex stream", vc * 16, false),
        ("bone_counts", layout.bone_counts_length as u64, true),
        ("bind_poses", layout.bind_poses_length as u64, true),
    ];

    for (label, size, is_storage) in checks {
        if size > max_buf {
            logger::warn!(
                "mesh {}: {} buffer ({} B) exceeds max_buffer_size ({} B)",
                data.asset_id,
                label,
                size,
                max_buf
            );
            return false;
        }
        if is_storage && size > max_storage {
            logger::warn!(
                "mesh {}: {} buffer ({} B) exceeds max_storage_buffer_binding_size ({} B)",
                data.asset_id,
                label,
                size,
                max_storage
            );
            return false;
        }
    }

    true
}

/// Creates core vertex and index buffers from the layout-validated `raw` slice.
pub(super) fn create_core_vertex_index_buffers(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
) -> CoreBuffers {
    profiling::scope!("asset::mesh_create_core_buffers");
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
    let vertex_stride_us = vertex_stride as usize;
    let index_count = compute_index_count(&data.submeshes);
    let index_count_u32 = index_count.max(0) as u32;

    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} vertices", data.asset_id)),
        contents: &raw[..layout.vertex_size],
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let ib_slice =
        &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} indices", data.asset_id)),
        contents: ib_slice,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });

    let index_format = wgpu_index_format(data.index_buffer_format);

    CoreBuffers {
        vb,
        ib,
        index_format,
        vertex_stride,
        vertex_stride_us,
        index_count_u32,
    }
}

fn upload_positions_normals(
    device: &wgpu::Device,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>) {
    if let Some((pb, nb)) = extract_float3_position_normal_as_vec4_streams(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    ) {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} positions_stream", data.asset_id)),
            contents: &pb,
            usage,
        });
        let nbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} normals_stream", data.asset_id)),
            contents: &nb,
            usage,
        });
        (Some(Arc::new(pbuf)), Some(Arc::new(nbuf)))
    } else {
        logger::warn!(
            "mesh {}: missing float3 position+normal attributes -- debug/deform path disabled",
            data.asset_id
        );
        (None, None)
    }
}

fn upload_uv0_color(
    device: &wgpu::Device,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>) {
    let uv0_buffer = uv0_float2_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    )
    .map(|uv_bytes| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} uv0_stream", data.asset_id)),
                contents: &uv_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        )
    });
    let color_buffer = color_float4_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    )
    .map(|color_bytes| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} color_stream", data.asset_id)),
                contents: &color_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        )
    });
    (uv0_buffer, color_buffer)
}

fn float4_default_stream_bytes(vertex_count: usize, default: [f32; 4]) -> Vec<u8> {
    let mut out = vec![0u8; vertex_count * 16];
    for chunk in out.chunks_exact_mut(16) {
        for (component, value) in default.iter().enumerate() {
            let o = component * 4;
            chunk[o..o + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn float2_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * 8]
}

fn create_vertex_stream_buffer(
    device: &wgpu::Device,
    asset_id: i32,
    label: &str,
    bytes: &[u8],
) -> Arc<wgpu::Buffer> {
    Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} {label}_stream")),
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }),
    )
}

pub(super) fn upload_extended_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
) -> ExtendedVertexStreams {
    let vc_usize = source.vertex_count;
    if vc_usize == 0 {
        return (None, None, None, None);
    }

    let tangent_bytes = tangent_stream_bytes(
        source.vertex_slice,
        source.index_slice,
        vc_usize,
        source.vertex_stride,
        source.vertex_attributes,
        source.index_format,
        source.submeshes,
    )
    .unwrap_or_else(|| float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]));

    let make_uv = |target: VertexAttributeType, label: &str| {
        let bytes = vertex_float2_stream_bytes(
            source.vertex_slice,
            vc_usize,
            source.vertex_stride,
            source.vertex_attributes,
            target,
        )
        .unwrap_or_else(|| float2_zero_stream_bytes(vc_usize));
        create_vertex_stream_buffer(device, asset_id, label, &bytes)
    };

    (
        Some(create_vertex_stream_buffer(
            device,
            asset_id,
            "tangent",
            &tangent_bytes,
        )),
        Some(make_uv(VertexAttributeType::UV1, "uv1")),
        Some(make_uv(VertexAttributeType::UV2, "uv2")),
        Some(make_uv(VertexAttributeType::UV3, "uv3")),
    )
}

pub(super) fn upload_uv1_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
    vertex_attributes: &[VertexAttributeDescriptor],
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv1_bytes = vertex_float2_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        vertex_attributes,
        VertexAttributeType::UV1,
    )
    .unwrap_or_else(|| float2_zero_stream_bytes(vc_usize));
    Some(create_vertex_stream_buffer(
        device, asset_id, "uv1", &uv1_bytes,
    ))
}

pub(super) fn upload_default_uv1_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv1_bytes = float2_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device, asset_id, "uv1", &uv1_bytes,
    ))
}

pub(super) fn upload_default_extended_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> ExtendedVertexStreams {
    if vc_usize == 0 {
        return (None, None, None, None);
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]);
    let uv_bytes = float2_zero_stream_bytes(vc_usize);
    (
        Some(create_vertex_stream_buffer(
            device,
            asset_id,
            "tangent",
            &tangent_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv1", &uv_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv2", &uv_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv3", &uv_bytes,
        )),
    )
}

/// Builds optional position/normal streams plus UV0 and vertex color buffers.
pub(super) fn extract_derived_vertex_streams(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    core: &CoreBuffers,
) -> DerivedStreams {
    profiling::scope!("asset::mesh_extract_derived_streams");
    let vc_usize = data.vertex_count.max(0) as usize;
    let vertex_slice = &raw[..layout.vertex_size];
    let (positions_buffer, normals_buffer) =
        upload_positions_normals(device, data, vertex_slice, vc_usize, core.vertex_stride_us);
    let (uv0_buffer, color_buffer) =
        upload_uv0_color(device, data, vertex_slice, vc_usize, core.vertex_stride_us);
    //perf xlinka: tangent/UV1-3 are big; build them only if a shader actually asks for them.
    DerivedStreams {
        positions_buffer,
        normals_buffer,
        uv0_buffer,
        color_buffer,
        tangent_buffer: None,
        uv1_buffer: None,
        uv2_buffer: None,
        uv3_buffer: None,
    }
}

fn upload_skeleton_bone_buffers(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    profiling::scope!("asset::mesh_upload_skeleton_buffers");
    let bp_raw = &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
    let bind_poses_arr = extract_bind_poses(bp_raw, data.bone_count as usize)?;
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    let skinning: Vec<Mat4> = bind_poses_arr
        .iter()
        .map(Mat4::from_cols_array_2d)
        .collect();

    let bc = &raw[layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
    let bw =
        &raw[layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
    let (bi_buf, bw_buf) = if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(bw, vc_usize) {
        let bi = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} bone_indices", data.asset_id)),
            contents: &ib,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bwt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} bone_weights_vec4", data.asset_id)),
            contents: &wb,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        (Some(Arc::new(bi)), Some(Arc::new(bwt)))
    } else {
        logger::warn!(
            "mesh {}: bone weight tail could not be repacked for GPU skinning",
            data.asset_id
        );
        (None, None)
    };

    let bc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bone_counts", data.asset_id)),
        contents: bc,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let bp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bind_poses", data.asset_id)),
        contents: &bp_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    Some(BoneSkinUpload {
        bone_counts_buffer: Some(Arc::new(bc_buf)),
        bone_indices_buffer: bi_buf,
        bone_weights_vec4_buffer: bw_buf,
        bind_poses_buffer: Some(Arc::new(bp_buf)),
        skinning_bind_matrices: skinning,
    })
}

fn upload_synthetic_blend_bone_buffers(
    device: &wgpu::Device,
    data: &MeshUploadData,
    vc_usize: usize,
) -> BoneSkinUpload {
    profiling::scope!("asset::mesh_upload_synthetic_bone_buffers");
    let (bind_poses_arr, bone_counts, bone_weights) =
        synthetic_bone_data_for_blendshape_only(data.vertex_count);
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    let skinning: Vec<Mat4> = bind_poses_arr
        .iter()
        .map(Mat4::from_cols_array_2d)
        .collect();
    let (bi_buf, bw_buf) = split_bone_weights_tail_for_gpu(&bone_weights, vc_usize).map_or(
        (None, None),
        |(ib, wb)| {
            let bi = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_indices synth", data.asset_id)),
                contents: &ib,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bwt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights_vec4 synth", data.asset_id)),
                contents: &wb,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            (Some(Arc::new(bi)), Some(Arc::new(bwt)))
        },
    );
    let bc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bone_counts synth", data.asset_id)),
        contents: &bone_counts,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let bp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bind_poses synth", data.asset_id)),
        contents: &bp_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    BoneSkinUpload {
        bone_counts_buffer: Some(Arc::new(bc_buf)),
        bone_indices_buffer: bi_buf,
        bone_weights_vec4_buffer: bw_buf,
        bind_poses_buffer: Some(Arc::new(bp_buf)),
        skinning_bind_matrices: skinning,
    }
}

/// Bone indices/weights, bind poses, and skinning matrices for skeleton or blendshape-only paths.
///
/// Returns [`None`] when the real-skeleton bind-pose slice is invalid ([`extract_bind_poses`]).
pub(super) fn upload_bone_and_skin_buffers(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    use_blendshapes: bool,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    profiling::scope!("asset::mesh_upload_bone_skin_buffers");
    if data.bone_count > 0 {
        upload_skeleton_bone_buffers(device, raw, data, layout, vc_usize)
    } else if use_blendshapes && data.vertex_count > 0 {
        Some(upload_synthetic_blend_bone_buffers(device, data, vc_usize))
    } else {
        Some(BoneSkinUpload {
            bone_counts_buffer: None,
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bind_poses_buffer: None,
            skinning_bind_matrices: Vec::new(),
        })
    }
}

/// Sparse GPU buffers and CPU scatter ranges produced by `layout::extract_blendshape_offsets`.
pub(super) struct BlendshapeBuffersUpload {
    /// Storage buffer of packed sparse position deltas (padded when empty; see `BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES`).
    pub sparse_buffer: Option<Arc<wgpu::Buffer>>,
    /// Storage buffer of per-frame `(first_entry, entry_count)` descriptor rows.
    pub shape_descriptor_buffer: Option<Arc<wgpu::Buffer>>,
    /// Copy of frame rows for CPU-side scatter dispatch.
    pub frame_ranges: Vec<BlendshapeFrameRange>,
    /// Per-shape spans into [`Self::frame_ranges`].
    pub shape_frame_spans: Vec<BlendshapeFrameSpan>,
    /// Logical blendshape slot count for weight indexing.
    pub num_blendshapes: u32,
}

/// Pads sparse CPU bytes to at least [`BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES`] for `wgpu` buffers.
pub(super) fn padded_sparse_bytes(sparse_deltas: &[u8]) -> Vec<u8> {
    let mut v = sparse_deltas.to_vec();
    let min = BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES as usize;
    if v.len() < min {
        v.resize(min, 0);
    }
    v
}

/// Builds sparse / descriptor GPU buffers (or empty upload when blendshapes are disabled).
pub(super) fn upload_blendshape_buffer(
    device: &wgpu::Device,
    gpu_limits: &GpuLimits,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    use_blendshapes: bool,
    max_buf: u64,
) -> BlendshapeBuffersUpload {
    profiling::scope!("asset::mesh_upload_blendshape_buffers");
    if !use_blendshapes {
        return BlendshapeBuffersUpload {
            sparse_buffer: None,
            shape_descriptor_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
        };
    }
    let Some(pack) =
        extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count)
    else {
        return BlendshapeBuffersUpload {
            sparse_buffer: None,
            shape_descriptor_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
        };
    };

    let n_u32 = pack.num_blendshapes.max(0) as u32;
    if !blendshape_sparse_buffers_fit_device(
        &pack,
        max_buf,
        gpu_limits.wgpu.max_storage_buffer_binding_size,
    ) {
        logger::warn!(
            "mesh {}: blendshapes dropped (sparse or descriptor bytes exceed buffer / binding limits)",
            data.asset_id
        );
        return BlendshapeBuffersUpload {
            sparse_buffer: None,
            shape_descriptor_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
        };
    }

    let sparse_bytes = padded_sparse_bytes(&pack.sparse_deltas);
    let desc_bytes = pack.shape_descriptor_bytes.clone();

    let sparse_label = format!("mesh {} blendshape_sparse", data.asset_id);
    let sparse_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&sparse_label),
        contents: &sparse_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let desc_label = format!("mesh {} blendshape_shape_desc", data.asset_id);
    let desc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&desc_label),
        contents: &desc_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    BlendshapeBuffersUpload {
        sparse_buffer: Some(Arc::new(sparse_buf)),
        shape_descriptor_buffer: Some(Arc::new(desc_buf)),
        frame_ranges: pack.frame_ranges,
        shape_frame_spans: pack.shape_frame_spans,
        num_blendshapes: n_u32,
    }
}

pub(super) fn sum_optional_buffer_bytes(buffers: &[Option<&Arc<wgpu::Buffer>>]) -> u64 {
    buffers
        .iter()
        .filter_map(|o| o.as_ref().map(|b| b.size()))
        .sum()
}

/// Sums VRAM for all optional mesh buffers plus fixed vertex/index sizes.
pub(super) fn resident_bytes_for_mesh_upload(
    core_vb: &wgpu::Buffer,
    core_ib: &wgpu::Buffer,
    derived: &DerivedStreams,
    bone_skin: &BoneSkinUpload,
    blend_sparse: Option<&Arc<wgpu::Buffer>>,
    blend_shape_desc: Option<&Arc<wgpu::Buffer>>,
) -> u64 {
    let mut n = core_vb.size() + core_ib.size();
    n += sum_optional_buffer_bytes(&[
        bone_skin.bone_counts_buffer.as_ref(),
        bone_skin.bone_indices_buffer.as_ref(),
        bone_skin.bone_weights_vec4_buffer.as_ref(),
        bone_skin.bind_poses_buffer.as_ref(),
        derived.positions_buffer.as_ref(),
        derived.normals_buffer.as_ref(),
        derived.uv0_buffer.as_ref(),
        derived.color_buffer.as_ref(),
        derived.tangent_buffer.as_ref(),
        derived.uv1_buffer.as_ref(),
        derived.uv2_buffer.as_ref(),
        derived.uv3_buffer.as_ref(),
    ]);
    if let Some(b) = blend_sparse {
        n += b.size();
    }
    if let Some(b) = blend_shape_desc {
        n += b.size();
    }
    n
}
