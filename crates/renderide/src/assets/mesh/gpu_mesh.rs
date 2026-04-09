//! GPU-resident mesh: wgpu buffers only; host layout preserved in one interleaved vertex buffer.

use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::shared::{
    IndexBufferFormat, MeshUploadData, RenderBoundingBox, SubmeshBufferDescriptor,
};

use super::layout::{
    color_float4_stream_bytes, compute_index_count, compute_mesh_buffer_layout,
    compute_vertex_stride, extract_bind_poses, extract_blendshape_offsets,
    extract_float3_position_normal_as_vec4_streams, index_bytes_per_element,
    split_bone_weights_tail_for_gpu, synthetic_bone_data_for_blendshape_only,
    uv0_float2_stream_bytes, MeshBufferLayout,
};

use crate::gpu::plan_blendshape_bind_chunks;

/// Resident mesh on GPU: no CPU geometry retained.
///
/// **Vertex groups** in Renderite are expressed through per-vertex bone influence streams
/// (`bone_counts` + bone weight tail) when the host provides skeleton data.
#[derive(Debug)]
pub struct GpuMesh {
    pub asset_id: i32,
    /// Full interleaved vertices as sent by the host (`vertex_attributes` order).
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_format: wgpu::IndexFormat,
    pub index_count: u32,
    /// Per-submesh `(first_index, index_count)` in elements of `index_format`.
    pub submeshes: Vec<(u32, u32)>,
    pub vertex_count: u32,
    pub vertex_stride: u32,
    pub bounds: RenderBoundingBox,
    /// Optional 1 byte per vertex (skinned / synthetic for blendshape-only).
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex joint indices as `vec4<u32>` (16 bytes / vertex) for skinning compute.
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex bone weights as `vec4<f32>` for skinning compute.
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    /// Column-major `float4x4` bind poses (64 bytes per bone).
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    /// Packed blendshape deltas (`BLENDSHAPE_OFFSET_GPU_STRIDE` × vertices × shapes).
    pub blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    pub num_blendshapes: u32,
    /// Decomposed position stream (`vec4<f32>` per vertex) for compute + debug raster.
    pub positions_buffer: Option<Arc<wgpu::Buffer>>,
    /// Bind-pose normal stream (`vec4<f32>` per vertex; xyz used). Serves as the skinning compute
    /// input; the forward pass binds [`Self::deformed_normals_buffer`] when skinning is active.
    pub normals_buffer: Option<Arc<wgpu::Buffer>>,
    /// Blendshape output and/or skinning input ping buffer (`vec4<f32>` per vertex).
    pub deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    /// Skinning output positions (`vec4<f32>` per vertex).
    pub deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    /// Skinning output normals in world space (`vec4<f32>` per vertex; xyz used), inverse-transpose
    /// LBS of bind-pose normals. Present when [`Self::has_skeleton`].
    pub deformed_normals_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec2<f32>` UV0 stream (`8` bytes/vertex) for embedded raster materials; zeros when uv0 is absent.
    pub uv0_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec4<f32>` color stream for UI/text embedded materials; defaults to opaque white when absent.
    pub color_buffer: Option<Arc<wgpu::Buffer>>,
    /// True when the host uploaded a real skeleton (`bone_count > 0`).
    pub has_skeleton: bool,
    /// Unity [`Mesh.bindposes`](https://docs.unity3d.com/ScriptReference/Mesh-bindposes.html):
    /// inverse bind matrices (mesh space → bone bind space). Per-frame palette is
    /// `world_bone * skinning_bind_matrices[i]`.
    pub skinning_bind_matrices: Vec<Mat4>,
    /// Approximate VRAM (bytes), used by [`crate::resources::VramAccounting`].
    pub resident_bytes: u64,
}

impl GpuMesh {
    /// Uploads mesh data from a raw byte slice covering at least `layout.total_buffer_length`.
    ///
    /// `raw` must be the mapping for `data.buffer` only for the duration of this call.
    pub fn upload(
        device: &wgpu::Device,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
    ) -> Option<Self> {
        if raw.len() < layout.total_buffer_length {
            logger::error!(
                "mesh {}: raw too short (need {}, got {})",
                data.asset_id,
                layout.total_buffer_length,
                raw.len()
            );
            return None;
        }

        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let vertex_stride_us = vertex_stride as usize;
        let index_count = compute_index_count(&data.submeshes);
        let index_count_u32 = index_count.max(0) as u32;
        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();

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
        let vc_usize = data.vertex_count.max(0) as usize;

        let vertex_slice = &raw[..layout.vertex_size];
        let (positions_buffer, normals_buffer) =
            match extract_float3_position_normal_as_vec4_streams(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
            ) {
                Some((pb, nb)) => {
                    let usage = wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST;
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
                }
                None => {
                    logger::warn!(
                        "mesh {}: missing float3 position+normal attributes — debug/deform path disabled",
                        data.asset_id
                    );
                    (None, None)
                }
            };

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

        let (
            bone_counts_buffer,
            bone_indices_buffer,
            bone_weights_vec4_buffer,
            bind_poses_buffer,
            skinning_bind_matrices,
        ) = if data.bone_count > 0 {
            let bp_raw =
                &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
            let bind_poses_arr = extract_bind_poses(bp_raw, data.bone_count as usize)?;
            let bp_bytes: Vec<u8> = bind_poses_arr
                .iter()
                .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                .collect();
            let skinning: Vec<Mat4> = bind_poses_arr
                .iter()
                .map(Mat4::from_cols_array_2d)
                .collect();

            let bc = &raw
                [layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
            let bw = &raw
                [layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
            let (bi_buf, bw_buf) = match split_bone_weights_tail_for_gpu(bw, vc_usize) {
                Some((ib, wb)) => {
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
                }
                None => {
                    logger::warn!(
                        "mesh {}: bone weight tail could not be repacked for GPU skinning",
                        data.asset_id
                    );
                    (None, None)
                }
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
            (
                Some(Arc::new(bc_buf)),
                bi_buf,
                bw_buf,
                Some(Arc::new(bp_buf)),
                skinning,
            )
        } else if use_blendshapes && data.vertex_count > 0 {
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
            let (bi_buf, bw_buf) = split_bone_weights_tail_for_gpu(&bone_weights, vc_usize)
                .map(|(ib, wb)| {
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
                })
                .unwrap_or((None, None));
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
            (
                Some(Arc::new(bc_buf)),
                bi_buf,
                bw_buf,
                Some(Arc::new(bp_buf)),
                skinning,
            )
        } else {
            (None, None, None, None, Vec::new())
        };

        let (blendshape_buffer, num_blendshapes) = if use_blendshapes {
            match extract_blendshape_offsets(
                raw,
                layout,
                &data.blendshape_buffers,
                data.vertex_count,
            ) {
                Some((pack, n)) if !pack.is_empty() => {
                    let vc_u32 = data.vertex_count.max(0) as u32;
                    let n_u32 = n.max(0) as u32;
                    let lims = device.limits();
                    if plan_blendshape_bind_chunks(
                        n_u32,
                        vc_u32,
                        lims.max_storage_buffer_binding_size,
                        lims.min_storage_buffer_offset_alignment,
                    )
                    .is_none()
                    {
                        logger::warn!(
                            "mesh {}: blendshapes dropped ({} shapes × {} verts exceed binding limit {} or offset alignment)",
                            data.asset_id,
                            n_u32,
                            vc_u32,
                            lims.max_storage_buffer_binding_size
                        );
                        (None, 0)
                    } else {
                        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("mesh {} blendshapes", data.asset_id)),
                            contents: &pack,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        });
                        (Some(Arc::new(buf)), n_u32)
                    }
                }
                _ => (None, 0),
            }
        } else {
            (None, 0)
        };

        let has_skeleton = data.bone_count > 0;
        let needs_blend_compute = num_blendshapes > 0;
        let needs_skin_compute = has_skeleton;

        let deform_usage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let deform_temp_buffer = if needs_blend_compute {
            let len = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
            Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("mesh {} deform_temp", data.asset_id)),
                size: len,
                usage: deform_usage,
                mapped_at_creation: false,
            })))
        } else {
            None
        };

        let (deformed_positions_buffer, deformed_normals_buffer) = if needs_skin_compute {
            let len = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
            let pos = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("mesh {} deformed_positions", data.asset_id)),
                size: len,
                usage: deform_usage,
                mapped_at_creation: false,
            })));
            let nrm = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("mesh {} deformed_normals", data.asset_id)),
                size: len,
                usage: deform_usage,
                mapped_at_creation: false,
            })));
            (pos, nrm)
        } else {
            (None, None)
        };

        let submeshes = validated_submesh_ranges(&data.submeshes, index_count_u32);

        let mut resident_bytes = vb.size() + ib.size();
        if let Some(ref b) = bone_counts_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = bone_indices_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = bone_weights_vec4_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = bind_poses_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = blendshape_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = positions_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = normals_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = deform_temp_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = deformed_positions_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = deformed_normals_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = uv0_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = color_buffer {
            resident_bytes += b.size();
        }

        Some(Self {
            asset_id: data.asset_id,
            vertex_buffer: Arc::new(vb),
            index_buffer: Arc::new(ib),
            index_format,
            index_count: index_count_u32,
            submeshes,
            vertex_count: data.vertex_count.max(0) as u32,
            vertex_stride,
            bounds: data.bounds,
            bone_counts_buffer,
            bone_indices_buffer,
            bone_weights_vec4_buffer,
            bind_poses_buffer,
            blendshape_buffer,
            num_blendshapes,
            positions_buffer,
            normals_buffer,
            deform_temp_buffer,
            deformed_positions_buffer,
            deformed_normals_buffer,
            uv0_buffer,
            color_buffer,
            has_skeleton,
            skinning_bind_matrices,
            resident_bytes,
        })
    }

    /// `true` when [`Self::positions_buffer`] and [`Self::normals_buffer`] exist for the debug mesh path.
    pub fn debug_streams_ready(&self) -> bool {
        self.positions_buffer.is_some() && self.normals_buffer.is_some()
    }
}

/// Builds layout and uploads; returns [`GpuMesh`] if validation and GPU creation succeed.
pub fn try_upload_mesh_from_raw(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
) -> Option<GpuMesh> {
    if data.buffer.length <= 0 {
        return None;
    }
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes);
    if vertex_stride <= 0 {
        logger::error!("mesh {}: invalid vertex stride", data.asset_id);
        return None;
    }
    let index_count = compute_index_count(&data.submeshes);
    let index_bytes = index_bytes_per_element(data.index_buffer_format);
    let layout = match compute_mesh_buffer_layout(
        vertex_stride,
        data.vertex_count,
        index_count,
        index_bytes,
        data.bone_count,
        data.bone_weight_count,
        Some(&data.blendshape_buffers),
    ) {
        Ok(l) => l,
        Err(e) => {
            logger::error!("mesh {}: layout error: {}", data.asset_id, e);
            return None;
        }
    };

    let expected_bone_weights_len = (data.bone_weight_count.max(0) * 8) as usize;
    let expected_bind_poses_len = (data.bone_count.max(0) * 64) as usize;
    if layout.bone_weights_length != expected_bone_weights_len {
        logger::error!("mesh {}: bone_weights layout mismatch", data.asset_id);
        return None;
    }
    if layout.bind_poses_length != expected_bind_poses_len {
        logger::error!("mesh {}: bind poses layout mismatch", data.asset_id);
        return None;
    }

    GpuMesh::upload(device, raw, data, &layout)
}

fn wgpu_index_format(f: IndexBufferFormat) -> wgpu::IndexFormat {
    match f {
        IndexBufferFormat::u_int16 => wgpu::IndexFormat::Uint16,
        IndexBufferFormat::u_int32 => wgpu::IndexFormat::Uint32,
    }
}

fn validated_submesh_ranges(
    submeshes: &[SubmeshBufferDescriptor],
    index_count_u32: u32,
) -> Vec<(u32, u32)> {
    if submeshes.is_empty() {
        if index_count_u32 > 0 {
            return vec![(0, index_count_u32)];
        }
        return Vec::new();
    }
    let valid: Vec<(u32, u32)> = submeshes
        .iter()
        .filter(|s| {
            s.index_count > 0
                && (s.index_start as i64 + s.index_count as i64) <= index_count_u32 as i64
        })
        .map(|s| (s.index_start as u32, s.index_count as u32))
        .collect();
    if valid.is_empty() && index_count_u32 > 0 {
        vec![(0, index_count_u32)]
    } else {
        valid
    }
}
