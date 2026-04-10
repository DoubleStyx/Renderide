//! GPU-resident mesh: wgpu buffers only; host layout preserved in one interleaved vertex buffer.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, MeshUploadData, MeshUploadHintFlag,
    RenderBoundingBox, SubmeshBufferDescriptor,
};

use super::layout::{
    color_float4_stream_bytes, compute_index_count, compute_mesh_buffer_layout,
    compute_vertex_stride, extract_bind_poses, extract_blendshape_offsets,
    extract_float3_position_normal_as_vec4_streams, index_bytes_per_element,
    split_bone_weights_tail_for_gpu, synthetic_bone_data_for_blendshape_only,
    uv0_float2_stream_bytes, MeshBufferLayout, BLENDSHAPE_OFFSET_GPU_STRIDE,
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
                    let pack_len = pack.len() as u64;
                    if pack_len > lims.max_buffer_size {
                        logger::warn!(
                            "mesh {}: blendshapes dropped (packed size {} bytes exceeds device max_buffer_size {})",
                            data.asset_id,
                            pack_len,
                            lims.max_buffer_size
                        );
                        (None, 0)
                    } else if plan_blendshape_bind_chunks(
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

    /// Whether `data`/`layout` match this mesh’s buffer sizes and optional derived streams so we can
    /// [`Self::write_in_place`] instead of allocating new buffers.
    fn compatible_for_in_place_update(
        &self,
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        raw: &[u8],
    ) -> bool {
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
        let no_gpu_blend = self.blendshape_buffer.is_none() && self.num_blendshapes == 0;

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

        let n_blend = blendshape_descriptor_count(&data.blendshape_buffers);
        if use_blendshapes && n_blend > 0 {
            let expected = n_blend as usize * vc_usize * BLENDSHAPE_OFFSET_GPU_STRIDE;
            if self.blendshape_buffer.as_ref().map(|b| b.size()) != Some(expected as u64)
                || self.num_blendshapes != n_blend
            {
                return false;
            }
        } else if self.num_blendshapes > 0 || self.blendshape_buffer.is_some() {
            return false;
        }

        if self.num_blendshapes > 0 {
            let need = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
            if self.deform_temp_buffer.as_ref().map(|b| b.size()) != Some(need) {
                return false;
            }
        }
        if self.has_skeleton {
            let need = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
            if self.deformed_positions_buffer.as_ref().map(|b| b.size()) != Some(need) {
                return false;
            }
            if self.deformed_normals_buffer.as_ref().map(|b| b.size()) != Some(need) {
                return false;
            }
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
            let bw = &raw
                [layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
            match split_bone_weights_tail_for_gpu(bw, vc_usize) {
                Some((ref ib, ref wb)) => {
                    if self.bone_indices_buffer.as_ref().map(|b| b.size()) != Some(ib.len() as u64)
                    {
                        return false;
                    }
                    if self.bone_weights_vec4_buffer.as_ref().map(|b| b.size())
                        != Some(wb.len() as u64)
                    {
                        return false;
                    }
                }
                None => {
                    if self.bone_indices_buffer.is_some() || self.bone_weights_vec4_buffer.is_some()
                    {
                        return false;
                    }
                }
            }
            if self.bone_counts_buffer.as_ref().map(|b| b.size())
                != Some(layout.bone_counts_length as u64)
            {
                return false;
            }
            if self.bind_poses_buffer.as_ref().map(|b| b.size())
                != Some(layout.bind_poses_length as u64)
            {
                return false;
            }
            if self.skinning_bind_matrices.len() != data.bone_count.max(0) as usize {
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

        if use_blendshapes && data.vertex_count > 0 {
            let (bind_poses_arr, bone_counts, bone_weights) =
                synthetic_bone_data_for_blendshape_only(data.vertex_count);
            if self.bone_counts_buffer.as_ref().map(|b| b.size()) != Some(bone_counts.len() as u64)
            {
                return false;
            }
            if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(&bone_weights, vc_usize) {
                if self.bone_indices_buffer.as_ref().map(|b| b.size()) != Some(ib.len() as u64) {
                    return false;
                }
                if self.bone_weights_vec4_buffer.as_ref().map(|b| b.size()) != Some(wb.len() as u64)
                {
                    return false;
                }
            } else {
                return false;
            }
            let bp_bytes: Vec<u8> = bind_poses_arr
                .iter()
                .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                .collect();
            if self.bind_poses_buffer.as_ref().map(|b| b.size()) != Some(bp_bytes.len() as u64) {
                return false;
            }
            if self.skinning_bind_matrices.len() != 1 {
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

        false
    }

    /// Overwrites vertex, index, and optional bone/blendshape/derived stream data using
    /// [`wgpu::Queue::write_buffer`], honoring [`MeshUploadHintFlag`] when set (otherwise full writes).
    fn write_in_place(
        &self,
        queue: &wgpu::Queue,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        hint: MeshUploadHintFlag,
    ) -> Option<GpuMesh> {
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

        if write_vertex {
            queue.write_buffer(self.vertex_buffer.as_ref(), 0, &raw[..layout.vertex_size]);
        }
        let vertex_slice = &raw[..layout.vertex_size];
        if write_vertex {
            if let (Some(pb), Some(nb), Some((pvec, nvec))) = (
                self.positions_buffer.as_ref(),
                self.normals_buffer.as_ref(),
                extract_float3_position_normal_as_vec4_streams(
                    vertex_slice,
                    vc_usize,
                    vertex_stride_us,
                    &data.vertex_attributes,
                )
                .as_ref(),
            ) {
                queue.write_buffer(pb.as_ref(), 0, pvec);
                queue.write_buffer(nb.as_ref(), 0, nvec);
            }

            if let (Some(uvb), Some(uv)) = (
                self.uv0_buffer.as_ref(),
                uv0_float2_stream_bytes(
                    vertex_slice,
                    vc_usize,
                    vertex_stride_us,
                    &data.vertex_attributes,
                ),
            ) {
                queue.write_buffer(uvb.as_ref(), 0, &uv);
            }

            if let (Some(cb), Some(c)) = (
                self.color_buffer.as_ref(),
                color_float4_stream_bytes(
                    vertex_slice,
                    vc_usize,
                    vertex_stride_us,
                    &data.vertex_attributes,
                ),
            ) {
                queue.write_buffer(cb.as_ref(), 0, &c);
            }
        }

        if write_ib {
            let ib_slice = &raw
                [layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
            queue.write_buffer(self.index_buffer.as_ref(), 0, ib_slice);
        }

        if needs_bone_buffers {
            if synthetic_bones && (full || write_bone_weights || write_bind_poses) {
                let (bind_poses_arr, bone_counts, bone_weights) =
                    synthetic_bone_data_for_blendshape_only(data.vertex_count);
                if let Some(bc) = &self.bone_counts_buffer {
                    queue.write_buffer(bc.as_ref(), 0, &bone_counts);
                }
                if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(&bone_weights, vc_usize) {
                    if let Some(bi) = &self.bone_indices_buffer {
                        queue.write_buffer(bi.as_ref(), 0, &ib);
                    }
                    if let Some(bwt) = &self.bone_weights_vec4_buffer {
                        queue.write_buffer(bwt.as_ref(), 0, &wb);
                    }
                }
                let bp_bytes: Vec<u8> = bind_poses_arr
                    .iter()
                    .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                    .collect();
                if let Some(bp) = &self.bind_poses_buffer {
                    queue.write_buffer(bp.as_ref(), 0, &bp_bytes);
                }
            } else if data.bone_count > 0 {
                if full || write_bone_weights {
                    let bc = &raw[layout.bone_counts_start
                        ..layout.bone_counts_start + layout.bone_counts_length];
                    let bw = &raw[layout.bone_weights_start
                        ..layout.bone_weights_start + layout.bone_weights_length];
                    if let Some(bcb) = &self.bone_counts_buffer {
                        queue.write_buffer(bcb.as_ref(), 0, bc);
                    }
                    if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(bw, vc_usize) {
                        if let Some(bi) = &self.bone_indices_buffer {
                            queue.write_buffer(bi.as_ref(), 0, &ib);
                        }
                        if let Some(bwt) = &self.bone_weights_vec4_buffer {
                            queue.write_buffer(bwt.as_ref(), 0, &wb);
                        }
                    }
                }
                if full || write_bind_poses {
                    let bp_raw = &raw[layout.bind_poses_start
                        ..layout.bind_poses_start + layout.bind_poses_length];
                    if let Some(bp) = &self.bind_poses_buffer {
                        let bind_poses_arr = extract_bind_poses(bp_raw, data.bone_count as usize)?;
                        let bp_bytes: Vec<u8> = bind_poses_arr
                            .iter()
                            .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                            .collect();
                        queue.write_buffer(bp.as_ref(), 0, &bp_bytes);
                    }
                }
            }
        }

        if write_blend {
            if let Some(bb) = &self.blendshape_buffer {
                let (pack, _) = extract_blendshape_offsets(
                    raw,
                    layout,
                    &data.blendshape_buffers,
                    data.vertex_count,
                )?;
                queue.write_buffer(bb.as_ref(), 0, &pack);
            }
        }

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
            blendshape_buffer: self.blendshape_buffer.clone(),
            num_blendshapes: self.num_blendshapes,
            positions_buffer: self.positions_buffer.clone(),
            normals_buffer: self.normals_buffer.clone(),
            deform_temp_buffer: self.deform_temp_buffer.clone(),
            deformed_positions_buffer: self.deformed_positions_buffer.clone(),
            deformed_normals_buffer: self.deformed_normals_buffer.clone(),
            uv0_buffer: self.uv0_buffer.clone(),
            color_buffer: self.color_buffer.clone(),
            has_skeleton: self.has_skeleton,
            skinning_bind_matrices: skinning,
            resident_bytes: self.resident_bytes,
        })
    }
}

/// Stable hash of host layout metadata and buffer byte sizes (for tracing and cache keys).
///
/// Does not hash the vertex/index payload bytes.
pub fn mesh_layout_fingerprint(data: &MeshUploadData, layout: &MeshBufferLayout) -> u64 {
    let mut h = DefaultHasher::new();
    data.asset_id.hash(&mut h);
    data.vertex_count.hash(&mut h);
    data.bone_count.hash(&mut h);
    data.bone_weight_count.hash(&mut h);
    (data.index_buffer_format as i32).hash(&mut h);
    data.vertex_attributes.len().hash(&mut h);
    for a in &data.vertex_attributes {
        (a.attribute as i32).hash(&mut h);
        (a.format as i32).hash(&mut h);
        a.dimensions.hash(&mut h);
    }
    data.submeshes.len().hash(&mut h);
    for s in &data.submeshes {
        (s.topology as i32).hash(&mut h);
        s.index_start.hash(&mut h);
        s.index_count.hash(&mut h);
    }
    data.blendshape_buffers.len().hash(&mut h);
    data.upload_hint.flags.0.hash(&mut h);
    layout.vertex_size.hash(&mut h);
    layout.index_buffer_length.hash(&mut h);
    layout.total_buffer_length.hash(&mut h);
    h.finish()
}

/// Fingerprint of inputs that determine [`compute_mesh_buffer_layout`] (no raw payload bytes).
pub fn mesh_upload_input_fingerprint(data: &MeshUploadData) -> u64 {
    let mut h = DefaultHasher::new();
    data.asset_id.hash(&mut h);
    data.vertex_count.hash(&mut h);
    data.bone_count.hash(&mut h);
    data.bone_weight_count.hash(&mut h);
    (data.index_buffer_format as i32).hash(&mut h);
    data.vertex_attributes.len().hash(&mut h);
    for a in &data.vertex_attributes {
        (a.attribute as i32).hash(&mut h);
        (a.format as i32).hash(&mut h);
        a.dimensions.hash(&mut h);
    }
    data.submeshes.len().hash(&mut h);
    for s in &data.submeshes {
        (s.topology as i32).hash(&mut h);
        s.index_start.hash(&mut h);
        s.index_count.hash(&mut h);
    }
    data.blendshape_buffers.len().hash(&mut h);
    for b in &data.blendshape_buffers {
        b.blendshape_index.hash(&mut h);
        b.data_flags.0.hash(&mut h);
        b.frame_weight.to_bits().hash(&mut h);
    }
    h.finish()
}

/// Computes [`MeshBufferLayout`] from [`MeshUploadData`] and validates bone region lengths.
pub fn compute_and_validate_mesh_layout(data: &MeshUploadData) -> Option<MeshBufferLayout> {
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
    Some(layout)
}

/// Builds layout and uploads; returns [`GpuMesh`] if validation and GPU creation succeed.
///
/// When `queue` and `existing` refer to the resident [`GpuMesh`] for `data.asset_id`, and topology
/// matches, **reuses** existing `wgpu::Buffer` allocations and uses [`wgpu::Queue::write_buffer`]
/// instead of allocating new buffers with `create_buffer_init`.
pub fn try_upload_mesh_from_raw(
    device: &wgpu::Device,
    queue: Option<&wgpu::Queue>,
    raw: &[u8],
    data: &MeshUploadData,
    existing: Option<&GpuMesh>,
    layout: &MeshBufferLayout,
) -> Option<GpuMesh> {
    if raw.len() < layout.total_buffer_length {
        logger::error!(
            "mesh {}: raw too short (need {}, got {})",
            data.asset_id,
            layout.total_buffer_length,
            raw.len()
        );
        return None;
    }

    let layout_fp = mesh_layout_fingerprint(data, layout);
    let hint = data.upload_hint.flags;

    if let (Some(queue), Some(existing)) = (queue, existing) {
        if existing.compatible_for_in_place_update(data, layout, raw) {
            if let Some(mesh) = existing.write_in_place(queue, raw, data, layout, hint) {
                logger::trace!(
                    "mesh {}: in-place upload (layout_fp={:#x})",
                    data.asset_id,
                    layout_fp
                );
                return Some(mesh);
            }
        }
    }

    logger::trace!(
        "mesh {}: full GPU buffer upload (layout_fp={:#x})",
        data.asset_id,
        layout_fp
    );
    GpuMesh::upload(device, raw, data, layout)
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

fn derived_streams_compatible_for_in_place(
    gpu: &GpuMesh,
    vertex_slice: &[u8],
    data: &MeshUploadData,
    vc_usize: usize,
    vertex_stride_us: usize,
) -> bool {
    let pos_norm = extract_float3_position_normal_as_vec4_streams(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    );
    match (
        &gpu.positions_buffer,
        &gpu.normals_buffer,
        pos_norm.as_ref(),
    ) {
        (Some(pb), Some(nb), Some((pvec, nvec))) => {
            if pb.size() != pvec.len() as u64 || nb.size() != nvec.len() as u64 {
                return false;
            }
        }
        (None, None, None) => {}
        _ => return false,
    }

    let uv_new = uv0_float2_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    );
    match (&gpu.uv0_buffer, uv_new.as_ref()) {
        (Some(b), Some(uv)) => {
            if b.size() != uv.len() as u64 {
                return false;
            }
        }
        (None, None) => {}
        _ => return false,
    }

    let color_new = color_float4_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    );
    match (&gpu.color_buffer, color_new.as_ref()) {
        (Some(b), Some(c)) => {
            if b.size() != c.len() as u64 {
                return false;
            }
        }
        (None, None) => {}
        _ => return false,
    }

    true
}

fn mesh_upload_hint_any_selective(h: MeshUploadHintFlag) -> bool {
    h.vertex_layout()
        || h.positions()
        || h.normals()
        || h.tangents()
        || h.colors()
        || h.uv0s()
        || h.uv1s()
        || h.uv2s()
        || h.uv3s()
        || h.uv4s()
        || h.uv5s()
        || h.uv6s()
        || h.uv7s()
        || h.geometry()
        || h.submesh_layout()
        || h.bone_weights()
        || h.bind_poses()
        || h.blendshapes()
}

fn mesh_upload_hint_touches_vertex_streams(h: MeshUploadHintFlag) -> bool {
    h.vertex_layout()
        || h.positions()
        || h.normals()
        || h.tangents()
        || h.colors()
        || h.uv0s()
        || h.uv1s()
        || h.uv2s()
        || h.uv3s()
        || h.uv4s()
        || h.uv5s()
        || h.uv6s()
        || h.uv7s()
}

fn blendshape_descriptor_count(descs: &[BlendshapeBufferDescriptor]) -> u32 {
    descs
        .iter()
        .map(|d| d.blendshape_index.max(0) as u32 + 1)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_upload_hint_any_selective_false_when_empty() {
        assert!(!mesh_upload_hint_any_selective(MeshUploadHintFlag(0)));
    }

    #[test]
    fn mesh_upload_hint_any_selective_true_for_geometry() {
        assert!(mesh_upload_hint_any_selective(MeshUploadHintFlag(
            MeshUploadHintFlag::GEOMETRY
        )));
    }

    #[test]
    fn mesh_upload_hint_touches_vertex_streams_for_positions() {
        assert!(mesh_upload_hint_touches_vertex_streams(MeshUploadHintFlag(
            MeshUploadHintFlag::POSITIONS
        )));
        assert!(!mesh_upload_hint_touches_vertex_streams(
            MeshUploadHintFlag(MeshUploadHintFlag::GEOMETRY)
        ));
    }
}
