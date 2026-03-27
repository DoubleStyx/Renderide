//! Upload [`MeshAsset`](crate::assets::MeshAsset) vertex and index data into [`super::buffers::GpuMeshBuffers`].

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::assets::{self, MeshAsset};
use crate::shared::{VertexAttributeFormat, VertexAttributeType};

use super::buffers::GpuMeshBuffers;
use super::decode::{read_color_float4, read_uv, read_vec3, read_vec4_f32};
use super::skinned::build_skinned_vertices;
use super::types::{
    BlendshapeOffset, VertexPosNormal, VertexPosNormalUv, VertexUiCanvas, VertexWithUv,
};

/// Creates GPU buffers for a mesh. Extracts position and smooth normal for normal debug shader.
///
/// When `ray_tracing_available` is true, the index buffer is created with
/// [`wgpu::BufferUsages::BLAS_INPUT`] so it can be used for BLAS builds.
pub fn create_mesh_buffers(
    device: &wgpu::Device,
    mesh: &MeshAsset,
    vertex_stride: usize,
    ray_tracing_available: bool,
) -> Option<GpuMeshBuffers> {
    if mesh.vertex_data.len() < 12 {
        return None;
    }
    if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
        return None;
    }
    let vc = mesh.vertex_count as usize;
    if vertex_stride == 0 {
        return None;
    }
    let required_vb = vertex_stride * vc;
    if required_vb > mesh.vertex_data.len() {
        return None;
    }
    let pos_info =
        assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position)
            .unwrap_or((0, 12));
    let normal_info =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::normal);
    let uv_info =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::uv0);
    let color_info =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::color);
    let tangent_info =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::tangent);

    let (pos_off, _) = pos_info;
    let (normal_off, normal_size, normal_format) =
        normal_info.unwrap_or((0, 0, VertexAttributeFormat::float32));
    let has_uvs = uv_info.map(|(_, s, _)| s >= 4).unwrap_or(false);

    let default_normal = [0.0f32, 1.0, 0.0];
    let default_uv = [0.0f32, 0.0];
    let (uv_off, uv_size, uv_format) = uv_info.unwrap_or((0, 0, VertexAttributeFormat::float32));

    let mut vertices = Vec::with_capacity(mesh.vertex_count as usize);
    let mut vertices_uv: Option<Vec<VertexWithUv>> = if has_uvs {
        Some(Vec::with_capacity(mesh.vertex_count as usize))
    } else {
        None
    };
    // Native UI canvas buffers when UV0 exists; vertex color defaults to white if absent.
    let build_ui_vertices = has_uvs;
    let mut vertices_ui: Option<Vec<VertexUiCanvas>> = if build_ui_vertices {
        Some(Vec::with_capacity(mesh.vertex_count as usize))
    } else {
        None
    };
    let mut vertices_pos_normal_uv: Option<Vec<VertexPosNormalUv>> = if has_uvs {
        Some(Vec::with_capacity(mesh.vertex_count as usize))
    } else {
        None
    };

    for i in 0..mesh.vertex_count as usize {
        let base = i * vertex_stride;
        if base + pos_off + 12 > mesh.vertex_data.len() {
            continue;
        }
        let px = f32::from_le_bytes(
            mesh.vertex_data[base + pos_off..base + pos_off + 4]
                .try_into()
                .ok()?,
        );
        let py = f32::from_le_bytes(
            mesh.vertex_data[base + pos_off + 4..base + pos_off + 8]
                .try_into()
                .ok()?,
        );
        let pz = f32::from_le_bytes(
            mesh.vertex_data[base + pos_off + 8..base + pos_off + 12]
                .try_into()
                .ok()?,
        );

        let mut normal = if normal_size > 0 {
            read_vec3(&mesh.vertex_data, base, normal_off, normal_format).unwrap_or(default_normal)
        } else {
            default_normal
        };
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 1e-6 {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }

        vertices.push(VertexPosNormal {
            position: [px, py, pz],
            normal,
        });

        if let Some(ref mut v_uv) = vertices_uv {
            let uv = if uv_size > 0 {
                read_uv(&mesh.vertex_data, base, uv_off, uv_format).unwrap_or(default_uv)
            } else {
                default_uv
            };
            v_uv.push(VertexWithUv {
                position: [px, py, pz],
                uv,
            });
        }

        if let Some(ref mut v_pnu) = vertices_pos_normal_uv {
            let uv = if uv_size > 0 {
                read_uv(&mesh.vertex_data, base, uv_off, uv_format).unwrap_or(default_uv)
            } else {
                default_uv
            };
            v_pnu.push(VertexPosNormalUv {
                position: [px, py, pz],
                normal,
                uv,
            });
        }

        if let Some(v_ui) = &mut vertices_ui {
            let color = if let Some((c_off, c_size, c_fmt)) = color_info {
                if c_size > 0 {
                    read_color_float4(&mesh.vertex_data, base, c_off, c_fmt)
                        .unwrap_or([1.0, 1.0, 1.0, 1.0])
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                }
            } else {
                [1.0, 1.0, 1.0, 1.0]
            };
            let uv = if uv_size > 0 {
                read_uv(&mesh.vertex_data, base, uv_off, uv_format).unwrap_or(default_uv)
            } else {
                default_uv
            };
            let aux = if let Some((t_off, t_size, t_fmt)) = tangent_info {
                if t_size >= 16 {
                    read_vec4_f32(&mesh.vertex_data, base, t_off, t_fmt)
                        .unwrap_or([0.0, 0.0, 0.0, 1.0])
                } else if t_size >= 12 {
                    let t =
                        read_vec3(&mesh.vertex_data, base, t_off, t_fmt).unwrap_or(default_normal);
                    [t[0], t[1], t[2], 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            } else if normal_size > 0 {
                let n = read_vec3(&mesh.vertex_data, base, normal_off, normal_format)
                    .unwrap_or(default_normal);
                [n[0], n[1], n[2], 0.0]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            };
            v_ui.push(VertexUiCanvas {
                position: [px, py, pz],
                uv,
                color,
                aux,
            });
        }
    }

    if vertices.len() < 3 {
        return None;
    }

    let vertex_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex buffer (pos+normal)"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }),
    );

    let vertex_buffer_uv = vertices_uv.map(|v_uv| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertex buffer (pos+uv)"),
                contents: bytemuck::cast_slice(&v_uv),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        )
    });

    let vertex_buffer_pos_normal_uv = vertices_pos_normal_uv.map(|v| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertex buffer (pos+normal+uv)"),
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        )
    });

    let vertex_buffer_ui = vertices_ui.and_then(|v_ui| {
        if v_ui.len() < 3 {
            None
        } else {
            Some(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("mesh vertex buffer (ui canvas)"),
                    contents: bytemuck::cast_slice(&v_ui),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            )))
        }
    });

    let (index_data, index_format, index_count) = match mesh.index_format {
        crate::shared::IndexBufferFormat::u_int16 => {
            let count = mesh.index_data.len() / 2;
            if count == 0 {
                return None;
            }
            (
                mesh.index_data.clone(),
                wgpu::IndexFormat::Uint16,
                count as u32,
            )
        }
        crate::shared::IndexBufferFormat::u_int32 => {
            let count = mesh.index_data.len() / 4;
            if count == 0 {
                return None;
            }
            (
                mesh.index_data.clone(),
                wgpu::IndexFormat::Uint32,
                count as u32,
            )
        }
    };

    let index_usage = if ray_tracing_available {
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT
    } else {
        wgpu::BufferUsages::INDEX
    };
    let index_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh index buffer"),
            contents: &index_data,
            usage: index_usage,
        }),
    );

    let submeshes: Vec<(u32, u32)> = if mesh.submeshes.is_empty() {
        vec![(0, index_count)]
    } else {
        let s: Vec<(u32, u32)> = mesh
            .submeshes
            .iter()
            .map(|s| (s.index_start as u32, s.index_count as u32))
            .filter(|(start, count)| *count > 0 && start.saturating_add(*count) <= index_count)
            .collect();
        if s.is_empty() {
            vec![(0, index_count)]
        } else {
            s
        }
    };

    let vertex_buffer_skinned = {
        let has_bind_poses = mesh.bind_poses.as_ref().is_some_and(|v| !v.is_empty());
        let has_bone_counts = mesh.bone_counts.as_ref().is_some_and(|v| !v.is_empty());
        let has_bone_weights = mesh.bone_weights.as_ref().is_some_and(|v| !v.is_empty());
        let bone_counts_match = mesh
            .bone_counts
            .as_ref()
            .map(|c| c.len())
            .is_some_and(|len| len == vc);
        if has_bind_poses && has_bone_counts && has_bone_weights && bone_counts_match {
            build_skinned_vertices(device, mesh, vertex_stride, &vertices).map(Arc::new)
        } else {
            None
        }
    };

    let (blendshape_buffer, num_blendshapes) = {
        let num = mesh.num_blendshapes.max(0) as u32;
        let expected_len = num as usize * vc * std::mem::size_of::<BlendshapeOffset>();
        if let Some(ref data) = mesh.blendshape_offsets {
            if num > 0 && data.len() >= expected_len {
                let buffer = Arc::new(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("mesh blendshape buffer"),
                        contents: &data[..expected_len],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                (Some(buffer), num)
            } else if vertex_buffer_skinned.is_some() {
                let dummy = [0u8; std::mem::size_of::<BlendshapeOffset>()];
                let buffer = Arc::new(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("mesh blendshape buffer (dummy)"),
                        contents: &dummy,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                (Some(buffer), 0)
            } else {
                (None, 0)
            }
        } else if vertex_buffer_skinned.is_some() {
            let dummy = [0u8; std::mem::size_of::<BlendshapeOffset>()];
            let buffer = Arc::new(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh blendshape buffer (dummy)"),
                    contents: &dummy,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            );
            (Some(buffer), 0)
        } else {
            (None, 0)
        }
    };

    Some(GpuMeshBuffers {
        vertex_buffer,
        vertex_buffer_pos_normal_uv,
        vertex_buffer_uv,
        vertex_buffer_ui,
        vertex_buffer_skinned,
        index_buffer,
        submeshes,
        index_format,
        has_uvs,
        blendshape_buffer,
        num_blendshapes,
    })
}

/// Computes vertex stride from mesh data when attribute layout is unknown.
pub fn compute_vertex_stride_from_mesh(mesh: &MeshAsset) -> usize {
    mesh.vertex_data.len() / mesh.vertex_count.max(1) as usize
}
