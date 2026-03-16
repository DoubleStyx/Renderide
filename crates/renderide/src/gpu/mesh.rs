//! wgpu mesh rendering with debug texture.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::{self, MeshAsset};
use crate::shared::{VertexAttributeFormat, VertexAttributeType};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
}

/// Vertex with position and UV for UV debug pipeline.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexWithUv {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Position + smooth normal for normal debug shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPosNormal {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Skinned vertex: position, normal, bone indices (4), bone weights (4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexSkinned {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub bone_indices: [i32; 4],
    pub bone_weights: [f32; 4],
}

/// Read a vec3 normal from vertex data at base+offset, converting from the given format to f32.
fn read_normal(data: &[u8], base: usize, offset: usize, format: VertexAttributeFormat) -> Option<[f32; 3]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 12 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 8..base + offset + 12].try_into().ok()?),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 6 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?)),
                    half_to_f32(u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?)),
                    half_to_f32(u16::from_le_bytes(data[base + offset + 4..base + offset + 6].try_into().ok()?)),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 3 <= data.len() {
                Some([
                    (data[base + offset] as f32 / 255.0) * 2.0 - 1.0,
                    (data[base + offset + 1] as f32 / 255.0) * 2.0 - 1.0,
                    (data[base + offset + 2] as f32 / 255.0) * 2.0 - 1.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm16 => {
            if base + offset + 6 <= data.len() {
                Some([
                    (u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?) as f32 / 65535.0) * 2.0 - 1.0,
                    (u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?) as f32 / 65535.0) * 2.0 - 1.0,
                    (u16::from_le_bytes(data[base + offset + 4..base + offset + 6].try_into().ok()?) as f32 / 65535.0) * 2.0 - 1.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Read a vec2 UV from vertex data at base+offset, converting from the given format to f32.
fn read_uv(data: &[u8], base: usize, offset: usize, format: VertexAttributeFormat) -> Option<[f32; 2]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 8 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 4 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?)),
                    half_to_f32(u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?)),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 2 <= data.len() {
                Some([
                    data[base + offset] as f32 / 255.0,
                    data[base + offset + 1] as f32 / 255.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm16 => {
            if base + offset + 4 <= data.len() {
                Some([
                    u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?) as f32 / 65535.0,
                    u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?) as f32 / 65535.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert IEEE 754 half-precision (f16) to f32.
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let f = (sign << 31) | (mant << 13);
        f32::from_bits(f) * 5.960464477539063e-8
    } else if exp == 31 {
        let f = (sign << 31) | 0x7F800000 | (mant << 13);
        f32::from_bits(f)
    } else {
        let f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f)
    }
}

/// Fallback cube with position+normal for normal debug pipeline (8 vertices, 12 triangles, 36 indices).
fn fallback_cube_pos_normal() -> (Vec<VertexPosNormal>, Vec<u16>) {
    let s = 0.5f32;
    let n = [0.0f32, 1.0, 0.0];
    let vertices = vec![
        VertexPosNormal { position: [-s, -s, -s], normal: n },
        VertexPosNormal { position: [s, -s, -s], normal: n },
        VertexPosNormal { position: [s, s, -s], normal: n },
        VertexPosNormal { position: [-s, s, -s], normal: n },
        VertexPosNormal { position: [-s, -s, s], normal: n },
        VertexPosNormal { position: [s, -s, s], normal: n },
        VertexPosNormal { position: [s, s, s], normal: n },
        VertexPosNormal { position: [-s, s, s], normal: n },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Fallback cube mesh (8 vertices, 12 triangles, 36 indices).
fn fallback_cube() -> (Vec<Vertex>, Vec<u16>) {
    let s = 0.5f32;
    let vertices = vec![
        Vertex { position: [-s, -s, -s] },
        Vertex { position: [s, -s, -s] },
        Vertex { position: [s, s, -s] },
        Vertex { position: [-s, s, -s] },
        Vertex { position: [-s, -s, s] },
        Vertex { position: [s, -s, s] },
        Vertex { position: [s, s, s] },
        Vertex { position: [-s, s, s] },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Fallback cube with UVs for UV debug shader.
fn fallback_cube_with_uv() -> (Vec<VertexWithUv>, Vec<u16>) {
    let s = 0.5f32;
    let vertices = vec![
        VertexWithUv { position: [-s, -s, -s], uv: [0.0, 0.0] },
        VertexWithUv { position: [s, -s, -s], uv: [1.0, 0.0] },
        VertexWithUv { position: [s, s, -s], uv: [1.0, 1.0] },
        VertexWithUv { position: [-s, s, -s], uv: [0.0, 1.0] },
        VertexWithUv { position: [-s, -s, s], uv: [0.0, 0.0] },
        VertexWithUv { position: [s, -s, s], uv: [1.0, 0.0] },
        VertexWithUv { position: [s, s, s], uv: [1.0, 1.0] },
        VertexWithUv { position: [-s, s, s], uv: [0.0, 1.0] },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

// MeshPipeline removed — see gpu::pipeline::PipelineManager and concrete pipelines.

/// Cached wgpu buffers for a mesh asset.
#[derive(Clone)]
pub struct GpuMeshBuffers {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub vertex_buffer_uv: Option<Arc<wgpu::Buffer>>,
    pub vertex_buffer_skinned: Option<Arc<wgpu::Buffer>>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub submeshes: Vec<(u32, u32)>,
    pub index_format: wgpu::IndexFormat,
    pub has_uvs: bool,
}

/// Creates GPU buffers for a mesh. Extracts position and smooth normal for normal debug shader.
pub fn create_mesh_buffers(
    device: &wgpu::Device,
    mesh: &MeshAsset,
    vertex_stride: usize,
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
    let pos_info = assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position)
        .unwrap_or((0, 12));
    let normal_info = assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::normal);
    let uv_info = assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::uv0);

    let (pos_off, _) = pos_info;
    let (normal_off, normal_size, normal_format) = normal_info
        .map(|(o, s, f)| (o, s, f))
        .unwrap_or((0, 0, VertexAttributeFormat::float32));
    let has_uvs = uv_info.map(|(_, s, _)| s >= 4).unwrap_or(false);

    let default_normal = [0.0f32, 1.0, 0.0];
    let default_uv = [0.0f32, 0.0];
    let (uv_off, uv_size, uv_format) = uv_info
        .map(|(o, s, f)| (o, s, f))
        .unwrap_or((0, 0, VertexAttributeFormat::float32));

    let mut vertices = Vec::with_capacity(mesh.vertex_count as usize);
    let mut vertices_uv: Option<Vec<VertexWithUv>> = if has_uvs {
        Some(Vec::with_capacity(mesh.vertex_count as usize))
    } else {
        None
    };

    for i in 0..mesh.vertex_count as usize {
        let base = i * vertex_stride;
        if base + pos_off + 12 > mesh.vertex_data.len() {
            continue;
        }
        let px = f32::from_le_bytes(mesh.vertex_data[base + pos_off..base + pos_off + 4].try_into().ok()?);
        let py = f32::from_le_bytes(mesh.vertex_data[base + pos_off + 4..base + pos_off + 8].try_into().ok()?);
        let pz = f32::from_le_bytes(mesh.vertex_data[base + pos_off + 8..base + pos_off + 12].try_into().ok()?);

        let mut normal = if normal_size > 0 {
            read_normal(&mesh.vertex_data, base, normal_off, normal_format)
                .unwrap_or(default_normal)
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
    }

    if vertices.len() < 3 {
        return None;
    }

    let vertex_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh vertex buffer (pos+normal)"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    }));

    let vertex_buffer_uv = vertices_uv.map(|v_uv| {
        Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex buffer (pos+uv)"),
            contents: bytemuck::cast_slice(&v_uv),
            usage: wgpu::BufferUsages::VERTEX,
        }))
    });

    let (index_data, index_format, index_count) = match mesh.index_format {
        crate::shared::IndexBufferFormat::u_int16 => {
            let count = mesh.index_data.len() / 2;
            if count == 0 {
                return None;
            }
            (mesh.index_data.clone(), wgpu::IndexFormat::Uint16, count as u32)
        }
        crate::shared::IndexBufferFormat::u_int32 => {
            let count = mesh.index_data.len() / 4;
            if count == 0 {
                return None;
            }
            (mesh.index_data.clone(), wgpu::IndexFormat::Uint32, count as u32)
        }
    };

    let index_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh index buffer"),
        contents: &index_data,
        usage: wgpu::BufferUsages::INDEX,
    }));

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

    let vertex_buffer_skinned = if mesh
        .bind_poses
        .as_ref()
        .map_or(false, |v| !v.is_empty())
    {
        build_skinned_vertices(device, mesh, vertex_stride, &vertices).map(Arc::new)
    } else {
        None
    };

    Some(GpuMeshBuffers {
        vertex_buffer,
        vertex_buffer_uv,
        vertex_buffer_skinned,
        index_buffer,
        submeshes,
        index_format,
        has_uvs,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoneWeightPod {
    weight: f32,
    bone_index: i32,
}

fn build_skinned_vertices(
    device: &wgpu::Device,
    mesh: &MeshAsset,
    vertex_stride: usize,
    base_vertices: &[VertexPosNormal],
) -> Option<wgpu::Buffer> {
    let bone_counts = mesh.bone_counts.as_ref()?;
    let bone_weights = mesh.bone_weights.as_ref()?;
    if bone_counts.len() != base_vertices.len() {
        return None;
    }
    let vc = base_vertices.len();
    let mut skinned = Vec::with_capacity(vc);
    let mut weight_offset = 0;
    for (i, v) in base_vertices.iter().enumerate() {
        let n = bone_counts.get(i).copied().unwrap_or(0) as usize;
        let n = n.min(4);
        let mut indices = [0i32; 4];
        let mut weights = [0.0f32; 4];
        for j in 0..n {
            if weight_offset + 8 <= bone_weights.len() {
                let w: BoneWeightPod = bytemuck::pod_read_unaligned(
                    &bone_weights[weight_offset..weight_offset + 8],
                );
                indices[j] = w.bone_index.max(0).min(255);
                weights[j] = w.weight;
                weight_offset += 8;
            }
        }
        skinned.push(VertexSkinned {
            position: v.position,
            normal: v.normal,
            bone_indices: indices,
            bone_weights: weights,
        });
    }
    Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh vertex buffer (skinned)"),
        contents: bytemuck::cast_slice(&skinned),
        usage: wgpu::BufferUsages::VERTEX,
    }))
}

/// Computes vertex stride from mesh data when attribute layout is unknown.
pub fn compute_vertex_stride_from_mesh(mesh: &MeshAsset) -> usize {
    mesh.vertex_data.len() / mesh.vertex_count.max(1) as usize
}
