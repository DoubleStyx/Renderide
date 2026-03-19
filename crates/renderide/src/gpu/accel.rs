//! Bottom Level Acceleration Structure (BLAS) cache and Top Level Acceleration Structure (TLAS)
//! for ray tracing.
//!
//! Builds and caches BLASes for non-skinned meshes when EXPERIMENTAL_RAY_QUERY is available.
//! TLAS is rebuilt each frame from non-overlay, non-skinned draw instances.
//! Used for future RTAO (Ray-Traced Ambient Occlusion) support.

use std::collections::HashMap;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::assets::{self, MeshAsset};
use crate::render::batch::SpaceDrawBatch;
use crate::shared::{VertexAttributeFormat, VertexAttributeType};

use super::mesh::GpuMeshBuffers;

/// Cache of BLASes keyed by mesh asset ID.
///
/// Only populated when [`GpuState::ray_tracing_available`] is true.
/// Kept in sync with [`GpuState::mesh_buffer_cache`] lifecycle.
pub struct AccelCache {
    /// BLAS per mesh asset ID. Non-skinned meshes only.
    blas_map: HashMap<i32, wgpu::Blas>,
}

impl AccelCache {
    /// Creates an empty acceleration structure cache.
    pub fn new() -> Self {
        Self {
            blas_map: HashMap::new(),
        }
    }

    /// Returns a reference to the BLAS for the given mesh asset ID, if present.
    pub fn get(&self, mesh_asset_id: i32) -> Option<&wgpu::Blas> {
        self.blas_map.get(&mesh_asset_id)
    }

    /// Inserts a BLAS for the given mesh asset ID.
    pub fn insert(&mut self, mesh_asset_id: i32, blas: wgpu::Blas) {
        self.blas_map.insert(mesh_asset_id, blas);
    }

    /// Removes the BLAS for the given mesh asset ID.
    pub fn remove(&mut self, mesh_asset_id: i32) -> Option<wgpu::Blas> {
        self.blas_map.remove(&mesh_asset_id)
    }
}

impl Default for AccelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads a vec3 position from vertex data at the given base offset.
fn read_position(
    data: &[u8],
    base: usize,
    offset: usize,
    format: VertexAttributeFormat,
) -> Option<[f32; 3]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 12 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                    f32::from_le_bytes(
                        data[base + offset + 8..base + offset + 12]
                            .try_into()
                            .ok()?,
                    ),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 6 <= data.len() {
                let half_to_f32 = |h: u16| {
                    let sign = (h >> 15) as u32;
                    let exp = ((h >> 10) & 0x1F) as u32;
                    let mant = (h & 0x3FF) as u32;
                    if exp == 0 {
                        let f = (sign << 31) | (mant << 13);
                        f32::from_bits(f) * 5.960_464_5e-8
                    } else if exp == 31 {
                        let f = (sign << 31) | 0x7F800000 | (mant << 13);
                        f32::from_bits(f)
                    } else {
                        let f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                        f32::from_bits(f)
                    }
                };
                Some([
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset..base + offset + 2].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 2..base + offset + 4].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 4..base + offset + 6].try_into().ok()?,
                    )),
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Builds a BLAS for a non-skinned mesh.
///
/// Caller must ensure the device has [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`] enabled.
/// Returns `None` if the mesh has no valid vertex/index data, is skinned, or BLAS creation fails.
///
/// Uses a position-only buffer (VertexFormat::Float32x3) as required by the BLAS API.
/// Geometry flags are set to [`wgpu::AccelerationStructureGeometryFlags::OPAQUE`].
pub fn build_blas_for_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &MeshAsset,
    gpu_buffers: &GpuMeshBuffers,
) -> Option<wgpu::Blas> {
    if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
        return None;
    }
    if mesh.vertex_data.len() < 12 {
        return None;
    }
    if gpu_buffers.vertex_buffer_skinned.is_some() {
        return None;
    }

    let vertex_stride = mesh.vertex_data.len() / mesh.vertex_count.max(1) as usize;
    if vertex_stride == 0 {
        return None;
    }

    let (pos_off, _) =
        assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position)?;
    let (_, _, pos_format) = assets::attribute_offset_size_format(
        &mesh.vertex_attributes,
        VertexAttributeType::position,
    )?;

    let mut positions = Vec::with_capacity(mesh.vertex_count as usize);
    for i in 0..mesh.vertex_count as usize {
        let base = i * vertex_stride;
        let pos = read_position(&mesh.vertex_data, base, pos_off, pos_format)?;
        positions.push(pos);
    }
    if positions.len() < 3 {
        return None;
    }

    let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BLAS position buffer"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let vertex_count = positions.len() as u32;
    let index_count = match mesh.index_format {
        crate::shared::IndexBufferFormat::u_int16 => mesh.index_data.len() / 2,
        crate::shared::IndexBufferFormat::u_int32 => mesh.index_data.len() / 4,
    } as u32;
    if index_count == 0 {
        return None;
    }

    let index_format = match mesh.index_format {
        crate::shared::IndexBufferFormat::u_int16 => wgpu::IndexFormat::Uint16,
        crate::shared::IndexBufferFormat::u_int32 => wgpu::IndexFormat::Uint32,
    };

    let size_descriptor = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count,
        index_format: Some(index_format),
        index_count: Some(index_count),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas_desc = wgpu::CreateBlasDescriptor {
        label: Some("mesh BLAS"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    };

    let sizes = wgpu::BlasGeometrySizeDescriptors::Triangles {
        descriptors: vec![size_descriptor.clone()],
    };

    let blas = device.create_blas(&blas_desc, sizes);

    let geometry = wgpu::BlasTriangleGeometry {
        size: &size_descriptor,
        vertex_buffer: &position_buffer,
        first_vertex: 0,
        vertex_stride: 12,
        index_buffer: Some(gpu_buffers.index_buffer.as_ref()),
        first_index: Some(0),
        transform_buffer: None,
        transform_buffer_offset: None,
    };

    let build_entry = wgpu::BlasBuildEntry {
        blas: &blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![geometry]),
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("BLAS build encoder"),
    });
    encoder.build_acceleration_structures(
        std::iter::once(&build_entry),
        std::iter::empty::<&wgpu::Tlas>(),
    );
    queue.submit(std::iter::once(encoder.finish()));

    Some(blas)
}

/// Removes the BLAS for the given mesh asset ID from the cache.
pub fn remove_blas(cache: &mut AccelCache, mesh_asset_id: i32) {
    cache.remove(mesh_asset_id);
}

/// Holds the current frame's TLAS, rebuilt each frame when ray tracing is available.
///
/// Stored in [`GpuState`] when [`GpuState::ray_tracing_available`]. Used by future RTAO pass.
pub struct RayTracingState {
    /// Current TLAS built from non-overlay, non-skinned draws. `None` when no instances.
    pub tlas: Option<wgpu::Tlas>,
    /// Reusable scratch buffer for TLAS instance data. Avoids per-frame Vec allocation.
    pub(crate) instance_scratch: Vec<(i32, [f32; 12])>,
}

impl RayTracingState {
    /// Creates an empty ray tracing state.
    pub fn new() -> Self {
        Self {
            tlas: None,
            instance_scratch: Vec::new(),
        }
    }
}

impl Default for RayTracingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a 4x4 model matrix to 3x4 affine row-major for [`wgpu::TlasInstance`].
fn matrix4_to_affine_3x4(m: &Mat4) -> [f32; 12] {
    let a = m.to_cols_array();
    [
        a[0], a[4], a[8], a[12], a[1], a[5], a[9], a[13], a[2], a[6], a[10], a[14],
    ]
}

/// Builds a TLAS from non-overlay, non-skinned draw instances.
///
/// Iterates `draw_batches`; for each non-overlay draw that is non-skinned and has a BLAS in
/// `accel_cache`, adds an instance with the draw's model matrix and BLAS reference.
/// Records the build into `encoder`. Caller must submit the encoder.
///
/// Uses `instance_scratch` to avoid per-frame Vec allocation. Caller should pass a reusable
/// buffer (e.g. [`RayTracingState::instance_scratch`]).
///
/// Returns `Some(Tlas)` when at least one instance was added, `None` otherwise.
/// Caller must ensure the device has [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`] enabled.
pub fn build_tlas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    accel_cache: &AccelCache,
    draw_batches: &[SpaceDrawBatch],
    instance_scratch: &mut Vec<(i32, [f32; 12])>,
) -> Option<wgpu::Tlas> {
    instance_scratch.clear();

    for batch in draw_batches {
        if batch.is_overlay {
            continue;
        }
        for d in &batch.draws {
            if d.is_skinned || d.mesh_asset_id < 0 {
                continue;
            }
            if accel_cache.get(d.mesh_asset_id).is_none() {
                continue;
            }
            let transform = matrix4_to_affine_3x4(&d.model_matrix);
            instance_scratch.push((d.mesh_asset_id, transform));
        }
    }

    if instance_scratch.is_empty() {
        return None;
    }

    let max_instances = instance_scratch.len() as u32;
    let tlas_desc = wgpu::CreateTlasDescriptor {
        label: Some("scene TLAS"),
        max_instances,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    };

    let mut tlas = device.create_tlas(&tlas_desc);

    for (i, (mesh_asset_id, transform)) in instance_scratch.iter().enumerate() {
        let Some(blas) = accel_cache.get(*mesh_asset_id) else {
            logger::warn!(
                "BLAS missing for mesh_asset_id={}, skipping TLAS instance",
                mesh_asset_id
            );
            continue;
        };
        let instance = wgpu::TlasInstance::new(blas, *transform, 0, 0xFF);
        tlas[i] = Some(instance);
    }

    encoder.build_acceleration_structures(
        std::iter::empty::<&wgpu::BlasBuildEntry>(),
        std::iter::once(&tlas),
    );

    Some(tlas)
}
