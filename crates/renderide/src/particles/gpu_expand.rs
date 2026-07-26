//! GPU point-particle billboard expansion into generated mesh streams.

use std::num::NonZeroU64;
use std::sync::OnceLock;

use glam::{IVec2, Vec3};
use wgpu::util::DeviceExt;

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu_resource::OnceGpu;

use super::types::PointParticle;

/// Compute workgroup width.
const EXPAND_WORKGROUP_SIZE: u32 = 64;

/// Shared lazy compute pipeline.
fn expand_pipeline() -> &'static ParticleBillboardExpandPipeline {
    static PIPELINE: OnceLock<ParticleBillboardExpandPipeline> = OnceLock::new();
    PIPELINE.get_or_init(ParticleBillboardExpandPipeline::default)
}

/// Returns the writable streams when all point-mesh buffers are allocated.
pub(crate) fn point_mesh_targets(mesh: &crate::assets::mesh::GpuMesh) -> Option<PointMeshTargets<'_>> {
    Some(PointMeshTargets {
        interleaved: mesh.vertex_buffer.as_deref()?,
        positions: mesh.positions_buffer.as_deref()?,
        normals: mesh.normals_buffer.as_deref()?,
        uv0: mesh.uv0_buffer.as_deref()?,
        color: mesh.color_buffer.as_deref()?,
        tangent: mesh.tangent_buffer.as_deref()?,
        raw_tangent: mesh.raw_tangent_buffer.as_deref()?,
        uv1: mesh.uv1_buffer.as_deref()?,
        indices: mesh.index_buffer.as_deref()?,
    })
}

/// Generated point-mesh stream buffers.
pub(crate) struct PointMeshTargets<'a> {
    pub(crate) interleaved: &'a wgpu::Buffer,
    pub(crate) positions: &'a wgpu::Buffer,
    pub(crate) normals: &'a wgpu::Buffer,
    pub(crate) uv0: &'a wgpu::Buffer,
    pub(crate) color: &'a wgpu::Buffer,
    pub(crate) tangent: &'a wgpu::Buffer,
    pub(crate) raw_tangent: &'a wgpu::Buffer,
    pub(crate) uv1: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
}

/// Uploads point instances and dispatches billboard expansion.
///
/// Capacity beyond `points.len()` is filled with degenerate geometry.
pub(crate) fn expand_point_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    targets: PointMeshTargets<'_>,
    points: &[PointParticle],
    capacity: u32,
    frame_grid_size: IVec2,
) {
    if capacity == 0 {
        return;
    }
    let instances = points
        .iter()
        .map(PointInstance::from_particle)
        .collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("particle_billboard_instances"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = ExpandParams {
        live_count: instances.len() as u32,
        capacity,
        frame_cols: frame_grid_size.x.max(0) as u32,
        frame_rows: frame_grid_size.y.max(0) as u32,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("particle_billboard_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particle_billboard_expand"),
    });
    expand_pipeline().encode(
        device,
        &mut encoder,
        ParticleExpandBinding {
            params: &params_buffer,
            instances: &instance_buffer,
            interleaved: targets.interleaved,
            positions: targets.positions,
            normals: targets.normals,
            uv0: targets.uv0,
            color: targets.color,
            tangent: targets.tangent,
            raw_tangent: targets.raw_tangent,
            uv1: targets.uv1,
            indices: targets.indices,
        },
        capacity,
    );
    queue.submit(std::iter::once(encoder.finish()));
}

/// One packed point-particle instance consumed by the expansion shader. Layout matches the WGSL
/// `PointInstance` struct exactly (five `vec4<f32>`, 80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointInstance {
    /// xyz = center position, w = roll (`glam` `Quat::to_euler(XYZ).2`).
    pos_roll: [f32; 4],
    /// Linear color, already photondust-linearized on decode.
    color: [f32; 4],
    /// x,y = size.xy, z = frame index as `f32`, w = has-frame flag (1.0/0.0).
    size_frame: [f32; 4],
    /// xyz = rotation * +Z (forward), w unused.
    forward: [f32; 4],
    /// xyz = rotation * +Y (up), w unused.
    up: [f32; 4],
}

impl PointInstance {
    /// Packs one point and precomputes its orientation vectors.
    pub(crate) fn from_particle(point: &PointParticle) -> Self {
        let (_, _, roll) = point.rotation.to_euler(glam::EulerRot::XYZ);
        let forward = point.rotation * Vec3::Z;
        let up = point.rotation * Vec3::Y;
        let (frame_index, has_frame) = match point.frame_index {
            Some(frame) => (f32::from(frame), 1.0),
            None => (0.0, 0.0),
        };
        Self {
            pos_roll: [point.position.x, point.position.y, point.position.z, roll],
            color: point.color.to_array(),
            size_frame: [point.size.x, point.size.y, frame_index, has_frame],
            forward: [forward.x, forward.y, forward.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
        }
    }
}

/// Uniform parameters for one expansion dispatch. Matches the WGSL `Params` struct (four `u32`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ExpandParams {
    /// Number of live particles at the front of the instance buffer.
    pub(crate) live_count: u32,
    /// Total slots (>= live_count) that receive vertices; trailing slots are degenerate.
    pub(crate) capacity: u32,
    /// Sprite-sheet columns (0 disables frame indexing).
    pub(crate) frame_cols: u32,
    /// Sprite-sheet rows (0 disables frame indexing).
    pub(crate) frame_rows: u32,
}

/// Buffers bound for one expansion dispatch.
pub(crate) struct ParticleExpandBinding<'a> {
    pub(crate) params: &'a wgpu::Buffer,
    pub(crate) instances: &'a wgpu::Buffer,
    pub(crate) interleaved: &'a wgpu::Buffer,
    pub(crate) positions: &'a wgpu::Buffer,
    pub(crate) normals: &'a wgpu::Buffer,
    pub(crate) uv0: &'a wgpu::Buffer,
    pub(crate) color: &'a wgpu::Buffer,
    pub(crate) tangent: &'a wgpu::Buffer,
    pub(crate) raw_tangent: &'a wgpu::Buffer,
    pub(crate) uv1: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
}

/// Lazily created compute pipeline for point-particle billboard expansion.
#[derive(Default)]
pub(crate) struct ParticleBillboardExpandPipeline {
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipeline: OnceGpu<wgpu::ComputePipeline>,
}

impl ParticleBillboardExpandPipeline {
    /// Records one expansion dispatch.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        binding: ParticleExpandBinding<'_>,
        capacity: u32,
    ) {
        if capacity == 0 {
            return;
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_billboard_expand"),
            layout: self.bind_group_layout(device),
            entries: &[
                entry(0, binding.params),
                entry(1, binding.instances),
                entry(2, binding.interleaved),
                entry(3, binding.positions),
                entry(4, binding.normals),
                entry(5, binding.uv0),
                entry(6, binding.color),
                entry(7, binding.tangent),
                entry(8, binding.raw_tangent),
                entry(9, binding.uv1),
                entry(10, binding.indices),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_billboard_expand"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.pipeline(device));
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = capacity.div_ceil(EXPAND_WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle_billboard_expand"),
                entries: &[
                    uniform_layout(0, size_of::<ExpandParams>() as u64),
                    storage_layout(1, true, size_of::<PointInstance>() as u64),
                    storage_layout(2, false, 4),
                    storage_layout(3, false, 16),
                    storage_layout(4, false, 16),
                    storage_layout(5, false, 8),
                    storage_layout(6, false, 16),
                    storage_layout(7, false, 16),
                    storage_layout(8, false, 16),
                    storage_layout(9, false, 8),
                    storage_layout(10, false, 4),
                ],
            })
        })
    }

    fn pipeline(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.pipeline.get_or_create(|| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle_billboard_expand"),
                bind_group_layouts: &[Some(self.bind_group_layout(device))],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particle_billboard_expand"),
                source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("particle_billboard_expand").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particle_billboard_expand"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            crate::profiling::note_resource_churn!(
                ComputePipeline,
                "particles::billboard_expand_pipeline"
            );
            pipeline
        })
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_layout(binding: u32, minimum_size: u64) -> wgpu::BindGroupLayoutEntry {
    buffer_layout(binding, wgpu::BufferBindingType::Uniform, minimum_size)
}

fn storage_layout(binding: u32, read_only: bool, minimum_size: u64) -> wgpu::BindGroupLayoutEntry {
    buffer_layout(
        binding,
        wgpu::BufferBindingType::Storage { read_only },
        minimum_size,
    )
}

fn buffer_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    minimum_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(minimum_size),
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_instance_layout_matches_shader() {
        // Five vec4<f32> => 80 bytes, 16-byte aligned array stride in the shader.
        assert_eq!(size_of::<PointInstance>(), 80);
        assert_eq!(size_of::<ExpandParams>(), 16);
        assert_eq!(size_of::<PointInstance>() % 16, 0);
    }

    #[test]
    fn packed_instance_precomputes_roll_forward_up() {
        let point = PointParticle {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: glam::Quat::IDENTITY,
            size: Vec3::new(0.5, 0.25, 0.0),
            color: glam::Vec4::new(0.1, 0.2, 0.3, 0.4),
            frame_index: Some(7),
        };
        let instance = PointInstance::from_particle(&point);
        assert_eq!(instance.pos_roll, [1.0, 2.0, 3.0, 0.0]);
        // Identity rotation: forward = +Z, up = +Y.
        assert!((instance.forward[2] - 1.0).abs() < 1e-6);
        assert!((instance.up[1] - 1.0).abs() < 1e-6);
        assert_eq!(instance.size_frame[0], 0.5);
        assert_eq!(instance.size_frame[1], 0.25);
        assert_eq!(instance.size_frame[2], 7.0);
        assert_eq!(instance.size_frame[3], 1.0);
    }
}
