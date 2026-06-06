//! Process-wide pipeline cache for [`super::ClusteredLightPass`].
//!
//! Mirrors the `LazyLock<PipelineCache>` shape used by the post-processing passes
//! (e.g. [`crate::passes::post_processing::AcesTonemapPass`]'s `AcesTonemapPipelineCache`),
//! specialized for compute dispatch. The bind-group layout and the compute pipeline are each
//! held in an [`OnceGpu`] slot so the first call lazily creates them on the active device,
//! and every subsequent call returns the cached references without locking.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::{CLUSTER_PARAMS_UNIFORM_SIZE, GpuLight};
use crate::gpu_resource::OnceGpu;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::world_mesh::cluster::{CLUSTER_COUNT_Z, TILE_SIZE, sanitize_cluster_clip_planes};

pub(super) const FROXEL_RECONSTRUCTION_GENERAL: u32 = 1 << 0;
const PROJECTION_XY_DEPTH_DEPENDENCE_EPSILON: f32 = 1e-6;

/// CPU layout for the compute shader `ClusterParams` uniform (WGSL `struct` + tail pad).
///
/// `world_to_view_scale` carries the world-to-view linear-scale factor so the shader can convert
/// `light.range` (world units) to view-space units before the cluster sphere/AABB test -- see
/// [`crate::world_mesh::cluster::ClusterFrameParams::world_to_view_scale_max`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ClusterParams {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    pub(super) near_clip: f32,
    pub(super) far_clip: f32,
    cluster_offset: u32,
    world_to_view_scale: f32,
    pub(super) froxel_reconstruction_flags: u32,
}

/// Descriptor for building the `ClusterParams` uniform from scene matrices and cluster grid metadata.
pub(super) struct ClusterParamsDesc {
    pub scene_view: glam::Mat4,
    pub proj: glam::Mat4,
    pub viewport: (u32, u32),
    pub cluster_count_x: u32,
    pub cluster_count_y: u32,
    pub light_count: u32,
    pub near: f32,
    pub far: f32,
    pub cluster_offset: u32,
    /// Max row length of the world-to-view linear part; multiplies `light.range` (world units)
    /// to view-space units inside the compute shader's culling test.
    pub world_to_view_scale: f32,
    /// Compute-side reconstruction flags derived from the final projection matrix.
    pub froxel_reconstruction_flags: u32,
}

/// Builds a [`ClusterParams`] from `desc`, applying the shared clip-plane sanitisation.
pub(super) fn build_params(desc: ClusterParamsDesc) -> ClusterParams {
    let inv_proj = desc.proj.inverse();
    let (near_clip, far_clip) = sanitize_cluster_clip_planes(desc.near, desc.far);
    ClusterParams {
        view: desc.scene_view.to_cols_array_2d(),
        proj: desc.proj.to_cols_array_2d(),
        inv_proj: inv_proj.to_cols_array_2d(),
        viewport_width: desc.viewport.0 as f32,
        viewport_height: desc.viewport.1 as f32,
        tile_size: TILE_SIZE,
        light_count: desc.light_count,
        cluster_count_x: desc.cluster_count_x,
        cluster_count_y: desc.cluster_count_y,
        cluster_count_z: CLUSTER_COUNT_Z,
        near_clip,
        far_clip,
        cluster_offset: desc.cluster_offset,
        world_to_view_scale: desc.world_to_view_scale,
        froxel_reconstruction_flags: desc.froxel_reconstruction_flags,
    }
}

/// Returns clustered-light froxel reconstruction flags for `proj`.
pub(super) fn froxel_reconstruction_flags_for_projection(proj: glam::Mat4) -> u32 {
    if projection_depth_depends_on_view_xy(proj) {
        FROXEL_RECONSTRUCTION_GENERAL
    } else {
        0
    }
}

fn projection_depth_depends_on_view_xy(proj: glam::Mat4) -> bool {
    [proj.x_axis.z, proj.y_axis.z, proj.x_axis.w, proj.y_axis.w]
        .into_iter()
        .any(|value| value.abs() > PROJECTION_XY_DEPTH_DEPENDENCE_EPSILON)
}

/// Writes one `ClusterParams` slot into the per-eye uniform buffer with std140 padding.
pub(super) fn write_cluster_params_padded(
    uploads: GraphUploadSink<'_>,
    buf: &wgpu::Buffer,
    params: &ClusterParams,
    buf_offset: u64,
) {
    let mut padded = [0u8; CLUSTER_PARAMS_UNIFORM_SIZE as usize];
    let src = bytemuck::bytes_of(params);
    padded[..src.len()].copy_from_slice(src);
    uploads.write_buffer(buf, buf_offset, &padded);
}

/// Process-wide cached compute pipeline + bind-group layout for the clustered-light dispatch.
#[derive(Default)]
pub(super) struct ClusteredLightPipelineCache {
    /// Cached bind group layout (params dyn-uniform, lights, ranges, indices).
    bgl: OnceGpu<wgpu::BindGroupLayout>,
    /// Cached compute pipeline created against the layout.
    pipeline: OnceGpu<wgpu::ComputePipeline>,
}

impl ClusteredLightPipelineCache {
    /// Returns the cached bind-group layout, creating it on first use.
    pub(super) fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bgl.get_or_create(|| compute_bind_group_layout(device))
    }

    /// Returns the cached compute pipeline, creating it on first use.
    ///
    /// The bind-group layout is fetched (or created) inside the pipeline's init closure so the
    /// pair stays consistent without taking a lock; both slots are independent
    /// [`std::sync::OnceLock`]s under the hood.
    pub(super) fn pipeline(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.pipeline.get_or_create(|| {
            let bgl = self.bind_group_layout(device);
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("clustered_light_pipeline_layout"),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("clustered_light"),
                source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("clustered_light").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("clustered_light"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            crate::profiling::note_resource_churn!(
                ComputePipeline,
                "passes::clustered_light_pipeline"
            );
            pipeline
        })
    }
}

/// Builds the compute bind-group layout for the clustered-light dispatch.
fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clustered_light_compute"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(CLUSTER_PARAMS_UNIFORM_SIZE),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<GpuLight>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(8),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
            },
        ],
    })
}

/// Returns the process-wide pipeline cache for [`super::ClusteredLightPass`].
pub(super) fn clustered_light_pipelines() -> &'static ClusteredLightPipelineCache {
    static CACHE: std::sync::LazyLock<ClusteredLightPipelineCache> =
        std::sync::LazyLock::new(ClusteredLightPipelineCache::default);
    &CACHE
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use super::{
        CLUSTER_PARAMS_UNIFORM_SIZE, ClusterParams, ClusterParamsDesc,
        FROXEL_RECONSTRUCTION_GENERAL, build_params, froxel_reconstruction_flags_for_projection,
    };
    use crate::camera::{Viewport, reverse_z_perspective};
    use crate::graph_inputs::OffscreenWriteTarget;

    /// `ClusterParams` must fit within the dynamic-offset slot reserved by
    /// `CLUSTER_PARAMS_UNIFORM_SIZE`; `write_cluster_params_padded` zero-pads the rest.
    #[test]
    fn cluster_params_struct_fits_uniform_slot() {
        assert!(
            size_of::<ClusterParams>() as u64 <= CLUSTER_PARAMS_UNIFORM_SIZE,
            "ClusterParams ({} bytes) exceeds CLUSTER_PARAMS_UNIFORM_SIZE ({} bytes)",
            size_of::<ClusterParams>(),
            CLUSTER_PARAMS_UNIFORM_SIZE,
        );
        assert_eq!(
            size_of::<ClusterParams>() % 16,
            0,
            "ClusterParams must be 16-byte aligned for WGSL std140 uniform layout"
        );
    }

    #[test]
    fn normal_perspective_uses_fast_froxel_reconstruction() {
        let proj = reverse_z_perspective(
            Viewport::new(1280, 720).aspect(),
            60f32.to_radians(),
            0.05,
            500.0,
        );

        assert_eq!(froxel_reconstruction_flags_for_projection(proj), 0);
    }

    #[test]
    fn offscreen_y_flipped_projection_uses_fast_froxel_reconstruction() {
        let proj = reverse_z_perspective(
            Viewport::new(1280, 720).aspect(),
            60f32.to_radians(),
            0.05,
            500.0,
        );
        let offscreen_proj = OffscreenWriteTarget::Untracked.render_projection(proj);

        assert_eq!(
            froxel_reconstruction_flags_for_projection(offscreen_proj),
            0
        );
    }

    #[test]
    fn asymmetric_projection_uses_fast_froxel_reconstruction() {
        let mut proj = reverse_z_perspective(
            Viewport::new(1280, 720).aspect(),
            60f32.to_radians(),
            0.05,
            500.0,
        );
        proj.z_axis.x = 0.25;
        proj.z_axis.y = -0.125;

        assert_eq!(froxel_reconstruction_flags_for_projection(proj), 0);
    }

    #[test]
    fn oblique_projection_uses_general_froxel_reconstruction() {
        let mut proj = reverse_z_perspective(
            Viewport::new(1280, 720).aspect(),
            60f32.to_radians(),
            0.05,
            500.0,
        );
        proj.x_axis.z = 0.25;

        assert_eq!(
            froxel_reconstruction_flags_for_projection(proj),
            FROXEL_RECONSTRUCTION_GENERAL
        );
    }

    #[test]
    fn cluster_params_pack_froxel_reconstruction_flags() {
        let mut proj = Mat4::IDENTITY;
        proj.y_axis.z = -0.25;
        let flags = froxel_reconstruction_flags_for_projection(proj);

        let params = build_params(ClusterParamsDesc {
            scene_view: Mat4::IDENTITY,
            proj,
            viewport: (1, 1),
            cluster_count_x: 1,
            cluster_count_y: 1,
            light_count: 0,
            near: 0.05,
            far: 10.0,
            cluster_offset: 0,
            world_to_view_scale: 1.0,
            froxel_reconstruction_flags: flags,
        });

        assert_eq!(params.froxel_reconstruction_flags, flags);
    }
}
