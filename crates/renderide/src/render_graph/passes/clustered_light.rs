//! Clustered forward lighting: compute pass assigns light indices per view-space cluster.
//!
//! Dispatches over a 3D grid (`16×16` pixel tiles × exponential Z slices). Uses the same
//! [`GpuLight`] buffer and cluster storage as raster `@group(0)` ([`crate::backend::FrameGpuResources`]).
//!
//! WGSL source: `shaders/source/compute/clustered_light.wgsl` (included at compile time).

use std::num::NonZeroU64;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::backend::{GpuLight, MAX_LIGHTS};
use crate::backend::{CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, TILE_SIZE};
use crate::render_graph::cluster_frame::{
    cluster_frame_params, cluster_frame_params_stereo, ClusterFrameParams,
};
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};

/// CPU layout for the compute shader `ClusterParams` uniform (WGSL `struct` + tail pad).
///
/// `cluster_offset` shifts cluster writes into the storage buffer so per-eye data occupies
/// `[0..N)` for eye 0 and `[N..2N)` for eye 1 (zero in mono mode).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterParams {
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
    near_clip: f32,
    far_clip: f32,
    cluster_offset: u32,
    _pad: [u8; 8],
}

const CLUSTERED_LIGHT_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/clustered_light.wgsl"
));

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
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuLight>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
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

fn ensure_compute_pipeline(
    device: &wgpu::Device,
) -> &'static (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    static CACHE: OnceLock<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> = OnceLock::new();
    CACHE.get_or_init(|| {
        let bgl = compute_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clustered_light_pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clustered_light"),
            source: wgpu::ShaderSource::Wgsl(CLUSTERED_LIGHT_SHADER_SRC.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clustered_light"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, bgl)
    })
}

fn write_cluster_params_padded(
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    params: &ClusterParams,
    buf_offset: u64,
) {
    let mut padded = [0u8; CLUSTER_PARAMS_UNIFORM_SIZE as usize];
    let src = bytemuck::bytes_of(params);
    padded[..src.len()].copy_from_slice(src);
    queue.write_buffer(buf, buf_offset, &padded);
}

/// Builds per-cluster light lists before the world forward pass.
#[derive(Debug, Default)]
pub struct ClusteredLightPass {
    logged_active_once: bool,
}

impl ClusteredLightPass {
    /// Creates a clustered light pass (pipeline is created lazily on first execute).
    pub fn new() -> Self {
        Self {
            logged_active_once: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_params(
        scene_view: Mat4,
        proj: Mat4,
        viewport: (u32, u32),
        cluster_count_x: u32,
        cluster_count_y: u32,
        light_count: u32,
        near: f32,
        far: f32,
        cluster_offset: u32,
    ) -> ClusterParams {
        let inv_proj = proj.inverse();
        ClusterParams {
            view: scene_view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_proj: inv_proj.to_cols_array_2d(),
            viewport_width: viewport.0 as f32,
            viewport_height: viewport.1 as f32,
            tile_size: TILE_SIZE,
            light_count,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z: CLUSTER_COUNT_Z,
            near_clip: near.max(0.01),
            far_clip: far,
            cluster_offset,
            _pad: [0; 8],
        }
    }
}

impl RenderPass for ClusteredLightPass {
    fn name(&self) -> &str {
        "ClusteredLight"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: vec![ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        let (vw, vh) = frame.viewport_px;
        if vw == 0 || vh == 0 {
            return Ok(());
        }

        let hc = frame.host_camera;
        let scene = frame.scene;
        let stereo = hc.vr_active && hc.stereo_views.is_some() && frame.multiview_stereo;

        let lights_upload: Vec<GpuLight> = frame.backend.frame_resources.frame_lights().to_vec();
        let Some(fgpu) = frame.backend.frame_resources.frame_gpu_mut() else {
            return Ok(());
        };

        fgpu.sync_cluster_viewport(ctx.device, (vw, vh), stereo);

        {
            let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
            fgpu.write_lights_buffer(&queue, lights_upload.as_slice());
        }

        let Some(refs) = fgpu
            .cluster_cache
            .get_buffers((vw, vh), CLUSTER_COUNT_Z, stereo)
        else {
            logger::trace!("ClusteredLight: cluster buffers missing after sync");
            return Ok(());
        };

        let light_count = lights_upload.len().min(MAX_LIGHTS) as u32;
        let viewport = (vw, vh);

        let eye_params: Vec<ClusterFrameParams> = if stereo {
            if let Some((left, right)) = cluster_frame_params_stereo(&hc, scene, viewport) {
                vec![left, right]
            } else if let Some(mono) = cluster_frame_params(&hc, scene, viewport) {
                vec![mono]
            } else {
                return Ok(());
            }
        } else if let Some(mono) = cluster_frame_params(&hc, scene, viewport) {
            vec![mono]
        } else {
            return Ok(());
        };

        let clusters_per_eye = eye_params[0].cluster_count_x as u32
            * eye_params[0].cluster_count_y as u32
            * CLUSTER_COUNT_Z;

        let (pipeline, bgl) = ensure_compute_pipeline(ctx.device);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clustered_light_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: refs.params_buffer,
                        offset: 0,
                        size: NonZeroU64::new(CLUSTER_PARAMS_UNIFORM_SIZE),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fgpu.lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: refs.cluster_light_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: refs.cluster_light_indices.as_entire_binding(),
                },
            ],
        });

        let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());

        for (eye_idx, cfp) in eye_params.iter().enumerate() {
            let cluster_offset = (eye_idx as u32) * clusters_per_eye;
            let buf_offset = (eye_idx as u64) * CLUSTER_PARAMS_UNIFORM_SIZE;
            let params = Self::build_params(
                cfp.world_to_view,
                cfp.proj,
                viewport,
                cfp.cluster_count_x,
                cfp.cluster_count_y,
                light_count,
                cfp.near_clip,
                cfp.far_clip,
                cluster_offset,
            );
            write_cluster_params_padded(&queue, refs.params_buffer, &params, buf_offset);

            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("clustered_light"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[buf_offset as u32]);
            pass.dispatch_workgroups(
                cfp.cluster_count_x.div_ceil(8),
                cfp.cluster_count_y.div_ceil(8),
                CLUSTER_COUNT_Z,
            );
        }

        if !self.logged_active_once {
            self.logged_active_once = true;
            let eye_count = eye_params.len();
            logger::info!(
                "ClusteredLight active (grid {}x{}x{} lights={} eyes={})",
                eye_params[0].cluster_count_x,
                eye_params[0].cluster_count_y,
                CLUSTER_COUNT_Z,
                light_count,
                eye_count,
            );
        }

        Ok(())
    }
}
