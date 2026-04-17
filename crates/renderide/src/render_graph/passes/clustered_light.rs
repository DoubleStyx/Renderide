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

use crate::backend::ClusterBufferRefs;
use crate::backend::GpuLight;
use crate::backend::{CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, TILE_SIZE};
use crate::gpu::GpuLimits;
use crate::render_graph::cluster_frame::{
    cluster_frame_params, cluster_frame_params_stereo, ClusterFrameParams,
};
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::frame_params::HostCameraFrame;
use crate::render_graph::handles::ResourceId;
use crate::render_graph::module::RenderModule;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::PassResources;
use crate::render_graph::{GraphBuilder, SharedRenderHandles};
use crate::scene::SceneCoordinator;

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

/// Descriptor for building the `ClusterParams` uniform from scene matrices and cluster grid metadata.
struct ClusterParamsDesc {
    scene_view: Mat4,
    proj: Mat4,
    viewport: (u32, u32),
    cluster_count_x: u32,
    cluster_count_y: u32,
    light_count: u32,
    near: f32,
    far: f32,
    cluster_offset: u32,
}

/// GPU and uniform state for per-eye clustered light compute dispatches.
struct ClusteredLightEyePassEnv<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
    queue: &'a wgpu::Queue,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    params_buffer: &'a wgpu::Buffer,
    eye_params: &'a [ClusterFrameParams],
    clusters_per_eye: u32,
    light_count: u32,
    viewport: (u32, u32),
    gpu_limits: &'a GpuLimits,
}

/// Per-eye cluster compute dispatches (params upload + 3D grid).
fn run_clustered_light_eye_passes(env: ClusteredLightEyePassEnv<'_>) {
    for (eye_idx, cfp) in env.eye_params.iter().enumerate() {
        let cluster_offset = (eye_idx as u32) * env.clusters_per_eye;
        let buf_offset = (eye_idx as u64) * CLUSTER_PARAMS_UNIFORM_SIZE;
        let params = ClusteredLightPass::build_params(ClusterParamsDesc {
            scene_view: cfp.world_to_view,
            proj: cfp.proj,
            viewport: env.viewport,
            cluster_count_x: cfp.cluster_count_x,
            cluster_count_y: cfp.cluster_count_y,
            light_count: env.light_count,
            near: cfp.near_clip,
            far: cfp.far_clip,
            cluster_offset,
        });
        write_cluster_params_padded(env.queue, env.params_buffer, &params, buf_offset);

        let dx = cfp.cluster_count_x.div_ceil(8);
        let dy = cfp.cluster_count_y.div_ceil(8);
        let dz = CLUSTER_COUNT_Z;
        if !env.gpu_limits.compute_dispatch_fits(dx, dy, dz) {
            logger::warn!(
                "ClusteredLight: dispatch {}×{}×{} exceeds max_compute_workgroups_per_dimension ({})",
                dx,
                dy,
                dz,
                env.gpu_limits.max_compute_workgroups_per_dimension()
            );
            continue;
        }

        let mut pass = env
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clustered_light"),
                timestamp_writes: None,
            });
        pass.set_pipeline(env.pipeline);
        pass.set_bind_group(0, env.bind_group, &[buf_offset as u32]);
        pass.dispatch_workgroups(dx, dy, dz);
    }
}

/// Resolves mono or stereo [`ClusterFrameParams`] rows for the current host camera and viewport.
fn clustered_light_eye_params_for_viewport(
    stereo: bool,
    hc: &HostCameraFrame,
    scene: &SceneCoordinator,
    viewport: (u32, u32),
) -> Option<Vec<ClusterFrameParams>> {
    if stereo {
        if let Some((left, right)) = cluster_frame_params_stereo(hc, scene, viewport) {
            Some(vec![left, right])
        } else if let Some(mono) = cluster_frame_params(hc, scene, viewport) {
            Some(vec![mono])
        } else {
            None
        }
    } else if let Some(mono) = cluster_frame_params(hc, scene, viewport) {
        Some(vec![mono])
    } else {
        None
    }
}

/// Builds per-cluster light lists before the world forward pass.
#[derive(Debug)]
pub struct ClusteredLightPass {
    cluster_buffers: ResourceId,
    light_buffer: ResourceId,
    logged_active_once: bool,
    /// Last [`crate::backend::cluster_gpu::ClusterBufferCache::version`] used for compute bind group; recreated when cluster buffers reallocate.
    cached_cluster_bind_version: Option<u64>,
    cached_compute_bind_group: Option<wgpu::BindGroup>,
}

impl ClusteredLightPass {
    /// Creates a clustered light pass (pipeline is created lazily on first execute).
    pub fn new(cluster_buffers: ResourceId, light_buffer: ResourceId) -> Self {
        Self {
            cluster_buffers,
            light_buffer,
            logged_active_once: false,
            cached_cluster_bind_version: None,
            cached_compute_bind_group: None,
        }
    }

    fn build_params(desc: ClusterParamsDesc) -> ClusterParams {
        let inv_proj = desc.proj.inverse();
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
            near_clip: desc.near.max(0.01),
            far_clip: desc.far,
            cluster_offset: desc.cluster_offset,
            _pad: [0; 8],
        }
    }

    /// Rebuilds compute bind group when cluster buffer storage is reallocated.
    fn ensure_cluster_compute_bind_group(
        &mut self,
        device: &wgpu::Device,
        cluster_ver: u64,
        refs: &ClusterBufferRefs<'_>,
        lights_buffer: &wgpu::Buffer,
        bgl: &wgpu::BindGroupLayout,
    ) -> &wgpu::BindGroup {
        let need_new_bind_group = self.cached_cluster_bind_version != Some(cluster_ver)
            || self.cached_compute_bind_group.is_none();
        if need_new_bind_group {
            self.cached_cluster_bind_version = Some(cluster_ver);
            self.cached_compute_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                            resource: lights_buffer.as_entire_binding(),
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
                }));
        }
        self.cached_compute_bind_group
            .as_ref()
            .expect("cached clustered_light bind group")
    }
}

/// Registers [`ClusteredLightPass`] on the main frame graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClusteredLightModule;

impl RenderModule for ClusteredLightModule {
    fn name(&self) -> &str {
        "clustered_light"
    }

    fn register(self: Box<Self>, builder: &mut GraphBuilder, handles: &SharedRenderHandles) {
        builder.add_pass(Box::new(ClusteredLightPass::new(
            handles.cluster_buffers,
            handles.light_buffer,
        )));
    }
}

impl RenderPass for ClusteredLightPass {
    fn name(&self) -> &str {
        "ClusteredLight"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: vec![self.cluster_buffers, self.light_buffer],
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

        let light_count = frame.backend.frame_resources.frame_light_count_u32();

        let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
        let Some(fgpu) = frame
            .backend
            .frame_resources
            .sync_cluster_viewport_ensure_lights_upload(ctx.device, &queue, (vw, vh), stereo)
        else {
            return Ok(());
        };

        let Some(refs) = fgpu
            .cluster_cache
            .get_buffers((vw, vh), CLUSTER_COUNT_Z, stereo)
        else {
            logger::trace!("ClusteredLight: cluster buffers missing after sync");
            return Ok(());
        };
        let viewport = (vw, vh);

        let Some(eye_params) =
            clustered_light_eye_params_for_viewport(stereo, &hc, scene, viewport)
        else {
            return Ok(());
        };

        let clusters_per_eye =
            eye_params[0].cluster_count_x * eye_params[0].cluster_count_y * CLUSTER_COUNT_Z;

        let (pipeline, bgl) = ensure_compute_pipeline(ctx.device);
        let cluster_ver = fgpu.cluster_cache.version;
        let bind_group = self.ensure_cluster_compute_bind_group(
            ctx.device,
            cluster_ver,
            &refs,
            &fgpu.lights_buffer,
            bgl,
        );

        run_clustered_light_eye_passes(ClusteredLightEyePassEnv {
            encoder: ctx.encoder,
            queue: &queue,
            pipeline,
            bind_group,
            params_buffer: refs.params_buffer,
            eye_params: &eye_params,
            clusters_per_eye,
            light_count,
            viewport,
            gpu_limits: ctx.gpu_limits,
        });

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
