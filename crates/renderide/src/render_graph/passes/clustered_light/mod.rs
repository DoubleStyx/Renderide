//! Clustered forward lighting: compute pass assigns light indices per view-space cluster.
//!
//! Dispatches over a 3D grid (`16×16` pixel tiles × exponential Z slices). Uses the same
//! [`GpuLight`] buffer and cluster storage as raster `@group(0)` ([`crate::backend::FrameGpuResources`]).
//!
//! WGSL source: `shaders/source/compute/clustered_light.wgsl` (composed by the build script and
//! loaded from the embedded shader registry at pipeline creation time).

mod cache;
mod froxel_cpu;

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use cache::ClusteredLightBindGroupCache;
use froxel_cpu::{FroxelLightPlanner, AUTO_CPU_FROXEL_LIGHT_THRESHOLD};

use crate::backend::GpuLight;
use crate::backend::{CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, TILE_SIZE};
use crate::config::ClusterAssignmentMode;
use crate::gpu::GpuLimits;
use crate::render_graph::cluster_frame::{
    cluster_frame_params, cluster_frame_params_stereo, sanitize_cluster_clip_planes,
    ClusterFrameParams,
};
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::{HostCameraFrame, PerViewFramePlanSlot};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, ImportedBufferHandle, StorageAccess,
};
use crate::render_graph::ViewId;
use crate::scene::SceneCoordinator;

/// CPU layout for the compute shader `ClusterParams` uniform (WGSL `struct` + tail pad).
///
/// `world_to_view_scale` carries the world-to-view linear-scale factor so the shader can convert
/// `light.range` (world units) to view-space units before the cluster sphere/AABB test — see
/// [`crate::render_graph::cluster_frame::ClusterFrameParams::world_to_view_scale_max`].
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
    world_to_view_scale: f32,
    _pad: [u8; 4],
}

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
            source: wgpu::ShaderSource::Wgsl(crate::embedded_shaders::CLUSTERED_LIGHT_WGSL.into()),
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
    upload_batch: &FrameUploadBatch,
    buf: &wgpu::Buffer,
    params: &ClusterParams,
    buf_offset: u64,
) {
    let mut padded = [0u8; CLUSTER_PARAMS_UNIFORM_SIZE as usize];
    let src = bytemuck::bytes_of(params);
    padded[..src.len()].copy_from_slice(src);
    upload_batch.write_buffer(buf, buf_offset, &padded);
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
    /// Max row length of the world-to-view linear part; multiplies `light.range` (world units)
    /// to view-space units inside the compute shader's culling test.
    world_to_view_scale: f32,
}

/// GPU and uniform state for per-eye clustered light compute dispatches.
struct ClusteredLightEyePassEnv<'a> {
    /// Active command encoder for this recording slice.
    encoder: &'a mut wgpu::CommandEncoder,
    /// Deferred [`wgpu::Queue::write_buffer`] sink shared with the rest of the frame.
    upload_batch: &'a FrameUploadBatch,
    /// Clustered-light compute pipeline.
    pipeline: &'a wgpu::ComputePipeline,
    /// Bind group with light/cluster/params resources.
    bind_group: &'a wgpu::BindGroup,
    /// Per-cluster light-count storage cleared before each eye dispatch.
    cluster_light_counts: &'a wgpu::Buffer,
    /// Uniform buffer holding per-eye [`ClusterFrameParams`].
    params_buffer: &'a wgpu::Buffer,
    /// Per-eye cluster frame params (one or two entries).
    eye_params: &'a [ClusterFrameParams],
    /// Number of clusters produced per eye.
    clusters_per_eye: u32,
    /// Scene light count (driving workgroup extent in Z).
    light_count: u32,
    /// Target viewport size in pixels.
    viewport: (u32, u32),
    /// Adapter limits for validating dispatch extents.
    gpu_limits: &'a GpuLimits,
    /// GPU profiler for the pass-level timestamp query on each eye's compute pass.
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Prepared work selected by clustered-light recording.
enum ClusteredLightRecordAction {
    /// Nothing to record for this view.
    Skip,
    /// CPU assignment already uploaded the cluster buffers.
    Done,
    /// Clear cluster counts because the frame has no lights.
    ClearZero(ClusteredLightClearData),
    /// Run the existing GPU scan compute path.
    GpuScan(ClusteredLightGpuScanData),
}

/// Data needed to clear empty per-cluster light counts.
struct ClusteredLightClearData {
    /// Shared cluster-count buffer.
    cluster_light_counts: wgpu::Buffer,
    /// Number of clusters produced per eye.
    clusters_per_eye: u32,
    /// Number of eyes represented by this view.
    eye_count: usize,
}

/// Inputs for CPU froxel assignment and upload.
struct CpuFroxelRecordData<'a> {
    /// Deferred upload sink for compatible cluster buffers.
    upload_batch: &'a FrameUploadBatch,
    /// CPU-side light rows packed during frame preparation.
    lights: &'a [GpuLight],
    /// Shared cluster-count buffer.
    cluster_light_counts: &'a wgpu::Buffer,
    /// Shared packed cluster-index buffer.
    cluster_light_indices: &'a wgpu::Buffer,
    /// Per-eye cluster frame params.
    eye_params: &'a [ClusterFrameParams],
    /// Number of clusters produced per eye.
    clusters_per_eye: u32,
    /// Graph view id.
    view_id: ViewId,
    /// Scene light count.
    light_count: u32,
}

/// Data needed to run the GPU clustered-light scan.
struct ClusteredLightGpuScanData {
    /// Graph view id.
    view_id: ViewId,
    /// Shared cluster-buffer cache version.
    cluster_ver: u64,
    /// Shared cluster-count buffer.
    cluster_light_counts: wgpu::Buffer,
    /// Shared packed cluster-index buffer.
    cluster_light_indices: wgpu::Buffer,
    /// Per-view cluster params uniform buffer.
    params_buffer: wgpu::Buffer,
    /// Frame light storage buffer.
    lights_buffer: wgpu::Buffer,
    /// Per-eye cluster frame params.
    eye_params: Vec<ClusterFrameParams>,
    /// Number of clusters produced per eye.
    clusters_per_eye: u32,
    /// Scene light count.
    light_count: u32,
    /// Target viewport size in pixels.
    viewport: (u32, u32),
}

/// Returns the byte range for a contiguous cluster-count slice.
fn cluster_count_clear_range(cluster_offset: u32, cluster_count: u32) -> Option<(u64, u64)> {
    let byte_offset = u64::from(cluster_offset).checked_mul(std::mem::size_of::<u32>() as u64)?;
    let byte_size = u64::from(cluster_count).checked_mul(std::mem::size_of::<u32>() as u64)?;
    Some((byte_offset, byte_size))
}

/// Returns the number of clusters in one eye's grid.
fn clusters_per_eye_for_params(params: &ClusterFrameParams) -> Option<u32> {
    params
        .cluster_count_x
        .checked_mul(params.cluster_count_y)?
        .checked_mul(CLUSTER_COUNT_Z)
}

/// Clears the shared cluster-count range when there are no active lights.
fn clear_zero_light_cluster_counts(
    encoder: &mut wgpu::CommandEncoder,
    cluster_light_counts: &wgpu::Buffer,
    clusters_per_eye: u32,
    eye_count: usize,
) {
    let Some(total_clusters) = u64::from(clusters_per_eye).checked_mul(eye_count as u64) else {
        logger::warn!(
            "ClusteredLight: zero-light cluster clear overflows for clusters_per_eye={} eyes={}",
            clusters_per_eye,
            eye_count
        );
        return;
    };
    let Some(counts_bytes) = total_clusters.checked_mul(std::mem::size_of::<u32>() as u64) else {
        logger::warn!(
            "ClusteredLight: zero-light count clear byte size overflows for {total_clusters} clusters"
        );
        return;
    };
    encoder.clear_buffer(cluster_light_counts, 0, Some(counts_bytes));
}

/// Logs the clustered-light activation banner once per pass instance.
fn log_clustered_light_active_once(
    logged_active_once: &AtomicBool,
    first_eye_params: &ClusterFrameParams,
    light_count: u32,
    eye_count: usize,
) {
    if logged_active_once
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }

    logger::info!(
        "ClusteredLight active (grid {}x{}x{} lights={} eyes={})",
        first_eye_params.cluster_count_x,
        first_eye_params.cluster_count_y,
        CLUSTER_COUNT_Z,
        light_count,
        eye_count,
    );
}

/// Per-eye cluster compute dispatches (params upload + 3D grid).
fn run_clustered_light_eye_passes(env: ClusteredLightEyePassEnv<'_>) {
    profiling::scope!("clustered_light::eye_passes");
    for (eye_idx, cfp) in env.eye_params.iter().enumerate() {
        let Some(cluster_offset) = (eye_idx as u32).checked_mul(env.clusters_per_eye) else {
            logger::warn!(
                "ClusteredLight: eye index {eye_idx} with {} clusters per eye overflows u32",
                env.clusters_per_eye
            );
            continue;
        };
        let Some((count_clear_offset, count_clear_size)) =
            cluster_count_clear_range(cluster_offset, env.clusters_per_eye)
        else {
            logger::warn!(
                "ClusteredLight: count clear range overflow for offset={} clusters={}",
                cluster_offset,
                env.clusters_per_eye
            );
            continue;
        };
        env.encoder.clear_buffer(
            env.cluster_light_counts,
            count_clear_offset,
            Some(count_clear_size),
        );
        let buf_offset = (eye_idx as u64) * CLUSTER_PARAMS_UNIFORM_SIZE;
        let (near, far) = cfp.sanitized_clip_planes();
        let params = ClusteredLightPass::build_params(ClusterParamsDesc {
            scene_view: cfp.world_to_view,
            proj: cfp.proj,
            viewport: env.viewport,
            cluster_count_x: cfp.cluster_count_x,
            cluster_count_y: cfp.cluster_count_y,
            light_count: env.light_count,
            near,
            far,
            cluster_offset,
            world_to_view_scale: cfp.world_to_view_scale_max(),
        });
        write_cluster_params_padded(env.upload_batch, env.params_buffer, &params, buf_offset);

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

        let pass_query = env
            .profiler
            .map(|p| p.begin_pass_query("clustered_light", env.encoder));
        let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut pass = env
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("clustered_light"),
                    timestamp_writes,
                });
            pass.set_pipeline(env.pipeline);
            pass.set_bind_group(0, env.bind_group, &[buf_offset as u32]);
            pass.dispatch_workgroups(dx, dy, dz);
        }
        if let (Some(p), Some(q)) = (env.profiler, pass_query) {
            p.end_query(env.encoder, q);
        }
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
        } else {
            cluster_frame_params(hc, scene, viewport).map(|mono| vec![mono])
        }
    } else {
        cluster_frame_params(hc, scene, viewport).map(|mono| vec![mono])
    }
}

/// Builds per-cluster light lists before the world forward pass.
#[derive(Debug)]
pub struct ClusteredLightPass {
    resources: ClusteredLightGraphResources,
    /// Assignment backend selected when this graph was compiled.
    assignment_mode: ClusterAssignmentMode,
    /// Logged once on first successful dispatch; uses an atomic to allow `record(&self, …)`.
    logged_active_once: AtomicBool,
    /// Logged once when CPU froxel mode is requested for a view that cannot safely use it.
    logged_cpu_fallback_once: AtomicBool,
    /// Per-view compute bind group cache: invalidated when the per-view cluster buffer version changes.
    bind_group_cache: ClusteredLightBindGroupCache,
}

/// Graph resources used by [`ClusteredLightPass`].
#[derive(Clone, Copy, Debug)]
pub struct ClusteredLightGraphResources {
    /// Imported light storage buffer.
    pub lights: ImportedBufferHandle,
    /// Imported per-cluster light-count storage buffer.
    pub cluster_light_counts: ImportedBufferHandle,
    /// Imported per-cluster light-index storage buffer.
    pub cluster_light_indices: ImportedBufferHandle,
    /// Transient uniform buffer for per-eye cluster parameters.
    pub params: BufferHandle,
}

impl ClusteredLightPass {
    /// Creates a clustered light pass (pipeline is created lazily on first execute).
    pub fn new(
        resources: ClusteredLightGraphResources,
        assignment_mode: ClusterAssignmentMode,
    ) -> Self {
        Self {
            resources,
            assignment_mode,
            logged_active_once: AtomicBool::new(false),
            logged_cpu_fallback_once: AtomicBool::new(false),
            bind_group_cache: ClusteredLightBindGroupCache::new(),
        }
    }

    fn build_params(desc: ClusterParamsDesc) -> ClusterParams {
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
            _pad: [0u8; 4],
        }
    }

    /// Returns the compute bind group for `view_id`, rebuilding it when `cluster_ver` changes.
    ///
    /// `params_buffer` is **per-view** and intentionally separated from `ClusterBufferRefs` to
    /// prevent a CPU write-order race in the shared `FrameUploadBatch` during parallel recording.
    fn ensure_cluster_compute_bind_group(
        &self,
        device: &wgpu::Device,
        view_id: ViewId,
        cluster_ver: u64,
        bufs: ClusterComputeBuffers<'_>,
        bgl: &wgpu::BindGroupLayout,
    ) -> Arc<wgpu::BindGroup> {
        self.bind_group_cache
            .get_or_rebuild(view_id, cluster_ver, || {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("clustered_light_compute"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: bufs.params,
                                offset: 0,
                                size: NonZeroU64::new(CLUSTER_PARAMS_UNIFORM_SIZE),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: bufs.lights.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bufs.counts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: bufs.indices.as_entire_binding(),
                        },
                    ],
                })
            })
    }

    /// Returns whether this pass should use CPU froxel assignment for the current view.
    fn should_use_cpu_froxel(&self, view_idx: usize, stereo: bool, light_count: u32) -> bool {
        match self.assignment_mode {
            ClusterAssignmentMode::GpuScan => false,
            ClusterAssignmentMode::Auto => {
                view_idx == 0 && stereo && light_count >= AUTO_CPU_FROXEL_LIGHT_THRESHOLD
            }
            ClusterAssignmentMode::CpuFroxel => {
                if view_idx == 0 {
                    true
                } else {
                    self.log_cpu_froxel_fallback(view_idx);
                    false
                }
            }
        }
    }

    /// Logs the shared-buffer ordering fallback once.
    fn log_cpu_froxel_fallback(&self, view_idx: usize) {
        if self
            .logged_cpu_fallback_once
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            logger::warn!(
                "ClusteredLight: CPU froxel assignment requested for view index {view_idx}, but only view 0 can safely use pre-submit CPU writes with the current shared cluster buffers; falling back to GPU scan for later views"
            );
        }
    }

    /// Selects and prepares the clustered-light work for the current graph view.
    fn prepare_record_action(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
    ) -> ClusteredLightRecordAction {
        let Some(frame) = ctx.frame.as_mut() else {
            return ClusteredLightRecordAction::Skip;
        };

        let (vw, vh) = frame.view.viewport_px;
        if vw == 0 || vh == 0 || frame.shared.frame_resources.frame_gpu().is_none() {
            return ClusteredLightRecordAction::Skip;
        }

        let hc = frame.view.host_camera;
        let scene = frame.shared.scene;
        let stereo = hc.vr_active && hc.stereo.is_some() && frame.view.multiview_stereo;
        let view_id = frame.view.view_id;
        let view_idx = ctx
            .blackboard
            .get::<PerViewFramePlanSlot>()
            .map_or(0, |plan| plan.view_idx);
        let light_count = frame.shared.frame_resources.frame_light_count_u32();

        let Some(refs) = frame.shared.frame_resources.shared_cluster_buffer_refs() else {
            logger::trace!("ClusteredLight: shared cluster buffers missing for {view_id:?}");
            return ClusteredLightRecordAction::Skip;
        };
        let cluster_light_counts = (*refs.cluster_light_counts).clone();
        let cluster_light_indices = (*refs.cluster_light_indices).clone();
        let cluster_ver = frame.shared.frame_resources.shared_cluster_version();

        let Some(params_buffer) = frame
            .shared
            .frame_resources
            .per_view_frame(view_id)
            .map(|s| s.cluster_params_buffer.clone())
        else {
            logger::trace!("ClusteredLight: per-view params buffer missing for {view_id:?}");
            return ClusteredLightRecordAction::Skip;
        };

        let viewport = (vw, vh);
        let Some(eye_params) =
            clustered_light_eye_params_for_viewport(stereo, &hc, scene, viewport)
        else {
            return ClusteredLightRecordAction::Skip;
        };
        let Some(clusters_per_eye) = clusters_per_eye_for_params(&eye_params[0]) else {
            logger::warn!(
                "ClusteredLight: cluster grid {}x{}x{} overflows u32",
                eye_params[0].cluster_count_x,
                eye_params[0].cluster_count_y,
                CLUSTER_COUNT_Z
            );
            return ClusteredLightRecordAction::Skip;
        };

        if self.should_use_cpu_froxel(view_idx, stereo, light_count)
            && self.try_record_cpu_froxel(CpuFroxelRecordData {
                upload_batch: ctx.upload_batch,
                lights: frame.shared.frame_resources.frame_lights(),
                cluster_light_counts: &cluster_light_counts,
                cluster_light_indices: &cluster_light_indices,
                eye_params: &eye_params,
                clusters_per_eye,
                view_id,
                light_count,
            })
        {
            return ClusteredLightRecordAction::Done;
        }

        if light_count == 0 {
            return ClusteredLightRecordAction::ClearZero(ClusteredLightClearData {
                cluster_light_counts,
                clusters_per_eye,
                eye_count: eye_params.len(),
            });
        }

        let Some(lights_buffer) = frame
            .shared
            .frame_resources
            .frame_gpu()
            .map(|fgpu| fgpu.lights_buffer.clone())
        else {
            return ClusteredLightRecordAction::Skip;
        };

        ClusteredLightRecordAction::GpuScan(ClusteredLightGpuScanData {
            view_id,
            cluster_ver,
            cluster_light_counts,
            cluster_light_indices,
            params_buffer,
            lights_buffer,
            eye_params,
            clusters_per_eye,
            light_count,
            viewport,
        })
    }

    /// Attempts CPU light-centric froxel assignment and uploads compatible cluster buffers.
    fn try_record_cpu_froxel(&self, data: CpuFroxelRecordData<'_>) -> bool {
        let Some(assignments) =
            FroxelLightPlanner::build(data.lights, data.eye_params, data.clusters_per_eye)
        else {
            logger::warn!(
                "ClusteredLight: CPU froxel assignment failed for {:?}; falling back to GPU scan",
                data.view_id
            );
            return false;
        };

        profiling::scope!("clustered_light::cpu_froxel_upload");
        data.upload_batch.write_buffer(
            data.cluster_light_counts,
            0,
            bytemuck::cast_slice(&assignments.counts),
        );
        if !assignments.indices.is_empty() {
            data.upload_batch.write_buffer(
                data.cluster_light_indices,
                0,
                bytemuck::cast_slice(&assignments.indices),
            );
        }
        logger::trace!(
            "ClusteredLight: CPU froxel assignment view={view_id:?} lights={} eyes={} memberships={} overflowed={}",
            data.light_count,
            data.eye_params.len(),
            assignments.stats.assigned_memberships,
            assignments.stats.overflowed_memberships,
            view_id = data.view_id
        );
        true
    }

    /// Records the existing GPU scan path for clustered-light assignment.
    fn record_gpu_scan(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
        data: &ClusteredLightGpuScanData,
    ) {
        let (pipeline, bgl) = ensure_compute_pipeline(ctx.device);
        let bind_group = self.ensure_cluster_compute_bind_group(
            ctx.device,
            data.view_id,
            data.cluster_ver,
            ClusterComputeBuffers {
                params: &data.params_buffer,
                lights: &data.lights_buffer,
                counts: &data.cluster_light_counts,
                indices: &data.cluster_light_indices,
            },
            bgl,
        );

        run_clustered_light_eye_passes(ClusteredLightEyePassEnv {
            encoder: ctx.encoder,
            upload_batch: ctx.upload_batch,
            pipeline,
            bind_group: &bind_group,
            cluster_light_counts: &data.cluster_light_counts,
            params_buffer: &data.params_buffer,
            eye_params: &data.eye_params,
            clusters_per_eye: data.clusters_per_eye,
            light_count: data.light_count,
            viewport: data.viewport,
            gpu_limits: ctx.gpu_limits,
            profiler: ctx.profiler,
        });
    }
}

/// Buffer refs needed to build the clustered-light compute bind group.
struct ClusterComputeBuffers<'a> {
    /// Per-view `ClusterParams` uniform (camera matrix, projection, etc.).
    params: &'a wgpu::Buffer,
    /// Scene lights storage (read-only).
    lights: &'a wgpu::Buffer,
    /// Shared per-cluster light-count storage (write).
    counts: &'a wgpu::Buffer,
    /// Shared per-cluster packed light-index storage (write).
    indices: &'a wgpu::Buffer,
}

impl ComputePass for ClusteredLightPass {
    fn name(&self) -> &str {
        "ClusteredLight"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.import_buffer(
            self.resources.lights,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.cluster_light_counts,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.import_buffer(
            self.resources.cluster_light_indices,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.write_buffer(
            self.resources.params,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::COMPUTE,
                dynamic_offset: true,
            },
        );
        Ok(())
    }

    fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        self.bind_group_cache.retire_views(retired_views);
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("clustered_light::record_dispatch");
        match self.prepare_record_action(ctx) {
            ClusteredLightRecordAction::Skip | ClusteredLightRecordAction::Done => {}
            ClusteredLightRecordAction::ClearZero(data) => {
                clear_zero_light_cluster_counts(
                    ctx.encoder,
                    &data.cluster_light_counts,
                    data.clusters_per_eye,
                    data.eye_count,
                );
            }
            ClusteredLightRecordAction::GpuScan(data) => {
                self.record_gpu_scan(ctx, &data);
                log_clustered_light_active_once(
                    &self.logged_active_once,
                    &data.eye_params[0],
                    data.light_count,
                    data.eye_params.len(),
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use crate::render_graph::cluster_frame::{sanitize_cluster_clip_planes, CLUSTER_NEAR_CLIP_MIN};

    use super::{
        cluster_count_clear_range, clusters_per_eye_for_params, ClusterParams, ClusterParamsDesc,
        ClusteredLightPass, CLUSTER_PARAMS_UNIFORM_SIZE,
    };

    /// `ClusterParams` must fit within the dynamic-offset slot reserved by
    /// `CLUSTER_PARAMS_UNIFORM_SIZE`; `write_cluster_params_padded` zero-pads the rest.
    #[test]
    fn cluster_params_struct_fits_uniform_slot() {
        assert!(
            std::mem::size_of::<ClusterParams>() as u64 <= CLUSTER_PARAMS_UNIFORM_SIZE,
            "ClusterParams ({} bytes) exceeds CLUSTER_PARAMS_UNIFORM_SIZE ({} bytes)",
            std::mem::size_of::<ClusterParams>(),
            CLUSTER_PARAMS_UNIFORM_SIZE,
        );
        assert_eq!(
            std::mem::size_of::<ClusterParams>() % 16,
            0,
            "ClusterParams must be 16-byte aligned for WGSL std140 uniform layout"
        );
    }

    /// Compute params apply the same cluster clip-plane sanitization as fragment lookup.
    #[test]
    fn cluster_params_use_shared_clip_plane_sanitization() {
        let params = ClusteredLightPass::build_params(ClusterParamsDesc {
            scene_view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            viewport: (1, 1),
            cluster_count_x: 1,
            cluster_count_y: 1,
            light_count: 0,
            near: 0.00001,
            far: 10.0,
            cluster_offset: 0,
            world_to_view_scale: 1.0,
        });
        let (near, far) = sanitize_cluster_clip_planes(0.00001, 10.0);

        assert_eq!(params.near_clip, near);
        assert_eq!(params.near_clip, CLUSTER_NEAR_CLIP_MIN);
        assert_eq!(params.far_clip, far);
    }

    /// Count clears address one `u32` per cluster.
    #[test]
    fn cluster_count_clear_range_uses_u32_stride() {
        assert_eq!(cluster_count_clear_range(3, 5), Some((12, 20)));
    }

    /// Reasonable grids fit in the checked per-eye cluster count.
    #[test]
    fn clusters_per_eye_checked_math_handles_reasonable_grid() {
        let params = crate::render_graph::cluster_frame::ClusterFrameParams {
            near_clip: 0.1,
            far_clip: 1000.0,
            world_to_view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            cluster_count_x: 4,
            cluster_count_y: 3,
            viewport_width: 128,
            viewport_height: 96,
        };

        assert_eq!(
            clusters_per_eye_for_params(&params),
            Some(4 * 3 * super::CLUSTER_COUNT_Z)
        );
    }
}
