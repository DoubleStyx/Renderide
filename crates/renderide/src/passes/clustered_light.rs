//! Clustered forward lighting: compute pass assigns light indices per view-space cluster.
//!
//! Dispatches over a 3D grid (`16×16` pixel tiles × exponential Z slices). Uses the same
//! [`crate::backend::GpuLight`] buffer and cluster storage as raster `@group(0)`
//! ([`crate::backend::FrameGpuResources`]).
//!
//! WGSL source: `shaders/passes/compute/clustered_light.wgsl` (composed by the build script and
//! loaded from the embedded shader registry at pipeline creation time).
//!
//! ## Module layout
//!
//! - [`pipeline`] owns the process-wide compute pipeline and `ClusterParams` uniform layout.
//! - [`eye_dispatch`] runs the per-eye dispatch loop (mono / stereo).
//! - [`record_action`] selects between skip / clear-zero / CPU froxel / GPU scan per tick.
//! - [`cache`] holds the per-view bind-group cache (lifecycle-coupled to retired views).
//! - [`froxel_cpu`] implements the CPU light-centric froxel planner used for view 0.

mod cache;
mod eye_dispatch;
mod froxel_cpu;
mod pipeline;
mod record_action;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use std::num::NonZeroU64;

use cache::ClusteredLightBindGroupCache;
use eye_dispatch::{
    ClusteredLightEyePassEnv, clear_zero_light_cluster_counts,
    clustered_light_eye_params_for_viewport, clusters_per_eye_for_params,
    log_clustered_light_active_once, run_clustered_light_eye_passes,
};
use froxel_cpu::AUTO_CPU_FROXEL_LIGHT_THRESHOLD;
use pipeline::clustered_light_pipelines;
use record_action::{
    ClusteredLightClearData, ClusteredLightGpuScanData, ClusteredLightRecordAction,
    CpuFroxelRecordData, try_record_cpu_froxel,
};

use crate::backend::CLUSTER_PARAMS_UNIFORM_SIZE;
use crate::camera::ViewId;
use crate::config::ClusterAssignmentMode;
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::PerViewFramePlanSlot;
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, ImportedBufferHandle, StorageAccess,
};

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
        let frame = &mut *ctx.pass_frame;

        let (vw, vh) = frame.view.viewport_px;
        if vw == 0 || vh == 0 || frame.shared.frame_resources.frame_gpu().is_none() {
            return ClusteredLightRecordAction::Skip;
        }

        let hc = frame.view.host_camera;
        let scene = frame.shared.scene;
        let stereo = frame.view.multiview_stereo && hc.active_stereo().is_some();
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
                crate::world_mesh::cluster_frame::CLUSTER_COUNT_Z
            );
            return ClusteredLightRecordAction::Skip;
        };

        if self.should_use_cpu_froxel(view_idx, stereo, light_count)
            && try_record_cpu_froxel(CpuFroxelRecordData {
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

    /// Records the GPU scan path for clustered-light assignment.
    fn record_gpu_scan(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
        data: &ClusteredLightGpuScanData,
    ) {
        let pipelines = clustered_light_pipelines();
        let pipeline = pipelines.pipeline(ctx.device);
        let bgl = pipelines.bind_group_layout(ctx.device);
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
