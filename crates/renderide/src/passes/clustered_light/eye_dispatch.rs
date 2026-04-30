//! Per-eye dispatch driver for the clustered-light compute pass.
//!
//! Walks the resolved [`ClusterFrameParams`] for each eye, clears the cluster-count slice for
//! that eye, uploads the corresponding `ClusterParams` uniform slot, and dispatches the compute
//! shader. Mono and stereo paths share the loop; eye 0 is always the mono / left-eye row.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::CLUSTER_PARAMS_UNIFORM_SIZE;
use crate::camera::HostCameraFrame;
use crate::gpu::GpuLimits;
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::scene::SceneCoordinator;
use crate::world_mesh::cluster_frame::{
    CLUSTER_COUNT_Z, ClusterFrameParams, cluster_frame_params, cluster_frame_params_stereo,
};

use super::pipeline::{ClusterParamsDesc, build_params, write_cluster_params_padded};

/// GPU and uniform state for per-eye clustered light compute dispatches.
pub(super) struct ClusteredLightEyePassEnv<'a> {
    /// Active command encoder for this recording slice.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Deferred [`wgpu::Queue::write_buffer`] sink shared with the rest of the frame.
    pub upload_batch: &'a FrameUploadBatch,
    /// Clustered-light compute pipeline.
    pub pipeline: &'a wgpu::ComputePipeline,
    /// Bind group with light/cluster/params resources.
    pub bind_group: &'a wgpu::BindGroup,
    /// Per-cluster light-count storage cleared before each eye dispatch.
    pub cluster_light_counts: &'a wgpu::Buffer,
    /// Uniform buffer holding per-eye [`ClusterFrameParams`].
    pub params_buffer: &'a wgpu::Buffer,
    /// Per-eye cluster frame params (one or two entries).
    pub eye_params: &'a [ClusterFrameParams],
    /// Number of clusters produced per eye.
    pub clusters_per_eye: u32,
    /// Scene light count (driving workgroup extent in Z).
    pub light_count: u32,
    /// Target viewport size in pixels.
    pub viewport: (u32, u32),
    /// Adapter limits for validating dispatch extents.
    pub gpu_limits: &'a GpuLimits,
    /// GPU profiler for the pass-level timestamp query on each eye's compute pass.
    pub profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Per-eye cluster compute dispatches (params upload + 3D grid).
pub(super) fn run_clustered_light_eye_passes(env: ClusteredLightEyePassEnv<'_>) {
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
        let params = build_params(ClusterParamsDesc {
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
        };
        if let (Some(p), Some(q)) = (env.profiler, pass_query) {
            p.end_query(env.encoder, q);
        }
    }
}

/// Resolves mono or stereo [`ClusterFrameParams`] rows for the current host camera and viewport.
pub(super) fn clustered_light_eye_params_for_viewport(
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

/// Returns the byte range for a contiguous cluster-count slice.
pub(super) fn cluster_count_clear_range(
    cluster_offset: u32,
    cluster_count: u32,
) -> Option<(u64, u64)> {
    let byte_offset = u64::from(cluster_offset).checked_mul(size_of::<u32>() as u64)?;
    let byte_size = u64::from(cluster_count).checked_mul(size_of::<u32>() as u64)?;
    Some((byte_offset, byte_size))
}

/// Returns the number of clusters in one eye's grid.
pub(super) fn clusters_per_eye_for_params(params: &ClusterFrameParams) -> Option<u32> {
    params
        .cluster_count_x
        .checked_mul(params.cluster_count_y)?
        .checked_mul(CLUSTER_COUNT_Z)
}

/// Clears the shared cluster-count range when there are no active lights.
pub(super) fn clear_zero_light_cluster_counts(
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
    let Some(counts_bytes) = total_clusters.checked_mul(size_of::<u32>() as u64) else {
        logger::warn!(
            "ClusteredLight: zero-light count clear byte size overflows for {total_clusters} clusters"
        );
        return;
    };
    encoder.clear_buffer(cluster_light_counts, 0, Some(counts_bytes));
}

/// Logs the clustered-light activation banner once per pass instance.
pub(super) fn log_clustered_light_active_once(
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

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use crate::world_mesh::cluster_frame::{CLUSTER_NEAR_CLIP_MIN, sanitize_cluster_clip_planes};

    use super::super::pipeline::{ClusterParamsDesc, build_params};
    use super::{cluster_count_clear_range, clusters_per_eye_for_params};

    /// Compute params apply the same cluster clip-plane sanitization as fragment lookup.
    #[test]
    fn cluster_params_use_shared_clip_plane_sanitization() {
        let params = build_params(ClusterParamsDesc {
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
        let params = crate::world_mesh::cluster_frame::ClusterFrameParams {
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
