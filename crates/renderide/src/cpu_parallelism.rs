//! Shared CPU parallelism admission policy for frame-critical renderer work.
//!
//! The renderer has many small opportunities to use Rayon, but spawning workers for two tiny
//! independent items can cost more than the work itself. This module keeps the frame hot paths on
//! one set of thresholds so view, graph, and draw-preparation stages make consistent decisions.

/// Minimum useful chunks before a Rayon fan-out is allowed.
pub(crate) const MIN_PARALLEL_CHUNKS: usize = 2;

/// Maximum worker count used when sizing renderer CPU work packets.
pub(crate) const REFERENCE_WORKER_CAP: usize = 16;

/// Minimum visibility-style items in one task packet.
pub(crate) const VISIBILITY_CULL_CHUNK_ITEMS: usize = 256;

/// Visible draw commands in one task packet.
pub(crate) const RENDER_COMMAND_CHUNK_DRAWS: usize = 64;

/// Renderable update rows in one task packet.
pub(crate) const RENDERABLE_UPDATE_CHUNK_ITEMS: usize = 32;

/// Lights in one task packet.
pub(crate) const LIGHT_WORK_CHUNK_LIGHTS: usize = 16;

/// Minimum lights before a light-work path may use Rayon.
pub(crate) const LIGHT_WORK_PARALLEL_MIN_LIGHTS: usize =
    LIGHT_WORK_CHUNK_LIGHTS * MIN_PARALLEL_CHUNKS;

/// Coarse render spaces in one task packet.
pub(crate) const COARSE_SPACE_CHUNK_ITEMS: usize = 2;

/// Independent mesh stream jobs in one task packet.
pub(crate) const MESH_STREAM_JOB_CHUNK_ITEMS: usize = 2;

/// Minimum mesh vertices before cross-stream CPU preparation may use Rayon.
pub(crate) const MESH_STREAM_JOB_MIN_VERTICES: usize = 512;

/// Blendshape extraction channel jobs in one task packet.
pub(crate) const BLENDSHAPE_CHANNEL_CHUNK_TASKS: usize = 2;

/// Minimum vertex-channel samples before blendshape extraction may use Rayon.
pub(crate) const BLENDSHAPE_CHANNEL_MIN_SAMPLES: usize = 4096;

/// Blendshape shapes in one sparse-pack task packet.
pub(crate) const BLENDSHAPE_PACK_CHUNK_SHAPES: usize = 2;

/// Minimum sparse blendshape entries before per-shape packing may use Rayon.
pub(crate) const BLENDSHAPE_PACK_MIN_SPARSE_ENTRIES: usize = 4096;

/// Bind-pose matrices in one extraction task packet.
pub(crate) const BIND_POSE_CHUNK_MATRICES: usize = 64;

/// 3D texture depth slices in one conversion task packet.
pub(crate) const TEXTURE3D_SLICE_CHUNK_SLICES: usize = 2;

/// Minimum 3D texture texels before per-slice conversion may use Rayon.
pub(crate) const TEXTURE3D_SLICE_MIN_TEXELS: usize = 16_384;

/// Minimum branchy relevance/material items in one task packet.
pub(crate) const RELEVANCE_PACKET_MIN_ITEMS: usize = 16;

/// Maximum branchy relevance/material items in one task packet.
pub(crate) const RELEVANCE_PACKET_MAX_ITEMS: usize = 512;

/// Target branchy relevance/material packets per worker.
pub(crate) const RELEVANCE_TARGET_PACKETS_PER_WORKER: usize = 6;

/// Baseline draw count where view-level frame work is usually large enough for Rayon.
const DRAW_HEAVY_PARALLEL_BASE_DRAWS: usize = 128;

/// Additional draw count per worker used to scale the draw-heavy gate on larger machines.
const DRAW_HEAVY_PARALLEL_DRAWS_PER_WORKER: usize = 16;

/// Admission decision for one parallel work site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ParallelAdmission {
    /// Run the work on the caller thread.
    Serial,
    /// Run the work through Rayon with the supplied minimum chunk size.
    Parallel {
        /// Minimum number of domain items owned by one worker split.
        chunk_size: usize,
    },
}

impl ParallelAdmission {
    /// Returns `true` when the work site should use Rayon.
    #[inline]
    pub(crate) const fn is_parallel(self) -> bool {
        matches!(self, Self::Parallel { .. })
    }

    /// Returns the admitted chunk size, or `None` for serial execution.
    #[inline]
    pub(crate) const fn chunk_size(self) -> Option<usize> {
        match self {
            Self::Serial => None,
            Self::Parallel { chunk_size } => Some(chunk_size),
        }
    }
}

/// Caps a Rayon worker count to the reference renderer scheduling bound.
#[inline]
pub(crate) const fn reference_worker_count(worker_count: usize) -> usize {
    let workers = if worker_count == 0 { 1 } else { worker_count };
    if workers > REFERENCE_WORKER_CAP {
        REFERENCE_WORKER_CAP
    } else {
        workers
    }
}

/// Returns the current Rayon worker count after applying the renderer scheduling cap.
#[inline]
pub(crate) fn current_reference_worker_count() -> usize {
    reference_worker_count(rayon::current_num_threads())
}

/// Returns `true` when `item_count` contains at least two full packets.
#[inline]
pub(crate) const fn has_two_chunks(item_count: usize, chunk_size: usize) -> bool {
    let chunk_size = if chunk_size == 0 { 1 } else { chunk_size };
    item_count >= chunk_size.saturating_mul(MIN_PARALLEL_CHUNKS)
}

/// Admits fixed-grain work when at least two task packets are available.
#[inline]
pub(crate) const fn admit_fixed_grain_items(
    item_count: usize,
    worker_count: usize,
    chunk_size: usize,
) -> ParallelAdmission {
    let chunk_size = if chunk_size == 0 { 1 } else { chunk_size };
    if reference_worker_count(worker_count) > 1 && has_two_chunks(item_count, chunk_size) {
        ParallelAdmission::Parallel { chunk_size }
    } else {
        ParallelAdmission::Serial
    }
}

/// Admits visible draw-command work using the reference draw packet size.
#[inline]
pub(crate) const fn admit_render_command_items(
    item_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    admit_fixed_grain_items(item_count, worker_count, RENDER_COMMAND_CHUNK_DRAWS)
}

/// Admits renderable update work using the reference renderable packet size.
#[inline]
pub(crate) const fn admit_renderable_update_items(
    item_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    admit_fixed_grain_items(item_count, worker_count, RENDERABLE_UPDATE_CHUNK_ITEMS)
}

/// Returns `true` when a space-level visibility-style fan-out has enough total work.
#[inline]
pub(crate) const fn has_visibility_parallel_work(item_count: usize, worker_count: usize) -> bool {
    reference_worker_count(worker_count) > 1
        && has_two_chunks(item_count, VISIBILITY_CULL_CHUNK_ITEMS)
}

/// Admits light work using the reference light packet size and light-count floor.
#[inline]
pub(crate) const fn admit_light_work_items(
    item_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if reference_worker_count(worker_count) > 1 && item_count >= LIGHT_WORK_PARALLEL_MIN_LIGHTS {
        ParallelAdmission::Parallel {
            chunk_size: LIGHT_WORK_CHUNK_LIGHTS,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Admits coarse render-space fan-outs when at least two space packets are available.
#[inline]
pub(crate) const fn admit_coarse_space_items(
    item_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    admit_fixed_grain_items(item_count, worker_count, COARSE_SPACE_CHUNK_ITEMS)
}

/// Admits multi-stream mesh CPU work when both job count and vertex count are large enough.
#[inline]
pub(crate) const fn admit_mesh_stream_jobs(
    job_count: usize,
    vertex_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if vertex_count >= MESH_STREAM_JOB_MIN_VERTICES {
        admit_fixed_grain_items(job_count, worker_count, MESH_STREAM_JOB_CHUNK_ITEMS)
    } else {
        ParallelAdmission::Serial
    }
}

/// Admits blendshape channel extraction when both task count and sample count are large enough.
#[inline]
pub(crate) const fn admit_blendshape_channel_tasks(
    task_count: usize,
    sample_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if sample_count >= BLENDSHAPE_CHANNEL_MIN_SAMPLES {
        admit_fixed_grain_items(task_count, worker_count, BLENDSHAPE_CHANNEL_CHUNK_TASKS)
    } else {
        ParallelAdmission::Serial
    }
}

/// Admits per-shape blendshape sparse packing when both shape count and sparse entry count are large enough.
#[inline]
pub(crate) const fn admit_blendshape_pack_shapes(
    shape_count: usize,
    sparse_entry_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if sparse_entry_count >= BLENDSHAPE_PACK_MIN_SPARSE_ENTRIES {
        admit_fixed_grain_items(shape_count, worker_count, BLENDSHAPE_PACK_CHUNK_SHAPES)
    } else {
        ParallelAdmission::Serial
    }
}

/// Admits bind-pose extraction when at least two matrix packets are available.
#[inline]
pub(crate) const fn admit_bind_pose_matrices(
    matrix_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    admit_fixed_grain_items(matrix_count, worker_count, BIND_POSE_CHUNK_MATRICES)
}

/// Admits 3D texture slice conversion when both slice count and texel count are large enough.
#[inline]
pub(crate) const fn admit_texture3d_slices(
    slice_count: usize,
    texel_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if texel_count >= TEXTURE3D_SLICE_MIN_TEXELS {
        admit_fixed_grain_items(slice_count, worker_count, TEXTURE3D_SLICE_CHUNK_SLICES)
    } else {
        ParallelAdmission::Serial
    }
}

/// Computes branchy relevance/material packet size using Unreal-style target packet counts.
#[inline]
pub(crate) const fn relevance_packet_size(item_count: usize, worker_count: usize) -> usize {
    let workers = reference_worker_count(worker_count);
    let target_packets = workers.saturating_mul(RELEVANCE_TARGET_PACKETS_PER_WORKER);
    let raw = if item_count == 0 || target_packets == 0 {
        RELEVANCE_PACKET_MIN_ITEMS
    } else {
        item_count.div_ceil(target_packets)
    };
    if raw < RELEVANCE_PACKET_MIN_ITEMS {
        RELEVANCE_PACKET_MIN_ITEMS
    } else if raw > RELEVANCE_PACKET_MAX_ITEMS {
        RELEVANCE_PACKET_MAX_ITEMS
    } else {
        raw
    }
}

/// Admits branchy relevance/material work using Unreal-style packet sizing.
#[inline]
pub(crate) const fn admit_relevance_items(
    item_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    let chunk_size = relevance_packet_size(item_count, worker_count);
    admit_fixed_grain_items(item_count, worker_count, chunk_size)
}

/// Records the admission decision for a reference-grain Rayon work site.
pub(crate) fn record_parallel_admission(
    _site_label: &'static str,
    work_units: usize,
    independent_items: usize,
    admission: ParallelAdmission,
) {
    profiling::scope!("rayon_admission", _site_label);
    let chunk_size = admission.chunk_size().unwrap_or(0);
    let chunk_count = if chunk_size == 0 {
        0
    } else {
        independent_items.div_ceil(chunk_size)
    };
    crate::profiling::plot_rayon_admission(crate::profiling::RayonAdmissionProfileSample {
        work_units: work_units as u64,
        independent_items: independent_items as u64,
        chunk_size: chunk_size as u64,
        chunk_count: chunk_count as u64,
        worker_count: current_reference_worker_count() as u64,
        parallel: u64::from(admission.is_parallel()),
    });
}

/// Compact description of one frame-critical CPU work site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameCpuWorkload {
    /// Independent view count represented by the work.
    view_count: usize,
    /// Estimated draw or renderer count represented by the work.
    total_draw_count: usize,
    /// Independent domain item count available for worker splits.
    independent_item_count: usize,
}

impl FrameCpuWorkload {
    /// Creates a workload from explicit view, draw, and independent-item counts.
    #[inline]
    pub(crate) const fn new(
        view_count: usize,
        total_draw_count: usize,
        independent_item_count: usize,
    ) -> Self {
        Self {
            view_count,
            total_draw_count,
            independent_item_count,
        }
    }

    /// Creates a workload for independent non-draw items such as render contexts.
    #[inline]
    pub(crate) const fn independent_items(item_count: usize) -> Self {
        Self::new(0, 0, item_count)
    }

    /// Creates a workload for view-owned draw work.
    #[inline]
    pub(crate) const fn view_draws(view_count: usize, total_draw_count: usize) -> Self {
        Self::new(view_count, total_draw_count, view_count)
    }

    /// Independent domain item count available for worker splits.
    #[inline]
    pub(crate) const fn independent_item_count(self) -> usize {
        self.independent_item_count
    }

    /// Independent view count represented by the work.
    #[inline]
    pub(crate) const fn view_count(self) -> usize {
        self.view_count
    }

    /// Estimated draw or renderer count represented by the work.
    #[inline]
    pub(crate) const fn total_draw_count(self) -> usize {
        self.total_draw_count
    }
}

/// Per-frame CPU parallelism thresholds derived from the active Rayon worker count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameParallelPolicy {
    /// Number of Rayon workers available to frame-critical work.
    worker_count: usize,
}

impl FrameParallelPolicy {
    /// Builds a policy from Rayon worker count.
    #[inline]
    pub(crate) const fn new(worker_count: usize) -> Self {
        Self {
            worker_count: reference_worker_count(worker_count),
        }
    }

    /// Builds a policy for the currently executing Rayon pool.
    #[inline]
    pub(crate) fn for_current_thread_pool() -> Self {
        Self::new(rayon::current_num_threads())
    }

    /// Draw count required before view-level frame work is considered heavy.
    #[inline]
    pub(crate) const fn draw_heavy_threshold(self) -> usize {
        let scaled = self
            .worker_count
            .saturating_mul(DRAW_HEAVY_PARALLEL_DRAWS_PER_WORKER);
        if scaled > DRAW_HEAVY_PARALLEL_BASE_DRAWS {
            scaled
        } else {
            DRAW_HEAVY_PARALLEL_BASE_DRAWS
        }
    }

    /// Minimum independent non-draw items required before worker fan-out is useful.
    #[inline]
    pub(crate) const fn independent_item_threshold(self) -> usize {
        if self.worker_count <= 1 {
            usize::MAX
        } else {
            MIN_PARALLEL_CHUNKS
        }
    }

    /// Returns `true` when the estimated draw work crosses the draw-heavy gate.
    #[inline]
    pub(crate) const fn is_draw_heavy(self, total_draw_count: usize) -> bool {
        self.worker_count > 1 && total_draw_count >= self.draw_heavy_threshold()
    }

    /// Decides whether independent item work should fan out through Rayon.
    #[inline]
    pub(crate) const fn admit_independent_items(
        self,
        workload: FrameCpuWorkload,
        chunk_size: usize,
    ) -> ParallelAdmission {
        let chunk_size = if chunk_size == 0 { 1 } else { chunk_size };
        let enough_chunks =
            workload.independent_item_count() >= chunk_size.saturating_mul(MIN_PARALLEL_CHUNKS);
        let enough_items = workload.independent_item_count() >= self.independent_item_threshold();
        if self.worker_count > 1 && enough_chunks && enough_items {
            ParallelAdmission::Parallel { chunk_size }
        } else {
            ParallelAdmission::Serial
        }
    }

    /// Decides whether view-level draw work should fan out through Rayon.
    #[inline]
    pub(crate) const fn admit_draw_heavy_views(
        self,
        workload: FrameCpuWorkload,
        chunk_size: usize,
    ) -> ParallelAdmission {
        let chunk_size = if chunk_size == 0 { 1 } else { chunk_size };
        let enough_views = workload.view_count() >= chunk_size.saturating_mul(MIN_PARALLEL_CHUNKS);
        if enough_views && self.is_draw_heavy(workload.total_draw_count()) {
            ParallelAdmission::Parallel { chunk_size }
        } else {
            ParallelAdmission::Serial
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BIND_POSE_CHUNK_MATRICES, BLENDSHAPE_CHANNEL_CHUNK_TASKS, BLENDSHAPE_CHANNEL_MIN_SAMPLES,
        BLENDSHAPE_PACK_CHUNK_SHAPES, BLENDSHAPE_PACK_MIN_SPARSE_ENTRIES, COARSE_SPACE_CHUNK_ITEMS,
        FrameCpuWorkload, FrameParallelPolicy, LIGHT_WORK_CHUNK_LIGHTS, ParallelAdmission,
        REFERENCE_WORKER_CAP, RELEVANCE_PACKET_MAX_ITEMS, RELEVANCE_PACKET_MIN_ITEMS,
        RENDER_COMMAND_CHUNK_DRAWS, RENDERABLE_UPDATE_CHUNK_ITEMS, TEXTURE3D_SLICE_CHUNK_SLICES,
        TEXTURE3D_SLICE_MIN_TEXELS, VISIBILITY_CULL_CHUNK_ITEMS, admit_bind_pose_matrices,
        admit_blendshape_channel_tasks, admit_blendshape_pack_shapes, admit_coarse_space_items,
        admit_light_work_items, admit_mesh_stream_jobs, admit_relevance_items,
        admit_render_command_items, admit_renderable_update_items, admit_texture3d_slices,
        has_two_chunks, has_visibility_parallel_work, reference_worker_count,
        relevance_packet_size,
    };

    #[test]
    fn draw_heavy_threshold_scales_with_worker_count() {
        assert_eq!(FrameParallelPolicy::new(1).draw_heavy_threshold(), 128);
        assert_eq!(FrameParallelPolicy::new(4).draw_heavy_threshold(), 128);
        assert_eq!(FrameParallelPolicy::new(8).draw_heavy_threshold(), 128);
        assert_eq!(FrameParallelPolicy::new(32).draw_heavy_threshold(), 256);
    }

    #[test]
    fn independent_items_require_multiple_chunks() {
        let policy = FrameParallelPolicy::new(8);
        assert_eq!(
            policy.admit_independent_items(FrameCpuWorkload::independent_items(1), 1),
            ParallelAdmission::Serial
        );
        assert!(
            policy
                .admit_independent_items(FrameCpuWorkload::independent_items(2), 1)
                .is_parallel()
        );
    }

    #[test]
    fn draw_heavy_views_require_multiple_views_and_enough_draws() {
        let policy = FrameParallelPolicy::new(4);
        assert_eq!(
            policy.admit_draw_heavy_views(FrameCpuWorkload::view_draws(1, 4096), 1),
            ParallelAdmission::Serial
        );
        assert_eq!(
            policy.admit_draw_heavy_views(FrameCpuWorkload::view_draws(2, 127), 1),
            ParallelAdmission::Serial
        );
        assert!(
            policy
                .admit_draw_heavy_views(FrameCpuWorkload::view_draws(2, 128), 1)
                .is_parallel()
        );
    }

    #[test]
    fn reference_worker_count_is_capped() {
        assert_eq!(reference_worker_count(0), 1);
        assert_eq!(reference_worker_count(4), 4);
        assert_eq!(reference_worker_count(64), REFERENCE_WORKER_CAP);
    }

    #[test]
    fn fixed_grain_admission_requires_two_full_chunks() {
        assert!(!has_two_chunks(
            RENDER_COMMAND_CHUNK_DRAWS * 2 - 1,
            RENDER_COMMAND_CHUNK_DRAWS
        ));
        assert!(has_two_chunks(
            RENDER_COMMAND_CHUNK_DRAWS * 2,
            RENDER_COMMAND_CHUNK_DRAWS
        ));
        assert_eq!(
            admit_render_command_items(RENDER_COMMAND_CHUNK_DRAWS * 2 - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_render_command_items(RENDER_COMMAND_CHUNK_DRAWS * 2, 8),
            ParallelAdmission::Parallel {
                chunk_size: RENDER_COMMAND_CHUNK_DRAWS
            }
        );
        assert_eq!(
            admit_render_command_items(RENDER_COMMAND_CHUNK_DRAWS * 2, 1),
            ParallelAdmission::Serial
        );
    }

    #[test]
    fn reference_grains_match_renderer_work_classes() {
        assert_eq!(
            admit_renderable_update_items(RENDERABLE_UPDATE_CHUNK_ITEMS * 2, 8),
            ParallelAdmission::Parallel {
                chunk_size: RENDERABLE_UPDATE_CHUNK_ITEMS
            }
        );
        assert!(!has_visibility_parallel_work(
            VISIBILITY_CULL_CHUNK_ITEMS,
            8
        ));
        assert!(has_visibility_parallel_work(
            VISIBILITY_CULL_CHUNK_ITEMS * 2,
            8
        ));
        assert_eq!(
            admit_light_work_items(LIGHT_WORK_CHUNK_LIGHTS * 2 - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_light_work_items(LIGHT_WORK_CHUNK_LIGHTS * 2, 8),
            ParallelAdmission::Parallel {
                chunk_size: LIGHT_WORK_CHUNK_LIGHTS
            }
        );
    }

    #[test]
    fn relevance_packet_size_uses_unreal_style_clamps() {
        assert_eq!(relevance_packet_size(1, 8), RELEVANCE_PACKET_MIN_ITEMS);
        assert_eq!(
            relevance_packet_size(RELEVANCE_PACKET_MIN_ITEMS * 6 * 8 + 1, 8),
            RELEVANCE_PACKET_MIN_ITEMS + 1
        );
        assert_eq!(
            relevance_packet_size(usize::MAX, 1),
            RELEVANCE_PACKET_MAX_ITEMS
        );
        assert_eq!(
            admit_relevance_items(RELEVANCE_PACKET_MIN_ITEMS * 2 - 1, 8),
            ParallelAdmission::Serial
        );
        assert!(admit_relevance_items(RELEVANCE_PACKET_MIN_ITEMS * 2, 8).is_parallel());
    }

    #[test]
    fn new_reference_grains_require_two_full_chunks() {
        assert_eq!(
            admit_coarse_space_items(COARSE_SPACE_CHUNK_ITEMS * 2 - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_coarse_space_items(COARSE_SPACE_CHUNK_ITEMS * 2, 8),
            ParallelAdmission::Parallel {
                chunk_size: COARSE_SPACE_CHUNK_ITEMS
            }
        );
        assert_eq!(
            admit_bind_pose_matrices(BIND_POSE_CHUNK_MATRICES * 2 - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_bind_pose_matrices(BIND_POSE_CHUNK_MATRICES * 2, 8),
            ParallelAdmission::Parallel {
                chunk_size: BIND_POSE_CHUNK_MATRICES
            }
        );
    }

    #[test]
    fn new_reference_grains_keep_work_floors() {
        assert_eq!(admit_mesh_stream_jobs(4, 511, 8), ParallelAdmission::Serial);
        assert!(admit_mesh_stream_jobs(4, 512, 8).is_parallel());
        assert_eq!(
            admit_blendshape_channel_tasks(4, BLENDSHAPE_CHANNEL_MIN_SAMPLES - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_blendshape_channel_tasks(4, BLENDSHAPE_CHANNEL_MIN_SAMPLES, 8),
            ParallelAdmission::Parallel {
                chunk_size: BLENDSHAPE_CHANNEL_CHUNK_TASKS
            }
        );
        assert_eq!(
            admit_blendshape_pack_shapes(4, BLENDSHAPE_PACK_MIN_SPARSE_ENTRIES - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_blendshape_pack_shapes(4, BLENDSHAPE_PACK_MIN_SPARSE_ENTRIES, 8),
            ParallelAdmission::Parallel {
                chunk_size: BLENDSHAPE_PACK_CHUNK_SHAPES
            }
        );
        assert_eq!(
            admit_texture3d_slices(4, TEXTURE3D_SLICE_MIN_TEXELS - 1, 8),
            ParallelAdmission::Serial
        );
        assert_eq!(
            admit_texture3d_slices(4, TEXTURE3D_SLICE_MIN_TEXELS, 8),
            ParallelAdmission::Parallel {
                chunk_size: TEXTURE3D_SLICE_CHUNK_SLICES
            }
        );
    }
}
