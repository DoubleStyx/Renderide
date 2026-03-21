//! CPU-side mesh draw preparation for the render graph: buffer ensures, projection, and draw collection.
//!
//! Used from the collect phase ([`prepare_mesh_draws_for_view`]) and from [`super::graph::RenderGraph::execute`]
//! when pre-collected draws are not supplied.

use nalgebra::Matrix4;

use super::mesh_draw::{CollectMeshDrawsContext, collect_mesh_draws};
use crate::render::batch::SpaceDrawBatch;
use crate::render::view::ViewParams;
use crate::session::Session;

/// Cached mesh draws: (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned).
pub(crate) type CachedMeshDraws = (
    Vec<super::mesh_draw::SkinnedBatchedDraw>,
    Vec<super::mesh_draw::SkinnedBatchedDraw>,
    Vec<super::mesh_draw::BatchedDraw>,
    Vec<super::mesh_draw::BatchedDraw>,
);

/// Reference to cached mesh draws for render pass context.
pub(crate) type CachedMeshDrawsRef<'a> = (
    &'a [super::mesh_draw::SkinnedBatchedDraw],
    &'a [super::mesh_draw::SkinnedBatchedDraw],
    &'a [super::mesh_draw::BatchedDraw],
    &'a [super::mesh_draw::BatchedDraw],
);

/// CPU mesh-draw prep counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshDrawPrepStats {
    /// Total draws visited across all batches before mesh/GPU validation.
    pub total_input_draws: usize,
    /// Total non-skinned draws visited.
    pub rigid_input_draws: usize,
    /// Total skinned draws visited.
    pub skinned_input_draws: usize,
    /// Submitted rigid draws after CPU culling/validation.
    pub submitted_rigid_draws: usize,
    /// Submitted skinned draws after validation.
    pub submitted_skinned_draws: usize,
    /// Rigid draws rejected by CPU frustum culling.
    pub frustum_culled_rigid_draws: usize,
    /// Skinned draws rejected by CPU frustum culling (bone-position AABB test).
    pub frustum_culled_skinned_draws: usize,
    /// Rigid draws kept because upload bounds were degenerate, so culling was skipped.
    pub skipped_cull_degenerate_bounds: usize,
    /// Draws skipped because `mesh_asset_id < 0`.
    pub skipped_invalid_mesh_asset_id: usize,
    /// Draws skipped because the mesh asset was not found.
    pub skipped_missing_mesh_asset: usize,
    /// Draws skipped because the mesh had no vertices or indices.
    pub skipped_empty_mesh: usize,
    /// Draws skipped because GPU buffers were not resident.
    pub skipped_missing_gpu_buffers: usize,
    /// Skinned draws skipped because bind poses were missing.
    pub skipped_skinned_missing_bind_poses: usize,
    /// Skinned draws skipped because bone IDs were missing or empty.
    pub skipped_skinned_missing_bone_ids: usize,
    /// Skinned draws skipped because bone ID count exceeded bind pose count.
    pub skipped_skinned_id_count_mismatch: usize,
    /// Skinned draws skipped because the skinned vertex buffer was missing.
    pub skipped_skinned_missing_vertex_buffer: usize,
}

impl MeshDrawPrepStats {
    /// Total draws submitted after prep.
    pub fn submitted_draws(&self) -> usize {
        self.submitted_rigid_draws + self.submitted_skinned_draws
    }

    /// Merges per-batch stats into running totals (used by [`super::mesh_draw::collect_mesh_draws`]).
    pub(crate) fn accumulate(&mut self, other: &Self) {
        self.total_input_draws += other.total_input_draws;
        self.rigid_input_draws += other.rigid_input_draws;
        self.skinned_input_draws += other.skinned_input_draws;
        self.submitted_rigid_draws += other.submitted_rigid_draws;
        self.submitted_skinned_draws += other.submitted_skinned_draws;
        self.frustum_culled_rigid_draws += other.frustum_culled_rigid_draws;
        self.frustum_culled_skinned_draws += other.frustum_culled_skinned_draws;
        self.skipped_cull_degenerate_bounds += other.skipped_cull_degenerate_bounds;
        self.skipped_invalid_mesh_asset_id += other.skipped_invalid_mesh_asset_id;
        self.skipped_missing_mesh_asset += other.skipped_missing_mesh_asset;
        self.skipped_empty_mesh += other.skipped_empty_mesh;
        self.skipped_missing_gpu_buffers += other.skipped_missing_gpu_buffers;
        self.skipped_skinned_missing_bind_poses += other.skipped_skinned_missing_bind_poses;
        self.skipped_skinned_missing_bone_ids += other.skipped_skinned_missing_bone_ids;
        self.skipped_skinned_id_count_mismatch += other.skipped_skinned_id_count_mismatch;
        self.skipped_skinned_missing_vertex_buffer += other.skipped_skinned_missing_vertex_buffer;
    }
}

/// Runs mesh-draw CPU collection for the main view and graph fallback paths.
///
/// [`Session`] is not [`Sync`] today (IPC queues), so per-batch worker threads cannot safely share
/// `&Session` yet. [`RenderConfig::parallel_mesh_draw_prep_batches`](crate::config::RenderConfig::parallel_mesh_draw_prep_batches) is reserved for when prep uses
/// owned snapshots or the session becomes shareable for read-only prep.
pub(super) fn run_collect_mesh_draws(
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    gpu: &mut crate::gpu::GpuState,
    proj: Matrix4<f32>,
    overlay_projection_override: Option<ViewParams>,
) -> (CachedMeshDraws, MeshDrawPrepStats) {
    let mut collect_ctx = CollectMeshDrawsContext {
        session,
        draw_batches,
        mesh_buffer_cache: &gpu.mesh_buffer_cache,
        rigid_frustum_cull_cache: &mut gpu.rigid_frustum_cull_cache,
        proj,
        overlay_projection_override,
    };
    let (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned, stats) =
        collect_mesh_draws(&mut collect_ctx);
    (
        (
            non_overlay_skinned,
            overlay_skinned,
            non_overlay_non_skinned,
            overlay_non_skinned,
        ),
        stats,
    )
}

/// Pre-collected mesh draws and view parameters for the main view.
///
/// Produced by [`prepare_mesh_draws_for_view`] during the collect phase for the same
/// render extent as the [`crate::render::RenderTarget`] passed into [`crate::render::RenderLoop::render_frame`]
/// (typically the acquired swapchain texture size, not window client area alone).
pub struct PreCollectedFrameData {
    /// Primary projection matrix for the main view.
    pub proj: Matrix4<f32>,
    /// Overlay projection override when overlays use orthographic.
    pub overlay_projection_override: Option<ViewParams>,
    /// Cached mesh draws for mesh and overlay passes.
    pub(crate) cached_mesh_draws: CachedMeshDraws,
    /// CPU-side mesh draw preparation counters for diagnostics.
    pub prep_stats: MeshDrawPrepStats,
}

/// Prepares mesh draws for the main view during the collect phase.
///
/// `viewport` must match the width and height of the swapchain (or other color target)
/// that will be rendered to in the same frame, so projection and cached draws agree
/// with the GPU viewport.
///
/// Runs [`ensure_mesh_buffers`] and [`run_collect_mesh_draws`] so this CPU work
/// is measured in the collect phase rather than the render phase.
pub fn prepare_mesh_draws_for_view(
    gpu: &mut crate::gpu::GpuState,
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    viewport: (u32, u32),
) -> PreCollectedFrameData {
    ensure_mesh_buffers(gpu, session, draw_batches);
    let (width, height) = viewport;
    let aspect = width as f32 / height.max(1) as f32;
    let view_params = ViewParams::perspective_from_session(session, aspect);
    let proj = view_params.to_projection_matrix();
    let overlay_projection_override =
        ViewParams::overlay_projection_for_frame(session, draw_batches, aspect);
    let (cached_mesh_draws, prep_stats) = run_collect_mesh_draws(
        session,
        draw_batches,
        gpu,
        proj,
        overlay_projection_override.clone(),
    );
    PreCollectedFrameData {
        proj,
        overlay_projection_override,
        cached_mesh_draws,
        prep_stats,
    }
}

/// Ensures all meshes referenced by draw batches are in the GPU mesh buffer cache.
///
/// When ray tracing is available, BLAS builds are submitted as separate queue submissions
/// (one per new mesh). After all builds, waits for those submissions to complete so the
/// TLAS build in the same frame (`update_tlas`) can safely reference the BLASes.
pub(super) fn ensure_mesh_buffers(
    gpu: &mut crate::gpu::GpuState,
    session: &crate::session::Session,
    draw_batches: &[SpaceDrawBatch],
) {
    let mesh_assets = session.asset_registry();
    let mut built_any_blas = false;
    for batch in draw_batches {
        for d in &batch.draws {
            if d.mesh_asset_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                continue;
            };
            if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                continue;
            }
            if !gpu.mesh_buffer_cache.contains_key(&d.mesh_asset_id) {
                let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                let stride = if stride > 0 {
                    stride
                } else {
                    crate::gpu::compute_vertex_stride_from_mesh(mesh)
                };
                let ray_tracing = gpu.ray_tracing_available;
                if let Some(b) =
                    crate::gpu::create_mesh_buffers(&gpu.device, mesh, stride, ray_tracing)
                {
                    gpu.mesh_buffer_cache.insert(d.mesh_asset_id, b.clone());
                    if let Some(ref mut accel) = gpu.accel_cache
                        && let Some(blas) =
                            crate::gpu::build_blas_for_mesh(&gpu.device, &gpu.queue, mesh, &b)
                    {
                        accel.insert(d.mesh_asset_id, blas);
                        built_any_blas = true;
                    }
                }
            }
        }
    }

    // Each build_blas_for_mesh call submits a separate queue submission. The TLAS build
    // (update_tlas) in the same frame records into the main encoder and references these
    // BLASes. Without waiting, the GPU may still be executing the BLAS build submissions
    // when the TLAS build tries to read them, causing a GPU fault / TDR crash on large scenes.
    if built_any_blas {
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }
}
