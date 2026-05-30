//! Explicit CPU render schedule shared by main and offscreen render paths.
//!
//! The schedule is intentionally a small typed orchestration layer, not a dynamic ECS scheduler.
//! It makes the renderer's CPU frame order visible and gives future async work stable phase
//! boundaries without changing the existing render-graph or view-plan contracts.

use crate::backend::RenderBackend;
use crate::camera::ViewId;
use crate::diagnostics::crash_context;
use crate::gpu::GpuContext;
use crate::render_graph::GraphExecuteError;
use crate::scene::SceneCoordinator;
use crate::world_mesh::WorldMeshDrawCollectParallelism;

use super::extract::{ExtractedFrame, PreparedViews};
use super::view_plan::{FrameViewPlan, ViewFamilyPlan};

/// Ordered CPU render phases for every graph-backed render submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) enum CpuRenderPhase {
    /// Extract immutable frame inputs from runtime state.
    Extract,
    /// Prepare material and asset state for this submission.
    AssetPrepare,
    /// Build the ordered view list for this submission.
    ViewPlanning,
    /// Queue visible draw candidates for planned views.
    DrawQueue,
    /// Sort and arrange queued draws into render-phase order.
    Sort,
    /// Prepare frame resources after draw order is known.
    ResourcePrepare,
    /// Record and submit render-graph commands.
    CommandRecord,
    /// Release frame-local or one-shot state.
    Cleanup,
}

impl CpuRenderPhase {
    /// Canonical phase order for one CPU render schedule.
    #[cfg(test)]
    pub(in crate::runtime) const ORDER: [Self; 8] = [
        Self::Extract,
        Self::AssetPrepare,
        Self::ViewPlanning,
        Self::DrawQueue,
        Self::Sort,
        Self::ResourcePrepare,
        Self::CommandRecord,
        Self::Cleanup,
    ];

    fn crash_context_phase(self) -> crash_context::CpuRenderPhase {
        match self {
            Self::Extract => crash_context::CpuRenderPhase::Extract,
            Self::AssetPrepare => crash_context::CpuRenderPhase::AssetPrepare,
            Self::ViewPlanning => crash_context::CpuRenderPhase::ViewPlanning,
            Self::DrawQueue => crash_context::CpuRenderPhase::DrawQueue,
            Self::Sort => crash_context::CpuRenderPhase::Sort,
            Self::ResourcePrepare => crash_context::CpuRenderPhase::ResourcePrepare,
            Self::CommandRecord => crash_context::CpuRenderPhase::CommandRecord,
            Self::Cleanup => crash_context::CpuRenderPhase::Cleanup,
        }
    }
}

/// Render path using the CPU render schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) enum RenderScheduleKind {
    /// Desktop or headless main-view render, including secondary render textures.
    Desktop,
    /// OpenXR HMD multiview render, including secondary render textures.
    Hmd,
    /// VR tick where only secondary render textures are submitted.
    VrSecondariesOnly,
    /// Host camera render task submitted for readback.
    CameraTask,
    /// Camera360 cubemap capture submitted before equirectangular projection.
    Camera360Capture,
    /// Reflection-probe cubemap capture.
    ReflectionProbeCapture,
}

impl RenderScheduleKind {
    /// Stable diagnostic label for this schedule kind.
    pub(in crate::runtime) const fn as_str(self) -> &'static str {
        match self {
            Self::Desktop => "desktop",
            Self::Hmd => "hmd",
            Self::VrSecondariesOnly => "vr-secondaries-only",
            Self::CameraTask => "camera-task",
            Self::Camera360Capture => "camera360-capture",
            Self::ReflectionProbeCapture => "reflection-probe-capture",
        }
    }
}

/// CPU render schedule instance for one graph-backed submission.
#[derive(Clone, Copy, Debug)]
pub(in crate::runtime) struct CpuRenderSchedule {
    kind: RenderScheduleKind,
}

impl CpuRenderSchedule {
    /// Builds a CPU render schedule for one render path.
    pub(in crate::runtime) const fn new(kind: RenderScheduleKind) -> Self {
        Self { kind }
    }

    /// Render path this schedule is executing.
    pub(in crate::runtime) const fn kind(self) -> RenderScheduleKind {
        self.kind
    }

    /// Unity-equivalent mesh LOD bias for this render path.
    pub(in crate::runtime) const fn mesh_lod_bias(self) -> f32 {
        match self.kind {
            RenderScheduleKind::Hmd | RenderScheduleKind::VrSecondariesOnly => 3.8,
            RenderScheduleKind::Desktop
            | RenderScheduleKind::CameraTask
            | RenderScheduleKind::Camera360Capture
            | RenderScheduleKind::ReflectionProbeCapture => 2.0,
        }
    }

    /// Runs `work` under a named CPU render phase.
    pub(in crate::runtime) fn run_phase<T>(
        self,
        phase: CpuRenderPhase,
        work: impl FnOnce() -> T,
    ) -> T {
        let _phase_scope = CpuRenderPhaseScope::new(phase);
        match phase {
            CpuRenderPhase::Extract => {
                profiling::scope!("render_schedule::extract");
                work()
            }
            CpuRenderPhase::AssetPrepare => {
                profiling::scope!("render_schedule::asset_prepare");
                work()
            }
            CpuRenderPhase::ViewPlanning => {
                profiling::scope!("render_schedule::view_planning");
                work()
            }
            CpuRenderPhase::DrawQueue => {
                profiling::scope!("render_schedule::draw_queue");
                work()
            }
            CpuRenderPhase::Sort => {
                profiling::scope!("render_schedule::sort");
                work()
            }
            CpuRenderPhase::ResourcePrepare => {
                profiling::scope!("render_schedule::resource_prepare");
                work()
            }
            CpuRenderPhase::CommandRecord => {
                profiling::scope!("render_schedule::command_record");
                work()
            }
            CpuRenderPhase::Cleanup => {
                profiling::scope!("render_schedule::cleanup");
                work()
            }
        }
    }
}

/// Crash-context guard for the active CPU render phase.
struct CpuRenderPhaseScope;

impl CpuRenderPhaseScope {
    /// Records `phase` until the scope is dropped.
    fn new(phase: CpuRenderPhase) -> Self {
        crash_context::set_cpu_render_phase(phase.crash_context_phase());
        Self
    }
}

impl Drop for CpuRenderPhaseScope {
    fn drop(&mut self) {
        crash_context::set_cpu_render_phase(crash_context::CpuRenderPhase::Unknown);
    }
}

/// Runs global asset/material preparation for a schedule-backed render submission.
pub(in crate::runtime) fn prepare_assets_for_schedule(backend: &mut RenderBackend) {
    backend.sync_material_shader_hot_reload();
    backend.drain_pipeline_build_completions();
}

/// Executes already prepared views through the draw, resource, command, and cleanup phases.
pub(in crate::runtime) fn execute_prepared_views<'a>(
    schedule: CpuRenderSchedule,
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    prepared_views: PreparedViews<'a>,
    inner_parallelism: WorldMeshDrawCollectParallelism,
) -> Result<(), GraphExecuteError> {
    execute_prepared_views_with_cleanup(
        schedule,
        gpu,
        backend,
        scene,
        prepared_views,
        inner_parallelism,
        |_| {},
    )
}

/// Executes one-shot view plans and retires their view-scoped resources during cleanup.
pub(in crate::runtime) fn execute_one_shot_view_plans<'a>(
    schedule: CpuRenderSchedule,
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    plans: Vec<FrameViewPlan<'a>>,
    inner_parallelism: WorldMeshDrawCollectParallelism,
) -> Result<(), GraphExecuteError> {
    let one_shot_views = view_ids_for_plans(&plans);
    schedule.run_phase(CpuRenderPhase::Extract, || {});
    schedule.run_phase(CpuRenderPhase::AssetPrepare, || {
        prepare_assets_for_schedule(backend);
    });
    let prepared_views = schedule.run_phase(CpuRenderPhase::ViewPlanning, || {
        PreparedViews::new(ViewFamilyPlan::new(plans))
    });
    execute_prepared_views_with_cleanup(
        schedule,
        gpu,
        backend,
        scene,
        prepared_views,
        inner_parallelism,
        move |backend| backend.retire_one_shot_views(&one_shot_views),
    )
}

/// Runs the schedule tail for prepared views and always executes the provided cleanup phase.
fn execute_prepared_views_with_cleanup<'a>(
    schedule: CpuRenderSchedule,
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    prepared_views: PreparedViews<'a>,
    inner_parallelism: WorldMeshDrawCollectParallelism,
    cleanup: impl FnOnce(&mut RenderBackend),
) -> Result<(), GraphExecuteError> {
    crash_context::set_prepared_view_count(prepared_views.plans().len());
    if prepared_views.is_empty() {
        logger::trace!(
            "render schedule skipped: kind={} no prepared views",
            schedule.kind().as_str()
        );
        schedule.run_phase(CpuRenderPhase::Cleanup, || cleanup(backend));
        return Ok(());
    }

    let view_draw_preparations = prepared_views
        .plans()
        .iter()
        .map(|plan| (plan.render_context(), plan.shader_permutation()))
        .collect::<Vec<_>>();
    let queued_draws = schedule.run_phase(CpuRenderPhase::DrawQueue, || {
        let shared =
            backend.extract_frame_shared(scene, inner_parallelism, &view_draw_preparations);
        ExtractedFrame::new(prepared_views, shared, schedule.mesh_lod_bias()).queue_draws()
    });
    let prepared_draws = schedule.run_phase(CpuRenderPhase::Sort, || queued_draws.sort_draws());
    let submit_frame = prepared_draws.into_submit_frame();
    schedule.run_phase(CpuRenderPhase::ResourcePrepare, || {
        submit_frame.prepare_resources(scene, backend);
    });
    let result = schedule.run_phase(CpuRenderPhase::CommandRecord, || {
        submit_frame.execute_after_resource_prepare(gpu, scene, backend)
    });
    schedule.run_phase(CpuRenderPhase::Cleanup, || cleanup(backend));
    result
}

/// Returns the one-shot view identifiers owned by `plans`.
fn view_ids_for_plans(plans: &[FrameViewPlan<'_>]) -> Vec<ViewId> {
    plans.iter().map(|plan| plan.view_id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_render_phase_order_is_explicit() {
        assert_eq!(
            CpuRenderPhase::ORDER,
            [
                CpuRenderPhase::Extract,
                CpuRenderPhase::AssetPrepare,
                CpuRenderPhase::ViewPlanning,
                CpuRenderPhase::DrawQueue,
                CpuRenderPhase::Sort,
                CpuRenderPhase::ResourcePrepare,
                CpuRenderPhase::CommandRecord,
                CpuRenderPhase::Cleanup,
            ]
        );
    }

    #[test]
    fn schedule_phase_scope_updates_crash_context() {
        let schedule = CpuRenderSchedule::new(RenderScheduleKind::Desktop);
        let observed = schedule.run_phase(CpuRenderPhase::DrawQueue, || {
            crash_context::snapshot().cpu_render_phase
        });

        assert_eq!(observed, crash_context::CpuRenderPhase::DrawQueue);
        assert_eq!(
            crash_context::snapshot().cpu_render_phase,
            crash_context::CpuRenderPhase::Unknown
        );
    }

    #[test]
    fn mesh_lod_bias_matches_render_schedule_kind() {
        assert_eq!(
            CpuRenderSchedule::new(RenderScheduleKind::Desktop).mesh_lod_bias(),
            2.0
        );
        assert_eq!(
            CpuRenderSchedule::new(RenderScheduleKind::CameraTask).mesh_lod_bias(),
            2.0
        );
        assert_eq!(
            CpuRenderSchedule::new(RenderScheduleKind::Hmd).mesh_lod_bias(),
            3.8
        );
        assert_eq!(
            CpuRenderSchedule::new(RenderScheduleKind::VrSecondariesOnly).mesh_lod_bias(),
            3.8
        );
    }
}
