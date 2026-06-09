//! Command-recording path selection for compiled graph execution.

use crate::cpu_parallelism::ParallelAdmission;
use crate::render_graph::pass::PassPhase;

use super::{
    CompiledRenderGraph, FrameView, FrameViewTarget, GraphCommandRecordingPath, PerViewWorkItem,
};

/// Command-recording strategy and parallelism metadata for one frame.
#[derive(Clone, Copy)]
pub(in crate::render_graph::compiled::exec) struct GraphCommandRecordingPlan {
    /// Selected command-buffer recording path.
    pub(in crate::render_graph::compiled::exec) path: GraphCommandRecordingPath,
    /// Selected parallelism strategy for per-view command recording.
    pub(in crate::render_graph::compiled::exec) strategy: GraphCommandRecordingStrategy,
    /// Estimated draw-equivalent work used by command-recording diagnostics.
    pub(in crate::render_graph::compiled::exec) estimated_per_view_record_work: usize,
    /// Rayon admission decision for per-view command recording.
    pub(in crate::render_graph::compiled::exec) per_view_record_admission: ParallelAdmission,
}

impl CompiledRenderGraph {
    /// Selects the command-recording path and captures its admission metrics.
    pub(in crate::render_graph::compiled::exec) fn graph_command_recording_plan(
        &self,
        views: &[FrameView<'_>],
        per_view_work_items: &[PerViewWorkItem],
    ) -> GraphCommandRecordingPlan {
        let (estimated_per_view_record_work, per_view_record_admission) =
            self.per_view_record_admission_for_work_items(per_view_work_items, views.len());
        let strategy = select_graph_command_recording_strategy(
            per_view_record_admission,
            self.schedule
                .recording_plan
                .phase_has_parallel_batches(PassPhase::PerView),
        );
        GraphCommandRecordingPlan {
            path: select_graph_command_recording_path(
                views.len(),
                single_view_targets_swapchain(views),
                strategy,
            ),
            strategy,
            estimated_per_view_record_work,
            per_view_record_admission,
        }
    }
}

/// Frame-level command-recording parallelism choice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::render_graph::compiled::exec) enum GraphCommandRecordingStrategy {
    /// Record all views and pass units serially.
    Serial,
    /// Record independent views across Rayon workers.
    AcrossViewsParallel,
    /// Record one view at a time, with scheduler-admitted pass units parallelized inside a view.
    InViewParallel,
}

fn single_view_targets_swapchain(views: &[FrameView<'_>]) -> bool {
    views.len() == 1 && matches!(&views[0].target, FrameViewTarget::Swapchain)
}

fn select_graph_command_recording_strategy(
    per_view_admission: ParallelAdmission,
    has_parallel_per_view_batches: bool,
) -> GraphCommandRecordingStrategy {
    if per_view_admission.is_parallel() {
        GraphCommandRecordingStrategy::AcrossViewsParallel
    } else if has_parallel_per_view_batches {
        GraphCommandRecordingStrategy::InViewParallel
    } else {
        GraphCommandRecordingStrategy::Serial
    }
}

fn select_graph_command_recording_path(
    view_count: usize,
    single_view_targets_swapchain: bool,
    strategy: GraphCommandRecordingStrategy,
) -> GraphCommandRecordingPath {
    profiling::scope!("graph::recording_path_selection");
    if view_count == 1
        && single_view_targets_swapchain
        && strategy == GraphCommandRecordingStrategy::Serial
    {
        GraphCommandRecordingPath::SingleSwapchainEncoder
    } else {
        GraphCommandRecordingPath::StandardCommandBuffers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_recording_path_selects_single_swapchain_encoder_for_serial_swapchain_view() {
        assert_eq!(
            select_graph_command_recording_path(1, true, GraphCommandRecordingStrategy::Serial),
            GraphCommandRecordingPath::SingleSwapchainEncoder
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_multi_view() {
        assert_eq!(
            select_graph_command_recording_path(2, false, GraphCommandRecordingStrategy::Serial),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_non_swapchain_view() {
        assert_eq!(
            select_graph_command_recording_path(1, false, GraphCommandRecordingStrategy::Serial),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_rayon_admitted_work() {
        assert_eq!(
            select_graph_command_recording_path(
                1,
                true,
                GraphCommandRecordingStrategy::AcrossViewsParallel
            ),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_scheduler_parallel_work() {
        assert_eq!(
            select_graph_command_recording_path(
                1,
                true,
                GraphCommandRecordingStrategy::InViewParallel
            ),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_strategy_prefers_across_view_parallelism() {
        assert_eq!(
            select_graph_command_recording_strategy(
                ParallelAdmission::Parallel { chunk_size: 1 },
                true
            ),
            GraphCommandRecordingStrategy::AcrossViewsParallel
        );
    }

    #[test]
    fn graph_recording_strategy_uses_in_view_parallelism_only_when_views_are_serial() {
        assert_eq!(
            select_graph_command_recording_strategy(ParallelAdmission::Serial, true),
            GraphCommandRecordingStrategy::InViewParallel
        );
    }
}
