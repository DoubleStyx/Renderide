//! Command-recording path selection for compiled graph execution.

use crate::cpu_parallelism::ParallelAdmission;
use crate::render_graph::pass::PassPhase;

use super::{
    CompiledRenderGraph, FrameView, FrameViewTarget, GraphCommandRecordingPath, PerViewWorkItem,
};

/// Command-recording strategy and parallelism metadata for one frame.
pub(in crate::render_graph::compiled::exec) struct GraphCommandRecordingPlan {
    /// Selected command-buffer recording path.
    pub(in crate::render_graph::compiled::exec) path: GraphCommandRecordingPath,
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
        GraphCommandRecordingPlan {
            path: select_graph_command_recording_path(
                views.len(),
                single_view_targets_swapchain(views),
                per_view_record_admission,
                self.schedule
                    .recording_plan
                    .phase_has_parallel_batches(PassPhase::PerView),
            ),
            estimated_per_view_record_work,
            per_view_record_admission,
        }
    }
}

fn single_view_targets_swapchain(views: &[FrameView<'_>]) -> bool {
    views.len() == 1 && matches!(&views[0].target, FrameViewTarget::Swapchain)
}

fn select_graph_command_recording_path(
    view_count: usize,
    single_view_targets_swapchain: bool,
    per_view_admission: ParallelAdmission,
    has_parallel_per_view_batches: bool,
) -> GraphCommandRecordingPath {
    profiling::scope!("graph::recording_path_selection");
    if view_count == 1
        && single_view_targets_swapchain
        && !per_view_admission.is_parallel()
        && !has_parallel_per_view_batches
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
            select_graph_command_recording_path(1, true, ParallelAdmission::Serial, false),
            GraphCommandRecordingPath::SingleSwapchainEncoder
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_multi_view() {
        assert_eq!(
            select_graph_command_recording_path(2, false, ParallelAdmission::Serial, false),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_non_swapchain_view() {
        assert_eq!(
            select_graph_command_recording_path(1, false, ParallelAdmission::Serial, false),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_rayon_admitted_work() {
        assert_eq!(
            select_graph_command_recording_path(
                1,
                true,
                ParallelAdmission::Parallel { chunk_size: 1 },
                false
            ),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }

    #[test]
    fn graph_recording_path_uses_standard_path_for_scheduler_parallel_work() {
        assert_eq!(
            select_graph_command_recording_path(1, true, ParallelAdmission::Serial, true),
            GraphCommandRecordingPath::StandardCommandBuffers
        );
    }
}
