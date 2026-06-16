//! Command-recording path selection for compiled graph execution.

use crate::config::CommandRecordingMode;
use crate::cpu_parallelism::{FrameParallelPolicy, ParallelAdmission};
use crate::render_graph::pass::PassPhase;

use super::{
    CompiledRenderGraph, FrameView, FrameViewTarget, GraphCommandRecordingPath, PerViewWorkItem,
};

const IN_VIEW_RECORD_PARALLEL_MIN_WORK: usize = 512;

/// Command-recording strategy and parallelism metadata for one frame.
#[derive(Clone, Copy)]
pub(in crate::render_graph::compiled::exec) struct GraphCommandRecordingPlan {
    /// Selected command-buffer recording path.
    pub(in crate::render_graph::compiled::exec) path: GraphCommandRecordingPath,
    /// Selected parallelism strategy for per-view command recording.
    pub(in crate::render_graph::compiled::exec) strategy: GraphCommandRecordingStrategy,
    /// Configured command-recording mode that selected this plan.
    pub(in crate::render_graph::compiled::exec) requested_mode: CommandRecordingMode,
    /// Estimated draw count visible to per-view command recording.
    pub(in crate::render_graph::compiled::exec) estimated_per_view_draw_count: usize,
    /// Estimated draw-equivalent work used by command-recording diagnostics.
    pub(in crate::render_graph::compiled::exec) estimated_per_view_record_work: usize,
    /// Automatic Rayon admission decision before any profiling override is applied.
    pub(in crate::render_graph::compiled::exec) auto_per_view_record_admission: ParallelAdmission,
    /// Effective Rayon admission decision for per-view command recording.
    pub(in crate::render_graph::compiled::exec) per_view_record_admission: ParallelAdmission,
    /// Automatic scheduler admission decision for splitting work inside one view.
    pub(in crate::render_graph::compiled::exec) auto_in_view_record_admitted: bool,
    /// Effective scheduler admission decision for splitting work inside one view.
    pub(in crate::render_graph::compiled::exec) in_view_record_admitted: bool,
    /// Whether the single-swapchain encoder path was selected or why it was unavailable.
    pub(in crate::render_graph::compiled::exec) single_swapchain_encoder_status:
        SingleSwapchainEncoderStatus,
}

impl CompiledRenderGraph {
    /// Selects the command-recording path and captures its admission metrics.
    pub(in crate::render_graph::compiled::exec) fn graph_command_recording_plan(
        &self,
        views: &[FrameView<'_>],
        per_view_work_items: &[PerViewWorkItem],
        requested_mode: CommandRecordingMode,
    ) -> GraphCommandRecordingPlan {
        let (
            estimated_per_view_draw_count,
            estimated_per_view_record_work,
            auto_per_view_record_admission,
        ) = self.per_view_record_admission_for_work_items(per_view_work_items, views.len());
        let per_view_record_admission = effective_per_view_record_admission(
            requested_mode,
            views.len(),
            auto_per_view_record_admission,
        );
        let has_parallel_per_view_batches = self
            .schedule
            .recording_plan
            .phase_has_parallel_batches(PassPhase::PerView);
        let has_split_per_view_batches = self
            .schedule
            .recording_plan
            .phase_batches(PassPhase::PerView)
            .nth(1)
            .is_some()
            || has_parallel_per_view_batches;
        let auto_in_view_record_admitted = auto_in_view_record_admitted(
            FrameParallelPolicy::for_current_thread_pool(),
            views.len(),
            estimated_per_view_record_work,
            has_split_per_view_batches,
        );
        let in_view_record_admitted = effective_in_view_record_admitted(
            requested_mode,
            auto_in_view_record_admitted,
            has_split_per_view_batches,
        );
        let strategy = select_graph_command_recording_strategy(
            requested_mode,
            views.len(),
            per_view_record_admission,
            in_view_record_admitted,
            has_split_per_view_batches,
        );
        let single_swapchain_encoder_status = single_swapchain_encoder_status(
            views.len(),
            single_view_targets_swapchain(views),
            strategy,
        );
        GraphCommandRecordingPlan {
            path: single_swapchain_encoder_status.path(),
            strategy,
            requested_mode,
            estimated_per_view_draw_count,
            estimated_per_view_record_work,
            auto_per_view_record_admission,
            per_view_record_admission,
            auto_in_view_record_admitted,
            in_view_record_admitted,
            single_swapchain_encoder_status,
        }
    }
}

/// Single-swapchain encoder selection outcome for command recording diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::render_graph::compiled::exec) enum SingleSwapchainEncoderStatus {
    /// The frame has more than one graph view.
    MultipleViews,
    /// The single view does not target the swapchain.
    NonSwapchainTarget,
    /// The selected strategy requires phase-specific command buffers.
    SplitRecordingStrategy,
    /// Frame-global work had already been split into multiple encoders.
    FrameGlobalSplitWorkload,
    /// The single-swapchain encoder path is active.
    Active,
}

impl SingleSwapchainEncoderStatus {
    /// Numeric value used by Tracy plots and compact diagnostics.
    pub(in crate::render_graph::compiled::exec) const fn as_plot_value(self) -> u64 {
        match self {
            Self::MultipleViews => 0,
            Self::NonSwapchainTarget => 1,
            Self::SplitRecordingStrategy => 2,
            Self::FrameGlobalSplitWorkload => 3,
            Self::Active => 4,
        }
    }

    /// Returns the command-buffer path implied by this status.
    const fn path(self) -> GraphCommandRecordingPath {
        match self {
            Self::Active => GraphCommandRecordingPath::SingleSwapchainEncoder,
            Self::MultipleViews
            | Self::NonSwapchainTarget
            | Self::SplitRecordingStrategy
            | Self::FrameGlobalSplitWorkload => GraphCommandRecordingPath::StandardCommandBuffers,
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
    /// Record one view at a time through scheduler-sized command buffers.
    InViewScheduler,
    /// Record independent views across Rayon workers and split each view through the scheduler.
    AcrossViewsWithInViewScheduler,
}

impl GraphCommandRecordingStrategy {
    /// Numeric value used by Tracy plots and compact diagnostics.
    pub(in crate::render_graph::compiled::exec) const fn as_plot_value(self) -> u64 {
        match self {
            Self::Serial => 0,
            Self::AcrossViewsParallel => 1,
            Self::InViewScheduler => 2,
            Self::AcrossViewsWithInViewScheduler => 3,
        }
    }

    /// Returns whether the strategy records views on independent Rayon workers.
    pub(in crate::render_graph::compiled::exec) const fn uses_across_view_parallelism(
        self,
    ) -> bool {
        matches!(
            self,
            Self::AcrossViewsParallel | Self::AcrossViewsWithInViewScheduler
        )
    }

    /// Returns whether each view should record through scheduler-sized command buffers.
    pub(in crate::render_graph::compiled::exec) const fn uses_in_view_scheduler(self) -> bool {
        matches!(
            self,
            Self::InViewScheduler | Self::AcrossViewsWithInViewScheduler
        )
    }

    /// Returns whether scheduler parallel batches should fan out inside one view.
    pub(in crate::render_graph::compiled::exec) const fn allows_in_view_parallel_batches(
        self,
    ) -> bool {
        matches!(self, Self::InViewScheduler)
    }
}

fn single_view_targets_swapchain(views: &[FrameView<'_>]) -> bool {
    views.len() == 1 && matches!(&views[0].target, FrameViewTarget::Swapchain)
}

fn select_graph_command_recording_strategy(
    requested_mode: CommandRecordingMode,
    view_count: usize,
    per_view_admission: ParallelAdmission,
    in_view_record_admitted: bool,
    has_split_per_view_batches: bool,
) -> GraphCommandRecordingStrategy {
    match requested_mode {
        CommandRecordingMode::Auto => match (
            view_count >= 2 && per_view_admission.is_parallel(),
            in_view_record_admitted,
        ) {
            (true, true) => GraphCommandRecordingStrategy::AcrossViewsWithInViewScheduler,
            (true, false) => GraphCommandRecordingStrategy::AcrossViewsParallel,
            (false, true) => GraphCommandRecordingStrategy::InViewScheduler,
            (false, false) => GraphCommandRecordingStrategy::Serial,
        },
        CommandRecordingMode::AcrossViews => {
            if view_count >= 2 && per_view_admission.is_parallel() {
                if in_view_record_admitted {
                    GraphCommandRecordingStrategy::AcrossViewsWithInViewScheduler
                } else {
                    GraphCommandRecordingStrategy::AcrossViewsParallel
                }
            } else if in_view_record_admitted {
                GraphCommandRecordingStrategy::InViewScheduler
            } else {
                GraphCommandRecordingStrategy::Serial
            }
        }
        CommandRecordingMode::Serial => GraphCommandRecordingStrategy::Serial,
        CommandRecordingMode::InView => {
            if has_split_per_view_batches {
                GraphCommandRecordingStrategy::InViewScheduler
            } else {
                GraphCommandRecordingStrategy::Serial
            }
        }
    }
}

fn effective_per_view_record_admission(
    requested_mode: CommandRecordingMode,
    view_count: usize,
    auto_admission: ParallelAdmission,
) -> ParallelAdmission {
    match requested_mode {
        CommandRecordingMode::Auto => auto_admission,
        CommandRecordingMode::AcrossViews if view_count >= 2 => {
            ParallelAdmission::Parallel { chunk_size: 1 }
        }
        CommandRecordingMode::Serial
        | CommandRecordingMode::AcrossViews
        | CommandRecordingMode::InView => ParallelAdmission::Serial,
    }
}

fn effective_in_view_record_admitted(
    requested_mode: CommandRecordingMode,
    auto_admitted: bool,
    has_split_per_view_batches: bool,
) -> bool {
    match requested_mode {
        CommandRecordingMode::Auto | CommandRecordingMode::AcrossViews => auto_admitted,
        CommandRecordingMode::InView => has_split_per_view_batches,
        CommandRecordingMode::Serial => false,
    }
}

fn auto_in_view_record_admitted(
    policy: FrameParallelPolicy,
    view_count: usize,
    estimated_record_work: usize,
    has_split_per_view_batches: bool,
) -> bool {
    if !has_split_per_view_batches || view_count == 0 {
        return false;
    }
    let per_view_work = estimated_record_work.div_ceil(view_count);
    per_view_work >= in_view_record_parallel_min_work(policy)
}

fn in_view_record_parallel_min_work(policy: FrameParallelPolicy) -> usize {
    policy
        .draw_heavy_threshold()
        .max(IN_VIEW_RECORD_PARALLEL_MIN_WORK)
}

fn single_swapchain_encoder_status(
    view_count: usize,
    single_view_targets_swapchain: bool,
    strategy: GraphCommandRecordingStrategy,
) -> SingleSwapchainEncoderStatus {
    profiling::scope!("graph::recording_path_selection");
    if view_count != 1 {
        return SingleSwapchainEncoderStatus::MultipleViews;
    }
    if !single_view_targets_swapchain {
        return SingleSwapchainEncoderStatus::NonSwapchainTarget;
    }
    if strategy != GraphCommandRecordingStrategy::Serial {
        return SingleSwapchainEncoderStatus::SplitRecordingStrategy;
    }
    SingleSwapchainEncoderStatus::Active
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_swapchain_status_selects_single_encoder_for_serial_swapchain_view() {
        assert_eq!(
            single_swapchain_encoder_status(1, true, GraphCommandRecordingStrategy::Serial),
            SingleSwapchainEncoderStatus::Active
        );
    }

    #[test]
    fn single_swapchain_status_reports_multi_view_disable_reason() {
        assert_eq!(
            single_swapchain_encoder_status(2, false, GraphCommandRecordingStrategy::Serial),
            SingleSwapchainEncoderStatus::MultipleViews
        );
    }

    #[test]
    fn single_swapchain_status_reports_non_swapchain_disable_reason() {
        assert_eq!(
            single_swapchain_encoder_status(1, false, GraphCommandRecordingStrategy::Serial),
            SingleSwapchainEncoderStatus::NonSwapchainTarget
        );
    }

    #[test]
    fn single_swapchain_status_reports_split_strategy_disable_reason() {
        assert_eq!(
            single_swapchain_encoder_status(
                1,
                true,
                GraphCommandRecordingStrategy::AcrossViewsWithInViewScheduler
            ),
            SingleSwapchainEncoderStatus::SplitRecordingStrategy
        );
    }

    #[test]
    fn graph_recording_strategy_prefers_auto_across_view_parallelism() {
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::Auto,
                2,
                ParallelAdmission::Parallel { chunk_size: 1 },
                false,
                true
            ),
            GraphCommandRecordingStrategy::AcrossViewsParallel
        );
    }

    #[test]
    fn graph_recording_strategy_splits_heavy_single_view_work() {
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::Auto,
                1,
                ParallelAdmission::Serial,
                true,
                true
            ),
            GraphCommandRecordingStrategy::InViewScheduler
        );
    }

    #[test]
    fn graph_recording_strategy_combines_across_and_in_view_work_for_heavy_multi_view() {
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::Auto,
                2,
                ParallelAdmission::Parallel { chunk_size: 1 },
                true,
                true
            ),
            GraphCommandRecordingStrategy::AcrossViewsWithInViewScheduler
        );
    }

    #[test]
    fn graph_recording_strategy_keeps_light_auto_work_serial() {
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::Auto,
                1,
                ParallelAdmission::Serial,
                false,
                true
            ),
            GraphCommandRecordingStrategy::Serial
        );
    }

    #[test]
    fn graph_recording_strategy_uses_forced_in_view_scheduler_when_available() {
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::InView,
                1,
                ParallelAdmission::Serial,
                false,
                true
            ),
            GraphCommandRecordingStrategy::InViewScheduler
        );
    }

    #[test]
    fn graph_recording_strategy_falls_back_when_forced_in_view_is_unavailable() {
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::InView,
                1,
                ParallelAdmission::Serial,
                true,
                false
            ),
            GraphCommandRecordingStrategy::Serial
        );
    }

    #[test]
    fn graph_recording_strategy_forces_across_view_admission_for_multi_view() {
        let admission = effective_per_view_record_admission(
            CommandRecordingMode::AcrossViews,
            2,
            ParallelAdmission::Serial,
        );

        assert_eq!(admission, ParallelAdmission::Parallel { chunk_size: 1 });
        assert_eq!(
            select_graph_command_recording_strategy(
                CommandRecordingMode::AcrossViews,
                2,
                admission,
                false,
                false
            ),
            GraphCommandRecordingStrategy::AcrossViewsParallel
        );
    }

    #[test]
    fn auto_in_view_record_admission_requires_split_batches() {
        assert!(!auto_in_view_record_admitted(
            FrameParallelPolicy::new(4),
            1,
            usize::MAX,
            false
        ));
    }

    #[test]
    fn auto_in_view_record_admission_uses_per_view_work() {
        assert!(auto_in_view_record_admitted(
            FrameParallelPolicy::new(4),
            2,
            1024,
            true
        ));
        assert!(!auto_in_view_record_admitted(
            FrameParallelPolicy::new(4),
            2,
            1022,
            true
        ));
    }
}
