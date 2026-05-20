//! Blackboard validation and seeding helpers for command recording.

use super::super::super::super::blackboard::Blackboard;
use super::super::super::super::context::GraphResolvedResources;
use super::super::super::super::error::GraphExecuteError;
use super::super::super::super::frame_params::{
    MsaaViewsSlot, PerViewFramePlan, PerViewFramePlanSlot,
};
use super::super::super::CompiledRenderGraph;
use super::super::super::helpers;
use super::super::types::PerViewRecordShared;
use crate::render_graph::post_process_settings::{
    AutoExposureSettingsSlot, AutoExposureSettingsValue, BloomSettingsSlot, BloomSettingsValue,
    GtaoSettingsSlot, GtaoSettingsValue, MotionBlurSettingsSlot, MotionBlurSettingsValue,
};

impl CompiledRenderGraph {
    /// Validates declared blackboard inputs immediately before one pass records.
    pub(super) fn validate_blackboard_inputs(
        &self,
        pass_idx: usize,
        pass_name: &str,
        blackboard: &Blackboard,
    ) -> Result<(), GraphExecuteError> {
        if !self.validation_mode.enabled() {
            return Ok(());
        }
        let Some(info) = self.pass_info.get(pass_idx) else {
            return Ok(());
        };
        for access in &info.blackboard_accesses {
            if !access.kind.requires_value() || blackboard.contains_type_id(access.slot.type_id) {
                continue;
            }
            if self.validation_mode.is_strict() {
                return Err(GraphExecuteError::MissingBlackboardSlot {
                    pass: pass_name.to_owned(),
                    slot: access.slot.type_name,
                });
            }
            logger::warn!(
                "render graph validation: pass `{pass_name}` requires blackboard slot `{}` but it was not present",
                access.slot.type_name
            );
        }
        Ok(())
    }

    /// Builds the per-view [`Blackboard`] seeded with MSAA views and preplanned frame data.
    pub(super) fn build_per_view_blackboard(
        &self,
        frame_params: &crate::render_graph::frame_params::GraphPassFrame<'_>,
        graph_resources: &GraphResolvedResources,
        initial_blackboard: Blackboard,
        per_view_frame_bg_and_buf: (std::sync::Arc<wgpu::BindGroup>, wgpu::Buffer),
        view_idx: usize,
    ) -> Blackboard {
        profiling::scope!("graph::per_view::build_blackboard");
        let mut view_blackboard = initial_blackboard;
        let mut graph_blackboard = Blackboard::new();
        if let Some(msaa_views) = helpers::resolve_forward_msaa_views_from_graph_resources(
            frame_params,
            graph_resources,
            self.main_graph_msaa_transient_handles,
        ) {
            graph_blackboard.insert::<MsaaViewsSlot>(msaa_views);
        }
        let (frame_bg, frame_buf) = per_view_frame_bg_and_buf;
        graph_blackboard.insert::<PerViewFramePlanSlot>(PerViewFramePlan {
            frame_bind_group: frame_bg,
            frame_uniform_buffer: frame_buf,
            view_idx,
        });
        view_blackboard.extend(graph_blackboard);
        view_blackboard
    }

    /// Seeds live post-processing settings into one per-view blackboard.
    pub(super) fn seed_live_post_process_settings(
        blackboard: &mut Blackboard,
        shared: &PerViewRecordShared<'_>,
        view_id: crate::camera::ViewId,
    ) {
        blackboard.insert::<GtaoSettingsSlot>(GtaoSettingsValue(shared.live_gtao_settings));
        blackboard.insert::<BloomSettingsSlot>(BloomSettingsValue(shared.live_bloom_settings));
        blackboard.insert::<MotionBlurSettingsSlot>(MotionBlurSettingsValue(
            shared.live_motion_blur_settings,
        ));
        blackboard.insert::<AutoExposureSettingsSlot>(AutoExposureSettingsValue::for_view(
            shared.live_auto_exposure_settings,
            shared.wall_frame_delta_seconds,
            view_id,
        ));
    }

    /// Builds the frame-global [`Blackboard`] seeded before frame-global passes record.
    pub(super) fn build_frame_global_blackboard() -> Blackboard {
        profiling::scope!("graph::frame_global::build_blackboard");
        Blackboard::new()
    }
}
