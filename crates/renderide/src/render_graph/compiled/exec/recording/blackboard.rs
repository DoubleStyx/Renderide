//! Blackboard validation and seeding helpers for command recording.

use super::super::super::super::blackboard::Blackboard;
use super::super::super::super::context::GraphResolvedResources;
use super::super::super::super::error::GraphExecuteError;
use super::super::super::CompiledRenderGraph;
use super::super::super::helpers;
use crate::graph_inputs::MsaaViewsSlot;

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

    /// Builds the per-view [`Blackboard`] seeded with graph-owned per-view resources.
    pub(super) fn build_per_view_blackboard(
        &self,
        frame_params: &crate::graph_inputs::GraphPassFrame<'_>,
        graph_resources: &GraphResolvedResources,
        initial_blackboard: Blackboard,
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
        view_blackboard.extend(graph_blackboard);
        view_blackboard
    }

    /// Builds the frame-global [`Blackboard`] seeded before frame-global passes record.
    pub(super) fn build_frame_global_blackboard() -> Blackboard {
        profiling::scope!("graph::frame_global::build_blackboard");
        Blackboard::new()
    }
}
