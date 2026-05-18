//! Runtime-owned per-tick scratch and phase gates.

use crate::scene::{ReflectionProbeOnChangesRenderRequest, RenderSpaceId};
use crate::shared::{CameraRenderTask, ReflectionProbeRenderResult, ReflectionProbeRenderTask};

use crate::runtime::offscreen_tasks::reflection_probe::{
    ActiveOnChangesReflectionProbeCapture, ActiveRealtimeReflectionProbeCapture,
};

/// Reflection-probe bake task plus the render space that carried it.
#[derive(Clone, Debug)]
pub(in crate::runtime) struct QueuedReflectionProbeRenderTask {
    /// Host render space containing the reflection probe.
    pub(in crate::runtime) render_space_id: RenderSpaceId,
    /// Host bake task payload.
    pub(in crate::runtime) task: ReflectionProbeRenderTask,
}

/// Per-tick gates and reusable view-planning scratch.
pub(in crate::runtime) struct RuntimeTickState {
    /// Set when asset integration completed for the current winit tick.
    did_integrate_this_tick: bool,
    /// Reusable per-frame scratch for secondary render-texture view collection.
    pub(in crate::runtime) secondary_view_tasks_scratch: Vec<(RenderSpaceId, f32, usize)>,
    /// Host camera readback tasks waiting for a GPU context before the next begin-frame send.
    pub(in crate::runtime) pending_camera_render_tasks: Vec<CameraRenderTask>,
    /// Host reflection-probe bake tasks waiting for a GPU context before the next begin-frame send.
    pub(in crate::runtime) pending_reflection_probe_render_tasks:
        Vec<QueuedReflectionProbeRenderTask>,
    /// Reflection-probe bake results waiting for the background IPC queue to accept them.
    pub(in crate::runtime) pending_reflection_probe_render_results:
        Vec<ReflectionProbeRenderResult>,
    /// OnChanges reflection-probe capture requests waiting for GPU processing.
    pub(in crate::runtime) pending_onchanges_reflection_probe_requests:
        Vec<ReflectionProbeOnChangesRenderRequest>,
    /// OnChanges reflection-probe captures that may span multiple ticks.
    pub(in crate::runtime) active_onchanges_reflection_probe_captures:
        Vec<ActiveOnChangesReflectionProbeCapture>,
    /// Next renderer-side OnChanges cubemap capture generation.
    pub(in crate::runtime) next_onchanges_reflection_probe_generation: u64,
    /// Realtime reflection-probe captures that may span multiple ticks.
    pub(in crate::runtime) active_realtime_reflection_probe_captures:
        Vec<ActiveRealtimeReflectionProbeCapture>,
    /// Next renderer-side realtime cubemap capture generation.
    pub(in crate::runtime) next_realtime_reflection_probe_generation: u64,
}

impl RuntimeTickState {
    /// Creates empty tick state.
    pub(in crate::runtime) fn new() -> Self {
        Self {
            did_integrate_this_tick: false,
            secondary_view_tasks_scratch: Vec::new(),
            pending_camera_render_tasks: Vec::new(),
            pending_reflection_probe_render_tasks: Vec::new(),
            pending_reflection_probe_render_results: Vec::new(),
            pending_onchanges_reflection_probe_requests: Vec::new(),
            active_onchanges_reflection_probe_captures: Vec::new(),
            next_onchanges_reflection_probe_generation: 1,
            active_realtime_reflection_probe_captures: Vec::new(),
            next_realtime_reflection_probe_generation: 1,
        }
    }

    /// Clears once-per-tick gates at the start of a new winit tick.
    pub(in crate::runtime) fn reset_for_tick(&mut self) {
        self.did_integrate_this_tick = false;
    }

    /// Whether asset integration already ran this tick.
    pub(in crate::runtime) fn did_integrate_assets_this_tick(&self) -> bool {
        self.did_integrate_this_tick
    }

    /// Marks asset integration as completed for this tick.
    pub(in crate::runtime) fn mark_integrated_assets_this_tick(&mut self) {
        self.did_integrate_this_tick = true;
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeTickState;

    #[test]
    fn asset_integration_gate_resets_per_tick() {
        let mut state = RuntimeTickState::new();

        assert!(!state.did_integrate_assets_this_tick());
        state.mark_integrated_assets_this_tick();
        assert!(state.did_integrate_assets_this_tick());
        state.reset_for_tick();
        assert!(!state.did_integrate_assets_this_tick());
    }
}
