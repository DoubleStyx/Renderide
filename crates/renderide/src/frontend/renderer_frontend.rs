//! [`RendererFrontend`] implementation.

use std::time::Instant;

use crate::connection::{ConnectionParams, InitError};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    InputState, OutputState, ReflectionProbeChangeRenderResult, RenderDecouplingConfig,
    RendererCommand, RendererInitData, VideoTextureClockErrorState,
};

use super::decoupling::{DecouplingActivationDecision, DecouplingState, DecouplingSubmitDecision};
use super::frame_start_performance::FrameStartPerformanceState;
use super::init_state::InitState;
use super::lockstep_state::LockstepState;
use super::output_policy::HostOutputPolicy;
use super::session::FrontendSession;
use super::transport::FrontendTransport;

/// IPC, shared memory, init sequence, lock-step, and host output state.
///
/// The facade owns no GPU pools or scene graph. Its fields are split by domain so pure transition
/// logic (init routing, begin-frame gating, decoupling, performance, output policy) stays separate
/// from side-effect adapters such as queue sends and shared-memory access.
pub struct RendererFrontend {
    transport: FrontendTransport,
    session: FrontendSession,
    lockstep: LockstepState,
    performance: FrameStartPerformanceState,
    output_policy: HostOutputPolicy,
    decoupling: DecouplingState,
}

impl RendererFrontend {
    /// Builds frontend state; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(params: Option<ConnectionParams>) -> Self {
        let standalone = params.is_none();
        Self {
            transport: FrontendTransport::new(params),
            session: FrontendSession::new(standalone),
            lockstep: LockstepState::new(standalone),
            performance: FrameStartPerformanceState::default(),
            output_policy: HostOutputPolicy::default(),
            decoupling: DecouplingState::default(),
        }
    }

    /// Lock-step: last host frame index echoed in outgoing [`crate::shared::FrameStartData`].
    pub fn last_frame_index(&self) -> i32 {
        self.lockstep.last_frame_index()
    }

    /// Whether the last [`crate::shared::FrameSubmitData`] was applied and another begin-frame may follow.
    pub fn last_frame_data_processed(&self) -> bool {
        self.lockstep.last_frame_data_processed()
    }

    /// Host requested an orderly renderer exit.
    pub fn shutdown_requested(&self) -> bool {
        self.session.shutdown_requested()
    }

    /// Records a host shutdown request.
    pub fn set_shutdown_requested(&mut self, value: bool) {
        self.session.set_shutdown_requested(value);
    }

    /// Unrecoverable IPC/init ordering error; stops begin-frame until reset.
    pub fn fatal_error(&self) -> bool {
        self.session.fatal_error()
    }

    /// Marks a fatal IPC/init error.
    pub fn set_fatal_error(&mut self, value: bool) {
        self.session.set_fatal_error(value);
    }

    /// Current host/renderer init handshake phase.
    pub fn init_state(&self) -> InitState {
        self.session.init_state()
    }

    /// Updates the init handshake phase.
    pub fn set_init_state(&mut self, state: InitState) {
        self.session.set_init_state(state);
    }

    /// Host [`RendererInitData`] waiting to be consumed after the SHM accessor is ready.
    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.session.pending_init()
    }

    /// Stores init payload until the runtime attaches shared memory and finalizes setup.
    pub fn set_pending_init(&mut self, data: RendererInitData) {
        self.session.set_pending_init(data);
    }

    /// Removes and returns pending init data once the consumer is ready.
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.session.take_pending_init()
    }

    /// Large-payload shared-memory accessor when the host mapped views are available.
    pub fn shared_memory(&self) -> Option<&SharedMemoryAccessor> {
        self.transport.shared_memory()
    }

    /// Mutable shared-memory accessor for mesh/texture uploads.
    pub fn shared_memory_mut(&mut self) -> Option<&mut SharedMemoryAccessor> {
        self.transport.shared_memory_mut()
    }

    /// Installs the SHM accessor produced after init handshake mapping.
    pub fn set_shared_memory(&mut self, shm: SharedMemoryAccessor) {
        self.transport.set_shared_memory(shm);
    }

    /// Mutable reference to the dual-queue IPC when connected.
    pub fn ipc_mut(&mut self) -> Option<&mut DualQueueIpc> {
        self.transport.ipc_mut()
    }

    /// Primary/background command queues when IPC is connected.
    pub fn ipc(&self) -> Option<&DualQueueIpc> {
        self.transport.ipc()
    }

    /// Disjoint mutable handles for backends that need both shared memory and IPC in one call.
    pub fn transport_pair_mut(
        &mut self,
    ) -> (Option<&mut SharedMemoryAccessor>, Option<&mut DualQueueIpc>) {
        self.transport.pair_mut()
    }

    /// Opens Primary/Background queues when connection parameters were provided at construction.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.transport.connect_ipc()
    }

    /// Whether [`Self::connect_ipc`] successfully opened the host queues.
    pub fn is_ipc_connected(&self) -> bool {
        self.transport.is_ipc_connected()
    }

    /// Clears per-tick outbound IPC drop flags on the dual queue.
    pub fn reset_ipc_outbound_drop_tick_flags(&mut self) {
        self.transport.reset_outbound_drop_tick_flags();
    }

    /// Whether any primary outbound send failed since the last drop-flag reset.
    pub fn ipc_outbound_primary_drop_this_tick(&self) -> bool {
        self.transport.outbound_primary_drop_this_tick()
    }

    /// Whether any background outbound send failed since the last drop-flag reset.
    pub fn ipc_outbound_background_drop_this_tick(&self) -> bool {
        self.transport.outbound_background_drop_this_tick()
    }

    /// Current consecutive outbound drop streaks per channel.
    pub fn ipc_consecutive_outbound_drop_streaks(&self) -> (u32, u32) {
        self.transport.consecutive_outbound_drop_streaks()
    }

    /// Records wall-clock spacing for FPS / [`crate::shared::PerformanceState`] before lock-step send.
    pub fn on_tick_frame_wall_clock(&mut self, now: Instant) {
        self.performance.on_tick_frame_wall_clock(now);
    }

    /// Stores the most recently completed GPU submit-to-idle interval for the next frame-start.
    pub fn set_perf_last_render_time_seconds(&mut self, render_time_seconds: Option<f32>) {
        self.performance
            .set_last_render_time_seconds(render_time_seconds);
    }

    /// Poll and sort commands by lifecycle priority.
    pub fn poll_commands(&mut self) -> Vec<RendererCommand> {
        self.transport.poll_commands()
    }

    /// Returns an empty command batch so its allocation is retained for the next poll.
    pub fn recycle_command_batch(&mut self, batch: Vec<RendererCommand>) {
        self.transport.recycle_command_batch(batch);
    }

    /// Whether a [`crate::shared::FrameStartData`] should be sent this tick.
    pub fn should_send_begin_frame(&self) -> bool {
        self.lockstep
            .begin_frame_decision(
                self.session.init_state().is_finalized(),
                self.session.fatal_error(),
                self.transport.is_ipc_connected(),
            )
            .is_allowed()
    }

    /// Whether the renderer is waiting for the host's next [`crate::shared::FrameSubmitData`].
    pub fn awaiting_frame_submit(&self) -> bool {
        self.lockstep.awaiting_submit()
    }

    /// Appends reflection-probe render completion rows for the next outgoing frame-start.
    pub fn enqueue_rendered_reflection_probes(
        &mut self,
        probes: impl IntoIterator<Item = ReflectionProbeChangeRenderResult>,
    ) {
        self.lockstep.enqueue_rendered_reflection_probes(probes);
    }

    /// Appends video texture clock-error samples for the next outgoing frame-start.
    pub fn enqueue_video_clock_errors(
        &mut self,
        errors: impl IntoIterator<Item = VideoTextureClockErrorState>,
    ) {
        self.lockstep.enqueue_video_clock_errors(errors);
    }

    /// Lock-step begin-frame: sends frame-start data with `inputs` when allowed.
    pub fn pre_frame(&mut self, inputs: InputState) {
        profiling::scope!("frontend::pre_frame_send");
        if !self.should_send_begin_frame() {
            return;
        }

        let performance = self.performance.step_for_frame_start();
        let (frame_start, commit) = self.lockstep.build_frame_start(inputs, performance);
        if let Some(ipc) = self.transport.ipc_mut()
            && !ipc.send_primary(RendererCommand::FrameStartData(frame_start))
        {
            logger::warn!(
                "IPC primary queue full: FrameStartData not sent; will retry on the next tick"
            );
            return;
        }
        self.lockstep.commit_begin_frame_sent(commit);
        self.decoupling.record_frame_start_sent(Instant::now());
    }

    /// Host wants relative mouse mode; merged into [`crate::shared::MouseState::is_active`].
    pub fn host_cursor_lock_requested(&self) -> bool {
        self.output_policy.cursor_lock_requested()
    }

    /// Updates cursor/window policy from a frame submit.
    pub fn apply_frame_submit_output(&mut self, output: Option<OutputState>) {
        self.output_policy.apply_frame_submit_output(output);
    }

    /// Last [`OutputState`] from a frame submit.
    pub fn last_output_state(&self) -> Option<&OutputState> {
        self.output_policy.last_output_state()
    }

    /// Takes the last one-shot [`OutputState`] so the winit layer can apply it once.
    pub fn take_pending_output_state(&mut self) -> Option<OutputState> {
        self.output_policy.take_pending_output_state()
    }

    /// Read-only handle to the host-driven decoupling state.
    pub fn decoupling_state(&self) -> &DecouplingState {
        &self.decoupling
    }

    /// Whether the renderer is currently running decoupled from host lock-step.
    pub fn is_decoupled(&self) -> bool {
        self.decoupling.is_active()
    }

    /// Replaces renderer-side decoupling thresholds with the host's config.
    pub fn set_decoupling_config(&mut self, cfg: RenderDecouplingConfig) {
        self.decoupling.apply_config(&cfg);
    }

    /// Per-tick decoupling activation check.
    pub fn update_decoupling_activation(&mut self, now: Instant) {
        let decision = self
            .decoupling
            .update_activation_for_tick(now, self.lockstep.awaiting_submit());
        if decision == DecouplingActivationDecision::Activate {
            logger::info!(
                "render decoupling activated: last_frame_index={} threshold_s={:.4} asset_budget_ms={}",
                self.lockstep.last_frame_index(),
                self.decoupling.activate_interval_seconds(),
                self.decoupling.effective_asset_integration_budget_ms(1),
            );
        }
    }

    /// Increments the renderer-tick counter feeding frame-start performance data.
    pub fn note_render_tick_complete(&mut self) {
        self.performance.note_render_tick_complete();
    }

    /// Updates lock-step state after the host submits a frame.
    pub fn note_frame_submit_processed(&mut self, frame_index: i32) {
        self.lockstep.note_frame_submit_processed(frame_index);
        let decision = self.decoupling.record_frame_submit_received(Instant::now());
        match decision {
            DecouplingSubmitDecision::Recouple => {
                logger::info!(
                    "render decoupling recoupled: frame_index={} stable_frame_count={} last_submit_ms={:.3}",
                    frame_index,
                    self.decoupling.recouple_frame_count(),
                    self.decoupling
                        .last_frame_begin_to_submit()
                        .map_or(-1.0, |duration| duration.as_secs_f64() * 1000.0),
                );
            }
            DecouplingSubmitDecision::ResetProgress => {
                logger::trace!(
                    "render decoupling recouple progress reset: frame_index={} last_submit_ms={:.3} threshold_s={:.4}",
                    frame_index,
                    self.decoupling
                        .last_frame_begin_to_submit()
                        .map_or(-1.0, |duration| duration.as_secs_f64() * 1000.0),
                    self.decoupling.activate_interval_seconds(),
                );
            }
            DecouplingSubmitDecision::AdvanceProgress(progress) => {
                logger::trace!(
                    "render decoupling recouple progress: frame_index={} progress={}/{} last_submit_ms={:.3}",
                    frame_index,
                    progress,
                    self.decoupling.recouple_frame_count(),
                    self.decoupling
                        .last_frame_begin_to_submit()
                        .map_or(-1.0, |duration| duration.as_secs_f64() * 1000.0),
                );
            }
            DecouplingSubmitDecision::Hold => {}
        }
    }

    /// Marks init received after `renderer_init_data`.
    pub fn on_init_received(&mut self) {
        self.session.mark_init_received();
        self.lockstep.mark_init_received();
    }
}
