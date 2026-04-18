//! [`RendererFrontend`] implementation.

use std::time::Instant;

use crate::connection::{ConnectionParams, InitError};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{FrameStartData, InputState, OutputState, RendererCommand, RendererInitData};

use super::begin_frame::begin_frame_allowed;
use super::init_state::InitState;

/// IPC, shared memory, init sequence, and lock-step fields. Does not own GPU pools or scene graph.
pub struct RendererFrontend {
    ipc: Option<DualQueueIpc>,
    params: Option<ConnectionParams>,
    init_state: InitState,
    pending_init: Option<RendererInitData>,
    shared_memory: Option<SharedMemoryAccessor>,
    /// After a successful frame submit application, host may expect another begin-frame.
    last_frame_data_processed: bool,
    last_frame_index: i32,
    sent_bootstrap_frame_start: bool,
    shutdown_requested: bool,
    fatal_error: bool,
    /// Latest host [`OutputState::lock_cursor`] from [`crate::shared::FrameSubmitData`].
    cursor_lock_requested: bool,
    /// Pending window policy from the last frame submit (applied in winit; consumed by the app).
    pending_output_state: Option<OutputState>,
    /// Last non-null [`OutputState`] from the host (retained for per-frame cursor policy).
    last_output_state: Option<OutputState>,
    /// Wall-clock start of the previous [`crate::app::RenderideApp::tick_frame`] (for FPS interval).
    last_tick_wall_start: Option<Instant>,
    /// Microseconds between the last two tick starts; fed into [`crate::frontend::frame_start_performance`].
    wall_interval_us_for_perf: u64,
    /// Total microseconds for the previous completed tick ([`crate::shared::PerformanceState::render_time`]).
    perf_last_total_us: u64,
    /// Exponentially smoothed FPS for optional [`crate::shared::FrameStartData::performance`].
    smoothed_fps: Option<f32>,
    /// Last wall time a non-null performance sample was attached to outgoing `frame_start_data`.
    last_perf_send: Option<Instant>,
}

impl RendererFrontend {
    /// Builds frontend; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(params: Option<ConnectionParams>) -> Self {
        let standalone = params.is_none();
        let init_state = if standalone {
            InitState::Finalized
        } else {
            InitState::default()
        };
        Self {
            ipc: None,
            params,
            init_state,
            pending_init: None,
            shared_memory: None,
            last_frame_data_processed: standalone,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            shutdown_requested: false,
            fatal_error: false,
            cursor_lock_requested: false,
            pending_output_state: None,
            last_output_state: None,
            last_tick_wall_start: None,
            wall_interval_us_for_perf: 0,
            perf_last_total_us: 0,
            smoothed_fps: None,
            last_perf_send: None,
        }
    }

    /// Lock-step: last host frame index echoed in outgoing [`FrameStartData`].
    pub fn last_frame_index(&self) -> i32 {
        self.last_frame_index
    }

    /// Whether the last [`crate::shared::FrameSubmitData`] was applied and another begin-frame may follow.
    pub fn last_frame_data_processed(&self) -> bool {
        self.last_frame_data_processed
    }

    /// Host requested an orderly renderer exit (IPC path).
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    /// Records a host shutdown request ([`RendererCommand::RendererShutdownRequest`] / shutdown).
    pub fn set_shutdown_requested(&mut self, value: bool) {
        self.shutdown_requested = value;
    }

    /// Unrecoverable IPC/init ordering error; stops begin-frame until reset.
    pub fn fatal_error(&self) -> bool {
        self.fatal_error
    }

    /// Marks a fatal IPC/init error (stops lock-step begin-frame).
    pub fn set_fatal_error(&mut self, value: bool) {
        self.fatal_error = value;
    }

    /// Current host/renderer init handshake phase.
    pub fn init_state(&self) -> InitState {
        self.init_state
    }

    /// Updates the init handshake phase (e.g. after processing [`RendererCommand::RendererInitData`]).
    pub fn set_init_state(&mut self, state: InitState) {
        self.init_state = state;
    }

    /// Host [`RendererInitData`] waiting to be consumed after the SHM accessor is ready.
    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.pending_init.as_ref()
    }

    /// Stores init payload until the runtime attaches shared memory and finalizes setup.
    pub fn set_pending_init(&mut self, data: RendererInitData) {
        self.pending_init = Some(data);
    }

    /// Removes and returns pending init data once the consumer is ready.
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.pending_init.take()
    }

    /// Large-payload shared-memory accessor when the host mapped views are available.
    pub fn shared_memory(&self) -> Option<&SharedMemoryAccessor> {
        self.shared_memory.as_ref()
    }

    /// Mutable shared-memory accessor for mesh/texture uploads.
    pub fn shared_memory_mut(&mut self) -> Option<&mut SharedMemoryAccessor> {
        self.shared_memory.as_mut()
    }

    /// Installs the SHM accessor produced after init handshake mapping.
    pub fn set_shared_memory(&mut self, shm: SharedMemoryAccessor) {
        self.shared_memory = Some(shm);
    }

    /// Mutable reference to the dual-queue IPC when connected.
    pub fn ipc_mut(&mut self) -> Option<&mut DualQueueIpc> {
        self.ipc.as_mut()
    }

    /// Primary/background command queues when IPC is connected.
    pub fn ipc(&self) -> Option<&DualQueueIpc> {
        self.ipc.as_ref()
    }

    /// Disjoint mutable handles for backends that need both shared memory and IPC in one call.
    pub fn transport_pair_mut(
        &mut self,
    ) -> (Option<&mut SharedMemoryAccessor>, Option<&mut DualQueueIpc>) {
        (self.shared_memory.as_mut(), self.ipc.as_mut())
    }

    /// Opens Primary/Background queues when connection parameters were provided at construction.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        let Some(ref p) = self.params.clone() else {
            return Ok(());
        };
        self.ipc = Some(DualQueueIpc::connect(p)?);
        Ok(())
    }

    /// Whether [`Self::connect_ipc`] successfully opened the host queues.
    pub fn is_ipc_connected(&self) -> bool {
        self.ipc.is_some()
    }

    /// Clears per-tick outbound IPC drop flags on the dual queue (no-op when IPC is disconnected).
    pub fn reset_ipc_outbound_drop_tick_flags(&mut self) {
        if let Some(ipc) = self.ipc.as_mut() {
            ipc.reset_outbound_drop_tick_flags();
        }
    }

    /// Whether any **primary** outbound send failed since the last [`Self::reset_ipc_outbound_drop_tick_flags`].
    pub fn ipc_outbound_primary_drop_this_tick(&self) -> bool {
        self.ipc
            .as_ref()
            .is_some_and(|i| i.had_outbound_primary_drop_this_tick())
    }

    /// Whether any **background** outbound send failed since the last [`Self::reset_ipc_outbound_drop_tick_flags`].
    pub fn ipc_outbound_background_drop_this_tick(&self) -> bool {
        self.ipc
            .as_ref()
            .is_some_and(|i| i.had_outbound_background_drop_this_tick())
    }

    /// Current consecutive outbound drop streaks per channel (`0` when disconnected or after a successful send).
    pub fn ipc_consecutive_outbound_drop_streaks(&self) -> (u32, u32) {
        self.ipc
            .as_ref()
            .map(|i| {
                (
                    i.consecutive_primary_drop_streak(),
                    i.consecutive_background_drop_streak(),
                )
            })
            .unwrap_or((0, 0))
    }

    /// Records wall-clock spacing for FPS / [`crate::shared::PerformanceState`] before lock-step
    /// [`Self::pre_frame`].
    ///
    /// Call once at the start of each winit tick with the same [`Instant`] used as that tick’s
    /// duration anchor (see [`Self::set_perf_last_total_us`]).
    pub fn on_tick_frame_wall_clock(&mut self, now: Instant) {
        let wall_interval_us = self
            .last_tick_wall_start
            .map(|t| now.duration_since(t).as_micros() as u64)
            .unwrap_or(0);
        self.wall_interval_us_for_perf = wall_interval_us;
        self.last_tick_wall_start = Some(now);
    }

    /// Stores the previous tick’s total duration so the next [`Self::pre_frame`] can populate
    /// [`crate::shared::PerformanceState::render_time`].
    pub fn set_perf_last_total_us(&mut self, total_us: u64) {
        self.perf_last_total_us = total_us;
    }

    /// Poll and sort commands so [`RendererCommand::RendererInitData`] runs before any other work
    /// in the same batch (then frame submits), avoiding a fatal `Uninitialized` ordering hazard.
    pub fn poll_commands(&mut self) -> Vec<RendererCommand> {
        let Some(ref mut ipc) = self.ipc else {
            return Vec::new();
        };
        let mut batch = ipc.poll();
        batch.sort_by_key(|c| match c {
            RendererCommand::RendererInitData(_) => 0u8,
            RendererCommand::FrameSubmitData(_) => 1,
            _ => 2,
        });
        batch
    }

    /// Whether a [`FrameStartData`] should be sent this tick (caller should supply [`InputState`] via [`Self::pre_frame`]).
    pub fn should_send_begin_frame(&self) -> bool {
        begin_frame_allowed(
            self.init_state.is_finalized(),
            self.fatal_error,
            self.ipc.is_some(),
            self.last_frame_data_processed,
            self.last_frame_index,
            self.sent_bootstrap_frame_start,
        )
    }

    /// Lock-step begin-frame: send [`FrameStartData`] with `inputs` when [`Self::should_send_begin_frame`].
    ///
    /// Call only when [`Self::should_send_begin_frame`] is true so [`InputState`] is not dropped on the floor.
    pub fn pre_frame(&mut self, inputs: InputState) {
        if !self.should_send_begin_frame() {
            return;
        }

        let bootstrap = self.last_frame_index < 0 && !self.sent_bootstrap_frame_start;
        let performance = super::frame_start_performance::step_frame_performance(
            self.wall_interval_us_for_perf,
            self.perf_last_total_us,
            &mut self.smoothed_fps,
            &mut self.last_perf_send,
            Instant::now(),
        );
        let frame_start = FrameStartData {
            last_frame_index: self.last_frame_index,
            performance,
            inputs: Some(inputs),
            ..Default::default()
        };
        if let Some(ref mut ipc) = self.ipc {
            if !ipc.send_primary(RendererCommand::FrameStartData(frame_start)) {
                logger::warn!(
                    "IPC primary queue full: FrameStartData not sent; will retry on the next tick"
                );
                return;
            }
        }
        self.last_frame_data_processed = false;
        if bootstrap {
            self.sent_bootstrap_frame_start = true;
        }
    }

    /// Host wants relative mouse mode; merged into [`crate::shared::MouseState::is_active`] when packing input.
    pub fn host_cursor_lock_requested(&self) -> bool {
        self.cursor_lock_requested
    }

    /// Updates cursor policy when the host includes [`OutputState`], and queues window chrome for the app.
    ///
    /// When `output` is `None`, [`Self::host_cursor_lock_requested`] is left unchanged (Unity only calls
    /// `HandleOutputState` when non-null); pending chrome is cleared for that frame.
    pub fn apply_frame_submit_output(&mut self, output: Option<OutputState>) {
        if let Some(ref o) = output {
            self.cursor_lock_requested = o.lock_cursor;
            self.last_output_state = Some(o.clone());
        }
        self.pending_output_state = output;
    }

    /// Last [`OutputState`] from a frame submit (for continuous cursor lock / warp each tick).
    pub fn last_output_state(&self) -> Option<&OutputState> {
        self.last_output_state.as_ref()
    }

    /// Takes the last [`OutputState`] so the winit layer can apply it once.
    pub fn take_pending_output_state(&mut self) -> Option<OutputState> {
        self.pending_output_state.take()
    }

    /// Updates lock-step state after the host submits a frame.
    pub fn note_frame_submit_processed(&mut self, frame_index: i32) {
        self.last_frame_index = frame_index;
        self.last_frame_data_processed = true;
    }

    /// Marks init received after `renderer_init_data` (shared memory may be created here).
    pub fn on_init_received(&mut self) {
        self.init_state = InitState::InitReceived;
        self.last_frame_data_processed = true;
    }
}
