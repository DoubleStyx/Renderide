//! Session: orchestrates IPC, scene, assets, and frame flow.
//!
//! Extension point for session state, draw batch collection.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::assets::AssetRegistry;
use crate::config::RenderConfig;
use crate::gpu::GpuState;
use crate::input::WindowInputState;
use crate::ipc::receiver::CommandReceiver;
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::render::batch::SpaceDrawBatch;
use crate::render::{RenderLoop, RenderTaskExecutor};
use crate::scene::{ResolvedLight, SceneGraph};
use crate::session::commands::{
    AssetContext, CommandContext, CommandDispatcher, CommandResult, FrameContext, SessionFlags,
};
use crate::session::frame_data::{
    apply_clip_and_output_state, select_primary_view, validate_active_non_overlay,
};
use crate::session::init::{InitError, get_connection_parameters, take_singleton_init};
use crate::session::state::{InitState, ViewState};
use crate::shared::{
    FrameStartData, FrameSubmitData, LightsBufferRendererConsumed, RendererCommand,
};

/// Accumulates phase times inside [`Session::collect_draw_batches_for_task`] when diagnostics are enabled.
#[derive(Default)]
pub(crate) struct SpaceCollectTimingSplit {
    /// Time spent in [`SceneGraph::compute_world_matrices`] for this space.
    world_matrices: Duration,
    /// Time spent filtering drawables, building entries, sorting, and creating the batch.
    filter_sort_batch: Duration,
}

/// Main session: coordinates command ingest, scene, and assets.
pub struct Session {
    receiver: CommandReceiver,
    scene_graph: SceneGraph,
    asset_registry: AssetRegistry,
    view_state: ViewState,
    shared_memory: Option<SharedMemoryAccessor>,
    dispatcher: CommandDispatcher,
    init_state: InitState,
    is_standalone: bool,
    shutdown: bool,
    fatal_error: bool,
    last_frame_index: i32,
    last_frame_data_processed: bool,
    sent_bootstrap_frame_start: bool,
    pending_mesh_unloads: Vec<i32>,
    pending_texture_unloads: Vec<i32>,
    pending_material_unloads: Vec<i32>,
    lock_cursor: bool,
    render_config: RenderConfig,
    pending_render_tasks: Vec<crate::shared::CameraRenderTask>,
    primary_camera_task: Option<crate::shared::CameraRenderTask>,
    primary_view_transform: Option<crate::shared::RenderTransform>,
    /// Space ID and override flag for the primary view (diagnostic).
    primary_view_space_id: Option<i32>,
    primary_view_override: Option<bool>,
    /// Whether view position comes from external source (e.g. VR head) — diagnostic.
    primary_view_position_is_external: Option<bool>,
    /// Root transform of primary space (for diagnostic: compare with view when override differs).
    primary_root_transform: Option<crate::shared::RenderTransform>,
    /// Resolved lights per space, populated each frame during collect_draw_batches.
    resolved_lights: HashMap<i32, Vec<ResolvedLight>>,
}

impl Session {
    /// Creates a new session.
    pub fn new() -> Self {
        Self {
            receiver: CommandReceiver::new(),
            scene_graph: SceneGraph::new(),
            asset_registry: AssetRegistry::new(),
            view_state: ViewState::default(),
            shared_memory: None,
            dispatcher: CommandDispatcher::new(),
            init_state: InitState::Uninitialized,
            is_standalone: false,
            shutdown: false,
            fatal_error: false,
            last_frame_index: -1,
            last_frame_data_processed: false,
            sent_bootstrap_frame_start: false,
            pending_mesh_unloads: Vec::new(),
            pending_texture_unloads: Vec::new(),
            pending_material_unloads: Vec::new(),
            lock_cursor: false,
            render_config: RenderConfig::load(),
            pending_render_tasks: Vec::new(),
            primary_camera_task: None,
            primary_view_transform: None,
            primary_view_space_id: None,
            primary_view_override: None,
            primary_view_position_is_external: None,
            primary_root_transform: None,
            resolved_lights: HashMap::new(),
        }
    }

    /// Initializes the session. Call once at startup.
    pub fn init(&mut self) -> Result<(), InitError> {
        if !take_singleton_init() {
            return Err(InitError::SingletonAlreadyExists);
        }

        if get_connection_parameters().is_none() {
            self.is_standalone = true;
            self.init_state = InitState::Finalized;
            return Ok(());
        }

        self.receiver.connect()?;
        if !self.receiver.is_connected() {
            self.is_standalone = true;
            self.init_state = InitState::Finalized;
        }
        Ok(())
    }

    /// Per-frame update. Returns Some(exit_code) to request exit.
    ///
    /// When a [`FrameStartData`](crate::shared::FrameStartData) is sent, input is sampled from
    /// `window_input` at that moment via [`WindowInputState::take_input_state`], so accumulated
    /// mouse and scroll deltas are not cleared on redraws that do not send frame start.
    pub fn update(&mut self, window_input: &mut WindowInputState) -> Option<i32> {
        if self.shutdown {
            return Some(0);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.handle_update(window_input);
        })) {
            Ok(()) => None,
            Err(e) => {
                logger::log_panic_payload(e, "Session update panic");
                self.fatal_error = true;
                Some(4)
            }
        }
    }

    /// Drains IPC commands, applies frame data, then may send `BeginFrame` for host lock-step.
    ///
    /// At log level **trace**, emits `FrameSubmitData` space ids and each `BeginFrame` send reason
    /// (`last_frame_data_processed` vs bootstrap) for debugging desync with the host.
    fn handle_update(&mut self, window_input: &mut WindowInputState) {
        self.process_commands();

        if self.init_state.is_finalized() && !self.fatal_error {
            let bootstrap = self.last_frame_index < 0 && !self.sent_bootstrap_frame_start;
            let processed_prev = self.last_frame_data_processed;
            let should_send = processed_prev || bootstrap;
            if should_send && self.receiver.is_connected() {
                logger::trace!(
                    "IPC lockstep: sending BeginFrame last_frame_index={} reason_processed_prev_frame={} bootstrap={}",
                    self.last_frame_index,
                    processed_prev,
                    bootstrap
                );
                self.send_begin_frame(window_input);
                self.last_frame_data_processed = false;
                if bootstrap {
                    self.sent_bootstrap_frame_start = true;
                }
            }
        }
    }

    fn process_commands(&mut self) {
        let commands = self.receiver.poll();

        // Frame submit must run before asset uploads in the same poll so scene updates and
        // render tasks are applied before new meshes/textures are used.
        let (frame_cmds, rest): (Vec<_>, Vec<_>) = commands
            .into_iter()
            .partition(|c| matches!(c, RendererCommand::frame_submit_data(_)));

        for cmd in frame_cmds {
            self.apply_command(cmd);
        }
        for cmd in rest {
            self.apply_command(cmd);
        }
    }

    fn apply_command(&mut self, cmd: RendererCommand) {
        let mut ctx = CommandContext {
            assets: AssetContext {
                shared_memory: &mut self.shared_memory,
                asset_registry: &mut self.asset_registry,
            },
            session_flags: SessionFlags {
                init_state: &mut self.init_state,
                shutdown: &mut self.shutdown,
                fatal_error: &mut self.fatal_error,
                last_frame_data_processed: &mut self.last_frame_data_processed,
            },
            frame: FrameContext {
                pending_mesh_unloads: &mut self.pending_mesh_unloads,
                pending_texture_unloads: &mut self.pending_texture_unloads,
                pending_material_unloads: &mut self.pending_material_unloads,
                pending_frame_data: None,
            },
            scene_graph: &mut self.scene_graph,
            view_state: &mut self.view_state,
            receiver: &mut self.receiver,
            render_config: &mut self.render_config,
            lock_cursor: &mut self.lock_cursor,
        };

        let result = self.dispatcher.dispatch(&cmd, &mut ctx);

        if result == CommandResult::FatalError {
            self.fatal_error = true;
            return;
        }

        if let Some(data) = ctx.frame.pending_frame_data {
            if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.process_frame_data(data);
            })) {
                logger::log_panic_payload(e, "process_frame_data panic");
                self.fatal_error = true;
            } else if self.init_state.is_finalized() {
                self.last_frame_data_processed = true;
            }
        }
    }

    /// Applies host [`FrameSubmitData`](crate::shared::FrameSubmitData): scene graph, validation, primary view, render tasks.
    ///
    /// At log level **trace**, logs `frame_index` and render space ids (Unity tears down spaces
    /// omitted from this list in a given frame).
    fn process_frame_data(&mut self, data: FrameSubmitData) {
        self.last_frame_index = data.frame_index;
        let space_ids: Vec<i32> = data.render_spaces.iter().map(|u| u.id).collect();
        logger::trace!(
            "FrameSubmitData: frame_index={} render_space_ids={:?}",
            data.frame_index,
            space_ids
        );

        apply_clip_and_output_state(
            &data,
            &mut self.view_state.near_clip,
            &mut self.view_state.far_clip,
            &mut self.view_state.desktop_fov,
            &mut self.lock_cursor,
        );
        self.render_config.near_clip = data.near_clip;
        self.render_config.far_clip = data.far_clip;
        self.render_config.desktop_fov = data.desktop_fov;

        self.primary_view_transform = None;
        self.primary_view_space_id = None;
        self.primary_view_override = None;
        self.primary_view_position_is_external = None;
        self.primary_root_transform = None;

        if let Some(ref mut shm) = self.shared_memory
            && let Err(e) = self.scene_graph.apply_frame_update(shm, &data)
        {
            logger::error!("Scene apply_frame_update: {}", e);
        }

        if let Err(e) = validate_active_non_overlay(&data) {
            logger::error!("Frame validation failed: {}", e);
            self.fatal_error = true;
            return;
        }

        if let Some(selection) = select_primary_view(&data) {
            self.primary_view_space_id = Some(selection.space_id);
            self.primary_view_override = Some(selection.override_view_position);
            self.primary_view_position_is_external = Some(selection.view_position_is_external);
            self.primary_root_transform = Some(selection.root_transform);
            self.primary_view_transform = Some(selection.view_transform);
        }

        self.pending_render_tasks = data.render_tasks;
        self.primary_camera_task = self.pending_render_tasks.first().cloned();
    }

    /// Sends [`RendererCommand::frame_start_data`] with [`FrameStartData::inputs`] taken from
    /// `window_input` at send time (see [`WindowInputState::take_input_state`]).
    fn send_begin_frame(&mut self, window_input: &mut WindowInputState) {
        let mut input = window_input.take_input_state();
        if let Some(ref mut m) = input.mouse {
            m.is_active = m.is_active || self.lock_cursor;
        }
        let frame_start = FrameStartData {
            last_frame_index: self.last_frame_index,
            performance: None,
            inputs: Some(input),
            rendered_reflection_probes: Vec::new(),
            video_clock_errors: Vec::new(),
        };
        self.receiver
            .send(RendererCommand::frame_start_data(frame_start));
    }

    /// Processes render tasks (camera renders to buffers). Runs in the RenderToAsset frame phase
    /// after the main window has been rendered. Requires GPU and render loop to be initialized;
    /// otherwise clears pending tasks without executing. Always drains
    /// [`crate::render::RenderLoop::drain_pending_camera_task_readbacks`] when GPU is active so async
    /// readbacks from prior ticks complete.
    pub fn process_render_tasks(
        &mut self,
        gpu: Option<&mut GpuState>,
        render_loop: Option<&mut RenderLoop>,
    ) {
        if let (Some(gpu), Some(render_loop)) = (gpu, render_loop) {
            let tasks = std::mem::take(&mut self.pending_render_tasks);
            if !tasks.is_empty() {
                RenderTaskExecutor::execute(gpu, render_loop, self, tasks);
            }
            render_loop.drain_pending_camera_task_readbacks(&gpu.device, self);
        } else {
            self.pending_render_tasks.clear();
        }
    }

    /// Drains mesh asset IDs unloaded this frame.
    pub fn drain_pending_mesh_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_mesh_unloads)
    }

    /// Drains texture2D asset IDs unloaded this frame.
    pub fn drain_pending_texture_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_texture_unloads)
    }

    /// Drains material IDs to unload. Caller should evict pipelines for these IDs.
    pub fn drain_pending_material_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_material_unloads)
    }

    /// Whether cursor lock was requested.
    pub fn cursor_lock_requested(&self) -> bool {
        self.lock_cursor
    }

    /// Returns the asset registry.
    /// Returns the scene graph for world matrix lookups (e.g. bone transforms).
    pub fn scene_graph(&self) -> &SceneGraph {
        &self.scene_graph
    }

    pub fn asset_registry(&self) -> &AssetRegistry {
        &self.asset_registry
    }

    /// Returns mutable shared memory accessor for render task result writes.
    pub fn shared_memory_mut(&mut self) -> Option<&mut SharedMemoryAccessor> {
        self.shared_memory.as_mut()
    }

    /// Returns the render configuration.
    pub fn render_config(&self) -> &RenderConfig {
        &self.render_config
    }

    /// Returns the primary camera task.
    pub fn primary_camera_task(&self) -> Option<&crate::shared::CameraRenderTask> {
        self.primary_camera_task.as_ref()
    }

    /// Number of offscreen render tasks queued for the current tick.
    pub fn pending_render_task_count(&self) -> usize {
        self.pending_render_tasks.len()
    }

    /// Returns the space ID for the primary view (diagnostic).
    pub fn primary_view_space_id(&self) -> Option<i32> {
        self.primary_view_space_id
    }

    /// Returns whether the primary view uses overridden position (diagnostic).
    pub fn primary_view_override(&self) -> Option<bool> {
        self.primary_view_override
    }

    /// Returns whether the primary view position is external (e.g. VR head) — diagnostic.
    pub fn primary_view_position_is_external(&self) -> Option<bool> {
        self.primary_view_position_is_external
    }

    /// Returns the primary space root transform (diagnostic: compare with view when override differs).
    pub fn primary_root_transform(&self) -> Option<&crate::shared::RenderTransform> {
        self.primary_root_transform.as_ref()
    }

    /// Returns the primary view transform.
    pub fn primary_view_transform(&self) -> Option<&crate::shared::RenderTransform> {
        self.primary_view_transform.as_ref()
    }

    /// Last frame index.
    pub fn last_frame_index(&self) -> i32 {
        self.last_frame_index
    }

    /// Near clip.
    pub fn near_clip(&self) -> f32 {
        self.view_state.near_clip
    }

    /// Far clip.
    pub fn far_clip(&self) -> f32 {
        self.view_state.far_clip
    }

    /// Desktop FOV.
    pub fn desktop_fov(&self) -> f32 {
        self.view_state.desktop_fov
    }

    /// Collects draw batches for rendering.
    ///
    /// **Overlay inclusion rules** (main view): Includes all spaces where `is_active` is true,
    /// including private overlays (user's dashboard, loading indicators). The main window is the
    /// user's own view and should show their local UI. For offscreen renders (e.g. mirrors,
    /// picture-in-picture), [`CameraRenderTask`](crate::shared::CameraRenderTask) uses
    /// `render_private_ui` to control private overlay inclusion. Skips draws where layer is Hidden.
    ///
    /// **Overlay view**: Overlay spaces use `primary_view_transform()` as view override
    /// (UpdateOverlayPositioning); overlay batches follow the head/camera.
    pub fn collect_draw_batches(&mut self) -> Vec<SpaceDrawBatch> {
        let space_ids: Vec<i32> = self
            .scene_graph
            .scenes()
            .iter()
            .filter(|(_, s)| s.is_active)
            .map(|(id, _)| *id)
            .collect();

        self.resolved_lights.clear();
        let mut batches = Vec::new();
        let overlay_view_override = self.primary_view_transform().cloned();
        let log_timings = self.render_config().log_collect_draw_batches_timing;
        let mut acc_world = Duration::ZERO;
        let mut acc_filter = Duration::ZERO;
        let mut acc_lights = Duration::ZERO;
        for space_id in space_ids {
            let mut per_space = if log_timings {
                Some(SpaceCollectTimingSplit::default())
            } else {
                None
            };
            batches.extend(self.collect_draw_batches_for_task(
                space_id,
                &[],
                &[],
                true,
                overlay_view_override,
                &mut per_space,
            ));
            if let Some(ref p) = per_space {
                acc_world += p.world_matrices;
                acc_filter += p.filter_sort_batch;
            }
            let lights_start = log_timings.then(Instant::now);
            let resolved = self
                .scene_graph
                .light_cache
                .resolve_lights_with_fallback(space_id, |tid| {
                    self.scene_graph.get_world_matrix(space_id, tid)
                });
            if let Some(start) = lights_start {
                acc_lights += start.elapsed();
            }
            if resolved.is_empty() {
                logger::trace!(
                    "resolved lights space_id={} count=0 (no lights in scene)",
                    space_id
                );
            } else {
                logger::trace!(
                    "resolved lights space_id={} count={} lights=[{}]",
                    space_id,
                    resolved.len(),
                    resolved
                        .iter()
                        .map(|l| format!(
                            "pos=({:.2},{:.2},{:.2}) dir=({:.2},{:.2},{:.2}) type={:?} intensity={:.2}",
                            l.world_position.x, l.world_position.y, l.world_position.z,
                            l.world_direction.x, l.world_direction.y, l.world_direction.z,
                            l.light_type, l.intensity
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                );
                self.resolved_lights.insert(space_id, resolved);
            }
        }
        let sort_start = log_timings.then(Instant::now);
        batches.sort_by_key(|b| b.is_overlay);
        let sort_ms = sort_start.map(|s| s.elapsed().as_secs_f64() * 1000.0);
        let overlay_count = batches.iter().filter(|b| b.is_overlay).count();
        let non_overlay_count = batches.len() - overlay_count;
        logger::trace!(
            "collected {} overlay batches, {} non-overlay batches (total={})",
            overlay_count,
            non_overlay_count,
            batches.len()
        );
        if log_timings {
            logger::trace!(
                "collect_draw_batches timing (ms): world_matrices={:.3} filter_sort_batch={:.3} lights_resolve={:.3} batch_sort={:.3}",
                acc_world.as_secs_f64() * 1000.0,
                acc_filter.as_secs_f64() * 1000.0,
                acc_lights.as_secs_f64() * 1000.0,
                sort_ms.unwrap_or(0.0),
            );
        }
        batches
    }

    /// Returns resolved lights for a space, if any. Populated during collect_draw_batches.
    pub fn resolved_lights_for_space(&self, space_id: i32) -> Option<&[ResolvedLight]> {
        self.resolved_lights.get(&space_id).map(|v| v.as_slice())
    }

    /// Sends one `LightsBufferRendererConsumed` per distinct buffer `global_unique_id` among
    /// resolved lights for the current frame, matching Renderite.Unity’s one-ack-per-buffer
    /// submission. Call after rendering to signal to the host that light data was consumed.
    pub fn send_lights_consumed_for_rendered_spaces(&mut self) {
        use std::collections::HashSet;

        let mut sent: HashSet<i32> = HashSet::new();
        for lights in self.resolved_lights.values() {
            for light in lights {
                if light.global_unique_id >= 0 && sent.insert(light.global_unique_id) {
                    self.receiver
                        .send(RendererCommand::lights_buffer_renderer_consumed(
                            LightsBufferRendererConsumed {
                                global_unique_id: light.global_unique_id,
                            },
                        ));
                }
            }
        }
    }

    /// Collects draw batches for a single space (e.g. CameraRenderTask).
    ///
    /// * `space_id` - Render space to collect from.
    /// * `only_render_list` - When non-empty, include only draws with `node_id` in this list.
    /// * `exclude_render_list` - When non-empty, exclude draws with `node_id` in this list.
    /// * `include_private` - When false, returns empty if the space is private.
    /// * `view_override` - When `Some` and the space is overlay, use this as the batch view
    ///   transform instead of `scene.view_transform`. Matches Unity overlay positioning: overlay
    ///   camera (view) is the head; see [`RenderSpace.UpdateOverlayPositioning`].
    /// * `timing` - When `Some`, accumulates per-phase durations for this space. Main-view
    ///   [`Session::collect_draw_batches`] passes `Some(default)` when
    ///   [`RenderConfig::log_collect_draw_batches_timing`] is enabled; other callers use `&mut None`.
    ///
    /// Skips draws where layer is Hidden. Returns at most one batch (for the given space).
    pub(crate) fn collect_draw_batches_for_task(
        &mut self,
        space_id: i32,
        only_render_list: &[i32],
        exclude_render_list: &[i32],
        include_private: bool,
        view_override: Option<crate::shared::RenderTransform>,
        timing: &mut Option<SpaceCollectTimingSplit>,
    ) -> Vec<SpaceDrawBatch> {
        let mut batches = Vec::new();

        let world_start = timing.is_some().then(Instant::now);
        if let Err(e) = self.scene_graph.compute_world_matrices(space_id) {
            logger::error!("Scene compute_world_matrices: {}", e);
            return batches;
        }
        if let (Some(start), Some(acc)) = (world_start, timing.as_mut()) {
            acc.world_matrices += start.elapsed();
        }

        let this = &*self;
        let scene = match this.scene_graph.get_scene(space_id) {
            Some(s) => s,
            None => return batches,
        };

        if !include_private && scene.is_private {
            return batches;
        }

        let filter_start = timing.is_some().then(Instant::now);
        let filtered = super::collect::filter_and_collect_drawables(
            scene,
            only_render_list,
            exclude_render_list,
            &this.scene_graph,
            space_id,
            this.asset_registry(),
            this.render_config(),
            this.render_config().use_debug_uv,
            this.render_config().use_pbr,
        );
        let mut draws = super::collect::build_draw_entries(filtered);
        draws.sort_by_key(|d| {
            (
                scene.is_overlay,
                -d.sort_key,
                d.pipeline_variant,
                d.material_id,
                d.mesh_asset_id,
            )
        });

        if let Some(batch) =
            super::collect::create_space_batch(space_id, scene, draws, view_override)
        {
            batches.push(batch);
        }

        if let (Some(start), Some(acc)) = (filter_start, timing.as_mut()) {
            acc.filter_sort_batch += start.elapsed();
        }

        batches
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
