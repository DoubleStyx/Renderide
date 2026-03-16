//! Session: orchestrates IPC, scene, assets, and frame flow.
//!
//! Extension point for session state, draw batch collection.

use crate::assets::{self, AssetRegistry};
use crate::config::RenderConfig;
use crate::gpu::PipelineVariant;
use crate::ipc::receiver::CommandReceiver;
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::render::batch::{DrawEntry, SpaceDrawBatch};
use crate::scene::{render_transform_to_matrix, SceneGraph};
use crate::shared::VertexAttributeType;
use crate::session::commands::{CommandContext, CommandDispatcher, CommandResult};
use crate::session::init::{get_connection_parameters, InitError, take_singleton_init};
use crate::session::state::ViewState;
use crate::shared::{FrameStartData, FrameSubmitData, InputState, RendererCommand};

/// Main session: coordinates command ingest, scene, and assets.
pub struct Session {
    receiver: CommandReceiver,
    scene_graph: SceneGraph,
    asset_registry: AssetRegistry,
    view_state: ViewState,
    shared_memory: Option<SharedMemoryAccessor>,
    dispatcher: CommandDispatcher,
    init_received: bool,
    init_finalized: bool,
    is_standalone: bool,
    shutdown: bool,
    fatal_error: bool,
    last_frame_index: i32,
    last_frame_data_processed: bool,
    sent_bootstrap_frame_start: bool,
    pending_input: Option<InputState>,
    pending_mesh_unloads: Vec<i32>,
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
            init_received: false,
            init_finalized: false,
            is_standalone: false,
            shutdown: false,
            fatal_error: false,
            last_frame_index: -1,
            last_frame_data_processed: false,
            sent_bootstrap_frame_start: false,
            pending_input: None,
            pending_mesh_unloads: Vec::new(),
            lock_cursor: false,
            render_config: RenderConfig::default(),
            pending_render_tasks: Vec::new(),
            primary_camera_task: None,
            primary_view_transform: None,
            primary_view_space_id: None,
            primary_view_override: None,
            primary_view_position_is_external: None,
            primary_root_transform: None,
        }
    }

    /// Initializes the session. Call once at startup.
    pub fn init(&mut self) -> Result<(), InitError> {
        if !take_singleton_init() {
            return Err(InitError::SingletonAlreadyExists);
        }

        if get_connection_parameters().is_none() {
            self.is_standalone = true;
            self.init_finalized = true;
            return Ok(());
        }

        self.receiver.connect()?;
        if !self.receiver.is_connected() {
            self.is_standalone = true;
            self.init_finalized = true;
        }
        Ok(())
    }

    /// Per-frame update. Returns Some(exit_code) to request exit.
    pub fn update(&mut self) -> Option<i32> {
        if self.shutdown {
            return Some(0);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.handle_update())) {
            Ok(()) => None,
            Err(_e) => {
                self.fatal_error = true;
                Some(4)
            }
        }
    }

    fn handle_update(&mut self) {
        self.process_commands();

        if self.init_finalized && !self.fatal_error {
            let bootstrap = self.last_frame_index < 0 && !self.sent_bootstrap_frame_start;
            let should_send = self.last_frame_data_processed || bootstrap;
            if should_send && self.receiver.is_connected() {
                self.send_begin_frame();
                self.last_frame_data_processed = false;
                if bootstrap {
                    self.sent_bootstrap_frame_start = true;
                }
            }
        }
    }

    fn process_commands(&mut self) {
        let commands = self.receiver.poll();

        let (mesh_cmds, other_cmds): (Vec<_>, Vec<_>) = commands.into_iter().partition(|c| {
            matches!(
                c,
                RendererCommand::mesh_upload_data(_) | RendererCommand::mesh_unload(_)
            )
        });

        for cmd in mesh_cmds {
            self.apply_command(cmd);
        }
        for cmd in other_cmds {
            self.apply_command(cmd);
        }
    }

    fn apply_command(&mut self, cmd: RendererCommand) {
        let mut ctx = CommandContext {
            shared_memory: &mut self.shared_memory,
            asset_registry: &mut self.asset_registry,
            scene_graph: &mut self.scene_graph,
            view_state: &mut self.view_state,
            receiver: &mut self.receiver,
            init_received: &mut self.init_received,
            init_finalized: &mut self.init_finalized,
            shutdown: &mut self.shutdown,
            fatal_error: &mut self.fatal_error,
            last_frame_data_processed: &mut self.last_frame_data_processed,
            pending_mesh_unloads: &mut self.pending_mesh_unloads,
            render_config: &mut self.render_config,
            lock_cursor: &mut self.lock_cursor,
            pending_frame_data: None,
        };

        let result = self.dispatcher.dispatch(cmd, &mut ctx);

        if result == CommandResult::FatalError {
            self.fatal_error = true;
            return;
        }

        if let Some(data) = ctx.pending_frame_data {
            if let Err(_e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.process_frame_data(data);
            })) {
                self.fatal_error = true;
            } else if self.init_finalized {
                self.last_frame_data_processed = true;
            }
        }
    }

    fn process_frame_data(&mut self, data: FrameSubmitData) {
        self.last_frame_index = data.frame_index;
        self.view_state.near_clip = data.near_clip;
        self.view_state.far_clip = data.far_clip;
        self.view_state.desktop_fov = data.desktop_fov;
        self.primary_view_transform = None;
        self.primary_view_space_id = None;
        self.primary_view_override = None;
        self.primary_view_position_is_external = None;
        self.primary_root_transform = None;

        if let Some(ref output) = data.output_state {
            self.lock_cursor = output.lock_cursor;
        }

        if let Some(ref mut shm) = self.shared_memory
            && let Err(e) = self.scene_graph.apply_frame_update(shm, &data) {
                crate::error!("Scene apply_frame_update: {}", e);
            }

        let active_non_overlay: Vec<_> = data
            .render_spaces
            .iter()
            .filter(|u| u.is_active && !u.is_overlay)
            .collect();
        if active_non_overlay.len() > 1 {
            self.fatal_error = true;
            return;
        }
        if let Some(update) = active_non_overlay.first() {
            self.primary_view_space_id = Some(update.id);
            self.primary_view_override = Some(update.override_view_position);
            self.primary_view_position_is_external = Some(update.view_position_is_external);
            self.primary_root_transform = Some(update.root_transform);
            // View selection: override (freecam) → overriden_view_transform; else → root_transform.
            // When view_position_is_external is true (e.g. VR/third-person), view may need to come
            // from input/head state; we use root for now since the host does not send a separate head pose.
            self.primary_view_transform = Some(if update.override_view_position {
                update.overriden_view_transform
            } else {
                update.root_transform
            });
        }
        if self.primary_view_transform.is_none()
            && let Some(first) = data.render_spaces.first() {
                self.primary_view_space_id = Some(first.id);
                self.primary_view_override = Some(first.override_view_position);
                self.primary_view_position_is_external = Some(first.view_position_is_external);
                self.primary_root_transform = Some(first.root_transform);
                self.primary_view_transform = Some(first.root_transform);
            }

        self.pending_render_tasks = data.render_tasks;
        self.primary_camera_task = self.pending_render_tasks.first().cloned();
    }

    fn send_begin_frame(&mut self) {
        let frame_start = FrameStartData {
            last_frame_index: self.last_frame_index,
            performance: None,
            inputs: self.pending_input.take(),
            rendered_reflection_probes: Vec::new(),
            video_clock_errors: Vec::new(),
        };
        self.receiver
            .send(RendererCommand::frame_start_data(frame_start));
    }

    /// Processes render tasks (camera renders to buffers). Stub for now.
    pub fn process_render_tasks(&mut self) {
        self.pending_render_tasks.clear();
    }

    /// Drains mesh asset IDs unloaded this frame.
    pub fn drain_pending_mesh_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_mesh_unloads)
    }

    /// Sets input for next FrameStartData.
    pub fn set_pending_input(&mut self, input: InputState) {
        self.pending_input = Some(input);
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

    /// Returns the primary camera task.
    pub fn primary_camera_task(&self) -> Option<&crate::shared::CameraRenderTask> {
        self.primary_camera_task.as_ref()
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
    pub fn collect_draw_batches(&mut self) -> Vec<SpaceDrawBatch> {
        let mut batches = Vec::new();
        let active_space_ids: Vec<i32> = self
            .scene_graph
            .scenes()
            .iter()
            .filter(|(_, s)| s.is_active && !s.is_overlay)
            .map(|(id, _)| *id)
            .collect();

        let mut draw_batch_samples: Option<(i32, usize, usize, usize, Vec<(i32, String)>)> = None;

        for space_id in active_space_ids {
            if let Err(e) = self.scene_graph.compute_world_matrices(space_id) {
                crate::error!("Scene compute_world_matrices: {}", e);
                continue;
            }
            let scene = match self.scene_graph.get_scene(space_id) {
                Some(s) => s,
                None => continue,
            };

            let mut draws = Vec::new();
            let mut samples = Vec::new();
            let use_debug_uv = self.render_config.use_debug_uv;
            let _frame_index = self.last_frame_index;
            let combined = scene.drawables.iter().map(|d| (d, false)).chain(
                scene.skinned_drawables.iter().map(|d| (d, true)),
            );
            for (entry, is_skinned) in combined {
                if entry.node_id < 0 {
                    continue;
                }
                if is_skinned {
                    if entry.bone_transform_ids.as_ref().map_or(true, |b| b.is_empty()) {
                        crate::warn!(
                            "Skinned draw skipped: bone_transform_ids missing or empty (node_id={})",
                            entry.node_id
                        );
                        continue;
                    }
                    if let Some(mesh) = self.asset_registry.get_mesh(entry.mesh_handle) {
                        if mesh.bind_poses.as_ref().map_or(true, |b| b.is_empty()) {
                            crate::warn!(
                                "Skinned draw skipped: mesh missing bind_poses (mesh={}, node_id={})",
                                entry.mesh_handle,
                                entry.node_id
                            );
                            continue;
                        }
                    }
                }
                let idx = entry.node_id as usize;
                let world_matrix = match self.scene_graph.get_world_matrix(space_id, idx) {
                    Some(m) => m,
                    None => {
                        if idx >= scene.nodes.len() {
                            continue;
                        }

                        render_transform_to_matrix(&scene.nodes[idx])
                    }
                };
                let material_id = entry.material_handle.unwrap_or(-1);
                if samples.len() < 3 {
                    let t = world_matrix.column(3);
                    samples.push((entry.node_id, format!("({:.2},{:.2},{:.2})", t.x, t.y, t.z)));
                }
                let pipeline_variant = if is_skinned {
                    PipelineVariant::Skinned
                } else {
                    compute_pipeline_variant(
                        false,
                        entry.mesh_handle,
                        use_debug_uv,
                        &self.asset_registry,
                    )
                };
                draws.push(DrawEntry {
                    model_matrix: world_matrix,
                    mesh_asset_id: entry.mesh_handle,
                    is_skinned,
                    material_id,
                    bone_transform_ids: if is_skinned {
                        entry.bone_transform_ids.clone()
                    } else {
                        None
                    },
                    pipeline_variant,
                });
            }

            draws.sort_by_key(|d| {
                (d.pipeline_variant.clone(), d.material_id, d.mesh_asset_id)
            });

            if !draws.is_empty() {
                if draw_batch_samples.is_none() && !samples.is_empty() {
                    draw_batch_samples = Some((
                        space_id,
                        scene.nodes.len(),
                        scene.drawables.len() + scene.skinned_drawables.len(),
                        draws.len(),
                        samples,
                    ));
                }
                batches.push(SpaceDrawBatch {
                    space_id,
                    is_overlay: scene.is_overlay,
                    view_transform: scene.view_transform,
                    draws,
                });
            }
        }

        batches.sort_by_key(|b| b.is_overlay);
        batches
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes pipeline variant from is_skinned, mesh UVs, and use_debug_uv.
fn compute_pipeline_variant(
    is_skinned: bool,
    mesh_asset_id: i32,
    use_debug_uv: bool,
    asset_registry: &AssetRegistry,
) -> PipelineVariant {
    if is_skinned {
        return PipelineVariant::Skinned;
    }
    let has_uvs = asset_registry
        .get_mesh(mesh_asset_id)
        .and_then(|m| {
            assets::attribute_offset_size_format(&m.vertex_attributes, VertexAttributeType::uv0)
        })
        .map(|(_, s, _)| s >= 4)
        .unwrap_or(false);
    if use_debug_uv && has_uvs {
        PipelineVariant::UvDebug
    } else {
        PipelineVariant::NormalDebug
    }
}
