//! Session: orchestrates IPC, scene, assets, and frame flow.
//!
//! Extension point for session state, draw batch collection.

use std::collections::{HashMap, HashSet};

use glam::Mat4;

use crate::assets::{self, AssetRegistry};
use crate::config::RenderConfig;
use crate::gpu::{GpuState, PipelineVariant};
use crate::ipc::receiver::CommandReceiver;
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::render::batch::{DrawEntry, SpaceDrawBatch};
use crate::render::{RenderLoop, RenderTaskExecutor};
use crate::scene::{Drawable, ResolvedLight, Scene, SceneGraph, render_transform_to_matrix};
use crate::session::commands::{
    AssetContext, CommandContext, CommandDispatcher, CommandResult, FrameContext, SessionFlags,
};
use crate::session::frame_data::{
    apply_clip_and_output_state, select_primary_view, validate_active_non_overlay,
};
use crate::session::init::{InitError, get_connection_parameters, take_singleton_init};
use crate::session::state::{InitState, ViewState};
use crate::shared::VertexAttributeType;
use crate::shared::{
    FrameStartData, FrameSubmitData, InputState, LayerType, LightsBufferRendererConsumed,
    RendererCommand,
};
use crate::stencil::{StencilOperation, StencilState};

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
    pending_input: Option<InputState>,
    pending_mesh_unloads: Vec<i32>,
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
            pending_input: None,
            pending_mesh_unloads: Vec::new(),
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
    pub fn update(&mut self) -> Option<i32> {
        if self.shutdown {
            return Some(0);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.handle_update())) {
            Ok(()) => None,
            Err(e) => {
                logger::log_panic_payload(e, "Session update panic");
                self.fatal_error = true;
                Some(4)
            }
        }
    }

    fn handle_update(&mut self) {
        self.process_commands();

        if self.init_state.is_finalized() && !self.fatal_error {
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

    fn process_frame_data(&mut self, data: FrameSubmitData) {
        self.last_frame_index = data.frame_index;

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

    /// Processes render tasks (camera renders to buffers). Runs in the RenderToAsset frame phase
    /// after the main window has been rendered. Requires GPU and render loop to be initialized;
    /// otherwise clears pending tasks without executing.
    pub fn process_render_tasks(
        &mut self,
        gpu: Option<&mut GpuState>,
        render_loop: Option<&mut RenderLoop>,
    ) {
        if let (Some(gpu), Some(render_loop)) = (gpu, render_loop) {
            let tasks = std::mem::take(&mut self.pending_render_tasks);
            RenderTaskExecutor::execute(gpu, render_loop, self, tasks);
        } else {
            self.pending_render_tasks.clear();
        }
    }

    /// Drains mesh asset IDs unloaded this frame.
    pub fn drain_pending_mesh_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_mesh_unloads)
    }

    /// Drains material IDs to unload. Caller should evict pipelines for these IDs.
    pub fn drain_pending_material_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_material_unloads)
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
        for space_id in space_ids {
            batches.extend(self.collect_draw_batches_for_task(
                space_id,
                &[],
                &[],
                true,
                overlay_view_override,
            ));
            let resolved = self
                .scene_graph
                .light_cache
                .resolve_lights_with_fallback(space_id, |tid| {
                    self.scene_graph.get_world_matrix(space_id, tid)
                });
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
        batches.sort_by_key(|b| b.is_overlay);
        let overlay_count = batches.iter().filter(|b| b.is_overlay).count();
        logger::trace!("collected {} overlay batches", overlay_count);
        batches
    }

    /// Returns resolved lights for a space, if any. Populated during collect_draw_batches.
    pub fn resolved_lights_for_space(&self, space_id: i32) -> Option<&[ResolvedLight]> {
        self.resolved_lights.get(&space_id).map(|v| v.as_slice())
    }

    /// Sends LightsBufferRendererConsumed for all resolved lights from the current frame.
    /// Call after rendering to signal to the host that light data was consumed.
    pub fn send_lights_consumed_for_rendered_spaces(&mut self) {
        for lights in self.resolved_lights.values() {
            for light in lights {
                if light.global_unique_id >= 0 {
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
    ///
    /// Skips draws where layer is Hidden. Returns at most one batch (for the given space).
    pub fn collect_draw_batches_for_task(
        &mut self,
        space_id: i32,
        only_render_list: &[i32],
        exclude_render_list: &[i32],
        include_private: bool,
        view_override: Option<crate::shared::RenderTransform>,
    ) -> Vec<SpaceDrawBatch> {
        let mut batches = Vec::new();

        if let Err(e) = self.scene_graph.compute_world_matrices(space_id) {
            logger::error!("Scene compute_world_matrices: {}", e);
            return batches;
        }

        let this = &*self;
        let scene = match this.scene_graph.get_scene(space_id) {
            Some(s) => s,
            None => return batches,
        };

        if !include_private && scene.is_private {
            return batches;
        }

        let filtered = filter_and_collect_drawables(
            scene,
            only_render_list,
            exclude_render_list,
            &this.scene_graph,
            space_id,
            this.asset_registry(),
            this.render_config.use_debug_uv,
            this.render_config.use_pbr,
        );
        let mut draws = build_draw_entries(filtered);
        draws.sort_by_key(|d| {
            (
                scene.is_overlay,
                -d.sort_key,
                d.pipeline_variant.clone(),
                d.material_id,
                d.mesh_asset_id,
            )
        });

        if let Some(batch) = create_space_batch(space_id, scene, draws, view_override) {
            batches.push(batch);
        }

        batches
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Filtered drawable with world matrix and pipeline variant.
///
/// Output of [`filter_and_collect_drawables`]; input to [`build_draw_entries`].
struct FilteredDrawable {
    drawable: Drawable,
    world_matrix: Mat4,
    pipeline_variant: PipelineVariant,
}

/// Filters drawables by layer, render lists, and skinned validity; collects world matrices.
///
/// Skips Hidden layer, applies only/exclude lists, validates bone_transform_ids and bind_poses
/// for skinned draws. Returns (Drawable, world_matrix, pipeline_variant) for each valid draw.
#[allow(clippy::too_many_arguments)]
fn filter_and_collect_drawables(
    scene: &Scene,
    only_render_list: &[i32],
    exclude_render_list: &[i32],
    scene_graph: &SceneGraph,
    space_id: i32,
    asset_registry: &AssetRegistry,
    use_debug_uv: bool,
    use_pbr: bool,
) -> Vec<FilteredDrawable> {
    let only_set: HashSet<i32> = only_render_list.iter().copied().collect();
    let exclude_set: HashSet<i32> = exclude_render_list.iter().copied().collect();
    let use_only = !only_set.is_empty();
    let use_exclude = !exclude_set.is_empty();

    let mut out = Vec::new();
    let combined = scene
        .drawables
        .iter()
        .map(|d| (d, false))
        .chain(scene.skinned_drawables.iter().map(|d| (d, true)));

    for (entry, is_skinned) in combined {
        if entry.node_id < 0 {
            continue;
        }
        if entry.layer == LayerType::hidden {
            continue;
        }
        if use_only && !only_set.contains(&entry.node_id) {
            continue;
        }
        if use_exclude && exclude_set.contains(&entry.node_id) {
            continue;
        }
        if is_skinned {
            if entry
                .bone_transform_ids
                .as_ref()
                .is_none_or(|b| b.is_empty())
            {
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (node_id={})",
                    entry.node_id
                );
                continue;
            }
            if let Some(mesh) = asset_registry.get_mesh(entry.mesh_handle)
                && mesh.bind_poses.as_ref().is_none_or(|b| b.is_empty())
            {
                logger::trace!(
                    "Skinned draw skipped: mesh missing bind_poses (mesh={}, node_id={})",
                    entry.mesh_handle,
                    entry.node_id
                );
                continue;
            }
        }
        let idx = entry.node_id as usize;
        let world_matrix = match scene_graph.get_world_matrix(space_id, idx) {
            Some(m) => m,
            None => {
                if idx >= scene.nodes.len() {
                    continue;
                }
                render_transform_to_matrix(&scene.nodes[idx])
            }
        };

        let stencil_state = resolve_overlay_stencil_state(scene.is_overlay, entry, asset_registry);
        let mut drawable = entry.clone();
        drawable.stencil_state = stencil_state;

        let pipeline_variant = compute_pipeline_variant_for_drawable(
            scene.is_overlay,
            is_skinned,
            &drawable,
            entry.mesh_handle,
            use_debug_uv,
            use_pbr,
            asset_registry,
        );
        out.push(FilteredDrawable {
            drawable,
            world_matrix,
            pipeline_variant,
        });
    }

    out
}

/// Builds draw entries from filtered drawables.
///
/// Converts [`FilteredDrawable`] tuples into [`DrawEntry`] for batch construction.
fn build_draw_entries(filtered: Vec<FilteredDrawable>) -> Vec<DrawEntry> {
    filtered
        .into_iter()
        .map(|f| {
            let material_id = f.drawable.material_handle.unwrap_or(-1);
            DrawEntry {
                model_matrix: f.world_matrix,
                node_id: f.drawable.node_id,
                mesh_asset_id: f.drawable.mesh_handle,
                is_skinned: f.drawable.is_skinned,
                material_id,
                sort_key: f.drawable.sort_key,
                bone_transform_ids: if f.drawable.is_skinned {
                    f.drawable.bone_transform_ids.clone()
                } else {
                    None
                },
                root_bone_transform_id: if f.drawable.is_skinned {
                    f.drawable.root_bone_transform_id
                } else {
                    None
                },
                blendshape_weights: if f.drawable.is_skinned {
                    f.drawable.blend_shape_weights.clone()
                } else {
                    None
                },
                pipeline_variant: f.pipeline_variant,
                stencil_state: f.drawable.stencil_state,
            }
        })
        .collect()
}

/// Creates a space batch if draws is non-empty.
///
/// Returns `None` when draws is empty; otherwise builds [`SpaceDrawBatch`] from scene metadata.
/// For overlay spaces, when `view_override` is `Some`, uses it as the batch view transform
/// (primary/head view) instead of `scene.view_transform` (root).
fn create_space_batch(
    space_id: i32,
    scene: &Scene,
    draws: Vec<DrawEntry>,
    view_override: Option<crate::shared::RenderTransform>,
) -> Option<SpaceDrawBatch> {
    if draws.is_empty() {
        return None;
    }
    let view_transform = if scene.is_overlay {
        view_override.unwrap_or(scene.view_transform)
    } else {
        scene.view_transform
    };
    Some(SpaceDrawBatch {
        space_id,
        is_overlay: scene.is_overlay,
        view_transform,
        draws,
    })
}

/// Resolves overlay stencil state from material property store when scene is overlay.
fn resolve_overlay_stencil_state(
    is_overlay: bool,
    entry: &Drawable,
    asset_registry: &AssetRegistry,
) -> Option<StencilState> {
    if !is_overlay {
        return None;
    }
    if let Some(block_id) = entry.material_override_block_id {
        StencilState::from_property_store(&asset_registry.material_property_store, block_id)
            .or(entry.stencil_state)
    } else {
        entry.stencil_state
    }
}

/// Computes pipeline variant for a drawable based on overlay, skinned, stencil, and mesh.
fn compute_pipeline_variant_for_drawable(
    is_overlay: bool,
    is_skinned: bool,
    drawable: &Drawable,
    mesh_asset_id: i32,
    use_debug_uv: bool,
    use_pbr: bool,
    asset_registry: &AssetRegistry,
) -> PipelineVariant {
    if is_overlay {
        if let Some(ref stencil) = drawable.stencil_state {
            if stencil.pass_op == StencilOperation::Replace && stencil.write_mask != 0 {
                if is_skinned {
                    PipelineVariant::OverlayStencilMaskWriteSkinned
                } else {
                    PipelineVariant::OverlayStencilMaskWrite
                }
            } else if stencil.pass_op == StencilOperation::Zero {
                if is_skinned {
                    PipelineVariant::OverlayStencilMaskClearSkinned
                } else {
                    PipelineVariant::OverlayStencilMaskClear
                }
            } else if is_skinned {
                PipelineVariant::OverlayStencilSkinned
            } else {
                PipelineVariant::OverlayStencilContent
            }
        } else if is_skinned {
            PipelineVariant::Skinned
        } else {
            compute_pipeline_variant(false, mesh_asset_id, use_debug_uv, false, asset_registry)
        }
    } else if is_skinned {
        if use_pbr {
            PipelineVariant::SkinnedPbr
        } else {
            PipelineVariant::Skinned
        }
    } else {
        compute_pipeline_variant(false, mesh_asset_id, use_debug_uv, use_pbr, asset_registry)
    }
}

/// Computes pipeline variant from is_skinned, mesh UVs, use_debug_uv, and use_pbr.
fn compute_pipeline_variant(
    is_skinned: bool,
    mesh_asset_id: i32,
    use_debug_uv: bool,
    use_pbr: bool,
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
    } else if use_pbr {
        PipelineVariant::Pbr
    } else {
        PipelineVariant::NormalDebug
    }
}
