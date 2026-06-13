//! Narrow backend access packet used by render-graph execution.

use std::sync::Arc;

mod warmup;

use crate::backend::AssetTransferQueue;
use crate::diagnostics::{DebugHudEncodeError, PerViewHudConfig, PerViewHudOutputs};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::graph_inputs::{
    GraphPassFrame, PerViewFramePlan, PerViewFramePlanSlot, PreRecordViewResourceLayout,
};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::occlusion::OcclusionSystem;
use crate::passes::post_processing::settings_slots::{
    AutoExposureSettingsSlot, AutoExposureSettingsValue, BloomSettingsSlot, BloomSettingsValue,
    GtaoSettingsSlot, GtaoSettingsValue, MotionBlurSettingsSlot, MotionBlurSettingsValue,
};
use crate::render_graph::TransientPool;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::compiled::FrameView;
use crate::render_graph::execution_backend::{
    GraphExecutionBackend, GraphFrameParamsSplit, GraphViewBlackboardPreparer,
};
use crate::render_graph::frame_upload_batch::{FrameUploadBatchStats, GraphUploadSink};
use crate::render_graph::upload_arena::PersistentUploadArena;

use super::super::debug_hud_bundle::DebugHudBundle;
use super::super::{
    FrameResourceManager, HistoryRegistry, WorldMeshDrawPlanSlot, WorldMeshOverlayDrawPlanSlot,
};

/// Live post-processing parameters seeded into per-view blackboards.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LivePostProcessingSettings {
    /// GTAO settings snapshot selected before graph execution borrows backend fields.
    pub(super) gtao: crate::config::GtaoSettings,
    /// Bloom settings snapshot selected before graph execution borrows backend fields.
    pub(super) bloom: crate::config::BloomSettings,
    /// Motion-blur settings snapshot selected before graph execution borrows backend fields.
    pub(super) motion_blur: crate::config::MotionBlurSettings,
    /// Auto-exposure settings snapshot selected before graph execution borrows backend fields.
    pub(super) auto_exposure: crate::config::AutoExposureSettings,
}

/// Narrow backend packet used by the render graph executor.
///
/// This is intentionally a bundle of disjoint backend owners instead of `&mut RenderBackend`:
/// graph execution can mutate transient/history/frame/HUD state without gaining access to IPC,
/// facade-only helpers, or unrelated backend orchestration fields.
pub(crate) struct BackendGraphAccess<'a> {
    /// Hi-Z and temporal occlusion state.
    pub(crate) occlusion: &'a mut OcclusionSystem,
    /// Frame-global and per-view bind resources.
    pub(crate) frame_resources: &'a mut FrameResourceManager,
    /// Material registry, routes, and property data.
    pub(crate) materials: &'a MaterialSystem,
    /// Asset upload queues and resident GPU pools.
    pub(crate) asset_transfers: &'a mut AssetTransferQueue,
    /// Mesh preprocess pipelines for frame-global deform work.
    pub(crate) mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    /// Mesh deform scratch buffers for frame-global deform work.
    pub(crate) mesh_deform_scratch: Option<&'a mut MeshDeformScratch>,
    /// Skin cache mutably borrowed by frame-global deform and shared by per-view draws afterwards.
    pub(crate) skin_cache: Option<&'a mut GpuSkinCache>,
    /// Backend-owned world-mesh forward frame planner.
    pub(crate) world_mesh_frame_planner: &'a crate::backend::BackendWorldMeshFramePlanner,
    /// Render-graph transient pool retained across frames.
    pub(crate) transient_pool: &'a mut TransientPool,
    /// Persistent ping-pong history registry.
    pub(crate) history_registry: &'a mut HistoryRegistry,
    /// Persistent upload staging arena used by graph buffer upload drains.
    pub(crate) upload_arena: &'a mut PersistentUploadArena,
    /// Latest frame-upload stats published for diagnostics.
    pub(crate) latest_upload_stats: &'a mut FrameUploadBatchStats,
    /// Debug HUD state and encoder.
    pub(crate) debug_hud: &'a mut DebugHudBundle,
    /// Scene-color format snapshot selected before graph execution borrows backend fields.
    pub(super) scene_color_format: wgpu::TextureFormat,
    /// GPU limits snapshot selected before graph execution borrows backend fields.
    pub(super) gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth-resolve resources selected before graph execution borrows backend fields.
    pub(super) msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Host-owned skin influence mode selected for mesh deform compute.
    pub(super) skin_weight_mode: crate::shared::SkinWeightMode,
    /// Live post-processing settings seeded into per-view blackboards.
    pub(super) live_post_processing: LivePostProcessingSettings,
    /// Live command-recording mode selected before graph execution borrows backend fields.
    pub(super) command_recording_mode: crate::config::CommandRecordingMode,
    /// Wall-frame delta snapshot in milliseconds.
    pub(super) wall_frame_time_ms: f64,
}

struct BackendViewBlackboardPreparer<'a> {
    world_mesh_frame_planner: &'a crate::backend::BackendWorldMeshFramePlanner,
    live_post_processing: LivePostProcessingSettings,
    wall_frame_delta_seconds: f32,
}

impl GraphViewBlackboardPreparer for BackendViewBlackboardPreparer<'_> {
    fn prepare_view_blackboard(
        &self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        gpu_limits: &GpuLimits,
        frame: &GraphPassFrame<'_>,
        frame_plan: &PerViewFramePlan,
        blackboard: &mut Blackboard,
    ) {
        blackboard.insert::<PerViewFramePlanSlot>(frame_plan.clone());
        blackboard.insert::<GtaoSettingsSlot>(GtaoSettingsValue(self.live_post_processing.gtao));
        blackboard.insert::<BloomSettingsSlot>(BloomSettingsValue(self.live_post_processing.bloom));
        blackboard.insert::<MotionBlurSettingsSlot>(MotionBlurSettingsValue(
            self.live_post_processing.motion_blur,
        ));
        blackboard.insert::<AutoExposureSettingsSlot>(AutoExposureSettingsValue::for_view(
            self.live_post_processing.auto_exposure,
            self.wall_frame_delta_seconds,
            frame.view.view_id,
        ));
        crate::backend::prepare_world_mesh_view_blackboard(
            self.world_mesh_frame_planner,
            device,
            uploads,
            gpu_limits,
            frame,
            frame_plan,
            blackboard,
        );
    }
}

impl<'a> BackendGraphAccess<'a> {
    /// Mutable transient pool for graph resource resolution.
    pub(crate) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        self.transient_pool
    }

    /// Shared history registry for import resolution.
    pub(crate) fn history_registry(&self) -> &HistoryRegistry {
        self.history_registry
    }

    /// Mutable history registry for frame advance and view-scoped registrations.
    pub(crate) fn history_registry_mut(&mut self) -> &mut HistoryRegistry {
        self.history_registry
    }

    /// Persistent upload staging arena.
    pub(crate) fn upload_arena_mut(&mut self) -> &mut PersistentUploadArena {
        self.upload_arena
    }

    /// Publishes upload drain stats for the diagnostics HUD.
    pub(crate) fn record_frame_upload_stats(&mut self, stats: FrameUploadBatchStats) {
        *self.latest_upload_stats = stats;
    }

    /// Scene-color format snapshot selected for this graph frame.
    pub(crate) fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        self.scene_color_format
    }

    /// GPU limits snapshot after attach.
    pub(crate) fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.gpu_limits.as_ref()
    }

    /// Optional MSAA depth resolve resources.
    pub(crate) fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>> {
        self.msaa_depth_resolve.clone()
    }

    /// Wall-frame delta snapshot for this graph frame, in seconds.
    pub(crate) fn wall_frame_delta_seconds(&self) -> f32 {
        (self.wall_frame_time_ms / 1000.0).clamp(0.0, 1.0) as f32
    }

    /// Shared material system.
    pub(crate) fn materials(&self) -> &MaterialSystem {
        self.materials
    }

    /// Shared occlusion state.
    pub(crate) fn occlusion(&self) -> &OcclusionSystem {
        self.occlusion
    }

    /// Mutable occlusion state.
    pub(crate) fn occlusion_mut(&mut self) -> &mut OcclusionSystem {
        self.occlusion
    }

    /// Optional mesh preprocess pipelines.
    pub(crate) fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        self.mesh_preprocess
    }

    /// Optional read-only skin cache for per-view forward draws.
    pub(crate) fn skin_cache(&self) -> Option<&GpuSkinCache> {
        self.skin_cache.as_deref()
    }

    /// Host-owned skin influence mode selected for mesh deform compute.
    pub(crate) fn skin_weight_mode(&self) -> crate::shared::SkinWeightMode {
        self.skin_weight_mode
    }

    /// Debug HUD flags consumed by per-view recording.
    pub(crate) fn per_view_hud_config(&self) -> PerViewHudConfig {
        self.debug_hud.per_view_config()
    }

    /// Whether the HUD will draw visible content this frame.
    pub(crate) fn debug_hud_has_visible_content(&self) -> bool {
        self.debug_hud.has_visible_content()
    }

    /// Render-graph command-recording mode selected for this frame.
    pub(crate) fn command_recording_mode(&self) -> crate::config::CommandRecordingMode {
        self.command_recording_mode
    }

    /// Encodes the debug HUD overlay.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), DebugHudEncodeError> {
        profiling::scope!("hud::encode");
        self.debug_hud
            .encode_overlay(device, queue, encoder, backbuffer, extent, profiler)
    }

    /// Clears cached input-capture state when HUD encoding is skipped.
    pub(crate) fn clear_debug_hud_input_capture(&mut self) {
        self.debug_hud.clear_input_capture();
    }

    /// Applies one deferred per-view HUD payload.
    pub(crate) fn apply_per_view_hud_outputs(&mut self, outputs: &PerViewHudOutputs) {
        self.debug_hud.apply_per_view_outputs(outputs);
    }

    /// Disjoint mutable borrows and attach-time snapshots for frame-global pass params.
    pub(crate) fn split_for_graph_frame_params(&mut self) -> GraphFrameParamsSplit<'_> {
        GraphFrameParamsSplit {
            occlusion: self.occlusion,
            frame_resources: self.frame_resources,
            materials: self.materials,
            asset_resources: self.asset_transfers,
            mesh_preprocess: self.mesh_preprocess,
            mesh_deform_scratch: self.mesh_deform_scratch.as_deref_mut(),
            mesh_deform_skin_cache: self.skin_cache.as_deref_mut(),
            skin_cache: None,
            gpu_limits: self.gpu_limits.clone(),
            msaa_depth_resolve: self.msaa_depth_resolve.clone(),
            skin_weight_mode: self.skin_weight_mode,
            debug_hud: self.debug_hud.per_view_config(),
        }
    }
}

impl GraphExecutionBackend for BackendGraphAccess<'_> {
    fn transient_pool_mut(&mut self) -> &mut TransientPool {
        BackendGraphAccess::transient_pool_mut(self)
    }

    fn history_registry(&self) -> &HistoryRegistry {
        BackendGraphAccess::history_registry(self)
    }

    fn history_registry_mut(&mut self) -> &mut HistoryRegistry {
        BackendGraphAccess::history_registry_mut(self)
    }

    fn upload_arena_mut(&mut self) -> &mut PersistentUploadArena {
        BackendGraphAccess::upload_arena_mut(self)
    }

    fn record_frame_upload_stats(&mut self, stats: FrameUploadBatchStats) {
        BackendGraphAccess::record_frame_upload_stats(self, stats);
    }

    fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        BackendGraphAccess::scene_color_format_wgpu(self)
    }

    fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        BackendGraphAccess::gpu_limits(self)
    }

    fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>> {
        BackendGraphAccess::msaa_depth_resolve(self)
    }

    fn frame_resources(&self) -> &dyn crate::render_graph::GraphFrameResources {
        self.frame_resources
    }

    fn frame_resources_mut(&mut self) -> &mut dyn crate::render_graph::GraphFrameResources {
        self.frame_resources
    }

    fn materials(&self) -> &MaterialSystem {
        BackendGraphAccess::materials(self)
    }

    fn asset_resources(&self) -> &dyn crate::render_graph::GraphAssetResources {
        self.asset_transfers
    }

    fn occlusion(&self) -> &dyn crate::occlusion::OcclusionGraphHook {
        BackendGraphAccess::occlusion(self)
    }

    fn occlusion_mut(&mut self) -> &mut dyn crate::occlusion::OcclusionGraphHook {
        BackendGraphAccess::occlusion_mut(self)
    }

    fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        BackendGraphAccess::mesh_preprocess(self)
    }

    fn skin_cache(&self) -> Option<&GpuSkinCache> {
        BackendGraphAccess::skin_cache(self)
    }

    fn skin_weight_mode(&self) -> crate::shared::SkinWeightMode {
        BackendGraphAccess::skin_weight_mode(self)
    }

    fn split_for_graph_frame_params(&mut self) -> GraphFrameParamsSplit<'_> {
        BackendGraphAccess::split_for_graph_frame_params(self)
    }

    fn pre_warm_view_assets_from_blackboards(
        &mut self,
        device: &wgpu::Device,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
    ) {
        BackendGraphAccess::pre_warm_view_assets_from_blackboards(
            self,
            device,
            views,
            view_layouts,
            resource_layouts,
        );
    }

    fn view_blackboard_preparer(&self) -> Box<dyn GraphViewBlackboardPreparer + '_> {
        Box::new(BackendViewBlackboardPreparer {
            world_mesh_frame_planner: self.world_mesh_frame_planner,
            live_post_processing: self.live_post_processing,
            wall_frame_delta_seconds: self.wall_frame_delta_seconds(),
        })
    }

    fn estimate_view_blackboard_prepare_draw_count(&self, blackboard: &Blackboard) -> usize {
        let world = blackboard
            .get::<WorldMeshDrawPlanSlot>()
            .map_or(0, crate::world_mesh::WorldMeshDrawPlan::draw_count);
        let overlay = blackboard
            .get::<WorldMeshOverlayDrawPlanSlot>()
            .map_or(0, crate::world_mesh::WorldMeshDrawPlan::draw_count);
        world.saturating_add(overlay)
    }

    fn per_view_hud_config(&self) -> PerViewHudConfig {
        BackendGraphAccess::per_view_hud_config(self)
    }

    fn command_recording_mode(&self) -> crate::config::CommandRecordingMode {
        BackendGraphAccess::command_recording_mode(self)
    }

    fn debug_hud_has_visible_content(&self) -> bool {
        BackendGraphAccess::debug_hud_has_visible_content(self)
    }

    fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), DebugHudEncodeError> {
        BackendGraphAccess::encode_debug_hud_overlay(
            self, device, queue, encoder, backbuffer, extent, profiler,
        )
    }

    fn clear_debug_hud_input_capture(&mut self) {
        BackendGraphAccess::clear_debug_hud_input_capture(self);
    }

    fn apply_per_view_hud_outputs(&mut self, outputs: &PerViewHudOutputs) {
        BackendGraphAccess::apply_per_view_hud_outputs(self, outputs);
    }
}
