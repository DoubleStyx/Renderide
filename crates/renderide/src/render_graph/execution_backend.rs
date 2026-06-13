//! Graph-facing resource contracts implemented by renderer-owned backend systems.

use std::sync::Arc;

use super::blackboard::Blackboard;
use super::compiled::FrameView;
use super::context::GraphResolvedResources;
use super::{HistoryRegistry, TransientPool};
use crate::frame_upload_batch::{FrameUploadBatchStats, GraphUploadSink};
use crate::gpu::driver_thread::SubmitToken;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::graph_inputs::{
    GraphAssetResources, GraphFrameResources, GraphPassFrame, PerViewFramePlan,
    PreRecordViewResourceLayout,
};
use crate::hud_contract::{DebugHudEncodeError, PerViewHudConfig, PerViewHudOutputs};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::occlusion::OcclusionGraphHook;
use crate::upload_arena::PersistentUploadArena;

/// Disjoint graph-frame parameter slices borrowed from the backend.
pub struct GraphFrameParamsSplit<'a> {
    /// Hi-Z and temporal occlusion state.
    pub occlusion: &'a dyn OcclusionGraphHook,
    /// Frame-global and per-view bind resources.
    pub frame_resources: &'a dyn GraphFrameResources,
    /// Material registry and caches.
    pub materials: &'a MaterialSystem,
    /// Resident asset/resource pools.
    pub asset_resources: &'a dyn GraphAssetResources,
    /// Optional mesh preprocess pipelines.
    pub mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    /// Optional mesh deform scratch for frame-global recording.
    pub mesh_deform_scratch: Option<&'a mut MeshDeformScratch>,
    /// Optional mutable skin cache for frame-global mesh deform.
    pub mesh_deform_skin_cache: Option<&'a mut GpuSkinCache>,
    /// Optional read-only skin cache for per-view forward draws.
    pub skin_cache: Option<&'a GpuSkinCache>,
    /// GPU limits snapshot.
    pub gpu_limits: Option<Arc<GpuLimits>>,
    /// Optional MSAA depth-resolve helpers.
    pub msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Host-owned skin influence mode for mesh deform compute.
    pub skin_weight_mode: crate::shared::SkinWeightMode,
    /// Per-view debug HUD switches.
    pub debug_hud: PerViewHudConfig,
}

/// Backend-owned per-view blackboard preparation that can be shared across Rayon workers.
pub trait GraphViewBlackboardPreparer: Sync {
    /// Lets backend-specific systems enrich one per-view blackboard before recording.
    fn prepare_view_blackboard(
        &self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        gpu_limits: &GpuLimits,
        frame: &GraphPassFrame<'_>,
        frame_plan: &PerViewFramePlan,
        blackboard: &mut Blackboard,
    );
}

/// Backend services required by compiled render-graph execution.
pub trait GraphExecutionBackend {
    /// Render-graph transient pool.
    fn transient_pool_mut(&mut self) -> &mut TransientPool;
    /// Schedules resolved transient resources for pool release after a driver submit completes.
    fn schedule_transient_release_after_submit(
        &mut self,
        token: SubmitToken,
        resources: Vec<GraphResolvedResources>,
    );
    /// Persistent graph history registry.
    fn history_registry(&self) -> &HistoryRegistry;
    /// Mutable persistent graph history registry.
    fn history_registry_mut(&mut self) -> &mut HistoryRegistry;
    /// Persistent upload staging arena.
    fn upload_arena_mut(&mut self) -> &mut PersistentUploadArena;
    /// Publishes upload drain stats for diagnostics.
    fn record_frame_upload_stats(&mut self, stats: FrameUploadBatchStats);
    /// Scene-color format selected for this graph frame.
    fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat;
    /// GPU limits snapshot after attach.
    fn gpu_limits(&self) -> Option<&Arc<GpuLimits>>;
    /// Optional MSAA depth-resolve resources.
    fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>>;
    /// Shared frame resources.
    fn frame_resources(&self) -> &dyn GraphFrameResources;
    /// Mutable frame resources.
    fn frame_resources_mut(&mut self) -> &mut dyn GraphFrameResources;
    /// Shared material system.
    fn materials(&self) -> &MaterialSystem;
    /// Shared asset resources.
    fn asset_resources(&self) -> &dyn GraphAssetResources;
    /// Shared occlusion state.
    fn occlusion(&self) -> &dyn OcclusionGraphHook;
    /// Mutable occlusion state.
    fn occlusion_mut(&mut self) -> &mut dyn OcclusionGraphHook;
    /// Optional mesh preprocess pipelines.
    fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines>;
    /// Optional read-only skin cache.
    fn skin_cache(&self) -> Option<&GpuSkinCache>;
    /// Host-owned skin influence mode selected for mesh deform compute.
    fn skin_weight_mode(&self) -> crate::shared::SkinWeightMode;
    /// Split frame params for frame-global recording.
    fn split_for_graph_frame_params(&mut self) -> GraphFrameParamsSplit<'_>;
    /// Warms assets required by caller-seeded per-view blackboards.
    fn pre_warm_view_assets_from_blackboards(
        &mut self,
        device: &wgpu::Device,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
    );
    /// Creates a thread-shareable per-view blackboard preparer for this graph execution.
    fn view_blackboard_preparer(&self) -> Box<dyn GraphViewBlackboardPreparer + '_>;
    /// Estimates the blackboard work size used to gate parallel pre-record preparation.
    fn estimate_view_blackboard_prepare_draw_count(&self, blackboard: &Blackboard) -> usize;
    /// Debug HUD flags consumed by per-view recording.
    fn per_view_hud_config(&self) -> PerViewHudConfig;
    /// Render-graph command-recording mode selected for this frame.
    fn command_recording_mode(&self) -> crate::config::CommandRecordingMode;
    /// Whether the HUD will draw visible content.
    fn debug_hud_has_visible_content(&self) -> bool;
    /// Encodes the debug HUD overlay.
    fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), DebugHudEncodeError>;
    /// Clears cached input-capture state when HUD encoding is skipped.
    fn clear_debug_hud_input_capture(&mut self);
    /// Applies per-view HUD outputs after recording.
    fn apply_per_view_hud_outputs(&mut self, outputs: &PerViewHudOutputs);
}
