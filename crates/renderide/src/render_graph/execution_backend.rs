//! Graph-facing resource contracts implemented by renderer-owned backend systems.

use std::sync::Arc;

use hashbrown::HashSet;

use super::blackboard::Blackboard;
use super::compiled::FrameView;
use super::frame_params::{GraphPassFrame, PerViewFramePlan, PreRecordViewResourceLayout};
use super::{HistoryRegistry, TransientPool};
use crate::camera::ViewId;
use crate::config::{AutoExposureSettings, BloomSettings, GtaoSettings, MotionBlurSettings};
use crate::diagnostics::{DebugHudEncodeError, PerViewHudConfig, PerViewHudOutputs};
use crate::gpu::frame_globals::SkyboxSpecularUniformParams;
use crate::gpu::{GpuLight, GpuLimits, MsaaDepthResolveResources};
use crate::gpu_pools::{
    CubemapPool, MeshPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{
    GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines, PaddedPerDrawUniforms, SkinCacheKey,
};
use crate::occlusion::OcclusionGraphHook;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::render_graph::upload_arena::PersistentUploadArena;

/// Cloned references to the shared clustered-light storage buffers.
#[derive(Clone)]
pub struct GraphClusterBufferRefs {
    /// Two `u32` words per cluster: compact-index offset and count.
    pub cluster_light_counts: wgpu::Buffer,
    /// Compact light-index storage addressed by each cluster range row.
    pub cluster_light_indices: wgpu::Buffer,
}

/// Graph-facing access to renderer frame resources.
pub trait GraphFrameResources: Send + Sync {
    /// Whether frame-global GPU resources were attached.
    fn has_frame_gpu(&self) -> bool;

    /// Packed GPU lights for one render view.
    fn frame_lights(&self, view_id: ViewId) -> &[GpuLight];

    /// Light count used in one view's frame uniforms and shaders.
    fn frame_light_count_u32(&self, view_id: ViewId) -> u32;

    /// View-local lights storage buffer.
    fn lights_buffer(&self, view_id: ViewId) -> Option<wgpu::Buffer>;

    /// Shared frame-uniform buffer.
    fn frame_uniform_buffer(&self) -> Option<wgpu::Buffer>;

    /// Shared clustered-light buffers.
    fn shared_cluster_buffer_refs(&self) -> Option<GraphClusterBufferRefs>;

    /// Current shared cluster-buffer version.
    fn shared_cluster_version(&self) -> u64;

    /// Per-view cluster-params uniform buffer.
    fn per_view_cluster_params_buffer(&self, view_id: ViewId) -> Option<wgpu::Buffer>;

    /// Per-view frame bind group and frame-uniform buffer.
    fn per_view_frame_bind_group_and_buffer(
        &self,
        view_id: ViewId,
    ) -> Option<(Arc<wgpu::BindGroup>, wgpu::Buffer)>;

    /// Ensures this view's per-draw slab can hold `draw_count` rows and returns its storage buffer.
    fn ensure_per_view_per_draw_capacity(
        &self,
        device: &wgpu::Device,
        view_id: ViewId,
        draw_count: usize,
    ) -> Option<wgpu::Buffer>;

    /// Gives callers mutable access to the per-view CPU slab-packing scratch.
    fn with_per_view_per_draw_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<PaddedPerDrawUniforms>, &mut Vec<u8>),
    ) -> bool;

    /// Gives callers mutable access to the per-view material-batch boundary scratch so it can be
    /// cleared and refilled without reallocating. Each tuple is an inclusive
    /// `(first_draw_idx, last_draw_idx)` span over the sorted world-mesh draw list. Returns
    /// `false` if the scratch slot has not yet been provisioned for this view.
    ///
    /// The boundary span type is duplicated as a tuple here rather than imported from
    /// `passes::world_mesh_forward` because `render_graph -> passes` is a forbidden layer edge
    /// (see `tests/architecture_layers.rs`).
    #[expect(
        clippy::type_complexity,
        reason = "callback Vec element type cannot be hoisted through the render_graph -> passes layer boundary"
    )]
    fn with_per_view_material_batch_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<(usize, usize)>),
    ) -> bool;

    /// Per-view per-draw storage buffer.
    fn per_view_per_draw_storage(&self, view_id: ViewId) -> Option<wgpu::Buffer>;

    /// Per-view per-draw bind group.
    fn per_view_per_draw_bind_group(&self, view_id: ViewId) -> Option<Arc<wgpu::BindGroup>>;

    /// Empty material bind group used by shaders without per-material resources.
    fn empty_material_bind_group(&self) -> Option<Arc<wgpu::BindGroup>>;

    /// Copies the current depth attachment into this view's sampled scene-depth snapshot.
    fn copy_scene_depth_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool;

    /// Copies the current HDR scene color into this view's sampled scene-color snapshot.
    fn copy_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool;

    /// Uniform parameters for the active skybox/reflection-probe specular source.
    fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams;

    /// Whether visible mesh-deform filtering has proven there is no work for this submission.
    fn visible_mesh_deform_filter_is_empty(&self) -> bool;

    /// Whether mesh deform has already recorded work for this graph submission.
    fn mesh_deform_dispatched_this_submission(&self) -> bool;

    /// Marks mesh deform work as recorded for this graph submission.
    fn set_mesh_deform_dispatched_this_submission(&self);

    /// Cloned visible mesh-deform filter for this submission's frame-global deform collection.
    fn visible_mesh_deform_keys_snapshot(&self) -> Option<HashSet<SkinCacheKey>>;

    /// Ensures per-view frame bind resources are resident.
    fn ensure_per_view_frame_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
        layout: PreRecordViewResourceLayout,
    ) -> bool;

    /// Ensures per-view per-draw resources are resident.
    fn ensure_per_view_per_draw_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> bool;

    /// Ensures per-view per-draw CPU scratch is resident.
    fn ensure_per_view_per_draw_scratch(&mut self, view_id: ViewId);

    /// Synchronizes shared frame resources before graph recording.
    fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        view_layouts: &[PreRecordViewResourceLayout],
    );
}

/// Graph-facing access to resident asset/resource pools.
pub trait GraphAssetResources: Send + Sync {
    /// Resident mesh pool.
    fn mesh_pool(&self) -> &MeshPool;
    /// Resident 2D texture pool.
    fn texture_pool(&self) -> &TexturePool;
    /// Resident 3D texture pool.
    fn texture3d_pool(&self) -> &Texture3dPool;
    /// Resident cubemap pool.
    fn cubemap_pool(&self) -> &CubemapPool;
    /// Host render-texture pool.
    fn render_texture_pool(&self) -> &RenderTexturePool;
    /// Resident video texture pool.
    fn video_texture_pool(&self) -> &VideoTexturePool;
}

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
    /// Per-view debug HUD switches.
    pub debug_hud: PerViewHudConfig,
}

/// Backend services required by compiled render-graph execution.
pub trait GraphExecutionBackend {
    /// Render-graph transient pool.
    fn transient_pool_mut(&mut self) -> &mut TransientPool;
    /// Persistent graph history registry.
    fn history_registry(&self) -> &HistoryRegistry;
    /// Mutable persistent graph history registry.
    fn history_registry_mut(&mut self) -> &mut HistoryRegistry;
    /// Persistent upload staging arena.
    fn upload_arena_mut(&mut self) -> &mut PersistentUploadArena;
    /// Scene-color format selected for this graph frame.
    fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat;
    /// GPU limits snapshot after attach.
    fn gpu_limits(&self) -> Option<&Arc<GpuLimits>>;
    /// Optional MSAA depth-resolve resources.
    fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>>;
    /// Live GTAO settings.
    fn live_gtao_settings(&self) -> GtaoSettings;
    /// Live bloom settings.
    fn live_bloom_settings(&self) -> BloomSettings;
    /// Live motion-blur settings.
    fn live_motion_blur_settings(&self) -> MotionBlurSettings;
    /// Live auto-exposure settings.
    fn live_auto_exposure_settings(&self) -> AutoExposureSettings;
    /// Wall-frame delta in seconds.
    fn wall_frame_delta_seconds(&self) -> f32;
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
    /// Split frame params for frame-global recording.
    fn split_for_graph_frame_params(&mut self) -> GraphFrameParamsSplit<'_>;
    /// Warms assets required by caller-seeded per-view blackboards.
    fn pre_warm_view_assets_from_blackboards(
        &mut self,
        device: &wgpu::Device,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
    );
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
    /// Debug HUD flags consumed by per-view recording.
    fn per_view_hud_config(&self) -> PerViewHudConfig;
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
