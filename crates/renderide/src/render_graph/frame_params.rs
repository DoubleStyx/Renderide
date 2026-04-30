//! Per-frame parameters shared across render graph passes (scene, backend slices, surface state).
//!
//! Cross-pass per-view state that is too large or too volatile to live on the pass struct lives
//! in the per-view [`crate::render_graph::blackboard::Blackboard`] via typed slots defined here.
//!
//! [`FrameRenderParams`] is a thin compositor over [`FrameSystemsShared`] (once-per-frame system
//! handles) and [`FrameRenderParamsView`] (per-view surface state). This separation keeps the
//! record path focused on view-local data while shared systems are borrowed through explicit
//! fields.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::assets::AssetTransferQueue;
use crate::backend::FrameResourceManager;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::MaterialSystem;
use crate::materials::ShaderPermutation;
use crate::materials::{
    MaterialPassDesc, MaterialPipelineDesc, MaterialPipelineSet, RasterFrontFace,
};
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::occlusion::OcclusionSystem;
use crate::occlusion::gpu::HiZGpuState;
use crate::scene::SceneCoordinator;
use crate::shared::CameraClearMode;
use crate::skybox::PreparedSkybox;

use super::OutputDepthMode;
use super::blackboard::BlackboardSlot;
use crate::world_mesh::cull::WorldMeshCullProjParams;
use crate::world_mesh::draw_prep::PipelineVariantKey;
use crate::world_mesh::draw_prep::{
    CameraTransformDrawFilter, InstancePlan, WorldMeshDrawCollection, WorldMeshDrawItem,
};

/// Pipeline state resolved during world-mesh forward preparation.
pub struct WorldMeshForwardPipelineState {
    /// Whether this view records multiview raster passes.
    pub use_multiview: bool,
    /// Material pipeline descriptor for this view's color/depth/sample state.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation used by material pipeline lookup.
    pub shader_perm: ShaderPermutation,
}

/// Per-view background clear contract propagated from host camera state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameViewClear {
    /// Host camera clear mode for this view.
    pub mode: CameraClearMode,
    /// Host background color used when [`CameraClearMode::Color`] is selected.
    pub color: glam::Vec4,
}

impl FrameViewClear {
    /// Main-view clear mode: render the active render-space skybox.
    pub fn skybox() -> Self {
        Self {
            mode: CameraClearMode::Skybox,
            color: glam::Vec4::ZERO,
        }
    }

    /// Color clear mode with the supplied linear RGBA background.
    pub fn color(color: glam::Vec4) -> Self {
        Self {
            mode: CameraClearMode::Color,
            color,
        }
    }

    /// Converts host camera state into a frame-view clear descriptor.
    pub fn from_camera_state(state: &crate::shared::CameraState) -> Self {
        Self {
            mode: state.clear_mode,
            color: state.background_color,
        }
    }
}

impl Default for FrameViewClear {
    fn default() -> Self {
        Self::skybox()
    }
}

/// Snapshot-dependent helper work required by a prefetched world-mesh view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorldMeshHelperNeeds {
    /// Whether any draw in the view samples the scene-depth snapshot for the intersection subpass.
    pub depth_snapshot: bool,
    /// Whether any draw in the view samples the scene-color snapshot for the grab-pass subpass.
    pub color_snapshot: bool,
}

impl WorldMeshHelperNeeds {
    /// Derives helper-pass requirements from the material flags on a collected draw list.
    pub fn from_collection(collection: &WorldMeshDrawCollection) -> Self {
        let mut needs = Self::default();
        for item in &collection.items {
            needs.depth_snapshot |= item.batch_key.embedded_uses_scene_depth_snapshot;
            needs.color_snapshot |= item.batch_key.embedded_uses_scene_color_snapshot;
            if needs.depth_snapshot && needs.color_snapshot {
                break;
            }
        }
        needs
    }
}

/// Per-view prefetched world-mesh data seeded before graph execution.
#[derive(Clone, Debug)]
pub struct PrefetchedWorldMeshViewDraws {
    /// Draw items and culling statistics collected for the view.
    pub collection: WorldMeshDrawCollection,
    /// Projection state used during culling, reused when capturing Hi-Z temporal feedback.
    pub cull_proj: Option<WorldMeshCullProjParams>,
    /// Helper snapshots and tail passes required by this view's collected materials.
    pub helper_needs: WorldMeshHelperNeeds,
}

impl PrefetchedWorldMeshViewDraws {
    /// Builds a prefetched view packet and derives helper-pass requirements from `collection`.
    pub fn new(
        collection: WorldMeshDrawCollection,
        cull_proj: Option<WorldMeshCullProjParams>,
    ) -> Self {
        let helper_needs = WorldMeshHelperNeeds::from_collection(&collection);
        Self {
            collection,
            cull_proj,
            helper_needs,
        }
    }

    /// Builds an explicit empty draw packet for views that should skip world-mesh work.
    pub fn empty() -> Self {
        Self::new(WorldMeshDrawCollection::empty(), None)
    }
}

/// Per-view forward-pass preparation shared by future split graph nodes.
pub struct PreparedWorldMeshForwardFrame {
    /// Sorted world mesh draw items for this view.
    pub draws: Vec<WorldMeshDrawItem>,
    /// Per-view [`InstancePlan`]: per-draw slab layout plus regular and intersection
    /// [`super::DrawGroup`]s that the forward pass turns into one `draw_indexed` each.
    ///
    /// Replaces the older `regular_indices` / `intersect_indices: Vec<usize>` pair.
    /// Decouples the per-draw slab layout from the sorted-draw order so that same-mesh
    /// instances merge regardless of where the sort placed individual members.
    pub plan: InstancePlan,
    /// Pipeline format/sample/multiview state.
    pub pipeline: WorldMeshForwardPipelineState,
    /// Scene snapshot helper work needed by the prepared draw list.
    pub helper_needs: WorldMeshHelperNeeds,
    /// Whether indexed draws may use base instance.
    pub supports_base_instance: bool,
    /// Whether the opaque/clear forward subpass was already recorded by a split graph node.
    pub opaque_recorded: bool,
    /// Whether the scene-depth snapshot for intersection draws was already recorded by a split graph node.
    pub depth_snapshot_recorded: bool,
    /// Whether the intersection/color-resolve tail raster was already recorded by a split graph node.
    pub tail_raster_recorded: bool,
    /// Per-batch resolved pipelines and bind groups, pre-computed by the prepare pass in parallel.
    ///
    /// One entry per unique `MaterialDrawBatchKey` run in `draws`, covering `[first_draw_idx,
    /// last_draw_idx]` (inclusive). Both raster sub-passes (opaque and intersect) share this
    /// list; each sub-pass only reads entries whose draw-index range overlaps its own index slice.
    pub precomputed_batches: Vec<MaterialBatchPacket>,
    /// Optional background draw prepared for the opaque subpass.
    pub skybox: Option<PreparedSkybox>,
}

/// Blackboard slot for per-view MSAA attachment views resolved from transient graph resources.
///
/// Populated by the executor (before per-view passes run) from
/// [`super::compiled::helpers::populate_forward_msaa_from_graph_resources`] output.
/// Replaces the six `msaa_*` fields that previously lived on [`FrameRenderParams`].
pub struct MsaaViewsSlot;
impl BlackboardSlot for MsaaViewsSlot {
    type Value = MsaaViews;
}

/// MSAA attachment views for the forward pass (resolved from graph transient textures).
///
/// Fields are read by [`crate::render_graph::passes::WorldMeshDepthSnapshotPass`] and
/// [`crate::render_graph::passes::WorldMeshForwardDepthResolvePass`] via the per-view blackboard.
/// The forward depth-snapshot/resolve helpers in `world_mesh_forward/execute_helpers.rs`
/// currently resolve MSAA views directly from graph transient textures; reading from this slot
/// is wired in but the consumer functions are migrated incrementally.
#[derive(Clone)]
#[expect(
    dead_code,
    reason = "fields are accessed via the blackboard slot; consumer migration is incremental"
)]
pub struct MsaaViews {
    /// Graph-owned multisampled color attachment view when MSAA is active.
    pub msaa_color_view: wgpu::TextureView,
    /// Graph-owned multisampled depth attachment view when MSAA is active.
    pub msaa_depth_view: wgpu::TextureView,
    /// R32Float intermediate view used by the MSAA depth resolve path.
    pub msaa_depth_resolve_r32_view: wgpu::TextureView,
    /// `true` when MSAA depth/R32 views are two-layer array views for stereo multiview.
    pub msaa_depth_is_array: bool,
    /// Per-eye single-layer views of stereo MSAA depth.
    pub msaa_stereo_depth_layer_views: Option<[wgpu::TextureView; 2]>,
    /// Per-eye single-layer views of stereo R32Float resolve targets.
    pub msaa_stereo_r32_layer_views: Option<[wgpu::TextureView; 2]>,
}

/// Blackboard slot key for the per-view world-mesh forward plan.
///
/// Populated by [`crate::render_graph::passes::WorldMeshForwardPreparePass`] (a [`crate::render_graph::pass::CallbackPass`])
/// and consumed by the four downstream forward passes (opaque, depth snapshot, intersect,
/// depth resolve). Replaces the `prepared_world_mesh_forward` field that previously lived on
/// [`FrameRenderParams`].
pub struct WorldMeshForwardPlanSlot;
impl BlackboardSlot for WorldMeshForwardPlanSlot {
    type Value = PreparedWorldMeshForwardFrame;
}

/// Blackboard slot key for pre-collected world-mesh draws (secondary cameras / prefetch path).
///
/// When set before the graph executes, [`crate::render_graph::passes::WorldMeshForwardPreparePass`]
/// skips draw collection and uses this list instead.
pub struct PrefetchedWorldMeshDrawsSlot;
impl BlackboardSlot for PrefetchedWorldMeshDrawsSlot {
    type Value = PrefetchedWorldMeshViewDraws;
}

/// One resolved per-batch draw packet covering a contiguous range of sorted draws with the same
/// [`crate::render_graph::MaterialDrawBatchKey`].
///
/// Populated by the prepare pass (parallel rayon fan-out) so the recording loop can drive
/// pipeline and bind-group state entirely from this table — no LRU lookups during `RenderPass`.
#[derive(Clone)]
pub struct MaterialBatchPacket {
    /// First draw index (into the sorted draw list) covered by this entry.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this entry.
    pub last_draw_idx: usize,
    /// Exact pipeline variant requested for this batch.
    pub(crate) pipeline_key: PipelineVariantKey,
    /// Front-face winding used by the resolved pipeline set.
    pub front_face: RasterFrontFace,
    /// Resolved `@group(1)` bind group for this batch's material, or `None` for the empty fallback.
    pub bind_group: Option<Arc<wgpu::BindGroup>>,
    /// Resolved pipeline set for this batch, or `None` when the pipeline is unavailable (skip draws).
    pub pipelines: Option<MaterialPipelineSet>,
    /// Material pass descriptors parallel to `pipelines` (zero-alloc static reference).
    pub declared_passes: &'static [MaterialPassDesc],
}

/// Blackboard slot for per-view HUD data collected during recording and merged on the main thread.
pub struct PerViewHudOutputsSlot;
impl BlackboardSlot for PerViewHudOutputsSlot {
    type Value = PerViewHudOutputs;
}

/// HUD payload produced by one view during recording.
#[derive(Default)]
pub struct PerViewHudOutputs {
    /// Latest world-mesh draw stats for the view when the main HUD is enabled.
    pub world_mesh_draw_stats: Option<crate::world_mesh::WorldMeshDrawStats>,
    /// Latest world-mesh draw-state rows for the view when the main HUD is enabled.
    pub world_mesh_draw_state_rows: Option<Vec<crate::world_mesh::WorldMeshDrawStateRow>>,
    /// Texture2D asset ids used by the view when the textures HUD is enabled.
    pub current_view_texture_2d_asset_ids: Vec<i32>,
}

/// Read-only HUD capture switches needed during per-view recording.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerViewHudConfig {
    /// Whether the main HUD wants world-mesh stats and rows from the current view.
    pub main_enabled: bool,
    /// Whether the textures HUD wants current-view Texture2D ids.
    pub textures_enabled: bool,
}

/// Blackboard slot for per-view frame bind group and uniform buffer.
///
/// Seeded into the per-view blackboard by the executor before running per-view passes.
/// The prepare pass writes frame uniforms to the buffer backing [`PerViewFramePlan::frame_bind_group`].
pub struct PerViewFramePlanSlot;
impl BlackboardSlot for PerViewFramePlanSlot {
    type Value = PerViewFramePlan;
}

/// Blackboard slot for the live [`crate::config::GtaoSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so
/// [`crate::render_graph::passes::post_processing::gtao::GtaoPass`] reads the current slider
/// values without rebuilding the compiled render graph. Slider changes don't flip
/// [`crate::render_graph::post_processing::chain::PostProcessChainSignature`] (which tracks
/// enable flags only) — this slot is the path that propagates parameter edits into the UBO.
pub struct GtaoSettingsSlot;
impl BlackboardSlot for GtaoSettingsSlot {
    type Value = GtaoSettingsValue;
}

/// Live [`crate::config::GtaoSettings`] carried on the per-view blackboard.
///
/// Wraps `GtaoSettings` by value; the blackboard slot trait needs a concrete type living in this
/// module and the inner settings type lives in `crate::config`.
#[derive(Clone, Copy, Debug)]
pub struct GtaoSettingsValue(pub crate::config::GtaoSettings);

/// Blackboard slot for the live [`crate::config::BloomSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so the
/// bloom passes read the current slider values without rebuilding the compiled render graph.
/// Non-topology edits (intensity, low-frequency boost, threshold, composite mode, …) flow in via
/// this slot; only the effective `max_mip_dimension` changes force a rebuild because it resizes
/// the mip-chain transient textures — the chain signature tracks that value explicitly.
pub struct BloomSettingsSlot;
impl BlackboardSlot for BloomSettingsSlot {
    type Value = BloomSettingsValue;
}

/// Live [`crate::config::BloomSettings`] carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct BloomSettingsValue(pub crate::config::BloomSettings);

/// Per-view frame bind group and uniform buffer for multi-view rendering.
///
/// Each view writes its own frame-uniform data to [`Self::frame_uniform_buffer`] in the prepare
/// pass. The forward raster pass binds [`Self::frame_bind_group`] at `@group(0)` so that each
/// view's camera / cluster parameters are independent.
#[derive(Clone)]
pub struct PerViewFramePlan {
    /// `@group(0)` bind group that uses this view's dedicated frame-uniform buffer.
    pub frame_bind_group: Arc<wgpu::BindGroup>,
    /// Per-view frame uniform buffer (written by the plan pass via `Queue::write_buffer`).
    ///
    /// [`wgpu::Buffer`] is internally ref-counted, so cloning is cheap.
    pub frame_uniform_buffer: wgpu::Buffer,
    /// Index of this view in the multi-view batch (0-based).
    pub view_idx: usize,
}

/// System handles shared across all views within a frame.
///
/// Shared systems borrowed by render graph passes while recording one frame.
pub struct FrameSystemsShared<'a> {
    /// World caches and mesh renderables after [`SceneCoordinator::flush_world_caches`].
    pub scene: &'a SceneCoordinator,
    /// Hi-Z pyramid GPU/CPU state and temporal culling for this frame.
    pub occlusion: &'a OcclusionSystem,
    /// Per-frame `@group(0/1/2)` binds, lights, per-draw slab, and CPU light scratch.
    pub frame_resources: &'a FrameResourceManager,
    /// Materials registry, embedded binds, and property store.
    pub materials: &'a MaterialSystem,
    /// Mesh/texture pools and upload queues.
    pub asset_transfers: &'a AssetTransferQueue,
    /// Skinning/blendshape compute pipelines (set after GPU attach, `None` before).
    pub mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    /// Deform scratch buffers for the `MeshDeformPass` (valid during frame-global recording only).
    pub mesh_deform_scratch: Option<&'a mut MeshDeformScratch>,
    /// Deformed mesh arenas for the frame-global mesh-deform pass.
    pub mesh_deform_skin_cache: Option<&'a mut GpuSkinCache>,
    /// Deformed mesh arenas for forward draws after mesh deform completes.
    pub skin_cache: Option<&'a GpuSkinCache>,
    /// Read-only HUD capture switches for deferred per-view diagnostics.
    pub debug_hud: PerViewHudConfig,
}

/// Per-view surface and camera state for one render target within a multi-view frame.
///
/// All fields are value types or immutable references: they are derived from the resolved view
/// target before recording begins and do not change during per-view pass execution. This is the
/// primary per-view context type; [`FrameRenderParams`] remains during a staged migration.
pub struct FrameRenderParamsView<'a> {
    /// Backing depth texture for the main forward pass (copy source for scene-depth snapshots).
    pub depth_texture: &'a wgpu::Texture,
    /// Depth attachment view for the main forward pass.
    pub depth_view: &'a wgpu::TextureView,
    /// Depth-only view for compute sampling (e.g. Hi-Z build); created once per view.
    pub depth_sample_view: Option<wgpu::TextureView>,
    /// Swapchain / main color format (output / compose target).
    pub surface_format: wgpu::TextureFormat,
    /// HDR scene-color format for forward shading ([`crate::config::RenderingSettings::scene_color_format`]).
    pub scene_color_format: wgpu::TextureFormat,
    /// Main surface extent in pixels (`width`, `height`) for projection.
    pub viewport_px: (u32, u32),
    /// Clip planes, FOV, and ortho task hint from the last host frame submission.
    pub host_camera: HostCameraFrame,
    /// When `true`, the forward pass targets 2-layer array attachments and may use multiview.
    pub multiview_stereo: bool,
    /// Optional transform filter for secondary cameras (selective / exclude lists).
    pub transform_draw_filter: Option<CameraTransformDrawFilter>,
    /// When rendering a secondary camera to a host render texture, the asset id of the color
    /// target being written. Materials must not sample that texture in the same pass.
    pub offscreen_write_render_texture_asset_id: Option<i32>,
    /// Which logical view this frame state belongs to.
    pub view_id: ViewId,
    /// Mutex-wrapped Hi-Z state resolved for this view before per-view recording starts.
    pub hi_z_slot: Arc<Mutex<HiZGpuState>>,
    /// Effective raster sample count for mesh forward (1 = off). Clamped to the GPU max for this view.
    pub sample_count: u32,
    /// GPU limits after attach (`None` only before a successful attach).
    pub gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth resolve pipelines when supported (cloned from the backend attach path).
    pub msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
}

/// Compositor over [`FrameSystemsShared`] and [`FrameRenderParamsView`].
///
/// Built with disjoint borrows from [`crate::backend::RenderBackend`] so passes do not take a
/// full backend handle.
pub struct FrameRenderParams<'a> {
    /// System handles shared across all views for this frame.
    pub shared: FrameSystemsShared<'a>,
    /// Per-view surface and camera state.
    pub view: FrameRenderParamsView<'a>,
}

impl FrameRenderParams<'_> {
    /// Output depth layout for Hi-Z and occlusion ([`OutputDepthMode::from_multiview_stereo`]).
    pub fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.view.multiview_stereo)
    }

    /// Disjoint material/pool/skin borrows for world-mesh forward raster encoding.
    pub(crate) fn world_mesh_forward_encode_refs(&self) -> WorldMeshForwardEncodeRefs<'_> {
        WorldMeshForwardEncodeRefs::from_frame_params(
            self.shared.materials,
            self.shared.asset_transfers,
            self.shared.skin_cache,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameViewClear, WorldMeshHelperNeeds};
    use crate::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::shared::{CameraClearMode, CameraState};
    use crate::world_mesh::WorldMeshDrawCollection;

    #[test]
    fn main_view_clear_defaults_to_skybox() {
        let clear = FrameViewClear::default();
        assert_eq!(clear.mode, CameraClearMode::Skybox);
        assert_eq!(clear.color, glam::Vec4::ZERO);
    }

    #[test]
    fn secondary_view_clear_comes_from_camera_state() {
        let state = CameraState {
            clear_mode: CameraClearMode::Color,
            background_color: glam::Vec4::new(0.1, 0.2, 0.3, 0.4),
            ..CameraState::default()
        };
        let clear = FrameViewClear::from_camera_state(&state);
        assert_eq!(clear.mode, CameraClearMode::Color);
        assert_eq!(clear.color, glam::Vec4::new(0.1, 0.2, 0.3, 0.4));
    }

    #[test]
    fn helper_needs_are_derived_from_scene_snapshot_usage_flags() {
        let regular = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        let mut depth = regular.clone();
        depth.batch_key.embedded_uses_scene_depth_snapshot = true;
        let mut color = regular.clone();
        color.batch_key.embedded_uses_scene_color_snapshot = true;

        let collection = WorldMeshDrawCollection {
            items: vec![regular.clone()],
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds::default()
        );

        let collection = WorldMeshDrawCollection {
            items: vec![regular.clone(), depth, color],
            draws_pre_cull: 3,
            draws_culled: 0,
            draws_hi_z_culled: 0,
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                color_snapshot: true,
            }
        );

        let mut refract_like = regular;
        refract_like.batch_key.embedded_uses_scene_depth_snapshot = true;
        refract_like.batch_key.embedded_uses_scene_color_snapshot = true;
        let collection = WorldMeshDrawCollection {
            items: vec![refract_like],
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                color_snapshot: true,
            }
        );
    }
}
