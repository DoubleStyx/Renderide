//! Per-frame parameters shared across render graph passes (scene, backend slices, surface state).
//!
//! Cross-pass per-view state that is too large or too volatile to live on the pass struct lives
//! in the per-view [`crate::render_graph::blackboard::Blackboard`] via typed slots defined here.
//!
//! [`FrameRenderParams`] is a thin compositor over [`FrameSystemsShared`] (once-per-frame mutable
//! system handles) and [`FrameRenderParamsView`] (per-view surface state). This separation is the
//! prerequisite for per-view parallel recording (Phase 4 milestone E): the shared handles will
//! gain interior mutability, and contexts will bind them directly without going through
//! [`FrameRenderParams`].

use std::sync::Arc;

use glam::{Mat4, Vec3};
use parking_lot::Mutex;

use crate::assets::AssetTransferQueue;
use crate::backend::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::backend::FrameResourceManager;
use crate::backend::MaterialSystem;
use crate::backend::OcclusionSystem;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::ShaderPermutation;
use crate::render_graph::occlusion::HiZGpuState;
use crate::scene::SceneCoordinator;
use crate::shared::HeadOutputDevice;

use super::blackboard::BlackboardSlot;
use super::world_mesh_draw_prep::{
    CameraTransformDrawFilter, WorldMeshDrawCollection, WorldMeshDrawItem,
};
use super::OutputDepthMode;

/// Identifies which Hi-Z / occlusion slot a view uses (main vs host render texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OcclusionViewId {
    /// Main window or OpenXR multiview (shared Hi-Z GPU state in [`crate::backend::OcclusionSystem`]).
    Main,
    /// Secondary camera writing to a host render texture (per-RT Hi-Z state).
    OffscreenRenderTexture(i32),
}

/// Latest camera-related fields from host [`crate::shared::FrameSubmitData`], updated each `frame_submit`.
#[derive(Clone, Copy, Debug)]
pub struct HostCameraFrame {
    /// Host lock-step frame index (`-1` before the first submit in standalone).
    pub frame_index: i32,
    /// Near clip distance from the host frame submission.
    pub near_clip: f32,
    /// Far clip distance from the host frame submission.
    pub far_clip: f32,
    /// Vertical field of view in **degrees** (matches host `desktopFOV`).
    pub desktop_fov_degrees: f32,
    /// Whether the host reported VR output as active for this frame.
    pub vr_active: bool,
    /// Init-time head output device selected by the host.
    pub output_device: HeadOutputDevice,
    /// `(orthographic_half_height, near, far)` from the first [`crate::shared::CameraRenderTask`] whose
    /// parameters use orthographic projection (overlay main-camera ortho override).
    pub primary_ortho_task: Option<(f32, f32, f32)>,
    /// When [`Self::vr_active`] and OpenXR supplies views, per-eye view–projection (reverse-Z), mapping
    /// **stage** space to clip. World mesh passes combine this with object transforms; the host
    /// `view_transform` is **not** multiplied again for stereo world draws (see `world_mesh_forward`).
    pub stereo_view_proj: Option<(Mat4, Mat4)>,
    /// Per-eye **view** matrices (world-to-view, with handedness fix applied) when stereo is active.
    ///
    /// Populated alongside [`Self::stereo_view_proj`] so the clustered lighting compute pass can
    /// decompose view and projection per eye without re-deriving from HMD poses.
    pub stereo_views: Option<(Mat4, Mat4)>,
    /// Legacy Unity `HeadOutput.transform` in renderer world space.
    ///
    /// Overlay render spaces are positioned relative to this transform each frame
    /// (`RenderingManager.HandleFrameUpdate -> RenderSpace.UpdateOverlayPositioning`).
    pub head_output_transform: Mat4,
    /// When set, non-VR mesh forward uses this world-to-view instead of the main render-space view.
    ///
    /// Used for secondary (render-texture) cameras so [`super::passes::world_mesh_forward::vp::compute_per_draw_vp_triple`]
    /// matches the offscreen projection, and so CPU frustum and Hi-Z temporal culling
    /// ([`super::world_mesh_cull`]) use the same world-to-view as the depth pyramid author pass.
    pub secondary_camera_world_to_view: Option<Mat4>,
    /// Optional override for cluster + forward projection (reverse-Z perspective or ortho).
    ///
    /// When both [`Self::cluster_view_override`] and [`Self::cluster_proj_override`] are set,
    /// [`super::cluster_frame::cluster_frame_params`] uses them instead of the desktop main-space camera.
    pub cluster_view_override: Option<Mat4>,
    /// Optional override projection for clustered light assignment (reverse-Z).
    pub cluster_proj_override: Option<Mat4>,
    /// World position for `@group(0)` camera uniforms when the secondary camera is active.
    pub secondary_camera_world_position: Option<Vec3>,
    /// Skips Hi-Z temporal state and uses uncull or frustum-only paths for secondary RT passes.
    pub suppress_occlusion_temporal: bool,
}

impl Default for HostCameraFrame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            near_clip: 0.01,
            far_clip: 10_000.0,
            desktop_fov_degrees: 60.0,
            vr_active: false,
            output_device: HeadOutputDevice::Screen,
            primary_ortho_task: None,
            stereo_view_proj: None,
            stereo_views: None,
            head_output_transform: Mat4::IDENTITY,
            secondary_camera_world_to_view: None,
            cluster_view_override: None,
            cluster_proj_override: None,
            secondary_camera_world_position: None,
            suppress_occlusion_temporal: false,
        }
    }
}

/// Pipeline state resolved during world-mesh forward preparation.
pub struct WorldMeshForwardPipelineState {
    /// Whether this view records multiview raster passes.
    pub use_multiview: bool,
    /// Material pipeline descriptor for this view's color/depth/sample state.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation used by material pipeline lookup.
    pub shader_perm: ShaderPermutation,
}

/// Per-view forward-pass preparation shared by future split graph nodes.
pub struct PreparedWorldMeshForwardFrame {
    /// Sorted world mesh draw items for this view.
    pub draws: Vec<WorldMeshDrawItem>,
    /// Draw indices that can be recorded in the opaque forward pass.
    pub regular_indices: Vec<usize>,
    /// Draw indices that need the post-depth-snapshot intersection pass.
    pub intersect_indices: Vec<usize>,
    /// Pipeline format/sample/multiview state.
    pub pipeline: WorldMeshForwardPipelineState,
    /// Whether indexed draws may use base instance.
    pub supports_base_instance: bool,
    /// Whether the opaque/clear forward subpass was already recorded by a split graph node.
    pub opaque_recorded: bool,
    /// Whether the scene-depth snapshot for intersection draws was already recorded by a split graph node.
    pub depth_snapshot_recorded: bool,
    /// Whether the intersection/color-resolve tail raster was already recorded by a split graph node.
    pub tail_raster_recorded: bool,
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
    type Value = WorldMeshDrawCollection;
}

/// Blackboard slot for precomputed per-batch `@group(1)` material bind groups.
///
/// Populated by [`crate::render_graph::passes::WorldMeshForwardPreparePass`] during the planning
/// phase. Each entry covers a contiguous range of sorted draws with the same material batch key.
/// The recording loop reads from this slot instead of performing per-batch LRU lookups.
pub struct PrecomputedMaterialBindsSlot;
impl BlackboardSlot for PrecomputedMaterialBindsSlot {
    type Value = Vec<PrecomputedMaterialBind>;
}

/// One precomputed `@group(1)` bind group covering a batch range in the sorted draw list.
#[derive(Clone)]
pub struct PrecomputedMaterialBind {
    /// First draw index (into the sorted draw list) covered by this bind group.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this bind group.
    pub last_draw_idx: usize,
    /// Resolved `@group(1)` bind group for this batch's material, or `None` for the empty fallback.
    pub bind_group: Option<std::sync::Arc<wgpu::BindGroup>>,
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
    pub world_mesh_draw_stats: Option<crate::render_graph::WorldMeshDrawStats>,
    /// Latest world-mesh draw-state rows for the view when the main HUD is enabled.
    pub world_mesh_draw_state_rows: Option<Vec<crate::render_graph::WorldMeshDrawStateRow>>,
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

/// Per-view frame bind group and uniform buffer for multi-view rendering.
///
/// Each view writes its own frame-uniform data to [`Self::frame_uniform_buffer`] in the prepare
/// pass. The forward raster pass binds [`Self::frame_bind_group`] at `@group(0)` so that each
/// view's camera / cluster parameters are independent.
#[derive(Clone)]
pub struct PerViewFramePlan {
    /// `@group(0)` bind group that uses this view's dedicated frame-uniform buffer.
    pub frame_bind_group: std::sync::Arc<wgpu::BindGroup>,
    /// Per-view frame uniform buffer (written by the plan pass via `Queue::write_buffer`).
    ///
    /// [`wgpu::Buffer`] is internally ref-counted, so cloning is cheap.
    pub frame_uniform_buffer: wgpu::Buffer,
    /// Index of this view in the multi-view batch (0-based).
    pub view_idx: usize,
}

/// System handles shared across all views within a frame.
///
/// Fields hold mutable references during milestones B–D. Phase 4 milestone E converts
/// mutation-at-record-time fields to interior-mut wrappers (`Mutex` / atomics) so that
/// multiple rayon workers can safely record different views concurrently.
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
    /// Which Hi-Z pyramid / temporal slot this view reads and writes.
    pub occlusion_view: OcclusionViewId,
    /// Mutex-wrapped Hi-Z state resolved for this view before per-view recording starts.
    pub hi_z_slot: Arc<Mutex<HiZGpuState>>,
    /// Effective raster sample count for mesh forward (1 = off). Clamped to the GPU max for this view.
    pub sample_count: u32,
    /// GPU limits after attach (`None` only before a successful attach).
    pub gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth resolve pipelines when supported (cloned from the backend attach path).
    pub msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
}

/// Transitional compositor over [`FrameSystemsShared`] and [`FrameRenderParamsView`].
///
/// Built with disjoint borrows from [`crate::backend::RenderBackend`] so passes do not take a
/// full backend handle. Removed after Phase 4 milestone E when the parallel path constructs
/// each sub-struct independently without this wrapper.
pub struct FrameRenderParams<'a> {
    /// System handles shared across all views for this frame.
    pub shared: FrameSystemsShared<'a>,
    /// Per-view surface and camera state.
    pub view: FrameRenderParamsView<'a>,
}

impl<'a> FrameRenderParams<'a> {
    /// Output depth layout for Hi-Z and occlusion ([`OutputDepthMode::from_multiview_stereo`]).
    pub fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.view.multiview_stereo)
    }

    /// Disjoint material/pool/skin borrows for world-mesh forward raster encoding.
    pub(crate) fn world_mesh_forward_encode_refs(&mut self) -> WorldMeshForwardEncodeRefs<'_> {
        WorldMeshForwardEncodeRefs::from_frame_params(
            self.shared.materials,
            self.shared.asset_transfers,
            self.shared.skin_cache,
        )
    }
}
