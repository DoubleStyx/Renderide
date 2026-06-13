//! Per-frame parameters shared across render graph passes (scene, backend slices, surface state).
//!
//! Cross-pass per-view state that is too large or too volatile to live on the pass struct lives
//! in the per-view [`crate::render_graph::blackboard::Blackboard`] via typed slots defined here.
//!
//! [`GraphPassFrame`] is a thin compositor over [`FrameSystemsShared`] (once-per-frame system
//! handles) and [`GraphPassFrameView`] (per-view surface state). This separation keeps the
//! record path focused on view-local data while shared systems are borrowed through explicit
//! fields.

use std::sync::Arc;

use hashbrown::HashSet;
use parking_lot::Mutex;

use crate::blackboard_contract::blackboard_slot;
use crate::camera::{HostCameraFrame, ViewId};
pub(crate) use crate::frame_contract::{
    FrameViewClear, OffscreenWriteTarget, RenderTextureSelfSampling, ViewPostProcessing,
    ViewWinding,
};
use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::frame_globals::SkyboxSpecularUniformParams;
use crate::gpu::{GpuLimits, GpuRetainedResources, MsaaDepthResolveResources};
use crate::gpu_pools::{
    CubemapPool, MeshPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};
use crate::hud_contract::{PerViewHudConfig, PerViewHudOutputs};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{
    GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines, PaddedPerDrawUniforms, SkinCacheKey,
};
use crate::occlusion::OcclusionGraphHook;
use crate::occlusion::gpu::HiZGpuState;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

/// Cloned references to the shared clustered-light storage buffers.
#[derive(Clone)]
pub struct GraphClusterBufferRefs {
    /// Two `u32` words per cluster: compact-index offset and count.
    pub cluster_light_counts: wgpu::Buffer,
    /// Compact light-index storage addressed by each cluster range row.
    pub cluster_light_indices: wgpu::Buffer,
}

/// Parameters required to encode the frame-global realtime shadow atlas.
pub struct ShadowAtlasEncodeParams<'a, 'encoder, 'upload> {
    /// WGPU device used for lazy pipeline creation.
    pub device: &'a wgpu::Device,
    /// Command encoder receiving shadow render passes.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Material registry, embedded binds, and property store.
    pub materials: &'a MaterialSystem,
    /// Resident asset/resource pools.
    pub asset_resources: &'a dyn GraphAssetResources,
    /// Optional skin cache populated by frame-global mesh deform.
    pub skin_cache: Option<&'a GpuSkinCache>,
    /// GPU limits snapshot for base-instance and capacity decisions.
    pub gpu_limits: &'a GpuLimits,
    /// Deferred upload sink for the shadow per-draw slab.
    pub uploads: GraphUploadSink<'upload>,
    /// Optional GPU profiler handle.
    pub profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Graph-facing access to renderer frame resources.
pub trait GraphFrameResources: Send + Sync {
    /// Whether frame-global GPU resources were attached.
    fn has_frame_gpu(&self) -> bool;

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

    /// Per-view frame bind group that binds named scene-color snapshots at the grab slots.
    fn per_view_named_scene_color_frame_bind_group(
        &self,
        view_id: ViewId,
    ) -> Option<Arc<wgpu::BindGroup>>;

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
        f: &mut dyn FnMut(&mut Vec<PaddedPerDrawUniforms>),
    ) -> bool;

    /// Gives callers mutable access to the per-view material-batch boundary scratch so it can be
    /// cleared and refilled without reallocating.
    #[expect(
        clippy::type_complexity,
        reason = "callback Vec element type cannot be hoisted through the graph_inputs boundary"
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

    /// Copies the current HDR scene color into this view's named scene-color snapshot.
    fn copy_named_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool;

    /// Uniform parameters for the active skybox/reflection-probe specular source.
    fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams;

    /// Whether mesh deform has already recorded work for this graph submission.
    fn mesh_deform_dispatched_this_submission(&self) -> bool;

    /// Marks mesh deform work as recorded for this graph submission.
    fn set_mesh_deform_dispatched_this_submission(&self);

    /// Cloned visible mesh-deform filter for this submission's frame-global deform collection.
    fn visible_mesh_deform_keys_snapshot(&self) -> Option<HashSet<SkinCacheKey>>;

    /// Whether a named frame-global pass has no work for the current graph submission.
    fn frame_global_pass_is_inactive(&self, pass_name: &str) -> bool;

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

    /// Retains frame-owned GPU handles that may be referenced by recorded command buffers.
    fn retain_submit_resources(&self, resources: &mut GpuRetainedResources);

    /// Whether any light-cookie atlas layers need frame-global synchronization.
    fn has_light_cookie_requests(&self) -> bool;

    /// Records light-cookie atlas clears and source blits.
    fn encode_light_cookie_atlas(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        asset_resources: &dyn GraphAssetResources,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    );

    /// Whether realtime shadow atlas rendering has work.
    fn has_shadow_atlas_requests(&self) -> bool;

    /// Records realtime shadow atlas layers.
    fn encode_shadow_atlas(&self, params: ShadowAtlasEncodeParams<'_, '_, '_>);
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

blackboard_slot! {
    /// Blackboard slot for per-view HUD data collected during recording and merged on the main thread.
    pub PerViewHudOutputsSlot => PerViewHudOutputs,
}

blackboard_slot! {
    /// Blackboard slot for per-view MSAA attachment views resolved from transient graph resources.
    ///
    /// Populated by the executor (before per-view passes run) from
    /// `resolve_forward_msaa_views_from_graph_resources` output.
    /// Replaces the six `msaa_*` fields that previously lived on [`GraphPassFrame`].
    pub MsaaViewsSlot => MsaaViews,
}

/// MSAA attachment views for the forward pass (resolved from graph transient textures).
///
/// Read by the world-mesh forward depth-snapshot and depth-resolve helpers via the per-view
/// blackboard ([`MsaaViewsSlot`]). Depth views are produced with `DepthOnly` aspect so they are
/// directly bindable as `texture_multisampled_2d<f32>` in the MSAA depth-resolve compute shader.
#[derive(Clone)]
pub struct MsaaViews {
    /// Depth-resolve views matching the active mono or stereo MSAA target shape.
    pub depth_resolve: MsaaDepthResolveViews,
}

/// Valid depth-resolve view bundle for mono or stereo MSAA.
#[derive(Clone)]
pub enum MsaaDepthResolveViews {
    /// Mono view uses one multisampled depth texture and one R32Float intermediate.
    Mono {
        /// Multisampled depth attachment view.
        msaa_depth_view: wgpu::TextureView,
        /// R32Float intermediate view used by the MSAA depth resolve path.
        r32_view: wgpu::TextureView,
    },
    /// Stereo multiview uses per-eye single-layer depth/R32 views and an array R32 output.
    Stereo(Box<MsaaStereoDepthResolveViews>),
}

/// Stereo depth-resolve views for a multiview MSAA forward target.
#[derive(Clone)]
pub struct MsaaStereoDepthResolveViews {
    /// Per-eye single-layer views of stereo MSAA depth.
    pub msaa_depth_layer_views: [wgpu::TextureView; 2],
    /// Per-eye single-layer views of stereo R32Float resolve targets.
    pub r32_layer_views: [wgpu::TextureView; 2],
    /// Two-layer R32Float array view used by the stereo resolve path.
    pub r32_array_view: wgpu::TextureView,
}

blackboard_slot! {
    /// Blackboard slot for per-view frame bind group and uniform buffer.
    ///
    /// Seeded into the per-view blackboard by the executor before running per-view passes.
    /// Backend world-mesh frame planning writes frame uniforms to the buffer backing
    /// [`PerViewFramePlan::frame_bind_group`].
    pub PerViewFramePlanSlot => PerViewFramePlan,
}

/// Per-view frame bind group and uniform buffer for multi-view rendering.
///
/// Each view writes its own frame-uniform data to [`Self::frame_uniform_buffer`] in the prepare
/// pass. The forward raster pass binds [`Self::frame_bind_group`] at `@group(0)` so that each
/// view's camera / cluster parameters are independent.
#[derive(Clone)]
pub struct PerViewFramePlan {
    /// `@group(0)` bind group that uses this view's dedicated frame-uniform buffer.
    pub frame_bind_group: Arc<wgpu::BindGroup>,
    /// Per-view frame uniform buffer (written by the plan pass via the graph upload sink).
    ///
    /// [`wgpu::Buffer`] is internally ref-counted, so cloning is cheap.
    pub frame_uniform_buffer: wgpu::Buffer,
    /// Index of this view in the multi-view batch (0-based).
    pub view_idx: usize,
}

/// Frame-resource layout needed before graph recording starts for one view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreRecordViewResourceLayout {
    /// Stable identity for the view that owns this layout.
    pub view_id: ViewId,
    /// Viewport width in physical pixels.
    pub width: u32,
    /// Viewport height in physical pixels.
    pub height: u32,
    /// Whether this view records as a two-layer multiview target.
    pub stereo: bool,
    /// Forward-pass sample count for this view.
    pub sample_count: u32,
    /// Depth snapshot format for material scene-depth sampling.
    pub depth_format: wgpu::TextureFormat,
    /// HDR scene-color snapshot format for grab-pass material sampling.
    pub color_format: wgpu::TextureFormat,
    /// Whether this view has materials that need a full-size scene-depth snapshot.
    pub needs_depth_snapshot: bool,
    /// Whether this view has materials that need a full-size scene-color snapshot.
    pub needs_color_snapshot: bool,
}

/// Opaque read-only scene borrow threaded through the render graph executor.
///
/// The `render_graph` layer never inspects the scene; it only carries this token from the
/// backend entry point to the [`FrameSystemsShared`] construction sites, where passes regain
/// typed access via [`FrameSystemsShared::scene`]. Keeping the executor scene-opaque makes the
/// "generic graph primitives" layering mechanically true: `render_graph` code cannot name
/// [`SceneCoordinator`] without going through this wrapper.
#[derive(Clone, Copy)]
pub struct GraphSceneView<'a> {
    /// Flushed scene coordinator for the frame being recorded.
    coordinator: &'a SceneCoordinator,
}

impl<'a> GraphSceneView<'a> {
    /// Wraps the flushed scene for one frame's graph execution.
    pub(crate) fn new(coordinator: &'a SceneCoordinator) -> Self {
        Self { coordinator }
    }

    /// Unwraps the typed scene borrow for pass-facing frame contracts.
    pub(crate) fn coordinator(self) -> &'a SceneCoordinator {
        self.coordinator
    }
}

/// System handles shared across all views within a frame.
///
/// Shared systems borrowed by render graph passes while recording one frame.
pub struct FrameSystemsShared<'a> {
    /// World caches and mesh renderables after [`SceneCoordinator::flush_world_caches`].
    pub scene: &'a SceneCoordinator,
    /// Hi-Z pyramid GPU/CPU state and temporal culling for this frame.
    pub occlusion: &'a dyn OcclusionGraphHook,
    /// Per-frame `@group(0/1/2)` binds, lights, per-draw slab, and CPU light scratch.
    pub frame_resources: &'a dyn GraphFrameResources,
    /// Materials registry, embedded binds, and property store.
    pub materials: &'a MaterialSystem,
    /// Mesh/texture pools and upload queues.
    pub asset_resources: &'a dyn GraphAssetResources,
    /// Skinning/blendshape compute pipelines (set after GPU attach, `None` before).
    pub mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    /// Deform scratch buffers for the `MeshDeformPass` (valid during frame-global recording only).
    pub mesh_deform_scratch: Option<&'a mut MeshDeformScratch>,
    /// Deformed mesh arenas for the frame-global mesh-deform pass.
    pub mesh_deform_skin_cache: Option<&'a mut GpuSkinCache>,
    /// Deformed mesh arenas for forward draws after mesh deform completes.
    pub skin_cache: Option<&'a GpuSkinCache>,
    /// Host-owned skin influence mode for mesh deform compute.
    pub skin_weight_mode: crate::shared::SkinWeightMode,
    /// Read-only HUD capture switches for deferred per-view diagnostics.
    pub debug_hud: PerViewHudConfig,
}

/// Per-view surface and camera state for one render target within a multi-view frame.
///
/// All fields are value types or immutable references: they are derived from the resolved view
/// target before recording begins and do not change during per-view pass execution. This is the
/// primary per-view context type; [`GraphPassFrame`] remains during a staged migration.
pub struct GraphPassFrameView<'a> {
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
    /// Render-context override scope used for transforms, materials, lights, and draw matrices.
    pub render_context: RenderingContext,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub frame_time_seconds: f32,
    /// When `true`, the forward pass targets 2-layer array attachments and may use multiview.
    pub multiview_stereo: bool,
    /// Offscreen target currently being written by this view.
    pub offscreen_write_target: OffscreenWriteTarget,
    /// Per-view winding policy used by raster pipeline key resolution.
    pub view_winding: ViewWinding,
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
    /// Post-processing permissions requested by this view.
    pub post_processing: ViewPostProcessing,
}

/// Compositor over [`FrameSystemsShared`] and [`GraphPassFrameView`].
///
/// Built with disjoint graph-facing borrows so passes do not take a full backend handle.
pub struct GraphPassFrame<'a> {
    /// System handles shared across all views for this frame.
    pub shared: FrameSystemsShared<'a>,
    /// Per-view surface and camera state.
    pub view: GraphPassFrameView<'a>,
}

#[cfg(test)]
mod tests {
    use super::{FrameViewClear, OffscreenWriteTarget, RenderTextureSelfSampling, ViewWinding};
    use crate::shared::{CameraClearMode, CameraState};

    #[test]
    fn offscreen_write_target_separates_projection_and_self_sampling() {
        assert!(!OffscreenWriteTarget::None.is_offscreen());
        assert_eq!(
            OffscreenWriteTarget::None.host_render_texture_asset_id(),
            None
        );

        assert!(OffscreenWriteTarget::Untracked.is_offscreen());
        assert_eq!(
            OffscreenWriteTarget::Untracked.host_render_texture_asset_id(),
            None
        );

        let host_target = OffscreenWriteTarget::host_render_texture(77);
        assert!(host_target.is_offscreen());
        assert_eq!(host_target.host_render_texture_asset_id(), Some(77));
        assert_eq!(
            host_target.render_texture_self_sampling(),
            Some(RenderTextureSelfSampling::Suppress)
        );
        assert!(host_target.suppresses_render_texture_sampling(77));
        assert!(!host_target.suppresses_render_texture_sampling(78));
    }

    #[test]
    fn offscreen_write_target_allows_previous_contents_for_self_sampling() {
        let host_target = OffscreenWriteTarget::host_render_texture_with_self_sampling(
            77,
            RenderTextureSelfSampling::AllowPreviousContents,
        );

        assert!(host_target.is_offscreen());
        assert_eq!(host_target.host_render_texture_asset_id(), Some(77));
        assert_eq!(
            host_target.render_texture_self_sampling(),
            Some(RenderTextureSelfSampling::AllowPreviousContents)
        );
        assert!(!host_target.suppresses_render_texture_sampling(77));
    }

    #[test]
    fn offscreen_write_target_flips_render_projection_y() {
        let projection = glam::Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let expected = glam::Mat4::from_diagonal(glam::Vec4::new(1.0, -1.0, 1.0, 1.0)) * projection;

        assert_eq!(
            OffscreenWriteTarget::host_render_texture(77).render_projection(projection),
            expected
        );
    }

    #[test]
    fn primary_write_target_keeps_render_projection_unchanged() {
        let projection = glam::Mat4::from_scale(glam::Vec3::new(2.0, 3.0, 4.0));

        assert_eq!(
            OffscreenWriteTarget::None.render_projection(projection),
            projection
        );
    }

    #[test]
    fn view_winding_combines_offscreen_and_reflection_parity() {
        assert!(!ViewWinding::normal().flips_front_face_for(OffscreenWriteTarget::None));
        assert!(
            ViewWinding::normal()
                .flips_front_face_for(OffscreenWriteTarget::host_render_texture(77))
        );
        assert!(ViewWinding::mirror_reflection().flips_front_face_for(OffscreenWriteTarget::None));
        assert!(
            !ViewWinding::mirror_reflection()
                .flips_front_face_for(OffscreenWriteTarget::host_render_texture(77))
        );
    }

    #[test]
    fn main_view_clear_defaults_to_skybox() {
        let clear = FrameViewClear::default();
        assert_eq!(clear.mode, CameraClearMode::Skybox);
        assert_eq!(clear.color, glam::Vec4::new(0.1, 0.1, 0.1, 1.0));
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
}
