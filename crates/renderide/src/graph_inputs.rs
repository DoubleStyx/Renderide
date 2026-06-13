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

use parking_lot::Mutex;

use crate::camera::{HostCameraFrame, ViewId};
use crate::color_space::DEFAULT_SKYBOX_CLEAR_COLOR;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::occlusion::OcclusionGraphHook;
use crate::occlusion::gpu::HiZGpuState;
use crate::render_graph::blackboard::blackboard_slot;
use crate::render_graph::compiled::ViewPostProcessing;
use crate::render_graph::execution_backend::{GraphAssetResources, GraphFrameResources};
use crate::scene::SceneCoordinator;
use crate::shared::{CameraClearMode, RenderingContext};

/// Offscreen target currently being written by a view.
///
/// The renderer uses this for two separate decisions: any offscreen target needs the offscreen
/// projection convention, while only host render textures need material self-sampling suppression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OffscreenWriteTarget {
    /// The view writes directly to the desktop swapchain or an external multiview target.
    #[default]
    None,
    /// The view writes to an offscreen target that is not a host render-texture asset.
    Untracked,
    /// The view writes to a host render texture with the supplied asset id and sampling policy.
    HostRenderTexture {
        /// Host render-texture asset id.
        asset_id: i32,
        /// Material sampling policy for this render texture while it is being written.
        self_sampling: RenderTextureSelfSampling,
    },
}

/// Material sampling policy for a render texture while a camera writes that same texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RenderTextureSelfSampling {
    /// Hide the render texture from materials while the view writes it.
    #[default]
    Suppress,
    /// Allow materials to sample the render texture contents completed before this view.
    AllowPreviousContents,
}

impl OffscreenWriteTarget {
    /// Builds a host render-texture target using the default same-target sampling suppression.
    #[inline]
    pub const fn host_render_texture(asset_id: i32) -> Self {
        Self::host_render_texture_with_self_sampling(asset_id, RenderTextureSelfSampling::Suppress)
    }

    /// Builds a host render-texture target with an explicit same-target sampling policy.
    #[inline]
    pub const fn host_render_texture_with_self_sampling(
        asset_id: i32,
        self_sampling: RenderTextureSelfSampling,
    ) -> Self {
        Self::HostRenderTexture {
            asset_id,
            self_sampling,
        }
    }

    /// Returns `true` when the view writes to any offscreen target.
    #[inline]
    pub const fn is_offscreen(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Applies the render-target projection convention for this write target.
    ///
    /// Offscreen color attachments are written in the host texture orientation, so their
    /// clip-space projection gets a Y flip. Screen-space consumers built from the view projection,
    /// including clustered-light froxels and frame unprojection constants, must use the same
    /// adjusted projection as the forward draw path.
    #[inline]
    pub(crate) fn render_projection(self, projection: glam::Mat4) -> glam::Mat4 {
        if self.is_offscreen() {
            offscreen_projection_y_flip() * projection
        } else {
            projection
        }
    }

    /// Returns the host render-texture asset id for this write target.
    #[inline]
    pub const fn host_render_texture_asset_id(self) -> Option<i32> {
        match self {
            Self::HostRenderTexture { asset_id, .. } => Some(asset_id),
            Self::None | Self::Untracked => None,
        }
    }

    /// Returns the same-target material sampling policy for this write target.
    #[inline]
    pub const fn render_texture_self_sampling(self) -> Option<RenderTextureSelfSampling> {
        match self {
            Self::HostRenderTexture { self_sampling, .. } => Some(self_sampling),
            Self::None | Self::Untracked => None,
        }
    }

    /// Returns `true` when material bindings should mask this render texture while rendering.
    #[inline]
    pub fn suppresses_render_texture_sampling(self, sampled_asset_id: i32) -> bool {
        self.host_render_texture_asset_id() == Some(sampled_asset_id)
            && self.render_texture_self_sampling() == Some(RenderTextureSelfSampling::Suppress)
    }
}

/// Per-view winding policy before draw-local transform parity is applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ViewWinding {
    /// Whether the camera view matrix mirrors handedness, as planar reflections do.
    mirror_reflection: bool,
}

impl ViewWinding {
    /// View policy for ordinary non-reflection cameras.
    #[inline]
    pub const fn normal() -> Self {
        Self {
            mirror_reflection: false,
        }
    }

    /// View policy for planar mirror reflection cameras.
    #[inline]
    pub const fn mirror_reflection() -> Self {
        Self {
            mirror_reflection: true,
        }
    }

    /// Returns whether final raster front-face winding must be flipped for this view.
    #[inline]
    pub const fn flips_front_face_for(self, write_target: OffscreenWriteTarget) -> bool {
        write_target.is_offscreen() ^ self.mirror_reflection
    }
}

#[inline]
fn offscreen_projection_y_flip() -> glam::Mat4 {
    glam::Mat4::from_diagonal(glam::Vec4::new(1.0, -1.0, 1.0, 1.0))
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
    #[inline]
    pub fn skybox() -> Self {
        Self {
            mode: CameraClearMode::Skybox,
            color: DEFAULT_SKYBOX_CLEAR_COLOR,
        }
    }

    /// Color clear mode with the supplied linear RGBA background.
    #[inline]
    pub fn color(color: glam::Vec4) -> Self {
        Self {
            mode: CameraClearMode::Color,
            color,
        }
    }

    /// Converts host camera state into a frame-view clear descriptor.
    #[inline]
    pub fn from_camera_state(state: &crate::shared::CameraState) -> Self {
        Self {
            mode: state.clear_mode,
            color: state.background_color,
        }
    }

    /// Converts host camera readback parameters into a frame-view clear descriptor.
    #[inline]
    pub fn from_camera_render_parameters(
        parameters: &crate::shared::CameraRenderParameters,
    ) -> Self {
        Self {
            mode: parameters.clear_mode,
            color: parameters.clear_color,
        }
    }
}

impl Default for FrameViewClear {
    #[inline]
    fn default() -> Self {
        Self::skybox()
    }
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
    pub debug_hud: crate::diagnostics::PerViewHudConfig,
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
        let expected = super::offscreen_projection_y_flip() * projection;

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
