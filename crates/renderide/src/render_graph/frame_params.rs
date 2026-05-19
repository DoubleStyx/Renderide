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
use crate::scene::SceneCoordinator;
use crate::shared::{CameraClearMode, RenderingContext};

use super::blackboard::BlackboardSlot;
use super::compiled::ViewPostProcessing;
use super::execution_backend::{GraphAssetResources, GraphFrameResources};
use crate::gpu::OutputDepthMode;

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
            color: DEFAULT_SKYBOX_CLEAR_COLOR,
        }
    }

    /// Color clear mode with the supplied linear RGBA background.
    #[cfg(test)]
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

    /// Converts host camera readback parameters into a frame-view clear descriptor.
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
    fn default() -> Self {
        Self::skybox()
    }
}

/// Blackboard slot for per-view MSAA attachment views resolved from transient graph resources.
///
/// Populated by the executor (before per-view passes run) from
/// [`super::compiled::helpers::populate_forward_msaa_from_graph_resources`] output.
/// Replaces the six `msaa_*` fields that previously lived on [`GraphPassFrame`].
pub struct MsaaViewsSlot;
impl BlackboardSlot for MsaaViewsSlot {
    type Value = MsaaViews;
}

/// MSAA attachment views for the forward pass (resolved from graph transient textures).
///
/// Read by the world-mesh forward depth-snapshot and depth-resolve helpers via the per-view
/// blackboard ([`MsaaViewsSlot`]). Depth views are produced with `DepthOnly` aspect so they are
/// directly bindable as `texture_multisampled_2d<f32>` in the MSAA depth-resolve compute shader.
#[derive(Clone)]
pub struct MsaaViews {
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

/// Blackboard slot for per-view frame bind group and uniform buffer.
///
/// Seeded into the per-view blackboard by the executor before running per-view passes.
/// Backend world-mesh frame planning writes frame uniforms to the buffer backing
/// [`PerViewFramePlan::frame_bind_group`].
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
    /// When `true`, the forward pass targets 2-layer array attachments and may use multiview.
    pub multiview_stereo: bool,
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

impl GraphPassFrame<'_> {
    /// Output depth layout for Hi-Z and occlusion ([`OutputDepthMode::from_multiview_stereo`]).
    pub fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.view.multiview_stereo)
    }
}

#[cfg(test)]
mod tests {
    use super::FrameViewClear;
    use crate::shared::{CameraClearMode, CameraState};

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
