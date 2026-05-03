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

use crate::assets::AssetTransferQueue;
use crate::backend::FrameResourceManager;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::occlusion::OcclusionSystem;
use crate::occlusion::gpu::HiZGpuState;
use crate::scene::SceneCoordinator;
use crate::shared::CameraClearMode;

use super::blackboard::BlackboardSlot;
use crate::gpu::OutputDepthMode;
use crate::world_mesh::draw_prep::CameraTransformDrawFilter;

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

/// Compositor over [`FrameSystemsShared`] and [`GraphPassFrameView`].
///
/// Built with disjoint borrows from [`crate::backend::BackendGraphAccess`] so passes do not take a
/// full backend handle.
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
    use super::FrameViewClear;
    use crate::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::shared::{CameraClearMode, CameraState};
    use crate::world_mesh::{WorldMeshDrawCollection, WorldMeshHelperNeeds};

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
