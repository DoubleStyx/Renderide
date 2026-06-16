//! [`FrameResourceManager`] struct definition and construction/attach lifecycle.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use hashbrown::HashSet;
use parking_lot::Mutex;

use crate::gpu::GpuLimits;
use crate::mesh_deform::SkinCacheKey;

use super::super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::super::frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
use super::super::light_gpu::GpuLight;
use super::super::per_draw_resources::PerDrawResources;
use super::super::per_view_resource_map::PerViewResourceMap;
use super::lights::LightVisibilityStats;
use super::per_view_state::{PerViewFrameState, PerViewPerDrawScratch, PreparedViewLights};
use super::shadows::ShadowFramePlan;

/// Per-frame GPU state: shared frame/light/cluster resources, per-view bind groups,
/// per-view per-draw storage slabs, and the CPU-side packed light buffer.
pub struct FrameResourceManager {
    /// Shared `@group(0)` frame globals (lights, fallback snapshots, bind group layout).
    pub(crate) frame_gpu: Option<FrameGpuResources>,
    /// Placeholder `@group(1)` for materials without per-material bindings.
    pub(crate) empty_material: Option<EmptyMaterialBindGroup>,
    /// Per-view frame uniform buffer and `@group(0)` bind group.
    ///
    /// Created lazily on first use per [`ViewId`]; retired when a secondary RT camera
    /// is destroyed via `retire_per_view_frame`.
    pub(super) per_view_frame: PerViewResourceMap<PerViewFrameState>,
    /// One grow-on-demand per-draw slab per stable render-view identity.
    ///
    /// Created lazily; keyed by [`ViewId`] so secondary RT cameras never compete
    /// with the main view (or each other) for buffer space.
    pub(super) per_view_draw: PerViewResourceMap<Mutex<PerDrawResources>>,
    /// Shared `@group(2)` bind group layout, reflected once at attach time.
    pub(super) per_draw_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    /// GPU limits stored at attach time for lazy per-view slab/cluster creation.
    pub(super) limits: Option<Arc<GpuLimits>>,
    /// Last packed lights for the first prepared view, retained for diagnostics and fallback callers.
    pub(super) light_scratch: Vec<GpuLight>,
    /// Per-view packed light sets keyed by render view identity.
    pub(super) per_view_lights: PerViewResourceMap<PreparedViewLights>,
    /// Latest aggregate light influence-volume culling stats.
    pub(super) light_visibility_stats: LightVisibilityStats,
    /// Shadow metadata and atlas render views planned for the current graph submission.
    pub(super) shadow_frame: ShadowFramePlan,
    /// Whether any packed light set subtracts in at least one signed-radiance channel.
    pub(super) signed_scene_color_required: bool,
    /// When true, [`crate::passes::MeshDeformPass`] already dispatched for the current graph
    /// submission.
    ///
    /// Reset when a new submitted draw packet installs its visible deform set.
    pub(super) mesh_deform_dispatched_this_submission: AtomicBool,
    /// Optional visible deform filter derived from prefetched per-view draw lists.
    pub(super) visible_mesh_deform_keys: Mutex<Option<HashSet<SkinCacheKey>>>,
    /// Reused per-view scratch for per-draw VP/pack before graph upload.
    ///
    /// Each view owns its own mutex-wrapped slot so rayon workers never alias the same scratch.
    pub(super) per_view_per_draw_scratch: PerViewResourceMap<Mutex<PerViewPerDrawScratch>>,
    /// One-shot guard for the [`crate::backend::light_gpu::MAX_LIGHTS`] overflow warning so a scene
    /// with too many lights does not spam logs every frame.
    pub(super) lights_overflow_warned: bool,
    /// One-shot guard for the signed scene-color activation log.
    pub(super) signed_scene_color_required_logged: bool,
}

impl Default for FrameResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameResourceManager {
    /// Creates an empty manager with no GPU resources.
    pub fn new() -> Self {
        Self {
            frame_gpu: None,
            empty_material: None,
            per_view_frame: PerViewResourceMap::new(),
            per_view_draw: PerViewResourceMap::new(),
            per_draw_bind_group_layout: None,
            limits: None,
            light_scratch: Vec::new(),
            per_view_lights: PerViewResourceMap::new(),
            light_visibility_stats: LightVisibilityStats::default(),
            shadow_frame: ShadowFramePlan::default(),
            signed_scene_color_required: false,
            mesh_deform_dispatched_this_submission: AtomicBool::new(false),
            visible_mesh_deform_keys: Mutex::new(None),
            per_view_per_draw_scratch: PerViewResourceMap::new(),
            lights_overflow_warned: false,
            signed_scene_color_required_logged: false,
        }
    }

    /// Allocates GPU resources for this manager. Called from
    /// [`crate::backend::RenderBackend::attach`].
    ///
    /// On success, `@group(0)` / `@group(1)` / `@group(2)` layout are present.
    /// `queue` initializes fallback sampled textures used by group-0 bindings.
    /// Per-view per-draw slabs and per-view frame bind resources are created lazily on first use.
    /// On error, frame bind fields remain unset (no partial attach).
    pub fn attach(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: Arc<GpuLimits>,
    ) -> Result<(), FrameGpuBindingsError> {
        let binds = FrameGpuBindings::try_new(device, queue, Arc::clone(&limits))?;
        self.frame_gpu = Some(binds.frame_gpu);
        self.empty_material = Some(binds.empty_material);
        self.per_draw_bind_group_layout = Some(binds.per_draw_bind_group_layout);
        self.limits = Some(limits);
        Ok(())
    }
}
