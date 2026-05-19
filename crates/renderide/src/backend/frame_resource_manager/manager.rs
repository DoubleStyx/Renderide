//! [`FrameResourceManager`] struct definition and construction/attach lifecycle.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use hashbrown::HashSet;
use parking_lot::Mutex;

use crate::gpu::GpuLimits;
use crate::mesh_deform::SkinCacheKey;
use crate::passes::WorldMeshShadowPhaseCache;
use crate::scene::{RenderSpaceId, ResolvedLight};
use crate::shared::QualityConfig;

use super::super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::super::frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
use super::super::light_gpu::GpuLight;
use super::super::per_draw_resources::PerDrawResources;
use super::super::per_view_resource_map::PerViewResourceMap;
use super::per_view_state::{PerViewFrameState, PerViewPerDrawScratch, PreparedViewLights};

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
    /// One grow-on-demand shadow-caster per-draw slab per stable render-view identity.
    pub(super) per_view_shadow_draw: PerViewResourceMap<Mutex<PerDrawResources>>,
    /// Shared `@group(2)` bind group layout, reflected once at attach time.
    pub(super) per_draw_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    /// GPU limits stored at attach time for lazy per-view slab/cluster creation.
    pub(super) limits: Option<Arc<GpuLimits>>,
    /// Last packed lights for the first prepared view, retained for diagnostics and fallback callers.
    pub(super) light_scratch: Vec<GpuLight>,
    /// Per-view packed light sets keyed by render view identity.
    pub(super) per_view_lights: PerViewResourceMap<PreparedViewLights>,
    /// Whether any packed light set subtracts in at least one signed-radiance channel.
    pub(super) signed_scene_color_required: bool,
    /// Reused each frame to flatten all spaces' [`ResolvedLight`] before ordering and GPU pack.
    pub(super) resolved_flatten_scratch: Vec<ResolvedLight>,
    /// Reused each view to collect active render spaces that should contribute lights.
    pub(super) light_space_ids_scratch: Vec<RenderSpaceId>,
    /// Latest host quality settings affecting realtime shadow-map allocation and planning.
    pub(super) quality_config: QualityConfig,
    /// When true, [`crate::passes::MeshDeformPass`] already dispatched for the current graph
    /// submission.
    ///
    /// Reset when a new submitted draw packet installs its visible deform set.
    pub(super) mesh_deform_dispatched_this_submission: AtomicBool,
    /// Optional visible deform filter derived from prefetched per-view draw lists.
    pub(super) visible_mesh_deform_keys: Mutex<Option<HashSet<SkinCacheKey>>>,
    /// Reused per-view scratch for per-draw VP/pack before
    /// [`crate::mesh_deform::write_per_draw_uniform_slab`].
    ///
    /// Each view owns its own mutex-wrapped slot so rayon workers never alias the same scratch.
    pub(super) per_view_per_draw_scratch: PerViewResourceMap<Mutex<PerViewPerDrawScratch>>,
    /// Reused per-view scratch for shadow-caster per-draw VP packing.
    pub(super) per_view_shadow_per_draw_scratch: PerViewResourceMap<Mutex<PerViewPerDrawScratch>>,
    /// Retained per-view shadow phase cache used to avoid rebuilding caster phases per light.
    pub(super) per_view_shadow_phase_cache: PerViewResourceMap<Mutex<WorldMeshShadowPhaseCache>>,
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
            per_view_shadow_draw: PerViewResourceMap::new(),
            per_draw_bind_group_layout: None,
            limits: None,
            light_scratch: Vec::new(),
            per_view_lights: PerViewResourceMap::new(),
            signed_scene_color_required: false,
            resolved_flatten_scratch: Vec::new(),
            light_space_ids_scratch: Vec::new(),
            quality_config: QualityConfig::default(),
            mesh_deform_dispatched_this_submission: AtomicBool::new(false),
            visible_mesh_deform_keys: Mutex::new(None),
            per_view_per_draw_scratch: PerViewResourceMap::new(),
            per_view_shadow_per_draw_scratch: PerViewResourceMap::new(),
            per_view_shadow_phase_cache: PerViewResourceMap::new(),
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
