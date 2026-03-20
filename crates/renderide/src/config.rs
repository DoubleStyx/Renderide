//! Render configuration types.
//!
//! Engine-agnostic configuration structures used by the renderer framework.
//!
//! ## Config loading precedence
//!
//! Use [`RenderConfig::load()`] as the single source of truth. Precedence:
//! 1. **Defaults** — hardcoded values below
//! 2. **Env vars** — override defaults (e.g. `RENDERIDE_DEBUG_BLENDSHAPES=1`, `RENDERIDE_VSYNC=1`)
//!
//! Extension point for config, feature flags.

/// Render configuration (clip planes, FOV, display settings).
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Near clip plane distance.
    pub near_clip: f32,
    /// Far clip plane distance.
    pub far_clip: f32,
    /// Desktop field of view in degrees.
    pub desktop_fov: f32,
    /// When true, the swapchain uses a vsync-aligned present mode ([`wgpu::PresentMode::AutoVsync`])
    /// to avoid tearing. When false (default), the swapchain uses [`wgpu::PresentMode::AutoNoVsync`]
    /// and the winit loop still paces redraws to the current monitor refresh rate (see
    /// [`crate::app`] `about_to_wait`), which limits CPU/GPU load without guaranteeing tear-free
    /// presentation.
    pub vsync: bool,
    /// When true, use UV debug pipeline for meshes that have UVs.
    pub use_debug_uv: bool,
    /// When true, main scene meshes use PBR pipeline instead of NormalDebug. Default true.
    pub use_pbr: bool,
    /// When true, apply the mesh root (drawable's model_matrix) to skinned MVP.
    /// Matches Unity SkinnedMeshRenderer: vertices are in mesh root local space.
    pub skinned_apply_mesh_root_transform: bool,
    /// When true, use root_bone_transform_id from BoneAssignment for root-relative bone matrices.
    /// Enables A/B testing of coordinate alignment. Default false.
    pub skinned_use_root_bone: bool,
    /// When true, log diagnostic info for the first skinned draw each frame.
    pub debug_skinned: bool,
    /// When true, log blendshape batch count and first few weights each frame.
    /// Can be enabled via RENDERIDE_DEBUG_BLENDSHAPES=1.
    pub debug_blendshapes: bool,
    /// When true, apply an extra Z flip to skinned MVP for handedness correction.
    /// Use when skinned meshes appear mirrored vs non-skinned. Default false.
    pub skinned_flip_handedness: bool,
    /// When true and ray tracing is available, RTAO (Ray-Traced Ambient Occlusion) may be used.
    /// Toggle for A/B testing. Default true.
    pub rtao_enabled: bool,
    /// RTAO strength: how much occlusion darkens the scene. 0 = no effect, 1 = full darkening.
    /// Default 0.5.
    pub rtao_strength: f32,
    /// RTAO ray max distance in world units. Rays beyond this are not considered occluded.
    /// Default 1.0.
    pub ao_radius: f32,
    /// When true, mesh draws outside the view frustum are skipped on the CPU: rigid meshes use
    /// local bounds transformed by the model matrix; non-overlay skinned meshes use a conservative
    /// world AABB derived from bone world origins (see [`crate::render::visibility::skinned`]).
    /// Default true.
    pub frustum_culling: bool,
    /// Reserved for future per-batch mesh-draw worker threads. Not active while [`crate::session::Session`]
    /// is not [`Sync`] (IPC). Disable with `RENDERIDE_PARALLEL_MESH_PREP=0` to match future defaults.
    pub parallel_mesh_draw_prep_batches: bool,
}

impl RenderConfig {
    /// Loads config from defaults, then env vars. Single source of truth for render config.
    ///
    /// Env vars: `RENDERIDE_DEBUG_BLENDSHAPES=1` enables blendshape debug logging.
    ///
    /// `RENDERIDE_NO_FRUSTUM_CULL=1` disables CPU frustum culling for rigid and skinned meshes.
    ///
    /// `RENDERIDE_PARALLEL_MESH_PREP=0` disables parallel per-batch mesh-draw collection.
    ///
    /// `RENDERIDE_NO_RTAO=1` disables RTAO even when ray tracing is available.
    ///
    /// `RENDERIDE_VSYNC=1` enables hardware vsync ([`Self::vsync`]); `RENDERIDE_VSYNC=0` forces it off.
    pub fn load() -> Self {
        let mut config = Self::default();
        if std::env::var("RENDERIDE_DEBUG_BLENDSHAPES").as_deref() == Ok("1") {
            config.debug_blendshapes = true;
        }
        if std::env::var("RENDERIDE_NO_FRUSTUM_CULL").as_deref() == Ok("1") {
            config.frustum_culling = false;
        }
        if std::env::var("RENDERIDE_PARALLEL_MESH_PREP").as_deref() == Ok("0") {
            config.parallel_mesh_draw_prep_batches = false;
        }
        if std::env::var("RENDERIDE_NO_RTAO").as_deref() == Ok("1") {
            config.rtao_enabled = false;
        }
        match std::env::var("RENDERIDE_VSYNC").as_deref() {
            Ok("1") | Ok("true") | Ok("yes") => config.vsync = true,
            Ok("0") | Ok("false") | Ok("no") => config.vsync = false,
            _ => {}
        }
        config
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            near_clip: 0.01,
            far_clip: 1024.0,
            desktop_fov: 75.0,
            vsync: false,
            use_debug_uv: false,
            use_pbr: true,
            skinned_apply_mesh_root_transform: true,
            skinned_use_root_bone: false,
            debug_skinned: false,
            debug_blendshapes: false,
            skinned_flip_handedness: false,
            rtao_enabled: true,
            rtao_strength: 1.0,
            ao_radius: 1.0,
            frustum_culling: true,
            parallel_mesh_draw_prep_batches: true,
        }
    }
}
