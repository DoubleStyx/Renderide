//! Shared types for mesh draw collection and GPU recording.

use crate::assets::MaterialPropertyStore;
use crate::config::RenderConfig;
use crate::gpu::{GpuMeshBuffers, PbrHostAlbedoMaterialBindKey, PipelineVariant};
use nalgebra::Matrix4;
use std::collections::HashMap;

/// Minimal context for mesh draw collection.
///
/// Used when caching results across passes so mesh and overlay passes share one collect per frame.
pub struct CollectMeshDrawsContext<'a> {
    pub(crate) session: &'a crate::session::Session,
    pub(crate) draw_batches: &'a [crate::render::SpaceDrawBatch],
    pub(crate) mesh_buffer_cache: &'a std::collections::HashMap<i32, GpuMeshBuffers>,
    pub(crate) rigid_frustum_cull_cache: &'a mut crate::render::visibility::RigidFrustumCullCache,
    pub(crate) proj: nalgebra::Matrix4<f32>,
    pub(crate) overlay_projection_override: Option<crate::render::ViewParams>,
}

/// Collected non-skinned draw for batch upload.
/// Uses mesh_asset_id for buffer lookup to avoid borrowing ctx across pass boundaries.
pub(crate) struct BatchedDraw {
    pub(crate) mesh_asset_id: i32,
    pub(crate) mvp: Matrix4<f32>,
    pub(crate) model: Matrix4<f32>,
    /// Host material asset id (submesh pairing after multi-material fan-out).
    pub(crate) material_asset_id: i32,
    pub(crate) pipeline_variant: PipelineVariant,
    pub(crate) is_overlay: bool,
    /// Per-draw stencil for GraphicsChunk masking. When `Some`, overlay uses stencil pipeline.
    pub(crate) stencil_state: Option<crate::stencil::StencilState>,
    /// Slot-0 mesh renderer property block for merged material property lookup.
    pub(crate) mesh_renderer_property_block_slot0_id: Option<i32>,
    /// Per-draw index range when splitting multi-material submeshes.
    pub(crate) submesh_index_range: Option<(u32, u32)>,
}

/// Collected skinned draw for batch upload.
/// Uses mesh_asset_id for buffer lookup to avoid borrowing ctx across pass boundaries.
pub(crate) struct SkinnedBatchedDraw {
    pub(crate) mesh_asset_id: i32,
    pub(crate) mvp: Matrix4<f32>,
    pub(crate) bone_matrices: Vec<[[f32; 4]; 4]>,
    pub(crate) blendshape_weights: Option<Vec<f32>>,
    pub(crate) num_vertices: u32,
    pub(crate) is_overlay: bool,
    /// Pipeline variant (Skinned or OverlayStencilSkinned).
    pub(crate) pipeline_variant: crate::gpu::PipelineVariant,
    /// Per-draw stencil for GraphicsChunk masking. When `Some`, overlay uses stencil pipeline.
    pub(crate) stencil_state: Option<crate::stencil::StencilState>,
    /// Per-draw index range when splitting multi-material submeshes.
    pub(crate) submesh_index_range: Option<(u32, u32)>,
}

/// Cache key for skinned bind groups.
pub(crate) type SkinnedBindGroupCacheKey = (PipelineVariant, i32);

/// Resources and per-frame tuning for PBR ray-query shadow bindings (scene group 1, bindings 5–7).
pub struct RtShadowBindParams<'a> {
    /// Uniform buffer for [`crate::gpu::pipeline::RtShadowUniforms`]; written each time a ray-query scene bind group is created.
    pub uniform_buffer: &'a wgpu::Buffer,
    pub atlas_view: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub soft_samples: u32,
    pub cone_scale: f32,
    pub shadow_mode: u32,
    pub full_viewport_width: u32,
    pub full_viewport_height: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub gbuffer_origin: [f32; 3],
}

/// Parameters for PBR scene bind group creation.
pub struct PbrSceneParams<'a> {
    pub(crate) view_position: [f32; 3],
    pub(crate) view_space_z_coeffs: [f32; 4],
    pub(crate) cluster_count_x: u32,
    pub(crate) cluster_count_y: u32,
    pub(crate) cluster_count_z: u32,
    pub(crate) near_clip: f32,
    pub(crate) far_clip: f32,
    pub(crate) light_count: u32,
    /// Matches clustered light compute and fragment cluster XY (16px tiles).
    pub(crate) viewport_width: u32,
    pub(crate) viewport_height: u32,
    pub(crate) light_buffer: &'a wgpu::Buffer,
    pub(crate) cluster_light_counts: &'a wgpu::Buffer,
    pub(crate) cluster_light_indices: &'a wgpu::Buffer,
    /// When true, mesh pass selects `*RayQuery` pipeline variants and binds the TLAS.
    pub(crate) use_ray_tracing_scene: bool,
}

/// Parameters for recording mesh draws; used to avoid borrowing ctx while encoder is active.
pub struct MeshDrawParams<'a> {
    pub(crate) pipeline_manager: &'a mut crate::gpu::PipelineManager,
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) config: &'a wgpu::SurfaceConfiguration,
    pub(crate) frame_index: u64,
    pub(crate) mesh_buffer_cache: &'a std::collections::HashMap<i32, GpuMeshBuffers>,
    /// Cache for skinned bind groups; keyed by (pipeline variant, mesh asset id).
    pub(crate) skinned_bind_group_cache: &'a mut HashMap<SkinnedBindGroupCacheKey, wgpu::BindGroup>,
    /// When true, overlay draws use depth-disabled pipelines for screen-space UI.
    pub(crate) overlay_orthographic: bool,
    /// When true, non-overlay mesh pass uses MRT pipelines (NormalDebugMRT, UvDebugMRT, SkinnedMRT).
    pub(crate) use_mrt: bool,
    /// When true, main scene non-skinned draws use PBR pipeline instead of NormalDebug.
    pub(crate) use_pbr: bool,
    /// Cluster buffers and light data for PBR. None when PBR cannot be used.
    pub(crate) pbr_scene: Option<PbrSceneParams<'a>>,
    /// Cache for PBR scene bind groups. Invalidated when light or cluster buffers change.
    pub(crate) pbr_scene_bind_group_cache:
        &'a mut HashMap<crate::gpu::PipelineVariant, wgpu::BindGroup>,
    /// Last light buffer version when cache was valid.
    pub(crate) last_pbr_scene_cache_light_version: &'a mut u64,
    /// Last cluster buffer version when cache was valid.
    pub(crate) last_pbr_scene_cache_cluster_version: &'a mut u64,
    /// Last TLAS generation when PBR scene bind groups were cached.
    pub(crate) last_pbr_scene_cache_tlas_generation: &'a mut u64,
    /// Current light buffer version (for cache invalidation).
    pub(crate) light_buffer_version: u64,
    /// Current cluster buffer version (for cache invalidation).
    pub(crate) cluster_buffer_version: u64,
    /// Current [`crate::gpu::RayTracingState::tlas_generation`] for PBR scene cache invalidation.
    pub(crate) pbr_tlas_generation: u64,
    /// TLAS for PBR ray-query scene bind group; points at [`crate::gpu::RayTracingState::tlas`].
    ///
    /// Valid only for this render pass encode: the mesh pass runs before the next frame replaces
    /// the TLAS. Used to avoid overlapping `&mut GpuState` with a `&Tlas` into the same state.
    pub(crate) pbr_tlas_ptr: Option<std::ptr::NonNull<wgpu::Tlas>>,
    /// Bind group 1 for [`crate::gpu::PipelineVariant::NormalDebugMRT`], [`crate::gpu::PipelineVariant::UvDebugMRT`], [`crate::gpu::PipelineVariant::SkinnedMRT`].
    pub(crate) mrt_gbuffer_origin_bind_group: Option<&'a wgpu::BindGroup>,
    /// Current [`crate::gpu::GpuState::rt_shadow_atlas_generation`] for PBR ray-query cache invalidation.
    pub(crate) rt_shadow_atlas_generation: u64,
    /// Last shadow-atlas generation applied to [`Self::pbr_scene_bind_group_cache`].
    pub(crate) last_pbr_scene_cache_rt_shadow_atlas_generation: &'a mut u64,
    /// When set with ray-query PBR, uploads [`crate::gpu::pipeline::RtShadowUniforms`] and binds group 1 slots 5–7.
    pub(crate) rt_shadow_bind: Option<RtShadowBindParams<'a>>,
    /// Material property store for [`PipelineVariant::Material`] host-unlit pipeline resolution.
    pub(crate) material_property_store: &'a MaterialPropertyStore,
    /// Frame render config (native UI property id maps, flags).
    pub(crate) render_config: &'a RenderConfig,
    /// Sampled scene depth bind group for native UI `OVERLAY`; overlay pass only.
    pub(crate) native_ui_scene_depth_bind: Option<&'a wgpu::BindGroup>,
    /// Asset registry for [`crate::assets::TextureAsset`] lookup during native UI draws.
    pub(crate) asset_registry: &'a crate::assets::AssetRegistry,
    /// Host Texture2D GPU cache; disjoint from other fields so recording avoids `&mut GpuState`.
    pub(crate) texture2d_gpu: &'a mut HashMap<i32, (wgpu::Texture, wgpu::TextureView)>,
    /// Last uploaded [`crate::assets::TextureAsset::data_version`] per resident GPU texture.
    pub(crate) texture2d_last_uploaded_version: &'a mut HashMap<i32, u64>,
    /// Native UI material bind groups keyed by texture/material state.
    pub(crate) material_gpu_resources: &'a mut crate::gpu::MaterialGpuResources,
    /// Cached group-0 bind groups for [`PipelineVariant::PbrHostAlbedo`] keyed by material + texture.
    pub(crate) pbr_host_albedo_bind_cache:
        &'a mut HashMap<PbrHostAlbedoMaterialBindKey, wgpu::BindGroup>,
}
