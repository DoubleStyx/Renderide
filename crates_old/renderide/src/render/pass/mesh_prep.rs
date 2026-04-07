//! CPU-side mesh draw preparation for the render graph: buffer ensures, projection, and draw collection.
//!
//! Used from the collect phase ([`prepare_mesh_draws_for_view`]) and from [`super::graph::RenderGraph::execute`]
//! when pre-collected draws are not supplied.
//!
//! ## Manual Canvas check (native UI)
//!
//! - Overlay: `native_ui_world_space` usually off; world UI: enable [`crate::config::RenderConfig::native_ui_world_space`].
//! - `use_native_ui_wgsl` on; shader allowlist matches host `set_shader` id (INI, env, or upload hint + optional force).
//! - `set_texture_2d_data` uses shared memory when the host supplies pixels; decode supports the atlas format.
//! - [`prefetch_native_ui_texture2d_gpu`] ensures `Texture2D` GPU uploads for native UI draws **and** for
//!   every material in the store classified as `UI_Unlit` (material-only `_MainTex` / `_MaskTex`), not only
//!   for meshes drawn this frame—so inventory and draw-time paths agree when assets are
//!   [`TextureAsset::ready_for_gpu`](crate::assets::TextureAsset::ready_for_gpu).
//! - After a frame with data, GPU texture residency should reflect UI slots (prefetch + draw-time ensure).

use std::collections::HashSet;

use nalgebra::Matrix4;

use super::material_draw_context::MaterialDrawContext;
use super::mesh_draw::{CollectMeshDrawsContext, collect_mesh_draws};
use crate::assets::texture2d_asset_id_from_packed;
use crate::assets::{
    MaterialPropertyLookupIds, NativeShaderRoute, resolve_native_shader_route,
    ui_text_unlit_material_uniform, ui_unlit_material_uniform,
};
use crate::gpu::GpuState;
use crate::gpu::PipelineVariant;
use crate::render::batch::SpaceDrawBatch;
use crate::render::view::ViewParams;
use crate::session::Session;

/// Cached mesh draws: (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned).
pub(crate) type CachedMeshDraws = (
    Vec<super::mesh_draw::SkinnedBatchedDraw>,
    Vec<super::mesh_draw::SkinnedBatchedDraw>,
    Vec<super::mesh_draw::BatchedDraw>,
    Vec<super::mesh_draw::BatchedDraw>,
);

/// Reference to cached mesh draws for render pass context.
pub(crate) type CachedMeshDrawsRef<'a> = (
    &'a [super::mesh_draw::SkinnedBatchedDraw],
    &'a [super::mesh_draw::SkinnedBatchedDraw],
    &'a [super::mesh_draw::BatchedDraw],
    &'a [super::mesh_draw::BatchedDraw],
);

/// CPU mesh-draw prep counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshDrawPrepStats {
    /// Total draws visited across all batches before mesh/GPU validation.
    pub total_input_draws: usize,
    /// Total non-skinned draws visited.
    pub rigid_input_draws: usize,
    /// Total skinned draws visited.
    pub skinned_input_draws: usize,
    /// Submitted rigid draws after CPU culling/validation.
    pub submitted_rigid_draws: usize,
    /// Submitted skinned draws after validation.
    pub submitted_skinned_draws: usize,
    /// Rigid draws rejected by CPU frustum culling.
    pub frustum_culled_rigid_draws: usize,
    /// Skinned draws rejected by CPU frustum culling (bone-position AABB test).
    pub frustum_culled_skinned_draws: usize,
    /// Rigid draws kept because upload bounds were degenerate, so culling was skipped.
    pub skipped_cull_degenerate_bounds: usize,
    /// Draws skipped because `mesh_asset_id < 0`.
    pub skipped_invalid_mesh_asset_id: usize,
    /// Draws skipped because the mesh asset was not found.
    pub skipped_missing_mesh_asset: usize,
    /// Draws skipped because the mesh had no vertices or indices.
    pub skipped_empty_mesh: usize,
    /// Draws skipped because GPU buffers were not resident.
    pub skipped_missing_gpu_buffers: usize,
    /// Skinned draws skipped because bind poses were missing.
    pub skipped_skinned_missing_bind_poses: usize,
    /// Skinned draws skipped because bone IDs were missing or empty.
    pub skipped_skinned_missing_bone_ids: usize,
    /// Skinned draws skipped because bone ID count exceeded bind pose count.
    pub skipped_skinned_id_count_mismatch: usize,
    /// Skinned draws skipped because the skinned vertex buffer was missing.
    pub skipped_skinned_missing_vertex_buffer: usize,
}

impl MeshDrawPrepStats {
    /// Total draws submitted after prep.
    pub fn submitted_draws(&self) -> usize {
        self.submitted_rigid_draws + self.submitted_skinned_draws
    }

    /// Merges per-batch stats into running totals (used by [`super::mesh_draw::collect_mesh_draws`]).
    pub(crate) fn accumulate(&mut self, other: &Self) {
        self.total_input_draws += other.total_input_draws;
        self.rigid_input_draws += other.rigid_input_draws;
        self.skinned_input_draws += other.skinned_input_draws;
        self.submitted_rigid_draws += other.submitted_rigid_draws;
        self.submitted_skinned_draws += other.submitted_skinned_draws;
        self.frustum_culled_rigid_draws += other.frustum_culled_rigid_draws;
        self.frustum_culled_skinned_draws += other.frustum_culled_skinned_draws;
        self.skipped_cull_degenerate_bounds += other.skipped_cull_degenerate_bounds;
        self.skipped_invalid_mesh_asset_id += other.skipped_invalid_mesh_asset_id;
        self.skipped_missing_mesh_asset += other.skipped_missing_mesh_asset;
        self.skipped_empty_mesh += other.skipped_empty_mesh;
        self.skipped_missing_gpu_buffers += other.skipped_missing_gpu_buffers;
        self.skipped_skinned_missing_bind_poses += other.skipped_skinned_missing_bind_poses;
        self.skipped_skinned_missing_bone_ids += other.skipped_skinned_missing_bone_ids;
        self.skipped_skinned_id_count_mismatch += other.skipped_skinned_id_count_mismatch;
        self.skipped_skinned_missing_vertex_buffer += other.skipped_skinned_missing_vertex_buffer;
    }
}

/// Runs mesh-draw CPU collection for the main view and graph fallback paths.
///
/// [`Session`] is not [`Sync`] today (IPC queues), so per-batch worker threads cannot safely share
/// `&Session` yet. [`RenderConfig::parallel_mesh_draw_prep_batches`](crate::config::RenderConfig::parallel_mesh_draw_prep_batches) is reserved for when prep uses
/// owned snapshots or the session becomes shareable for read-only prep.
pub(super) fn run_collect_mesh_draws(
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    gpu: &mut crate::gpu::GpuState,
    proj: Matrix4<f32>,
    overlay_projection_override: Option<ViewParams>,
) -> (CachedMeshDraws, MeshDrawPrepStats) {
    let mut collect_ctx = CollectMeshDrawsContext {
        session,
        draw_batches,
        mesh_buffer_cache: &gpu.mesh_buffer_cache,
        rigid_frustum_cull_cache: &mut gpu.rigid_frustum_cull_cache,
        proj,
        overlay_projection_override,
    };
    let (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned, stats) =
        collect_mesh_draws(&mut collect_ctx);
    (
        (
            non_overlay_skinned,
            overlay_skinned,
            non_overlay_non_skinned,
            overlay_non_skinned,
        ),
        stats,
    )
}

/// Pre-collected mesh draws and view parameters for the main view.
///
/// Produced by [`prepare_mesh_draws_for_view`] during the collect phase for the same
/// render extent as the [`crate::render::RenderTarget`] passed into [`crate::render::RenderLoop::render_frame`]
/// (typically the acquired swapchain texture size, not window client area alone).
pub struct PreCollectedFrameData {
    /// Primary projection matrix for the main view.
    pub proj: Matrix4<f32>,
    /// Overlay projection override when overlays use orthographic.
    pub overlay_projection_override: Option<ViewParams>,
    /// Cached mesh draws for mesh and overlay passes.
    pub(crate) cached_mesh_draws: CachedMeshDraws,
    /// CPU-side mesh draw preparation counters for diagnostics.
    pub prep_stats: MeshDrawPrepStats,
}

/// Prepares mesh draws for the main view during the collect phase.
///
/// `viewport` must match the width and height of the swapchain (or other color target)
/// that will be rendered to in the same frame, so projection and cached draws agree
/// with the GPU viewport.
///
/// Runs [`ensure_mesh_buffers`] and [`run_collect_mesh_draws`] so this CPU work
/// is measured in the collect phase rather than the render phase.
pub fn prepare_mesh_draws_for_view(
    gpu: &mut crate::gpu::GpuState,
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    viewport: (u32, u32),
) -> PreCollectedFrameData {
    ensure_mesh_buffers(gpu, session, draw_batches);
    prefetch_native_ui_texture2d_gpu(session, gpu, draw_batches);
    let (width, height) = viewport;
    let aspect = width as f32 / height.max(1) as f32;
    let view_params = ViewParams::perspective_from_session(session, aspect);
    let proj = view_params.to_projection_matrix();
    let overlay_projection_override =
        ViewParams::overlay_projection_for_frame(session, draw_batches, aspect);
    let (cached_mesh_draws, prep_stats) = run_collect_mesh_draws(
        session,
        draw_batches,
        gpu,
        proj,
        overlay_projection_override.clone(),
    );
    PreCollectedFrameData {
        proj,
        overlay_projection_override,
        cached_mesh_draws,
        prep_stats,
    }
}

/// Uploads host `Texture2D` mips referenced by native UI materials before pass encoding.
///
/// Walks this frame’s draw batches (merged material + property-block lookup) and then
/// **every** `set_shader` material in the store resolved as [`NativeUiShaderFamily::UiUnlit`]
/// using material-only lookup (same material store walk as
/// [`log_world_unlit_material_inventory_if_enabled`](crate::assets::log_world_unlit_material_inventory_if_enabled),
/// which logs world `Shader "Unlit"` rows instead when that diagnostic is enabled)
/// so textures are GPU-resident even when a material is not drawn this frame.
fn prefetch_native_ui_texture2d_gpu(
    session: &Session,
    gpu: &mut GpuState,
    batches: &[SpaceDrawBatch],
) {
    let rc = session.render_config();
    if !rc.use_native_ui_wgsl {
        return;
    }
    let reg = session.asset_registry();
    let store = &reg.material_property_store;
    for batch in batches {
        for d in &batch.draws {
            let (material_id, is_ui_unlit) = match d.pipeline_variant {
                PipelineVariant::NativeUiUnlit { material_id }
                | PipelineVariant::NativeUiUnlitStencil { material_id } => (material_id, true),
                PipelineVariant::NativeUiTextUnlit { material_id }
                | PipelineVariant::NativeUiTextUnlitStencil { material_id } => (material_id, false),
                _ => continue,
            };
            let lookup = MaterialDrawContext::for_non_skinned_draw(
                material_id,
                d.mesh_renderer_property_block_slot0_id,
            )
            .property_lookup;
            if is_ui_unlit {
                let (_, main_tex, mask_tex) =
                    ui_unlit_material_uniform(store, lookup, &rc.ui_unlit_property_ids);
                prefetch_packed_texture2d(gpu, reg, main_tex);
                prefetch_packed_texture2d(gpu, reg, mask_tex);
            } else {
                let (_, atlas) =
                    ui_text_unlit_material_uniform(store, lookup, &rc.ui_text_unlit_property_ids);
                prefetch_packed_texture2d(gpu, reg, atlas);
            }
        }
    }
    for (material_id, shader_id) in store.iter_material_shader_bindings() {
        let family = resolve_native_shader_route(Some(shader_id), reg);
        match family {
            NativeShaderRoute::UiUnlit => {
                let lookup = MaterialPropertyLookupIds {
                    material_asset_id: material_id,
                    mesh_property_block_slot0: None,
                };
                let (_, main_tex, mask_tex) =
                    ui_unlit_material_uniform(store, lookup, &rc.ui_unlit_property_ids);
                prefetch_packed_texture2d(gpu, reg, main_tex);
                prefetch_packed_texture2d(gpu, reg, mask_tex);
            }
            NativeShaderRoute::WorldUnlit => {
                // Prefetch world Shader "Unlit" textures so they are GPU-resident before
                // the mesh pass bind records them (avoids first-frame grey fallback).
                let lookup = MaterialPropertyLookupIds {
                    material_asset_id: material_id,
                    mesh_property_block_slot0: None,
                };
                let (_, main_tex, mask_tex) = crate::assets::world_unlit_material_uniform(
                    store,
                    lookup,
                    &rc.world_unlit_property_ids,
                );
                prefetch_packed_texture2d(gpu, reg, main_tex);
                prefetch_packed_texture2d(gpu, reg, mask_tex);
            }
            _ => {}
        }
    }
}

fn prefetch_packed_texture2d(gpu: &mut GpuState, reg: &crate::assets::AssetRegistry, packed: i32) {
    if packed == 0 {
        return;
    }
    let Some(tex_id) = texture2d_asset_id_from_packed(packed) else {
        return;
    };
    let Some(tex) = reg.get_texture(tex_id) else {
        return;
    };
    let _ = gpu.ensure_texture2d_gpu(tex_id, tex);
}

/// Ensures all meshes referenced by draw batches are in the GPU mesh buffer cache.
///
/// When ray tracing is available, index buffers use [`wgpu::BufferUsages::BLAS_INPUT`]. BLAS objects
/// are built only while [`crate::gpu::needs_scene_ray_tracing_accel`] is true; when RTAO and RT
/// shadows are both off, BLAS builds are skipped. When tracing is re-enabled, a pass fills any
/// missing BLAS entries for meshes referenced this frame so [`crate::gpu::update_tlas`] can run.
///
/// BLAS builds are submitted as separate queue submissions (one per new mesh). After all builds,
/// waits for those submissions to complete so the TLAS build in the same frame (`update_tlas`)
/// can safely reference the BLASes.
pub(super) fn ensure_mesh_buffers(
    gpu: &mut crate::gpu::GpuState,
    session: &crate::session::Session,
    draw_batches: &[SpaceDrawBatch],
) {
    let mesh_assets = session.asset_registry();
    let rc = session.render_config();
    let need_accel = crate::gpu::needs_scene_ray_tracing_accel(
        gpu.ray_tracing_available,
        rc.rtao_enabled,
        rc.ray_traced_shadows_enabled,
    );
    let mut built_any_blas = false;
    for batch in draw_batches {
        for d in &batch.draws {
            if d.mesh_asset_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                continue;
            };
            if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                continue;
            }
            if !gpu.mesh_buffer_cache.contains_key(&d.mesh_asset_id) {
                let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                let stride = if stride > 0 {
                    stride
                } else {
                    crate::gpu::compute_vertex_stride_from_mesh(mesh)
                };
                let ray_tracing = gpu.ray_tracing_available;
                if let Some(b) =
                    crate::gpu::create_mesh_buffers(&gpu.device, mesh, stride, ray_tracing)
                {
                    gpu.mesh_buffer_cache.insert(d.mesh_asset_id, b.clone());
                    if need_accel
                        && let Some(ref mut accel) = gpu.accel_cache
                        && let Some(blas) =
                            crate::gpu::build_blas_for_mesh(&gpu.device, &gpu.queue, mesh, &b)
                    {
                        accel.insert(d.mesh_asset_id, blas);
                        built_any_blas = true;
                    }
                }
            }
        }
    }

    if need_accel {
        let Some(ref mut accel) = gpu.accel_cache else {
            // No accel cache: poll only if we already built BLAS above (defensive).
            if built_any_blas {
                let _ = gpu.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
            }
            return;
        };
        let mut seen = HashSet::new();
        for batch in draw_batches {
            for d in &batch.draws {
                if d.mesh_asset_id < 0 || !seen.insert(d.mesh_asset_id) {
                    continue;
                }
                if accel.get(d.mesh_asset_id).is_some() {
                    continue;
                }
                let Some(buffers) = gpu.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                    continue;
                };
                let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                    continue;
                };
                if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                    continue;
                }
                if let Some(blas) =
                    crate::gpu::build_blas_for_mesh(&gpu.device, &gpu.queue, mesh, buffers)
                {
                    accel.insert(d.mesh_asset_id, blas);
                    built_any_blas = true;
                }
            }
        }
    }

    // Each build_blas_for_mesh call submits a separate queue submission. The TLAS build
    // (update_tlas) in the same frame records into the main encoder and references these
    // BLASes. Without waiting, the GPU may still be executing the BLAS build submissions
    // when the TLAS build tries to read them, causing a GPU fault / TDR crash on large scenes.
    if built_any_blas {
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }
}
