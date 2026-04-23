//! Pre-record hoists that warm caches and allocate per-view state before the per-view record
//! loop so the loop can later fan out across rayon workers without mutating shared backend state.

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use super::super::super::context::GraphResolvedResources;
use super::super::super::error::GraphExecuteError;
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameView, FrameViewTarget, MultiViewExecutionContext};
use super::{GraphResolveKey, TransientTextureResolveSurfaceParams};
use crate::assets::material::MaterialDictionary;
use crate::materials::{
    embedded_stem_needs_extended_vertex_streams, resolve_raster_pipeline, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::world_mesh_draw_prep::FramePreparedRenderables;

impl CompiledRenderGraph {
    /// Warms the [`crate::materials::MaterialRegistry`] pipeline cache for every prefetched draw
    /// before the per-view record loop begins.
    ///
    /// Pre-warming uses [`crate::materials::MaterialPipelineDesc`] derived from each view's
    /// surface format, depth format, sample count, and multiview mask so cache keys match the
    /// keys the record path will request. Views without prefetched draws (lazy-collect path)
    /// are skipped — they will fall back to lazy cache build during recording as before.
    pub(super) fn pre_warm_pipeline_cache_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) {
        use crate::materials::MaterialPipelineDesc;
        use std::num::NonZeroU32;
        profiling::scope!("graph::pre_warm_pipelines");
        let Some(reg) = mv_ctx.backend.materials.material_registry() else {
            return;
        };
        for view in views {
            let Some(collection) = view.prefetched_world_mesh_draws.as_ref() else {
                continue;
            };
            if collection.items.is_empty() {
                continue;
            }
            let host_camera = view.host_camera;
            let (viewport, multiview_stereo) = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => {
                    let stereo = host_camera.vr_active && host_camera.stereo_views.is_some();
                    (ext.extent_px, stereo)
                }
                FrameViewTarget::OffscreenRt(ext) => (ext.extent_px, false),
                FrameViewTarget::Swapchain => (mv_ctx.gpu.surface_extent_px(), false),
            };
            let _ = viewport;
            let surface_format = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => ext.surface_format,
                FrameViewTarget::OffscreenRt(ext) => ext.color_format,
                FrameViewTarget::Swapchain => mv_ctx.gpu.config_format(),
            };
            let depth_stencil_format = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => ext.depth_texture.format(),
                FrameViewTarget::OffscreenRt(ext) => ext.depth_texture.format(),
                FrameViewTarget::Swapchain => {
                    let Ok((depth_tex, _)) = mv_ctx.gpu.ensure_depth_target() else {
                        continue;
                    };
                    depth_tex.format()
                }
            };
            let sample_count = match &view.target {
                FrameViewTarget::ExternalMultiview(_) => {
                    mv_ctx.gpu.swapchain_msaa_effective_stereo().max(1)
                }
                FrameViewTarget::OffscreenRt(_) => 1,
                FrameViewTarget::Swapchain => mv_ctx.gpu.swapchain_msaa_effective().max(1),
            };
            let use_multiview = multiview_stereo
                && host_camera.vr_active
                && host_camera.stereo_view_proj.is_some()
                && mv_ctx.gpu_limits.supports_multiview;
            let pass_desc = MaterialPipelineDesc {
                surface_format,
                depth_stencil_format: Some(depth_stencil_format),
                sample_count,
                multiview_mask: if use_multiview {
                    NonZeroU32::new(3)
                } else {
                    None
                },
            };
            let shader_perm = if use_multiview {
                crate::pipelines::SHADER_PERM_MULTIVIEW_STEREO
            } else {
                crate::pipelines::ShaderPermutation(0)
            };

            // Walk unique (shader_asset_id, blend_mode, render_state) tuples to avoid duplicate
            // cache calls for draws that share the same batch key.
            let mut seen: std::collections::HashSet<(
                i32,
                crate::materials::MaterialBlendMode,
                crate::materials::MaterialRenderState,
            )> = std::collections::HashSet::new();
            for item in &collection.items {
                let key = (
                    item.batch_key.shader_asset_id,
                    item.batch_key.blend_mode,
                    item.batch_key.render_state,
                );
                if !seen.insert(key) {
                    continue;
                }
                let _ = reg.pipeline_for_shader_asset(
                    item.batch_key.shader_asset_id,
                    &pass_desc,
                    shader_perm,
                    item.batch_key.blend_mode,
                    item.batch_key.render_state,
                );
            }
        }
    }

    /// Eagerly allocates per-view frame state ([`crate::backend::FrameResourceManager::per_view_frame_or_create`])
    /// and per-view per-draw resources ([`crate::backend::FrameResourceManager::per_view_per_draw_or_create`])
    /// for every view in `views` before per-view recording begins.
    ///
    /// Hoists the lazy `&mut backend.frame_resources.*_or_create` calls out of the per-view
    /// recording loop so that loop can later borrow `backend` shared across rayon workers
    /// without colliding on the per-view resource maps (`per_view_frame`, `per_view_draw`).
    /// Also primes a freshly added secondary RT camera so its first frame does not pay the
    /// cluster-buffer / frame-uniform-buffer allocation cost mid-recording.
    pub(super) fn pre_warm_per_view_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::pre_warm_per_view");
        let mut mesh_ids_needing_extended_streams = std::collections::HashSet::new();
        let mut any_view_missing_prefetch = false;
        for view in views {
            let occlusion_view = view.occlusion_view_id();
            let host_camera = view.host_camera;
            let (viewport, stereo) = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => {
                    let stereo = host_camera.vr_active && host_camera.stereo_views.is_some();
                    (ext.extent_px, stereo)
                }
                FrameViewTarget::OffscreenRt(ext) => (ext.extent_px, false),
                FrameViewTarget::Swapchain => (mv_ctx.gpu.surface_extent_px(), false),
            };
            let _ = mv_ctx.backend.frame_resources.per_view_frame_or_create(
                occlusion_view,
                mv_ctx.device,
                viewport,
                stereo,
            );
            let _ = mv_ctx.backend.occlusion.ensure_hi_z_state(occlusion_view);
            let _ = mv_ctx
                .backend
                .frame_resources
                .per_view_per_draw_or_create(occlusion_view, mv_ctx.device);
            let _ = mv_ctx
                .backend
                .frame_resources
                .per_view_per_draw_scratch_or_create(occlusion_view);
            if let Some(collection) = view.prefetched_world_mesh_draws.as_ref() {
                for item in &collection.items {
                    if item.batch_key.embedded_needs_extended_vertex_streams
                        && item.mesh_asset_id >= 0
                    {
                        mesh_ids_needing_extended_streams.insert(item.mesh_asset_id);
                    }
                }
            } else {
                any_view_missing_prefetch = true;
            }
        }
        if any_view_missing_prefetch {
            // Entry points that hand the graph a [`FrameView`] without prefetched draws — notably
            // the OpenXR stereo view assembled in
            // [`crate::runtime::RendererRuntime::render_frame`] — otherwise leave the mesh set
            // above empty, so `ensure_extended_vertex_streams` never runs for materials whose
            // vertex shader reads `@location(4)` or higher. The per-view record path uses an
            // immutable `&MeshPool` and cannot upload those streams itself; a miss there silently
            // drops the draw. Mirror the scene walk that
            // [`crate::runtime::secondary_cameras`] performs for desktop multi-view, scoped to
            // the main render context, and pre-warm the same streams here.
            collect_fallback_extended_stream_mesh_ids(
                mv_ctx,
                &mut mesh_ids_needing_extended_streams,
            );
        }
        for mesh_asset_id in mesh_ids_needing_extended_streams {
            let _ = mv_ctx
                .backend
                .asset_transfers
                .mesh_pool
                .ensure_extended_vertex_streams(mv_ctx.device, mesh_asset_id);
        }
        Ok(())
    }

    /// Pre-synchronizes shared frame resources for every unique per-view layout before recording.
    ///
    /// This hoists the shared `FrameGpuResources::sync_cluster_viewport` and one-time lights upload
    /// out of the per-view record path so rayon workers only touch per-view state during recording.
    pub(super) fn pre_sync_shared_frame_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) {
        profiling::scope!("graph::pre_sync_frame_gpu");
        let mut viewports_and_stereo = Vec::with_capacity(views.len());
        for view in views {
            let host_camera = view.host_camera;
            let (viewport, stereo) = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => {
                    let stereo = host_camera.vr_active && host_camera.stereo_views.is_some();
                    (ext.extent_px, stereo)
                }
                FrameViewTarget::OffscreenRt(ext) => (ext.extent_px, false),
                FrameViewTarget::Swapchain => (mv_ctx.gpu.surface_extent_px(), false),
            };
            viewports_and_stereo.push((viewport.0, viewport.1, stereo));
        }
        mv_ctx.backend.frame_resources.pre_record_sync_for_views(
            mv_ctx.device,
            mv_ctx.queue_arc.as_ref(),
            &viewports_and_stereo,
        );
    }

    /// Pre-resolves transient textures and buffers for every view's [`GraphResolveKey`].
    ///
    /// Hoists the transient-pool allocation out of the per-view record loop so that the loop
    /// itself no longer calls `backend.transient_pool_mut()`. This is a prerequisite for parallel
    /// per-view recording (Milestone E): concurrent workers cannot share `&mut` access to the
    /// pool, but they can share `&` access to the resulting `transient_by_key` map.
    ///
    /// Imported textures/buffers still resolve per-view inside the record loop because their
    /// bindings (backbuffer, per-view cluster refs) differ across views that share a key.
    pub(super) fn pre_resolve_transients_for_views(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::pre_resolve_transients");
        for view in views {
            let resolved = Self::resolve_view_from_target(
                &view.target,
                mv_ctx.gpu,
                mv_ctx.backbuffer_view_holder,
            )?;
            let key = GraphResolveKey::from_resolved(&resolved);
            if let Entry::Vacant(v) = transient_by_key.entry(key) {
                let mut resources = GraphResolvedResources::with_capacity(
                    self.transient_textures.len(),
                    self.transient_buffers.len(),
                    self.imported_textures.len(),
                    self.imported_buffers.len(),
                );
                let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                    resolved.viewport_px,
                    mv_ctx.gpu_limits.max_texture_dimension_2d(),
                );
                let scene_color_format = mv_ctx.backend.scene_color_format_wgpu();
                self.resolve_transient_textures(
                    mv_ctx.device,
                    mv_ctx.backend.transient_pool_mut(),
                    TransientTextureResolveSurfaceParams {
                        viewport_px: alloc_viewport,
                        surface_format: resolved.surface_format,
                        depth_stencil_format: resolved.depth_texture.format(),
                        scene_color_format,
                        sample_count: resolved.sample_count,
                        multiview_stereo: resolved.multiview_stereo,
                    },
                    &mut resources,
                )?;
                self.resolve_transient_buffers(
                    mv_ctx.device,
                    mv_ctx.backend.transient_pool_mut(),
                    alloc_viewport,
                    &mut resources,
                )?;
                v.insert(resources);
            }
        }
        Ok(())
    }
}

/// Fallback scene walk that mirrors [`crate::runtime::secondary_cameras::RendererRuntime::render_frame`]'s
/// frame-scope renderable expansion, scoped to the scene's active main render context.
///
/// Invoked by [`CompiledRenderGraph::pre_warm_per_view_resources_for_views`] when at least one
/// view arrives without prefetched world-mesh draws. Uploads tangent / UV1..3 streams for every
/// mesh whose resolved material stem has a vertex shader that reads `@location(4)` or higher so
/// the per-view record path's read-only [`crate::resources::MeshPool`] finds those streams ready
/// instead of silently skipping the draw. Today this is exercised by the OpenXR stereo view
/// assembled in [`crate::runtime::RendererRuntime::render_frame`]; any other caller that passes
/// `prefetched_world_mesh_draws: None` will also reuse this fallback.
fn collect_fallback_extended_stream_mesh_ids(
    mv_ctx: &MultiViewExecutionContext<'_>,
    out: &mut std::collections::HashSet<i32>,
) {
    profiling::scope!("graph::pre_warm_per_view_fallback_scene_walk");
    let Some(reg) = mv_ctx.backend.materials.material_registry() else {
        return;
    };
    let router = &reg.router;
    let property_store = mv_ctx.backend.materials.material_property_store();
    let dict = MaterialDictionary::new(property_store);
    let render_context = mv_ctx.scene.active_main_render_context();
    let mesh_pool = &mv_ctx.backend.asset_transfers.mesh_pool;
    let prepared =
        FramePreparedRenderables::build_for_frame(mv_ctx.scene, mesh_pool, render_context);
    if prepared.is_empty() {
        return;
    }
    for (mesh_asset_id, material_asset_id) in prepared.mesh_material_pairs() {
        if mesh_asset_id < 0 {
            continue;
        }
        let shader_asset_id = dict
            .shader_asset_for_material(material_asset_id)
            .unwrap_or(-1);
        let needs = match resolve_raster_pipeline(shader_asset_id, router) {
            RasterPipelineKind::EmbeddedStem(stem) => {
                embedded_stem_needs_extended_vertex_streams(stem.as_ref(), ShaderPermutation(0))
            }
            RasterPipelineKind::DebugWorldNormals => false,
        };
        if needs {
            out.insert(mesh_asset_id);
        }
    }
}
