//! Pre-record hoists that warm caches and allocate per-view state before the per-view record
//! loop so the loop can later fan out across rayon workers without mutating shared backend state.

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use super::super::super::context::GraphResolvedResources;
use super::super::super::error::GraphExecuteError;
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameView, MultiViewExecutionContext};
use super::{GraphResolveKey, TransientTextureResolveSurfaceParams};
use crate::backend::{HistoryResourceScope, TextureHistorySpec};
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::ShaderPermutation;
use crate::render_graph::occlusion::HIZ_MAX_MIPS;
use crate::render_graph::world_mesh_draw_prep::PipelineVariantKey;
use crate::render_graph::{
    hi_z_pyramid_dimensions, mip_levels_for_extent, HistorySlotId, OutputDepthMode,
};

impl CompiledRenderGraph {
    /// Prepares shared frame resources, per-view resource slots, mesh streams, and material
    /// pipelines for every view before command recording begins.
    pub(super) fn prepare_view_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::prepare_view_resources");
        Self::pre_sync_shared_frame_resources_for_views(mv_ctx, views);
        Self::pre_warm_per_view_resources_for_views(mv_ctx, views)?;
        Self::register_history_resources_for_views(mv_ctx, views)?;
        Self::pre_warm_pipeline_cache_for_views(mv_ctx, views);
        Ok(())
    }

    /// Warms the [`crate::materials::MaterialRegistry`] pipeline cache for every prefetched draw
    /// before the per-view record loop begins.
    ///
    /// Pre-warming uses [`crate::materials::MaterialPipelineDesc`] derived from each view's
    /// surface format, depth format, sample count, and multiview mask so cache keys match the
    /// keys the record path will request.
    ///
    pub(super) fn pre_warm_pipeline_cache_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) {
        profiling::scope!("graph::pre_warm_pipelines");
        if mv_ctx.backend.materials.material_registry().is_none() {
            return;
        }

        let mut compile_requests: Vec<PipelineVariantKey> = Vec::new();
        for view in views {
            let Some((pass_desc, shader_perm)) = view_pipeline_pass_desc(mv_ctx, view) else {
                continue;
            };
            let Some(collection) = view.world_mesh_draw_plan.as_prefetched() else {
                continue;
            };
            if !collection.items.is_empty() {
                collect_unique_pipeline_requests(
                    &collection.items,
                    pass_desc,
                    shader_perm,
                    &mut compile_requests,
                );
            }
        }

        if compile_requests.is_empty() {
            return;
        }

        let Some(reg) = mv_ctx.backend.materials.material_registry() else {
            return;
        };
        // Fan pipeline misses out to the rayon pool so multiple new permutations compile in
        // parallel instead of serially blocking the main thread. `MaterialPipelineCache`
        // releases its mutex before `create_shader_module` / `create_render_pipeline` and
        // elides duplicate inserts on re-lock, so concurrent callers are safe.
        use rayon::prelude::*;
        compile_requests.par_iter().for_each(|req| {
            profiling::scope!("graph::pre_warm_pipelines::compile");
            let pass_desc = req.pass_desc();
            let _ = reg.pipeline_for_shader_asset(
                req.shader_asset_id,
                &pass_desc,
                req.shader_perm,
                req.blend_mode,
                req.render_state,
                req.front_face,
            );
        });
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
        for view in views {
            let occlusion_view = view.occlusion_view_id();
            let viewport = view.target.extent_px(mv_ctx.gpu);
            let stereo = view.is_multiview_stereo_active();
            let Ok(depth_format) = view.target.depth_format(mv_ctx.gpu) else {
                continue;
            };
            let helper_needs = view.world_mesh_draw_plan.helper_needs();
            let layout = crate::backend::PreRecordViewResourceLayout {
                width: viewport.0,
                height: viewport.1,
                stereo,
                depth_format,
                color_format: mv_ctx.backend.scene_color_format_wgpu(),
                needs_depth_snapshot: helper_needs.depth_snapshot,
                needs_color_snapshot: helper_needs.color_snapshot,
            };
            let _ = mv_ctx.backend.frame_resources.per_view_frame_or_create(
                occlusion_view,
                mv_ctx.device,
                layout,
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
            if let Some(collection) = view.world_mesh_draw_plan.as_prefetched() {
                for item in &collection.items {
                    if item.batch_key.embedded_needs_extended_vertex_streams
                        && item.mesh_asset_id >= 0
                    {
                        mesh_ids_needing_extended_streams.insert(item.mesh_asset_id);
                    }
                }
            }
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

    /// Registers view-scoped history resources required by ping-pong graph imports.
    ///
    /// Hi-Z still owns CPU snapshots and readback policy through [`crate::backend::OcclusionSystem`],
    /// but its graph-declared persistent pyramid now has a registry-backed lifetime keyed by
    /// [`HistorySlotId::HI_Z`] plus the view's [`crate::render_graph::OcclusionViewId`].
    pub(super) fn register_history_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::register_history_resources");
        for view in views {
            let viewport = view.target.extent_px(mv_ctx.gpu);
            let mode = OutputDepthMode::from_multiview_stereo(view.is_multiview_stereo_active());
            let Some(spec) = hi_z_history_spec(viewport, mode) else {
                continue;
            };
            mv_ctx
                .backend
                .history_registry_mut()
                .register_texture_scoped(
                    HistorySlotId::HI_Z,
                    HistoryResourceScope::View(view.occlusion_view_id()),
                    spec,
                )?;
        }
        mv_ctx
            .backend
            .history_registry()
            .ensure_resources(mv_ctx.device);
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
        let mut view_layouts = Vec::with_capacity(views.len());
        let color_format = mv_ctx.backend.scene_color_format_wgpu();
        for view in views {
            let viewport = view.target.extent_px(mv_ctx.gpu);
            let stereo = view.is_multiview_stereo_active();
            let Ok(depth_format) = view.target.depth_format(mv_ctx.gpu) else {
                continue;
            };
            let helper_needs = view.world_mesh_draw_plan.helper_needs();
            view_layouts.push(crate::backend::PreRecordViewResourceLayout {
                width: viewport.0,
                height: viewport.1,
                stereo,
                depth_format,
                color_format,
                needs_depth_snapshot: helper_needs.depth_snapshot,
                needs_color_snapshot: helper_needs.color_snapshot,
            });
        }
        mv_ctx.backend.frame_resources.pre_record_sync_for_views(
            mv_ctx.device,
            mv_ctx.queue_arc.as_ref(),
            &view_layouts,
        );
    }

    /// Pre-resolves transient textures and buffers for every view's [`GraphResolveKey`].
    ///
    /// Hoists transient-pool allocation out of the per-view record loop so recording can read the
    /// resulting `transient_by_key` map without mutating the shared pool.
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
                    self.subresources.len(),
                );
                let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                    resolved.viewport_px,
                    mv_ctx.gpu_limits.max_texture_dimension_2d(),
                );
                let scene_color_format = mv_ctx.backend.scene_color_format_wgpu();
                self.resolve_transient_textures(
                    mv_ctx.device,
                    mv_ctx.gpu_limits,
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
                    mv_ctx.gpu_limits,
                    mv_ctx.backend.transient_pool_mut(),
                    alloc_viewport,
                    &mut resources,
                )?;
                self.resolve_subresource_views(&mut resources);
                v.insert(resources);
            }
        }
        Ok(())
    }
}

/// Builds the registry spec for the current view's Hi-Z pyramid texture.
fn hi_z_history_spec(
    full_extent_px: (u32, u32),
    mode: OutputDepthMode,
) -> Option<TextureHistorySpec> {
    let (bw, bh) = hi_z_pyramid_dimensions(full_extent_px.0, full_extent_px.1);
    if bw == 0 || bh == 0 {
        return None;
    }
    Some(TextureHistorySpec {
        label: "hi_z_history",
        format: wgpu::TextureFormat::R32Float,
        extent: wgpu::Extent3d {
            width: bw,
            height: bh,
            depth_or_array_layers: match mode {
                OutputDepthMode::DesktopSingle => 1,
                OutputDepthMode::StereoArray { .. } => 2,
            },
        },
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        mip_level_count: mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS).max(1),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
    })
}

/// Resolves the view's surface / depth / sample-count / multiview attributes into the
/// [`MaterialPipelineDesc`] + [`ShaderPermutation`] pair used as the pipeline cache key, or
/// returns `None` when the swapchain depth target is unavailable this tick.
fn view_pipeline_pass_desc(
    mv_ctx: &mut MultiViewExecutionContext<'_>,
    view: &FrameView<'_>,
) -> Option<(MaterialPipelineDesc, ShaderPermutation)> {
    use std::num::NonZeroU32;
    let host_camera = view.host_camera;
    let multiview_stereo = view.is_multiview_stereo_active();
    let surface_format = view.target.color_format(mv_ctx.gpu);
    let depth_stencil_format = view.target.depth_format(mv_ctx.gpu).ok()?;
    let sample_count = view.target.sample_count(mv_ctx.gpu);
    let use_multiview =
        multiview_stereo && host_camera.stereo.is_some() && mv_ctx.gpu_limits.supports_multiview;
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
        ShaderPermutation(0)
    };
    Some((pass_desc, shader_perm))
}

/// Appends unique `(shader_asset_id, blend_mode, render_state, front_face)` permutations from `items` to
/// `out`, stamped with the view's `pass_desc` and `shader_perm`. Duplicates within this view
/// are elided; the LRU cache handles cross-view dedup.
fn collect_unique_pipeline_requests(
    items: &[crate::render_graph::world_mesh_draw_prep::WorldMeshDrawItem],
    pass_desc: MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    out: &mut Vec<PipelineVariantKey>,
) {
    let mut seen: std::collections::HashSet<(
        i32,
        crate::materials::MaterialBlendMode,
        crate::materials::MaterialRenderState,
        crate::materials::RasterFrontFace,
        bool,
    )> = std::collections::HashSet::new();
    for item in items {
        let grab_pass = item.batch_key.embedded_requires_grab_pass;
        let key = (
            item.batch_key.shader_asset_id,
            item.batch_key.blend_mode,
            item.batch_key.render_state,
            item.batch_key.front_face,
            grab_pass,
        );
        if !seen.insert(key) {
            continue;
        }
        out.push(PipelineVariantKey::for_draw_item(
            item,
            pass_desc,
            shader_perm,
        ));
    }
}
