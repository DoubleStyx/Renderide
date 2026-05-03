//! Pre-record hoists that warm caches and allocate per-view state before the per-view record
//! loop so the loop can later fan out across rayon workers without mutating shared backend state.

use hashbrown::HashMap;
use hashbrown::HashSet;
use hashbrown::hash_map::Entry;

use super::super::super::context::GraphResolvedResources;
use super::super::super::error::GraphExecuteError;
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameView, MultiViewExecutionContext};
use super::{GraphResolveKey, TransientTextureResolveSurfaceParams};
use crate::backend::{HistoryResourceScope, TextureHistorySpec};
use crate::gpu::OutputDepthMode;
use crate::materials::{MaterialPipelineDesc, MaterialPipelineVariantSpec, ShaderPermutation};
use crate::occlusion::gpu::HIZ_MAX_MIPS;
use crate::occlusion::{hi_z_pyramid_dimensions, mip_levels_for_extent};
use crate::passes::PipelineVariantKey;
use crate::render_graph::HistorySlotId;

impl CompiledRenderGraph {
    /// Prepares shared frame resources, per-view resource slots, mesh streams, and material
    /// pipelines for every view before command recording begins.
    pub(super) fn prepare_view_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::prepare_view_resources");
        // Derive each view's `PreRecordViewResourceLayout` once and reuse it across the four
        // per-view sub-phases below. Previously every phase ran its own `viewport / stereo /
        // depth_format / helper_needs / layout` recomputation; some of them rebuilt the same
        // `Vec<PreRecordViewResourceLayout>` from scratch every frame.
        //
        // `Option<...>` is per-view so a swapchain whose depth target failed to resolve this
        // tick maps to `None` and every phase short-circuits for that index in lock-step.
        let view_layouts: Vec<Option<crate::backend::PreRecordViewResourceLayout>> =
            build_view_layouts(mv_ctx, views);
        Self::pre_sync_skybox_specular_environment(mv_ctx);
        Self::pre_sync_shared_frame_resources_for_views(mv_ctx, &view_layouts);
        Self::pre_warm_per_view_resources_for_views(mv_ctx, views, &view_layouts)?;
        Self::register_history_resources_for_views(mv_ctx, views)?;
        Self::pre_warm_pipeline_cache_for_views(mv_ctx, views);
        Ok(())
    }

    /// Resolves the active main skybox prefiltered cubemap before per-view `@group(0)` bind
    /// groups are cached. While the IBL bake is in flight (or no source resolves) the frame
    /// falls back to the existing black cubemap binding.
    pub(super) fn pre_sync_skybox_specular_environment(
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
    ) {
        profiling::scope!("graph::pre_sync_skybox_specular");
        let source = mv_ctx
            .backend
            .active_ibl_cubemap_source(mv_ctx.scene, mv_ctx.gpu_limits);
        let _ = mv_ctx
            .backend
            .frame_resources
            .sync_skybox_specular_environment(mv_ctx.device, source);
    }

    /// Warms the [`crate::materials::MaterialRegistry`] pipeline cache for every prefetched draw
    /// before the per-view record loop begins.
    ///
    /// Pre-warming uses [`crate::materials::MaterialPipelineDesc`] derived from each view's
    /// surface format, depth format, sample count, and multiview mask so cache keys match the
    /// keys the record path will request.
    ///
    pub(super) fn pre_warm_pipeline_cache_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &[FrameView<'_>],
    ) {
        profiling::scope!("graph::pre_warm_pipelines");
        let Some(reg) = mv_ctx.backend.materials.material_registry() else {
            return;
        };

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
                    reg,
                    &mut compile_requests,
                );
            }
        }

        if compile_requests.is_empty() {
            return;
        }
        logger::trace!(
            "graph pipeline prewarm: compiling {} warmed-miss variant(s)",
            compile_requests.len()
        );

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
                MaterialPipelineVariantSpec {
                    permutation: req.shader_perm,
                    blend_mode: req.blend_mode,
                    render_state: req.render_state,
                    front_face: req.front_face,
                    primitive_topology: req.primitive_topology,
                },
            );
        });
        // Stamp the warmed set so the next frame's walk can skip these variants entirely. We
        // mark every requested key, even if its compile failed: a later retry would re-add it.
        reg.mark_pipeline_variants_warmed(compile_requests.iter().copied());
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
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &[FrameView<'_>],
        view_layouts: &[Option<crate::backend::PreRecordViewResourceLayout>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::pre_warm_per_view");
        let mut mesh_ids_needing_uv1_stream: HashSet<i32> = HashSet::new();
        let mut mesh_ids_needing_extended_streams: HashSet<i32> = HashSet::new();
        for (view, layout_opt) in views.iter().zip(view_layouts.iter()) {
            let view_id = view.view_id();
            let Some(layout) = *layout_opt else {
                continue;
            };
            let _ = mv_ctx.backend.frame_resources.per_view_frame_or_create(
                view_id,
                mv_ctx.device,
                layout,
            );
            let _ = mv_ctx.backend.occlusion.ensure_hi_z_state(view_id);
            let _ = mv_ctx
                .backend
                .frame_resources
                .per_view_per_draw_or_create(view_id, mv_ctx.device);
            let _ = mv_ctx
                .backend
                .frame_resources
                .per_view_per_draw_scratch_or_create(view_id);
            if let Some(collection) = view.world_mesh_draw_plan.as_prefetched() {
                for item in &collection.items {
                    if item.mesh_asset_id < 0 {
                        continue;
                    }
                    if item.batch_key.embedded_needs_uv1
                        && !item.batch_key.embedded_needs_extended_vertex_streams
                    {
                        mesh_ids_needing_uv1_stream.insert(item.mesh_asset_id);
                    }
                    if item.batch_key.embedded_needs_extended_vertex_streams {
                        mesh_ids_needing_extended_streams.insert(item.mesh_asset_id);
                    }
                }
            }
        }
        logger::trace!(
            "graph pre-warm per-view resources: views={} uv1_stream_meshes={} extended_stream_meshes={}",
            views.len(),
            mesh_ids_needing_uv1_stream.len(),
            mesh_ids_needing_extended_streams.len(),
        );
        for mesh_asset_id in mesh_ids_needing_uv1_stream {
            let _ = mv_ctx
                .backend
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv1_vertex_stream(mv_ctx.device, mesh_asset_id);
        }
        for mesh_asset_id in mesh_ids_needing_extended_streams {
            let _ = mv_ctx
                .backend
                .asset_transfers
                .mesh_pool_mut()
                .ensure_extended_vertex_streams(mv_ctx.device, mesh_asset_id);
        }
        Ok(())
    }

    /// Registers view-scoped history resources required by ping-pong graph imports.
    ///
    /// Hi-Z still owns CPU snapshots and readback policy through [`crate::occlusion::OcclusionSystem`],
    /// but its graph-declared persistent pyramid now has a registry-backed lifetime keyed by
    /// [`HistorySlotId::HI_Z`] plus the view's [`crate::camera::ViewId`].
    pub(super) fn register_history_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
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
                    HistoryResourceScope::View(view.view_id()),
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
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        view_layouts: &[Option<crate::backend::PreRecordViewResourceLayout>],
    ) {
        profiling::scope!("graph::pre_sync_frame_gpu");
        // Reuse the precomputed layouts from `prepare_view_resources_for_views` instead of
        // walking views again. Skips views whose depth target failed to resolve this tick
        // (matching the prior behaviour of dropping them silently).
        let layouts: Vec<crate::backend::PreRecordViewResourceLayout> =
            view_layouts.iter().filter_map(|layout| *layout).collect();
        mv_ctx.backend.frame_resources.pre_record_sync_for_views(
            mv_ctx.device,
            mv_ctx.queue_arc.as_ref(),
            &layouts,
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
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &mut [FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::pre_resolve_transients");
        for view in views {
            let resolved = Self::resolve_view_from_target(
                view.view_id(),
                &view.target,
                mv_ctx.gpu,
                mv_ctx.backbuffer_view_holder.as_ref(),
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

/// Computes one [`crate::backend::PreRecordViewResourceLayout`] per view. Returns `None` for any
/// view whose `depth_format` cannot be resolved this tick (matches the prior per-phase `continue`
/// behaviour); both pre-warm sub-phases short-circuit on `None` so no view is half-prepared.
fn build_view_layouts(
    mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
    views: &[FrameView<'_>],
) -> Vec<Option<crate::backend::PreRecordViewResourceLayout>> {
    let color_format = mv_ctx.backend.scene_color_format_wgpu();
    views
        .iter()
        .map(|view| {
            let viewport = view.target.extent_px(mv_ctx.gpu);
            let stereo = view.is_multiview_stereo_active();
            let depth_format = view.target.depth_format(mv_ctx.gpu).ok()?;
            let helper_needs = view.world_mesh_draw_plan.helper_needs();
            Some(crate::backend::PreRecordViewResourceLayout {
                width: viewport.0,
                height: viewport.1,
                stereo,
                depth_format,
                color_format,
                needs_depth_snapshot: helper_needs.depth_snapshot,
                needs_color_snapshot: helper_needs.color_snapshot,
            })
        })
        .collect()
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
    mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
    view: &FrameView<'_>,
) -> Option<(MaterialPipelineDesc, ShaderPermutation)> {
    use std::num::NonZeroU32;
    let host_camera = view.host_camera;
    let multiview_stereo = view.is_multiview_stereo_active();
    let surface_format = view.target.color_format(mv_ctx.gpu);
    let depth_stencil_format = view.target.depth_format(mv_ctx.gpu).ok()?;
    let sample_count = view.target.sample_count(mv_ctx.gpu);
    let use_multiview = multiview_stereo
        && host_camera.active_stereo().is_some()
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
        crate::materials::SHADER_PERM_MULTIVIEW_STEREO
    } else {
        ShaderPermutation(0)
    };
    Some((pass_desc, shader_perm))
}

/// Appends unique `(shader_asset_id, blend_mode, render_state, front_face)` permutations from `items` to
/// `out`, stamped with the view's `pass_desc` and `shader_perm`. Duplicates within this view are
/// elided; cross-frame dedup against `registry`'s warmed-variants set skips keys the pre-warm
/// path has already requested in a prior frame, eliminating the rayon dispatch on steady state.
fn collect_unique_pipeline_requests(
    items: &[crate::world_mesh::draw_prep::WorldMeshDrawItem],
    pass_desc: MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    registry: &crate::materials::MaterialRegistry,
    out: &mut Vec<PipelineVariantKey>,
) {
    let mut seen: HashSet<(
        i32,
        crate::materials::MaterialBlendMode,
        crate::materials::MaterialRenderState,
        crate::materials::RasterFrontFace,
        crate::materials::RasterPrimitiveTopology,
        bool,
    )> = HashSet::new();
    for item in items {
        let grab_pass = item.batch_key.embedded_uses_scene_color_snapshot;
        let key = (
            item.batch_key.shader_asset_id,
            item.batch_key.blend_mode,
            item.batch_key.render_state,
            item.batch_key.front_face,
            item.batch_key.primitive_topology,
            grab_pass,
        );
        if !seen.insert(key) {
            continue;
        }
        let variant_key = PipelineVariantKey::for_draw_item(item, pass_desc, shader_perm);
        if registry.is_pipeline_variant_warmed(&variant_key) {
            continue;
        }
        out.push(variant_key);
    }
}
