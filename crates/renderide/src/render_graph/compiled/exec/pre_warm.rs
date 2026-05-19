//! Pre-record hoists that warm caches and allocate per-view state before the per-view record
//! loop so the loop can later fan out across rayon workers without mutating shared backend state.

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;

use super::super::super::context::GraphResolvedResources;
use super::super::super::error::GraphExecuteError;
use super::super::super::frame_params::PreRecordViewResourceLayout;
use super::super::super::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use super::super::super::history::{HistoryResourceScope, TextureHistorySpec};
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameView, MultiViewExecutionContext};
use super::{GraphResolveKey, TransientTextureResolveSurfaceParams};
use crate::gpu::OutputDepthMode;
use crate::occlusion::gpu::HIZ_MAX_MIPS;
use crate::occlusion::{hi_z_pyramid_dimensions, mip_levels_for_extent};
use crate::render_graph::HistorySlotId;

impl CompiledRenderGraph {
    /// Prepares shared frame resources, per-view resource slots, mesh streams, and material
    /// pipelines for every view before command recording begins.
    pub(super) fn prepare_view_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        upload_batch: &FrameUploadBatch,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::prepare_view_resources");
        // Derive each view's `PreRecordViewResourceLayout` once and reuse it across the four
        // per-view sub-phases below. A per-view `None` keeps depth-target resolution failures
        // aligned across every phase for that index.
        let view_layouts: Vec<Option<PreRecordViewResourceLayout>> =
            build_view_layouts(mv_ctx, views);
        Self::pre_warm_per_view_resources_for_views(mv_ctx, views, &view_layouts)?;
        Self::pre_sync_shared_frame_resources_for_views(mv_ctx, &view_layouts, upload_batch);
        Self::register_history_resources_for_views(mv_ctx, views)?;
        Ok(())
    }

    /// Eagerly allocates per-view frame and per-draw resources
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
        view_layouts: &[Option<PreRecordViewResourceLayout>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::pre_warm_per_view");
        let mut prepared_frame_layouts = Vec::with_capacity(views.len());
        for (view, layout_opt) in views.iter().zip(view_layouts.iter()) {
            let view_id = view.view_id();
            let Some(layout) = *layout_opt else {
                continue;
            };
            if !mv_ctx
                .backend
                .frame_resources_mut()
                .ensure_per_view_frame_resources(view_id, mv_ctx.device, layout)
            {
                logger::warn!(
                    "graph pre-warm: per-view frame resources unavailable for view {view_id:?} layout={layout:?}"
                );
                return Err(GraphExecuteError::MissingPerViewResources {
                    view_id,
                    resource: "frame",
                });
            }
            prepared_frame_layouts.push((view_id, layout));
            let _ = mv_ctx.backend.occlusion().ensure_hi_z_state(view_id);
            if !mv_ctx
                .backend
                .frame_resources_mut()
                .ensure_per_view_per_draw_resources(view_id, mv_ctx.device)
            {
                logger::warn!(
                    "graph pre-warm: per-draw resources unavailable for view {view_id:?}"
                );
                return Err(GraphExecuteError::MissingPerViewResources {
                    view_id,
                    resource: "per-draw",
                });
            }
            mv_ctx
                .backend
                .frame_resources_mut()
                .ensure_per_view_per_draw_scratch(view_id);
        }
        for (view_id, layout) in prepared_frame_layouts {
            if !mv_ctx
                .backend
                .frame_resources_mut()
                .ensure_per_view_frame_resources(view_id, mv_ctx.device, layout)
            {
                logger::warn!(
                    "graph pre-warm: per-view frame resources became stale for view {view_id:?} layout={layout:?}"
                );
                return Err(GraphExecuteError::MissingPerViewResources {
                    view_id,
                    resource: "frame",
                });
            }
        }
        logger::trace!("graph pre-warm per-view resources: views={}", views.len(),);
        mv_ctx
            .backend
            .pre_warm_view_assets_from_blackboards(mv_ctx.device, views, view_layouts);
        Ok(())
    }

    /// Registers view-scoped history resources required by ping-pong graph imports.
    ///
    /// Hi-Z still owns CPU snapshots and readback policy through [`crate::occlusion::OcclusionSystem`],
    /// but its graph-declared persistent pyramid now has a registry-backed lifetime keyed by
    /// [`HistorySlotId::HI_Z`] plus the view's [`crate::camera::ViewId`].
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
    /// This hoists shared cluster synchronization and per-view light uploads out of the per-view
    /// record path so rayon workers only read view-local state during recording.
    pub(super) fn pre_sync_shared_frame_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        upload_batch: &FrameUploadBatch,
    ) {
        profiling::scope!("graph::pre_sync_frame_gpu");
        // Reuse the precomputed layouts from `prepare_view_resources_for_views` so every shared
        // frame sync phase sees the same valid-view set.
        let layouts: Vec<PreRecordViewResourceLayout> =
            view_layouts.iter().filter_map(|layout| *layout).collect();
        mv_ctx
            .backend
            .frame_resources_mut()
            .pre_record_sync_for_views(
                mv_ctx.device,
                GraphUploadSink::pre_record(upload_batch),
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
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::pre_resolve_transients");
        for view in views {
            let resolved = Self::resolve_view_from_target(
                view.view_id(),
                view.post_processing,
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

/// Computes one [`PreRecordViewResourceLayout`] per view. Returns `None` for any
/// view whose `depth_format` cannot be resolved this tick (matches the prior per-phase `continue`
/// behaviour); both pre-warm sub-phases short-circuit on `None` so no view is half-prepared.
fn build_view_layouts(
    mv_ctx: &mut MultiViewExecutionContext<'_>,
    views: &[FrameView<'_>],
) -> Vec<Option<PreRecordViewResourceLayout>> {
    let color_format = mv_ctx.backend.scene_color_format_wgpu();
    views
        .iter()
        .map(|view| {
            let viewport = view.target.extent_px(mv_ctx.gpu);
            let stereo = view.is_multiview_stereo_active();
            let depth_format = view.target.depth_format(mv_ctx.gpu).ok()?;
            let sample_count = match &view.target {
                super::super::FrameViewTarget::Swapchain => {
                    mv_ctx.gpu.msaa().swapchain_msaa_effective().max(1)
                }
                super::super::FrameViewTarget::ExternalMultiview(_) => {
                    mv_ctx.gpu.msaa().swapchain_msaa_effective_stereo().max(1)
                }
                super::super::FrameViewTarget::OffscreenRt(ext) => ext
                    .sample_count_policy
                    .resolve(mv_ctx.gpu.msaa().swapchain_msaa_effective()),
            };
            Some(PreRecordViewResourceLayout {
                view_id: view.view_id(),
                width: viewport.0,
                height: viewport.1,
                stereo,
                sample_count,
                depth_format,
                color_format,
                needs_depth_snapshot: view.resource_hints.needs_depth_snapshot,
                needs_color_snapshot: view.resource_hints.needs_color_snapshot,
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
