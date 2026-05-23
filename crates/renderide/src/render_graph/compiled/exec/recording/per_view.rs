//! Per-view command-buffer recording.

use hashbrown::HashMap;
use std::time::Instant;

use crate::diagnostics::PerViewHudOutputsSlot;
use crate::graph_inputs::FrameSystemsShared;
use crate::render_graph::blackboard::GraphCommandStatsSlot;
use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::pass::PassPhase;

use super::super::super::helpers;
use super::super::super::{CompiledRenderGraph, ResolvedView};
use super::super::{
    GraphResolveKey, PerViewEncodeOutput, PerViewRecordShared, PerViewWorkItem,
    ResolvedOffscreenColorCopy, elapsed_ms,
};

impl CompiledRenderGraph {
    /// Records the per-view pass phase into one command buffer for `work_item`.
    pub(in crate::render_graph::compiled::exec) fn record_one_view(
        &self,
        shared: &PerViewRecordShared<'_>,
        work_item: PerViewWorkItem,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewEncodeOutput, GraphExecuteError> {
        profiling::scope!("graph::per_view");
        let encode_start = Instant::now();
        let device = shared.device;
        let PerViewWorkItem {
            view_idx,
            host_camera,
            render_context,
            frame_time_seconds,
            clear,
            initial_blackboard,
            resolved,
            ..
        } = work_item;

        let mut encoder = {
            profiling::scope!("graph::per_view::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-per-view"),
            })
        };
        let gpu_query = profiler.map(|p| p.begin_query("graph::per_view", &mut encoder));

        let resolved_view = resolved.as_resolved();
        let resolved_resources =
            self.resolve_per_view_graph_resources(shared, &resolved_view, transient_by_key)?;
        let graph_resources: &GraphResolvedResources = &resolved_resources;

        let mut frame_params = Self::build_per_view_frame_params(
            shared,
            &resolved_view,
            &host_camera,
            render_context,
            frame_time_seconds,
            clear,
        );
        let mut view_blackboard =
            self.build_per_view_blackboard(&frame_params, graph_resources, initial_blackboard);

        {
            profiling::scope!("graph::per_view::pass_loop");
            self.record_phase_steps(
                PassPhase::PerView,
                Some(view_idx),
                &resolved_view,
                graph_resources,
                &mut frame_params,
                &mut view_blackboard,
                &mut encoder,
                shared.device,
                shared.gpu_limits,
                upload_batch,
                profiler,
            )?;
        }
        let offscreen_copy_recorded = Self::record_offscreen_color_copy(
            &mut encoder,
            resolved.offscreen_color_copy.as_ref(),
            profiler,
        );
        if let Some(query) = gpu_query
            && let Some(prof) = profiler
        {
            prof.end_query(&mut encoder, query);
        }
        let mut command_stats = view_blackboard
            .get::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        if resolved.offscreen_color_copy.is_some() {
            command_stats.record_copy_result(offscreen_copy_recorded);
        }
        let hud_outputs = view_blackboard.take::<PerViewHudOutputsSlot>();
        let encode_ms = elapsed_ms(encode_start);
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            let finish_ms = elapsed_ms(finish_start);
            (command_buffer, finish_ms)
        };
        Ok(PerViewEncodeOutput {
            command_buffer,
            hud_outputs,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    /// Resolves this view's transient/imported graph resources from pre-record shared state.
    fn resolve_per_view_graph_resources(
        &self,
        shared: &PerViewRecordShared<'_>,
        resolved: &ResolvedView<'_>,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<GraphResolvedResources, GraphExecuteError> {
        profiling::scope!("graph::per_view::resolve_transients");
        let key = GraphResolveKey::from_resolved(resolved);
        let mut resolved_resources = transient_by_key.get(&key).cloned().ok_or_else(|| {
            logger::warn!("pre-resolve: missing transient resources for view key {key:?}");
            GraphExecuteError::MissingTransientResources
        })?;
        self.resolve_imported_textures(resolved, shared.history, &mut resolved_resources)?;
        self.resolve_imported_buffers(
            shared.frame_resources,
            shared.history,
            resolved,
            &mut resolved_resources,
        )?;
        Ok(resolved_resources)
    }

    /// Records the final scratch-to-render-texture copy for a partial offscreen viewport.
    fn record_offscreen_color_copy(
        encoder: &mut wgpu::CommandEncoder,
        copy: Option<&ResolvedOffscreenColorCopy>,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> bool {
        let Some(copy) = copy else {
            return false;
        };
        if copy.extent_px.0 == 0 || copy.extent_px.1 == 0 {
            return false;
        }
        profiling::scope!("graph::per_view::offscreen_color_copy");
        let copy_query =
            profiler.map(|p| p.begin_query("graph::per_view::offscreen_color_copy", encoder));
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &copy.source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &copy.destination_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: copy.destination_origin_px.0,
                    y: copy.destination_origin_px.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: copy.extent_px.0,
                height: copy.extent_px.1,
                depth_or_array_layers: 1,
            },
        );
        if let Some(query) = copy_query
            && let Some(profiler) = profiler
        {
            profiler.end_query(encoder, query);
        }
        true
    }

    /// Builds [`crate::graph_inputs::GraphPassFrame`] for one per-view pass batch.
    fn build_per_view_frame_params<'a>(
        shared: &'a PerViewRecordShared<'a>,
        resolved: &'a ResolvedView<'a>,
        host_camera: &crate::camera::HostCameraFrame,
        render_context: crate::shared::RenderingContext,
        frame_time_seconds: f32,
        clear: crate::graph_inputs::FrameViewClear,
    ) -> crate::graph_inputs::GraphPassFrame<'a> {
        profiling::scope!("graph::per_view::build_frame_params");
        let hi_z_slot = shared.occlusion.ensure_hi_z_state(resolved.view_id);
        helpers::frame_render_params_from_shared(
            FrameSystemsShared {
                scene: shared.scene,
                occlusion: shared.occlusion,
                frame_resources: shared.frame_resources,
                materials: shared.materials,
                asset_resources: shared.asset_resources,
                mesh_preprocess: shared.mesh_preprocess,
                mesh_deform_scratch: None,
                mesh_deform_skin_cache: None,
                skin_cache: shared.skin_cache,
                debug_hud: shared.debug_hud,
            },
            helpers::GraphPassFrameViewInputs {
                resolved,
                scene_color_format: shared.scene_color_format,
                host_camera,
                render_context,
                frame_time_seconds,
                clear,
                post_processing: resolved.post_processing,
                gpu_limits: shared.gpu_limits_arc.clone(),
                msaa_depth_resolve: shared.msaa_depth_resolve.clone(),
                hi_z_slot,
            },
        )
    }
}
