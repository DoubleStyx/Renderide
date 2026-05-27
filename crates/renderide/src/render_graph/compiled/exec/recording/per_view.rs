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
use crate::render_graph::schedule::{
    RecordingBatch, RecordingBatchKind, RecordingUnit, RenderPassMaterializationGroup,
};

use super::super::super::helpers;
use super::super::super::{CompiledRenderGraph, ResolvedView};
use super::super::{
    GraphResolveKey, PerViewEncodeOutput, PerViewRecordShared, PerViewWorkItem,
    ResolvedOffscreenColorCopy, elapsed_ms,
};

struct PerViewUnitEncodeOutput {
    command_buffer: wgpu::CommandBuffer,
    encode_ms: f64,
    finish_ms: f64,
    command_stats: crate::render_graph::blackboard::GraphCommandStats,
}

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

        let (command_buffers, command_stats, encode_ms, finish_ms) = if self
            .schedule
            .recording_plan
            .phase_has_parallel_batches(PassPhase::PerView)
        {
            self.record_one_view_scheduler_v2(
                shared,
                view_idx,
                &resolved_view,
                graph_resources,
                &host_camera,
                render_context,
                frame_time_seconds,
                clear,
                &mut frame_params,
                &mut view_blackboard,
                resolved.offscreen_color_copy.as_ref(),
                upload_batch,
                profiler,
            )?
        } else {
            self.record_one_view_flat(
                shared,
                view_idx,
                &resolved_view,
                graph_resources,
                &mut frame_params,
                &mut view_blackboard,
                resolved.offscreen_color_copy.as_ref(),
                upload_batch,
                profiler,
            )?
        };
        let hud_outputs = view_blackboard.take::<PerViewHudOutputsSlot>();
        let encode_ms = encode_ms.max(elapsed_ms(encode_start));
        Ok(PerViewEncodeOutput {
            command_buffers,
            hud_outputs,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "flat recording keeps existing borrow scopes explicit"
    )]
    fn record_one_view_flat<'a>(
        &self,
        shared: &'a PerViewRecordShared<'a>,
        view_idx: usize,
        resolved_view: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
        view_blackboard: &mut crate::render_graph::blackboard::Blackboard,
        offscreen_color_copy: Option<&ResolvedOffscreenColorCopy>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<
        (
            Vec<wgpu::CommandBuffer>,
            crate::render_graph::blackboard::GraphCommandStats,
            f64,
            f64,
        ),
        GraphExecuteError,
    > {
        let device = shared.device;
        let encode_start = Instant::now();
        let mut encoder = {
            profiling::scope!("graph::per_view::create_encoder");
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-graph-per-view"),
            })
        };
        let gpu_query = profiler.map(|p| p.begin_query("graph::per_view", &mut encoder));
        {
            profiling::scope!("graph::per_view::pass_loop");
            self.record_phase_steps(
                PassPhase::PerView,
                Some(view_idx),
                resolved_view,
                graph_resources,
                frame_params,
                view_blackboard,
                &mut encoder,
                shared.device,
                shared.gpu_limits,
                upload_batch,
                profiler,
            )?;
        }
        let offscreen_copy_recorded =
            Self::record_offscreen_color_copy(&mut encoder, offscreen_color_copy, profiler);
        if let Some(query) = gpu_query
            && let Some(prof) = profiler
        {
            prof.end_query(&mut encoder, query);
        }
        let mut command_stats = view_blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        if offscreen_color_copy.is_some() {
            command_stats.record_copy_result(offscreen_copy_recorded);
        }
        let encode_ms = elapsed_ms(encode_start);
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            let finish_ms = elapsed_ms(finish_start);
            (command_buffer, finish_ms)
        };
        Ok((vec![command_buffer], command_stats, encode_ms, finish_ms))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "scheduler v2 needs independent serial and worker borrows"
    )]
    fn record_one_view_scheduler_v2<'a>(
        &self,
        shared: &'a PerViewRecordShared<'a>,
        view_idx: usize,
        resolved_view: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        host_camera: &crate::camera::HostCameraFrame,
        render_context: crate::shared::RenderingContext,
        frame_time_seconds: f32,
        clear: crate::graph_inputs::FrameViewClear,
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
        view_blackboard: &mut crate::render_graph::blackboard::Blackboard,
        offscreen_color_copy: Option<&ResolvedOffscreenColorCopy>,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<
        (
            Vec<wgpu::CommandBuffer>,
            crate::render_graph::blackboard::GraphCommandStats,
            f64,
            f64,
        ),
        GraphExecuteError,
    > {
        profiling::scope!("graph::per_view::scheduler_v2");
        let mut command_buffers = Vec::new();
        let mut parallel_stats = crate::render_graph::blackboard::GraphCommandStats::default();
        let mut encode_ms = 0.0;
        let mut finish_ms = 0.0;
        for batch in self
            .schedule
            .recording_plan
            .phase_batches(PassPhase::PerView)
        {
            match batch.kind {
                RecordingBatchKind::Serial => {
                    for unit_idx in batch.start_unit..batch.end_unit {
                        let output = self.record_serial_unit(
                            shared,
                            view_idx,
                            resolved_view,
                            graph_resources,
                            frame_params,
                            view_blackboard,
                            unit_idx,
                            upload_batch,
                            profiler,
                        )?;
                        encode_ms += output.encode_ms;
                        finish_ms += output.finish_ms;
                        command_buffers.push(output.command_buffer);
                    }
                }
                RecordingBatchKind::Parallel => {
                    let outputs = self.record_parallel_batch(
                        shared,
                        view_idx,
                        resolved_view,
                        graph_resources,
                        host_camera,
                        render_context,
                        frame_time_seconds,
                        clear,
                        view_blackboard,
                        batch,
                        upload_batch,
                        profiler,
                    )?;
                    for output in outputs {
                        encode_ms += output.encode_ms;
                        finish_ms += output.finish_ms;
                        parallel_stats.add(output.command_stats);
                        command_buffers.push(output.command_buffer);
                    }
                }
            }
        }
        if let Some(copy_output) =
            Self::record_offscreen_color_copy_command(shared.device, offscreen_color_copy, profiler)
        {
            let (command_buffer, recorded, copy_encode_ms, copy_finish_ms) = copy_output;
            encode_ms += copy_encode_ms;
            finish_ms += copy_finish_ms;
            let mut stats = crate::render_graph::blackboard::GraphCommandStats::default();
            stats.record_copy_result(recorded);
            parallel_stats.add(stats);
            command_buffers.push(command_buffer);
        } else if offscreen_color_copy.is_some() {
            let mut stats = crate::render_graph::blackboard::GraphCommandStats::default();
            stats.record_copy_result(false);
            parallel_stats.add(stats);
        }
        let mut command_stats = view_blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        command_stats.add(parallel_stats);
        Ok((command_buffers, command_stats, encode_ms, finish_ms))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "unit recording mirrors pass context construction"
    )]
    fn record_serial_unit<'a>(
        &self,
        shared: &'a PerViewRecordShared<'a>,
        view_idx: usize,
        resolved_view: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
        view_blackboard: &mut crate::render_graph::blackboard::Blackboard,
        unit_idx: usize,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<PerViewUnitEncodeOutput, GraphExecuteError> {
        let unit = self.schedule.recording_plan.units[unit_idx];
        self.record_unit_command_buffer(
            shared.device,
            shared.gpu_limits,
            view_idx,
            resolved_view,
            graph_resources,
            frame_params,
            view_blackboard,
            unit,
            upload_batch,
            profiler,
            "render-graph-per-view-unit",
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "parallel workers rebuild view-local frame params"
    )]
    fn record_parallel_batch<'a>(
        &self,
        shared: &'a PerViewRecordShared<'a>,
        view_idx: usize,
        resolved_view: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        host_camera: &crate::camera::HostCameraFrame,
        render_context: crate::shared::RenderingContext,
        frame_time_seconds: f32,
        clear: crate::graph_inputs::FrameViewClear,
        view_blackboard: &crate::render_graph::blackboard::Blackboard,
        batch: RecordingBatch,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
    ) -> Result<Vec<PerViewUnitEncodeOutput>, GraphExecuteError> {
        profiling::scope!("graph::per_view::scheduler_v2::parallel_batch");
        use rayon::prelude::*;
        let mut outputs = (batch.start_unit..batch.end_unit)
            .into_par_iter()
            .map(|unit_idx| {
                let unit = self.schedule.recording_plan.units[unit_idx];
                let mut frame_params = Self::build_per_view_frame_params(
                    shared,
                    resolved_view,
                    host_camera,
                    render_context,
                    frame_time_seconds,
                    clear,
                );
                let mut local_blackboard = view_blackboard.clone_read_only();
                local_blackboard.insert_untracked::<GraphCommandStatsSlot>(
                    crate::render_graph::blackboard::GraphCommandStats::default(),
                );
                let output = self.record_unit_command_buffer(
                    shared.device,
                    shared.gpu_limits,
                    view_idx,
                    resolved_view,
                    graph_resources,
                    &mut frame_params,
                    &mut local_blackboard,
                    unit,
                    upload_batch,
                    profiler,
                    "render-graph-per-view-parallel-unit",
                )?;
                Ok((unit_idx, output))
            })
            .collect::<Result<Vec<_>, GraphExecuteError>>()?;
        outputs.sort_unstable_by_key(|(unit_idx, _)| *unit_idx);
        Ok(outputs.into_iter().map(|(_, output)| output).collect())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "unit recording keeps frame and blackboard borrows narrow"
    )]
    fn record_unit_command_buffer<'a>(
        &self,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        view_idx: usize,
        resolved_view: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
        blackboard: &mut crate::render_graph::blackboard::Blackboard,
        unit: RecordingUnit,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
        encoder_label: &'static str,
    ) -> Result<PerViewUnitEncodeOutput, GraphExecuteError> {
        let encode_start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(encoder_label),
        });
        let query_label = self.recording_unit_label(unit);
        let gpu_query = profiler.map(|p| p.begin_query(query_label.as_str(), &mut encoder));
        self.record_unit_into_encoder(
            view_idx,
            resolved_view,
            graph_resources,
            frame_params,
            blackboard,
            &mut encoder,
            device,
            gpu_limits,
            upload_batch,
            profiler,
            unit,
        )?;
        if let Some(query) = gpu_query
            && let Some(prof) = profiler
        {
            prof.end_query(&mut encoder, query);
        }
        let command_stats = blackboard
            .get_untracked::<GraphCommandStatsSlot>()
            .copied()
            .unwrap_or_default();
        let encode_ms = elapsed_ms(encode_start);
        let (command_buffer, finish_ms) = {
            profiling::scope!("CommandEncoder::finish::graph_per_view_unit");
            let finish_start = Instant::now();
            let command_buffer = encoder.finish();
            (command_buffer, elapsed_ms(finish_start))
        };
        Ok(PerViewUnitEncodeOutput {
            command_buffer,
            encode_ms,
            finish_ms,
            command_stats,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "shares dispatch path with flat recording"
    )]
    fn record_unit_into_encoder<'a>(
        &self,
        view_idx: usize,
        resolved_view: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::graph_inputs::GraphPassFrame<'a>,
        blackboard: &mut crate::render_graph::blackboard::Blackboard,
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        upload_batch: &FrameUploadBatch,
        profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
        unit: RecordingUnit,
    ) -> Result<(), GraphExecuteError> {
        if unit.is_materialized_group()
            && self.try_execute_raster_materialization_group(
                RenderPassMaterializationGroup {
                    start_step: unit.start_step,
                    end_step: unit.end_step,
                },
                PassPhase::PerView,
                Some(view_idx),
                graph_resources,
                frame_params,
                blackboard,
                encoder,
                device,
                upload_batch,
                profiler,
            )?
        {
            return Ok(());
        }
        for step in &self.schedule.steps[unit.start_step..unit.end_step] {
            self.execute_pass_node(
                step.pass_idx,
                step.frame_upload_scope(Some(view_idx)),
                resolved_view,
                graph_resources,
                frame_params,
                blackboard,
                encoder,
                device,
                gpu_limits,
                upload_batch,
                profiler,
            )?;
        }
        Ok(())
    }

    fn recording_unit_label(&self, unit: RecordingUnit) -> String {
        let mut label = String::from("graph::per_view::unit(");
        for (idx, step) in self.schedule.steps[unit.start_step..unit.end_step]
            .iter()
            .enumerate()
        {
            if idx != 0 {
                label.push_str(" + ");
            }
            label.push_str(self.passes[step.pass_idx].profiling_label().as_ref());
        }
        label.push(')');
        label
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

    fn record_offscreen_color_copy_command(
        device: &wgpu::Device,
        copy: Option<&ResolvedOffscreenColorCopy>,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Option<(wgpu::CommandBuffer, bool, f64, f64)> {
        let copy = copy?;
        if copy.extent_px.0 == 0 || copy.extent_px.1 == 0 {
            return None;
        }
        let encode_start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-per-view-offscreen-copy"),
        });
        let recorded = Self::record_offscreen_color_copy(&mut encoder, Some(copy), profiler);
        let encode_ms = elapsed_ms(encode_start);
        let finish_start = Instant::now();
        let command_buffer = encoder.finish();
        let finish_ms = elapsed_ms(finish_start);
        Some((command_buffer, recorded, encode_ms, finish_ms))
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
