//! `gtao_prefilter_mip0` compute pass -- GTAO view-space depth pyramid.
//!
//! The effect samples a prefiltered view-space depth pyramid rather than the raw hardware depth
//! buffer. This pass writes all five depth levels in one dispatch, reducing graph-pass overhead
//! and keeping the small mip reductions in workgroup-local memory.

use std::borrow::Cow;

use super::pipeline::{GtaoParamsGpu, GtaoPipelines, VIEW_DEPTH_MIP_COUNT};
use crate::graph_inputs::PerViewFramePlanSlot;
use crate::passes::helpers::missing_pass_resource;
use crate::passes::post_processing::settings_slots::GtaoSettingsSlot;
use crate::profiling::compute_pass_timestamp_writes;
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, SubresourceHandle,
    TextureAccess, TextureHandle,
};

const GTAO_PREFILTER_WORKGROUP_SIZE: u32 = 8;
const GTAO_PREFILTER_MIP0_PIXELS_PER_INVOCATION: u32 = 2;

/// Source and destination resources for the combined GTAO depth prefilter.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoDepthPrefilterResources {
    /// Imported raw scene depth used when building mip 0.
    pub depth: ImportedTextureHandle,
    /// Imported frame-uniforms buffer used when the per-view buffer slot is absent.
    pub frame_uniforms: ImportedBufferHandle,
    /// Destination view-space depth textures, one per GTAO depth level.
    pub output_textures: [TextureHandle; VIEW_DEPTH_MIP_COUNT as usize],
    /// Destination view-space depth writable subresources.
    pub output_mips: [SubresourceHandle; VIEW_DEPTH_MIP_COUNT as usize],
}

/// Computes all view-space depth levels, dispatching one workgroup layer per stereo eye.
pub(super) struct GtaoDepthPrefilterPass {
    resources: GtaoDepthPrefilterResources,
    settings: crate::config::GtaoSettings,
    pipelines: &'static GtaoPipelines,
    view_depth_multiview_stereo: bool,
}

impl GtaoDepthPrefilterPass {
    /// Creates a combined raw-depth-to-view-depth prefilter pass.
    pub(super) fn new(
        resources: GtaoDepthPrefilterResources,
        settings: crate::config::GtaoSettings,
        pipelines: &'static GtaoPipelines,
        view_depth_multiview_stereo: bool,
    ) -> Self {
        Self {
            resources,
            settings,
            pipelines,
            view_depth_multiview_stereo,
        }
    }

    fn resolve_frame_uniform_buffer(
        &self,
        ctx: &ComputePassCtx<'_, '_, '_>,
    ) -> Result<wgpu::Buffer, RenderPassError> {
        if let Some(plan) = ctx.blackboard.get::<PerViewFramePlanSlot>() {
            return Ok(plan.frame_uniform_buffer.clone());
        }
        ctx.graph_resources
            .imported_buffer(self.resources.frame_uniforms)
            .map(|resolved| resolved.buffer.clone())
            .ok_or_else(|| missing_pass_resource(self.name(), "frame_uniforms not resolved"))
    }

    fn live_params(&self, ctx: &ComputePassCtx<'_, '_, '_>) -> GtaoParamsGpu {
        let live = ctx
            .blackboard
            .get::<GtaoSettingsSlot>()
            .map_or(self.settings, |slot| slot.0);
        GtaoParamsGpu::from_settings(live, 0.0, false)
    }

    fn bind_group_multiview_stereo(&self) -> bool {
        self.view_depth_multiview_stereo
    }

    fn dispatch_layer_count(&self, view_multiview_stereo: bool) -> u32 {
        if self.view_depth_multiview_stereo {
            super::stereo_array_layer_count(view_multiview_stereo)
        } else {
            1
        }
    }

    fn dispatch_grid(&self, mip0_extent: (u32, u32)) -> Option<(u32, u32)> {
        let mip1_width = mip0_extent
            .0
            .max(1)
            .div_ceil(GTAO_PREFILTER_MIP0_PIXELS_PER_INVOCATION);
        let mip1_height = mip0_extent
            .1
            .max(1)
            .div_ceil(GTAO_PREFILTER_MIP0_PIXELS_PER_INVOCATION);
        let gx = mip1_width.div_ceil(GTAO_PREFILTER_WORKGROUP_SIZE);
        let gy = mip1_height.div_ceil(GTAO_PREFILTER_WORKGROUP_SIZE);
        (gx > 0 && gy > 0).then_some((gx, gy))
    }

    fn output_extent(
        &self,
        ctx: &ComputePassCtx<'_, '_, '_>,
    ) -> Result<(u32, u32), RenderPassError> {
        let m0 = self.resources.output_textures[0];
        let t0 = ctx.graph_resources.transient_texture(m0).ok_or_else(|| {
            missing_pass_resource(self.name(), format_args!("missing output mip {m0:?}"))
        })?;
        Ok((t0.width, t0.height))
    }
}

fn raw_depth_view_label(bind_group_multiview_stereo: bool) -> &'static str {
    if bind_group_multiview_stereo {
        "gtao_prefilter_raw_depth_stereo"
    } else {
        "gtao_prefilter_raw_depth_mono"
    }
}

fn prefilter_view_dimension(bind_group_multiview_stereo: bool) -> wgpu::TextureViewDimension {
    if bind_group_multiview_stereo {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

impl ComputePass for GtaoDepthPrefilterPass {
    fn name(&self) -> &str {
        "GtaoDepthPrefilter"
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("GtaoDepthPrefilter.combined")
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.read_optional_blackboard::<PerViewFramePlanSlot>();
        b.read_optional_blackboard::<crate::passes::WorldMeshForwardPlanSlot>();
        b.read_blackboard::<GtaoSettingsSlot>();
        b.import_texture(
            self.resources.depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        b.import_buffer(
            self.resources.frame_uniforms,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::COMPUTE,
                dynamic_offset: false,
            },
        );
        for output_mip in self.resources.output_mips {
            b.write_texture_subresource(
                output_mip,
                TextureAccess::Storage {
                    stages: wgpu::ShaderStages::COMPUTE,
                    access: StorageAccess::WriteOnly,
                },
            );
        }
        Ok(())
    }

    fn should_record(&self, ctx: &ComputePassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(super::gtao_view_recording_needed(
            ctx.blackboard,
            &ctx.pass_frame.view,
        ))
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::gtao_depth_prefilter");
        let mip0_extent = self.output_extent(ctx)?;
        let Some((gx, gy)) = self.dispatch_grid(mip0_extent) else {
            return Ok(());
        };

        let params = self.live_params(ctx);
        let params_buffer = self.pipelines.params.get(ctx.device);
        ctx.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let frame_uniform_buffer = self.resolve_frame_uniform_buffer(ctx)?;
        let bind_group_multiview_stereo = self.bind_group_multiview_stereo();
        let view_multiview_stereo = ctx.pass_frame.view.multiview_stereo;
        let layer_count = self.dispatch_layer_count(view_multiview_stereo);
        let source_view =
            ctx.pass_frame
                .view
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some(raw_depth_view_label(bind_group_multiview_stereo)),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    dimension: Some(prefilter_view_dimension(bind_group_multiview_stereo)),
                    base_array_layer: 0,
                    array_layer_count: Some(layer_count),
                    ..Default::default()
                });
        crate::profiling::note_resource_churn!(
            TextureView,
            "passes::gtao_prefilter_raw_depth_view"
        );

        let [m0, m1, m2, m3, m4] = self.resources.output_mips;
        let Some(output_view0) = ctx.graph_resources.subresource_view(m0) else {
            return Ok(());
        };
        let Some(output_view1) = ctx.graph_resources.subresource_view(m1) else {
            return Ok(());
        };
        let Some(output_view2) = ctx.graph_resources.subresource_view(m2) else {
            return Ok(());
        };
        let Some(output_view3) = ctx.graph_resources.subresource_view(m3) else {
            return Ok(());
        };
        let Some(output_view4) = ctx.graph_resources.subresource_view(m4) else {
            return Ok(());
        };

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao_prefilter_combined"),
            layout: self
                .pipelines
                .depth_prefilter
                .bind_group_layout(ctx.device, bind_group_multiview_stereo),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(output_view0),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(output_view1),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_view2),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(output_view3),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(output_view4),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "passes::gtao_prefilter_combined_bg");

        let pipeline = self
            .pipelines
            .depth_prefilter
            .pipeline(ctx.device, bind_group_multiview_stereo);
        dispatch_prefilter(
            ctx,
            self.profiling_label().as_ref(),
            pipeline.as_ref(),
            &bind_group,
            gx,
            gy,
            layer_count,
        );
        Ok(())
    }
}

fn dispatch_prefilter(
    ctx: &mut ComputePassCtx<'_, '_, '_>,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    gx: u32,
    gy: u32,
    layer_count: u32,
) {
    profiling::scope!("post_processing::gtao_depth_prefilter::dispatch");
    let pass_query = ctx
        .profiler
        .map(|profiler| profiler.begin_pass_query(label, ctx.encoder));
    {
        let mut cpass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
            });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(gx, gy, layer_count);
    }
    if let (Some(query), Some(profiler)) = (pass_query, ctx.profiler) {
        profiler.end_query(ctx.encoder, query);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GtaoSettings;

    fn resources() -> GtaoDepthPrefilterResources {
        GtaoDepthPrefilterResources {
            depth: ImportedTextureHandle(0),
            frame_uniforms: ImportedBufferHandle(0),
            output_textures: [
                TextureHandle(0),
                TextureHandle(1),
                TextureHandle(2),
                TextureHandle(3),
                TextureHandle(4),
            ],
            output_mips: [
                SubresourceHandle(0),
                SubresourceHandle(1),
                SubresourceHandle(2),
                SubresourceHandle(3),
                SubresourceHandle(4),
            ],
        }
    }

    #[test]
    fn stereo_graph_mono_view_uses_array_bindings_with_one_dispatch_layer() {
        let pass = GtaoDepthPrefilterPass::new(
            resources(),
            GtaoSettings::default(),
            super::super::gtao_pipelines(),
            true,
        );

        assert!(pass.bind_group_multiview_stereo());
        assert_eq!(
            prefilter_view_dimension(pass.bind_group_multiview_stereo()),
            wgpu::TextureViewDimension::D2Array
        );
        assert_eq!(pass.dispatch_layer_count(false), 1);
    }

    #[test]
    fn stereo_graph_stereo_view_dispatches_both_layers() {
        let pass = GtaoDepthPrefilterPass::new(
            resources(),
            GtaoSettings::default(),
            super::super::gtao_pipelines(),
            true,
        );

        assert_eq!(pass.dispatch_layer_count(true), 2);
    }

    #[test]
    fn mono_graph_uses_d2_bindings_and_single_layer() {
        let pass = GtaoDepthPrefilterPass::new(
            resources(),
            GtaoSettings::default(),
            super::super::gtao_pipelines(),
            false,
        );

        assert!(!pass.bind_group_multiview_stereo());
        assert_eq!(
            prefilter_view_dimension(pass.bind_group_multiview_stereo()),
            wgpu::TextureViewDimension::D2
        );
        assert_eq!(pass.dispatch_layer_count(false), 1);
    }

    #[test]
    fn dispatch_grid_targets_mip1_extent() {
        let pass = GtaoDepthPrefilterPass::new(
            resources(),
            GtaoSettings::default(),
            super::super::gtao_pipelines(),
            false,
        );

        assert_eq!(pass.dispatch_grid((1920, 1080)), Some((120, 68)));
        assert_eq!(pass.dispatch_grid((1, 1)), Some((1, 1)));
    }
}
