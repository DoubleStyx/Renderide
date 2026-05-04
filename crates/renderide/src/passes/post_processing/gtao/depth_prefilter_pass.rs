//! `gtao_prefilter_*` compute passes -- XeGTAO view-space depth mip chain.
//!
//! XeGTAO samples a prefiltered view-space depth pyramid rather than the raw hardware depth
//! buffer. The mips reduce large-radius bandwidth, stabilize horizon samples, and prevent
//! distant depth discontinuities from dominating contact shadows.

use std::borrow::Cow;

use super::pipeline::{GtaoParamsGpu, GtaoPipelines};
use crate::passes::helpers::missing_pass_resource;
use crate::passes::post_processing::settings_slot::GtaoSettingsSlot;
use crate::profiling::compute_pass_timestamp_writes;
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::PerViewFramePlanSlot;
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, SubresourceHandle,
    TextureAccess,
};

const GTAO_PREFILTER_WORKGROUP_SIZE: u32 = 8;

/// Source/destination subresources for one depth prefilter node.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoDepthPrefilterResources {
    /// Imported raw scene depth used when building mip 0.
    pub depth: ImportedTextureHandle,
    /// Imported frame-uniforms buffer used when the per-view buffer slot is absent.
    pub frame_uniforms: ImportedBufferHandle,
    /// Previous view-space depth mip for downsample nodes.
    pub source_mip: Option<SubresourceHandle>,
    /// Destination view-space depth mip/layer.
    pub output_mip: SubresourceHandle,
}

/// Computes one layer of one view-space depth mip.
pub(super) struct GtaoDepthPrefilterPass {
    resources: GtaoDepthPrefilterResources,
    settings: crate::config::GtaoSettings,
    pipelines: &'static GtaoPipelines,
    mip_level: u32,
    layer: u32,
    profile_label: String,
}

impl GtaoDepthPrefilterPass {
    /// Creates a raw-depth-to-mip0 prefilter pass.
    pub(super) fn mip0(
        resources: GtaoDepthPrefilterResources,
        settings: crate::config::GtaoSettings,
        pipelines: &'static GtaoPipelines,
        layer: u32,
    ) -> Self {
        Self {
            resources,
            settings,
            pipelines,
            mip_level: 0,
            layer,
            profile_label: format!("GtaoDepthPrefilter.mip0.layer{layer}"),
        }
    }

    /// Creates a downsample pass from mip `mip_level - 1` into `mip_level`.
    pub(super) fn downsample(
        resources: GtaoDepthPrefilterResources,
        settings: crate::config::GtaoSettings,
        pipelines: &'static GtaoPipelines,
        mip_level: u32,
        layer: u32,
    ) -> Self {
        Self {
            resources,
            settings,
            pipelines,
            mip_level,
            layer,
            profile_label: format!("GtaoDepthPrefilter.mip{mip_level}.layer{layer}"),
        }
    }

    fn is_mip0(&self) -> bool {
        self.resources.source_mip.is_none()
    }

    fn output_extent(&self, viewport_px: (u32, u32)) -> (u32, u32) {
        (
            (viewport_px.0.max(1) >> self.mip_level).max(1),
            (viewport_px.1.max(1) >> self.mip_level).max(1),
        )
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

    fn dispatch_grid(&self, viewport_px: (u32, u32)) -> Option<(u32, u32)> {
        let (width, height) = self.output_extent(viewport_px);
        let gx = width.div_ceil(GTAO_PREFILTER_WORKGROUP_SIZE);
        let gy = height.div_ceil(GTAO_PREFILTER_WORKGROUP_SIZE);
        (gx > 0 && gy > 0).then_some((gx, gy))
    }

    fn record_mip0(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
        output_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
        gx: u32,
        gy: u32,
    ) -> Result<(), RenderPassError> {
        let frame_uniform_buffer = self.resolve_frame_uniform_buffer(ctx)?;
        let source_view =
            ctx.pass_frame
                .view
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gtao_prefilter_raw_depth_layer"),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: self.layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao_prefilter_mip0"),
            layout: self
                .pipelines
                .depth_prefilter
                .mip0_bind_group_layout(ctx.device),
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
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });
        let pipeline = self.pipelines.depth_prefilter.mip0_pipeline(ctx.device);
        dispatch_prefilter(
            ctx,
            self.profile_label.as_str(),
            pipeline.as_ref(),
            &bind_group,
            gx,
            gy,
        );
        Ok(())
    }

    fn record_downsample(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
        output_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
        gx: u32,
        gy: u32,
    ) -> Result<(), RenderPassError> {
        let Some(source_mip) = self.resources.source_mip else {
            return Err(missing_pass_resource(self.name(), "missing source mip"));
        };
        let Some(source_view) = ctx.graph_resources.subresource_view(source_mip) else {
            return Ok(());
        };
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao_prefilter_downsample"),
            layout: self
                .pipelines
                .depth_prefilter
                .downsample_bind_group_layout(ctx.device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });
        let pipeline = self
            .pipelines
            .depth_prefilter
            .downsample_pipeline(ctx.device);
        dispatch_prefilter(
            ctx,
            self.profile_label.as_str(),
            pipeline.as_ref(),
            &bind_group,
            gx,
            gy,
        );
        Ok(())
    }
}

impl ComputePass for GtaoDepthPrefilterPass {
    fn name(&self) -> &str {
        "GtaoDepthPrefilter"
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.profile_label.as_str())
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        if let Some(source_mip) = self.resources.source_mip {
            b.read_texture_subresource(
                source_mip,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                },
            );
        } else {
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
        }
        b.write_texture_subresource(
            self.resources.output_mip,
            TextureAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        Ok(())
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::gtao_depth_prefilter");
        let Some(output_view) = ctx
            .graph_resources
            .subresource_view(self.resources.output_mip)
        else {
            return Ok(());
        };
        let params = self.live_params(ctx);
        let params_buffer = self.pipelines.params.get(ctx.device);
        ctx.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let Some((gx, gy)) = self.dispatch_grid(ctx.pass_frame.view.viewport_px) else {
            return Ok(());
        };

        if self.is_mip0() {
            return self.record_mip0(ctx, output_view, params_buffer, gx, gy);
        }
        self.record_downsample(ctx, output_view, params_buffer, gx, gy)
    }
}

fn dispatch_prefilter(
    ctx: &mut ComputePassCtx<'_, '_, '_>,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    gx: u32,
    gy: u32,
) {
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
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    if let (Some(query), Some(profiler)) = (pass_query, ctx.profiler) {
        profiler.end_query(ctx.encoder, query);
    }
}
