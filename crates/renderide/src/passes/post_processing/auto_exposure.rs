//! Histogram-based auto-exposure post-processing effect.
//!
//! The effect is split into two graph passes: a compute pass that meters HDR scene color and
//! updates per-view exposure EV state, followed by a fullscreen raster pass that multiplies HDR
//! scene color by the current exposure before tonemapping.

mod pipeline;

use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::{Arc, LazyLock};

use hashbrown::HashMap;
use parking_lot::Mutex;

use pipeline::{
    AutoExposureParamsGpu, AutoExposurePipelineCache, HISTOGRAM_WORKGROUP_HEIGHT,
    HISTOGRAM_WORKGROUP_WIDTH, ViewAutoExposureGpuState,
};

use crate::camera::ViewId;
use crate::config::PostProcessingSettings;
use crate::passes::helpers::{
    color_attachment, missing_pass_resource, read_fragment_sampled_texture,
};
use crate::passes::post_processing::settings_slot::AutoExposureSettingsSlot;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::{ComputePassCtx, GraphResolvedResources, RasterPassCtx};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::raster_stereo_mask_override;
use crate::render_graph::pass::{ComputePass, PassBuilder, RasterPass};
use crate::render_graph::post_processing::{EffectPasses, PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{TextureAccess, TextureHandle};

/// Compute pass that meters scene luminance and updates persistent exposure EV state.
pub struct AutoExposureComputePass {
    input: TextureHandle,
    state_cache: Arc<AutoExposureStateCache>,
    pipelines: &'static AutoExposurePipelineCache,
}

impl AutoExposureComputePass {
    fn new(input: TextureHandle, state_cache: Arc<AutoExposureStateCache>) -> Self {
        Self {
            input,
            state_cache,
            pipelines: auto_exposure_pipelines(),
        }
    }
}

impl ComputePass for AutoExposureComputePass {
    fn name(&self) -> &str {
        "AutoExposureCompute"
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("AutoExposure.compute")
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.read_texture_resource(
            self.input,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        Ok(())
    }

    fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        self.state_cache.retire_views(retired_views);
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::auto_exposure::compute");
        let frame = &*ctx.pass_frame;
        let graph_resources = ctx.graph_resources;
        let Some(tex) = graph_resources.transient_texture(self.input) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing transient input {:?}", self.input),
            ));
        };

        let layer_count = auto_exposure_layer_count(tex.array_layers, frame.view.multiview_stereo);
        let settings = ctx
            .blackboard
            .get::<AutoExposureSettingsSlot>()
            .copied()
            .unwrap_or_default();
        let params = AutoExposureParamsGpu::from_settings(
            settings.settings,
            settings.delta_seconds,
            layer_count,
        );
        let state = self.state_cache.ensure(ctx.device, frame.view.view_id);
        ctx.write_buffer(&state.settings, 0, bytemuck::bytes_of(&params));

        let bind_group = self.pipelines.compute_bind_group(
            ctx.device,
            &tex.texture,
            frame.view.multiview_stereo,
            &state,
        );

        let size = tex.texture.size();
        let groups_x = size.width.div_ceil(HISTOGRAM_WORKGROUP_WIDTH).max(1);
        let groups_y = size.height.div_ceil(HISTOGRAM_WORKGROUP_HEIGHT).max(1);
        let pass_query = ctx
            .profiler
            .map(|profiler| profiler.begin_pass_query("auto_exposure", ctx.encoder));
        let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut cpass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("auto_exposure"),
                    timestamp_writes,
                });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(self.pipelines.histogram_pipeline(ctx.device));
            cpass.dispatch_workgroups(groups_x, groups_y, layer_count);
            cpass.set_pipeline(self.pipelines.average_pipeline(ctx.device));
            cpass.dispatch_workgroups(1, 1, 1);
        }
        if let (Some(profiler), Some(query)) = (ctx.profiler, pass_query) {
            profiler.end_query(ctx.encoder, query);
        }

        Ok(())
    }
}

/// Fullscreen pass that applies the current exposure EV to HDR scene color.
pub struct AutoExposureApplyPass {
    input: TextureHandle,
    output: TextureHandle,
    state_cache: Arc<AutoExposureStateCache>,
    pipelines: &'static AutoExposurePipelineCache,
}

impl AutoExposureApplyPass {
    fn new(
        input: TextureHandle,
        output: TextureHandle,
        state_cache: Arc<AutoExposureStateCache>,
    ) -> Self {
        Self {
            input,
            output,
            state_cache,
            pipelines: auto_exposure_pipelines(),
        }
    }
}

impl RasterPass for AutoExposureApplyPass {
    fn name(&self) -> &str {
        "AutoExposureApply"
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("AutoExposure.apply")
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.input);
        color_attachment(b, self.output, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        raster_stereo_mask_override(ctx, template)
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::auto_exposure::apply");
        let frame = &*ctx.pass_frame;
        let graph_resources = ctx.graph_resources;
        let Some(tex) = graph_resources.transient_texture(self.input) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing transient input {:?}", self.input),
            ));
        };

        let output_format = output_attachment_format(self.output, graph_resources);
        let pipeline =
            self.pipelines
                .apply_pipeline(ctx.device, output_format, frame.view.multiview_stereo);
        let state = self.state_cache.ensure(ctx.device, frame.view.view_id);
        let bind_group = self.pipelines.apply_bind_group(
            ctx.device,
            &tex.texture,
            frame.view.multiview_stereo,
            &state,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Effect descriptor that contributes auto-exposure compute and apply passes to the chain.
pub struct AutoExposureEffect;

impl PostProcessEffect for AutoExposureEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::AutoExposure
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.auto_exposure.enabled
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let state_cache = Arc::new(AutoExposureStateCache::default());
        let compute = builder.add_compute_pass(Box::new(AutoExposureComputePass::new(
            input,
            Arc::clone(&state_cache),
        )));
        let apply = builder.add_raster_pass(Box::new(AutoExposureApplyPass::new(
            input,
            output,
            state_cache,
        )));
        builder.add_edge(compute, apply);
        EffectPasses {
            first: compute,
            last: apply,
        }
    }
}

#[derive(Default)]
struct AutoExposureStateCache {
    per_view: Mutex<HashMap<ViewId, Arc<ViewAutoExposureGpuState>>>,
}

impl AutoExposureStateCache {
    fn ensure(&self, device: &wgpu::Device, view_id: ViewId) -> Arc<ViewAutoExposureGpuState> {
        let mut per_view = self.per_view.lock();
        Arc::clone(
            per_view
                .entry(view_id)
                .or_insert_with(|| Arc::new(ViewAutoExposureGpuState::new(device))),
        )
    }

    fn retire_views(&self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        let mut per_view = self.per_view.lock();
        for view_id in retired_views {
            per_view.remove(view_id);
        }
    }
}

fn auto_exposure_pipelines() -> &'static AutoExposurePipelineCache {
    static CACHE: LazyLock<AutoExposurePipelineCache> =
        LazyLock::new(AutoExposurePipelineCache::default);
    &CACHE
}

fn auto_exposure_layer_count(texture_layers: u32, multiview_stereo: bool) -> u32 {
    if multiview_stereo {
        texture_layers.clamp(1, 2)
    } else {
        1
    }
}

fn output_attachment_format(
    output: TextureHandle,
    graph_resources: &GraphResolvedResources,
) -> wgpu::TextureFormat {
    graph_resources
        .transient_texture(output)
        .map_or(wgpu::TextureFormat::Rgba16Float, |t| t.texture.format())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::render_graph::GraphBuilder;
    use crate::render_graph::pass::PassBuilder;
    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::resources::{
        AccessKind, TransientArrayLayers, TransientExtent, TransientSampleCount,
        TransientTextureDesc, TransientTextureFormat,
    };

    fn fake_textures(builder: &mut GraphBuilder) -> (TextureHandle, TextureHandle) {
        let desc = || TransientTextureDesc {
            label: "auto_exposure_hdr",
            format: TransientTextureFormat::SceneColorHdr,
            extent: TransientExtent::Backbuffer,
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Frame,
            base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            alias: true,
        };
        (
            builder.create_texture(desc()),
            builder.create_texture(desc()),
        )
    }

    #[test]
    fn compute_setup_declares_compute_sampled_input() {
        let mut graph = GraphBuilder::new();
        let (input, _) = fake_textures(&mut graph);
        let cache = Arc::new(AutoExposureStateCache::default());
        let mut pass = AutoExposureComputePass::new(input, cache);
        let mut builder = PassBuilder::new("AutoExposureCompute");

        pass.setup(&mut builder).expect("setup");
        let setup = builder.finish().expect("finish");

        assert_eq!(setup.kind, PassKind::Compute);
        assert!(
            setup.accesses.iter().any(|access| matches!(
                &access.access,
                AccessKind::Texture(TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::COMPUTE,
                    ..
                })
            )),
            "expected sampled HDR input read"
        );
    }

    #[test]
    fn apply_setup_declares_sampled_input_and_raster_output() {
        let mut graph = GraphBuilder::new();
        let (input, output) = fake_textures(&mut graph);
        let cache = Arc::new(AutoExposureStateCache::default());
        let mut pass = AutoExposureApplyPass::new(input, output, cache);
        let mut builder = PassBuilder::new("AutoExposureApply");

        pass.setup(&mut builder).expect("setup");
        let setup = builder.finish().expect("finish");

        assert_eq!(setup.kind, PassKind::Raster);
        assert!(
            setup.accesses.iter().any(|access| matches!(
                &access.access,
                AccessKind::Texture(TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )),
            "expected sampled HDR input read"
        );
        assert_eq!(setup.color_attachments.len(), 1);
    }

    #[test]
    fn effect_id_and_enable_gate_match_settings() {
        let effect = AutoExposureEffect;
        assert_eq!(effect.id(), PostProcessEffectId::AutoExposure);
        assert!(effect.is_enabled(&PostProcessingSettings::default()));

        let mut settings = PostProcessingSettings::default();
        settings.auto_exposure.enabled = false;
        assert!(!effect.is_enabled(&settings));
    }

    #[test]
    fn layer_count_is_shared_for_stereo_views() {
        assert_eq!(auto_exposure_layer_count(2, true), 2);
        assert_eq!(auto_exposure_layer_count(6, true), 2);
        assert_eq!(auto_exposure_layer_count(2, false), 1);
    }
}
