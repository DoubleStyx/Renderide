//! Ground-Truth Ambient Occlusion (Jimenez et al. 2016) render pass.
//!
//! Reads the chain's HDR scene color and the scene depth buffer, reconstructs view-space normals
//! from depth derivatives, evaluates the analytic cosine-weighted GTAO integral with a single
//! horizon direction per pixel (spatially jittered 4×4 + per-frame phase rotation), applies the
//! multi-bounce fit for near-field indirect light, and writes the HDR scene color modulated by
//! the resulting visibility factor. Must run **before** tonemapping so AO acts on linear light.
//!
//! Multiview is handled the same way as [`crate::render_graph::passes::AcesTonemapPass`]: two
//! pipeline variants (mono / multiview) picked via a `multiview_mask_override` of
//! `NonZeroU32::new(3)` in stereo, with `#ifdef MULTIVIEW` in the shader selecting
//! `@builtin(view_index)` and the depth-array sample path.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::OnceLock;

use pipeline::{GtaoParamsGpu, GtaoPipelineCache};

use crate::config::{GtaoSettings, PostProcessingSettings};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::{GtaoSettingsSlot, PerViewFramePlanSlot};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::post_processing::{EffectPasses, PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, TextureAccess, TextureHandle,
};

/// Graph handles for [`GtaoPass`].
#[derive(Clone, Copy, Debug)]
pub struct GtaoGraphResources {
    /// HDR scene-color input (chain stage's previous output, or the forward HDR target).
    pub input: TextureHandle,
    /// HDR chain output written by this pass (input × AO factor).
    pub output: TextureHandle,
    /// Frame depth texture (declared as a sampled dependency so the scheduler sees the read;
    /// the record path builds its own depth-only `TextureView` from `frame.view.depth_texture`
    /// since `ResolvedImportedTexture` only exposes the attachment view).
    pub depth: ImportedTextureHandle,
    /// Legacy-path fallback for the frame-uniforms buffer. In normal per-view rendering, the
    /// actual buffer bound at record time is [`PerViewFramePlan::frame_uniform_buffer`] (read
    /// from the blackboard); this import is kept only for graph-scheduling declaration and
    /// fallback when the per-view slot is absent.
    pub frame_uniforms: ImportedBufferHandle,
}

/// Fullscreen render pass applying GTAO to `input`, writing modulated HDR to `output`.
pub struct GtaoPass {
    /// Graph handles for this pass instance.
    resources: GtaoGraphResources,
    /// Live GTAO tunables captured at chain-build time and rewritten into the GPU UBO each record.
    settings: GtaoSettings,
    /// Process-wide cached pipelines, bind-group layouts, sampler, and params UBO.
    pipelines: &'static GtaoPipelineCache,
}

impl GtaoPass {
    /// Creates a new GTAO pass instance capturing the current settings snapshot.
    pub fn new(resources: GtaoGraphResources, settings: GtaoSettings) -> Self {
        Self {
            resources,
            settings,
            pipelines: gtao_pipelines(),
        }
    }
}

/// Process-wide pipeline cache shared by every GTAO pass instance.
fn gtao_pipelines() -> &'static GtaoPipelineCache {
    static CACHE: OnceLock<GtaoPipelineCache> = OnceLock::new();
    CACHE.get_or_init(GtaoPipelineCache::default)
}

impl RasterPass for GtaoPass {
    fn name(&self) -> &str {
        "Gtao"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.input,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.import_texture(
            self.resources.depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.import_buffer(
            self.resources.frame_uniforms,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        );
        let mut r = b.raster();
        r.color(
            self.resources.output,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            Option::<TextureHandle>::None,
        );
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let stereo = ctx
            .frame
            .as_ref()
            .is_some_and(|frame| frame.view.multiview_stereo);
        if stereo {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::gtao");
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(input_tex) = graph_resources.transient_texture(self.resources.input) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!(
                    "{} (missing transient input {:?})",
                    self.name(),
                    self.resources.input
                ),
            });
        };

        // Bind the per-view frame-uniforms buffer when the per-view plan is populated (the
        // default render path). The imported `frame_uniforms` handle resolves to the shared
        // `FrameResourceManager` buffer, which is only written by the shared-frame path —
        // binding it in per-view mode would leave the shader reading zeros and producing NaN
        // through `linearize_depth` / `view_pos_from_uv`, which `saturate` collapses to 0.
        let per_view_buffer = ctx
            .blackboard
            .get::<PerViewFramePlanSlot>()
            .map(|plan| plan.frame_uniform_buffer.clone());
        let frame_uniform_buffer = match per_view_buffer {
            Some(buf) => buf,
            None => match graph_resources.imported_buffer(self.resources.frame_uniforms) {
                Some(resolved) => resolved.buffer.clone(),
                None => {
                    return Err(RenderPassError::MissingFrameParams {
                        pass: format!("{} (frame_uniforms not resolved)", self.name()),
                    });
                }
            },
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let target_format = output_attachment_format(self.resources.output, graph_resources);

        let settings = ctx
            .blackboard
            .get::<GtaoSettingsSlot>()
            .map(|slot| slot.0)
            .unwrap_or(self.settings);
        let params = GtaoParamsGpu {
            radius_world: settings.radius_meters.max(0.0),
            max_pixel_radius: settings.max_pixel_radius.max(1.0),
            intensity: settings.intensity.max(0.0),
            step_count: settings.step_count.max(1),
            falloff_range: settings.falloff_range.clamp(0.05, 1.0),
            albedo_multibounce: settings.albedo_multibounce.clamp(0.0, 0.99),
            align_pad_tail: [0.0; 2],
        };
        let params_buffer = self.pipelines.params_buffer(ctx.device);
        // Route through the deferred `upload_batch` sink rather than calling
        // `ctx.queue.write_buffer` directly so every queue write goes through the driver
        // thread (matches the project's single-producer invariant for wgpu's queue).
        ctx.upload_batch
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let pipeline = self
            .pipelines
            .pipeline(ctx.device, target_format, multiview_stereo);
        let bind_group = self.pipelines.bind_group(
            ctx.device,
            multiview_stereo,
            &input_tex.texture,
            frame.view.depth_texture,
            &frame_uniform_buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Resolves the wgpu format the GTAO color attachment is bound to this frame.
fn output_attachment_format(
    output: TextureHandle,
    graph_resources: &crate::render_graph::context::GraphResolvedResources,
) -> wgpu::TextureFormat {
    graph_resources
        .transient_texture(output)
        .map(|t| t.texture.format())
        .unwrap_or(wgpu::TextureFormat::Rgba16Float)
}

/// Effect descriptor that contributes a [`GtaoPass`] to the post-processing chain.
pub struct GtaoEffect {
    /// Snapshot of the GTAO settings used when building the pass for this frame.
    pub settings: GtaoSettings,
    /// Imported depth texture handle (declared as a sampled read for scheduling).
    pub depth: ImportedTextureHandle,
    /// Imported frame-uniforms buffer handle (fallback / scheduling; actual bind sources from
    /// [`PerViewFramePlanSlot`] at record time).
    pub frame_uniforms: ImportedBufferHandle,
}

impl PostProcessEffect for GtaoEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::Gtao
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.gtao.enabled
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let pass_id = builder.add_raster_pass(Box::new(GtaoPass::new(
            GtaoGraphResources {
                input,
                output,
                depth: self.depth,
                frame_uniforms: self.frame_uniforms,
            },
            self.settings,
        )));
        EffectPasses::single(pass_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::pass::PassBuilder;
    use crate::render_graph::resources::{
        AccessKind, BufferImportSource, ImportSource, ImportedBufferDecl, ImportedTextureDecl,
        TextureAccess, TransientArrayLayers, TransientExtent, TransientSampleCount,
        TransientTextureDesc, TransientTextureFormat,
    };
    use crate::render_graph::GraphBuilder;

    fn fake_graph() -> (
        GraphBuilder,
        TextureHandle,
        TextureHandle,
        ImportedTextureHandle,
        ImportedBufferHandle,
    ) {
        let mut builder = GraphBuilder::new();
        let desc = || TransientTextureDesc {
            label: "pp_hdr",
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
        let input = builder.create_texture(desc());
        let output = builder.create_texture(desc());
        let depth = builder.import_texture(ImportedTextureDecl {
            label: "frame_depth",
            source: ImportSource::FrameTarget(
                crate::render_graph::resources::FrameTargetRole::DepthAttachment,
            ),
            initial_access: TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
            final_access: TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        });
        let frame_uniforms = builder.import_buffer(ImportedBufferDecl {
            label: "frame_uniforms",
            source: BufferImportSource::BackendFrameResource(
                crate::render_graph::resources::BackendFrameBufferKind::FrameUniforms,
            ),
            initial_access: BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
            final_access: BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        });
        (builder, input, output, depth, frame_uniforms)
    }

    #[test]
    fn setup_declares_sampled_input_and_raster_output_and_uniform_buffer() {
        let (_builder, input, output, depth, frame_uniforms) = fake_graph();
        let mut pass = GtaoPass::new(
            GtaoGraphResources {
                input,
                output,
                depth,
                frame_uniforms,
            },
            GtaoSettings::default(),
        );
        let mut b = PassBuilder::new("Gtao");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        let sampled_reads = setup
            .accesses
            .iter()
            .filter(|a| {
                matches!(
                    &a.access,
                    AccessKind::Texture(TextureAccess::Sampled {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        ..
                    })
                )
            })
            .count();
        assert!(
            sampled_reads >= 2,
            "expected sampled reads for both HDR input and depth (got {sampled_reads})"
        );
        assert!(
            setup.accesses.iter().any(|a| matches!(
                &a.access,
                AccessKind::Buffer(BufferAccess::Uniform {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )),
            "expected uniform-buffer read of frame_uniforms"
        );
        assert_eq!(setup.color_attachments.len(), 1);
    }

    #[test]
    fn gtao_effect_id_label() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
            frame_uniforms: ImportedBufferHandle(0),
        };
        assert_eq!(e.id(), PostProcessEffectId::Gtao);
        assert_eq!(e.id().label(), "GTAO");
    }

    #[test]
    fn gtao_effect_is_gated_by_master_and_per_effect_enable() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
            frame_uniforms: ImportedBufferHandle(0),
        };
        let mut s = PostProcessingSettings::default();
        assert!(!e.is_enabled(&s), "defaults: master off, gtao off");
        s.enabled = true;
        assert!(!e.is_enabled(&s), "master on but gtao off");
        s.gtao.enabled = true;
        assert!(e.is_enabled(&s), "master on + gtao on");
        s.enabled = false;
        assert!(!e.is_enabled(&s), "master off disables even if gtao on");
    }
}
