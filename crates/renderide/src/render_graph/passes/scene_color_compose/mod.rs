//! Samples HDR scene color into the per-view display target (swapchain / XR / offscreen RT).
//!
//! This pass is the integration point for a future post-processing stack (exposure, bloom, tonemap,
//! color grading): insert additional passes before this node, or extend the compose shader.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::OnceLock;

use pipeline::SceneColorComposePipelineCache;

use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{ImportedTextureHandle, TextureAccess, TextureHandle};

/// Graph handles for [`SceneColorComposePass`].
#[derive(Clone, Copy, Debug)]
pub struct SceneColorComposeGraphResources {
    /// Resolved single-sample HDR scene color ([`crate::render_graph::resources::TransientTextureFormat::SceneColorHdr`]).
    pub scene_color_hdr: TextureHandle,
    /// Imported frame color (output).
    pub frame_color: ImportedTextureHandle,
}

/// Fullscreen blit from HDR scene color to the displayable color target.
pub struct SceneColorComposePass {
    resources: SceneColorComposeGraphResources,
    pipelines: &'static SceneColorComposePipelineCache,
}

impl SceneColorComposePass {
    /// Creates a scene-color compose pass instance.
    pub fn new(resources: SceneColorComposeGraphResources) -> Self {
        Self {
            resources,
            pipelines: compose_pipelines(),
        }
    }
}

fn compose_pipelines() -> &'static SceneColorComposePipelineCache {
    static CACHE: OnceLock<SceneColorComposePipelineCache> = OnceLock::new();
    CACHE.get_or_init(SceneColorComposePipelineCache::default)
}

impl RasterPass for SceneColorComposePass {
    fn name(&self) -> &str {
        "SceneColorCompose"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.scene_color_hdr,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        {
            let mut r = b.raster();
            r.color(
                self.resources.frame_color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
        }
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
        profiling::scope!("scene_color_compose::record");
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
        let Some(tex) = graph_resources.transient_texture(self.resources.scene_color_hdr) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!(
                    "{} (missing transient scene_color_hdr {:?})",
                    self.name(),
                    self.resources.scene_color_hdr
                ),
            });
        };
        let pipeline = self.pipelines.pipeline(
            ctx.device,
            frame.view.surface_format,
            frame.view.multiview_stereo,
        );
        let bind_group =
            self.pipelines
                .bind_group(ctx.device, &tex.texture, frame.view.multiview_stereo);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    use crate::render_graph::pass::PassBuilder;

    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::resources::{
        AccessKind, FrameTargetRole, ImportSource, ImportedTextureDecl, TextureAccess,
        TransientArrayLayers, TransientExtent, TransientSampleCount, TransientTextureDesc,
        TransientTextureFormat,
    };
    use crate::render_graph::GraphBuilder;

    #[test]
    fn setup_declares_sampled_hdr_and_frame_color_raster() {
        let mut builder = GraphBuilder::new();
        let hdr = builder.create_texture(TransientTextureDesc {
            label: "scene_color_hdr",
            format: TransientTextureFormat::SceneColorHdr,
            extent: TransientExtent::Custom {
                width: 4,
                height: 4,
            },
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Fixed(1),
            base_usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            alias: true,
        });
        let frame_color = builder.import_texture(ImportedTextureDecl {
            label: "frame_color",
            source: ImportSource::FrameTarget(FrameTargetRole::ColorAttachment),
            initial_access: TextureAccess::ColorAttachment {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
                resolve_to: None,
            },
            final_access: TextureAccess::Present,
        });
        let mut pass = SceneColorComposePass::new(SceneColorComposeGraphResources {
            scene_color_hdr: hdr,
            frame_color,
        });
        let mut b = PassBuilder::new("SceneColorCompose");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        assert!(
            setup.accesses.iter().any(|a| {
                matches!(
                    &a.access,
                    AccessKind::Texture(TextureAccess::Sampled {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        ..
                    })
                )
            }),
            "expected sampled HDR read"
        );
        assert_eq!(setup.color_attachments.len(), 1);
    }
}
