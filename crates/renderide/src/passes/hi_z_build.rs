//! Builds a CPU-readable hierarchical depth pyramid from the main depth attachment after the forward pass.

use crate::occlusion::HiZBuildInput;
use crate::render_graph::context::{ComputePassCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::params::{
    GraphPassParameters, PassParameterField, PassParameterSchema,
};
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{ImportedTextureHandle, StorageAccess, TextureAccess};

/// Compute + copy pass that samples main depth and stages mips for next-frame occlusion.
#[derive(Debug)]
pub struct HiZBuildPass {
    resources: HiZBuildGraphResources,
}

/// Graph resources used by [`HiZBuildPass`].
#[derive(Clone, Copy, Debug)]
pub struct HiZBuildGraphResources {
    /// Imported single-sample depth texture for this view.
    pub depth: ImportedTextureHandle,
    /// Imported ping-pong Hi-Z pyramid output.
    pub hi_z_current: ImportedTextureHandle,
}

impl HiZBuildPass {
    /// Creates a Hi-Z build pass instance.
    pub fn new(resources: HiZBuildGraphResources) -> Self {
        Self { resources }
    }
}

impl GraphPassParameters for HiZBuildGraphResources {
    fn schema(&self) -> PassParameterSchema {
        PassParameterSchema::new("HiZBuildGraphResources")
            .with_field(PassParameterField::new("depth", "sampled_input"))
            .with_field(PassParameterField::new("hi_z_current", "storage_output"))
    }

    fn declare(&self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.import_texture(
            self.depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        b.import_texture(
            self.hi_z_current,
            TextureAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        Ok(())
    }
}

impl ComputePass for HiZBuildPass {
    fn name(&self) -> &str {
        "HiZBuild"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.parameters(&self.resources)
    }

    fn should_record(&self, ctx: &ComputePassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        let frame = &*ctx.pass_frame;
        Ok(!frame.view.host_camera.suppress_occlusion_temporal
            && ctx.depth_view.is_some()
            && frame.view.depth_sample_view.is_some())
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("hi_z::encode_pyramid");
        if ctx.depth_view.is_none() {
            return Ok(());
        }
        let frame = &mut *ctx.pass_frame;
        let Some(depth_sample_view) = frame.view.depth_sample_view.as_ref() else {
            return Ok(());
        };
        let hi_z_history = ctx
            .graph_resources
            .imported_texture(self.resources.hi_z_current)
            .ok_or_else(|| RenderPassError::UnresolvedImportedTexture {
                pass: self.name().to_owned(),
                resource: "hi_z_current",
            })?
            .history
            .clone()
            .ok_or_else(|| RenderPassError::HistoryImportWithoutBacking {
                pass: self.name().to_owned(),
                resource: "hi_z_current",
            })?;
        let mode = frame.output_depth_mode();
        frame.shared.occlusion.encode_hi_z_build_pass(
            crate::occlusion::gpu::HiZBuildRecord {
                device: ctx.device,
                limits: ctx.gpu_limits,
                encoder: ctx.encoder,
            },
            frame.view.hi_z_slot.as_ref(),
            HiZBuildInput {
                depth_view: depth_sample_view,
                history_texture: &hi_z_history.texture,
                history_mip_views: &hi_z_history.mip_views,
                extent: frame.view.viewport_px,
                mode,
            },
            ctx.profiler,
        );
        Ok(())
    }

    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        // Hi-Z staging-buffer `map_async` now runs from a
        // [`wgpu::Queue::on_submitted_work_done`] callback installed in
        // [`crate::render_graph::compiled::exec::CompiledRenderGraph::execute_multi_view`],
        // so this post-submit hook is a no-op on the main thread.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::PassKind;
    use crate::render_graph::resources::{AccessKind, ResourceAccess, TextureResourceHandle};

    #[test]
    fn setup_declares_typed_depth_and_hiz_imports() {
        let mut pass = HiZBuildPass::new(HiZBuildGraphResources {
            depth: ImportedTextureHandle(1),
            hi_z_current: ImportedTextureHandle(2),
        });
        let mut builder = PassBuilder::new("HiZBuild");
        pass.setup(&mut builder).expect("setup");
        let setup = builder.finish().expect("finish");

        assert_eq!(setup.kind, PassKind::Compute);
        assert_eq!(setup.accesses.len(), 2);
        assert!(
            setup.accesses.iter().any(|access| {
                matches!(
                    access,
                    ResourceAccess {
                        resource: crate::render_graph::resources::ResourceHandle::Texture(
                            TextureResourceHandle::Imported(ImportedTextureHandle(1))
                        ),
                        access: AccessKind::Texture(TextureAccess::Sampled {
                            stages: wgpu::ShaderStages::COMPUTE,
                            ..
                        }),
                        ..
                    }
                )
            }),
            "expected sampled depth import"
        );
        assert!(
            setup.accesses.iter().any(|access| {
                matches!(
                    access,
                    ResourceAccess {
                        resource: crate::render_graph::resources::ResourceHandle::Texture(
                            TextureResourceHandle::Imported(ImportedTextureHandle(2))
                        ),
                        access: AccessKind::Texture(TextureAccess::Storage {
                            stages: wgpu::ShaderStages::COMPUTE,
                            access: StorageAccess::WriteOnly,
                            ..
                        }),
                        ..
                    }
                )
            }),
            "expected write-only Hi-Z storage import"
        );
        let schema = setup.parameter_schema.expect("parameter schema");
        assert_eq!(schema.name, "HiZBuildGraphResources");
    }
}
