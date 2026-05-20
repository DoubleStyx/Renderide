//! Compiles setup declarations into retained pass metadata.

use super::super::compiled::CompiledPassInfo;
use super::super::pass::{
    ColorAttachmentTemplate, DepthAttachmentTemplate, PassSetup, RenderPassTemplate,
};
use super::decl::SetupEntry;

/// Compiles setup metadata for the retained pass order.
pub(super) fn compile_pass_info(setups: &[SetupEntry], ordered: &[usize]) -> Vec<CompiledPassInfo> {
    ordered
        .iter()
        .copied()
        .map(|idx| {
            let setup = &setups[idx];
            let raster_template = compile_raster_template(&setup.setup);
            CompiledPassInfo {
                name: setup.name.clone(),
                #[cfg(test)]
                kind: setup.setup.kind,
                workload_flags: setup.setup.workload_flags,
                accesses: setup.setup.accesses.clone(),
                blackboard_accesses: setup.setup.blackboard_accesses.clone(),
                parameter_schema: setup.setup.parameter_schema.clone(),
                #[cfg(test)]
                multiview_mask: setup.setup.multiview_mask,
                raster_template,
                merge_hint: setup.setup.merge_hint,
            }
        })
        .collect()
}

/// Compiles raster attachment metadata for a pass setup.
fn compile_raster_template(setup: &PassSetup) -> Option<RenderPassTemplate> {
    let color_attachments: Vec<ColorAttachmentTemplate> = setup
        .color_attachments
        .iter()
        .map(|color| ColorAttachmentTemplate {
            target: color.target,
            load: color.load,
            store: color.store,
            resolve_to: color.resolve_to,
        })
        .collect();
    let depth_stencil_attachment =
        setup
            .depth_stencil_attachment
            .as_ref()
            .map(|depth| DepthAttachmentTemplate {
                target: depth.target,
                depth: depth.depth,
                stencil: depth.stencil,
            });
    (!color_attachments.is_empty() || depth_stencil_attachment.is_some()).then_some(
        RenderPassTemplate {
            color_attachments,
            depth_stencil_attachment,
            multiview_mask: setup.multiview_mask,
        },
    )
}
