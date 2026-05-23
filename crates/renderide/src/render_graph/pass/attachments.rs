//! Shared attachment-declaration helpers used by [`super::builder::RasterPassBuilder`].
//!
//! Both single-sample and frame-sampled (MSAA) attachments share the same logical shape: declare
//! the underlying color/depth access on the relevant resource(s), optionally declare a resolve
//! target, and push the matching template entry. Centralising that shape here keeps
//! [`super::builder::RasterPassBuilder`] methods to thin wrappers that pick a
//! [`TextureAttachmentTarget`] and [`TextureAttachmentResolve`] strategy.

use super::builder::PassBuilder;
use super::setup::{RasterColorAttachmentSetup, RasterDepthAttachmentSetup};
use super::template::{AttachmentOps, AttachmentStoreOp};
use crate::render_graph::resources::{
    TextureAccess, TextureAttachmentResolve, TextureAttachmentTarget, TextureResourceHandle,
};

/// Records a color attachment declaration on `parent`, handling both single-sample and
/// frame-sampled (MSAA) targets uniformly. Resolve declaration is shared across both paths.
pub(super) fn declare_color_attachment(
    parent: &mut PassBuilder<'_>,
    target: TextureAttachmentTarget,
    ops: wgpu::Operations<wgpu::Color>,
    resolve_to: Option<TextureResourceHandle>,
) {
    let resolve = match target {
        TextureAttachmentTarget::Resource(handle) => {
            parent.write_texture_resource(
                handle,
                TextureAccess::ColorAttachment {
                    load: ops.load,
                    store: ops.store,
                    resolve_to,
                },
            );
            resolve_to.map(TextureAttachmentResolve::Always)
        }
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            parent.write_texture_resource(
                single_sample,
                TextureAccess::ColorAttachment {
                    load: ops.load,
                    store: ops.store,
                    resolve_to: None,
                },
            );
            parent.write_texture_resource(
                multisampled,
                TextureAccess::ColorAttachment {
                    load: ops.load,
                    store: ops.store,
                    resolve_to,
                },
            );
            resolve_to.map(TextureAttachmentResolve::FrameMultisampled)
        }
    };
    if let Some(resolve_handle) = resolve_to {
        parent.write_texture_resource(resolve_handle, TextureAccess::Present);
    }
    parent.color_attachments.push(RasterColorAttachmentSetup {
        target,
        load: ops.load,
        store: AttachmentStoreOp::static_op(ops.store),
        resolve_to: resolve,
    });
}

/// Records a depth/stencil attachment declaration on `parent`, handling both single-sample and
/// frame-sampled (MSAA) targets uniformly.
pub(super) fn declare_depth_attachment(
    parent: &mut PassBuilder<'_>,
    target: TextureAttachmentTarget,
    depth: wgpu::Operations<f32>,
    stencil: Option<wgpu::Operations<u32>>,
) {
    match target {
        TextureAttachmentTarget::Resource(handle) => {
            parent
                .write_texture_resource(handle, TextureAccess::DepthAttachment { depth, stencil });
        }
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            parent.write_texture_resource(
                single_sample,
                TextureAccess::DepthAttachment { depth, stencil },
            );
            parent.write_texture_resource(
                multisampled,
                TextureAccess::DepthAttachment { depth, stencil },
            );
        }
    }
    parent.depth_stencil_attachment = Some(RasterDepthAttachmentSetup {
        target,
        depth: AttachmentOps::from_operations(depth),
        stencil,
    });
}
