//! Compiled setup data for a single pass node.
//!
//! [`PassSetup`] is produced by [`super::builder::PassBuilder::finish`] and consumed by the
//! graph compiler for edge synthesis, lifetime analysis, and template compilation.

use std::num::NonZeroU32;

use super::node::{PassKind, PassMergeHint};
use crate::render_graph::error::SetupError;
use crate::render_graph::resources::{
    ResourceAccess, TextureAttachmentResolve, TextureAttachmentTarget,
};

/// Setup-time color attachment declaration.
#[derive(Clone, Debug)]
pub struct RasterColorAttachmentSetup {
    /// Attachment target (single-sample, MSAA, or frame-sampled selection).
    pub(crate) target: TextureAttachmentTarget,
    /// Load operation for the attachment.
    pub(crate) load: wgpu::LoadOp<wgpu::Color>,
    /// Store operation for the attachment.
    pub(crate) store: wgpu::StoreOp,
    /// Optional MSAA resolve target.
    pub(crate) resolve_to: Option<TextureAttachmentResolve>,
}

/// Setup-time depth/stencil attachment declaration.
#[derive(Clone, Debug)]
pub struct RasterDepthAttachmentSetup {
    /// Attachment target.
    pub(crate) target: TextureAttachmentTarget,
    /// Depth operations.
    pub(crate) depth: wgpu::Operations<f32>,
    /// Optional stencil operations.
    pub(crate) stencil: Option<wgpu::Operations<u32>>,
}

/// Compiled setup data for one pass.
#[derive(Clone, Debug)]
pub struct PassSetup {
    /// Command domain (raster, compute, copy, or callback).
    pub(crate) kind: PassKind,
    /// Declared resource accesses for dependency synthesis and lifetime analysis.
    pub(crate) accesses: Vec<ResourceAccess>,
    /// Color attachment declarations for graph-managed raster passes.
    pub(crate) color_attachments: Vec<RasterColorAttachmentSetup>,
    /// Depth/stencil attachment declaration for graph-managed raster passes.
    pub(crate) depth_stencil_attachment: Option<RasterDepthAttachmentSetup>,
    /// Optional multiview mask for graph-managed raster passes.
    pub(crate) multiview_mask: Option<NonZeroU32>,
    /// When `true`, the pass is retained even when it has no import-writing successors.
    pub(crate) cull_exempt: bool,
    /// Backend merge hint; see [`PassMergeHint`].
    pub(crate) merge_hint: PassMergeHint,
}

impl PassSetup {
    /// Validates the combination of kind and declared attachments.
    pub(crate) fn validate(self) -> Result<Self, SetupError> {
        let has_attachment = !self.color_attachments.is_empty()
            || self.depth_stencil_attachment.is_some()
            || self.accesses.iter().any(ResourceAccess::is_attachment);
        match self.kind {
            PassKind::Raster if !has_attachment => Err(SetupError::RasterWithoutAttachments),
            PassKind::Compute | PassKind::Copy if has_attachment => {
                Err(SetupError::NonRasterPassHasAttachment)
            }
            PassKind::Callback if !self.accesses.is_empty() => {
                Err(SetupError::CallbackPassHasAccesses)
            }
            _ => Ok(self),
        }
    }
}
