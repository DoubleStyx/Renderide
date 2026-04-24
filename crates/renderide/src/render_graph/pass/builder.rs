//! Setup-time pass builder: resource access declarations and attachment intent.
//!
//! [`PassBuilder`] is the single entry-point a pass's `setup` method uses to declare:
//! - What kind of GPU work it performs ([`PassKind`] via [`PassBuilder::raster`],
//!   [`PassBuilder::compute`], [`PassBuilder::copy`], or [`PassBuilder::callback`]).
//! - Which resources it reads or writes (textures/buffers, transient or imported).
//! - For raster passes: color and depth-stencil attachments with their load/store ops.
//! - Whether the pass is exempt from dead-pass culling ([`PassBuilder::cull_exempt`]).

use std::num::NonZeroU32;

use super::node::{PassKind, PassMergeHint};
use super::setup::{PassSetup, RasterColorAttachmentSetup, RasterDepthAttachmentSetup};
use crate::render_graph::error::SetupError;
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, BufferResourceHandle, ImportedBufferHandle, ImportedTextureHandle,
    ResourceAccess, StorageAccess, TextureAccess, TextureAttachmentResolve,
    TextureAttachmentTarget, TextureHandle, TextureResourceHandle,
};

/// Setup-time builder used by a pass to declare resource access and command kind.
pub struct PassBuilder<'a> {
    pub(crate) name: &'a str,
    pub(crate) kind: PassKind,
    pub(crate) accesses: Vec<ResourceAccess>,
    pub(crate) color_attachments: Vec<RasterColorAttachmentSetup>,
    pub(crate) depth_stencil_attachment: Option<RasterDepthAttachmentSetup>,
    pub(crate) multiview_mask: Option<NonZeroU32>,
    pub(crate) cull_exempt: bool,
    pub(crate) merge_hint: PassMergeHint,
}

impl<'a> PassBuilder<'a> {
    /// Creates a builder starting in [`PassKind::Callback`] kind (no-GPU default).
    pub(crate) fn new(name: &'a str) -> Self {
        Self {
            name,
            kind: PassKind::Callback,
            accesses: Vec::new(),
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            multiview_mask: None,
            cull_exempt: false,
            merge_hint: PassMergeHint::default(),
        }
    }

    pub(crate) fn finish(self) -> Result<PassSetup, SetupError> {
        PassSetup {
            kind: self.kind,
            accesses: self.accesses,
            color_attachments: self.color_attachments,
            depth_stencil_attachment: self.depth_stencil_attachment,
            multiview_mask: self.multiview_mask,
            cull_exempt: self.cull_exempt,
            merge_hint: self.merge_hint,
        }
        .validate()
    }

    /// Name of the pass currently declaring setup.
    pub fn pass_name(&self) -> &str {
        self.name
    }

    /// Declares this pass as raster and returns the attachment builder.
    pub fn raster(&mut self) -> RasterPassBuilder<'_, 'a> {
        self.kind = PassKind::Raster;
        RasterPassBuilder { parent: self }
    }

    /// Declares this pass as compute.
    pub fn compute(&mut self) {
        self.kind = PassKind::Compute;
    }

    /// Declares this pass as copy-only.
    pub fn copy(&mut self) {
        self.kind = PassKind::Copy;
    }

    /// Declares this pass as a callback (CPU-only, no encoder or GPU resources).
    pub fn callback(&mut self) {
        self.kind = PassKind::Callback;
    }

    /// Keeps the pass even when it has no graph-visible export.
    pub fn cull_exempt(&mut self) {
        self.cull_exempt = true;
    }

    /// Sets the backend merge hint for this pass. See [`PassMergeHint`] for details.
    ///
    /// The current wgpu executor ignores the hint; it exists so passes can annotate their
    /// attachment-reuse intent today, ready to be consumed by a future subpass-aware backend.
    pub fn merge_hint(&mut self, hint: PassMergeHint) {
        self.merge_hint = hint;
    }

    /// Declares a transient texture read.
    pub fn read_texture(&mut self, handle: TextureHandle, access: TextureAccess) {
        self.read_texture_resource(handle, access);
    }

    /// Declares a transient texture write.
    pub fn write_texture(&mut self, handle: TextureHandle, access: TextureAccess) {
        self.write_texture_resource(handle, access);
    }

    /// Declares an imported texture access. Direction is inferred from the access type.
    pub fn import_texture(&mut self, handle: ImportedTextureHandle, access: TextureAccess) {
        let reads = access.reads();
        let writes = access.writes();
        self.accesses.push(ResourceAccess::texture(
            TextureResourceHandle::Imported(handle),
            access,
            reads,
            writes,
        ));
    }

    /// Declares a transient buffer read.
    pub fn read_buffer(&mut self, handle: BufferHandle, access: BufferAccess) {
        self.read_buffer_resource(handle, access);
    }

    /// Declares a transient buffer write.
    pub fn write_buffer(&mut self, handle: BufferHandle, access: BufferAccess) {
        self.write_buffer_resource(handle, access);
    }

    /// Declares an imported buffer access. Direction is inferred from the access type.
    pub fn import_buffer(&mut self, handle: ImportedBufferHandle, access: BufferAccess) {
        let reads = access.reads();
        let writes = access.writes();
        self.accesses.push(ResourceAccess::buffer(
            BufferResourceHandle::Imported(handle),
            access,
            reads,
            writes,
        ));
    }

    /// Declares a texture read for either transient or imported handles.
    pub fn read_texture_resource(
        &mut self,
        handle: impl Into<TextureResourceHandle>,
        access: TextureAccess,
    ) {
        self.accesses
            .push(ResourceAccess::texture(handle.into(), access, true, false));
    }

    /// Declares a texture write for either transient or imported handles.
    pub fn write_texture_resource(
        &mut self,
        handle: impl Into<TextureResourceHandle>,
        access: TextureAccess,
    ) {
        let reads = access.reads();
        self.accesses
            .push(ResourceAccess::texture(handle.into(), access, reads, true));
    }

    /// Declares a buffer read for either transient or imported handles.
    pub fn read_buffer_resource(
        &mut self,
        handle: impl Into<BufferResourceHandle>,
        access: BufferAccess,
    ) {
        self.accesses
            .push(ResourceAccess::buffer(handle.into(), access, true, false));
    }

    /// Declares a buffer write for either transient or imported handles.
    pub fn write_buffer_resource(
        &mut self,
        handle: impl Into<BufferResourceHandle>,
        access: BufferAccess,
    ) {
        let reads = matches!(
            access,
            BufferAccess::Storage {
                access: StorageAccess::ReadWrite,
                ..
            }
        );
        self.accesses
            .push(ResourceAccess::buffer(handle.into(), access, reads, true));
    }
}

/// Raster-pass setup helper that records attachments and multiview state.
pub struct RasterPassBuilder<'b, 'a> {
    pub(crate) parent: &'b mut PassBuilder<'a>,
}

impl RasterPassBuilder<'_, '_> {
    /// Declares a color attachment.
    pub fn color(
        &mut self,
        handle: impl Into<TextureResourceHandle>,
        ops: wgpu::Operations<wgpu::Color>,
        resolve_to: Option<impl Into<TextureResourceHandle>>,
    ) {
        let handle = handle.into();
        let resolve_to = resolve_to.map(Into::into);
        self.parent.write_texture_resource(
            handle,
            TextureAccess::ColorAttachment {
                load: ops.load,
                store: ops.store,
                resolve_to,
            },
        );
        if let Some(target) = resolve_to {
            self.parent
                .write_texture_resource(target, TextureAccess::Present);
        }
        self.parent
            .color_attachments
            .push(RasterColorAttachmentSetup {
                target: TextureAttachmentTarget::Resource(handle),
                load: ops.load,
                store: ops.store,
                resolve_to: resolve_to.map(TextureAttachmentResolve::Always),
            });
    }

    /// Declares a color attachment that switches between single-sample and MSAA targets per frame.
    pub fn frame_sampled_color(
        &mut self,
        single_sample: impl Into<TextureResourceHandle>,
        multisampled: impl Into<TextureResourceHandle>,
        ops: wgpu::Operations<wgpu::Color>,
        resolve_to: Option<impl Into<TextureResourceHandle>>,
    ) {
        let single_sample = single_sample.into();
        let multisampled = multisampled.into();
        let resolve_to = resolve_to.map(Into::into);
        self.parent.write_texture_resource(
            single_sample,
            TextureAccess::ColorAttachment {
                load: ops.load,
                store: ops.store,
                resolve_to: None,
            },
        );
        self.parent.write_texture_resource(
            multisampled,
            TextureAccess::ColorAttachment {
                load: ops.load,
                store: ops.store,
                resolve_to,
            },
        );
        if let Some(target) = resolve_to {
            self.parent
                .write_texture_resource(target, TextureAccess::Present);
        }
        self.parent
            .color_attachments
            .push(RasterColorAttachmentSetup {
                target: TextureAttachmentTarget::FrameSampled {
                    single_sample,
                    multisampled,
                },
                load: ops.load,
                store: ops.store,
                resolve_to: resolve_to.map(TextureAttachmentResolve::FrameMultisampled),
            });
    }

    /// Declares a depth/stencil attachment.
    pub fn depth(
        &mut self,
        handle: impl Into<TextureResourceHandle>,
        depth: wgpu::Operations<f32>,
        stencil: Option<wgpu::Operations<u32>>,
    ) {
        let handle = handle.into();
        self.parent
            .write_texture_resource(handle, TextureAccess::DepthAttachment { depth, stencil });
        self.parent.depth_stencil_attachment = Some(RasterDepthAttachmentSetup {
            target: TextureAttachmentTarget::Resource(handle),
            depth,
            stencil,
        });
    }

    /// Declares a depth/stencil attachment that switches between single-sample and MSAA targets per frame.
    pub fn frame_sampled_depth(
        &mut self,
        single_sample: impl Into<TextureResourceHandle>,
        multisampled: impl Into<TextureResourceHandle>,
        depth: wgpu::Operations<f32>,
        stencil: Option<wgpu::Operations<u32>>,
    ) {
        let single_sample = single_sample.into();
        let multisampled = multisampled.into();
        self.parent.write_texture_resource(
            single_sample,
            TextureAccess::DepthAttachment { depth, stencil },
        );
        self.parent.write_texture_resource(
            multisampled,
            TextureAccess::DepthAttachment { depth, stencil },
        );
        self.parent.depth_stencil_attachment = Some(RasterDepthAttachmentSetup {
            target: TextureAttachmentTarget::FrameSampled {
                single_sample,
                multisampled,
            },
            depth,
            stencil,
        });
    }

    /// Declares a multiview render-pass mask.
    pub fn multiview(&mut self, mask: NonZeroU32) {
        self.parent.multiview_mask = Some(mask);
    }
}
