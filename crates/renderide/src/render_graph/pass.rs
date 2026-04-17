//! [`RenderPass`] trait plus setup-time pass builders.

use std::num::NonZeroU32;

use super::compiled::{DepthAttachmentTemplate, RenderPassTemplate};
use super::context::{GraphRasterPassContext, RenderPassContext};
use super::error::{RenderPassError, SetupError};
use super::resources::{
    BufferAccess, BufferHandle, BufferResourceHandle, ImportedBufferHandle, ImportedTextureHandle,
    ResourceAccess, TextureAccess, TextureAttachmentResolve, TextureAttachmentTarget,
    TextureHandle, TextureResourceHandle,
};

/// Whether a render pass runs once per frame or once per view in a multi-view tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassPhase {
    /// Runs once per frame regardless of view count (e.g. mesh deform).
    FrameGlobal,
    /// Runs once per view (e.g. clustered light compute, forward raster, Hi-Z build).
    PerView,
}

/// Command domain declared by a pass during setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// Raster pass with at least one color or depth attachment declaration.
    Raster,
    /// Compute pass.
    Compute,
    /// Copy-only pass.
    Copy,
    /// Callback/side-effect pass without graph-visible GPU resources.
    Callback,
}

/// Group execution scope. Frame-global groups run once per tick; per-view groups run once per
/// [`super::FrameView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupScope {
    /// Runs once per tick.
    FrameGlobal,
    /// Runs once for each frame view.
    PerView,
}

impl From<PassPhase> for GroupScope {
    fn from(value: PassPhase) -> Self {
        match value {
            PassPhase::FrameGlobal => Self::FrameGlobal,
            PassPhase::PerView => Self::PerView,
        }
    }
}

/// Compiled setup data for one pass.
#[derive(Clone, Debug)]
pub(crate) struct PassSetup {
    pub(crate) kind: PassKind,
    pub(crate) accesses: Vec<ResourceAccess>,
    pub(crate) color_attachments: Vec<RasterColorAttachmentSetup>,
    pub(crate) depth_stencil_attachment: Option<RasterDepthAttachmentSetup>,
    pub(crate) multiview_mask: Option<NonZeroU32>,
    pub(crate) cull_exempt: bool,
}

impl PassSetup {
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

/// Setup-time color attachment declaration.
#[derive(Clone, Debug)]
pub(crate) struct RasterColorAttachmentSetup {
    pub(crate) target: TextureAttachmentTarget,
    pub(crate) load: wgpu::LoadOp<wgpu::Color>,
    pub(crate) store: wgpu::StoreOp,
    pub(crate) resolve_to: Option<TextureAttachmentResolve>,
}

/// Setup-time depth/stencil attachment declaration.
#[derive(Clone, Debug)]
pub(crate) struct RasterDepthAttachmentSetup {
    pub(crate) target: TextureAttachmentTarget,
    pub(crate) depth: wgpu::Operations<f32>,
    pub(crate) stencil: Option<wgpu::Operations<u32>>,
}

/// Setup-time builder used by a pass to declare resource access and command kind.
pub struct PassBuilder<'a> {
    name: &'a str,
    kind: PassKind,
    accesses: Vec<ResourceAccess>,
    color_attachments: Vec<RasterColorAttachmentSetup>,
    depth_stencil_attachment: Option<RasterDepthAttachmentSetup>,
    multiview_mask: Option<NonZeroU32>,
    cull_exempt: bool,
}

impl<'a> PassBuilder<'a> {
    pub(crate) fn new(name: &'a str) -> Self {
        Self {
            name,
            kind: PassKind::Callback,
            accesses: Vec::new(),
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            multiview_mask: None,
            cull_exempt: false,
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

    /// Keeps the pass even when it has no graph-visible export.
    pub fn cull_exempt(&mut self) {
        self.cull_exempt = true;
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
                access: super::resources::StorageAccess::ReadWrite,
                ..
            }
        );
        self.accesses
            .push(ResourceAccess::buffer(handle.into(), access, reads, true));
    }
}

/// Raster-pass setup helper. It records attachments and multiview state.
pub struct RasterPassBuilder<'b, 'a> {
    parent: &'b mut PassBuilder<'a>,
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

/// One node in the render graph.
pub trait RenderPass: Send {
    /// Stable name for logging and errors.
    fn name(&self) -> &str;

    /// Declares pass kind, resource accesses, and attachment intent.
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Records GPU commands for this pass into `ctx.encoder`.
    ///
    /// Runtime execution still routes through the existing command encoder while the graph-owned
    /// resource allocator is brought online; setup data is already the source of scheduling truth.
    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError>;

    /// Whether this raster pass expects the graph to open `wgpu::RenderPass` from setup data.
    ///
    /// Existing legacy raster passes return `false` until their encode helpers are ported.
    fn graph_managed_raster(&self) -> bool {
        false
    }

    /// Runtime multiview mask for a graph-owned raster pass.
    fn graph_raster_multiview_mask(
        &self,
        _ctx: &GraphRasterPassContext<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        template.multiview_mask
    }

    /// Runtime stencil operations for a graph-owned raster pass.
    fn graph_raster_stencil_ops(
        &self,
        _ctx: &GraphRasterPassContext<'_, '_>,
        depth: &DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        depth.stencil
    }

    /// Records commands into a graph-owned raster pass.
    fn execute_graph_raster(
        &mut self,
        _ctx: &mut GraphRasterPassContext<'_, '_>,
        _rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        Ok(())
    }

    /// Scheduling phase for multi-view execution. Defaults to per-view.
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }
}
