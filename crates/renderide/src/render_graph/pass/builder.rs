//! Setup-time pass builder: resource access declarations and attachment intent.
//!
//! [`PassBuilder`] is the single entry-point a pass's `setup` method uses to declare:
//! - What kind of GPU work it performs ([`PassKind`] via [`PassBuilder::raster`] or
//!   [`PassBuilder::compute`] / [`PassBuilder::encoder`]).
//! - Scheduler policy flags such as never-cull, never-merge, never-parallel, and
//!   async-compute-capable metadata.
//! - Which resources it reads or writes (textures/buffers, transient or imported).
//! - For raster passes: color and depth-stencil attachments with their load/store ops.
//! - Whether the pass is exempt from dead-pass culling ([`PassBuilder::cull_exempt`]).

use std::num::NonZeroU32;

use super::attachments::{declare_color_attachment, declare_depth_attachment};
use super::node::{PassKind, PassMergeHint, PassWorkloadFlags};
use super::params::{
    BlackboardAccessDecl, BlackboardAccessKind, GraphPassParameters, PassParameterSchema,
};
use super::setup::{PassSetup, RasterColorAttachmentSetup, RasterDepthAttachmentSetup};
use crate::render_graph::error::SetupError;
#[cfg(test)]
use crate::render_graph::resources::StorageAccess;
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, BufferResourceHandle, ImportedBufferHandle, ImportedTextureHandle,
    ResourceAccess, SubresourceHandle, TextureAccess, TextureAttachmentTarget, TextureHandle,
    TextureResourceHandle,
};

/// Setup-time builder used by a pass to declare resource access and command kind.
pub struct PassBuilder<'a> {
    pub(crate) _name: &'a str,
    pub(crate) kind: PassKind,
    pub(crate) flags: PassWorkloadFlags,
    pub(crate) accesses: Vec<ResourceAccess>,
    pub(crate) color_attachments: Vec<RasterColorAttachmentSetup>,
    pub(crate) depth_stencil_attachment: Option<RasterDepthAttachmentSetup>,
    pub(crate) blackboard_accesses: Vec<BlackboardAccessDecl>,
    pub(crate) parameter_schema: Option<PassParameterSchema>,
    pub(crate) multiview_mask: Option<NonZeroU32>,
    pub(crate) cull_exempt: bool,
    pub(crate) requires_async_compute: bool,
    pub(crate) merge_hint: PassMergeHint,
}

impl<'a> PassBuilder<'a> {
    /// Creates a builder starting in [`PassKind::Compute`] kind.
    pub(crate) fn new(name: &'a str) -> Self {
        Self {
            _name: name,
            kind: PassKind::Compute,
            flags: PassWorkloadFlags::EMPTY,
            accesses: Vec::new(),
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            blackboard_accesses: Vec::new(),
            parameter_schema: Some(PassParameterSchema::new(name)),
            multiview_mask: None,
            cull_exempt: false,
            requires_async_compute: false,
            merge_hint: PassMergeHint::default(),
        }
    }

    pub(crate) fn finish(self) -> Result<PassSetup, SetupError> {
        let workload_flags = self.flags.with(PassWorkloadFlags::for_kind(self.kind));
        PassSetup {
            kind: self.kind,
            workload_flags,
            accesses: self.accesses,
            color_attachments: self.color_attachments,
            depth_stencil_attachment: self.depth_stencil_attachment,
            blackboard_accesses: self.blackboard_accesses,
            parameter_schema: self.parameter_schema,
            multiview_mask: self.multiview_mask,
            cull_exempt: self.cull_exempt,
            requires_async_compute: self.requires_async_compute,
            merge_hint: self.merge_hint,
        }
        .validate()
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

    /// Declares this pass as encoder-driven mixed work.
    pub fn encoder(&mut self) {
        self.kind = PassKind::Encoder;
    }

    /// Declares a graph-facing parameter schema for this pass.
    pub fn parameter_schema(&mut self, schema: PassParameterSchema) {
        self.parameter_schema = Some(schema);
    }

    /// Declares resources and metadata from a typed parameter struct.
    pub fn parameters<P: GraphPassParameters>(&mut self, params: &P) -> Result<(), SetupError> {
        self.parameter_schema(params.schema());
        params.declare(self)
    }

    /// Declares a required blackboard slot read.
    pub fn read_blackboard<S: crate::render_graph::blackboard::BlackboardSlot>(&mut self) {
        self.blackboard_accesses
            .push(BlackboardAccessDecl::new::<S>(
                BlackboardAccessKind::RequiredRead,
            ));
    }

    /// Declares an optional blackboard slot read.
    pub fn read_optional_blackboard<S: crate::render_graph::blackboard::BlackboardSlot>(&mut self) {
        self.blackboard_accesses
            .push(BlackboardAccessDecl::new::<S>(
                BlackboardAccessKind::OptionalRead,
            ));
    }

    /// Declares a blackboard slot write.
    pub fn write_blackboard<S: crate::render_graph::blackboard::BlackboardSlot>(&mut self) {
        self.blackboard_accesses
            .push(BlackboardAccessDecl::new::<S>(BlackboardAccessKind::Write));
    }

    /// Keeps the pass even when it has no graph-visible export.
    pub fn cull_exempt(&mut self) {
        self.cull_exempt = true;
        self.flags.insert(PassWorkloadFlags::NEVER_CULL);
    }

    /// Prevents the scheduler from folding this pass into a render-pass merge group.
    pub fn never_merge(&mut self) {
        self.flags.insert(PassWorkloadFlags::NEVER_MERGE);
    }

    /// Prevents future scheduler backends from recording this pass off the main path.
    pub fn never_parallel(&mut self) {
        self.flags.insert(PassWorkloadFlags::NEVER_PARALLEL);
    }

    /// Marks this compute pass as eligible for async-compute scheduling on a future backend.
    ///
    /// This flag is metadata in the current wgpu executor. It does not move the pass to a
    /// separate queue.
    pub fn async_compute_capable(&mut self) {
        self.flags.insert(PassWorkloadFlags::ASYNC_COMPUTE_CAPABLE);
    }

    /// Requires async-compute execution.
    ///
    /// Scheduler v1 does not expose a multi-queue backend, so this is rejected during setup
    /// validation instead of silently falling back to the graphics queue.
    #[cfg(test)]
    pub fn require_async_compute(&mut self) {
        self.requires_async_compute = true;
        self.flags.insert(PassWorkloadFlags::ASYNC_COMPUTE_CAPABLE);
    }

    /// Sets the backend merge hint for this pass. See [`PassMergeHint`] for details.
    ///
    /// Scheduler v1 uses this metadata when grouping adjacent raster passes that target the same
    /// attachments. The wgpu executor materializes compatible groups into one render pass.
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

    /// Declares a read from a transient texture subresource.
    pub fn read_texture_subresource(&mut self, handle: SubresourceHandle, access: TextureAccess) {
        self.accesses.push(ResourceAccess::texture_subresource(
            handle, access, true, false,
        ));
    }

    /// Declares a write to a transient texture subresource.
    pub fn write_texture_subresource(&mut self, handle: SubresourceHandle, access: TextureAccess) {
        let reads = access.reads();
        self.accesses.push(ResourceAccess::texture_subresource(
            handle, access, reads, true,
        ));
    }

    /// Declares a transient buffer read.
    #[cfg(test)]
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
    #[cfg(test)]
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
        #[cfg(test)]
        let reads = matches!(
            access,
            BufferAccess::Storage {
                access: StorageAccess::ReadWrite,
                ..
            }
        );
        #[cfg(not(test))]
        let reads = false;
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
        declare_color_attachment(
            self.parent,
            TextureAttachmentTarget::Resource(handle.into()),
            ops,
            resolve_to.map(Into::into),
        );
    }

    /// Declares a color attachment that switches between single-sample and MSAA targets per frame.
    pub fn frame_sampled_color(
        &mut self,
        single_sample: impl Into<TextureResourceHandle>,
        multisampled: impl Into<TextureResourceHandle>,
        ops: wgpu::Operations<wgpu::Color>,
        resolve_to: Option<impl Into<TextureResourceHandle>>,
    ) {
        declare_color_attachment(
            self.parent,
            TextureAttachmentTarget::FrameSampled {
                single_sample: single_sample.into(),
                multisampled: multisampled.into(),
            },
            ops,
            resolve_to.map(Into::into),
        );
    }

    /// Declares a depth/stencil attachment.
    #[cfg(test)]
    pub fn depth(
        &mut self,
        handle: impl Into<TextureResourceHandle>,
        depth: wgpu::Operations<f32>,
        stencil: Option<wgpu::Operations<u32>>,
    ) {
        declare_depth_attachment(
            self.parent,
            TextureAttachmentTarget::Resource(handle.into()),
            depth,
            stencil,
        );
    }

    /// Declares a depth/stencil attachment that switches between single-sample and MSAA targets per frame.
    pub fn frame_sampled_depth(
        &mut self,
        single_sample: impl Into<TextureResourceHandle>,
        multisampled: impl Into<TextureResourceHandle>,
        depth: wgpu::Operations<f32>,
        stencil: Option<wgpu::Operations<u32>>,
    ) {
        declare_depth_attachment(
            self.parent,
            TextureAttachmentTarget::FrameSampled {
                single_sample: single_sample.into(),
                multisampled: multisampled.into(),
            },
            depth,
            stencil,
        );
    }

    /// Declares a multiview render-pass mask.
    #[cfg(test)]
    pub fn multiview(&mut self, mask: NonZeroU32) {
        self.parent.multiview_mask = Some(mask);
    }
}
