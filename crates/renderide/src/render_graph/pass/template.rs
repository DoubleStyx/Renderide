//! Render-pass attachment templates produced during graph setup and consumed while recording.

use crate::render_graph::resources::{TextureAttachmentResolve, TextureAttachmentTarget};

/// Store operation for an attachment whose concrete target can depend on frame sample count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentStoreOp {
    /// One store operation applies for every concrete target.
    Static(wgpu::StoreOp),
    /// Store operations for a frame-sampled target.
    FrameSampled {
        /// Store operation when the frame sample count is one.
        single_sample: wgpu::StoreOp,
        /// Store operation when the frame sample count is greater than one.
        multisampled: wgpu::StoreOp,
    },
}

impl AttachmentStoreOp {
    /// Creates a store operation that applies to every target lane.
    pub const fn static_op(store: wgpu::StoreOp) -> Self {
        Self::Static(store)
    }

    /// Creates a frame-sampled store operation, collapsing identical lanes.
    pub fn frame_sampled(single_sample: wgpu::StoreOp, multisampled: wgpu::StoreOp) -> Self {
        if store_ops_match(single_sample, multisampled) {
            Self::Static(single_sample)
        } else {
            Self::FrameSampled {
                single_sample,
                multisampled,
            }
        }
    }

    /// Resolves the concrete store operation for the frame sample count.
    pub fn resolve(self, sample_count: u32) -> wgpu::StoreOp {
        match self {
            Self::Static(store) => store,
            Self::FrameSampled {
                single_sample,
                multisampled,
            } => {
                if sample_count > 1 {
                    multisampled
                } else {
                    single_sample
                }
            }
        }
    }

    /// Returns whether every possible concrete target stores.
    pub const fn stores_for_all_targets(self) -> bool {
        match self {
            Self::Static(wgpu::StoreOp::Store) => true,
            Self::Static(wgpu::StoreOp::Discard) => false,
            Self::FrameSampled {
                single_sample,
                multisampled,
            } => {
                matches!(single_sample, wgpu::StoreOp::Store)
                    && matches!(multisampled, wgpu::StoreOp::Store)
            }
        }
    }
}

/// Attachment operations with target-aware store behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttachmentOps<T> {
    /// Attachment load operation.
    pub load: wgpu::LoadOp<T>,
    /// Attachment store operation.
    pub store: AttachmentStoreOp,
}

impl<T> AttachmentOps<T> {
    /// Converts concrete wgpu operations into target-aware attachment operations.
    pub fn from_operations(ops: wgpu::Operations<T>) -> Self {
        Self {
            load: ops.load,
            store: AttachmentStoreOp::static_op(ops.store),
        }
    }

    /// Resolves concrete wgpu operations for the frame sample count.
    pub fn resolve(self, sample_count: u32) -> wgpu::Operations<T> {
        wgpu::Operations {
            load: self.load,
            store: self.store.resolve(sample_count),
        }
    }
}

/// Returns whether two wgpu store operations are the same.
fn store_ops_match(a: wgpu::StoreOp, b: wgpu::StoreOp) -> bool {
    matches!(
        (a, b),
        (wgpu::StoreOp::Store, wgpu::StoreOp::Store)
            | (wgpu::StoreOp::Discard, wgpu::StoreOp::Discard)
    )
}

/// Compiled render-pass attachment template.
#[derive(Clone, Debug)]
pub struct RenderPassTemplate {
    /// Color attachments in declaration order.
    pub color_attachments: Vec<ColorAttachmentTemplate>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthAttachmentTemplate>,
    /// Optional multiview mask.
    pub multiview_mask: Option<std::num::NonZeroU32>,
}

/// Color attachment template.
#[derive(Clone, Debug)]
pub struct ColorAttachmentTemplate {
    /// Color target handle.
    pub target: TextureAttachmentTarget,
    /// Load operation.
    pub load: wgpu::LoadOp<wgpu::Color>,
    /// Store operation.
    pub store: AttachmentStoreOp,
    /// Optional resolve target.
    pub resolve_to: Option<TextureAttachmentResolve>,
}

/// Depth/stencil attachment template.
#[derive(Clone, Debug)]
pub struct DepthAttachmentTemplate {
    /// Depth/stencil target handle.
    pub target: TextureAttachmentTarget,
    /// Depth operations.
    pub depth: AttachmentOps<f32>,
    /// Optional stencil operations.
    pub stencil: Option<wgpu::Operations<u32>>,
}
