//! Context types passed to each pass's recording method for one encoder slice.
//!
//! Context types correspond to the pass kinds in [`super::pass::PassNode`]:
//! - [`RasterPassCtx`] -- graph has already opened `wgpu::RenderPass`; pass records draws.
//! - [`ComputePassCtx`] -- pass receives the raw `wgpu::CommandEncoder` for compute work.
//! - [`EncoderPassCtx`] -- pass receives the raw `wgpu::CommandEncoder` for mixed work.
//!
//! [`PostSubmitContext`] is shared across all pass kinds for post-submit hooks.
//!
//! ## Lifetime parameters
//!
//! Contexts use up to three lifetime parameters:
//! - `'a` -- immutable GPU handles (device, limits, graph resources, views).
//! - `'encoder` -- mutable encoder borrow (compute and encoder contexts).
//! - `'frame` -- mutable scene/backend frame params borrow.
//!
//! Multi-view graph execution creates a **separate** encoder (and thus a separate context) for
//! frame-global work vs each view; passes never share one encoder across those slices.

mod pass;
mod resolved;

pub use pass::{ComputePassCtx, EncoderPassCtx, PostSubmitContext, RasterPassCtx};
pub use resolved::{
    GraphResolvedResources, ResolvedGraphBuffer, ResolvedGraphTexture, ResolvedImportedBuffer,
    ResolvedImportedHistoryTexture, ResolvedImportedTexture,
};
