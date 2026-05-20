//! Pass-node trait hierarchy, builder, and setup data.
//!
//! ## Pass kinds
//!
//! The render graph stores `Vec<PassNode>`. Each node wraps one of three typed pass traits:
//!
//! | Kind | Trait | GPU work |
//! |------|-------|----------|
//! | [`PassKind::Raster`] | [`RasterPass`] | Graph opens render pass; pass records draws. |
//! | [`PassKind::Compute`] | [`ComputePass`] | Pass receives raw encoder; dispatches compute. |
//! | [`PassKind::Encoder`] | [`EncoderPass`] | Pass receives raw encoder; records mixed copy/render work. |
//!
//! ## Setup flow
//!
//! During graph build, each pass's [`RasterPass::setup`] / [`ComputePass::setup`] is called with a
//! [`PassBuilder`]. The builder accumulates resource declarations, attachment templates, and the
//! pass kind flag (`raster()` / `compute()` / `encoder()`).
//! [`PassBuilder::finish`] validates the combination and emits a [`PassSetup`].

mod attachments;
pub mod builder;
pub mod compute;
pub mod encoder;
pub mod node;
pub mod params;
pub mod raster;
pub(crate) mod setup;
pub mod template;

pub use builder::PassBuilder;
pub use compute::ComputePass;
pub use encoder::EncoderPass;
pub use node::PassMergeHint;
pub use node::{GroupScope, PassKind, PassNode, PassPhase, PassWorkloadFlags};
pub use params::{
    BlackboardAccessDecl, BlackboardSeedDecl, BlackboardSlotKey, PassParameterSchema,
};
pub use raster::RasterPass;
pub use setup::PassSetup;
pub use template::{ColorAttachmentTemplate, DepthAttachmentTemplate, RenderPassTemplate};
