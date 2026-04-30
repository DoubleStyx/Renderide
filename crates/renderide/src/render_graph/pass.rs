//! Pass-node trait hierarchy, builder, and setup data.
//!
//! ## Pass kinds
//!
//! The render graph stores `Vec<PassNode>`. Each node wraps one of four typed pass traits:
//!
//! | Kind | Trait | GPU work |
//! |------|-------|----------|
//! | [`PassKind::Raster`] | [`RasterPass`] | Graph opens render pass; pass records draws. |
//! | [`PassKind::Compute`] | [`ComputePass`] | Pass receives raw encoder; dispatches compute. |
//! | [`PassKind::Copy`] | [`CopyPass`] | Pass receives raw encoder; copies/clears. |
//! | [`PassKind::Callback`] | [`CallbackPass`] | CPU-only; no encoder; uploads / blackboard writes. |
//!
//! ## Setup flow
//!
//! During graph build, each pass's [`RasterPass::setup`] / [`ComputePass::setup`] / etc. is
//! called with a [`PassBuilder`]. The builder accumulates resource declarations, attachment
//! templates, and the pass kind flag (`raster()` / `compute()` / `copy()` / `callback()`).
//! [`PassBuilder::finish`] validates the combination and emits a [`PassSetup`].

pub mod builder;
pub mod callback;
pub mod compute;
pub mod copy;
pub mod node;
pub mod raster;
pub(crate) mod setup;

pub use builder::{PassBuilder, RasterPassBuilder};
pub use callback::CallbackPass;
pub use compute::ComputePass;
pub use copy::CopyPass;
pub use node::{GroupScope, PassKind, PassMergeHint, PassNode, PassPhase};
pub use raster::RasterPass;
pub use setup::PassSetup;
