//! Post-processing stack framework: trait, signature, and graph wiring helpers.
//!
//! Effects are inserted between the world-mesh forward HDR producer
//! ([`crate::render_graph::passes::WorldMeshForwardOpaquePass`]) and the displayable target blit
//! ([`crate::render_graph::passes::SceneColorComposePass`]). Each effect is a single render pass
//! that reads one HDR float texture and writes another; the [`PostProcessChain`] handles
//! ping-pong allocation between effects.
//!
//! See [`crate::render_graph::passes::post_processing`] for concrete effect implementations
//! (currently: [`crate::render_graph::passes::post_processing::AcesTonemapPass`]).

mod chain;
mod effect;

pub use chain::{ChainOutput, PostProcessChain, PostProcessChainSignature};
pub use effect::{PostProcessEffect, PostProcessEffectId};
