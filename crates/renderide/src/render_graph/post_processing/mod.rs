//! Post-processing stack framework: trait, signature, and graph wiring helpers.
//!
//! Effects are inserted between the world-mesh forward HDR producer
//! ([`crate::render_graph::passes::WorldMeshForwardOpaquePass`]) and the displayable target blit
//! ([`crate::render_graph::passes::SceneColorComposePass`]). Each effect registers a subgraph on
//! the builder whose head samples one HDR float texture and whose tail writes another; the
//! [`PostProcessChain`] allocates the ping-pong HDR slots and wires edges between effects. Most
//! effects contribute a single raster pass (GTAO, ACES tonemap); a few (bloom) register a mip
//! ladder terminating in a single composite pass.
//!
//! See [`crate::render_graph::passes::post_processing`] for concrete effect implementations:
//! [`GtaoPass`](crate::render_graph::passes::post_processing::GtaoPass),
//! [`BloomEffect`](crate::render_graph::passes::post_processing::BloomEffect), and
//! [`AcesTonemapPass`](crate::render_graph::passes::post_processing::AcesTonemapPass).

mod chain;
pub(crate) mod effect;

pub use chain::{ChainOutput, PostProcessChain, PostProcessChainSignature};
pub use effect::{EffectPasses, PostProcessEffect, PostProcessEffectId};
