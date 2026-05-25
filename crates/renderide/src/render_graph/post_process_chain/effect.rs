//! [`PostProcessEffect`] trait and identity enum used by [`super::PostProcessChain`].
//!
//! Effects read one HDR float texture and write another, but are free to register an arbitrary
//! subgraph in between (a single raster pass, a compute -> raster pair, a bloom mip-chain ladder,
//! etc.). They are added to the chain in execution order; each enabled effect's terminal pass
//! hands its output to the next effect (or to [`crate::passes::SceneColorComposePass`]
//! for the final one).

use crate::config::PostProcessingSettings;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::TextureHandle;

/// Stable identity for a regular post-processing chain effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostProcessEffectId {
    /// Dual-filter physically-based bloom, pre-tonemap HDR.
    Bloom,
    /// Histogram-based exposure adaptation, pre-tonemap HDR scale.
    AutoExposure,
    /// Screen-space motion blur, post-bloom and pre-tonemap.
    MotionBlur,
    /// Stephen Hill ACES Fitted tonemap (HDR linear -> display-referred 0..1 linear).
    AcesTonemap,
    /// Analytic AgX tonemap (HDR linear -> display-referred 0..1 linear).
    AgxTonemap,
}

impl PostProcessEffectId {
    /// Stable short label for logs and diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bloom => "Bloom",
            Self::AutoExposure => "Auto Exposure",
            Self::MotionBlur => "Motion Blur",
            Self::AcesTonemap => "ACES Tonemap",
            Self::AgxTonemap => "AgX Tonemap",
        }
    }
}

/// Pass range an effect contributed to the graph, or an exact pass-through result.
///
/// Returned by [`PostProcessEffect::register`]. `first` is the head of the subgraph (used as the
/// `to` endpoint for edges from the previous chain stage); `last` is the tail that terminates in
/// the effect's output texture (used as the `from` endpoint for edges into the next stage). For
/// single-pass effects both values are the same `PassId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectPasses {
    /// The effect is mathematically identical to forwarding its input handle.
    PassThrough,
    /// The effect registered one or more graph passes.
    Registered {
        /// First pass added by the effect (head of its subgraph).
        first: PassId,
        /// Last pass added by the effect (tail that writes the effect's output texture).
        last: PassId,
    },
}

impl EffectPasses {
    /// Helper for single-pass effects that contribute exactly one pass.
    pub fn single(pass: PassId) -> Self {
        Self::Registered {
            first: pass,
            last: pass,
        }
    }

    /// Helper for multi-pass effects that contribute a first and last pass.
    pub fn registered(first: PassId, last: PassId) -> Self {
        Self::Registered { first, last }
    }

    /// Helper for effects that are exact no-ops for the current graph shape.
    pub fn pass_through() -> Self {
        Self::PassThrough
    }
}

/// One effect contributing a subgraph to a [`super::PostProcessChain`].
///
/// Trait objects are stored in the chain in execution order. The chain calls [`Self::is_enabled`]
/// against the live [`PostProcessingSettings`] to decide whether to register the effect, and
/// [`Self::register`] to attach it to the graph builder. The effect must sample `input` (HDR
/// scene color, fragment stage) somewhere in its subgraph and write `output` (single color
/// attachment, HDR format) as its terminal pass. Any intermediate transient textures/resources
/// required by the effect are declared on `builder` and stay scoped to the effect.
pub trait PostProcessEffect: Send + Sync {
    /// Stable identity (also used for logging).
    fn id(&self) -> PostProcessEffectId;

    /// Whether this effect is configured for the current settings snapshot.
    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool;

    /// Registers this effect's passes against `builder`.
    ///
    /// Returning [`EffectPasses::PassThrough`] is allowed only when the effect is exactly
    /// equivalent to sampling the input and writing it unchanged to the output.
    fn register(
        &self,
        builder: &mut GraphBuilder,
        settings: &PostProcessingSettings,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses;
}
