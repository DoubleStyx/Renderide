//! [`PostProcessEffect`] trait and identity enum used by [`super::PostProcessChain`].
//!
//! Effects are render passes with a stable identity that read one HDR float texture and write
//! another. They are added to the chain in execution order; each enabled effect contributes one
//! raster pass whose input is the previous effect's output (or the forward HDR target for the
//! first one) and whose output is the next ping-pong slot (or the chain's final output for the
//! last one).

use crate::config::PostProcessingSettings;
use crate::render_graph::pass::RasterPass;
use crate::render_graph::resources::TextureHandle;

/// Stable identity for a post-processing effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostProcessEffectId {
    /// Ground-Truth Ambient Occlusion (Jimenez et al. 2016), pre-tonemap HDR modulation.
    Gtao,
    /// Stephen Hill ACES Fitted tonemap (HDR linear → display-referred 0..1 linear).
    AcesTonemap,
}

impl PostProcessEffectId {
    /// Stable short label for logs and diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Gtao => "GTAO",
            Self::AcesTonemap => "ACES Tonemap",
        }
    }
}

/// One effect that contributes a single raster pass to a [`super::PostProcessChain`].
///
/// Trait objects are stored in the chain in execution order. The chain calls [`Self::is_enabled`]
/// against the live [`PostProcessingSettings`] to decide whether to allocate a pass for this
/// effect, and [`Self::build_pass`] to build the boxed [`RasterPass`] when it does.
pub trait PostProcessEffect: Send + Sync {
    /// Stable identity (also used for logging).
    fn id(&self) -> PostProcessEffectId;

    /// Whether this effect should run for the current settings snapshot.
    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool;

    /// Builds the raster pass for this effect, given the chain-allocated input/output handles.
    ///
    /// The pass must read `input` (sampled, fragment stage) and write `output` (single color
    /// attachment). The chain handles ping-pong wiring; effect implementations only need to know
    /// their immediate input and output.
    fn build_pass(&self, input: TextureHandle, output: TextureHandle) -> Box<dyn RasterPass>;
}
