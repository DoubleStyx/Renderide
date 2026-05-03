//! Tonemapping configuration. Persisted as `[post_processing.tonemap]`.

use serde::{Deserialize, Serialize};

use crate::labeled_enum;

/// Tonemapping configuration. Persisted as `[post_processing.tonemap]`.
///
/// Tonemapping converts unbounded HDR scene-referred radiance to a bounded display-referred
/// linear signal. Output values are in `[0, 1]` linear sRGB so the existing sRGB swapchain
/// encodes gamma correctly without a separate gamma pass.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TonemapSettings {
    /// Selected tonemapping curve (see [`TonemapMode`]).
    pub mode: TonemapMode,
}

labeled_enum! {
    /// Tonemapping curve selector for [`TonemapSettings::mode`].
    ///
    /// Adding a new variant only requires extending the macro declaration and any new
    /// post-processing pass that consumes it; the chain signature in
    /// [`crate::render_graph::cache::PostProcessChainSignature`] does not need to change unless
    /// the new mode introduces additional render-graph passes.
    pub enum TonemapMode: "tonemap mode" {
        default => AcesFitted;

        /// No tonemapping (raw HDR is passed through, identical to the master-disabled path but
        /// kept as an explicit option so the master toggle can stay enabled while only other
        /// future effects run).
        None => {
            persist: "none",
            label: "None (HDR pass-through)",
        },
        /// Stephen Hill ACES Fitted (sRGB -> AP1, RRT+ODT, AP1 -> sRGB). High-quality reference
        /// curve used by Bevy and Unity HDRP.
        AcesFitted => {
            persist: "aces_fitted",
            label: "ACES Fitted (Hill)",
        },
    }
}
