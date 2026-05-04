//! Concrete post-processing render passes registered on the
//! [`crate::render_graph::post_processing::PostProcessChain`].
//!
//! The chain currently ships with four effects, executed in this order:
//! 1. [`GtaoEffect`] -- Ground-Truth Ambient Occlusion (pre-tonemap HDR modulation, with an
//!    XeGTAO-style depth-aware bilateral denoise stage between AO production and apply).
//! 2. [`AutoExposureEffect`] -- histogram-based exposure adaptation (pre-bloom HDR scale).
//! 3. [`BloomEffect`] -- dual-filter physically-based bloom (post-exposure, pre-tonemap HDR scatter).
//! 4. [`AcesTonemapPass`] -- Stephen Hill ACES Fitted tonemap.
//!
//! Future effects (color grading, etc.) live alongside them as sibling sub-modules and implement
//! [`crate::render_graph::post_processing::PostProcessEffect`].

mod aces_tonemap;
mod auto_exposure;
mod bloom;
mod gtao;
pub mod settings_slot;

pub use aces_tonemap::{AcesTonemapEffect, AcesTonemapGraphResources, AcesTonemapPass};
pub(crate) use auto_exposure::AutoExposureStateCache;
pub use auto_exposure::{AutoExposureApplyPass, AutoExposureComputePass, AutoExposureEffect};
pub use bloom::BloomEffect;
pub use gtao::GtaoEffect;
pub use settings_slot::{
    AutoExposureSettingsSlot, AutoExposureSettingsValue, BloomSettingsSlot, BloomSettingsValue,
    GtaoSettingsSlot, GtaoSettingsValue,
};
