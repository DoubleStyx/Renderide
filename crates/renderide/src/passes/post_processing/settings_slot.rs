//! Per-view blackboard slots that propagate live post-processing settings into the chain.

use crate::render_graph::blackboard::BlackboardSlot;

/// Blackboard slot for the live [`crate::config::GtaoSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so
/// [`crate::passes::post_processing::GtaoPass`] reads the current slider values without
/// rebuilding the compiled render graph. Slider changes don't flip
/// [`crate::render_graph::post_processing::chain::PostProcessChainSignature`] (which tracks
/// enable flags only) — this slot is the path that propagates parameter edits into the UBO.
pub struct GtaoSettingsSlot;
impl BlackboardSlot for GtaoSettingsSlot {
    type Value = GtaoSettingsValue;
}

/// Live [`crate::config::GtaoSettings`] carried on the per-view blackboard.
///
/// Wraps `GtaoSettings` by value; the blackboard slot trait needs a concrete type living in this
/// module and the inner settings type lives in `crate::config`.
#[derive(Clone, Copy, Debug)]
pub struct GtaoSettingsValue(pub crate::config::GtaoSettings);

/// Blackboard slot for the live [`crate::config::BloomSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so the
/// bloom passes read the current slider values without rebuilding the compiled render graph.
/// Non-topology edits (intensity, low-frequency boost, threshold, composite mode, …) flow in via
/// this slot; only the effective `max_mip_dimension` changes force a rebuild because it resizes
/// the mip-chain transient textures — the chain signature tracks that value explicitly.
pub struct BloomSettingsSlot;
impl BlackboardSlot for BloomSettingsSlot {
    type Value = BloomSettingsValue;
}

/// Live [`crate::config::BloomSettings`] carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct BloomSettingsValue(pub crate::config::BloomSettings);
