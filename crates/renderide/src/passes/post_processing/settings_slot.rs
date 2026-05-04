//! Per-view blackboard slots that propagate live post-processing settings into the chain.

use crate::render_graph::blackboard::BlackboardSlot;

/// Blackboard slot for the live [`crate::config::GtaoSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so
/// the GTAO chain ([`crate::passes::post_processing::GtaoEffect`]) reads the current slider
/// values without rebuilding the compiled render graph. Non-topology slider changes don't
/// flip [`crate::render_graph::post_processing::chain::PostProcessChainSignature`] -- this
/// slot is the path that propagates those edits into the per-stage UBO writes.
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
/// Non-topology edits (intensity, low-frequency boost, threshold, composite mode, ...) flow in via
/// this slot; only the effective `max_mip_dimension` changes force a rebuild because it resizes
/// the mip-chain transient textures -- the chain signature tracks that value explicitly.
pub struct BloomSettingsSlot;
impl BlackboardSlot for BloomSettingsSlot {
    type Value = BloomSettingsValue;
}

/// Live [`crate::config::BloomSettings`] carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct BloomSettingsValue(pub crate::config::BloomSettings);

/// Blackboard slot for the live [`crate::config::AutoExposureSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so the
/// auto-exposure histogram pass can update its GPU settings buffer without rebuilding the graph.
/// The frame delta is carried alongside the settings because exposure adaptation is temporal.
pub struct AutoExposureSettingsSlot;
impl BlackboardSlot for AutoExposureSettingsSlot {
    type Value = AutoExposureSettingsValue;
}

/// Live auto-exposure settings and frame delta carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct AutoExposureSettingsValue {
    /// Current renderer-config auto-exposure settings.
    pub settings: crate::config::AutoExposureSettings,
    /// Wall-clock delta for temporal adaptation, in seconds.
    pub delta_seconds: f32,
}

impl Default for AutoExposureSettingsValue {
    fn default() -> Self {
        Self {
            settings: crate::config::AutoExposureSettings::default(),
            delta_seconds: 1.0 / 60.0,
        }
    }
}
