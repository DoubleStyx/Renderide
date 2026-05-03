//! Numeric limits for [`super::RenderingSettings`]: max-frame-latency bounds and the resolved
//! clamping helper, expressed in terms of [`crate::config::value::Clamped`].

use crate::config::value::Clamped;

/// Default value for [`super::RenderingSettings::max_frame_latency`]. Matches wgpu and Bevy.
pub const DEFAULT_MAX_FRAME_LATENCY: u32 = 2;

/// Inclusive lower bound for [`super::RenderingSettings::max_frame_latency`] (wgpu's hard minimum).
pub const MIN_MAX_FRAME_LATENCY: u32 = 1;

/// Inclusive upper bound for [`super::RenderingSettings::max_frame_latency`].
///
/// wgpu does not hard-enforce a maximum, but values above `3` only add presentation latency
/// without measurably improving throughput on the backends Renderide targets.
pub const MAX_MAX_FRAME_LATENCY: u32 = 3;

/// Type-level result of resolving a raw `max_frame_latency` config value into the wgpu-safe range.
///
/// Construction goes through [`Clamped::with_default_for_zero`] so a stray `0` resolves back to
/// [`DEFAULT_MAX_FRAME_LATENCY`] rather than being promoted to `MIN_MAX_FRAME_LATENCY`.
pub type MaxFrameLatency = Clamped<MIN_MAX_FRAME_LATENCY, MAX_MAX_FRAME_LATENCY>;

/// Default helper for `#[serde(default = ...)]` so a missing field round-trips to
/// [`DEFAULT_MAX_FRAME_LATENCY`] rather than `0`.
pub(super) fn default_max_frame_latency() -> u32 {
    DEFAULT_MAX_FRAME_LATENCY
}

/// Resolves a raw `max_frame_latency` to the typed [`MaxFrameLatency`] range.
///
/// Mirrors the historical `RenderingSettings::resolved_max_frame_latency` helper but goes
/// through the shared [`Clamped`] primitive so the algebra is reusable.
pub fn resolve_max_frame_latency(raw: u32) -> MaxFrameLatency {
    MaxFrameLatency::with_default_for_zero(raw, DEFAULT_MAX_FRAME_LATENCY)
}
