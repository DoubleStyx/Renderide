//! Cached pipelines and bind layout for [`super::AcesTonemapPass`].
//!
//! Delegates to the shared
//! [`FullscreenD2ArraySampledPipelineCache`](crate::passes::helpers::FullscreenD2ArraySampledPipelineCache):
//! one filterable D2-array texture + linear-clamp sampler, mono/multiview pipelines keyed by
//! output format. WGSL is sourced from the runtime shader package.

use crate::passes::helpers::define_fullscreen_d2_array_pipeline_cache;

/// Upper bound for cached ACES bind groups before the cache is flushed.
///
/// The scene-color transient texture is stable across most frames -- the cache normally holds
/// one or two entries (mono + multiview). This cap protects against unbounded growth when the
/// swapchain / MSAA setting flips repeatedly and the transient pool cycles allocations.
const MAX_CACHED_BIND_GROUPS: usize = 8;

define_fullscreen_d2_array_pipeline_cache! {
    /// GPU state shared by all ACES tonemap passes (bind layout + sampler + per-format pipelines).
    pub(super) AcesTonemapPipelineCache {
        base: "aces_tonemap",
        sampled_view: "aces_tonemap_sampled",
        mono: "aces_tonemap_default",
        multiview: "aces_tonemap_multiview",
        max_bind_groups: MAX_CACHED_BIND_GROUPS,
        churn_site: "passes::aces_tonemap_bind_group",
    }
}
