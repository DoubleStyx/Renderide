//! Cached pipelines and bind layout for [`super::AgxTonemapPass`].
//!
//! Delegates to the shared
//! [`FullscreenD2ArraySampledPipelineCache`](crate::passes::helpers::FullscreenD2ArraySampledPipelineCache):
//! one filterable D2-array texture + linear-clamp sampler, mono/multiview pipelines keyed by
//! output format. WGSL is sourced from the runtime shader package.

use crate::passes::helpers::define_fullscreen_d2_array_pipeline_cache;

/// Upper bound for cached AgX bind groups before the cache is flushed.
const MAX_CACHED_BIND_GROUPS: usize = 8;

define_fullscreen_d2_array_pipeline_cache! {
    /// GPU state shared by all AgX tonemap passes (bind layout + sampler + per-format pipelines).
    pub(super) AgxTonemapPipelineCache {
        base: "agx_tonemap",
        sampled_view: "agx_tonemap_sampled",
        mono: "agx_tonemap_default",
        multiview: "agx_tonemap_multiview",
        max_bind_groups: MAX_CACHED_BIND_GROUPS,
        churn_site: "passes::agx_tonemap_bind_group",
    }
}
