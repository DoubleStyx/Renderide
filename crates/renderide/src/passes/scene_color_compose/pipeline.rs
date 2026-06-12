//! Cached pipelines and bind layout for [`super::SceneColorComposePass`].
//!
//! Delegates to the shared
//! [`FullscreenD2ArraySampledPipelineCache`](crate::passes::helpers::FullscreenD2ArraySampledPipelineCache):
//! one filterable D2-array texture + linear-clamp sampler, mono/multiview pipelines keyed by
//! output format. WGSL is sourced from the runtime shader package.

use crate::passes::helpers::define_fullscreen_d2_array_pipeline_cache;

/// Upper bound for cached scene-color-compose bind groups before the cache is flushed.
///
/// Normally one or two entries (mono + multiview). The cap protects against unbounded growth
/// when the transient pool cycles allocations (resize / MSAA toggle).
const MAX_CACHED_BIND_GROUPS: usize = 8;

define_fullscreen_d2_array_pipeline_cache! {
    /// GPU state shared by all compose passes (bind layout + sampler + per-format pipelines).
    pub(super) SceneColorComposePipelineCache {
        base: "scene_color_compose",
        sampled_view: "scene_color_compose_sampled",
        mono: "scene_color_compose_default",
        multiview: "scene_color_compose_multiview",
        max_bind_groups: MAX_CACHED_BIND_GROUPS,
        churn_site: "passes::scene_color_compose_bind_group",
    }
}
