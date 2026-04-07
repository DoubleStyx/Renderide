//! Light cache and resolved light types for rendering.
//!
//! Stores light data per space, merges with scene transforms, and produces
//! world-space resolved lights for the render loop.
//!
//! **LightsBufferRenderer** paths use buffer submissions plus incremental
//! `LightsBufferRendererUpdate` batches; **regular** `Light` components use
//! [`LightCache::apply_regular_lights_update`]. FrooxEngine sends **incremental** dirty batches,
//! matching `ChangesHandlingRenderableComponentManager` on the host. The cache **merges** each
//! batch into persistent per-slot storage (like Unity’s `RenderableStateChangeManager`), then
//! flattens into [`LightCache::spaces`](LightCache::spaces) for resolve.

mod cache;
mod types;

pub use cache::LightCache;
pub use types::{CachedLight, ResolvedLight, light_casts_shadows};

#[cfg(test)]
mod tests;
