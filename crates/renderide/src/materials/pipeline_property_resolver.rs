//! Caches the interned [`MaterialPipelinePropertyIds`] so per-frame and per-draw call sites do
//! not re-intern the same Unity property names every time.
//!
//! The renderer evaluates blend mode and render-state overrides multiple times per frame: once
//! per material in [`crate::backend::facade::frame_packet`] and again per draw inside
//! [`crate::materials::material_blend_mode_for_lookup`] /
//! [`crate::materials::render_state::material_render_state_for_lookup`]. Both paths need the same
//! `MaterialPipelinePropertyIds`. Without a resolver, every call constructs a new
//! [`MaterialPipelinePropertyIds`] which goes through ~14
//! [`crate::materials::host_data::PropertyIdRegistry::intern`] calls; the resolver caches the
//! result behind a `Mutex` so only the first call pays that cost.

use std::sync::Arc;

use parking_lot::Mutex;

use super::host_data::PropertyIdRegistry;
use super::material_passes::MaterialPipelinePropertyIds;

/// Caches a [`MaterialPipelinePropertyIds`] keyed off a [`PropertyIdRegistry`].
///
/// One resolver lives next to the registry that minted its property ids. Cloning is cheap (the
/// inner cache is `Arc<Mutex<_>>`). The first [`Self::resolve`] call interns through the
/// registry; subsequent calls return the cached snapshot until [`Self::invalidate`] is called.
#[derive(Clone)]
pub struct PipelinePropertyResolver {
    registry: Arc<PropertyIdRegistry>,
    cache: Arc<Mutex<Option<MaterialPipelinePropertyIds>>>,
}

impl PipelinePropertyResolver {
    /// Creates a resolver over `registry`. Construction does not intern anything.
    pub fn new(registry: Arc<PropertyIdRegistry>) -> Self {
        Self {
            registry,
            cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns a copy of the resolved [`MaterialPipelinePropertyIds`], interning property names
    /// through the underlying [`PropertyIdRegistry`] on the first call.
    pub fn resolve(&self) -> MaterialPipelinePropertyIds {
        let mut slot = self.cache.lock();
        if let Some(ids) = slot.as_ref() {
            return *ids;
        }
        let ids = MaterialPipelinePropertyIds::new(self.registry.as_ref());
        *slot = Some(ids);
        ids
    }

    /// Drops the cached ids so the next [`Self::resolve`] re-interns. Useful when the underlying
    /// registry has been replaced (e.g. tests, hot-reload).
    pub fn invalidate(&self) {
        *self.cache.lock() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_returns_consistent_ids_across_calls() {
        let registry = Arc::new(PropertyIdRegistry::new());
        let resolver = PipelinePropertyResolver::new(Arc::clone(&registry));
        let first = resolver.resolve();
        let second = resolver.resolve();
        assert_eq!(first.src_blend, second.src_blend);
        assert_eq!(first.cull, second.cull);
    }

    #[test]
    fn invalidate_forces_reintern() {
        let registry = Arc::new(PropertyIdRegistry::new());
        let resolver = PipelinePropertyResolver::new(Arc::clone(&registry));
        let _ = resolver.resolve();
        resolver.invalidate();
        let _ = resolver.resolve();
    }
}
