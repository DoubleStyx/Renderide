//! Tracks which logical render views are currently active.
//!
//! Runtime view planning is the authoritative source of which [`crate::render_graph::ViewId`]s
//! exist in a frame. This registry turns that frame-local list into a stable ownership boundary
//! for backend systems that keep view-scoped state across frames.

use hashbrown::HashSet;

use crate::render_graph::ViewId;

/// Retained set of view identities that currently own backend state.
#[derive(Default)]
pub(crate) struct ViewResourceRegistry {
    /// View ids that were active on the most recent sync.
    active_views: HashSet<ViewId>,
}

impl ViewResourceRegistry {
    /// Creates an empty view-resource registry.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Updates the retained active-view set and returns views that were retired.
    pub(crate) fn sync_active_views<I>(&mut self, active_views: I) -> Vec<ViewId>
    where
        I: IntoIterator<Item = ViewId>,
    {
        let next_active: HashSet<ViewId> = active_views.into_iter().collect();
        let retired: Vec<ViewId> = self
            .active_views
            .difference(&next_active)
            .copied()
            .collect();
        self.active_views = next_active;
        retired
    }

    /// Number of views retained after the last sync.
    #[cfg(test)]
    pub(crate) fn active_view_count(&self) -> usize {
        self.active_views.len()
    }
}

#[cfg(test)]
mod tests {
    use super::ViewResourceRegistry;
    use crate::render_graph::ViewId;
    use crate::scene::RenderSpaceId;

    /// Builds a secondary-camera view id for registry tests.
    fn secondary_view(render_space_id: i32, renderable_index: i32) -> ViewId {
        ViewId::secondary_camera(RenderSpaceId(render_space_id), renderable_index)
    }

    /// Sync returns views that disappeared from the previous active set.
    #[test]
    fn sync_returns_only_retired_views() {
        let mut registry = ViewResourceRegistry::new();
        let first = secondary_view(10, 0);
        let second = secondary_view(10, 1);

        assert!(registry.sync_active_views([ViewId::Main, first]).is_empty());

        let retired = registry.sync_active_views([ViewId::Main, second]);
        assert_eq!(retired, vec![first]);
        assert_eq!(registry.active_view_count(), 2);
    }

    /// Repeated secondary-view churn keeps retained registry state bounded.
    #[test]
    fn repeated_create_and_unload_stays_bounded() {
        let mut registry = ViewResourceRegistry::new();

        for renderable_index in 0..128 {
            let view = secondary_view(42, renderable_index);
            let retired = registry.sync_active_views([ViewId::Main, view]);
            if renderable_index == 0 {
                assert!(retired.is_empty());
            } else {
                assert_eq!(retired.len(), 1);
            }
            assert_eq!(registry.active_view_count(), 2);
        }

        let retired = registry.sync_active_views([ViewId::Main]);
        assert_eq!(retired.len(), 1);
        assert_eq!(registry.active_view_count(), 1);
    }
}
