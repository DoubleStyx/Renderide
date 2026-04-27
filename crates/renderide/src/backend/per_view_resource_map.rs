//! Small keyed owner for resources that live per logical render view.

use hashbrown::HashMap;

use crate::render_graph::OcclusionViewId;

/// Per-view resource map with the repeated create/get/retire lifecycle used by frame resources.
pub(crate) struct PerViewResourceMap<T> {
    /// Resources keyed by stable occlusion/render view identity.
    entries: HashMap<OcclusionViewId, T>,
}

impl<T> Default for PerViewResourceMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PerViewResourceMap<T> {
    /// Creates an empty per-view resource map.
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Returns a shared reference for `view_id`.
    pub(crate) fn get(&self, view_id: OcclusionViewId) -> Option<&T> {
        self.entries.get(&view_id)
    }

    /// Returns a mutable reference for `view_id`.
    pub(crate) fn get_mut(&mut self, view_id: OcclusionViewId) -> Option<&mut T> {
        self.entries.get_mut(&view_id)
    }

    /// Returns true when a resource exists for `view_id`.
    pub(crate) fn contains_key(&self, view_id: OcclusionViewId) -> bool {
        self.entries.contains_key(&view_id)
    }

    /// Returns the existing resource or inserts one built by `create`.
    pub(crate) fn get_or_insert_with<F>(&mut self, view_id: OcclusionViewId, create: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        self.entries.entry(view_id).or_insert_with(create)
    }

    /// Removes the resource for `view_id`, returning true when one existed.
    pub(crate) fn retire(&mut self, view_id: OcclusionViewId) -> bool {
        self.entries.remove(&view_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_or_insert_reuses_existing_entry() {
        let mut map = PerViewResourceMap::new();
        *map.get_or_insert_with(OcclusionViewId::Main, || 7) = 8;
        let value = map.get_or_insert_with(OcclusionViewId::Main, || 99);
        assert_eq!(*value, 8);
    }

    #[test]
    fn retire_removes_only_target_view() {
        let mut map = PerViewResourceMap::new();
        map.get_or_insert_with(OcclusionViewId::Main, || 1);
        map.get_or_insert_with(OcclusionViewId::OffscreenRenderTexture(4), || 2);

        assert!(map.retire(OcclusionViewId::Main));
        assert!(map.get(OcclusionViewId::Main).is_none());
        assert_eq!(
            map.get(OcclusionViewId::OffscreenRenderTexture(4)).copied(),
            Some(2)
        );
    }
}
