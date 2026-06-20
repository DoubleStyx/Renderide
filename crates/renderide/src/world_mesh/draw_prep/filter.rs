//! Camera transform filtering for world-mesh draw collection.

use hashbrown::HashSet;

use crate::scene::{RenderSpaceId, RenderSpaceRead, SceneSpaceRead};

/// Selective / exclude transform lists for secondary cameras.
#[derive(Clone, Debug, Default)]
pub struct CameraTransformDrawFilter {
    /// When `Some`, only these transform node ids are drawn.
    pub only: Option<HashSet<i32>>,
    /// Transform roots skipped after the optional selective list has been applied.
    pub exclude: HashSet<i32>,
}

impl CameraTransformDrawFilter {
    /// Returns whether this filter has at least one selective transform root.
    pub(crate) fn has_selective_roots(&self) -> bool {
        self.only.as_ref().is_some_and(|only| !only.is_empty())
    }

    /// Returns `true` if `node_id` should be rendered under this filter.
    #[inline]
    #[cfg(test)]
    pub fn passes(&self, node_id: i32) -> bool {
        if let Some(only) = &self.only {
            only.contains(&node_id) && !self.exclude.contains(&node_id)
        } else {
            !self.exclude.contains(&node_id)
        }
    }

    /// Returns `true` if `node_id` should be rendered, treating filter entries as transform roots.
    ///
    /// Host camera selective/exclude lists are transform ids. Dashboard and UI cameras commonly list
    /// a parent transform, so child renderers must inherit that decision.
    pub fn passes_scene_node(
        &self,
        scene: &(impl SceneSpaceRead + ?Sized),
        space_id: RenderSpaceId,
        node_id: i32,
    ) -> bool {
        if let Some(only) = &self.only {
            if only.is_empty() {
                return false;
            }
            node_or_ancestor_in_set(scene, space_id, node_id, only)
                && (self.exclude.is_empty()
                    || !node_or_ancestor_in_set(scene, space_id, node_id, &self.exclude))
        } else {
            if self.exclude.is_empty() {
                return true;
            }
            !node_or_ancestor_in_set(scene, space_id, node_id, &self.exclude)
        }
    }

    /// Precomputes `passes_scene_node` for every node in `space_id` so per-draw filtering
    /// becomes an O(1) index lookup instead of repeated ancestor walks.
    ///
    /// Returns `None` when the space is missing; otherwise returns a `Vec<bool>` of length
    /// `space.local_transforms().len()` where `mask[node_id as usize] == true` iff the draw should render.
    pub fn build_pass_mask(
        &self,
        scene: &(impl SceneSpaceRead + ?Sized),
        space_id: RenderSpaceId,
    ) -> Option<Vec<bool>> {
        let space = scene.space(space_id)?;
        let n = space.local_transforms().len();
        if let Some(only) = &self.only {
            if only.is_empty() {
                return Some(vec![false; n]);
            }
            let mut mask = ancestor_membership_mask(scene, space_id, only);
            if !self.exclude.is_empty() {
                let excl = ancestor_membership_mask(scene, space_id, &self.exclude);
                for (pass, excluded) in mask.iter_mut().zip(excl) {
                    *pass &= !excluded;
                }
            }
            Some(mask)
        } else if self.exclude.is_empty() {
            Some(vec![true; n])
        } else {
            let excl = ancestor_membership_mask(scene, space_id, &self.exclude);
            Some(excl.into_iter().map(|e| !e).collect())
        }
    }
}

fn node_or_ancestor_in_set(
    scene: &(impl SceneSpaceRead + ?Sized),
    space_id: RenderSpaceId,
    node_id: i32,
    set: &HashSet<i32>,
) -> bool {
    if node_id < 0 || set.is_empty() {
        return false;
    }
    let Some(space) = scene.space(space_id) else {
        return false;
    };
    let mut cursor = node_id;
    let parents = space.node_parents();
    let node_count = space.local_transforms().len();
    for _ in 0..node_count {
        if set.contains(&cursor) {
            return true;
        }
        let Some(&parent) = parents.get(cursor as usize) else {
            return false;
        };
        if parent < 0 || parent == cursor || parent as usize >= node_count {
            return false;
        }
        cursor = parent;
    }
    false
}

/// Memoized ancestor-membership scan: for every node in `space_id`, returns whether it or any
/// ancestor appears in `set`. Amortized O(nodes), one pass with a path-painting cache.
fn ancestor_membership_mask(
    scene: &(impl SceneSpaceRead + ?Sized),
    space_id: RenderSpaceId,
    set: &HashSet<i32>,
) -> Vec<bool> {
    let Some(space) = scene.space(space_id) else {
        return Vec::new();
    };
    let n = space.local_transforms().len();
    if n == 0 || set.is_empty() {
        return vec![false; n];
    }
    // 0 = unknown, 1 = true, 2 = false
    let mut cache: Vec<u8> = vec![0; n];
    let mut path: Vec<usize> = Vec::with_capacity(32);
    for start in 0..n {
        if cache[start] != 0 {
            continue;
        }
        path.clear();
        let mut cur = start as i32;
        let hit;
        loop {
            if cur < 0 {
                hit = false;
                break;
            }
            let cu = cur as usize;
            if cu >= n {
                hit = false;
                break;
            }
            match cache[cu] {
                1 => {
                    hit = true;
                    break;
                }
                2 => {
                    hit = false;
                    break;
                }
                _ => {}
            }
            if set.contains(&cur) {
                cache[cu] = 1;
                hit = true;
                break;
            }
            path.push(cu);
            if path.len() > n {
                hit = false;
                break;
            }
            let Some(&parent) = space.node_parents().get(cu) else {
                hit = false;
                break;
            };
            if parent < 0 || parent == cur {
                hit = false;
                break;
            }
            cur = parent;
        }
        let marker = if hit { 1u8 } else { 2u8 };
        for &p in &path {
            cache[p] = marker;
        }
    }
    cache.into_iter().map(|v| v == 1).collect()
}

/// Builds a filter from a host [`crate::scene::CameraRenderableEntry`].
pub fn draw_filter_from_camera_entry(
    entry: &crate::scene::CameraRenderableEntry,
) -> CameraTransformDrawFilter {
    if entry.selective_transform_ids.is_empty() {
        CameraTransformDrawFilter {
            only: None,
            exclude: entry.exclude_transform_ids.iter().copied().collect(),
        }
    } else {
        CameraTransformDrawFilter {
            only: Some(entry.selective_transform_ids.iter().copied().collect()),
            exclude: entry.exclude_transform_ids.iter().copied().collect(),
        }
    }
}

#[cfg(test)]
mod tests;
