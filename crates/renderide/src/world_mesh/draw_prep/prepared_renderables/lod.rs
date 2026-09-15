//! Prepared LOD-group snapshots derived from frame-prepared renderer runs.

use hashbrown::{HashMap, HashSet};

use crate::render_contract::ParticleDrawKind;
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, WorldMeshSceneRead};

use super::{FramePreparedDraw, FramePreparedRenderables, FramePreparedRun};

type PreparedLodAabb = (glam::Vec3, glam::Vec3);

/// Ordinal lookup used while recomputing prepared LOD group bounds.
///
/// Normal scene tables have dense ordinals and retain the cache-friendly flat-vector path. Asset
/// residency can make the prepared subset extremely sparse, though, so a high live ordinal must
/// not force a vector sized like the entire scene table.
enum PreparedLodBoundsByOrdinal {
    Dense(Vec<Option<PreparedLodAabb>>),
    Sparse(HashMap<usize, Option<PreparedLodAabb>>),
}

impl PreparedLodBoundsByOrdinal {
    fn from_rows(rows: Vec<(usize, Option<PreparedLodAabb>)>) -> Self {
        let Some(max_ordinal) = rows.iter().map(|(ordinal, _)| *ordinal).max() else {
            return Self::Dense(Vec::new());
        };
        let dense_limit = rows.len().saturating_mul(4).saturating_add(64);
        if max_ordinal <= dense_limit {
            let mut dense = vec![None; max_ordinal + 1];
            for (ordinal, bounds) in rows {
                dense[ordinal] = bounds;
            }
            Self::Dense(dense)
        } else {
            let mut sparse = HashMap::with_capacity(rows.len());
            for (ordinal, bounds) in rows {
                sparse.insert(ordinal, bounds);
            }
            Self::Sparse(sparse)
        }
    }

    fn world_aabb(&self, ordinal: usize) -> Option<PreparedLodAabb> {
        match self {
            Self::Dense(bounds) => bounds.get(ordinal).copied().flatten(),
            Self::Sparse(bounds) => bounds.get(&ordinal).copied().flatten(),
        }
    }
}

/// One renderer referenced by a prepared LOD entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::world_mesh::draw_prep) struct FramePreparedLodRenderer {
    /// Stable per-space renderer ordinal used by dense visibility bitsets.
    pub(in crate::world_mesh::draw_prep) renderer_ordinal: usize,
}

/// One prepared LOD row with live renderer membership pre-resolved.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::world_mesh::draw_prep) struct FramePreparedLodEntry {
    /// Threshold copied from scene LOD state.
    pub(in crate::world_mesh::draw_prep) screen_relative_transition_height: f32,
    /// Live renderer ordinals selected by this LOD row.
    pub(in crate::world_mesh::draw_prep) renderers: Vec<FramePreparedLodRenderer>,
}

/// One prepared LOD group with membership and view-invariant bounds pre-resolved.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::world_mesh::draw_prep) struct FramePreparedLodGroup {
    /// Render space that owns the LOD group.
    pub(in crate::world_mesh::draw_prep) space_id: RenderSpaceId,
    /// Index into the scene render space's LOD group table.
    pub(in crate::world_mesh::draw_prep) scene_group_index: usize,
    /// Whether any referenced renderer is in the overlay layer.
    pub(in crate::world_mesh::draw_prep) any_overlay: bool,
    /// Cached group bounds when every referenced renderer has view-invariant geometry.
    pub(in crate::world_mesh::draw_prep) world_aabb: Option<(glam::Vec3, glam::Vec3)>,
    /// Ordered LOD entries with stale renderer references removed.
    pub(in crate::world_mesh::draw_prep) lods: Vec<FramePreparedLodEntry>,
}

impl FramePreparedRenderables {
    /// Refreshes cached LOD group bounds when the scene's LOD rows are provably unchanged,
    /// otherwise rebuilds membership from scratch.
    ///
    /// Bounds refits move renderers; they cannot change which renderer belongs to which LOD row.
    /// Rebuilding membership anyway meant clearing the whole table and rehashing every prepared
    /// run on every transform move, which measured 2.4 ms/frame in a 3.5k renderer world.
    pub(super) fn refresh_lod_groups_for_bounds_refit<S>(
        &mut self,
        scene: &S,
        touched_spaces: &HashSet<RenderSpaceId>,
        changed_ordinals: Option<&HashSet<(RenderSpaceId, usize)>>,
    ) where
        S: WorldMeshSceneRead + ?Sized,
    {
        let signature = self.lod_membership_signature(scene);
        if self.lod_membership_signature == Some(signature) {
            self.refresh_lod_group_bounds_for(touched_spaces, changed_ordinals);
            return;
        }
        self.rebuild_lod_groups(Some(scene));
    }

    /// Recomputes the cached group AABBs in `touched_spaces` from current draw cull geometry.
    ///
    /// Uses a flat vector for ordinary dense renderer ordinals and an adaptive sparse map when
    /// asset residency leaves only high-ordinal renderers prepared.
    ///
    /// Groups outside `touched_spaces` keep their cached bounds. A group's AABB is the union of its
    /// own space's renderer cull geometry, so a space nothing moved in cannot have stale bounds.
    /// Refreshes group bounds, restricted to groups owning `changed_ordinals` when supplied.
    fn refresh_lod_group_bounds_for(
        &mut self,
        touched_spaces: &HashSet<RenderSpaceId>,
        changed_ordinals: Option<&HashSet<(RenderSpaceId, usize)>>,
    ) {
        if self.lod_groups.is_empty() || touched_spaces.is_empty() {
            return;
        }
        // No group in any touched space references a renderer, so the full run scan below cannot
        // change a single cached AABB. In a one-space city world that scan is ~37k runs per patch.
        if !touched_spaces.iter().any(|space_id| {
            self.lod_member_ordinals
                .get(space_id)
                .is_some_and(|ordinals| !ordinals.is_empty())
        }) {
            return;
        }
        profiling::scope!("mesh::prepared_renderables::refresh_lod_group_bounds");
        // Resolve the affected groups FIRST, then gather bounds for only those groups' members.
        // Gathering every member in the space and only then filtering groups left the whole pass
        // O(scene), which is why scoping the group loop alone barely moved it (1617 -> 1133us). A
        // group's AABB is a union, so recomputing one still needs all of ITS members. -xlinka
        let affected = changed_ordinals.map(|changed| {
            let mut slots = HashSet::new();
            for key in changed {
                if let Some(groups) = self.lod_groups_by_ordinal.get(key) {
                    slots.extend(groups.iter().copied());
                }
            }
            slots
        });
        if affected.as_ref().is_some_and(HashSet::is_empty) {
            return;
        }
        let mut rows_by_space: HashMap<RenderSpaceId, Vec<(usize, Option<PreparedLodAabb>)>> =
            HashMap::with_capacity(touched_spaces.len());
        {
            // Only the members of groups we will actually recompute need a bounds row.
            let want = |space_id: RenderSpaceId, ordinal: usize, rows: &mut HashMap<RenderSpaceId, Vec<(usize, Option<PreparedLodAabb>)>>| {
                let Some(&draw_index) = self
                    .lod_member_ordinals
                    .get(&space_id)
                    .and_then(|members| members.get(&ordinal))
                else {
                    return;
                };
                let Some(draw) = self.draws.get(draw_index) else {
                    return;
                };
                // Scene LOD rows reference only static/skinned renderers. A generated particle run
                // has its own ordinal range and must not overwrite a scene renderer's bounds.
                if draw.particle_draw.kind != ParticleDrawKind::None {
                    return;
                }
                rows.entry(space_id).or_default().push((
                    ordinal,
                    draw.cull_geometry.and_then(|geometry| geometry.world_aabb),
                ));
            };
            match affected.as_ref() {
                Some(slots) => {
                    for &group_index in slots {
                        let Some(group) = self.lod_groups.get(group_index) else {
                            continue;
                        };
                        if !touched_spaces.contains(&group.space_id) {
                            continue;
                        }
                        for lod in &group.lods {
                            for renderer in &lod.renderers {
                                want(group.space_id, renderer.renderer_ordinal, &mut rows_by_space);
                            }
                        }
                    }
                }
                None => {
                    for space_id in touched_spaces {
                        let ordinals = self
                            .lod_member_ordinals
                            .get(space_id)
                            .map(|members| members.keys().copied().collect::<Vec<_>>())
                            .unwrap_or_default();
                        for ordinal in ordinals {
                            want(*space_id, ordinal, &mut rows_by_space);
                        }
                    }
                }
            }
        }
        let bounds_by_space = rows_by_space
            .into_iter()
            .map(|(space_id, rows)| (space_id, PreparedLodBoundsByOrdinal::from_rows(rows)))
            .collect::<HashMap<_, _>>();
        for (group_index, group) in self.lod_groups.iter_mut().enumerate() {
            if !touched_spaces.contains(&group.space_id) {
                continue;
            }
            if affected
                .as_ref()
                .is_some_and(|slots| !slots.contains(&group_index))
            {
                continue;
            }
            let Some(space_bounds) = bounds_by_space.get(&group.space_id) else {
                group.world_aabb = None;
                continue;
            };
            let mut world_aabb = None;
            let mut view_dependent = false;
            for lod in &group.lods {
                for renderer in &lod.renderers {
                    // A renderer that dropped out of the run table has no resolvable bounds, so
                    // the group falls back to per-view bounds exactly like the full rebuild does.
                    match space_bounds.world_aabb(renderer.renderer_ordinal) {
                        Some(bounds) if !view_dependent => {
                            union_prepared_lod_aabb(&mut world_aabb, bounds);
                        }
                        Some(_) => {}
                        None => {
                            view_dependent = true;
                            world_aabb = None;
                        }
                    }
                }
            }
            group.world_aabb = world_aabb;
        }
    }

    /// Hashes everything on the scene side that LOD group membership is resolved from.
    ///
    /// Cheap because LOD rows are orders of magnitude fewer than renderers. The prepared-side
    /// inputs (renderer ordinals, overlay flags) are not hashed: every caller of
    /// [`Self::refresh_lod_groups_for_bounds_refit`] is a bounds or stable-in-place refit, and
    /// `prepared_range_is_stable` already guarantees those fields survive it unchanged. The run
    /// and draw counts are folded in as a shape guard.
    fn lod_membership_signature<S>(&self, scene: &S) -> u64
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        profiling::scope!("mesh::prepared_renderables::lod_membership_signature");
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        self.runs.len().hash(&mut hasher);
        self.draws.len().hash(&mut hasher);
        self.active_space_ids.hash(&mut hasher);
        // Each space announces its own LOD mutations, so folding in the version is equivalent to
        // hashing every group's every renderer and costs one integer per space instead. -xlinka
        for &space_id in &self.active_space_ids {
            scene.lod_generation(space_id).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Rebuilds pre-resolved LOD groups from the active scene spaces and current prepared draws.
    pub(in crate::world_mesh::draw_prep) fn rebuild_lod_groups<S>(&mut self, scene: Option<&S>)
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        self.rebuild_lod_groups_for_spaces(scene, None);
    }

    /// Rebuilds LOD groups, restricted to `only_spaces` when supplied.
    ///
    /// A LOD update in one space used to rebuild every group in every space: 7554us per call and 82
    /// calls during a Dark City load, purely to re-resolve members that could not have moved.
    /// Groups outside the set keep their resolved membership. -xlinka
    pub(in crate::world_mesh::draw_prep) fn rebuild_lod_groups_for_spaces<S>(
        &mut self,
        scene: Option<&S>,
        only_spaces: Option<&HashSet<RenderSpaceId>>,
    ) where
        S: WorldMeshSceneRead + ?Sized,
    {
        match only_spaces {
            Some(spaces) => {
                self.lod_groups.retain(|group| !spaces.contains(&group.space_id));
                self.lod_member_ordinals
                    .retain(|space_id, _| !spaces.contains(space_id));
            }
            None => {
                self.lod_groups.clear();
                self.lod_member_ordinals.clear();
            }
        }
        self.lod_groups_by_ordinal.clear();
        let Some(scene) = scene else {
            self.lod_membership_signature = None;
            return;
        };
        self.lod_membership_signature = Some(self.lod_membership_signature(scene));
        profiling::scope!("mesh::prepared_renderables::rebuild_lod_groups");
        // Only renderers a LOD group actually references need resolving. Building the lookup over
        // every run inserted ~37k entries in a city world to answer a few thousand questions.
        let mut wanted = HashSet::new();
        for &space_id in &self.active_space_ids {
            let Some(lod_groups) = scene.lod_groups(space_id) else {
                continue;
            };
            for group in lod_groups {
                for lod in &group.lods {
                    for renderer_ref in &lod.renderers {
                        wanted.insert((space_id, renderer_ref.instance_id));
                    }
                }
            }
        }
        if wanted.is_empty() {
            return;
        }
        let renderer_lookup = build_lod_renderer_lookup(&self.draws, &self.runs, &wanted);
        for &space_id in &self.active_space_ids {
            if only_spaces.is_some_and(|spaces| !spaces.contains(&space_id)) {
                continue;
            }
            let Some(lod_groups) = scene.lod_groups(space_id) else {
                continue;
            };
            for (scene_group_index, group) in lod_groups.iter().enumerate() {
                let mut view_dependent_bounds = false;
                let mut prepared_group = FramePreparedLodGroup {
                    space_id,
                    scene_group_index,
                    any_overlay: false,
                    world_aabb: None,
                    lods: Vec::new(),
                };
                for lod in &group.lods {
                    let mut prepared_lod = FramePreparedLodEntry {
                        screen_relative_transition_height: lod.screen_relative_transition_height,
                        renderers: Vec::with_capacity(lod.renderers.len()),
                    };
                    for renderer_ref in &lod.renderers {
                        let key = (space_id, renderer_ref.instance_id);
                        let Some(renderer) = renderer_lookup.get(&key).copied() else {
                            continue;
                        };
                        prepared_group.any_overlay |= renderer.is_overlay;
                        if let Some(bounds) = renderer.world_aabb {
                            if !view_dependent_bounds {
                                union_prepared_lod_aabb(&mut prepared_group.world_aabb, bounds);
                            }
                        } else {
                            view_dependent_bounds = true;
                            prepared_group.world_aabb = None;
                        }
                        prepared_lod.renderers.push(FramePreparedLodRenderer {
                            renderer_ordinal: renderer.renderer_ordinal,
                        });
                        self.lod_member_ordinals
                            .entry(space_id)
                            .or_default()
                            .insert(renderer.renderer_ordinal, renderer.draw_index);
                    }
                    prepared_group.lods.push(prepared_lod);
                }
                if prepared_group
                    .lods
                    .iter()
                    .any(|lod| !lod.renderers.is_empty())
                {
                    self.lod_groups.push(prepared_group);
                }
            }
        }
        // Group indices shift whenever groups are retained and rebuilt, so the ordinal index is
        // rebuilt from the final vector rather than maintained during the loop.
        self.lod_groups_by_ordinal.clear();
        for (group_index, group) in self.lod_groups.iter().enumerate() {
            for lod in &group.lods {
                for renderer in &lod.renderers {
                    let slots = self
                        .lod_groups_by_ordinal
                        .entry((group.space_id, renderer.renderer_ordinal))
                        .or_default();
                    if !slots.contains(&group_index) {
                        slots.push(group_index);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod ordinal_lookup_tests {
    use super::{PreparedLodAabb, PreparedLodBoundsByOrdinal};

    fn bounds(value: f32) -> PreparedLodAabb {
        (glam::Vec3::splat(value), glam::Vec3::splat(value + 1.0))
    }

    #[test]
    fn high_sparse_renderer_ordinal_uses_bounded_storage() {
        let low = bounds(1.0);
        let high = bounds(4.0);
        let lookup = PreparedLodBoundsByOrdinal::from_rows(vec![
            (2, Some(low)),
            (1_000_000_000, Some(high)),
        ]);

        assert!(matches!(&lookup, PreparedLodBoundsByOrdinal::Sparse(_)));
        assert_eq!(lookup.world_aabb(2), Some(low));
        assert_eq!(lookup.world_aabb(1_000_000_000), Some(high));
        assert_eq!(lookup.world_aabb(3), None);
    }

    #[test]
    fn dense_renderer_ordinals_keep_flat_lookup() {
        let lookup = PreparedLodBoundsByOrdinal::from_rows(
            (0..32)
                .map(|ordinal| (ordinal, Some(bounds(ordinal as f32))))
                .collect(),
        );

        assert!(matches!(&lookup, PreparedLodBoundsByOrdinal::Dense(_)));
        assert_eq!(lookup.world_aabb(31), Some(bounds(31.0)));
    }
}

/// Renderer metadata used while rebuilding prepared LOD groups.
#[derive(Clone, Copy)]
struct PreparedLodRendererLookup {
    /// Stable renderer ordinal.
    renderer_ordinal: usize,
    /// Whether the renderer is in the overlay layer.
    is_overlay: bool,
    /// View-invariant renderer AABB when available.
    world_aabb: Option<(glam::Vec3, glam::Vec3)>,
    /// First draw row of the renderer's run, used to re-read bounds without a run sweep.
    draw_index: usize,
}

/// Builds a lookup from stable renderer identity to prepared LOD metadata.
fn build_lod_renderer_lookup(
    draws: &[FramePreparedDraw],
    runs: &[FramePreparedRun],
    wanted: &HashSet<(RenderSpaceId, MeshRendererInstanceId)>,
) -> HashMap<(RenderSpaceId, MeshRendererInstanceId), PreparedLodRendererLookup> {
    let mut lookup = HashMap::with_capacity(wanted.len());
    for run in runs {
        let Some(first) = draws.get(run.start as usize) else {
            continue;
        };
        if !wanted.contains(&(first.space_id, first.instance_id)) {
            continue;
        }
        lookup.insert(
            (first.space_id, first.instance_id),
            PreparedLodRendererLookup {
                renderer_ordinal: first.renderer_ordinal,
                is_overlay: first.is_overlay,
                world_aabb: first.cull_geometry.and_then(|geometry| geometry.world_aabb),
                draw_index: run.start as usize,
            },
        );
    }
    lookup
}

/// Expands `dst` to include a prepared renderer AABB.
fn union_prepared_lod_aabb(
    dst: &mut Option<(glam::Vec3, glam::Vec3)>,
    bounds: (glam::Vec3, glam::Vec3),
) {
    match dst {
        Some((min, max)) => {
            *min = min.min(bounds.0);
            *max = max.max(bounds.1);
        }
        None => *dst = Some(bounds),
    }
}
