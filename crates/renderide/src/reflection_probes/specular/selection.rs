use glam::{Vec3, Vec3A};
use hashbrown::HashMap;

use crate::scene::RenderSpaceId;

/// Maximum number of probes in one BVH leaf.
const BVH_LEAF_SIZE: usize = 8;
const MIN_BLEND_DISTANCE: f32 = 1e-6;
const MAX_LOCAL_PROBES: usize = 4;

/// Per-draw reflection-probe selection stored in the per-draw slab.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReflectionProbeDrawSelection {
    /// Selected reflection-probe atlas indices (1 global + max 4 local)
    pub atlas_indices: [u16; 5],
    /// Bit mask which indicates, for each probe, if it is of lower importance than its predecessors
    pub importance_mask: u8,
}

/// CPU-side selector snapshot used during world-mesh draw collection.
#[derive(Default)]
pub struct ReflectionProbeFrameSelection {
    spaces: HashMap<RenderSpaceId, ReflectionProbeSpatialIndex>,
}

impl ReflectionProbeFrameSelection {
    /// Selects up to two probes for one object AABB.
    #[must_use]
    pub fn select(
        &self,
        space_id: RenderSpaceId,
        object_aabb: (Vec3, Vec3),
    ) -> ReflectionProbeDrawSelection {
        if let Some(selection) = self
            .spaces
            .get(&space_id)
            .map(|index| index.select(object_aabb))
            && selection.importance_mask > 0
        {
            return selection;
        }
        ReflectionProbeDrawSelection::default()
    }

    pub(super) fn rebuild_spatial<I>(&mut self, probes: I)
    where
        I: IntoIterator<Item = (RenderSpaceId, SpatialProbe)>,
    {
        self.spaces.clear();
        let mut by_space: HashMap<RenderSpaceId, Vec<SpatialProbe>> = HashMap::new();
        for (space_id, probe) in probes {
            by_space.entry(space_id).or_default().push(probe);
        }
        for (space_id, probes) in by_space {
            self.spaces
                .insert(space_id, ReflectionProbeSpatialIndex::build(probes));
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SpatialProbe {
    pub(super) renderable_index: i32,
    pub(super) atlas_index: u16,
    pub(super) importance: i32,
    pub(super) aabb_min: Vec3A,
    pub(super) aabb_max: Vec3A,
    pub(super) influence_aabb_min: Vec3A,
    pub(super) influence_aabb_max: Vec3A,
    pub(super) center: Vec3A,
    pub(super) volume: f32,
    pub(super) skybox: bool,
}

/// A BVH over reflection-probe AABBs for one render space.
#[derive(Default)]
pub struct ReflectionProbeSpatialIndex {
    probes: Vec<SpatialProbe>,
    order: Vec<usize>,
    nodes: Vec<BvhNode>,
    root: Option<usize>,
}

impl ReflectionProbeSpatialIndex {
    pub(super) fn build(probes: Vec<SpatialProbe>) -> Self {
        let mut out = Self {
            order: (0..probes.len()).collect(),
            probes,
            nodes: Vec::new(),
            root: None,
        };
        if !out.probes.is_empty() {
            let mut order = std::mem::take(&mut out.order);
            let end = order.len();
            out.root = Some(out.build_node(&mut order, 0, end));
            out.order = order;
        }
        out
    }

    /// Selects up to two probes for one object AABB.
    #[must_use]
    pub fn select(&self, object_aabb: (Vec3, Vec3)) -> ReflectionProbeDrawSelection {
        let object_min = Vec3A::from(object_aabb.0);
        let object_max = Vec3A::from(object_aabb.1);
        if self.root.is_none() || !aabb_valid(object_aabb.0, object_aabb.1) {
            return ReflectionProbeDrawSelection::default();
        }
        let object_center = object_center(object_min, object_max);
        let mut top: Vec<ProbeScore> = Vec::new();
        let mut fallback: Option<ProbeScore> = None;
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root.unwrap_or(0));
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if !aabb_intersects(node.aabb_min, node.aabb_max, object_min, object_max) {
                continue;
            }
            if node.count > 0 {
                for &probe_index in &self.order[node.start..node.start + node.count] {
                    let probe = &self.probes[probe_index];
                    let influence_intersection = intersection_volume_vec3a(
                        probe.influence_aabb_min,
                        probe.influence_aabb_max,
                        object_min,
                        object_max,
                    );
                    if influence_intersection < MIN_BLEND_DISTANCE {
                        continue;
                    }
                    let score = ProbeScore {
                        atlas_index: probe.atlas_index,
                        importance: probe.importance,
                        influence_intersection,
                        probe_volume: probe.volume,
                        center_distance_sq: (probe.center - object_center).length_squared(),
                        renderable_index: probe.renderable_index,
                        skybox: probe.skybox,
                    };
                    insert_probe_score(&mut top, score);
                    // Keep the best skybox probe as a fallback,
                    // even if it has lower importance
                    if probe.skybox {
                        fallback = fallback
                            .filter(|&best| score_better(best, score))
                            .or(Some(score));
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        selection_from_scores(top, fallback)
    }

    fn build_node(&mut self, order: &mut [usize], start: usize, end: usize) -> usize {
        let (aabb_min, aabb_max) = bounds_for_order(&self.probes, &order[start..end]);
        let index = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb_min,
            aabb_max,
            start,
            count: 0,
            left: 0,
            right: 0,
        });
        let count = end - start;
        if count <= BVH_LEAF_SIZE {
            self.nodes[index].count = count;
            return index;
        }
        let axis = largest_axis(aabb_max - aabb_min);
        order[start..end].sort_unstable_by(|&a, &b| {
            let ac = axis_value(self.probes[a].center, axis);
            let bc = axis_value(self.probes[b].center, axis);
            ac.total_cmp(&bc).then_with(|| {
                self.probes[a]
                    .renderable_index
                    .cmp(&self.probes[b].renderable_index)
            })
        });
        let mid = start + count / 2;
        let left = self.build_node(order, start, mid);
        let right = self.build_node(order, mid, end);
        self.nodes[index].left = left;
        self.nodes[index].right = right;
        index
    }
}

#[derive(Clone, Copy)]
struct BvhNode {
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    start: usize,
    count: usize,
    left: usize,
    right: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProbeScore {
    atlas_index: u16,
    importance: i32,
    influence_intersection: f32,
    probe_volume: f32,
    center_distance_sq: f32,
    renderable_index: i32,
    skybox: bool,
}

fn insert_probe_score(top: &mut Vec<ProbeScore>, score: ProbeScore) {
    for i in 0..top.len() {
        if score_better(score, top[i]) {
            top.insert(i, score);
            if top.len() > MAX_LOCAL_PROBES {
                top.pop();
            }
            return;
        }
    }
    if top.len() < MAX_LOCAL_PROBES {
        top.push(score);
    }
}

/// Order of preference:
/// 1. Largest importance set by creator
/// 2. Non-skybox preferred over skybox
/// 3. Largest influence intersection
/// 4. Smallest probe volume
/// 5. Closest to the center
/// 6. Highest renderable index (ie most recently activated)
fn score_better(a: ProbeScore, b: ProbeScore) -> bool {
    a.importance
        .cmp(&b.importance)
        .reverse()
        .then_with(|| a.skybox.cmp(&b.skybox))
        .then_with(|| {
            a.influence_intersection
                .total_cmp(&b.influence_intersection)
                .reverse()
        })
        .then_with(|| a.probe_volume.total_cmp(&b.probe_volume))
        .then_with(|| a.center_distance_sq.total_cmp(&b.center_distance_sq))
        .then_with(|| a.renderable_index.cmp(&b.renderable_index).reverse())
        .is_lt()
}

fn selection_from_scores(
    top: Vec<ProbeScore>,
    fallback: Option<ProbeScore>,
) -> ReflectionProbeDrawSelection {
    let mut atlas_indices = [0u16; 5];
    let mut importance_mask = 0u8;
    let mut importance = i32::MAX;
    if let Some(probe) = fallback {
        atlas_indices[0] = probe.atlas_index;
    }
    for (i, probe) in top.iter().take(MAX_LOCAL_PROBES).enumerate() {
        atlas_indices[i + 1] = probe.atlas_index;
        if probe.importance < importance {
            importance_mask |= 1 << i;
        }
        importance = probe.importance;
    }
    ReflectionProbeDrawSelection {
        atlas_indices,
        importance_mask,
    }
}

fn bounds_for_order(probes: &[SpatialProbe], order: &[usize]) -> (Vec3A, Vec3A) {
    let mut min = Vec3A::splat(f32::INFINITY);
    let mut max = Vec3A::splat(f32::NEG_INFINITY);
    for &index in order {
        min = min.min(probes[index].influence_aabb_min);
        max = max.max(probes[index].influence_aabb_max);
    }
    (min, max)
}

fn aabb_intersects(a_min: Vec3A, a_max: Vec3A, b_min: Vec3A, b_max: Vec3A) -> bool {
    a_min.cmple(b_max).all() && a_max.cmpge(b_min).all()
}

pub(super) fn aabb_valid(min: Vec3, max: Vec3) -> bool {
    min.is_finite() && max.is_finite() && (max - min).cmpgt(Vec3::ZERO).all()
}

pub(super) fn sanitized_blend_distance(blend_distance: f32) -> f32 {
    if blend_distance.is_finite() {
        blend_distance.max(0.0)
    } else {
        0.0
    }
}

pub(super) fn expanded_aabb(min: Vec3, max: Vec3, blend_distance: f32) -> (Vec3A, Vec3A) {
    let expansion = Vec3A::splat(sanitized_blend_distance(blend_distance));
    (Vec3A::from(min) - expansion, Vec3A::from(max) + expansion)
}

pub(super) fn aabb_volume(min: Vec3, max: Vec3) -> f32 {
    aabb_volume_vec3a(Vec3A::from(min), Vec3A::from(max))
}

fn aabb_volume_vec3a(min: Vec3A, max: Vec3A) -> f32 {
    let d = (max - min).max(Vec3A::ZERO);
    d.x * d.y * d.z
}

pub(super) fn intersection_volume_vec3a(
    a_min: Vec3A,
    a_max: Vec3A,
    b_min: Vec3A,
    b_max: Vec3A,
) -> f32 {
    let d = (a_max.min(b_max) - a_min.max(b_min)).max(Vec3A::ZERO);
    d.x * d.y * d.z
}

fn object_center(min: Vec3A, max: Vec3A) -> Vec3A {
    (min + max) * 0.5
}

fn largest_axis(v: Vec3A) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

fn axis_value(v: Vec3A, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(index: i32, atlas: u16, importance: i32, min: Vec3, max: Vec3) -> SpatialProbe {
        probe_with_blend(index, atlas, importance, min, max, 0.0)
    }

    fn probe_with_blend(
        index: i32,
        atlas: u16,
        importance: i32,
        min: Vec3,
        max: Vec3,
        blend_distance: f32,
    ) -> SpatialProbe {
        full_probe(index, atlas, importance, min, max, blend_distance, false)
    }

    fn skybox_probe(index: i32, atlas: u16, importance: i32, min: Vec3, max: Vec3) -> SpatialProbe {
        full_probe(index, atlas, importance, min, max, 0.0, true)
    }

    fn full_probe(
        index: i32,
        atlas: u16,
        importance: i32,
        min: Vec3,
        max: Vec3,
        blend_distance: f32,
        skybox: bool,
    ) -> SpatialProbe {
        let blend_distance = sanitized_blend_distance(blend_distance);
        let (influence_aabb_min, influence_aabb_max) = expanded_aabb(min, max, blend_distance);
        SpatialProbe {
            renderable_index: index,
            atlas_index: atlas,
            importance,
            aabb_min: Vec3A::from(min),
            aabb_max: Vec3A::from(max),
            influence_aabb_min,
            influence_aabb_max,
            center: Vec3A::from((min + max) * 0.5),
            volume: aabb_volume(min, max),
            skybox,
        }
    }

    #[test]
    fn higher_priority_overrides_lower_priority() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 0, Vec3::splat(-100.0), Vec3::splat(100.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.25), Vec3::splat(0.25)));

        assert_eq!(selection, ReflectionProbeDrawSelection::three(2, 1, 2));
    }

    #[test]
    fn frame_selection_returns_default_when_no_probe_hits() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(Vec::new());

        let draw = selection.select(space_id, (Vec3::splat(-1.0), Vec3::splat(1.0)));

        assert_eq!(draw, ReflectionProbeDrawSelection::default());
    }

    #[test]
    fn frame_selection_uses_probe_hit() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial([(
            space_id,
            probe(0, 3, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        )]);

        let draw = selection.select(space_id, (Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(
            draw,
            ReflectionProbeDrawSelection {
                first_atlas_index: 3,
                second_atlas_index: 0,
                fallback_atlas_index: 0,
                hit_count: 1,
            }
        );
    }

    #[test]
    fn blend_distance_selects_probe_outside_original_bounds() {
        let index = ReflectionProbeSpatialIndex::build(vec![probe_with_blend(
            0,
            1,
            1,
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            0.75,
        )]);

        let selection = index.select((Vec3::new(1.25, -0.25, -0.25), Vec3::new(1.5, 0.25, 0.25)));

        assert_eq!(selection, ReflectionProbeDrawSelection::one(1, 0));
    }

    #[test]
    fn blend_distance_stops_selecting_after_influence_bounds() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial([(
            space_id,
            probe_with_blend(0, 1, 1, Vec3::splat(-1.0), Vec3::splat(1.0), 0.25),
        )]);

        let draw = selection.select(
            space_id,
            (Vec3::new(1.5, -0.1, -0.1), Vec3::new(1.75, 0.1, 0.1)),
        );

        assert_eq!(draw, ReflectionProbeDrawSelection::default());
    }

    #[test]
    fn higher_priority_overrides_lower_priority_in_blend_fringe() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe_with_blend(0, 1, 0, Vec3::splat(-5.0), Vec3::splat(5.0), 0.0),
            probe_with_blend(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0), 1.0),
        ]);

        let selection = index.select((Vec3::new(1.2, -0.1, -0.1), Vec3::new(1.4, 0.1, 0.1)));

        assert_eq!(selection, ReflectionProbeDrawSelection::three(2, 1, 1));
    }

    #[test]
    fn same_importance_selects_two_by_intersection_volume() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(
                0,
                1,
                1,
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            probe(
                1,
                2,
                1,
                Vec3::new(0.0, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
            probe(
                2,
                3,
                1,
                Vec3::new(0.75, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
        ]);

        let selection = index.select((Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 2);
        assert_eq!(selection.second_atlas_index, 1);
        assert_eq!(selection.fallback_atlas_index, 0);
    }

    #[test]
    fn contained_same_importance_probe_selects_inner_in_higher_priority_when_object_fully_inside() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-10.0), Vec3::splat(10.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection, ReflectionProbeDrawSelection::two(2, 1, 2));
    }

    #[test]
    fn contained_same_importance_probe_blends_when_object_partially_leaves_inner() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-10.0), Vec3::splat(10.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 1);
        assert_eq!(selection.second_atlas_index, 2);
        assert_eq!(selection.fallback_atlas_index, 1);
    }

    #[test]
    fn identical_same_importance_probe_boxes_use_intersection_blend() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 2);
        assert_eq!(selection.second_atlas_index, 1);
        assert_eq!(selection.fallback_atlas_index, 2);
    }

    #[test]
    fn skybox_probe_terrible_candidate_but_used_as_fallback() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 0, Vec3::splat(-1.0), Vec3::splat(1.0)),
            probe(1, 2, 0, Vec3::splat(-1.0), Vec3::splat(1.0)),
            skybox_probe(2, 3, 0, Vec3::splat(-1000.0), Vec3::splat(1000.0)),
            skybox_probe(3, 4, 0, Vec3::splat(-10_000.0), Vec3::splat(10_000.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 2);
        assert_eq!(selection.second_atlas_index, 1);
        assert_eq!(selection.fallback_atlas_index, 3);
    }

    #[test]
    fn probes_of_different_importance_respect_hierarchy_and_have_fallback() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 2, Vec3::splat(-1.0), Vec3::splat(1.0)),
            probe(1, 2, 1, Vec3::splat(-3.0), Vec3::splat(3.0)),
            skybox_probe(2, 3, 0, Vec3::splat(-1000.0), Vec3::splat(1000.0)),
            skybox_probe(3, 4, 0, Vec3::splat(-10_000.0), Vec3::splat(10_000.0)),
        ]);

        let selection = index.select((Vec3::splat(-5.0), Vec3::splat(5.0)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 1);
        assert_eq!(selection.second_atlas_index, 2);
        assert_eq!(selection.fallback_atlas_index, 3);
    }

    #[test]
    fn bvh_matches_bruteforce_candidates() {
        let probes: Vec<_> = (0..32)
            .map(|i| {
                let x = i as f32 * 0.5;
                probe(
                    i,
                    (i + 1) as u16,
                    1,
                    Vec3::new(x, -1.0, -1.0),
                    Vec3::new(x + 1.0, 1.0, 1.0),
                )
            })
            .collect();
        let index = ReflectionProbeSpatialIndex::build(probes.clone());
        let object = (Vec3::new(4.2, -0.25, -0.25), Vec3::new(6.1, 0.25, 0.25));
        let selection = index.select(object);

        let mut brute: Vec<_> = probes
            .iter()
            .filter_map(|probe| {
                let v = intersection_volume_vec3a(
                    probe.aabb_min,
                    probe.aabb_max,
                    Vec3A::from(object.0),
                    Vec3A::from(object.1),
                );
                (v > 0.0).then_some((probe.atlas_index, v))
            })
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1));

        assert_eq!(selection.first_atlas_index, brute[0].0);
        assert_eq!(selection.second_atlas_index, brute[1].0);
    }
}
