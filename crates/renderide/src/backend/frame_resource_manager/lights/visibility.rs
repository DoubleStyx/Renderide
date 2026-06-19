//! Per-view CPU visibility filtering for resolved light influence volumes.

use glam::{Vec3, Vec3A};

use crate::scene::{RenderSpaceId, ResolvedLight, SceneLightRead, light_contributes};
use crate::shared::LightType;
use crate::world_mesh::culling::{WorldMeshCullInput, world_aabb_visible_for_cull};

use super::super::view_desc::FrameLightCullDesc;

const LIGHT_BVH_LEAF_SIZE: usize = 8;
const LIGHT_SPATIAL_LINEAR_LIMIT: usize = 64;
const SPOT_BOUNDS_AXIS_LEN_SQ_EPSILON: f32 = 1e-12;
const SPOT_BOUNDS_WIDE_COS_HALF: f32 = 0.5;
const LIGHT_AABB_EXTENT_EPSILON: f32 = 1e-7;

/// Filters resolved lights for one render space and optional view frustum.
pub(super) fn filter_resolved_lights_for_view<S>(
    scene: &S,
    space_id: RenderSpaceId,
    cull: Option<&FrameLightCullDesc>,
    lights: &mut Vec<ResolvedLight>,
) -> LightVisibilityStats
where
    S: SceneLightRead + ?Sized,
{
    filter_resolved_lights_for_view_with_stats(scene, space_id, cull, lights)
}

fn filter_resolved_lights_for_view_with_stats<S>(
    scene: &S,
    space_id: RenderSpaceId,
    cull: Option<&FrameLightCullDesc>,
    lights: &mut Vec<ResolvedLight>,
) -> LightVisibilityStats
where
    S: SceneLightRead + ?Sized,
{
    profiling::scope!("render::prepare_lights::filter_visibility");
    let mut stats = LightVisibilityStats {
        space_count: 1,
        lights_before_cull: lights.len(),
        ..LightVisibilityStats::default()
    };
    let Some(cull) = cull else {
        profiling::scope!("render::prepare_lights::filter_contributors");
        stats.cull_disabled_spaces = 1;
        lights.retain(light_contributes);
        stats.non_contributing_lights = stats.lights_before_cull.saturating_sub(lights.len());
        stats.lights_after_cull = lights.len();
        return stats;
    };

    let culling = WorldMeshCullInput {
        proj: cull.proj,
        host_camera: &cull.host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let mut keep = vec![false; lights.len()];
    let mut indexed = Vec::new();

    for (source_index, light) in lights.iter().enumerate() {
        if !light_contributes(light) {
            stats.non_contributing_lights = stats.non_contributing_lights.saturating_add(1);
            continue;
        }
        let Some(bounds) = light_influence_bounds(light) else {
            keep[source_index] = true;
            stats.fallback_lights = stats.fallback_lights.saturating_add(1);
            continue;
        };
        indexed.push(IndexedLight {
            source_index,
            aabb_min: bounds.min,
            aabb_max: bounds.max,
            center: (bounds.min + bounds.max) * 0.5,
        });
    }

    stats.indexed_lights = indexed.len();
    if !indexed.is_empty() {
        if indexed.len() <= LIGHT_SPATIAL_LINEAR_LIMIT {
            stats.linear_queries = stats.linear_queries.saturating_add(1);
            mark_visible_lights_linear(scene, space_id, &culling, &indexed, &mut keep, &mut stats);
        } else {
            stats.bvh_queries = stats.bvh_queries.saturating_add(1);
            let bvh = LightBvh::build(&indexed);
            bvh.mark_visible(scene, space_id, &culling, &mut keep, &mut stats);
        }
    }

    filter_lights_by_keep(lights, &keep);
    stats.lights_after_cull = lights.len();
    stats
}

fn mark_visible_lights_linear<S>(
    scene: &S,
    space_id: RenderSpaceId,
    culling: &WorldMeshCullInput<'_>,
    entries: &[IndexedLight],
    keep: &mut [bool],
    stats: &mut LightVisibilityStats,
) where
    S: SceneLightRead + ?Sized,
{
    profiling::scope!("render::prepare_lights::filter_visibility_linear");
    for entry in entries {
        stats.light_aabb_tests = stats.light_aabb_tests.saturating_add(1);
        if light_aabb_visible(scene, space_id, culling, entry.aabb_min, entry.aabb_max) {
            keep[entry.source_index] = true;
        } else {
            stats.rejected_lights = stats.rejected_lights.saturating_add(1);
        }
    }
}

fn filter_lights_by_keep(lights: &mut Vec<ResolvedLight>, keep: &[bool]) {
    let mut index = 0usize;
    lights.retain(|_| {
        let keep_light = keep.get(index).copied().unwrap_or(false);
        index = index.saturating_add(1);
        keep_light
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LightInfluenceBounds {
    min: Vec3A,
    max: Vec3A,
}

fn light_influence_bounds(light: &ResolvedLight) -> Option<LightInfluenceBounds> {
    match light.light_type {
        LightType::Directional => None,
        LightType::Point => point_light_bounds(light),
        LightType::Spot => spot_light_bounds(light).or_else(|| point_light_bounds(light)),
    }
}

fn point_light_bounds(light: &ResolvedLight) -> Option<LightInfluenceBounds> {
    if !light.world_position.is_finite() || !light.range.is_finite() || light.range <= 0.0 {
        return None;
    }
    let center = Vec3A::from(light.world_position);
    let radius = Vec3A::splat(light.range);
    valid_light_bounds(center - radius, center + radius)
}

fn spot_light_bounds(light: &ResolvedLight) -> Option<LightInfluenceBounds> {
    let axis = normalized_axis(light.world_direction)?;
    let half_angle = light.spot_angle.clamp(0.0, 180.0).to_radians() * 0.5;
    if !half_angle.is_finite() {
        return None;
    }
    let (sin_half, cos_half) = half_angle.sin_cos();
    if cos_half <= SPOT_BOUNDS_WIDE_COS_HALF || sin_half <= LIGHT_AABB_EXTENT_EPSILON {
        return None;
    }
    let range = light.range;
    if !range.is_finite() || range <= 0.0 || !light.world_position.is_finite() {
        return None;
    }

    let apex = light.world_position;
    let cap_center = apex + axis * (range * cos_half);
    let cap_radius = range * sin_half;
    if !cap_center.is_finite() || !cap_radius.is_finite() || cap_radius <= LIGHT_AABB_EXTENT_EPSILON
    {
        return None;
    }

    let radial_extent = Vec3::new(
        (1.0 - axis.x * axis.x).max(0.0).sqrt(),
        (1.0 - axis.y * axis.y).max(0.0).sqrt(),
        (1.0 - axis.z * axis.z).max(0.0).sqrt(),
    ) * cap_radius;
    let cap_min = cap_center - radial_extent;
    let cap_max = cap_center + radial_extent;
    valid_light_bounds(
        Vec3A::from(apex.min(cap_min)),
        Vec3A::from(apex.max(cap_max)),
    )
}

fn normalized_axis(axis: Vec3) -> Option<Vec3> {
    let len_sq = axis.length_squared();
    (len_sq.is_finite() && len_sq > SPOT_BOUNDS_AXIS_LEN_SQ_EPSILON)
        .then(|| axis * len_sq.sqrt().recip())
}

fn valid_light_bounds(min: Vec3A, max: Vec3A) -> Option<LightInfluenceBounds> {
    if !vec3a_is_finite(min) || !vec3a_is_finite(max) {
        return None;
    }
    let extent = max - min;
    if extent.cmplt(Vec3A::ZERO).any() || extent.max_element() <= LIGHT_AABB_EXTENT_EPSILON {
        return None;
    }
    Some(LightInfluenceBounds { min, max })
}

fn vec3a_is_finite(v: Vec3A) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LightVisibilityStats {
    /// Render spaces visited while resolving view light packs.
    pub(crate) space_count: usize,
    /// Render spaces prepared without a culling descriptor.
    pub(crate) cull_disabled_spaces: usize,
    /// Resolved lights before contribution and culling filters.
    pub(crate) lights_before_cull: usize,
    /// Resolved lights discarded because they cannot contribute visible direct lighting.
    pub(crate) non_contributing_lights: usize,
    /// Light influence volumes with finite bounds tested against the view.
    pub(crate) indexed_lights: usize,
    /// Lights kept conservatively because they could not be spatially indexed.
    pub(crate) fallback_lights: usize,
    /// Bounded light influence volumes rejected before clustered-light packing.
    pub(crate) rejected_lights: usize,
    /// Lights kept after contribution and frustum filters, before `MAX_LIGHTS` truncation.
    pub(crate) lights_after_cull: usize,
    /// Lights retained in packed GPU light arrays after `MAX_LIGHTS` truncation.
    pub(crate) packed_lights: usize,
    /// Lights kept by culling but dropped because the GPU light buffer reached `MAX_LIGHTS`.
    pub(crate) max_lights_culled: usize,
    /// Number of space-level light BVH traversals used this frame.
    pub(crate) bvh_queries: usize,
    /// Number of space-level linear light scans used this frame.
    pub(crate) linear_queries: usize,
    /// Per-light AABB frustum tests executed by linear runs or BVH leaves.
    pub(crate) light_aabb_tests: usize,
    /// BVH node AABB frustum tests executed before leaf light tests.
    pub(crate) bvh_node_tests: usize,
    /// BVH nodes rejected as a group before testing their contained lights.
    pub(crate) bvh_nodes_culled: usize,
}

impl LightVisibilityStats {
    /// Records the final GPU packing count for one prepared view light pack.
    pub(super) fn note_view_pack(&mut self, resolved_len: usize, packed_len: usize) {
        self.packed_lights = self.packed_lights.saturating_add(packed_len);
        self.max_lights_culled = self
            .max_lights_culled
            .saturating_add(resolved_len.saturating_sub(packed_len));
    }

    /// Adds another visibility sample into this one, saturating each field.
    pub(super) fn add(&mut self, other: Self) {
        self.space_count = self.space_count.saturating_add(other.space_count);
        self.cull_disabled_spaces = self
            .cull_disabled_spaces
            .saturating_add(other.cull_disabled_spaces);
        self.lights_before_cull = self
            .lights_before_cull
            .saturating_add(other.lights_before_cull);
        self.non_contributing_lights = self
            .non_contributing_lights
            .saturating_add(other.non_contributing_lights);
        self.indexed_lights = self.indexed_lights.saturating_add(other.indexed_lights);
        self.fallback_lights = self.fallback_lights.saturating_add(other.fallback_lights);
        self.rejected_lights = self.rejected_lights.saturating_add(other.rejected_lights);
        self.lights_after_cull = self
            .lights_after_cull
            .saturating_add(other.lights_after_cull);
        self.packed_lights = self.packed_lights.saturating_add(other.packed_lights);
        self.max_lights_culled = self
            .max_lights_culled
            .saturating_add(other.max_lights_culled);
        self.bvh_queries = self.bvh_queries.saturating_add(other.bvh_queries);
        self.linear_queries = self.linear_queries.saturating_add(other.linear_queries);
        self.light_aabb_tests = self.light_aabb_tests.saturating_add(other.light_aabb_tests);
        self.bvh_node_tests = self.bvh_node_tests.saturating_add(other.bvh_node_tests);
        self.bvh_nodes_culled = self.bvh_nodes_culled.saturating_add(other.bvh_nodes_culled);
    }
}

#[derive(Clone, Copy)]
struct IndexedLight {
    source_index: usize,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    center: Vec3A,
}

#[derive(Clone, Copy)]
struct LightBvhNode {
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    light_count: usize,
    start: usize,
    count: usize,
    left: usize,
    right: usize,
}

struct LightBvh {
    entries: Vec<IndexedLight>,
    order: Vec<usize>,
    nodes: Vec<LightBvhNode>,
    root: Option<usize>,
}

impl LightBvh {
    fn build(entries: &[IndexedLight]) -> Self {
        let mut bvh = Self {
            entries: entries.to_vec(),
            order: (0..entries.len()).collect(),
            nodes: Vec::new(),
            root: None,
        };
        if !entries.is_empty() {
            let mut order = std::mem::take(&mut bvh.order);
            let end = order.len();
            bvh.root = Some(bvh.build_node(&mut order, 0, end));
            bvh.order = order;
        }
        bvh
    }

    fn build_node(&mut self, order: &mut [usize], start: usize, end: usize) -> usize {
        let (aabb_min, aabb_max) = bounds_for_order(&self.entries, &order[start..end]);
        let count = end - start;
        let node_index = self.nodes.len();
        self.nodes.push(LightBvhNode {
            aabb_min,
            aabb_max,
            light_count: count,
            start,
            count: 0,
            left: 0,
            right: 0,
        });
        if count <= LIGHT_BVH_LEAF_SIZE {
            self.nodes[node_index].count = count;
            return node_index;
        }

        let axis = largest_axis(aabb_max - aabb_min);
        order[start..end].sort_unstable_by(|&a, &b| {
            axis_value(self.entries[a].center, axis)
                .total_cmp(&axis_value(self.entries[b].center, axis))
                .then_with(|| {
                    self.entries[a]
                        .source_index
                        .cmp(&self.entries[b].source_index)
                })
        });
        let mid = start + count / 2;
        let left = self.build_node(order, start, mid);
        let right = self.build_node(order, mid, end);
        self.nodes[node_index].left = left;
        self.nodes[node_index].right = right;
        node_index
    }

    fn mark_visible<S>(
        &self,
        scene: &S,
        space_id: RenderSpaceId,
        culling: &WorldMeshCullInput<'_>,
        keep: &mut [bool],
        stats: &mut LightVisibilityStats,
    ) where
        S: SceneLightRead + ?Sized,
    {
        profiling::scope!("render::prepare_lights::filter_visibility_bvh");
        if let Some(root) = self.root {
            self.mark_visible_node(root, scene, space_id, culling, keep, stats);
        }
    }

    fn mark_visible_node<S>(
        &self,
        node_index: usize,
        scene: &S,
        space_id: RenderSpaceId,
        culling: &WorldMeshCullInput<'_>,
        keep: &mut [bool],
        stats: &mut LightVisibilityStats,
    ) where
        S: SceneLightRead + ?Sized,
    {
        let node = self.nodes[node_index];
        stats.bvh_node_tests = stats.bvh_node_tests.saturating_add(1);
        if !light_aabb_visible(scene, space_id, culling, node.aabb_min, node.aabb_max) {
            stats.bvh_nodes_culled = stats.bvh_nodes_culled.saturating_add(1);
            stats.rejected_lights = stats.rejected_lights.saturating_add(node_light_count(node));
            return;
        }
        if node.count > 0 {
            for &entry_index in &self.order[node.start..node.start + node.count] {
                let entry = self.entries[entry_index];
                stats.light_aabb_tests = stats.light_aabb_tests.saturating_add(1);
                if light_aabb_visible(scene, space_id, culling, entry.aabb_min, entry.aabb_max) {
                    keep[entry.source_index] = true;
                } else {
                    stats.rejected_lights = stats.rejected_lights.saturating_add(1);
                }
            }
        } else {
            self.mark_visible_node(node.left, scene, space_id, culling, keep, stats);
            self.mark_visible_node(node.right, scene, space_id, culling, keep, stats);
        }
    }
}

fn node_light_count(node: LightBvhNode) -> usize {
    node.light_count
}

fn light_aabb_visible<S>(
    scene: &S,
    space_id: RenderSpaceId,
    culling: &WorldMeshCullInput<'_>,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
) -> bool
where
    S: SceneLightRead + ?Sized,
{
    world_aabb_visible_for_cull(
        scene,
        space_id,
        false,
        culling,
        Vec3::from(aabb_min),
        Vec3::from(aabb_max),
    )
}

fn bounds_for_order(entries: &[IndexedLight], order: &[usize]) -> (Vec3A, Vec3A) {
    let mut aabb_min = Vec3A::splat(f32::INFINITY);
    let mut aabb_max = Vec3A::splat(f32::NEG_INFINITY);
    for &entry_index in order {
        let entry = entries[entry_index];
        aabb_min = aabb_min.min(entry.aabb_min);
        aabb_max = aabb_max.max(entry.aabb_max);
    }
    (aabb_min, aabb_max)
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
    use glam::Mat4;

    use crate::camera::HostCameraFrame;
    use crate::scene::SceneCoordinator;
    use crate::shared::{RenderTransform, ShadowType};
    use crate::world_mesh::WorldMeshCullProjParams;

    use super::*;

    fn resolved_light(light_type: LightType, position: Vec3, range: f32) -> ResolvedLight {
        ResolvedLight {
            world_position: position,
            world_direction: Vec3::Z,
            world_right: Vec3::X,
            world_up: Vec3::Y,
            color: Vec3::ONE,
            intensity: 1.0,
            range,
            spot_angle: 45.0,
            light_type,
            shadow_type: ShadowType::None,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_map_resolution: 0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            cookie_texture_asset_id: -1,
        }
    }

    fn assert_vec3a_near(actual: Vec3A, expected: Vec3, epsilon: f32) {
        let delta = Vec3::from(actual) - expected;
        assert!(
            delta.abs().max_element() <= epsilon,
            "actual={:?} expected={:?} delta={:?}",
            Vec3::from(actual),
            expected,
            delta
        );
    }

    fn test_scene_and_cull(
        space_id: RenderSpaceId,
    ) -> (SceneCoordinator, HostCameraFrame, FrameLightCullDesc) {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
        let host_camera = HostCameraFrame::default();
        let cull = FrameLightCullDesc {
            host_camera,
            proj: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
        };
        (scene, host_camera, cull)
    }

    #[test]
    fn point_light_bounds_use_range_sphere_aabb() {
        let light = resolved_light(LightType::Point, Vec3::new(1.0, 2.0, 3.0), 4.0);
        let bounds = light_influence_bounds(&light).expect("point bounds");

        assert_vec3a_near(bounds.min, Vec3::new(-3.0, -2.0, -1.0), 1e-5);
        assert_vec3a_near(bounds.max, Vec3::new(5.0, 6.0, 7.0), 1e-5);
    }

    #[test]
    fn spot_light_bounds_use_stable_cone_aabb() {
        let mut light = resolved_light(LightType::Spot, Vec3::ZERO, 10.0);
        light.spot_angle = 60.0;
        let bounds = light_influence_bounds(&light).expect("spot bounds");

        assert_vec3a_near(bounds.min, Vec3::new(-5.0, -5.0, 0.0), 1e-4);
        assert_vec3a_near(
            bounds.max,
            Vec3::new(5.0, 5.0, 10.0 * 30.0f32.to_radians().cos()),
            1e-4,
        );
    }

    #[test]
    fn wide_and_degenerate_spot_bounds_fall_back_to_range_sphere() {
        let mut wide = resolved_light(LightType::Spot, Vec3::new(1.0, 0.0, 0.0), 3.0);
        wide.spot_angle = 150.0;
        let wide_bounds = light_influence_bounds(&wide).expect("wide fallback");
        assert_vec3a_near(wide_bounds.min, Vec3::new(-2.0, -3.0, -3.0), 1e-5);
        assert_vec3a_near(wide_bounds.max, Vec3::new(4.0, 3.0, 3.0), 1e-5);

        let mut degenerate = resolved_light(LightType::Spot, Vec3::ZERO, 2.0);
        degenerate.world_direction = Vec3::ZERO;
        let degenerate_bounds = light_influence_bounds(&degenerate).expect("degenerate fallback");
        assert_vec3a_near(degenerate_bounds.min, Vec3::splat(-2.0), 1e-5);
        assert_vec3a_near(degenerate_bounds.max, Vec3::splat(2.0), 1e-5);
    }

    #[test]
    fn linear_visibility_culls_off_frustum_lights_and_keeps_directional() {
        let space_id = RenderSpaceId(1);
        let (scene, _host_camera, cull) = test_scene_and_cull(space_id);
        let mut lights = vec![
            resolved_light(LightType::Point, Vec3::ZERO, 0.25),
            resolved_light(LightType::Point, Vec3::new(4.0, 0.0, 0.0), 0.25),
            resolved_light(LightType::Directional, Vec3::new(100.0, 0.0, 0.0), 0.0),
        ];

        let stats =
            filter_resolved_lights_for_view_with_stats(&scene, space_id, Some(&cull), &mut lights);

        assert_eq!(stats.space_count, 1);
        assert_eq!(stats.lights_before_cull, 3);
        assert_eq!(stats.bvh_queries, 0);
        assert_eq!(stats.linear_queries, 1);
        assert_eq!(stats.indexed_lights, 2);
        assert_eq!(stats.fallback_lights, 1);
        assert_eq!(stats.rejected_lights, 1);
        assert_eq!(stats.lights_after_cull, 2);
        assert_eq!(stats.light_aabb_tests, 2);
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].light_type, LightType::Point);
        assert_eq!(lights[1].light_type, LightType::Directional);
    }

    #[test]
    fn disabled_culling_reports_contributor_filter_counts() {
        let space_id = RenderSpaceId(5);
        let (scene, _host_camera, _cull) = test_scene_and_cull(space_id);
        let mut lights = vec![
            resolved_light(LightType::Point, Vec3::ZERO, 0.25),
            resolved_light(LightType::Point, Vec3::new(4.0, 0.0, 0.0), 0.25),
        ];
        lights[1].intensity = 0.0;

        let stats = filter_resolved_lights_for_view(&scene, space_id, None, &mut lights);

        assert_eq!(stats.space_count, 1);
        assert_eq!(stats.cull_disabled_spaces, 1);
        assert_eq!(stats.lights_before_cull, 2);
        assert_eq!(stats.non_contributing_lights, 1);
        assert_eq!(stats.rejected_lights, 0);
        assert_eq!(stats.lights_after_cull, 1);
        assert_eq!(stats.indexed_lights, 0);
        assert_eq!(stats.light_aabb_tests, 0);
        assert_eq!(lights.len(), 1);
    }

    #[test]
    fn bvh_visibility_activates_for_large_sets_and_preserves_order() {
        let space_id = RenderSpaceId(2);
        let (scene, _host_camera, cull) = test_scene_and_cull(space_id);
        let mut lights = (0..80)
            .map(|index| {
                let visible = index % 4 == 0 || index % 4 == 1;
                let x = if visible {
                    -0.5 + index as f32 * 0.001
                } else {
                    4.0
                };
                let mut light = resolved_light(LightType::Point, Vec3::new(x, 0.0, 0.0), 0.1);
                light.color = Vec3::splat(index as f32 + 1.0);
                light
            })
            .collect::<Vec<_>>();

        let stats =
            filter_resolved_lights_for_view_with_stats(&scene, space_id, Some(&cull), &mut lights);

        assert_eq!(stats.bvh_queries, 1);
        assert_eq!(stats.linear_queries, 0);
        assert_eq!(stats.indexed_lights, 80);
        assert_eq!(stats.rejected_lights, 40);
        assert!(stats.bvh_node_tests > 0);
        assert!(stats.light_aabb_tests <= 80);
        assert_eq!(lights.len(), 40);
        let kept_indices = lights
            .iter()
            .map(|light| light.color.x as usize - 1)
            .collect::<Vec<_>>();
        let expected = (0..80)
            .filter(|index| index % 4 == 0 || index % 4 == 1)
            .collect::<Vec<_>>();
        assert_eq!(kept_indices, expected);
    }

    #[test]
    fn small_visibility_sets_use_linear_fallback() {
        let space_id = RenderSpaceId(3);
        let (scene, _host_camera, cull) = test_scene_and_cull(space_id);
        let mut lights = (0..LIGHT_SPATIAL_LINEAR_LIMIT)
            .map(|_| resolved_light(LightType::Point, Vec3::ZERO, 0.1))
            .collect::<Vec<_>>();

        let stats =
            filter_resolved_lights_for_view_with_stats(&scene, space_id, Some(&cull), &mut lights);

        assert_eq!(stats.bvh_queries, 0);
        assert_eq!(stats.linear_queries, 1);
        assert_eq!(stats.light_aabb_tests, LIGHT_SPATIAL_LINEAR_LIMIT);
        assert_eq!(lights.len(), LIGHT_SPATIAL_LINEAR_LIMIT);
    }

    #[test]
    fn stereo_visibility_keeps_lights_visible_to_either_eye() {
        let space_id = RenderSpaceId(4);
        let (scene, host_camera, mut cull) = test_scene_and_cull(space_id);
        cull.host_camera = HostCameraFrame {
            vr_active: true,
            ..host_camera
        };
        cull.proj.vr_stereo = Some((
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(-2.0, 0.0, 0.0)),
        ));
        let mut lights = vec![resolved_light(
            LightType::Point,
            Vec3::new(2.0, 0.0, 0.0),
            0.1,
        )];

        let stats =
            filter_resolved_lights_for_view_with_stats(&scene, space_id, Some(&cull), &mut lights);

        assert_eq!(stats.rejected_lights, 0);
        assert_eq!(stats.lights_after_cull, 1);
        assert_eq!(lights.len(), 1);
    }
}
