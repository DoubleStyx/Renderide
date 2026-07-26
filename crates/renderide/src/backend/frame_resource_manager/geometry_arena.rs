//! Frame-global geometry-arena population planning.

use std::sync::Arc;

use hashbrown::HashSet;

use crate::shared::ShadowCastMode;
use crate::world_mesh::{WorldMeshDrawItem, WorldMeshDrawPlan};

use super::manager::{FrameResourceManager, GeometryArenaFramePlan};
use super::shadows::ShadowFramePlan;

/// Retained draw-array identities for one arena frame plan.
#[derive(Default)]
struct GeometryArenaFramePlanKey {
    world_draws: Vec<Arc<[WorldMeshDrawItem]>>,
    shadow_source_draws: Vec<Arc<[WorldMeshDrawItem]>>,
}

impl GeometryArenaFramePlanKey {
    fn clear(&mut self) {
        self.world_draws.clear();
        self.shadow_source_draws.clear();
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        arc_slice_identities_match(&self.world_draws, &other.world_draws)
            && arc_slice_identities_match(&self.shadow_source_draws, &other.shadow_source_draws)
    }
}

fn arc_slice_identities_match<T>(left: &[Arc<[T]>], right: &[Arc<[T]>]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| Arc::ptr_eq(left, right))
}

/// Retained arena-plan key plus allocation-free scratch for the miss path.
#[derive(Default)]
pub(super) struct GeometryArenaFramePlanCache {
    initialized: bool,
    key: GeometryArenaFramePlanKey,
    pending_key: GeometryArenaFramePlanKey,
    shadow_caster_indices: Vec<usize>,
    seen_meshes: HashSet<i32>,
    seen_caster_sets: HashSet<usize>,
    #[cfg(test)]
    hits: u64,
    #[cfg(test)]
    misses: u64,
}

impl FrameResourceManager {
    /// Collects static meshes needed by this graph's views and refreshed shadow layers.
    pub(crate) fn prepare_geometry_arena_for_views<'a, I>(&mut self, views: I)
    where
        I: IntoIterator<Item = &'a WorldMeshDrawPlan>,
    {
        collect_geometry_arena_frame_plan(
            views,
            &self.shadow_frame,
            &mut self.geometry_arena_frame,
            &mut self.geometry_arena_frame_cache,
        );
    }

    /// Current frame-global geometry-arena population plan.
    pub(super) fn geometry_arena_frame_plan(&self) -> &GeometryArenaFramePlan {
        &self.geometry_arena_frame
    }
}

fn collect_geometry_arena_frame_plan<'a, I>(
    views: I,
    shadow_plan: &ShadowFramePlan,
    plan: &mut GeometryArenaFramePlan,
    cache: &mut GeometryArenaFramePlanCache,
) -> bool
where
    I: IntoIterator<Item = &'a WorldMeshDrawPlan>,
{
    build_pending_plan_key(views, shadow_plan, cache);
    if cache.initialized && cache.key.ptr_eq(&cache.pending_key) {
        cache.pending_key.clear();
        #[cfg(feature = "tracy")]
        tracy_client::plot!("world_mesh::geometry_arena_plan_cache_hit", 1.0);
        #[cfg(test)]
        {
            cache.hits = cache.hits.saturating_add(1);
        }
        return true;
    }

    plan.mesh_asset_ids.clear();
    plan.input_draws = 0;
    plan.deformed_draws = 0;

    cache.seen_meshes.clear();
    for draws in &cache.pending_key.world_draws {
        for item in draws.iter() {
            plan.input_draws = plan.input_draws.saturating_add(1);
            if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
                continue;
            }
            if item.skinned || item.world_space_deformed || item.blendshape_deformed {
                plan.deformed_draws = plan.deformed_draws.saturating_add(1);
                continue;
            }
            if cache.seen_meshes.insert(item.mesh_asset_id) {
                plan.mesh_asset_ids.push(item.mesh_asset_id);
            }
        }
    }

    // Include shadow-only meshes only for layers refreshed this frame.
    for &caster_set_index in &cache.shadow_caster_indices {
        let Some(caster_set) = shadow_plan.caster_sets.get(caster_set_index) else {
            continue;
        };
        for item in caster_set.draws.iter() {
            plan.input_draws = plan.input_draws.saturating_add(1);
            if item.skinned || item.world_space_deformed || item.blendshape_deformed {
                plan.deformed_draws = plan.deformed_draws.saturating_add(1);
                continue;
            }
            if cache.seen_meshes.insert(item.mesh_asset_id) {
                plan.mesh_asset_ids.push(item.mesh_asset_id);
            }
        }
    }

    std::mem::swap(&mut cache.key, &mut cache.pending_key);
    cache.pending_key.clear();
    cache.initialized = true;
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("world_mesh::geometry_arena_plan_cache_hit", 0.0);
        tracy_client::plot!(
            "world_mesh::geometry_arena_plan_scanned_draws",
            plan.input_draws as f64
        );
    }
    #[cfg(test)]
    {
        cache.misses = cache.misses.saturating_add(1);
    }
    false
}

/// Captures draw-array identities without scanning their rows.
fn build_pending_plan_key<'a, I>(
    views: I,
    shadow_plan: &ShadowFramePlan,
    cache: &mut GeometryArenaFramePlanCache,
) where
    I: IntoIterator<Item = &'a WorldMeshDrawPlan>,
{
    cache.pending_key.clear();
    for view in views {
        let Some(collection) = view.as_prefetched() else {
            continue;
        };
        cache
            .pending_key
            .world_draws
            .push(Arc::clone(&collection.items));
    }

    cache.shadow_caster_indices.clear();
    cache.seen_caster_sets.clear();
    for &layer_idx in &shadow_plan.rendering_layer_indices {
        let Some(view) = shadow_plan.render_views.get(layer_idx as usize) else {
            continue;
        };
        if !cache.seen_caster_sets.insert(view.caster_set_index) {
            continue;
        }
        let Some(caster_set) = shadow_plan.caster_sets.get(view.caster_set_index) else {
            continue;
        };
        cache.shadow_caster_indices.push(view.caster_set_index);
        cache
            .pending_key
            .shadow_source_draws
            .push(Arc::clone(&caster_set.source_draws));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshDrawPlan};

    fn draw(mesh_asset_id: i32) -> WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id,
            node_id: mesh_asset_id,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        })
    }

    fn plan(items: Vec<WorldMeshDrawItem>) -> WorldMeshDrawPlan {
        plan_from_items(items.into())
    }

    fn plan_from_items(items: Arc<[WorldMeshDrawItem]>) -> WorldMeshDrawPlan {
        WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws::new(
            WorldMeshDrawCollection {
                draws_pre_cull: items.len(),
                items,
                draws_culled: 0,
                draws_hi_z_culled: 0,
                visibility: Default::default(),
                arrangement: Default::default(),
            },
            None,
        )))
    }

    #[test]
    fn frame_plan_deduplicates_static_meshes_across_views() {
        let first = plan(vec![draw(10), draw(20), draw(10)]);
        let second = plan(vec![draw(20), draw(30)]);
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        collect_geometry_arena_frame_plan(
            [&first, &second],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        );

        assert_eq!(frame.mesh_asset_ids, [10, 20, 30]);
        assert_eq!(frame.input_draws, 5);
        assert_eq!(frame.deformed_draws, 0);
    }

    #[test]
    fn frame_plan_excludes_deformed_and_shadow_only_draws() {
        let mut skinned = draw(20);
        skinned.skinned = true;
        let mut world_deformed = draw(30);
        world_deformed.world_space_deformed = true;
        let mut blendshape = draw(40);
        blendshape.blendshape_deformed = true;
        let mut shadow_only = draw(50);
        shadow_only.shadow_cast_mode = ShadowCastMode::ShadowOnly;
        let visible = plan(vec![
            draw(10),
            skinned,
            world_deformed,
            blendshape,
            shadow_only,
        ]);
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        collect_geometry_arena_frame_plan(
            [&visible],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        );

        assert_eq!(frame.mesh_asset_ids, [10]);
        assert_eq!(frame.input_draws, 5);
        assert_eq!(frame.deformed_draws, 3);
    }

    #[test]
    fn frame_plan_includes_static_casters_for_rendering_shadow_layers() {
        use std::sync::Arc;

        use glam::{Mat4, Vec3};

        use crate::backend::frame_resource_manager::{ShadowCasterSet, ShadowRenderView};

        let visible = plan(vec![draw(10)]);
        let mut deformed = draw(30);
        deformed.skinned = true;
        let shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                source_draws: Arc::from([draw(10), draw(20), deformed.clone()]),
                draws: Arc::from([draw(10), draw(20), deformed]),
                instance_plan: Default::default(),
                slab_slot_offset: 0,
            }],
            render_views: vec![ShadowRenderView::for_tests(
                crate::gpu::SHADOW_VIEW_KIND_DIRECTIONAL,
                Mat4::IDENTITY,
                Vec3::ZERO,
                1.0,
                0.0,
            )],
            rendering_layer_indices: vec![0],
            ..ShadowFramePlan::default()
        };
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        collect_geometry_arena_frame_plan([&visible], &shadow_plan, &mut frame, &mut cache);

        assert_eq!(frame.mesh_asset_ids, [10, 20]);
        assert_eq!(frame.input_draws, 4);
        assert_eq!(frame.deformed_draws, 1);
    }

    #[test]
    fn frame_plan_ignores_casters_for_cached_shadow_layers() {
        use std::sync::Arc;

        use glam::{Mat4, Vec3};

        use crate::backend::frame_resource_manager::{ShadowCasterSet, ShadowRenderView};

        let shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                source_draws: Arc::from([draw(20)]),
                draws: Arc::from([draw(20)]),
                instance_plan: Default::default(),
                slab_slot_offset: 0,
            }],
            render_views: vec![ShadowRenderView::for_tests(
                crate::gpu::SHADOW_VIEW_KIND_DIRECTIONAL,
                Mat4::IDENTITY,
                Vec3::ZERO,
                1.0,
                0.0,
            )],
            rendering_layer_indices: Vec::new(),
            ..ShadowFramePlan::default()
        };
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        collect_geometry_arena_frame_plan(std::iter::empty(), &shadow_plan, &mut frame, &mut cache);

        assert!(frame.mesh_asset_ids.is_empty());
        assert_eq!(frame.input_draws, 0);
    }

    #[test]
    fn frame_plan_reuses_shared_draw_array_across_new_view_packets() {
        let items: Arc<[WorldMeshDrawItem]> = Arc::from([draw(10), draw(20), draw(10)]);
        let first = plan_from_items(Arc::clone(&items));
        // Camera/Hi-Z refreshes replace the outer prefetched packet while retaining this immutable
        // item array. Arena planning should therefore stay a cache hit.
        let refreshed = plan_from_items(Arc::clone(&items));
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        assert!(!collect_geometry_arena_frame_plan(
            [&first],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));
        assert!(collect_geometry_arena_frame_plan(
            [&refreshed],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));

        assert_eq!(frame.mesh_asset_ids, [10, 20]);
        assert_eq!(frame.input_draws, 3);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn frame_plan_invalidates_for_new_draw_array_identity() {
        let first = plan(vec![draw(10)]);
        let replacement = plan(vec![draw(20)]);
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        assert!(!collect_geometry_arena_frame_plan(
            [&first],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));
        assert!(!collect_geometry_arena_frame_plan(
            [&replacement],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));

        assert_eq!(frame.mesh_asset_ids, [20]);
        assert_eq!(cache.misses, 2);
        assert_eq!(cache.hits, 0);
    }

    #[test]
    fn frame_plan_reuses_shadow_source_identity_when_filtered_packet_is_rebuilt() {
        use glam::{Mat4, Vec3};

        use crate::backend::frame_resource_manager::{ShadowCasterSet, ShadowRenderView};

        let source: Arc<[WorldMeshDrawItem]> = Arc::from([draw(20)]);
        let mut shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                source_draws: Arc::clone(&source),
                draws: Arc::from([draw(20)]),
                instance_plan: Default::default(),
                slab_slot_offset: 0,
            }],
            render_views: vec![ShadowRenderView::for_tests(
                crate::gpu::SHADOW_VIEW_KIND_DIRECTIONAL,
                Mat4::IDENTITY,
                Vec3::ZERO,
                1.0,
                0.0,
            )],
            rendering_layer_indices: vec![0],
            ..ShadowFramePlan::default()
        };
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        assert!(!collect_geometry_arena_frame_plan(
            std::iter::empty(),
            &shadow_plan,
            &mut frame,
            &mut cache,
        ));
        shadow_plan.caster_sets[0].draws = Arc::from([draw(20)]);
        assert!(collect_geometry_arena_frame_plan(
            std::iter::empty(),
            &shadow_plan,
            &mut frame,
            &mut cache,
        ));

        assert_eq!(frame.mesh_asset_ids, [20]);
        assert_eq!(frame.input_draws, 1);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn frame_plan_invalidates_when_shadow_refresh_selection_changes() {
        use glam::{Mat4, Vec3};

        use crate::backend::frame_resource_manager::{ShadowCasterSet, ShadowRenderView};

        let source: Arc<[WorldMeshDrawItem]> = Arc::from([draw(20)]);
        let mut shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                source_draws: source,
                draws: Arc::from([draw(20)]),
                instance_plan: Default::default(),
                slab_slot_offset: 0,
            }],
            render_views: vec![ShadowRenderView::for_tests(
                crate::gpu::SHADOW_VIEW_KIND_DIRECTIONAL,
                Mat4::IDENTITY,
                Vec3::ZERO,
                1.0,
                0.0,
            )],
            rendering_layer_indices: vec![0],
            ..ShadowFramePlan::default()
        };
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        assert!(!collect_geometry_arena_frame_plan(
            std::iter::empty(),
            &shadow_plan,
            &mut frame,
            &mut cache,
        ));
        shadow_plan.rendering_layer_indices.clear();
        assert!(!collect_geometry_arena_frame_plan(
            std::iter::empty(),
            &shadow_plan,
            &mut frame,
            &mut cache,
        ));

        assert!(frame.mesh_asset_ids.is_empty());
        assert_eq!(frame.input_draws, 0);
        assert_eq!(cache.misses, 2);
    }
}
