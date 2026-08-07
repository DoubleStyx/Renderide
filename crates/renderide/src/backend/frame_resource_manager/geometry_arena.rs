//! Frame-global geometry-arena population planning.

use std::sync::Arc;

use hashbrown::HashSet;

use crate::shared::ShadowCastMode;
use crate::world_mesh::{WorldMeshDrawItem, WorldMeshDrawPlan, WorldMeshPhase};

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

/// Everything the arena plan output depends on, reduced to one hash plus the two counters.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct ArenaPlanContentSignature {
    fingerprint: u64,
    input_draws: usize,
    deformed_draws: usize,
}

/// Returns whether a draw sources its geometry from deform output rather than the static arena.
#[inline]
fn draw_is_deformed(item: &WorldMeshDrawItem) -> bool {
    item.skinned || item.world_space_deformed || item.blendshape_deformed
}

/// Hashes exactly the per-item facts that reach `mesh_asset_ids`.
///
/// Draw arrays are rebuilt every frame in a live world because [`WorldMeshDrawItem`] carries
/// `rigid_world_matrix` and `world_aabb`, so pointer identity fails as soon as anything moves even
/// though the mesh set is unchanged. Only the mesh id and the two skip predicates below are read by
/// the population walk, and none of them move with a transform. Hashing them costs one cheap pass
/// and lets the far more expensive dedup pass (a hash-set insert per draw) be skipped.
fn arena_plan_content_signature(
    key: &GeometryArenaFramePlanKey,
    shadow_plan: &ShadowFramePlan,
    shadow_rendering_layers: &[(usize, usize)],
) -> ArenaPlanContentSignature {
    use std::hash::{Hash, Hasher};

    profiling::scope!("world_mesh::geometry_arena_plan_signature");
    let mut hasher = ahash::AHasher::default();
    let mut input_draws = 0usize;
    let mut deformed_draws = 0usize;

    key.world_draws.len().hash(&mut hasher);
    for draws in &key.world_draws {
        draws.len().hash(&mut hasher);
        for item in draws.iter() {
            input_draws = input_draws.saturating_add(1);
            if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
                2u8.hash(&mut hasher);
                continue;
            }
            if draw_is_deformed(item) {
                deformed_draws = deformed_draws.saturating_add(1);
                1u8.hash(&mut hasher);
                continue;
            }
            0u8.hash(&mut hasher);
            item.mesh_asset_id.hash(&mut hasher);
        }
    }

    // Hash the selection itself: dropping a layer from the refresh set changes the output even when
    // every retained caster array is untouched.
    shadow_rendering_layers.hash(&mut hasher);
    for &(layer_idx, caster_set_index) in shadow_rendering_layers {
        let (Some(view), Some(caster_set)) = (
            shadow_plan.render_views.get(layer_idx),
            shadow_plan.caster_sets.get(caster_set_index),
        ) else {
            continue;
        };
        for phase in WorldMeshPhase::PRIMARY_FORWARD {
            let groups = view.groups(phase);
            groups.len().hash(&mut hasher);
            for group in groups {
                let Some(item) = caster_set.draws.get(group.representative_draw_idx) else {
                    continue;
                };
                input_draws = input_draws.saturating_add(1);
                if draw_is_deformed(item) {
                    deformed_draws = deformed_draws.saturating_add(1);
                    1u8.hash(&mut hasher);
                    continue;
                }
                0u8.hash(&mut hasher);
                item.mesh_asset_id.hash(&mut hasher);
            }
        }
    }

    ArenaPlanContentSignature {
        fingerprint: hasher.finish(),
        input_draws,
        deformed_draws,
    }
}

/// Retained arena-plan key plus allocation-free scratch for the miss path.
#[derive(Default)]
pub(super) struct GeometryArenaFramePlanCache {
    initialized: bool,
    key: GeometryArenaFramePlanKey,
    pending_key: GeometryArenaFramePlanKey,
    /// Content signature of the retained plan, used when pointer identity fails.
    signature: ArenaPlanContentSignature,
    shadow_caster_indices: Vec<usize>,
    /// `(render view index, caster set index)` for every shadow layer rendering this frame.
    shadow_rendering_layers: Vec<(usize, usize)>,
    seen_meshes: HashSet<i32>,
    seen_caster_sets: HashSet<usize>,
    #[cfg(test)]
    hits: u64,
    #[cfg(test)]
    content_hits: u64,
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

    // Pointer identity is gone but the mesh set usually is not: a moving renderer rewrites its draw
    // item without touching which mesh it draws. Pay one cheap hashing pass to find out before
    // paying the dedup pass.
    let signature = arena_plan_content_signature(
        &cache.pending_key,
        shadow_plan,
        &cache.shadow_rendering_layers,
    );
    if cache.initialized && signature == cache.signature {
        plan.input_draws = signature.input_draws;
        plan.deformed_draws = signature.deformed_draws;
        std::mem::swap(&mut cache.key, &mut cache.pending_key);
        cache.pending_key.clear();
        #[cfg(feature = "tracy")]
        tracy_client::plot!("world_mesh::geometry_arena_plan_cache_hit", 1.0);
        #[cfg(test)]
        {
            cache.content_hits = cache.content_hits.saturating_add(1);
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

    // Include shadow meshes only for layers refreshed this frame, and only for the caster groups
    // that layer actually draws. The shared caster set is collected without any culling, so it runs
    // an order of magnitude larger than what any layer renders (measured 6659 caster draws against
    // 669 camera draws). `visible_groups` is already culled against the layer's own shadow view, so
    // walking it instead is both cheaper and exact: a caster no layer draws needs no residency.
    // Instanced groups share one mesh, so the representative draw carries the whole group's id.
    for &(layer_idx, caster_set_index) in &cache.shadow_rendering_layers {
        let (Some(view), Some(caster_set)) = (
            shadow_plan.render_views.get(layer_idx),
            shadow_plan.caster_sets.get(caster_set_index),
        ) else {
            continue;
        };
        for phase in WorldMeshPhase::PRIMARY_FORWARD {
            for group in view.groups(phase) {
                let Some(item) = caster_set.draws.get(group.representative_draw_idx) else {
                    continue;
                };
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
    }

    std::mem::swap(&mut cache.key, &mut cache.pending_key);
    cache.pending_key.clear();
    cache.signature = signature;
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
    cache.shadow_rendering_layers.clear();
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
            .shadow_rendering_layers
            .push((layer_idx as usize, view.caster_set_index));
        cache
            .pending_key
            .shadow_source_draws
            .push(Arc::clone(&caster_set.source_draws));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::frame_resource_manager::ShadowRenderView;
    use crate::world_mesh::DrawGroup;
    use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshDrawPlan};
    use glam::{Mat4, Vec3};

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

    /// Shadow layer that draws the given caster-set draw indices, one group each.
    fn caster_layer_drawing(representative_draw_indices: &[usize]) -> ShadowRenderView {
        let mut view = ShadowRenderView::for_tests(
            crate::gpu::SHADOW_VIEW_KIND_DIRECTIONAL,
            Mat4::IDENTITY,
            Vec3::ZERO,
            1.0,
            0.0,
        );
        let groups = representative_draw_indices
            .iter()
            .map(|&idx| DrawGroup {
                representative_draw_idx: idx,
                instance_range: idx as u32..idx as u32 + 1,
                material_packet_idx: 0,
            })
            .collect();
        view.set_visible_groups_for_tests(WorldMeshPhase::ForwardOpaque, groups);
        view
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

        use crate::backend::frame_resource_manager::ShadowCasterSet;

        let visible = plan(vec![draw(10)]);
        let mut deformed = draw(30);
        deformed.skinned = true;
        let shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                first_dynamic_instance: u32::MAX,
                source_draws: Arc::from([draw(10), draw(20), deformed.clone()]),
                draws: Arc::from([draw(10), draw(20), deformed]),
                instance_plan: Default::default(),
                slab_slot_offset: 0,
            }],
            render_views: vec![caster_layer_drawing(&[0, 1, 2])],
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

        use crate::backend::frame_resource_manager::ShadowCasterSet;

        let shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                first_dynamic_instance: u32::MAX,
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
    fn frame_plan_reuses_mesh_set_when_only_transforms_moved() {
        // a renderer that moves rewrites its draw item, so the array is a fresh Arc with a fresh
        // matrix. the mesh set is identical, so population must not run again.
        let mut moved = draw(10);
        moved.rigid_world_matrix = Some(Mat4::from_translation(Vec3::splat(5.0)));
        let first = plan(vec![draw(10), draw(20)]);
        let after_motion = plan(vec![moved, draw(20)]);
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        assert!(!collect_geometry_arena_frame_plan(
            [&first],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));
        assert!(collect_geometry_arena_frame_plan(
            [&after_motion],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));

        assert_eq!(frame.mesh_asset_ids, [10, 20]);
        assert_eq!(frame.input_draws, 2);
        assert_eq!(cache.misses, 1);
        assert_eq!(
            cache.hits, 0,
            "pointer identity is gone once the array is rebuilt"
        );
        assert_eq!(cache.content_hits, 1);
    }

    #[test]
    fn frame_plan_content_hit_reports_the_same_counters_as_a_rebuild() {
        let mut skinned = draw(30);
        skinned.skinned = true;
        let mut shadow_only = draw(40);
        shadow_only.shadow_cast_mode = ShadowCastMode::ShadowOnly;
        let items = || vec![draw(10), skinned.clone(), shadow_only.clone(), draw(20)];
        let mut rebuilt = GeometryArenaFramePlan::default();
        let mut rebuilt_cache = GeometryArenaFramePlanCache::default();
        collect_geometry_arena_frame_plan(
            [&plan(items())],
            &ShadowFramePlan::default(),
            &mut rebuilt,
            &mut rebuilt_cache,
        );

        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();
        collect_geometry_arena_frame_plan(
            [&plan(items())],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        );
        assert!(collect_geometry_arena_frame_plan(
            [&plan(items())],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));

        assert_eq!(cache.content_hits, 1);
        assert_eq!(frame.mesh_asset_ids, rebuilt.mesh_asset_ids);
        assert_eq!(frame.input_draws, rebuilt.input_draws);
        assert_eq!(frame.deformed_draws, rebuilt.deformed_draws);
    }

    #[test]
    fn frame_plan_content_signature_separates_skip_reasons_from_a_real_mesh() {
        // a deformed draw and a shadow-only draw both contribute no mesh id. they must still hash
        // differently from each other and from a drawn mesh, or a swap between them goes unnoticed.
        let mut skinned = draw(10);
        skinned.skinned = true;
        let mut shadow_only = draw(10);
        shadow_only.shadow_cast_mode = ShadowCastMode::ShadowOnly;
        let mut frame = GeometryArenaFramePlan::default();
        let mut cache = GeometryArenaFramePlanCache::default();

        collect_geometry_arena_frame_plan(
            [&plan(vec![skinned])],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        );
        assert!(!collect_geometry_arena_frame_plan(
            [&plan(vec![shadow_only])],
            &ShadowFramePlan::default(),
            &mut frame,
            &mut cache,
        ));
        assert_eq!(cache.content_hits, 0);
        assert_eq!(frame.deformed_draws, 0);
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
        use crate::backend::frame_resource_manager::ShadowCasterSet;

        let source: Arc<[WorldMeshDrawItem]> = Arc::from([draw(20)]);
        let mut shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                first_dynamic_instance: u32::MAX,
                source_draws: Arc::clone(&source),
                draws: Arc::from([draw(20)]),
                instance_plan: Default::default(),
                slab_slot_offset: 0,
            }],
            render_views: vec![caster_layer_drawing(&[0])],
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
        use crate::backend::frame_resource_manager::ShadowCasterSet;

        let source: Arc<[WorldMeshDrawItem]> = Arc::from([draw(20)]);
        let mut shadow_plan = ShadowFramePlan {
            caster_sets: vec![ShadowCasterSet {
                first_dynamic_instance: u32::MAX,
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
