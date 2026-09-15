use crate::materials::ShaderPermutation;
use crate::materials::host_data::{
    MaterialDictionary, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::{
    EmbeddedTangentFallbackMode, MaterialPipelinePropertyIds, MaterialRouter, RasterFrontFace,
    RasterPipelineKind, RasterPrimitiveTopology,
};

use super::{FrameMaterialBatchCache, PendingMaterialResolve, TouchOutcome};
use crate::world_mesh::materials::{
    MaterialResolveCtx, batch_key_for_slot_cached, compute_batch_key_hash,
};

fn make_test_deps() -> (MaterialPropertyStore, MaterialRouter, PropertyIdRegistry) {
    let store = MaterialPropertyStore::new();
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let reg = PropertyIdRegistry::new();
    (store, router, reg)
}

/// Directly exercise the private `touch_or_refresh` path so we can unit-test generation
/// invalidation without setting up a `SceneCoordinator`. `refresh_for_frame` is the
/// production entry; it wraps the same per-key logic over a scene walk.
fn touch(
    cache: &mut FrameMaterialBatchCache,
    mat: i32,
    pb: Option<i32>,
    ctx: MaterialResolveCtx<'_>,
    frame: u64,
) -> TouchOutcome {
    cache.frame_counter = frame;
    let rgen = ctx.router.generation();
    cache.touch_or_refresh(mat, pb, ctx, rgen, frame)
}

/// Helper that bundles the four handles into a [`MaterialResolveCtx`] for a test call site.
fn make_ctx<'a>(
    dict: &'a MaterialDictionary<'a>,
    router: &'a MaterialRouter,
    ids: &'a MaterialPipelinePropertyIds,
    perm: ShaderPermutation,
) -> MaterialResolveCtx<'a> {
    MaterialResolveCtx {
        dict,
        router,
        pipeline_property_ids: ids,
        shader_perm: perm,
    }
}

#[test]
fn first_touch_resolves_and_inserts_entry() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    touch(
        &mut cache,
        42,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        1,
    );
    assert!(cache.get(42, None).is_some());
    // Unknown material id -> shader id -1.
    assert_eq!(cache.get(42, None).unwrap().shader_asset_id, -1);
}

#[test]
fn cached_batch_hash_matches_every_draw_local_raster_variant() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let ctx = make_ctx(&dict, &router, &ids, ShaderPermutation::default());
    let mut cache = FrameMaterialBatchCache::new();
    touch(&mut cache, 42, Some(7), ctx, 1);

    for skinned in [false, true] {
        for front_face in [
            RasterFrontFace::Clockwise,
            RasterFrontFace::CounterClockwise,
        ] {
            for primitive_topology in [
                RasterPrimitiveTopology::TriangleList,
                RasterPrimitiveTopology::PointList,
            ] {
                let (batch_key, _, cached_hash) = batch_key_for_slot_cached(
                    42,
                    Some(7),
                    skinned,
                    front_face,
                    primitive_topology,
                    &cache,
                    ctx,
                );
                assert_eq!(cached_hash, Some(compute_batch_key_hash(&batch_key)));
            }
        }
    }
}

#[test]
fn cached_pbsvoronoicrystal_batch_keeps_generated_tangent_policy() {
    let (mut store, mut router, reg) = make_test_deps();
    store.set_shader_asset_for_material(7, 99);
    router.set_shader_pipeline(
        99,
        RasterPipelineKind::EmbeddedStem(std::sync::Arc::from("pbsvoronoicrystal_default")),
    );
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();

    touch(
        &mut cache,
        7,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation::default()),
        1,
    );

    let resolved = cache.get(7, None).expect("cached material batch");
    assert!(resolved.embedded_needs_tangent);
    assert_eq!(
        resolved.embedded_tangent_fallback_mode,
        EmbeddedTangentFallbackMode::GenerateMissing
    );
}

#[test]
fn unchanged_entry_is_reused_without_reresolve() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    touch(
        &mut cache,
        1,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        1,
    );
    let before = cache.entries.get(&(1, None)).unwrap().clone();
    touch(
        &mut cache,
        1,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        2,
    );
    let after = cache.entries.get(&(1, None)).unwrap();
    assert_eq!(before.material_gen, after.material_gen);
    assert_eq!(before.router_gen, after.router_gen);
    // last_used_frame advanced but generations did not -- confirms no re-resolve.
    assert_eq!(after.last_used_frame, 2);
}

#[test]
fn touch_or_refresh_reports_miss_hit_and_stale() {
    let (mut store, router, reg) = make_test_deps();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    {
        let dict = MaterialDictionary::new(&store);
        assert_eq!(
            touch(
                &mut cache,
                1,
                None,
                make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
                1,
            ),
            TouchOutcome::Miss
        );
        assert_eq!(
            touch(
                &mut cache,
                1,
                None,
                make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
                2,
            ),
            TouchOutcome::Hit
        );
    }

    store.set_material(1, 7, MaterialPropertyValue::Float(0.5));
    let dict = MaterialDictionary::new(&store);
    assert_eq!(
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            3,
        ),
        TouchOutcome::Stale
    );
}

#[test]
fn staged_prepared_resolves_apply_entries_in_key_order() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let ctx = make_ctx(&dict, &router, &ids, ShaderPermutation(2));
    let pending = vec![
        PendingMaterialResolve {
            material_asset_id: 11,
            property_block_id: None,
            material_gen: 1,
            property_block_gen: 0,
        },
        PendingMaterialResolve {
            material_asset_id: 12,
            property_block_id: Some(4),
            material_gen: 2,
            property_block_gen: 3,
        },
    ];

    let updates = super::resolve_pending_material_batches(pending, ctx, router.generation(), 9);
    let mut cache = FrameMaterialBatchCache::new();
    cache.apply_resolved_material_updates(updates);

    assert_eq!(cache.len(), 2);
    assert!(cache.get(11, None).is_some());
    assert!(cache.get(12, Some(4)).is_some());
    assert_eq!(
        cache.entries.get(&(11, None)).map(|entry| (
            entry.material_gen,
            entry.property_block_gen,
            entry.shader_perm,
            entry.last_used_frame
        )),
        Some((1, 0, ShaderPermutation(2), 9))
    );
    assert_eq!(
        cache.entries.get(&(12, Some(4))).map(|entry| (
            entry.material_gen,
            entry.property_block_gen,
            entry.shader_perm,
            entry.last_used_frame
        )),
        Some((2, 3, ShaderPermutation(2), 9))
    );
}

#[test]
fn prepared_key_classification_stamps_hits_without_pending_resolves() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let ctx = make_ctx(&dict, &router, &ids, ShaderPermutation(0));
    let mut cache = FrameMaterialBatchCache::new();
    let keys = (0..super::MATERIAL_CLASSIFY_PARALLEL_MIN_KEYS)
        .map(|index| (index as i32, None))
        .collect::<Vec<_>>();
    for &(material_asset_id, property_block_id) in &keys {
        touch(&mut cache, material_asset_id, property_block_id, ctx, 1);
    }

    let (stats, pending) = cache.classify_prepared_keys(&keys, ctx, router.generation(), 2);

    assert_eq!(stats.hits, keys.len());
    assert_eq!(stats.stale, 0);
    assert_eq!(stats.misses, 0);
    assert!(pending.is_empty());
    assert!(keys.iter().all(|key| {
        cache
            .entries
            .get(key)
            .is_some_and(|entry| entry.last_used_frame == 2)
    }));
}

#[test]
fn material_mutation_invalidates_entry() {
    let (mut store, router, reg) = make_test_deps();
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    {
        let dict = MaterialDictionary::new(&store);
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
    };
    let gen_before = cache.entries.get(&(1, None)).unwrap().material_gen;
    store.set_material(1, 7, MaterialPropertyValue::Float(0.25));
    {
        let dict = MaterialDictionary::new(&store);
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            2,
        );
    };
    let gen_after = cache.entries.get(&(1, None)).unwrap().material_gen;
    assert_ne!(gen_before, gen_after);
}

#[test]
fn router_mutation_invalidates_entry() {
    let (store, mut router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    touch(
        &mut cache,
        1,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        1,
    );
    let rgen_before = cache.entries.get(&(1, None)).unwrap().router_gen;
    router.set_shader_pipeline(
        7,
        RasterPipelineKind::EmbeddedStem(std::sync::Arc::from("x_default")),
    );
    touch(
        &mut cache,
        1,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        2,
    );
    let rgen_after = cache.entries.get(&(1, None)).unwrap().router_gen;
    assert_ne!(rgen_before, rgen_after);
}

#[test]
fn shader_perm_mismatch_triggers_reresolve() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    touch(
        &mut cache,
        1,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        1,
    );
    touch(
        &mut cache,
        1,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(1)),
        2,
    );
    assert_eq!(
        cache.entries.get(&(1, None)).unwrap().shader_perm,
        ShaderPermutation(1)
    );
}

#[test]
fn prepared_fast_path_requires_matching_live_set_signature() {
    let (store, router, _reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let mut cache = FrameMaterialBatchCache::new();
    let router_gen = router.generation();
    let dict_gen = dict.global_generation();
    let shader_perm = ShaderPermutation(0);

    cache.record_refresh_snapshot(router_gen, dict_gen, shader_perm, Some(11));

    assert!(cache.try_prepared_fast_path_skip(router_gen, dict_gen, shader_perm, 11));
    assert!(!cache.try_prepared_fast_path_skip(router_gen, dict_gen, shader_perm, 12));

    cache.record_refresh_snapshot(router_gen, dict_gen, shader_perm, None);

    assert!(!cache.try_prepared_fast_path_skip(router_gen, dict_gen, shader_perm, 11));
}

#[test]
fn property_block_id_produces_separate_entry() {
    let (store, router, reg) = make_test_deps();
    let dict = MaterialDictionary::new(&store);
    let ids = MaterialPipelinePropertyIds::new(&reg);
    let mut cache = FrameMaterialBatchCache::new();
    touch(
        &mut cache,
        10,
        None,
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        1,
    );
    touch(
        &mut cache,
        10,
        Some(99),
        make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
        1,
    );
    assert_eq!(cache.len(), 2);
}
