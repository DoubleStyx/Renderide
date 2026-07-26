use super::*;

#[test]
fn closed_space_filter_matches_runtime_cubemap_space() {
    let mut spaces = HashSet::new();
    spaces.insert(RenderSpaceId(7));

    assert!(specular_ibl_key_matches_closed_spaces(
        &SkyboxIblKey::RuntimeCubemap {
            render_space_id: 7,
            renderable_index: 0,
            generation: 1,
            mip_levels: 1,
            storage_v_inverted: true,
            face_size: 128,
        },
        &spaces,
    ));
}

#[test]
fn closed_space_filter_does_not_match_uploaded_asset_keys() {
    let mut spaces = HashSet::new();
    spaces.insert(RenderSpaceId(7));

    assert!(!specular_ibl_key_matches_closed_spaces(
        &SkyboxIblKey::Cubemap {
            material_asset_id: 21,
            material_generation: 1,
            route_hash: 99,
            asset_id: 7,
            allocation_generation: 1,
            mip_levels_resident: 1,
            content_generation: 1,
            storage_v_inverted: false,
            face_size: 128,
        },
        &spaces,
    ));
}

#[test]
fn mark_dirty_spaces_invalidates_sync_signature() {
    let mut system = ReflectionProbeSpecularSystem::new();
    system.sync_signature = Some(SpecularSyncSignature {
        face_size: 256,
        max_local_reflection_probes: 4,
        ready: Vec::new(),
    });

    system.mark_render_spaces_dirty([RenderSpaceId(7)]);

    assert!(system.dirty_spaces.contains(&RenderSpaceId(7)));
    assert!(system.sync_signature.is_none());
}

#[test]
fn final_ready_generation_reports_runtime_probe_generation() {
    let mut system = ReflectionProbeSpecularSystem::new();
    system.runtime_final_ready_generation.insert(
        ProbeIdentity {
            space_id: RenderSpaceId(7),
            renderable_index: 3,
        },
        42,
    );

    assert_eq!(system.final_ready_generation(7, 3), Some(42));
    assert_eq!(system.final_ready_generation(7, 4), None);
}

#[test]
fn purge_spaces_removes_final_ready_generations() {
    let mut system = ReflectionProbeSpecularSystem::new();
    system.runtime_final_ready_generation.insert(
        ProbeIdentity {
            space_id: RenderSpaceId(7),
            renderable_index: 3,
        },
        42,
    );
    let mut spaces = HashSet::new();
    spaces.insert(RenderSpaceId(7));

    system.purge_render_space_resources(&spaces);

    assert_eq!(system.final_ready_generation(7, 3), None);
}

#[test]
fn space_summary_normalize_orders_ready_rows() {
    let mut summary = CachedSpaceSummary {
        ready: vec![
            ready_summary(RenderSpaceId(2), 0, 4, true),
            ready_summary(RenderSpaceId(1), 5, 5, false),
            ready_summary(RenderSpaceId(1), 4, 6, false),
            ready_summary(RenderSpaceId(1), 4, 4, true),
        ],
        ..Default::default()
    };

    summary.normalize();

    let order = summary
        .ready
        .iter()
        .map(|probe| {
            (
                probe.identity.space_id,
                probe.identity.renderable_index,
                probe.mip_levels,
                probe.has_sh2,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        order,
        vec![
            (RenderSpaceId(1), 4, 4, true),
            (RenderSpaceId(1), 4, 6, false),
            (RenderSpaceId(1), 5, 5, false),
            (RenderSpaceId(2), 0, 4, true),
        ]
    );
}

#[test]
fn sync_signature_tracks_selection_inputs() {
    let ready = vec![ready_summary(RenderSpaceId(1), 2, 4, false)];
    let with_sh2 = vec![ready_summary(RenderSpaceId(1), 2, 4, true)];

    assert_ne!(
        SpecularSyncSignature {
            face_size: 128,
            max_local_reflection_probes: 4,
            ready: ready.clone(),
        },
        SpecularSyncSignature {
            face_size: 128,
            max_local_reflection_probes: 8,
            ready: ready.clone(),
        }
    );
    assert_ne!(
        SpecularSyncSignature {
            face_size: 128,
            max_local_reflection_probes: 4,
            ready,
        },
        SpecularSyncSignature {
            face_size: 128,
            max_local_reflection_probes: 4,
            ready: with_sh2,
        }
    );
}

#[test]
fn collected_resources_extend_cached_preserves_cached_probe_sets() {
    let identity = ProbeIdentity {
        space_id: RenderSpaceId(3),
        renderable_index: 9,
    };
    let key = runtime_key(RenderSpaceId(3), 9);
    let capture_key = RuntimeReflectionProbeCaptureKey {
        space_id: RenderSpaceId(3),
        renderable_index: 9,
    };
    let cache = CachedSpace {
        summary: CachedSpaceSummary {
            active_keys: std::iter::once(key.clone()).collect(),
            active_capture_keys: std::iter::once(capture_key).collect(),
            active_identities: std::iter::once(identity).collect(),
            ready: Vec::new(),
        },
        ready: Vec::new(),
    };
    let mut collected = CollectedProbeResources::default();

    collected.extend_cached(&cache);

    assert!(collected.active_keys.contains(&key));
    assert!(collected.active_capture_keys.contains(&capture_key));
    assert!(collected.active_identities.contains(&identity));
}

#[test]
fn new_system_reports_default_stats() {
    assert_eq!(
        ReflectionProbeSpecularSystem::new().last_stats(),
        MaintainStats::default()
    );
}

#[test]
fn runtime_ibl_bakes_are_sliced_for_every_host_mode() {
    for mode in [
        ReflectionProbeTimeSlicingMode::AllFacesAtOnce,
        ReflectionProbeTimeSlicingMode::IndividualFaces,
        ReflectionProbeTimeSlicingMode::NoTimeSlicing,
    ] {
        assert_eq!(runtime_ibl_policy(mode), IblBakePolicy::UnityTimeSliced);
    }
}

#[test]
fn reflection_probe_cube_byte_count_includes_every_face_and_mip() {
    assert_eq!(reflection_probe_cube_bytes(256), 4_194_288);
}

#[test]
fn reflection_probe_face_size_stays_inside_the_atlas_budget() {
    assert_eq!(budgeted_reflection_probe_face_size(256, 127), 256);
    assert_eq!(budgeted_reflection_probe_face_size(256, 128), 128);
    assert_eq!(budgeted_reflection_probe_face_size(256, 340), 128);
}

#[test]
fn atlas_growth_is_geometric_and_clamped_to_device_slots() {
    assert_eq!(atlas_growth_capacity(2, 341), 2);
    assert_eq!(atlas_growth_capacity(3, 341), 4);
    assert_eq!(atlas_growth_capacity(129, 341), 256);
    assert_eq!(atlas_growth_capacity(257, 341), 341);
}

#[test]
fn atlas_shrinks_at_the_two_to_one_capacity_boundary() {
    assert!(atlas_capacity_needs_reallocation(128, 33, 64));
    assert!(!atlas_capacity_needs_reallocation(128, 65, 128));
}

#[test]
fn face_size_transition_waits_for_every_target_cube() {
    let old = runtime_key_with_face_size(RenderSpaceId(1), 1, 256);
    let new = runtime_key_with_face_size(RenderSpaceId(1), 1, 128);

    assert!(atlas_face_transition_pending(256, 128, [&old]));
    assert!(atlas_face_transition_pending(256, 128, [&old, &new]));
    assert!(!atlas_face_transition_pending(256, 128, std::iter::empty()));
    assert!(!atlas_face_transition_pending(256, 128, [&new]));
    assert!(!atlas_face_transition_pending(128, 128, [&old]));
}

#[test]
fn shared_filtered_content_uses_one_texture_request_and_small_capacity() {
    let source = atlas_request(RenderSpaceId(1), 10);
    let requests = (0..65)
        .map(|index| {
            let mut request = source.clone();
            request.identity = AtlasProbeIdentity {
                space_id: RenderSpaceId(2),
                renderable_index: index,
            };
            request
        })
        .collect();

    let unique = deduplicate_atlas_requests(requests);

    assert_eq!(unique.len(), 1);
    let capacity = atlas_growth_capacity(
        (unique.len() + usize::from(FIRST_PROBE_ATLAS_SLOT)) as u16,
        341,
    );
    assert_eq!(capacity, 2);
    assert!(reflection_probe_cube_bytes(256) * u64::from(capacity) < 9 * 1024 * 1024);
}

#[test]
fn solid_colors_deduplicate_by_texels_not_probe_identity() {
    let first_key = solid_color_key(1, 99, 256);
    let second_key = solid_color_key(2, 99, 256);
    let first = atlas_request_for_key(RenderSpaceId(1), 1, first_key);
    let second = atlas_request_for_key(RenderSpaceId(1), 2, second_key);

    let unique = deduplicate_atlas_requests(vec![first, second]);

    assert_eq!(unique.len(), 1);
}

#[test]
fn atlas_placement_preserves_stable_identity_slots_across_request_reordering() {
    let first = atlas_request(RenderSpaceId(1), 10);
    let second = atlas_request(RenderSpaceId(1), 20);
    let previous = vec![
        None,
        Some(resident_probe_for_request(&first)),
        Some(resident_probe_for_request(&second)),
        None,
    ];

    let plan = plan_atlas_placements(&[second, first], &previous, 4);

    assert_eq!(plan.placements[0].slot, 2);
    assert!(plan.placements[0].copy_source.is_none());
    assert_eq!(plan.placements[1].slot, 1);
    assert!(plan.placements[1].copy_source.is_none());
}

#[test]
fn atlas_placement_reuses_resident_texels_for_a_new_identity_and_shared_key() {
    let source = atlas_request(RenderSpaceId(1), 10);
    let shared_key = source.key.clone();
    let first = AtlasProbeRequest {
        identity: AtlasProbeIdentity {
            space_id: RenderSpaceId(2),
            renderable_index: 30,
        },
        texture_key: AtlasTextureKey::from(&shared_key),
        key: shared_key.clone(),
        mip_levels: source.mip_levels,
    };
    let second = AtlasProbeRequest {
        identity: AtlasProbeIdentity {
            space_id: RenderSpaceId(2),
            renderable_index: 31,
        },
        texture_key: AtlasTextureKey::from(&shared_key),
        key: shared_key,
        mip_levels: source.mip_levels,
    };
    let previous = vec![None, Some(resident_probe_for_request(&source)), None, None];

    let plan = plan_atlas_placements(&[first, second], &previous, 4);

    assert_eq!(plan.placements[0].slot, 1);
    assert!(plan.placements[0].copy_source.is_none());
    assert_eq!(plan.placements[1].slot, 2);
    assert!(matches!(
        plan.placements[1].copy_source,
        Some(PlannedAtlasCopySource::ResidentSlot(1))
    ));
}

#[test]
fn atlas_placement_requests_completed_cube_for_a_new_key() {
    let request = atlas_request(RenderSpaceId(4), 12);

    let plan = plan_atlas_placements(std::slice::from_ref(&request), &[None, None], 2);

    assert_eq!(plan.placements[0].slot, 1);
    assert!(matches!(
        plan.placements[0].copy_source,
        Some(PlannedAtlasCopySource::CompletedCube)
    ));
}

#[test]
fn atlas_repack_compacts_a_high_resident_slot_without_losing_external_texels() {
    let request = atlas_request(RenderSpaceId(4), 12);
    let mut previous = vec![None; 128];
    previous[100] = Some(resident_probe_for_request(&request));

    let repack = plan_atlas_repack(std::slice::from_ref(&request), &previous, 2);

    assert_eq!(repack.slots[1], Some(resident_probe_for_request(&request)));
    assert_eq!(
        repack.copies,
        vec![AtlasRepackCopy {
            source_slot: 100,
            destination_slot: 1,
            mip_levels: request.mip_levels,
        }]
    );
}

fn atlas_request(space_id: RenderSpaceId, renderable_index: i32) -> AtlasProbeRequest {
    let key = runtime_key(space_id, renderable_index);
    atlas_request_for_key(space_id, renderable_index, key)
}

fn atlas_request_for_key(
    space_id: RenderSpaceId,
    renderable_index: i32,
    key: SkyboxIblKey,
) -> AtlasProbeRequest {
    AtlasProbeRequest {
        identity: AtlasProbeIdentity {
            space_id,
            renderable_index,
        },
        texture_key: AtlasTextureKey::from(&key),
        key,
        mip_levels: 8,
    }
}

fn ready_summary(
    space_id: RenderSpaceId,
    renderable_index: i32,
    mip_levels: u32,
    has_sh2: bool,
) -> ReadyProbeSummary {
    ReadyProbeSummary {
        identity: ProbeIdentity {
            space_id,
            renderable_index,
        },
        key: runtime_key(space_id, renderable_index),
        mip_levels,
        has_sh2,
        spatial: SpatialProbeSummary {
            renderable_index,
            importance: 0,
            aabb_min: [0; 3],
            aabb_max: [0; 3],
            influence_aabb_min: [0; 3],
            influence_aabb_max: [0; 3],
            center: [0; 3],
            volume: 0,
            skybox: false,
        },
    }
}

fn runtime_key(space_id: RenderSpaceId, renderable_index: i32) -> SkyboxIblKey {
    runtime_key_with_face_size(space_id, renderable_index, 128)
}

fn runtime_key_with_face_size(
    space_id: RenderSpaceId,
    renderable_index: i32,
    face_size: u32,
) -> SkyboxIblKey {
    SkyboxIblKey::RuntimeCubemap {
        render_space_id: space_id.0,
        renderable_index,
        generation: 1,
        mip_levels: 1,
        storage_v_inverted: true,
        face_size,
    }
}

fn solid_color_key(identity: u64, color_hash: u64, face_size: u32) -> SkyboxIblKey {
    SkyboxIblKey::SolidColor {
        identity,
        color_hash,
        face_size,
    }
}
