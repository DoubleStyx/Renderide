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
    }
}

fn runtime_key(space_id: RenderSpaceId, renderable_index: i32) -> SkyboxIblKey {
    SkyboxIblKey::RuntimeCubemap {
        render_space_id: space_id.0,
        renderable_index,
        generation: 1,
        mip_levels: 1,
        storage_v_inverted: true,
        face_size: 128,
    }
}
