use glam::{Vec3, Vec3A, Vec4};

use crate::backend::AssetTransferQueue;
use crate::backend::frame_gpu::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_METADATA_BOX_PROJECTION,
    REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL,
};
use crate::scene::{
    ReflectionProbeEntry, RenderSpaceId, SceneCoordinator, reflection_probe_skybox_only,
    reflection_probe_use_box_projection,
};
use crate::shared::{ReflectionProbeClear, ReflectionProbeState, ReflectionProbeType, RenderSH2};
use crate::skybox::specular::{
    CubemapIblSource, RuntimeCubemapIblSource, SkyboxIblSource, solid_color_ibl_source,
};
use crate::world_mesh::culling::world_aabb_from_local_bounds;

use super::captures::{RuntimeReflectionProbeCaptureKey, RuntimeReflectionProbeCaptureStore};
use super::selection::{
    SpatialProbe, aabb_valid, aabb_volume, expanded_aabb, sanitized_blend_distance,
};

pub(super) fn resolve_probe_source(
    space_id: RenderSpaceId,
    probe: &ReflectionProbeEntry,
    assets: &AssetTransferQueue,
    captures: &RuntimeReflectionProbeCaptureStore,
) -> Option<SkyboxIblSource> {
    let state = probe.state;
    if state.intensity <= 0.0 {
        return None;
    }
    if state.clear_flags == ReflectionProbeClear::Color {
        let color = state.background_color;
        return Some(solid_color_ibl_source(
            color_probe_identity(probe.renderable_index, color),
            color.to_array(),
        ));
    }
    if state.r#type == ReflectionProbeType::Baked {
        return resolve_baked_probe_source(state, assets);
    }
    if state.r#type == ReflectionProbeType::OnChanges {
        return resolve_runtime_capture_source(space_id, probe, captures);
    }
    None
}

fn resolve_runtime_capture_source(
    space_id: RenderSpaceId,
    probe: &ReflectionProbeEntry,
    captures: &RuntimeReflectionProbeCaptureStore,
) -> Option<SkyboxIblSource> {
    let capture = captures.get(RuntimeReflectionProbeCaptureKey {
        space_id,
        renderable_index: probe.renderable_index,
    })?;
    Some(SkyboxIblSource::RuntimeCubemap(RuntimeCubemapIblSource {
        render_space_id: capture.key.space_id.0,
        renderable_index: capture.key.renderable_index,
        generation: capture.generation,
        face_size: capture.face_size,
        mip_levels: capture.mip_levels,
        texture: capture.texture.clone(),
        view: capture.view.clone(),
    }))
}

pub(super) fn resolve_baked_probe_source(
    state: ReflectionProbeState,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    if state.cubemap_asset_id < 0 {
        return None;
    }
    let cubemap = assets.cubemap_pool().get(state.cubemap_asset_id)?;
    if cubemap.mip_levels_resident == 0 {
        return None;
    }
    Some(SkyboxIblSource::Cubemap(CubemapIblSource {
        material_asset_id: -1,
        material_generation: 0,
        route_hash: 0,
        asset_id: state.cubemap_asset_id,
        allocation_generation: cubemap.allocation_generation,
        face_size: cubemap.size,
        mip_levels_resident: cubemap.mip_levels_resident,
        content_generation: cubemap.content_generation,
        storage_v_inverted: cubemap.storage_v_inverted,
        view: cubemap.view.clone(),
    }))
}

fn color_probe_identity(renderable_index: i32, color: Vec4) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for bits in [
        renderable_index as u32,
        color.x.to_bits(),
        color.y.to_bits(),
        color.z.to_bits(),
        color.w.to_bits(),
    ] {
        hash ^= u64::from(bits);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

pub(super) fn spatial_probe_for_state(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    probe: &ReflectionProbeEntry,
    render_context: crate::shared::RenderingContext,
    atlas_index: u16,
) -> Option<SpatialProbe> {
    if probe.transform_id < 0 {
        return None;
    }
    let box_size = probe.state.box_size.abs();
    if box_size.cmplt(Vec3::splat(1e-6)).any() {
        return None;
    }
    let world =
        scene.world_matrix_for_context(space_id, probe.transform_id as usize, render_context)?;
    let bounds = crate::shared::RenderBoundingBox {
        center: Vec3::ZERO,
        extents: box_size * 0.5,
    };
    let (min, max) = world_aabb_from_local_bounds(&bounds, world)?;
    if !aabb_valid(min, max) {
        return None;
    }
    let blend_distance = sanitized_blend_distance(probe.state.blend_distance);
    let (influence_aabb_min, influence_aabb_max) = expanded_aabb(min, max, blend_distance);
    Some(SpatialProbe {
        renderable_index: probe.renderable_index,
        atlas_index,
        importance: probe.state.importance,
        aabb_min: Vec3A::from(min),
        aabb_max: Vec3A::from(max),
        influence_aabb_min,
        influence_aabb_max,
        center: Vec3A::from(world.transform_point3(Vec3::ZERO)),
        volume: aabb_volume(min, max),
        blend_distance,
        skybox: reflection_probe_skybox_only(probe.state.flags),
    })
}

pub(super) fn metadata_for_spatial(
    spatial: &SpatialProbe,
    state: ReflectionProbeState,
    sh2: Option<&RenderSH2>,
) -> GpuReflectionProbeMetadata {
    let flags = if reflection_probe_use_box_projection(state.flags) {
        REFLECTION_PROBE_METADATA_BOX_PROJECTION
    } else {
        0
    };
    let sh2_source = if sh2.is_some() {
        REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL
    } else {
        0.0
    };
    GpuReflectionProbeMetadata {
        box_min: [
            spatial.aabb_min.x,
            spatial.aabb_min.y,
            spatial.aabb_min.z,
            sanitized_blend_distance(state.blend_distance),
        ],
        box_max: [
            spatial.aabb_max.x,
            spatial.aabb_max.y,
            spatial.aabb_max.z,
            0.0,
        ],
        position: [spatial.center.x, spatial.center.y, spatial.center.z, 0.0],
        params: [state.intensity.max(0.0), 0.0, flags as f32, sh2_source],
        sh2: sh2.map_or([[0.0; 4]; 9], pack_render_sh2_raw),
    }
}

fn pack_render_sh2_raw(sh: &RenderSH2) -> [[f32; 4]; 9] {
    [
        [sh.sh0.x, sh.sh0.y, sh.sh0.z, 0.0],
        [sh.sh1.x, sh.sh1.y, sh.sh1.z, 0.0],
        [sh.sh2.x, sh.sh2.y, sh.sh2.z, 0.0],
        [sh.sh3.x, sh.sh3.y, sh.sh3.z, 0.0],
        [sh.sh4.x, sh.sh4.y, sh.sh4.z, 0.0],
        [sh.sh5.x, sh.sh5.y, sh.sh5.z, 0.0],
        [sh.sh6.x, sh.sh6.y, sh.sh6.z, 0.0],
        [sh.sh7.x, sh.sh7.y, sh.sh7.z, 0.0],
        [sh.sh8.x, sh.sh8.y, sh.sh8.z, 0.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(index: i32, atlas: u16, importance: i32, min: Vec3, max: Vec3) -> SpatialProbe {
        let (influence_aabb_min, influence_aabb_max) = expanded_aabb(min, max, 0.0);
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
            blend_distance: 0.0,
            skybox: false,
        }
    }

    #[test]
    fn missing_baked_cubemap_is_not_a_probe_source() {
        let assets = AssetTransferQueue::new();
        let state = ReflectionProbeState {
            intensity: 1.0,
            cubemap_asset_id: 42,
            r#type: ReflectionProbeType::Baked,
            ..ReflectionProbeState::default()
        };

        assert!(resolve_baked_probe_source(state, &assets).is_none());
    }

    #[test]
    fn missing_onchanges_capture_is_not_a_specular_source() {
        let assets = AssetTransferQueue::new();
        let captures = RuntimeReflectionProbeCaptureStore::default();
        let probe = ReflectionProbeEntry {
            renderable_index: 5,
            transform_id: 1,
            state: ReflectionProbeState {
                intensity: 1.0,
                flags: 0b001,
                r#type: ReflectionProbeType::OnChanges,
                ..ReflectionProbeState::default()
            },
        };

        let source = resolve_probe_source(RenderSpaceId(7), &probe, &assets, &captures);

        assert!(source.is_none());
    }

    #[test]
    fn spatial_probe_metadata_without_sh2_keeps_specular_enabled() {
        let spatial = probe(0, 3, 0, Vec3::splat(-1.0), Vec3::splat(1.0));
        let state = ReflectionProbeState {
            flags: 0b100,
            r#type: ReflectionProbeType::OnChanges,
            intensity: 1.0,
            ..ReflectionProbeState::default()
        };

        let metadata = metadata_for_spatial(&spatial, state, None);

        assert_eq!(metadata.params, [1.0, 0.0, 1.0, 0.0]);
        assert_eq!(metadata.sh2, [[0.0; 4]; 9]);
    }

    #[test]
    fn skybox_only_spatial_probe_metadata_marks_local_sh2_source() {
        let spatial = probe(0, 3, 0, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sh = RenderSH2 {
            sh0: Vec3::ONE,
            ..RenderSH2::default()
        };
        let state = ReflectionProbeState {
            flags: 0b001,
            r#type: ReflectionProbeType::OnChanges,
            intensity: 1.0,
            ..ReflectionProbeState::default()
        };

        let metadata = metadata_for_spatial(&spatial, state, Some(&sh));

        assert_eq!(
            metadata.params[3],
            REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL
        );
    }

    #[test]
    fn non_skybox_spatial_probe_metadata_marks_local_sh2_source() {
        let spatial = probe(0, 3, 0, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sh = RenderSH2 {
            sh0: Vec3::ONE,
            ..RenderSH2::default()
        };
        let state = ReflectionProbeState {
            flags: 0b001,
            clear_flags: ReflectionProbeClear::Color,
            intensity: 1.0,
            ..ReflectionProbeState::default()
        };

        let metadata = metadata_for_spatial(&spatial, state, Some(&sh));

        assert_eq!(
            metadata.params[3],
            REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL
        );
    }

    #[test]
    fn spatial_probe_metadata_stores_sanitized_blend_distance() {
        let spatial = probe(0, 3, 0, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sh = RenderSH2::default();
        let metadata = metadata_for_spatial(
            &spatial,
            ReflectionProbeState {
                blend_distance: 2.5,
                intensity: 1.0,
                ..ReflectionProbeState::default()
            },
            Some(&sh),
        );
        let negative = metadata_for_spatial(
            &spatial,
            ReflectionProbeState {
                blend_distance: -1.0,
                intensity: 1.0,
                ..ReflectionProbeState::default()
            },
            Some(&sh),
        );

        assert_eq!(metadata.box_min[3], 2.5);
        assert_eq!(negative.box_min[3], 0.0);
    }
}
