//! Source-key resolution for reflection-probe SH2 projection tasks.

use super::task_rows::TaskHeader;
use super::{
    CubemapResidency, CubemapSourceMaterialIdentity, DEFAULT_SAMPLE_SIZE, GpuSh2Source,
    Sh2SourceKey, constant_color_sh2,
};
use crate::reflection_probes::specular::{
    RuntimeReflectionProbeCaptureKey, RuntimeReflectionProbeCaptureStore,
};
use crate::scene::{
    ReflectionProbeEntry, RenderSpaceId, SceneCoordinator, reflection_probe_solid_color,
};
use crate::shared::{ReflectionProbeType, RenderSH2};

/// Either a synchronous CPU result or a GPU source to project.
pub(super) enum Sh2ResolvedSource {
    /// CPU-computed SH2.
    Cpu(Box<RenderSH2>),
    /// GPU-computed SH2 source.
    Gpu(GpuSh2Source),
    /// Source is expected to become available later.
    Postpone,
}

/// Resolves a host task into a cache key and source payload.
pub(super) fn resolve_task_source(
    scene: &SceneCoordinator,
    assets: &crate::backend::AssetTransferQueue,
    captures: &RuntimeReflectionProbeCaptureStore,
    render_space_id: i32,
    task: TaskHeader,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if task.renderable_index < 0 || task.reflection_probe_renderable_index < 0 {
        return None;
    }
    let space = scene.space(RenderSpaceId(render_space_id))?;
    let probe = space
        .reflection_probes()
        .get(task.reflection_probe_renderable_index as usize)?;
    let state = probe.state;

    if reflection_probe_solid_color(state) {
        let color = state.background_color * state.intensity.max(0.0);
        let key = Sh2SourceKey::ConstantColor {
            render_space_id,
            color_bits: color.to_array().map(|f| f.to_bits()),
        };
        return Some((
            key,
            Sh2ResolvedSource::Cpu(Box::new(constant_color_sh2(color.truncate()))),
        ));
    }
    if state.r#type == ReflectionProbeType::Baked {
        if state.cubemap_asset_id < 0 {
            return None;
        }
        let asset_id = state.cubemap_asset_id;
        let identity = CubemapSourceMaterialIdentity::DIRECT_PROBE;
        let Some(cubemap) = assets.cubemap_pool().get(asset_id) else {
            return Some((
                Sh2SourceKey::cubemap(
                    render_space_id,
                    identity,
                    asset_id,
                    CubemapResidency::default(),
                ),
                Sh2ResolvedSource::Postpone,
            ));
        };
        let key = Sh2SourceKey::cubemap(
            render_space_id,
            identity,
            asset_id,
            cubemap_residency_from_pool(cubemap),
        );
        if cubemap.mip_levels_resident == 0 {
            return Some((key, Sh2ResolvedSource::Postpone));
        }
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
                asset_id,
                storage_v_inverted: cubemap.storage_v_inverted,
            }),
        ));
    }

    if matches!(
        state.r#type,
        ReflectionProbeType::OnChanges | ReflectionProbeType::Realtime
    ) {
        return resolve_runtime_capture_source(render_space_id, probe, captures);
    }
    None
}

fn resolve_runtime_capture_source(
    render_space_id: i32,
    probe: &ReflectionProbeEntry,
    captures: &RuntimeReflectionProbeCaptureStore,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    let renderable_index = probe.renderable_index;
    let key = RuntimeReflectionProbeCaptureKey {
        space_id: RenderSpaceId(render_space_id),
        renderable_index,
    };
    let Some(capture) = captures.get(key) else {
        return Some((
            Sh2SourceKey::RuntimeCubemap {
                render_space_id,
                renderable_index,
                generation: 0,
                size: 0,
                sample_size: DEFAULT_SAMPLE_SIZE,
            },
            Sh2ResolvedSource::Postpone,
        ));
    };
    let key = Sh2SourceKey::RuntimeCubemap {
        render_space_id,
        renderable_index,
        generation: capture.generation,
        size: capture.face_size,
        sample_size: DEFAULT_SAMPLE_SIZE,
    };
    Some((
        key,
        Sh2ResolvedSource::Gpu(GpuSh2Source::RuntimeCubemap {
            texture: capture.texture.clone(),
            view: capture.view.clone(),
        }),
    ))
}

fn cubemap_residency_from_pool(
    cubemap: &crate::gpu_pools::pools::cubemap::GpuCubemap,
) -> CubemapResidency {
    CubemapResidency {
        allocation_generation: cubemap.allocation_generation,
        size: cubemap.size,
        resident_mips: cubemap.mip_levels_resident,
        content_generation: cubemap.content_generation,
        storage_v_inverted: cubemap.storage_v_inverted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AssetTransferQueue;
    use crate::shared::{ReflectionProbeClear, ReflectionProbeState};

    #[test]
    fn missing_runtime_capture_postpones_onchanges_probe() {
        let captures = RuntimeReflectionProbeCaptureStore::default();

        let probe = ReflectionProbeEntry {
            renderable_index: 3,
            transform_id: 12,
            state: ReflectionProbeState {
                clear_flags: ReflectionProbeClear::Skybox,
                ..Default::default()
            },
        };

        let (key, source) = resolve_runtime_capture_source(7, &probe, &captures)
            .expect("missing captures should return a stable postponed key");

        assert_eq!(
            key,
            Sh2SourceKey::RuntimeCubemap {
                render_space_id: 7,
                renderable_index: 3,
                generation: 0,
                size: 0,
                sample_size: DEFAULT_SAMPLE_SIZE,
            }
        );
        assert!(matches!(source, Sh2ResolvedSource::Postpone));
    }

    #[test]
    fn missing_runtime_capture_key_ignores_background_color() {
        let captures = RuntimeReflectionProbeCaptureStore::default();
        let skybox_probe = ReflectionProbeEntry {
            renderable_index: 3,
            transform_id: 12,
            state: ReflectionProbeState {
                clear_flags: ReflectionProbeClear::Skybox,
                background_color: glam::Vec4::new(0.1, 0.2, 0.3, 1.0),
                ..Default::default()
            },
        };
        let color_probe = ReflectionProbeEntry {
            renderable_index: skybox_probe.renderable_index,
            transform_id: skybox_probe.transform_id,
            state: ReflectionProbeState {
                clear_flags: ReflectionProbeClear::Color,
                background_color: glam::Vec4::new(0.9, 0.8, 0.7, 1.0),
                flags: 0,
                ..Default::default()
            },
        };

        let (skybox_key, skybox_source) =
            resolve_runtime_capture_source(7, &skybox_probe, &captures)
                .expect("missing skybox capture should return a postponed key");
        let (color_key, color_source) = resolve_runtime_capture_source(7, &color_probe, &captures)
            .expect("missing color capture should return a postponed key");

        assert_eq!(color_key, skybox_key);
        assert!(matches!(skybox_source, Sh2ResolvedSource::Postpone));
        assert!(matches!(color_source, Sh2ResolvedSource::Postpone));
    }

    #[test]
    fn realtime_task_without_capture_postpones_runtime_cubemap_source() {
        let mut scene = SceneCoordinator::new();
        let space_id = RenderSpaceId(7);
        scene.test_seed_space_identity_worlds(space_id, Vec::new(), Vec::new());
        scene.test_push_reflection_probes(
            space_id,
            [ReflectionProbeEntry {
                renderable_index: 0,
                transform_id: 12,
                state: ReflectionProbeState {
                    renderable_index: 0,
                    clear_flags: ReflectionProbeClear::Skybox,
                    r#type: ReflectionProbeType::Realtime,
                    ..Default::default()
                },
            }],
        );
        let assets = AssetTransferQueue::new();
        let captures = RuntimeReflectionProbeCaptureStore::default();

        let (key, source) = resolve_task_source(
            &scene,
            &assets,
            &captures,
            space_id.0,
            TaskHeader {
                renderable_index: 42,
                reflection_probe_renderable_index: 0,
            },
        )
        .expect("realtime probes should resolve to a postponed runtime cubemap source");

        assert_eq!(
            key,
            Sh2SourceKey::RuntimeCubemap {
                render_space_id: space_id.0,
                renderable_index: 0,
                generation: 0,
                size: 0,
                sample_size: DEFAULT_SAMPLE_SIZE,
            }
        );
        assert!(matches!(source, Sh2ResolvedSource::Postpone));
    }
}
