//! Reflection-probe renderable state mirrored from host updates.

use crate::color_space::srgb_vec4_rgb_to_linear;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    REFLECTION_PROBE_CHANGE_RENDER_TASK_HOST_ROW_BYTES, REFLECTION_PROBE_STATE_HOST_ROW_BYTES,
    ReflectionProbeChangeRenderResult, ReflectionProbeChangeRenderTask, ReflectionProbeClear,
    ReflectionProbeRenderablesUpdate, ReflectionProbeState, ReflectionProbeType,
};

use super::dense_update::{push_dense_additions, swap_remove_dense_indices_with_update};
use super::error::SceneError;
use super::render_space::RenderSpaceState;
use super::transforms::TransformRemovalEvent;
use super::world::fixup_transform_id;

/// One dense reflection-probe renderable entry inside a render space.
#[derive(Debug, Clone)]
pub struct ReflectionProbeEntry {
    /// Dense renderable index assigned by the host.
    pub renderable_index: i32,
    /// Dense transform index that owns the probe component.
    pub transform_id: i32,
    /// Latest probe state row sent by the host; `background_color` is normalized to linear RGB on apply.
    pub state: ReflectionProbeState,
}

/// Owned reflection-probe update extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedReflectionProbeRenderablesUpdate {
    /// Dense renderable removal indices terminated by a negative entry.
    pub removals: Vec<i32>,
    /// Added probe transform indices terminated by a negative entry.
    pub additions: Vec<i32>,
    /// Probe state rows terminated by `renderable_index < 0`.
    pub states: Vec<ReflectionProbeState>,
    /// OnChanges render requests terminated by `renderable_index < 0`.
    pub changed_probes_to_render: Vec<ReflectionProbeChangeRenderTask>,
}

/// Host-visible changed-probe completions plus renderer-side scene capture requests.
#[derive(Default, Debug)]
pub struct DrainedReflectionProbeRenderChanges {
    /// Render completions that can be returned to the host immediately.
    pub completed: Vec<ReflectionProbeChangeRenderResult>,
    /// OnChanges probes that need scene cubemap capture before completion.
    pub scene_captures: Vec<ReflectionProbeOnChangesRenderRequest>,
}

/// One host OnChanges reflection probe render request that needs scene capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReflectionProbeOnChangesRenderRequest {
    /// Host render-space id that owns the probe.
    pub render_space_id: i32,
    /// Dense reflection-probe renderable index.
    pub renderable_index: i32,
    /// Host unique id echoed in the eventual completion row.
    pub unique_id: i32,
}

/// Returns whether a probe state requests skybox-only rendering.
#[inline]
pub fn reflection_probe_skybox_only(flags: u8) -> bool {
    flags & 0b001 != 0
}

/// Returns whether a probe state is a fixed solid color.
#[inline]
pub fn reflection_probe_solid_color(state: ReflectionProbeState) -> bool {
    state.clear_flags == ReflectionProbeClear::Color
        && reflection_probe_skybox_only(state.flags)
        && state.r#type != ReflectionProbeType::Baked
}

/// Returns whether a probe state requests HDR rendering.
#[inline]
#[cfg(test)]
pub fn reflection_probe_hdr(flags: u8) -> bool {
    flags & 0b010 != 0
}

/// Returns whether a probe state uses box projection.
#[inline]
pub fn reflection_probe_use_box_projection(flags: u8) -> bool {
    flags & 0b100 != 0
}

/// Reads every reflection-probe shared-memory buffer for one render-space update.
pub(crate) fn extract_reflection_probe_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &ReflectionProbeRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedReflectionProbeRenderablesUpdate, SceneError> {
    let mut out = ExtractedReflectionProbeRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("reflection probe removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("reflection probe additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("reflection probe states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<ReflectionProbeState>(
                &update.states,
                REFLECTION_PROBE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.changed_probes_to_render.length > 0 {
        let ctx = format!("reflection probe changed renders scene_id={scene_id}");
        out.changed_probes_to_render = shm
            .access_copy_memory_packable_rows::<ReflectionProbeChangeRenderTask>(
                &update.changed_probes_to_render,
                REFLECTION_PROBE_CHANGE_RENDER_TASK_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

fn update_moved_reflection_probe(probe: &mut ReflectionProbeEntry, index: i32) {
    probe.renderable_index = index;
}

fn build_added_reflection_probe(transform_id: i32, renderable_index: i32) -> ReflectionProbeEntry {
    ReflectionProbeEntry {
        renderable_index,
        transform_id,
        state: ReflectionProbeState::default(),
    }
}

/// Applies a pre-extracted reflection-probe update to one render space.
pub(crate) fn apply_reflection_probe_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedReflectionProbeRenderablesUpdate,
) {
    profiling::scope!("scene::apply_reflection_probes");
    space.pending_reflection_probe_render_changes.clear();

    swap_remove_dense_indices_with_update(
        &mut space.reflection_probes,
        &extracted.removals,
        update_moved_reflection_probe,
    );
    push_dense_additions(
        &mut space.reflection_probes,
        &extracted.additions,
        &build_added_reflection_probe,
    );
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.reflection_probes.get_mut(idx) else {
            continue;
        };
        let mut state = *state;
        state.background_color = srgb_vec4_rgb_to_linear(state.background_color);
        entry.renderable_index = state.renderable_index;
        entry.state = state;
    }
    space.pending_reflection_probe_render_changes.extend(
        extracted
            .changed_probes_to_render
            .iter()
            .take_while(|task| task.renderable_index >= 0)
            .copied(),
    );
}

/// Updates cached probe transform indices after dense transform swap-removals.
pub(crate) fn fixup_reflection_probes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() || space.reflection_probes.is_empty() {
        return;
    }
    for removal in removals {
        for probe in &mut space.reflection_probes {
            probe.transform_id = fixup_transform_id(
                probe.transform_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
    }
    space
        .reflection_probes
        .retain(|probe| probe.transform_id >= 0);
}

/// Drains changed-probe render requests into immediate completions or OnChanges capture requests.
pub(crate) fn drain_reflection_probe_render_changes(
    space: &mut RenderSpaceState,
) -> DrainedReflectionProbeRenderChanges {
    let mut out = DrainedReflectionProbeRenderChanges::default();
    for task in space.pending_reflection_probe_render_changes.drain(..) {
        let idx = task.renderable_index as usize;
        let Some(entry) = space.reflection_probes.get(idx) else {
            logger::warn!(
                "reflection probe changed render ignored: render_space={} renderable_index={} not found",
                space.id.0,
                task.renderable_index
            );
            out.completed
                .push(changed_probe_completion(space.id.0, task.unique_id, true));
            continue;
        };
        if reflection_probe_solid_color(entry.state) {
            out.completed
                .push(changed_probe_completion(space.id.0, task.unique_id, false));
        } else if entry.state.r#type == ReflectionProbeType::OnChanges {
            out.scene_captures
                .push(ReflectionProbeOnChangesRenderRequest {
                    render_space_id: space.id.0,
                    renderable_index: task.renderable_index,
                    unique_id: task.unique_id,
                });
        } else if entry.state.r#type == ReflectionProbeType::Realtime {
            out.completed
                .push(changed_probe_completion(space.id.0, task.unique_id, false));
        }
    }
    out
}

/// Builds one changed-probe completion row.
pub(crate) const fn changed_probe_completion(
    render_space_id: i32,
    unique_id: i32,
    require_reset: bool,
) -> ReflectionProbeChangeRenderResult {
    ReflectionProbeChangeRenderResult {
        render_space_id,
        render_probe_unique_id: unique_id,
        require_reset: require_reset as u8,
        _padding: [0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_helpers_decode_host_bits() {
        assert!(reflection_probe_skybox_only(0b001));
        assert!(reflection_probe_hdr(0b010));
        assert!(reflection_probe_use_box_projection(0b100));
        assert!(!reflection_probe_skybox_only(0b010));
        assert!(!reflection_probe_hdr(0b001));
        assert!(!reflection_probe_use_box_projection(0b011));
    }

    #[test]
    fn solid_color_helper_requires_color_skybox_only_non_baked_probe() {
        assert!(reflection_probe_solid_color(ReflectionProbeState {
            clear_flags: ReflectionProbeClear::Color,
            flags: 0b001,
            r#type: ReflectionProbeType::OnChanges,
            ..ReflectionProbeState::default()
        }));
        assert!(reflection_probe_solid_color(ReflectionProbeState {
            clear_flags: ReflectionProbeClear::Color,
            flags: 0b001,
            r#type: ReflectionProbeType::Realtime,
            ..ReflectionProbeState::default()
        }));
        assert!(!reflection_probe_solid_color(ReflectionProbeState {
            clear_flags: ReflectionProbeClear::Color,
            flags: 0,
            r#type: ReflectionProbeType::OnChanges,
            ..ReflectionProbeState::default()
        }));
        assert!(!reflection_probe_solid_color(ReflectionProbeState {
            clear_flags: ReflectionProbeClear::Skybox,
            flags: 0b001,
            r#type: ReflectionProbeType::OnChanges,
            ..ReflectionProbeState::default()
        }));
        assert!(!reflection_probe_solid_color(ReflectionProbeState {
            clear_flags: ReflectionProbeClear::Color,
            flags: 0b001,
            r#type: ReflectionProbeType::Baked,
            ..ReflectionProbeState::default()
        }));
    }

    #[test]
    fn reflection_probe_update_linearizes_background_color() {
        let mut space = RenderSpaceState::default();
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: 0,
            transform_id: 1,
            state: ReflectionProbeState::default(),
        });
        let update = ExtractedReflectionProbeRenderablesUpdate {
            states: vec![ReflectionProbeState {
                renderable_index: 0,
                background_color: glam::Vec4::new(0.5, 0.04045, 1.25, 0.33),
                ..ReflectionProbeState::default()
            }],
            ..ExtractedReflectionProbeRenderablesUpdate::default()
        };

        apply_reflection_probe_renderables_update_extracted(&mut space, &update);

        let color = space.reflection_probes[0].state.background_color;
        assert!((color.x - 0.214_041_14).abs() < 0.000_001);
        assert!((color.y - (0.04045 / 12.92)).abs() < 0.000_001);
        assert!((color.z - 1.633_811_8).abs() < 0.000_001);
        assert_eq!(color.w, 0.33);
    }

    #[test]
    fn changed_skybox_onchanges_probe_queues_capture_request() {
        let mut space = RenderSpaceState {
            id: crate::scene::RenderSpaceId(12),
            ..RenderSpaceState::default()
        };
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: 0,
            transform_id: 1,
            state: ReflectionProbeState {
                renderable_index: 0,
                flags: 0b001,
                r#type: ReflectionProbeType::OnChanges,
                ..ReflectionProbeState::default()
            },
        });
        space
            .pending_reflection_probe_render_changes
            .push(ReflectionProbeChangeRenderTask {
                renderable_index: 0,
                unique_id: 77,
            });

        let results = drain_reflection_probe_render_changes(&mut space);

        assert!(results.completed.is_empty());
        assert_eq!(
            results.scene_captures,
            vec![ReflectionProbeOnChangesRenderRequest {
                render_space_id: 12,
                renderable_index: 0,
                unique_id: 77,
            }]
        );
        assert!(space.pending_reflection_probe_render_changes.is_empty());
    }

    #[test]
    fn changed_onchanges_scene_probe_queues_capture_request() {
        let mut space = RenderSpaceState {
            id: crate::scene::RenderSpaceId(12),
            ..RenderSpaceState::default()
        };
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: 0,
            transform_id: 1,
            state: ReflectionProbeState {
                renderable_index: 0,
                r#type: ReflectionProbeType::OnChanges,
                ..ReflectionProbeState::default()
            },
        });
        space
            .pending_reflection_probe_render_changes
            .push(ReflectionProbeChangeRenderTask {
                renderable_index: 0,
                unique_id: 88,
            });

        let results = drain_reflection_probe_render_changes(&mut space);

        assert!(results.completed.is_empty());
        assert_eq!(
            results.scene_captures,
            vec![ReflectionProbeOnChangesRenderRequest {
                render_space_id: 12,
                renderable_index: 0,
                unique_id: 88,
            }]
        );
    }

    #[test]
    fn changed_color_probe_returns_immediate_completion() {
        let mut space = RenderSpaceState {
            id: crate::scene::RenderSpaceId(12),
            ..RenderSpaceState::default()
        };
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: 0,
            transform_id: 1,
            state: ReflectionProbeState {
                renderable_index: 0,
                clear_flags: ReflectionProbeClear::Color,
                r#type: ReflectionProbeType::OnChanges,
                flags: 0x1,
                ..ReflectionProbeState::default()
            },
        });
        space
            .pending_reflection_probe_render_changes
            .push(ReflectionProbeChangeRenderTask {
                renderable_index: 0,
                unique_id: 99,
            });

        let results = drain_reflection_probe_render_changes(&mut space);

        assert_eq!(results.completed.len(), 1);
        assert_eq!(results.completed[0].render_space_id, 12);
        assert_eq!(results.completed[0].render_probe_unique_id, 99);
        assert_eq!(results.completed[0].require_reset, 0);
        assert!(results.scene_captures.is_empty());
    }

    #[test]
    fn changed_realtime_probe_returns_immediate_completion() {
        let mut space = RenderSpaceState {
            id: crate::scene::RenderSpaceId(12),
            ..RenderSpaceState::default()
        };
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: 0,
            transform_id: 1,
            state: ReflectionProbeState {
                renderable_index: 0,
                r#type: ReflectionProbeType::Realtime,
                ..ReflectionProbeState::default()
            },
        });
        space
            .pending_reflection_probe_render_changes
            .push(ReflectionProbeChangeRenderTask {
                renderable_index: 0,
                unique_id: 101,
            });

        let results = drain_reflection_probe_render_changes(&mut space);

        assert_eq!(results.completed.len(), 1);
        assert_eq!(results.completed[0].render_space_id, 12);
        assert_eq!(results.completed[0].render_probe_unique_id, 101);
        assert_eq!(results.completed[0].require_reset, 0);
        assert!(results.scene_captures.is_empty());
        assert!(space.pending_reflection_probe_render_changes.is_empty());
    }

    #[test]
    fn changed_missing_probe_returns_reset_completion() {
        let mut space = RenderSpaceState {
            id: crate::scene::RenderSpaceId(13),
            ..RenderSpaceState::default()
        };
        space
            .pending_reflection_probe_render_changes
            .push(ReflectionProbeChangeRenderTask {
                renderable_index: 9,
                unique_id: 55,
            });

        let results = drain_reflection_probe_render_changes(&mut space);

        assert_eq!(results.completed.len(), 1);
        assert_eq!(results.completed[0].render_space_id, 13);
        assert_eq!(results.completed[0].render_probe_unique_id, 55);
        assert_eq!(results.completed[0].require_reset, 1);
        assert!(results.scene_captures.is_empty());
    }
}
