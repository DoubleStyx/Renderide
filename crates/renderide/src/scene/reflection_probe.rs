//! Reflection-probe renderable state mirrored from host updates.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    ReflectionProbeChangeRenderResult, ReflectionProbeChangeRenderTask,
    ReflectionProbeRenderablesUpdate, ReflectionProbeState,
    REFLECTION_PROBE_CHANGE_RENDER_TASK_HOST_ROW_BYTES, REFLECTION_PROBE_STATE_HOST_ROW_BYTES,
};

use super::error::SceneError;
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

/// One dense reflection-probe renderable entry inside a render space.
#[derive(Debug, Clone)]
pub struct ReflectionProbeEntry {
    /// Dense renderable index assigned by the host.
    pub renderable_index: i32,
    /// Dense transform index that owns the probe component.
    pub transform_id: i32,
    /// Latest probe state row sent by the host.
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

/// Returns whether a probe state requests skybox-only rendering.
#[inline]
pub fn reflection_probe_skybox_only(flags: u8) -> bool {
    flags & 0b001 != 0
}

/// Returns whether a probe state requests HDR rendering.
#[inline]
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

/// Applies a pre-extracted reflection-probe update to one render space.
pub(crate) fn apply_reflection_probe_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedReflectionProbeRenderablesUpdate,
) {
    profiling::scope!("scene::apply_reflection_probes");
    space.pending_reflection_probe_render_changes.clear();

    for &raw in extracted.removals.iter().take_while(|&&i| i >= 0) {
        let idx = raw as usize;
        if idx < space.reflection_probes.len() {
            space.reflection_probes.swap_remove(idx);
        }
    }
    for &transform_id in extracted.additions.iter().take_while(|&&i| i >= 0) {
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: -1,
            transform_id,
            state: ReflectionProbeState::default(),
        });
    }
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.reflection_probes.get_mut(idx) else {
            continue;
        };
        entry.renderable_index = state.renderable_index;
        entry.state = *state;
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

/// Converts supported changed-probe render requests into host-visible completion rows.
pub(crate) fn drain_supported_reflection_probe_render_results(
    space: &mut RenderSpaceState,
) -> Vec<ReflectionProbeChangeRenderResult> {
    let mut out = Vec::new();
    for task in space.pending_reflection_probe_render_changes.drain(..) {
        let idx = task.renderable_index as usize;
        let Some(entry) = space.reflection_probes.get(idx) else {
            logger::warn!(
                "reflection probe changed render ignored: render_space={} renderable_index={} not found",
                space.id.0,
                task.renderable_index
            );
            continue;
        };
        if entry.state.clear_flags == crate::shared::ReflectionProbeClear::Color
            || reflection_probe_skybox_only(entry.state.flags)
        {
            out.push(ReflectionProbeChangeRenderResult {
                render_space_id: space.id.0,
                render_probe_unique_id: task.unique_id,
                require_reset: 0,
                _padding: [0; 3],
            });
        } else {
            logger::debug!(
                "reflection probe changed render not completed: render_space={} renderable_index={} requires scene capture",
                space.id.0,
                task.renderable_index
            );
        }
    }
    out
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
    fn changed_skybox_probe_drains_completion_result() {
        let mut space = RenderSpaceState::default();
        space.reflection_probes.push(ReflectionProbeEntry {
            renderable_index: 0,
            transform_id: 1,
            state: ReflectionProbeState {
                renderable_index: 0,
                flags: 0b001,
                ..ReflectionProbeState::default()
            },
        });
        space
            .pending_reflection_probe_render_changes
            .push(ReflectionProbeChangeRenderTask {
                renderable_index: 0,
                unique_id: 77,
            });

        let results = drain_supported_reflection_probe_render_results(&mut space);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].render_probe_unique_id, 77);
        assert!(space.pending_reflection_probe_render_changes.is_empty());
    }
}
