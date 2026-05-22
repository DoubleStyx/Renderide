//! Per-render-space mesh-swap LOD group mirror state.

use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::{non_negative_i32s, swap_remove_dense_indices};
use crate::scene::error::SceneError;
use crate::scene::meshes::types::MeshRendererInstanceId;
use crate::scene::overrides::{MeshRendererOverrideTarget, decode_packed_mesh_renderer_target};
use crate::scene::render_space::RenderSpaceState;
use crate::shared::packing_extras::LOD_GROUP_STATE_HOST_ROW_BYTES;
use crate::shared::{LODGroupRenderablesUpdate, LODGroupState, LODState};

/// Static or skinned renderer table selected by a packed LOD renderer id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LodRendererKind {
    /// Static mesh renderer table.
    Static,
    /// Skinned mesh renderer table.
    Skinned,
}

/// Renderer membership in one LOD entry, keyed by stable renderer-local identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LodRendererRef {
    /// Static versus skinned renderer list.
    pub(crate) kind: LodRendererKind,
    /// Stable identity assigned when the renderer entry was created.
    pub(crate) instance_id: MeshRendererInstanceId,
    /// Dense index from the host row at apply time, used as a fast lookup hint.
    pub(crate) renderable_index_hint: usize,
}

/// One ordered LOD entry for a group.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LodEntry {
    /// Unity-style relative screen-height threshold for this LOD.
    pub(crate) screen_relative_transition_height: f32,
    /// Host fade width. Stored for parity; mesh-swap selection does not consume it yet.
    pub(crate) fade_transition_width: f32,
    /// Renderers shown when this LOD is selected.
    pub(crate) renderers: Vec<LodRendererRef>,
}

/// One host LOD group in a render space.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LodGroupEntry {
    /// Dense transform index that owns the LOD group component.
    pub(crate) node_id: i32,
    /// Host cross-fade flag. Stored for future fade support.
    pub(crate) cross_fade: bool,
    /// Host animated cross-fade flag. Stored for future fade support.
    pub(crate) animate_cross_fading: bool,
    /// Ordered LOD entries from nearest/highest-detail to farthest/lowest-detail.
    pub(crate) lods: Vec<LodEntry>,
}

/// Owned per-space LOD group payload extracted from shared memory.
#[derive(Default, Debug)]
pub(in crate::scene) struct ExtractedLodGroupRenderablesUpdate {
    /// Dense LOD-group removal indices, terminated by `< 0`.
    pub(in crate::scene) removals: Vec<i32>,
    /// New LOD-group node ids, terminated by `< 0`.
    pub(in crate::scene) additions: Vec<i32>,
    /// Per-group state rows, terminated by `renderable_index < 0`.
    pub(in crate::scene) states: Vec<LODGroupState>,
    /// Sequential LOD rows consumed by [`LODGroupState::lod_count`].
    pub(in crate::scene) lod_states: Vec<LODState>,
    /// Sequential packed static/skinned renderer ids consumed by [`LODState::renderer_count`].
    pub(in crate::scene) packed_mesh_renderer_ids: Vec<i32>,
}

/// Reads every shared-memory buffer referenced by [`LODGroupRenderablesUpdate`].
pub(in crate::scene) fn extract_lod_group_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &LODGroupRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedLodGroupRenderablesUpdate, SceneError> {
    let mut out = ExtractedLodGroupRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("lod_group removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("lod_group additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("lod_group states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<LODGroupState>(
                &update.states,
                LOD_GROUP_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.lod_states.length > 0 {
        let ctx = format!("lod_group lod_states scene_id={scene_id}");
        out.lod_states = shm
            .access_copy_diagnostic_with_context::<LODState>(&update.lod_states, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.packed_mesh_renderer_ids.length > 0 {
        let ctx = format!("lod_group packed_mesh_renderer_ids scene_id={scene_id}");
        out.packed_mesh_renderer_ids = shm
            .access_copy_diagnostic_with_context::<i32>(
                &update.packed_mesh_renderer_ids,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Applies an extracted LOD group update to one render space.
pub(in crate::scene) fn apply_lod_group_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedLodGroupRenderablesUpdate,
    scene_id: i32,
) {
    profiling::scope!("scene::apply_lod_groups");

    swap_remove_dense_indices(&mut space.lod_groups, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        space.lod_groups.push(LodGroupEntry {
            node_id,
            ..Default::default()
        });
    }

    let mut lod_cursor = 0usize;
    let mut renderer_cursor = 0usize;
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let lods = consume_lod_entries(
            space,
            extracted,
            state.lod_count.max(0) as usize,
            &mut lod_cursor,
            &mut renderer_cursor,
            scene_id,
        );
        let Some(group) = space.lod_groups.get_mut(idx) else {
            logger::warn!(
                "LOD group state ignored: scene_id={} renderable_index={} group_count={}",
                scene_id,
                idx,
                space.lod_groups.len()
            );
            continue;
        };
        group.cross_fade = state.cross_fade;
        group.animate_cross_fading = state.animate_cross_fading;
        group.lods = lods;
    }
}

/// Consumes the ordered LOD rows and renderer id rows for one group state.
fn consume_lod_entries(
    space: &RenderSpaceState,
    extracted: &ExtractedLodGroupRenderablesUpdate,
    lod_count: usize,
    lod_cursor: &mut usize,
    renderer_cursor: &mut usize,
    scene_id: i32,
) -> Vec<LodEntry> {
    let mut lods = Vec::with_capacity(lod_count);
    for _ in 0..lod_count {
        let Some(lod_state) = extracted.lod_states.get(*lod_cursor) else {
            logger::warn!(
                "LOD group row truncated: scene_id={} requested_lod_rows={} available_lod_rows={}",
                scene_id,
                lod_count,
                extracted.lod_states.len()
            );
            break;
        };
        *lod_cursor += 1;
        let renderer_count = lod_state.renderer_count.max(0) as usize;
        let renderers =
            consume_lod_renderers(space, extracted, renderer_count, renderer_cursor, scene_id);
        lods.push(LodEntry {
            screen_relative_transition_height: lod_state.screen_relative_transition_height,
            fade_transition_width: lod_state.fade_transition_width,
            renderers,
        });
    }
    lods
}

/// Consumes and resolves renderer ids for one LOD row.
fn consume_lod_renderers(
    space: &RenderSpaceState,
    extracted: &ExtractedLodGroupRenderablesUpdate,
    renderer_count: usize,
    renderer_cursor: &mut usize,
    scene_id: i32,
) -> Vec<LodRendererRef> {
    let mut renderers = Vec::with_capacity(renderer_count);
    for _ in 0..renderer_count {
        let Some(packed) = extracted
            .packed_mesh_renderer_ids
            .get(*renderer_cursor)
            .copied()
        else {
            logger::warn!(
                "LOD group renderer id slab truncated: scene_id={} requested_renderer_rows={} available_renderer_rows={}",
                scene_id,
                renderer_count,
                extracted.packed_mesh_renderer_ids.len()
            );
            break;
        };
        *renderer_cursor += 1;
        if let Some(renderer_ref) = resolve_lod_renderer_ref(space, packed) {
            renderers.push(renderer_ref);
        }
    }
    renderers
}

/// Resolves one packed static/skinned renderer id into a stable LOD renderer reference.
fn resolve_lod_renderer_ref(space: &RenderSpaceState, packed: i32) -> Option<LodRendererRef> {
    match decode_packed_mesh_renderer_target(packed) {
        MeshRendererOverrideTarget::Static(index) => {
            let renderable_index_hint = usize::try_from(index).ok()?;
            let renderer = space.static_mesh_renderers.get(renderable_index_hint)?;
            Some(LodRendererRef {
                kind: LodRendererKind::Static,
                instance_id: renderer.instance_id,
                renderable_index_hint,
            })
        }
        MeshRendererOverrideTarget::Skinned(index) => {
            let renderable_index_hint = usize::try_from(index).ok()?;
            let renderer = space.skinned_mesh_renderers.get(renderable_index_hint)?;
            Some(LodRendererRef {
                kind: LodRendererKind::Skinned,
                instance_id: renderer.base.instance_id,
                renderable_index_hint,
            })
        }
        MeshRendererOverrideTarget::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::meshes::types::{SkinnedMeshRenderer, StaticMeshRenderer};

    /// Builds a static renderer with a stable instance id and node.
    fn static_renderer(id: u64, node_id: i32) -> StaticMeshRenderer {
        StaticMeshRenderer {
            instance_id: MeshRendererInstanceId(id),
            node_id,
            ..Default::default()
        }
    }

    /// Builds a skinned renderer with a stable instance id and node.
    fn skinned_renderer(id: u64, node_id: i32) -> SkinnedMeshRenderer {
        SkinnedMeshRenderer {
            base: static_renderer(id, node_id),
            ..Default::default()
        }
    }

    /// Host packed id for a static renderer.
    fn packed_static(index: i32) -> i32 {
        index
    }

    /// Host packed id for a skinned renderer.
    fn packed_skinned(index: i32) -> i32 {
        (1i32 << 30) | index
    }

    #[test]
    fn additions_grow_lod_group_table() {
        let mut space = RenderSpaceState::default();
        let extracted = ExtractedLodGroupRenderablesUpdate {
            additions: vec![7, 8, -1],
            ..Default::default()
        };

        apply_lod_group_renderables_update_extracted(&mut space, &extracted, 1);

        assert_eq!(space.lod_groups.len(), 2);
        assert_eq!(space.lod_groups[0].node_id, 7);
        assert_eq!(space.lod_groups[1].node_id, 8);
    }

    #[test]
    fn removals_use_dense_swap_remove() {
        let mut space = RenderSpaceState {
            lod_groups: vec![
                LodGroupEntry {
                    node_id: 10,
                    ..Default::default()
                },
                LodGroupEntry {
                    node_id: 11,
                    ..Default::default()
                },
                LodGroupEntry {
                    node_id: 12,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let extracted = ExtractedLodGroupRenderablesUpdate {
            removals: vec![0, -1],
            ..Default::default()
        };

        apply_lod_group_renderables_update_extracted(&mut space, &extracted, 1);

        assert_eq!(space.lod_groups.len(), 2);
        assert_eq!(space.lod_groups[0].node_id, 12);
        assert_eq!(space.lod_groups[1].node_id, 11);
    }

    #[test]
    fn state_consumes_lod_and_renderer_slabs_in_host_order() {
        let mut space = RenderSpaceState {
            lod_groups: vec![LodGroupEntry {
                node_id: 20,
                ..Default::default()
            }],
            static_mesh_renderers: vec![static_renderer(100, 1), static_renderer(101, 2)],
            skinned_mesh_renderers: vec![skinned_renderer(200, 3)],
            ..Default::default()
        };
        let extracted = ExtractedLodGroupRenderablesUpdate {
            states: vec![
                LODGroupState {
                    renderable_index: 0,
                    lod_count: 2,
                    cross_fade: true,
                    animate_cross_fading: true,
                },
                LODGroupState {
                    renderable_index: -1,
                    ..Default::default()
                },
            ],
            lod_states: vec![
                LODState {
                    screen_relative_transition_height: 0.7,
                    fade_transition_width: 0.1,
                    renderer_count: 2,
                },
                LODState {
                    screen_relative_transition_height: 0.2,
                    fade_transition_width: 0.0,
                    renderer_count: 1,
                },
            ],
            packed_mesh_renderer_ids: vec![packed_static(0), packed_skinned(0), packed_static(1)],
            ..Default::default()
        };

        apply_lod_group_renderables_update_extracted(&mut space, &extracted, 42);

        let group = &space.lod_groups[0];
        assert!(group.cross_fade);
        assert!(group.animate_cross_fading);
        assert_eq!(group.lods.len(), 2);
        assert_eq!(group.lods[0].screen_relative_transition_height, 0.7);
        assert_eq!(group.lods[0].renderers.len(), 2);
        assert_eq!(
            group.lods[0].renderers[0].instance_id,
            MeshRendererInstanceId(100)
        );
        assert_eq!(
            group.lods[0].renderers[1].instance_id,
            MeshRendererInstanceId(200)
        );
        assert_eq!(
            group.lods[1].renderers[0].instance_id,
            MeshRendererInstanceId(101)
        );
    }

    #[test]
    fn renderer_refs_keep_instance_ids_after_renderer_swap_remove() {
        let mut space = RenderSpaceState {
            lod_groups: vec![LodGroupEntry {
                node_id: 20,
                ..Default::default()
            }],
            static_mesh_renderers: vec![
                static_renderer(100, 1),
                static_renderer(101, 2),
                static_renderer(102, 3),
            ],
            ..Default::default()
        };
        let extracted = ExtractedLodGroupRenderablesUpdate {
            states: vec![LODGroupState {
                renderable_index: 0,
                lod_count: 1,
                ..Default::default()
            }],
            lod_states: vec![LODState {
                screen_relative_transition_height: 0.5,
                fade_transition_width: 0.0,
                renderer_count: 1,
            }],
            packed_mesh_renderer_ids: vec![packed_static(1)],
            ..Default::default()
        };

        apply_lod_group_renderables_update_extracted(&mut space, &extracted, 1);
        space.static_mesh_renderers.swap_remove(0);

        assert_eq!(
            space.lod_groups[0].lods[0].renderers[0].instance_id,
            MeshRendererInstanceId(101)
        );
        assert_eq!(
            space.static_mesh_renderers[0].instance_id,
            MeshRendererInstanceId(102)
        );
    }
}
