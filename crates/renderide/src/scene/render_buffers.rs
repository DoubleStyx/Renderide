//! PhotonDust render-buffer renderer rows mirrored from host render-space updates.

use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::{
    for_each_row_with_par_dispatch, non_negative_i32s, swap_remove_dense_indices,
};
use crate::scene::error::SceneError;
use crate::scene::render_space::RenderSpaceState;
use crate::scene::transforms::TransformRemovalEvent;
use crate::scene::world::fixup_transform_id;
use crate::shared::{
    BILLBOARD_RENDER_BUFFER_STATE_HOST_ROW_BYTES, BillboardAlignment, BillboardRenderBufferState,
    BillboardRenderBufferUpdate, MESH_RENDER_BUFFER_STATE_HOST_ROW_BYTES, MeshAlignment,
    MeshRenderBufferState, MeshRenderBufferUpdate, MotionVectorMode,
    TRAILS_RENDERER_STATE_HOST_ROW_BYTES, TrailTextureMode, TrailsRendererState,
    TrailsRendererUpdate,
};

/// One PhotonDust billboard renderer attached to a dense scene transform.
#[derive(Clone, Copy, Debug)]
pub struct BillboardRenderBufferEntry {
    /// Dense transform index this renderer attaches to.
    pub node_id: i32,
    /// Point render-buffer asset id consumed by this renderer.
    pub point_render_buffer_asset_id: i32,
    /// Material asset id assigned to the billboard renderer.
    pub material_asset_id: i32,
    /// Minimum billboard screen size requested by the host.
    pub min_billboard_screen_size: f32,
    /// Maximum billboard screen size requested by the host.
    pub max_billboard_screen_size: f32,
    /// Billboard orientation mode requested by the host.
    pub alignment: BillboardAlignment,
    /// Motion-vector mode requested by the host.
    pub motion_vector_mode: MotionVectorMode,
}

impl Default for BillboardRenderBufferEntry {
    fn default() -> Self {
        Self {
            node_id: -1,
            point_render_buffer_asset_id: -1,
            material_asset_id: -1,
            min_billboard_screen_size: 0.0,
            max_billboard_screen_size: 0.0,
            alignment: BillboardAlignment::default(),
            motion_vector_mode: MotionVectorMode::default(),
        }
    }
}

/// One PhotonDust mesh-particle renderer attached to a dense scene transform.
#[derive(Clone, Copy, Debug)]
pub struct MeshRenderBufferEntry {
    /// Dense transform index this renderer attaches to.
    pub node_id: i32,
    /// Point render-buffer asset id consumed by this renderer.
    pub point_render_buffer_asset_id: i32,
    /// Material asset id assigned to the mesh-particle renderer.
    pub material_asset_id: i32,
    /// Source mesh asset id instanced by this renderer.
    pub mesh_asset_id: i32,
    /// Mesh-particle orientation mode requested by the host.
    pub alignment: MeshAlignment,
}

impl Default for MeshRenderBufferEntry {
    fn default() -> Self {
        Self {
            node_id: -1,
            point_render_buffer_asset_id: -1,
            material_asset_id: -1,
            mesh_asset_id: -1,
            alignment: MeshAlignment::default(),
        }
    }
}

/// One PhotonDust trail renderer attached to a dense scene transform.
#[derive(Clone, Copy, Debug)]
pub struct TrailRenderBufferEntry {
    /// Dense transform index this renderer attaches to.
    pub node_id: i32,
    /// Trail render-buffer asset id consumed by this renderer.
    pub trails_render_buffer_asset_id: i32,
    /// Material asset id assigned to the trail renderer.
    pub material_asset_id: i32,
    /// Texture-coordinate generation mode for the generated ribbon mesh.
    pub texture_mode: TrailTextureMode,
    /// Motion-vector mode requested by the host.
    pub motion_vector_mode: MotionVectorMode,
    /// Whether lighting data should be available for the trail mesh.
    pub generate_lighting_data: bool,
}

impl Default for TrailRenderBufferEntry {
    fn default() -> Self {
        Self {
            node_id: -1,
            trails_render_buffer_asset_id: -1,
            material_asset_id: -1,
            texture_mode: TrailTextureMode::default(),
            motion_vector_mode: MotionVectorMode::default(),
            generate_lighting_data: false,
        }
    }
}

/// Owned billboard render-buffer renderer update payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedBillboardRenderBufferUpdate {
    /// Billboard renderer removal indices.
    pub removals: Vec<i32>,
    /// Billboard renderer addition transform ids.
    pub additions: Vec<i32>,
    /// Per-renderer host state rows.
    pub states: Vec<BillboardRenderBufferState>,
}

/// Owned mesh render-buffer renderer update payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedMeshRenderBufferUpdate {
    /// Mesh-particle renderer removal indices.
    pub removals: Vec<i32>,
    /// Mesh-particle renderer addition transform ids.
    pub additions: Vec<i32>,
    /// Per-renderer host state rows.
    pub states: Vec<MeshRenderBufferState>,
}

/// Owned trail renderer update payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedTrailRendererUpdate {
    /// Trail renderer removal indices.
    pub removals: Vec<i32>,
    /// Trail renderer addition transform ids.
    pub additions: Vec<i32>,
    /// Per-renderer host state rows.
    pub states: Vec<TrailsRendererState>,
}

/// Reads every shared-memory buffer referenced by [`BillboardRenderBufferUpdate`].
pub(crate) fn extract_billboard_render_buffer_update(
    shm: &mut SharedMemoryAccessor,
    update: &BillboardRenderBufferUpdate,
    scene_id: i32,
) -> Result<ExtractedBillboardRenderBufferUpdate, SceneError> {
    let mut out = ExtractedBillboardRenderBufferUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("billboard render-buffer removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("billboard render-buffer additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("billboard render-buffer states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<BillboardRenderBufferState>(
                &update.states,
                BILLBOARD_RENDER_BUFFER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Reads every shared-memory buffer referenced by [`MeshRenderBufferUpdate`].
pub(crate) fn extract_mesh_render_buffer_update(
    shm: &mut SharedMemoryAccessor,
    update: &MeshRenderBufferUpdate,
    scene_id: i32,
) -> Result<ExtractedMeshRenderBufferUpdate, SceneError> {
    let mut out = ExtractedMeshRenderBufferUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("mesh render-buffer removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("mesh render-buffer additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("mesh render-buffer states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<MeshRenderBufferState>(
                &update.states,
                MESH_RENDER_BUFFER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Reads every shared-memory buffer referenced by [`TrailsRendererUpdate`].
pub(crate) fn extract_trail_renderer_update(
    shm: &mut SharedMemoryAccessor,
    update: &TrailsRendererUpdate,
    scene_id: i32,
) -> Result<ExtractedTrailRendererUpdate, SceneError> {
    let mut out = ExtractedTrailRendererUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("trail renderer removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("trail renderer additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("trail renderer states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<TrailsRendererState>(
                &update.states,
                TRAILS_RENDERER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Mutates billboard renderer rows using a pre-extracted payload.
pub(crate) fn apply_billboard_render_buffer_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedBillboardRenderBufferUpdate,
) {
    profiling::scope!("scene::apply_billboard_render_buffers");
    swap_remove_dense_indices(&mut space.billboard_render_buffers, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        space
            .billboard_render_buffers
            .push(BillboardRenderBufferEntry {
                node_id,
                ..Default::default()
            });
    }
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let Some(entry) = space
            .billboard_render_buffers
            .get_mut(state.renderable_index as usize)
        else {
            continue;
        };
        entry.point_render_buffer_asset_id = state.point_render_buffer_asset_id;
        entry.material_asset_id = state.material_asset_id;
        entry.min_billboard_screen_size = state.min_billboard_screen_size;
        entry.max_billboard_screen_size = state.max_billboard_screen_size;
        entry.alignment = state.alignment;
        entry.motion_vector_mode = state.motion_vector_mode;
    }
}

/// Mutates mesh-particle renderer rows using a pre-extracted payload.
pub(crate) fn apply_mesh_render_buffer_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedMeshRenderBufferUpdate,
) {
    profiling::scope!("scene::apply_mesh_render_buffers");
    swap_remove_dense_indices(&mut space.mesh_render_buffers, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        space.mesh_render_buffers.push(MeshRenderBufferEntry {
            node_id,
            ..Default::default()
        });
    }
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let Some(entry) = space
            .mesh_render_buffers
            .get_mut(state.renderable_index as usize)
        else {
            continue;
        };
        entry.point_render_buffer_asset_id = state.point_render_buffer_asset_id;
        entry.material_asset_id = state.material_asset_id;
        entry.mesh_asset_id = state.mesh_asset_id;
        entry.alignment = state.alignment;
    }
}

/// Mutates trail renderer rows using a pre-extracted payload.
pub(crate) fn apply_trail_renderer_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedTrailRendererUpdate,
) {
    profiling::scope!("scene::apply_trail_renderers");
    swap_remove_dense_indices(&mut space.trail_render_buffers, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        space.trail_render_buffers.push(TrailRenderBufferEntry {
            node_id,
            ..Default::default()
        });
    }
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let Some(entry) = space
            .trail_render_buffers
            .get_mut(state.renderable_index as usize)
        else {
            continue;
        };
        entry.trails_render_buffer_asset_id = state.trails_render_buffer_asset_id;
        entry.material_asset_id = state.material_asset_id;
        entry.texture_mode = state.texture_mode;
        entry.motion_vector_mode = state.motion_vector_mode;
        entry.generate_lighting_data = state.generate_lighting_data != 0;
    }
}

/// Remaps render-buffer renderer transform ids after dense transform swap-removals.
pub(crate) fn fixup_render_buffers_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() {
        return;
    }
    for removal in removals {
        let removed_id = removal.removed_index;
        let last_index = removal.last_index_before_swap;
        for_each_row_with_par_dispatch(&mut space.billboard_render_buffers, |entry| {
            entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
        });
        for_each_row_with_par_dispatch(&mut space.mesh_render_buffers, |entry| {
            entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
        });
        for_each_row_with_par_dispatch(&mut space.trail_render_buffers, |entry| {
            entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
        });
        space
            .billboard_render_buffers
            .retain(|entry| entry.node_id >= 0);
        space.mesh_render_buffers.retain(|entry| entry.node_id >= 0);
        space
            .trail_render_buffers
            .retain(|entry| entry.node_id >= 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn billboard_state_updates_existing_row() {
        let mut space = RenderSpaceState::default();
        apply_billboard_render_buffer_update_extracted(
            &mut space,
            &ExtractedBillboardRenderBufferUpdate {
                additions: vec![7, -1],
                states: vec![BillboardRenderBufferState {
                    renderable_index: 0,
                    point_render_buffer_asset_id: 22,
                    material_asset_id: 33,
                    min_billboard_screen_size: 1.0,
                    max_billboard_screen_size: 4.0,
                    alignment: BillboardAlignment::Facing,
                    motion_vector_mode: MotionVectorMode::Object,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        assert_eq!(space.billboard_render_buffers.len(), 1);
        let row = space.billboard_render_buffers[0];
        assert_eq!(row.node_id, 7);
        assert_eq!(row.point_render_buffer_asset_id, 22);
        assert_eq!(row.material_asset_id, 33);
        assert_eq!(row.alignment, BillboardAlignment::Facing);
    }

    #[test]
    fn trail_fixup_removes_deleted_transform_rows() {
        let mut space = RenderSpaceState {
            trail_render_buffers: vec![
                TrailRenderBufferEntry {
                    node_id: 0,
                    ..Default::default()
                },
                TrailRenderBufferEntry {
                    node_id: 2,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        fixup_render_buffers_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 2,
                last_index_before_swap: 2,
            }],
        );

        assert_eq!(space.trail_render_buffers.len(), 1);
        assert_eq!(space.trail_render_buffers[0].node_id, 0);
    }
}
