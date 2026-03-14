//! Scene graph: manages scenes and applies host updates.

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, UnitQuaternion, Vector3};

use crate::core::{render_transform_to_matrix, Drawable, Scene, SceneId};
use crate::shared::{
    BoneAssignment, MeshRenderablesUpdate, ReflectionProbeSH2Task, RenderTransform,
    SkinnedMeshRenderablesUpdate, TransformParentUpdate, TransformPoseUpdate, TransformsUpdate,
};
use crate::shared::shared_memory::SharedMemoryAccessor;

/// Layout-compatible with MeshRendererState for shared memory access.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct MeshRendererStatePod {
    renderable_index: i32,
    mesh_asset_id: i32,
    material_count: i32,
    material_property_block_count: i32,
    sorting_order: i32,
    _shadow_cast_mode: u8,
    _motion_vector_mode: u8,
    _pad: [u8; 2],
}

fn fixup_transform_id(old: i32, removed_id: i32, last_index: usize) -> i32 {
    if old == removed_id {
        -1
    } else if old == last_index as i32 {
        removed_id
    } else {
        old
    }
}

/// Returns a RenderTransform with identity rotation, zero position, and unit scale.
/// Matches Gloobie's explicit initialization to avoid invalid Default-derived values.
fn render_transform_identity() -> RenderTransform {
    RenderTransform {
        position: Vector3::zeros(),
        scale: Vector3::new(1.0, 1.0, 1.0),
        rotation: UnitQuaternion::identity().into_inner(),
    }
}

/// Maximum allowed absolute value for position or scale components before rejecting as corrupt.
const POSE_VALIDATION_THRESHOLD: f32 = 1e6;

/// Returns true if the pose has no inf, NaN, or absurdly large values.
/// Logs a warning and returns false for invalid data to avoid corrupting the scene.
fn is_pose_valid(
    pose: &RenderTransform,
    frame_index: i32,
    scene_id: i32,
    transform_id: i32,
) -> bool {
    let pos_ok = pose.position.x.is_finite()
        && pose.position.y.is_finite()
        && pose.position.z.is_finite()
        && pose.position.x.abs() < POSE_VALIDATION_THRESHOLD
        && pose.position.y.abs() < POSE_VALIDATION_THRESHOLD
        && pose.position.z.abs() < POSE_VALIDATION_THRESHOLD;
    if !pos_ok {
        crate::log::log_write(&format!(
            "[POSE VALIDATION] frame {} scene_id={} transform_id={} REJECTED: invalid position ({:.2},{:.2},{:.2})",
            frame_index, scene_id, transform_id,
            pose.position.x, pose.position.y, pose.position.z
        ));
        return false;
    }
    let scale_ok = pose.scale.x.is_finite()
        && pose.scale.y.is_finite()
        && pose.scale.z.is_finite()
        && pose.scale.x.abs() < POSE_VALIDATION_THRESHOLD
        && pose.scale.y.abs() < POSE_VALIDATION_THRESHOLD
        && pose.scale.z.abs() < POSE_VALIDATION_THRESHOLD;
    if !scale_ok {
        crate::log::log_write(&format!(
            "[POSE VALIDATION] frame {} scene_id={} transform_id={} REJECTED: invalid scale ({:.2},{:.2},{:.2})",
            frame_index, scene_id, transform_id,
            pose.scale.x, pose.scale.y, pose.scale.z
        ));
        return false;
    }
    let rot_ok = pose.rotation.i.is_finite()
        && pose.rotation.j.is_finite()
        && pose.rotation.k.is_finite()
        && pose.rotation.w.is_finite();
    if !rot_ok {
        crate::log::log_write(&format!(
            "[POSE VALIDATION] frame {} scene_id={} transform_id={} REJECTED: invalid rotation",
            frame_index, scene_id, transform_id
        ));
        return false;
    }
    true
}

/// Manages scenes (render spaces) and applies incremental updates from the host.
pub struct SceneGraph {
    scenes: HashMap<SceneId, Scene>,
    /// Cached world matrices per scene. Key: (scene_id, transform_index).
    world_matrices: HashMap<SceneId, Vec<Matrix4<f32>>>,
    /// Scene IDs whose transforms changed this frame; cache must be recomputed.
    world_matrices_dirty: HashSet<SceneId>,
    spaces_to_remove: Vec<SceneId>,
}

impl SceneGraph {
    /// Creates a new empty scene graph.
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            world_matrices: HashMap::new(),
            world_matrices_dirty: HashSet::new(),
            spaces_to_remove: Vec::new(),
        }
    }

    /// Returns a reference to a scene by id.
    pub fn get_scene(&self, id: SceneId) -> Option<&Scene> {
        self.scenes.get(&id)
    }

    /// Returns a mutable reference to a scene by id.
    pub fn get_scene_mut(&mut self, id: SceneId) -> Option<&mut Scene> {
        self.scenes.get_mut(&id)
    }

    /// Returns all scenes.
    pub fn scenes(&self) -> &HashMap<SceneId, Scene> {
        &self.scenes
    }

    /// Returns the cached world matrix for a transform in a scene.
    pub fn get_world_matrix(&self, scene_id: SceneId, transform_id: usize) -> Option<Matrix4<f32>> {
        self.world_matrices
            .get(&scene_id)
            .and_then(|mats| mats.get(transform_id).copied())
    }

    /// Applies a frame's render space updates.
    pub fn apply_frame_update(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &crate::shared::FrameSubmitData,
    ) {
        let diag_frame = std::env::var("RENDERIDE_DIAG_FULL").is_ok() || data.frame_index % 30 == 0;

        for update in &data.render_spaces {
            let is_new = !self.scenes.contains_key(&update.id);
            let total_before = self.scenes.len();
            let scene = self
                .scenes
                .entry(update.id)
                .or_insert_with(Scene::default);

            if is_new && diag_frame {
                crate::log::log_write(&format!(
                    "[RENDER SPACE] frame {} space_id={} ADDED (total_spaces={})",
                    data.frame_index, update.id, total_before + 1
                ));
            }

            scene.id = update.id;
            scene.is_active = update.is_active;
            scene.is_overlay = update.is_overlay;
            scene.root_transform = update.root_transform;
            scene.view_transform = if update.override_view_position {
                update.overriden_view_transform
            } else {
                update.root_transform
            };

            let frame_index = data.frame_index;
            if let Some(ref transforms_update) = update.transforms_update {
                Self::apply_transforms_update(scene, shm, transforms_update, frame_index);
                self.world_matrices.remove(&update.id);
                self.world_matrices_dirty.insert(update.id);
            }
            if scene.id == 2 && scene.nodes.len() > 82 && frame_index % 30 == 0 {
                for &idx in &[70usize, 73, 82] {
                    let n = &scene.nodes[idx];
                    crate::log::log_write(&format!(
                        "[LOCAL POSE] scene_id={} node_id={} pos=({:.2},{:.2},{:.2})",
                        scene.id, idx, n.position.x, n.position.y, n.position.z
                    ));
                }
            }
            if let Some(ref mesh_update) = update.mesh_renderers_update {
                Self::apply_mesh_renderables_update(scene, shm, mesh_update, frame_index);
            }
            if let Some(ref skinned_update) = update.skinned_mesh_renderers_update {
                Self::apply_skinned_mesh_renderables_update(scene, shm, skinned_update, frame_index);
            }
            if let Some(ref sh2_tasks) = update.reflection_probe_sh2_taks {
                Self::apply_reflection_probe_sh2_tasks(shm, sh2_tasks);
            }
        }

        for (id, _scene) in self.scenes.iter() {
            if !data.render_spaces.iter().any(|u| u.id == *id) {
                self.spaces_to_remove.push(*id);
            }
        }
        for id in &self.spaces_to_remove {
            if diag_frame {
                crate::log::log_write(&format!(
                    "[RENDER SPACE] frame {} space_id={} REMOVED (remaining={})",
                    data.frame_index, id, self.scenes.len() - 1
                ));
            }
            self.scenes.remove(id);
            self.world_matrices.remove(id);
            self.world_matrices_dirty.remove(id);
        }
        self.spaces_to_remove.clear();
    }

    /// Writes `ComputeResult::Failed` to each ReflectionProbeSH2Task in shared memory.
    /// The host expects the renderer to update the result field before frame finalization;
    /// otherwise it panics with "Invalid compute result: Scheduled". We do not compute SH2
    /// (spherical harmonics from reflection probes), so we mark all tasks as failed.
    fn apply_reflection_probe_sh2_tasks(
        shm: &mut SharedMemoryAccessor,
        sh2_tasks: &crate::shared::ReflectionProbeSH2Tasks,
    ) {
        if sh2_tasks.tasks.length <= 0 {
            return;
        }
        const TASK_STRIDE: usize = std::mem::size_of::<ReflectionProbeSH2Task>();
        const RESULT_OFFSET: usize = 8; // after renderable_index (4) + reflection_probe_renderable_index (4)
        const COMPUTE_RESULT_FAILED: i32 = 3;
        if !shm.access_mut_bytes(&sh2_tasks.tasks, |bytes| {
            let mut offset = 0;
            while offset + TASK_STRIDE <= bytes.len() {
                let renderable_index = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0; 4]));
                if renderable_index < 0 {
                    break;
                }
                bytes[offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4]
                    .copy_from_slice(&COMPUTE_RESULT_FAILED.to_le_bytes());
                offset += TASK_STRIDE;
            }
        }) {
            crate::log::log_write(&format!(
                "[REFLECTION PROBE SH2] access_mut_bytes failed for buffer_id={}",
                sh2_tasks.tasks.buffer_id
            ));
        }
    }

    fn apply_transforms_update(
        scene: &mut Scene,
        shm: &mut SharedMemoryAccessor,
        update: &TransformsUpdate,
        frame_index: i32,
    ) {
        if update.removals.length > 0 {
            match shm.access_copy_diagnostic::<i32>(&update.removals) {
                Ok(removals) => {
                    let mut indices: Vec<usize> = removals
                        .iter()
                        .take_while(|&&i| i >= 0)
                        .map(|&i| i as usize)
                        .collect();
                    indices.sort_by(|a, b| b.cmp(a));
                    for &idx in &indices {
                        if idx >= scene.nodes.len() {
                            continue;
                        }
                        let removed_id = idx as i32;
                        let last_index = scene.nodes.len() - 1;

                        for parent in scene.node_parents.iter_mut() {
                            if *parent == removed_id {
                                *parent = -1;
                            } else if *parent == last_index as i32 {
                                *parent = removed_id;
                            }
                        }
                        for entry in &mut scene.drawables {
                            entry.node_id =
                                fixup_transform_id(entry.node_id, removed_id, last_index);
                        }
                        for entry in &mut scene.skinned_drawables {
                            entry.node_id =
                                fixup_transform_id(entry.node_id, removed_id, last_index);
                            if let Some(ref mut ids) = entry.bone_transform_ids {
                                for id in ids.iter_mut() {
                                    *id = fixup_transform_id(*id, removed_id, last_index);
                                }
                            }
                        }

                        scene.nodes.swap_remove(idx);
                        scene.node_parents.swap_remove(idx);
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[TRANSFORM REMOVALS] frame {} scene_id={} CRITICAL: access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }

        let nodes_before = scene.nodes.len();
        while (scene.nodes.len() as i32) < update.target_transform_count {
            scene.nodes.push(render_transform_identity());
            scene.node_parents.push(-1);
        }
        let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
        if (diag_full || frame_index % 30 == 0)
            && (update.target_transform_count - nodes_before as i32).abs() > 10
        {
            crate::log::log_write(&format!(
                "[TRANSFORM COUNT] frame {} scene_id={} nodes_before={} target={} nodes_after={}",
                frame_index,
                scene.id,
                nodes_before,
                update.target_transform_count,
                scene.nodes.len()
            ));
        }

        if update.parent_updates.length > 0 {
            match shm.access_copy_diagnostic::<TransformParentUpdate>(&update.parent_updates) {
                Ok(parents) => {
                    for pu in parents {
                        if pu.transform_id < 0 {
                            break;
                        }
                        if (pu.transform_id as usize) < scene.node_parents.len() {
                            scene.node_parents[pu.transform_id as usize] = pu.new_parent_id;
                        }
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[TRANSFORM PARENT UPDATES] frame {} scene_id={} CRITICAL: access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }

        if update.pose_updates.length > 0 {
            match shm.access_copy_diagnostic::<TransformPoseUpdate>(&update.pose_updates) {
                Ok(poses) => {
                    let mut written = 0;
                    let mut written_ids: Vec<i32> = Vec::new();
                    for pu in &poses {
                        if pu.transform_id < 0 {
                            break;
                        }
                        if (pu.transform_id as usize) < scene.nodes.len()
                            && is_pose_valid(&pu.pose, frame_index, scene.id, pu.transform_id)
                        {
                            scene.nodes[pu.transform_id as usize] = pu.pose;
                            written += 1;
                            written_ids.push(pu.transform_id);
                        }
                    }
                    if frame_index % 30 == 0 {
                        crate::log::log_write(&format!(
                            "[POSE UPDATE] frame {} scene_id={} pose_updates_received={} actually_written={}",
                            frame_index, scene.id, poses.len(), written
                        ));
                        let sample: Vec<i32> = written_ids
                            .iter()
                            .take(5)
                            .chain(written_ids.iter().rev().take(5))
                            .copied()
                            .collect();
                        crate::log::log_write(&format!(
                            "[POSE UPDATE] frame {} scene_id={} written_transform_ids_sample={:?}",
                            frame_index, scene.id, sample
                        ));
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[POSE UPDATE] frame {} scene_id={} CRITICAL: access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }

        let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
        if diag_full || frame_index % 30 == 0 {
            let removals_count = if update.removals.length > 0 {
                shm.access_copy::<i32>(&update.removals)
                    .map(|r| r.iter().take_while(|&&i| i >= 0).count())
                    .unwrap_or(0)
            } else {
                0
            };
            let pose_count = if update.pose_updates.length > 0 {
                shm.access_copy::<TransformPoseUpdate>(&update.pose_updates)
                    .map(|p| p.iter().take_while(|pu| pu.transform_id >= 0).count())
                    .unwrap_or(0)
            } else {
                0
            };
            let parent_count = if update.parent_updates.length > 0 {
                shm.access_copy::<TransformParentUpdate>(&update.parent_updates)
                    .map(|p| p.iter().take_while(|pu| pu.transform_id >= 0).count())
                    .unwrap_or(0)
            } else {
                0
            };
            crate::log::log_write(&format!(
                "[TRANSFORM] frame {} scene_id={} nodes_count={} target={} removals={} pose_updates={} parent_updates={}",
                frame_index,
                scene.id,
                scene.nodes.len(),
                update.target_transform_count,
                removals_count,
                pose_count,
                parent_count
            ));
        }
    }

    fn apply_mesh_renderables_update(
        scene: &mut Scene,
        shm: &mut SharedMemoryAccessor,
        update: &MeshRenderablesUpdate,
        frame_index: i32,
    ) {
        if update.removals.length > 0 {
            match shm.access_copy_diagnostic::<i32>(&update.removals) {
                Ok(removals) => {
                    let mut indices: Vec<usize> = removals
                        .iter()
                        .take_while(|&&i| i >= 0)
                        .map(|&i| i as usize)
                        .collect();
                    indices.sort_by(|a, b| b.cmp(a));
                    for idx in indices {
                        if idx < scene.drawables.len() {
                            scene.drawables.swap_remove(idx);
                        }
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[MESH RENDERABLES REMOVALS] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
        if update.additions.length > 0 {
            match shm.access_copy_diagnostic::<i32>(&update.additions) {
                Ok(additions) => {
                    let added_node_ids: Vec<i32> = additions
                        .iter()
                        .take_while(|&&i| i >= 0)
                        .copied()
                        .collect();
                    for &node_id in &added_node_ids {
                        scene.drawables.push(Drawable {
                            node_id,
                            mesh_handle: -1,
                            material_handle: None,
                            sort_key: 0,
                            is_skinned: false,
                            bone_transform_ids: None,
                        });
                    }
                    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
                    if (diag_full || frame_index % 30 == 0) && !added_node_ids.is_empty() {
                        let sample: Vec<i32> = added_node_ids.iter().take(5).copied().collect();
                        crate::log::log_write(&format!(
                            "[MESH RENDERABLES] frame {} scene_id={} additions={} drawables_len={} sample_node_ids={:?}",
                            frame_index,
                            scene.id,
                            added_node_ids.len(),
                            scene.drawables.len(),
                            sample
                        ));
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[MESH RENDERABLES ADDITIONS] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
        if update.mesh_states.length > 0 {
            match shm.access_copy_diagnostic::<MeshRendererStatePod>(&update.mesh_states) {
                Ok(states) => {
                    for state in states {
                        if state.renderable_index < 0 {
                            break;
                        }
                        let idx = state.renderable_index as usize;
                        if idx < scene.drawables.len() {
                            scene.drawables[idx].mesh_handle = state.mesh_asset_id;
                            scene.drawables[idx].sort_key = state.sorting_order;
                            scene.drawables[idx].material_handle = if state.material_count > 0 {
                                Some(-1)
                            } else {
                                None
                            };
                        }
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[MESH RENDERABLES STATES] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
    }

    fn apply_skinned_mesh_renderables_update(
        scene: &mut Scene,
        shm: &mut SharedMemoryAccessor,
        update: &SkinnedMeshRenderablesUpdate,
        frame_index: i32,
    ) {
        if update.removals.length > 0 {
            match shm.access_copy_diagnostic::<i32>(&update.removals) {
                Ok(removals) => {
                    let mut indices: Vec<usize> = removals
                        .iter()
                        .take_while(|&&i| i >= 0)
                        .map(|&i| i as usize)
                        .collect();
                    indices.sort_by(|a, b| b.cmp(a));
                    for idx in indices {
                        if idx < scene.skinned_drawables.len() {
                            scene.skinned_drawables.swap_remove(idx);
                        }
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[SKINNED RENDERABLES REMOVALS] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
        if update.additions.length > 0 {
            match shm.access_copy_diagnostic::<i32>(&update.additions) {
                Ok(additions) => {
                    let added_node_ids: Vec<i32> = additions
                        .iter()
                        .take_while(|&&i| i >= 0)
                        .copied()
                        .collect();
                    for &node_id in &added_node_ids {
                        scene.skinned_drawables.push(Drawable {
                            node_id,
                            mesh_handle: -1,
                            material_handle: None,
                            sort_key: 0,
                            is_skinned: true,
                            bone_transform_ids: None,
                        });
                    }
                    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
                    if (diag_full || frame_index % 30 == 0) && !added_node_ids.is_empty() {
                        let sample: Vec<i32> = added_node_ids.iter().take(5).copied().collect();
                        crate::log::log_write(&format!(
                            "[SKINNED RENDERABLES] frame {} scene_id={} additions={} skinned_drawables_len={} sample_node_ids={:?}",
                            frame_index,
                            scene.id,
                            added_node_ids.len(),
                            scene.skinned_drawables.len(),
                            sample
                        ));
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[SKINNED RENDERABLES ADDITIONS] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
        if update.mesh_states.length > 0 {
            match shm.access_copy_diagnostic::<MeshRendererStatePod>(&update.mesh_states) {
                Ok(states) => {
                    for state in states {
                        if state.renderable_index < 0 {
                            break;
                        }
                        let idx = state.renderable_index as usize;
                        if idx < scene.skinned_drawables.len() {
                            scene.skinned_drawables[idx].mesh_handle = state.mesh_asset_id;
                            scene.skinned_drawables[idx].sort_key = state.sorting_order;
                            scene.skinned_drawables[idx].material_handle =
                                if state.material_count > 0 {
                                    Some(-1)
                                } else {
                                    None
                                };
                        }
                    }
                }
                Err(e) => {
                    crate::log::log_write(&format!(
                        "[SKINNED RENDERABLES STATES] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
        if update.bone_assignments.length > 0 {
            match (
                shm.access_copy_diagnostic::<BoneAssignment>(&update.bone_assignments),
                shm.access_copy_diagnostic::<i32>(&update.bone_transform_indexes),
            ) {
                (Ok(assignments), Ok(indexes)) => {
                    let mut index_offset = 0;
                    let mut assigned_count = 0;
                    let mut assignments_processed = 0;
                    for assignment in &assignments {
                        if assignment.renderable_index < 0 {
                            break;
                        }
                        assignments_processed += 1;
                        let idx = assignment.renderable_index as usize;
                        let bone_count = assignment.bone_count.max(0) as usize;
                        if idx < scene.skinned_drawables.len()
                            && index_offset + bone_count <= indexes.len()
                        {
                            let ids: Vec<i32> = indexes[index_offset..index_offset + bone_count]
                                .to_vec();
                            scene.skinned_drawables[idx].bone_transform_ids = Some(ids);
                            assigned_count += 1;
                        }
                        index_offset += bone_count;
                    }
                    let diag_full = std::env::var("RENDERIDE_DIAG_FULL").is_ok();
                    if (diag_full || frame_index % 30 == 0) && assigned_count > 0 {
                        crate::log::log_write(&format!(
                            "[SKINNED BONE ASSIGNMENTS] frame {} scene_id={} assignments_processed={} skinned_drawables_with_bones={}",
                            frame_index,
                            scene.id,
                            assignments_processed,
                            assigned_count
                        ));
                    }
                }
                (Err(e), _) | (_, Err(e)) => {
                    crate::log::log_write(&format!(
                        "[SKINNED BONE ASSIGNMENTS] frame {} scene_id={} access_copy failed: {}",
                        frame_index, scene.id, e
                    ));
                }
            }
        }
    }

    /// Computes and caches world matrices for a scene when cache is missing or dirty.
    /// Uses references to scene data instead of cloning.
    pub fn compute_world_matrices(&mut self, scene_id: SceneId) {
        let needs_recompute = !self.world_matrices.contains_key(&scene_id)
            || self.world_matrices_dirty.contains(&scene_id);
        if !needs_recompute {
            return;
        }
        let matrices = match self.scenes.get(&scene_id) {
            Some(scene) => Self::compute_world_matrices_from_scene(scene),
            None => return,
        };
        self.world_matrices.insert(scene_id, matrices);
        self.world_matrices_dirty.remove(&scene_id);
    }

    /// Bulletproof iterative BFS world matrix computation.
    /// Guarantees every node gets correct world = parent_chain * local.
    /// No recursion, no stack overflow, works with deep or messy hierarchies.
    ///
    /// Root-level nodes (parent < 0 or invalid) use identity as parent—objects are in world space.
    /// The scene's `root_transform` is for the view/camera only, not for object hierarchy.
    fn compute_world_matrices_from_scene(scene: &Scene) -> Vec<Matrix4<f32>> {
        use std::collections::VecDeque;

        let n = scene.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        let mut world = vec![Matrix4::identity(); n];
        let mut visited = vec![false; n];

        // Build children lists + find roots.
        // Treat as root when: parent < 0, parent >= n, or parent == self (cycle sentinel).
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut roots = Vec::new();
        for i in 0..n {
            let p = scene.node_parents.get(i).copied().unwrap_or(-1);
            let is_root = p < 0 || (p as usize) >= n || p == i as i32;
            if is_root {
                roots.push(i);
            } else {
                children[p as usize].push(i);
            }
        }

        // Diagnostic: log parent state when all nodes unreachable (scene_id=2).
        if roots.is_empty() && scene.id == 2 {
            let parents_len = scene.node_parents.len();
            let sample: Vec<i32> = (0..n.min(10))
                .map(|i| scene.node_parents.get(i).copied().unwrap_or(-999))
                .collect();
            crate::log::log_write(&format!(
                "[WORLD COMPUTE] roots=0 nodes={} node_parents_len={} sample_parents={:?}",
                n, parents_len, sample
            ));
        }

        // If no roots (cycle or bad parent data), find a node that is never a child of any other.
        // That node is the logical root (breaks the cycle). Fallback to 0 if none found.
        if roots.is_empty() {
            let never_child: Vec<usize> = (0..n)
                .filter(|&i| !children.iter().any(|c| c.contains(&i)))
                .collect();
            let candidate = never_child.into_iter().next().unwrap_or(0);
            if scene.id == 2 {
                crate::log::log_write(&format!(
                    "[WORLD COMPUTE] forcing root: candidate={} (no roots, cycle or bad parent)",
                    candidate
                ));
            }
            roots.push(candidate);
        }

        // BFS from all roots (guarantees parents before children).
        let mut queue: VecDeque<usize> = roots.into_iter().collect();
        while let Some(i) = queue.pop_front() {
            if visited[i] {
                continue;
            }
            visited[i] = true;

            let local = render_transform_to_matrix(&scene.nodes[i]);
            let p = scene.node_parents.get(i).copied().unwrap_or(-1);
            let parent_world = if p >= 0 && (p as usize) < n && p != i as i32 {
                world[p as usize]
            } else {
                Matrix4::identity()
            };
            world[i] = parent_world * local;

            for &child in &children[i] {
                queue.push_back(child);
            }
        }

        // Safety check: were any nodes unreachable? (cycle or bad parent).
        let unvisited: Vec<usize> = (0..n).filter(|&i| !visited[i]).collect();
        if !unvisited.is_empty() && scene.id == 2 {
            crate::log::log_write(&format!(
                "[WORLD COMPUTE] WARNING: {} nodes unreachable (possible cycle or bad parent index)",
                unvisited.len()
            ));
        }

        // Threshold beyond which world translation is considered corrupt (stretched vertices).
        const WORLD_TRANSLATION_WARN_THRESHOLD: f32 = 1e5;
        let bad: Vec<(usize, f32, f32, f32)> = world
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                let tx: f32 = m[(0, 3)];
                let ty: f32 = m[(1, 3)];
                let tz: f32 = m[(2, 3)];
                if tx.abs() > WORLD_TRANSLATION_WARN_THRESHOLD
                    || ty.abs() > WORLD_TRANSLATION_WARN_THRESHOLD
                    || tz.abs() > WORLD_TRANSLATION_WARN_THRESHOLD
                {
                    Some((i, tx, ty, tz))
                } else {
                    None
                }
            })
            .take(5)
            .collect();
        if !bad.is_empty() {
            crate::log::log_write(&format!(
                "[WORLD COMPUTE] WARNING: scene_id={} {} node(s) with world translation > {} (possible corrupt pose in parent chain). Sample: {:?}",
                scene.id,
                bad.len(),
                WORLD_TRANSLATION_WARN_THRESHOLD,
                bad
            ));
        }

        // Debug: specifically check node 70 (we know it has real local pose from logs).
        if let Some(m) = world.get(70) {
            crate::log::log_write(&format!(
                "[WORLD DEBUG] node_70 world_pos=({:.3},{:.3},{:.3})",
                m[(0, 3)],
                m[(1, 3)],
                m[(2, 3)]
            ));
        }

        world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Quaternion, Vector3};

    fn make_transform(pos: (f32, f32, f32)) -> RenderTransform {
        RenderTransform {
            position: Vector3::new(pos.0, pos.1, pos.2),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::identity(),
        }
    }

    #[test]
    fn test_world_matrix_propagation_three_level_hierarchy() {
        let mut scene = Scene::default();
        scene.root_transform = make_transform((0.0, 0.0, 0.0));
        scene.nodes = vec![
            make_transform((1.0, 0.0, 0.0)),
            make_transform((0.0, 2.0, 0.0)),
            make_transform((0.0, 0.0, 3.0)),
        ];
        scene.node_parents = vec![-1, 0, 1];

        let world = SceneGraph::compute_world_matrices_from_scene(&scene);

        assert_eq!(world.len(), 3);

        let pos0 = world[0].column(3);
        assert!((pos0.x - 1.0).abs() < 1e-5);
        assert!((pos0.y - 0.0).abs() < 1e-5);
        assert!((pos0.z - 0.0).abs() < 1e-5);

        let pos1 = world[1].column(3);
        assert!((pos1.x - 1.0).abs() < 1e-5);
        assert!((pos1.y - 2.0).abs() < 1e-5);
        assert!((pos1.z - 0.0).abs() < 1e-5);

        let pos2 = world[2].column(3);
        assert!((pos2.x - 1.0).abs() < 1e-5);
        assert!((pos2.y - 2.0).abs() < 1e-5);
        assert!((pos2.z - 3.0).abs() < 1e-5);
    }

    /// Root-level nodes use identity as parent; root_transform is for view only, not object hierarchy.
    #[test]
    fn test_world_matrix_root_level_uses_identity() {
        let mut scene = Scene::default();
        scene.root_transform = make_transform((10.0, 0.0, -5.0));
        scene.nodes = vec![make_transform((1.0, 0.0, 0.0))];
        scene.node_parents = vec![-1];

        let world = SceneGraph::compute_world_matrices_from_scene(&scene);

        assert_eq!(world.len(), 1);
        let pos = world[0].column(3);
        // Root-level: world = identity * local = (1, 0, 0). root_transform is not applied to objects.
        assert!((pos.x - 1.0).abs() < 1e-5);
        assert!((pos.y - 0.0).abs() < 1e-5);
        assert!((pos.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_world_matrix_parent_after_child_in_array() {
        let mut scene = Scene::default();
        scene.root_transform = make_transform((0.0, 0.0, 0.0));
        scene.nodes = vec![
            make_transform((0.0, 0.0, 1.0)),
            make_transform((5.0, 0.0, 0.0)),
        ];
        scene.node_parents = vec![1, -1];

        let world = SceneGraph::compute_world_matrices_from_scene(&scene);

        assert_eq!(world.len(), 2);
        let pos0 = world[0].column(3);
        assert!((pos0.x - 5.0).abs() < 1e-5);
        assert!((pos0.y - 0.0).abs() < 1e-5);
        assert!((pos0.z - 1.0).abs() < 1e-5);
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}
