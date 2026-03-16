//! Scene graph: manages scenes and applies host updates.
//!
//! Extension point for scene graph, hierarchy.

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, UnitQuaternion, Vector3};

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Drawable, Scene, SceneId};
use crate::shared::{
    BoneAssignment, MeshRenderablesUpdate, ReflectionProbeSH2Task, RenderTransform,
    SkinnedMeshRenderablesUpdate, TransformParentUpdate, TransformPoseUpdate, TransformsUpdate,
};

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

/// Validates pose data; rejects NaN, inf, or values with abs > 1e6.
pub struct PoseValidation<'a> {
    pose: &'a RenderTransform,
    /// Frame index for error logging context.
    pub frame_index: i32,
    /// Scene ID for error logging context.
    pub scene_id: i32,
    /// Transform ID for error logging context.
    pub transform_id: i32,
}

impl PoseValidation<'_> {
    /// Returns true if the pose has no inf, NaN, or absurdly large values.
    pub fn is_valid(&self) -> bool {
        let pos_ok = self.pose.position.x.is_finite()
            && self.pose.position.y.is_finite()
            && self.pose.position.z.is_finite()
            && self.pose.position.x.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.position.y.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.position.z.abs() < POSE_VALIDATION_THRESHOLD;
        if !pos_ok {
            return false;
        }
        let scale_ok = self.pose.scale.x.is_finite()
            && self.pose.scale.y.is_finite()
            && self.pose.scale.z.is_finite()
            && self.pose.scale.x.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.scale.y.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.scale.z.abs() < POSE_VALIDATION_THRESHOLD;
        if !scale_ok {
            return false;
        }
        
        self.pose.rotation.i.is_finite()
            && self.pose.rotation.j.is_finite()
            && self.pose.rotation.k.is_finite()
            && self.pose.rotation.w.is_finite()
    }
}

/// Error returned by scene graph operations.
#[derive(Debug)]
pub enum SceneError {
    /// Shared memory access failed.
    SharedMemoryAccess(String),
    /// Cycle detected in transform hierarchy.
    CycleDetected { scene_id: i32, transform_id: i32 },
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneError::SharedMemoryAccess(msg) => write!(f, "Shared memory access: {}", msg),
            SceneError::CycleDetected {
                scene_id,
                transform_id,
            } => {
                write!(
                    f,
                    "Cycle detected in scene {} at transform {}",
                    scene_id, transform_id
                )
            }
        }
    }
}

impl std::error::Error for SceneError {}

/// Per-scene cache for world matrices and computed flags.
struct SceneCache {
    world_matrices: Vec<Matrix4<f32>>,
    computed: Vec<bool>,
}

/// Manages scenes (render spaces) and applies incremental updates from the host.
pub struct SceneGraph {
    scenes: HashMap<SceneId, Scene>,
    /// Per-scene cache: world matrices and computed flags.
    scene_caches: HashMap<SceneId, SceneCache>,
    /// Scene IDs that need at least one transform recomputed.
    world_matrices_dirty: HashSet<SceneId>,
    spaces_to_remove: Vec<SceneId>,
}

impl SceneGraph {
    /// Creates a new empty scene graph.
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            scene_caches: HashMap::new(),
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

    /// Test-only: marks a transform and its descendants uncomputed to exercise incremental recompute.
    #[cfg(test)]
    pub fn test_invalidate_transform(&mut self, scene_id: SceneId, transform_id: usize) {
        if let Some(cache) = self.scene_caches.get_mut(&scene_id) {
            if transform_id < cache.computed.len() {
                cache.computed[transform_id] = false;
                if let Some(scene) = self.scenes.get(&scene_id) {
                    Self::mark_descendants_uncomputed(
                        &scene.node_parents,
                        &mut cache.computed,
                    );
                }
                self.world_matrices_dirty.insert(scene_id);
            }
        }
    }

    /// Returns the cached world matrix for a transform in a scene.
    pub fn get_world_matrix(&self, scene_id: SceneId, transform_id: usize) -> Option<Matrix4<f32>> {
        self.scene_caches
            .get(&scene_id)
            .and_then(|c| c.world_matrices.get(transform_id).copied())
    }

    /// Computes bone matrices for skinned mesh rendering.
    /// Each output matrix is `world_matrix(bone_id) * bind_pose[i]`, or when `root_bone_transform_id`
    /// is provided: `world[bone] * inverse(world[root]) * bind_pose` for root-relative alignment.
    ///
    /// The host sends bind poses in Unity-style column-major format (see Unity Mesh.bindposes):
    /// each matrix is 64 bytes with column 0 (4 floats), then column 1, then column 2, then column 3.
    /// After `bytemuck::pod_read_unaligned`, `bind[col][row]` = M[row][col].
    pub fn compute_bone_matrices(
        &self,
        space_id: i32,
        bone_transform_ids: &[i32],
        bind_poses: &[[[f32; 4]; 4]],
        root_bone_transform_id: Option<i32>,
    ) -> Vec<[[f32; 4]; 4]> {
        if bone_transform_ids.len() > bind_poses.len() {
            crate::warn!(
                "Bone count mismatch: bone_transform_ids.len()={} > bind_poses.len()={}",
                bone_transform_ids.len(),
                bind_poses.len()
            );
        }
        let inv_root = root_bone_transform_id
            .filter(|&id| id >= 0)
            .and_then(|id| self.get_world_matrix(space_id, id as usize))
            .and_then(|m| m.try_inverse())
            .unwrap_or_else(Matrix4::identity);
        let use_root = root_bone_transform_id.is_some_and(|id| id >= 0);

        let mut out = Vec::with_capacity(bone_transform_ids.len().min(bind_poses.len()));
        for (i, &tid) in bone_transform_ids.iter().enumerate() {
            let bind = bind_poses
                .get(i)
                .copied()
                .unwrap_or(Matrix4::identity().into());
            let bind_mat = Matrix4::from_fn(|r, c| bind[c][r]);
            let world = if tid >= 0 {
                self.get_world_matrix(space_id, tid as usize)
                    .unwrap_or_else(Matrix4::identity)
            } else {
                Matrix4::identity()
            };
            let combined = if use_root {
                world * inv_root * bind_mat
            } else {
                world * bind_mat
            };
            out.push(combined.into());
        }
        out
    }

    /// Applies a frame's render space updates.
    pub fn apply_frame_update(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &crate::shared::FrameSubmitData,
    ) -> Result<(), SceneError> {
        for update in &data.render_spaces {
            if let Some(ref sh2_tasks) = update.reflection_probe_sh2_taks {
                Self::apply_reflection_probe_sh2_tasks(shm, sh2_tasks);
            }
        }

        for update in &data.render_spaces {
            {
                let scene = self.scenes.entry(update.id).or_default();
                scene.id = update.id;
                scene.is_active = update.is_active;
                scene.is_overlay = update.is_overlay;
                scene.root_transform = update.root_transform;
                scene.view_transform = if update.override_view_position {
                    update.overriden_view_transform
                } else {
                    update.root_transform
                };
            }

            let frame_index = data.frame_index;
            let transform_removals = if let Some(ref transforms_update) = update.transforms_update {
                Self::apply_transforms_update(self, update.id, shm, transforms_update, frame_index)?
            } else {
                Vec::new()
            };

            let scene = self.scenes.get_mut(&update.id).expect("scene exists after entry");
            if let Some(ref mesh_update) = update.mesh_renderers_update {
                Self::apply_mesh_renderables_update(scene, shm, mesh_update, frame_index)?;
            }
            if let Some(ref skinned_update) = update.skinned_mesh_renderers_update {
                Self::apply_skinned_mesh_renderables_update(
                    scene,
                    shm,
                    skinned_update,
                    frame_index,
                    &transform_removals,
                )?;
            }
        }

        for (id, _scene) in self.scenes.iter() {
            if !data.render_spaces.iter().any(|u| u.id == *id) {
                self.spaces_to_remove.push(*id);
            }
        }
        for id in &self.spaces_to_remove {
            self.scenes.remove(id);
            self.scene_caches.remove(id);
            self.world_matrices_dirty.remove(id);
        }
        self.spaces_to_remove.clear();
        Ok(())
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
                let renderable_index =
                    i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0; 4]));
                if renderable_index < 0 {
                    break;
                }
                bytes[offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4]
                    .copy_from_slice(&COMPUTE_RESULT_FAILED.to_le_bytes());
                offset += TASK_STRIDE;
            }
        }) {}
    }

    fn apply_transforms_update(
        &mut self,
        scene_id: SceneId,
        shm: &mut SharedMemoryAccessor,
        update: &TransformsUpdate,
        frame_index: i32,
    ) -> Result<Vec<(i32, usize)>, SceneError> {
        let scene = self
            .scenes
            .get_mut(&scene_id)
            .expect("scene exists when applying transforms update");
        let cache = self
            .scene_caches
            .entry(scene_id)
            .or_insert_with(|| SceneCache {
                world_matrices: Vec::new(),
                computed: Vec::new(),
            });
        let mut transform_removals = Vec::new();

        if cache.world_matrices.len() != scene.nodes.len() {
            cache.world_matrices.resize(scene.nodes.len(), Matrix4::identity());
            cache.computed.resize(scene.nodes.len(), false);
        }

        if update.removals.length > 0 {
            let removals = shm
                .access_copy_diagnostic::<i32>(&update.removals)
                .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
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

                for (i, parent) in scene.node_parents.iter_mut().enumerate() {
                    if *parent == removed_id {
                        *parent = -1;
                        if i < cache.computed.len() {
                            cache.computed[i] = false;
                        }
                    } else if *parent == last_index as i32 {
                        *parent = removed_id;
                    }
                }
                for entry in &mut scene.drawables {
                    entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
                }
                transform_removals.push((removed_id, last_index));

                scene.nodes.swap_remove(idx);
                scene.node_parents.swap_remove(idx);
                if idx < cache.world_matrices.len() {
                    cache.world_matrices.swap_remove(idx);
                    cache.computed.swap_remove(idx);
                }
            }
        }

        while (scene.nodes.len() as i32) < update.target_transform_count {
            scene.nodes.push(render_transform_identity());
            scene.node_parents.push(-1);
            cache.world_matrices.push(Matrix4::identity());
            cache.computed.push(false);
        }

        let mut changed_indices = std::collections::HashSet::new();

        if update.parent_updates.length > 0 {
            let parents = shm
                .access_copy_diagnostic::<TransformParentUpdate>(&update.parent_updates)
                .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
            for pu in parents {
                if pu.transform_id < 0 {
                    break;
                }
                if (pu.transform_id as usize) < scene.node_parents.len() {
                    scene.node_parents[pu.transform_id as usize] = pu.new_parent_id;
                    changed_indices.insert(pu.transform_id as usize);
                }
            }
        }

        if update.pose_updates.length > 0 {
            let poses = shm
                .access_copy_diagnostic::<TransformPoseUpdate>(&update.pose_updates)
                .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
            for pu in &poses {
                if pu.transform_id < 0 {
                    break;
                }
                if (pu.transform_id as usize) < scene.nodes.len() {
                    let validation = PoseValidation {
                        pose: &pu.pose,
                        frame_index,
                        scene_id: scene.id,
                        transform_id: pu.transform_id,
                    };
                    if validation.is_valid() {
                        scene.nodes[pu.transform_id as usize] = pu.pose;
                    } else {
                        crate::error!(
                            "Invalid pose scene={} transform={} frame={}: using identity",
                            scene.id,
                            pu.transform_id,
                            frame_index
                        );
                        scene.nodes[pu.transform_id as usize] = render_transform_identity();
                    }
                    changed_indices.insert(pu.transform_id as usize);
                }
            }
        }

        for i in &changed_indices {
            if *i < cache.computed.len() {
                cache.computed[*i] = false;
            }
        }
        Self::mark_descendants_uncomputed(&scene.node_parents, &mut cache.computed);
        self.world_matrices_dirty.insert(scene_id);

        Ok(transform_removals)
    }

    fn apply_mesh_renderables_update(
        scene: &mut Scene,
        shm: &mut SharedMemoryAccessor,
        update: &MeshRenderablesUpdate,
        _frame_index: i32,
    ) -> Result<(), SceneError> {
        if update.removals.length > 0 {
            let removals = shm
                .access_copy_diagnostic::<i32>(&update.removals)
                .map_err(SceneError::SharedMemoryAccess)?;
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
        if update.additions.length > 0 {
            let additions = shm
                .access_copy_diagnostic::<i32>(&update.additions)
                .map_err(SceneError::SharedMemoryAccess)?;
            let added_node_ids: Vec<i32> =
                additions.iter().take_while(|&&i| i >= 0).copied().collect();
            for &node_id in &added_node_ids {
                scene.drawables.push(Drawable {
                    node_id,
                    mesh_handle: -1,
                    material_handle: None,
                    sort_key: 0,
                    is_skinned: false,
                    bone_transform_ids: None,
                    root_bone_transform_id: None,
                });
            }
        }
        if update.mesh_states.length > 0 {
            let states = shm
                .access_copy_diagnostic::<MeshRendererStatePod>(&update.mesh_states)
                .map_err(SceneError::SharedMemoryAccess)?;
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
        Ok(())
    }

    /// Applies skinned mesh renderable updates. Bone transform IDs are fixup'd when transforms
    /// are removed via swap_remove: references to the removed ID become -1; references to the
    /// last index (now swapped into the removed slot) become the removed ID.
    fn apply_skinned_mesh_renderables_update(
        scene: &mut Scene,
        shm: &mut SharedMemoryAccessor,
        update: &SkinnedMeshRenderablesUpdate,
        _frame_index: i32,
        transform_removals: &[(i32, usize)],
    ) -> Result<(), SceneError> {
        for &(removed_id, last_index) in transform_removals {
            for entry in &mut scene.skinned_drawables {
                entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
                if let Some(ref mut ids) = entry.bone_transform_ids {
                    for id in ids.iter_mut() {
                        *id = fixup_transform_id(*id, removed_id, last_index);
                    }
                }
                if let Some(rid) = entry.root_bone_transform_id {
                    entry.root_bone_transform_id = Some(fixup_transform_id(rid, removed_id, last_index));
                }
            }
        }

        if update.removals.length > 0 {
            let removals = shm
                .access_copy_diagnostic::<i32>(&update.removals)
                .map_err(SceneError::SharedMemoryAccess)?;
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
        if update.additions.length > 0 {
            let additions = shm
                .access_copy_diagnostic::<i32>(&update.additions)
                .map_err(SceneError::SharedMemoryAccess)?;
            let added_node_ids: Vec<i32> =
                additions.iter().take_while(|&&i| i >= 0).copied().collect();
            for &node_id in &added_node_ids {
                scene.skinned_drawables.push(Drawable {
                    node_id,
                    mesh_handle: -1,
                    material_handle: None,
                    sort_key: 0,
                    is_skinned: true,
                    bone_transform_ids: None,
                    root_bone_transform_id: None,
                });
            }
        }
        if update.mesh_states.length > 0 {
            let states = shm
                .access_copy_diagnostic::<MeshRendererStatePod>(&update.mesh_states)
                .map_err(SceneError::SharedMemoryAccess)?;
            for state in states {
                if state.renderable_index < 0 {
                    break;
                }
                let idx = state.renderable_index as usize;
                if idx < scene.skinned_drawables.len() {
                    scene.skinned_drawables[idx].mesh_handle = state.mesh_asset_id;
                    scene.skinned_drawables[idx].sort_key = state.sorting_order;
                    scene.skinned_drawables[idx].material_handle = if state.material_count > 0 {
                        Some(-1)
                    } else {
                        None
                    };
                }
            }
        }
        if update.bone_assignments.length > 0 {
            let assignments = shm
                .access_copy_diagnostic::<BoneAssignment>(&update.bone_assignments)
                .map_err(SceneError::SharedMemoryAccess)?;
            let indexes = shm
                .access_copy_diagnostic::<i32>(&update.bone_transform_indexes)
                .map_err(SceneError::SharedMemoryAccess)?;
            let mut index_offset = 0;
            for assignment in &assignments {
                if assignment.renderable_index < 0 {
                    break;
                }
                let idx = assignment.renderable_index as usize;
                let bone_count = assignment.bone_count.max(0) as usize;
                if idx < scene.skinned_drawables.len() && index_offset + bone_count <= indexes.len()
                {
                    let ids: Vec<i32> = indexes[index_offset..index_offset + bone_count].to_vec();
                    scene.skinned_drawables[idx].bone_transform_ids = Some(ids);
                    scene.skinned_drawables[idx].root_bone_transform_id =
                        if assignment.root_bone_transform_id >= 0 {
                            Some(assignment.root_bone_transform_id)
                        } else {
                            None
                        };
                }
                index_offset += bone_count;
            }
        }
        Ok(())
    }

    /// Marks descendants of uncomputed transforms as uncomputed.
    /// Walks each node's parent chain to find the uppermost uncomputed ancestor;
    /// if found, marks all nodes in that chain as uncomputed.
    fn mark_descendants_uncomputed(node_parents: &[i32], computed: &mut [bool]) {
        let n = computed.len();
        if n == 0 {
            return;
        }
        let mut checked = vec![false; n];
        for transform_index in (0..n).rev() {
            if checked[transform_index] {
                continue;
            }
            let mut maybe_last_non_computed: Option<usize> = None;
            let mut id = transform_index;
            let mut steps = 0;
            while id < n && steps < n {
                steps += 1;
                if !computed[id] {
                    maybe_last_non_computed = Some(id);
                }
                if checked[id] {
                    break;
                }
                let p = node_parents.get(id).copied().unwrap_or(-1);
                if p < 0 || (p as usize) >= n || p == id as i32 {
                    break;
                }
                id = p as usize;
            }
            if let Some(last_non_computed) = maybe_last_non_computed {
                let mut id = transform_index;
                let mut steps = 0;
                while id != last_non_computed && id < n && steps < n {
                    steps += 1;
                    computed[id] = false;
                    checked[id] = true;
                    let p = node_parents.get(id).copied().unwrap_or(-1);
                    if p < 0 || (p as usize) >= n || p == id as i32 {
                        break;
                    }
                    id = p as usize;
                }
            } else {
                let mut id = transform_index;
                let mut steps = 0;
                while id < n && steps < n {
                    steps += 1;
                    checked[id] = true;
                    let p = node_parents.get(id).copied().unwrap_or(-1);
                    if p < 0 || (p as usize) >= n || p == id as i32 {
                        break;
                    }
                    id = p as usize;
                }
            }
            checked[transform_index] = true;
        }
    }

    /// Computes and caches world matrices for a scene. Uses incremental recomputation:
    /// only transforms with `computed[i] == false` are recomputed.
    pub fn compute_world_matrices(&mut self, scene_id: SceneId) -> Result<(), SceneError> {
        let n = self
            .scenes
            .get(&scene_id)
            .map(|s| s.nodes.len())
            .unwrap_or(0);
        if n == 0 {
            self.scene_caches.remove(&scene_id);
            self.world_matrices_dirty.remove(&scene_id);
            return Ok(());
        }

        let cache = self
            .scene_caches
            .entry(scene_id)
            .or_insert_with(|| SceneCache {
                world_matrices: Vec::new(),
                computed: Vec::new(),
            });

        let needs_resize = cache.world_matrices.len() != n;
        if needs_resize {
            cache.world_matrices.resize(n, Matrix4::identity());
            cache.computed.resize(n, false);
            for c in cache.computed.iter_mut() {
                *c = false;
            }
        }

        if !self.world_matrices_dirty.contains(&scene_id)
            && !needs_resize
            && cache.computed.iter().all(|&c| c)
        {
            return Ok(());
        }

        let scene = self.scenes.get(&scene_id).expect("scene exists");
        Self::compute_world_matrices_incremental(
            scene,
            &mut cache.world_matrices,
            &mut cache.computed,
        )?;
        self.world_matrices_dirty.remove(&scene_id);
        Ok(())
    }

    /// Incremental world matrix computation: only recomputes nodes with `computed[i] == false`.
    /// Walks up from each uncomputed node to find the first computed ancestor, then multiplies down.
    fn compute_world_matrices_incremental(
        scene: &Scene,
        world_matrices: &mut [Matrix4<f32>],
        computed: &mut [bool],
    ) -> Result<(), SceneError> {
        let n = scene.nodes.len();
        let node_parents = &scene.node_parents;
        let nodes = &scene.nodes;
        let mut stack = Vec::with_capacity(64.min(n));

        for transform_index in (0..n).rev() {
            if computed[transform_index] {
                continue;
            }

            let mut maybe_uppermost_matrix: Option<Matrix4<f32>> = None;
            let mut id = transform_index;
            let mut steps = 0;
            while id < n && steps < n {
                steps += 1;
                if computed[id] {
                    maybe_uppermost_matrix = Some(world_matrices[id]);
                    break;
                }
                stack.push(id);
                let p = node_parents.get(id).copied().unwrap_or(-1);
                if p < 0 || (p as usize) >= n || p == id as i32 {
                    break;
                }
                id = p as usize;
            }

            let mut parent_matrix = match maybe_uppermost_matrix {
                Some(m) => m,
                None => {
                    let top = match stack.pop() {
                        Some(t) => t,
                        None => continue,
                    };
                    let local = super::render_transform_to_matrix(&nodes[top]);
                    let uppermost = Matrix4::<f32>::identity() * local;
                    world_matrices[top] = uppermost;
                    computed[top] = true;
                    uppermost
                }
            };

            while let Some(child_id) = stack.pop() {
                let local = super::render_transform_to_matrix(&nodes[child_id]);
                parent_matrix = parent_matrix * local;
                world_matrices[child_id] = parent_matrix;
                computed[child_id] = true;
            }
        }

        Ok(())
    }

    /// Full iterative DFS world matrix computation with cycle detection.
    /// Used by tests; root-level nodes use identity as parent.
    #[cfg(test)]
    fn compute_world_matrices_from_scene(scene: &Scene) -> Vec<Matrix4<f32>> {
        let n = scene.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        let mut world = vec![Matrix4::identity(); n];
        let mut visited = vec![false; n];
        let mut in_stack = vec![false; n];

        let mut stack: Vec<usize> = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            stack.push(start);
            in_stack[start] = true;
            while let Some(&i) = stack.last() {
                if visited[i] {
                    in_stack[i] = false;
                    stack.pop();
                    continue;
                }
                let p = scene.node_parents.get(i).copied().unwrap_or(-1);
                let p_usize = if p >= 0 && (p as usize) < n && p != i as i32 {
                    p as usize
                } else {
                    let local = super::render_transform_to_matrix(&scene.nodes[i]);
                    world[i] = Matrix4::<f32>::identity() * local;
                    visited[i] = true;
                    in_stack[i] = false;
                    stack.pop();
                    continue;
                };

                if in_stack[p_usize] {
                    crate::warn!(
                        "Cycle detected in scene {} at transform {} (parent {}); treating as root",
                        scene.id,
                        i,
                        p
                    );
                    let local = super::render_transform_to_matrix(&scene.nodes[i]);
                    world[i] = Matrix4::<f32>::identity() * local;
                    visited[i] = true;
                    in_stack[i] = false;
                    stack.pop();
                    continue;
                }

                if !visited[p_usize] {
                    stack.push(p_usize);
                    in_stack[p_usize] = true;
                    continue;
                }

                let local = super::render_transform_to_matrix(&scene.nodes[i]);
                world[i] = world[p_usize] * local;
                visited[i] = true;
                in_stack[i] = false;
                stack.pop();
            }
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
    fn test_pose_validation_rejects_nan() {
        let mut pose = make_transform((0.0, 0.0, 0.0));
        pose.position.x = f32::NAN;
        let v = PoseValidation {
            pose: &pose,
            frame_index: 0,
            scene_id: 0,
            transform_id: 0,
        };
        assert!(!v.is_valid());
    }

    #[test]
    fn test_pose_validation_rejects_inf() {
        let mut pose = make_transform((0.0, 0.0, 0.0));
        pose.scale.y = f32::INFINITY;
        let v = PoseValidation {
            pose: &pose,
            frame_index: 0,
            scene_id: 0,
            transform_id: 0,
        };
        assert!(!v.is_valid());
    }

    #[test]
    fn test_pose_validation_rejects_large() {
        let pose = RenderTransform {
            position: Vector3::new(2e6, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::identity(),
        };
        let v = PoseValidation {
            pose: &pose,
            frame_index: 0,
            scene_id: 0,
            transform_id: 0,
        };
        assert!(!v.is_valid());
    }

    #[test]
    fn test_pose_validation_accepts_valid() {
        let pose = make_transform((1.0, 2.0, 3.0));
        let v = PoseValidation {
            pose: &pose,
            frame_index: 0,
            scene_id: 0,
            transform_id: 0,
        };
        assert!(v.is_valid());
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

    #[test]
    fn test_compute_world_matrices_cycle_detection() {
        let mut scene = Scene::default();
        scene.root_transform = make_transform((0.0, 0.0, 0.0));
        scene.nodes = vec![
            make_transform((1.0, 0.0, 0.0)),
            make_transform((0.0, 2.0, 0.0)),
        ];
        scene.node_parents = vec![1, 0];

        let world = SceneGraph::compute_world_matrices_from_scene(&scene);

        assert_eq!(world.len(), 2);
        assert!(!world[0].column(3).x.is_nan());
        assert!(!world[1].column(3).x.is_nan());
    }

    #[test]
    fn test_compute_world_matrices_incremental_after_pose_change() {
        let mut graph = SceneGraph::new();
        let mut scene = Scene::default();
        scene.id = 0;
        scene.nodes = vec![
            make_transform((1.0, 0.0, 0.0)),
            make_transform((0.0, 2.0, 0.0)),
            make_transform((0.0, 0.0, 3.0)),
        ];
        scene.node_parents = vec![-1, 0, 1];
        graph.scenes.insert(0, scene);

        graph.compute_world_matrices(0).unwrap();
        let mat_before = graph.get_world_matrix(0, 2).unwrap();
        let pos2_before = mat_before.column(3);
        assert!((pos2_before.x - 1.0).abs() < 1e-5);
        assert!((pos2_before.y - 2.0).abs() < 1e-5);
        assert!((pos2_before.z - 3.0).abs() < 1e-5);

        graph.get_scene_mut(0).unwrap().nodes[0] =
            make_transform((10.0, 0.0, 0.0));
        graph.test_invalidate_transform(0, 0);

        graph.compute_world_matrices(0).unwrap();
        let mat_after = graph.get_world_matrix(0, 2).unwrap();
        let pos2_after = mat_after.column(3);
        assert!((pos2_after.x - 10.0).abs() < 1e-5);
        assert!((pos2_after.y - 2.0).abs() < 1e-5);
        assert!((pos2_after.z - 3.0).abs() < 1e-5);
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}
