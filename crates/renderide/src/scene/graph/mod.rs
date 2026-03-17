//! Scene graph: manages scenes and applies host updates.
//!
//! Extension point for scene graph, hierarchy.

mod error;
mod pods;
mod pose;
mod updates;
mod world_matrices;

use std::collections::{HashMap, HashSet};

use nalgebra::Matrix4;

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Scene, SceneId};

pub use error::SceneError;
pub use pose::PoseValidation;

#[allow(unused_imports)]
use world_matrices::{compute_world_matrices_incremental, mark_descendants_uncomputed, SceneCache};

/// Manages scenes (render spaces) and applies incremental updates from the host.
pub struct SceneGraph {
    scenes: HashMap<SceneId, Scene>,
    scene_caches: HashMap<SceneId, SceneCache>,
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
                    mark_descendants_uncomputed(
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
            logger::trace!(
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
                updates::apply_reflection_probe_sh2_tasks(shm, sh2_tasks);
            }
        }

        for update in &data.render_spaces {
            {
                let scene = self.scenes.entry(update.id).or_default();
                scene.id = update.id;
                scene.is_active = update.is_active;
                scene.is_overlay = update.is_overlay;
                scene.is_private = update.is_private;
                scene.root_transform = update.root_transform;
                scene.view_transform = if update.override_view_position {
                    update.overriden_view_transform
                } else {
                    update.root_transform
                };
            }

            let frame_index = data.frame_index;
            let transform_removals = if let Some(ref transforms_update) = update.transforms_update {
                let scene = self.scenes.get_mut(&update.id).expect("scene exists after entry");
                let cache = self.scene_caches.entry(update.id).or_insert_with(|| SceneCache {
                    world_matrices: Vec::new(),
                    computed: Vec::new(),
                });
                updates::apply_transforms_update(
                    scene,
                    cache,
                    &mut self.world_matrices_dirty,
                    update.id,
                    shm,
                    transforms_update,
                    frame_index,
                )?
            } else {
                Vec::new()
            };

            let scene = self.scenes.get_mut(&update.id).expect("scene exists after entry");
            if let Some(ref layers_update) = update.layers_update {
                updates::apply_layers_update(scene, shm, layers_update)?;
            }
            if let Some(ref mesh_update) = update.mesh_renderers_update {
                updates::apply_mesh_renderables_update(scene, shm, mesh_update, frame_index)?;
            }
            if let Some(ref skinned_update) = update.skinned_mesh_renderers_update {
                updates::apply_skinned_mesh_renderables_update(
                    scene,
                    shm,
                    skinned_update,
                    frame_index,
                    &transform_removals,
                )?;
            }
            if let Some(ref mat_override_update) = update.render_material_overrides_update {
                updates::apply_render_material_overrides_update(scene, shm, mat_override_update)?;
            }
            updates::sync_drawable_layers(scene);
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
        compute_world_matrices_incremental(
            scene,
            &mut cache.world_matrices,
            &mut cache.computed,
        )?;
        self.world_matrices_dirty.remove(&scene_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::RenderTransform;
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

        let world = world_matrices::compute_world_matrices_from_scene(&scene);

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

        let world = world_matrices::compute_world_matrices_from_scene(&scene);

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

        let world = world_matrices::compute_world_matrices_from_scene(&scene);

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

        let world = world_matrices::compute_world_matrices_from_scene(&scene);

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

        graph.get_scene_mut(0).unwrap().nodes[0] = make_transform((10.0, 0.0, 0.0));
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
