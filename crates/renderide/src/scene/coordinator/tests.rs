//! Unit tests for [`super::SceneCoordinator`].
//!
//! The tests are split into two subject areas: [`apply`] covers the phase-orchestration
//! predicates that decide whether the render world header / extracted update needs a re-flush
//! plus the apply-extracted commit, and [`queries`] covers the read-only accessors
//! (render-space iteration, blit-for-display, world matrices, overlay matrices, and degenerate
//! scale flags). Shared test-only helpers on [`super::SceneCoordinator`] live here so both
//! subject files can use them.

use crate::scene::overrides::RenderTransformOverrideEntry;
use crate::scene::render_space::{LayerAssignmentEntry, RenderSpaceState};
use crate::scene::{SkinnedMeshRenderer, StaticMeshRenderer};
use crate::shared::{LayerType, RenderTransform, RenderingContext};

use super::super::ids::RenderSpaceId;
use super::super::world::{WorldTransformCache, compute_world_matrices_for_space};
use super::SceneCoordinator;

mod apply;
mod queries;

impl SceneCoordinator {
    /// Overrides [`RenderSpaceState::is_active`] for a seeded space (unit tests only).
    pub(crate) fn test_set_space_active(&mut self, id: RenderSpaceId, is_active: bool) {
        if let Some(space) = self.spaces.get_mut(&id) {
            space.is_active = is_active;
        }
    }

    /// Overrides [`RenderSpaceState::is_overlay`] for a seeded space (unit tests only).
    pub(crate) fn test_set_space_overlay(&mut self, id: RenderSpaceId, is_overlay: bool) {
        if let Some(space) = self.spaces.get_mut(&id) {
            space.is_overlay = is_overlay;
        }
    }

    /// Overrides [`RenderSpaceState::is_private`] for a seeded space (unit tests only).
    pub(crate) fn test_set_space_private(&mut self, id: RenderSpaceId, is_private: bool) {
        if let Some(space) = self.spaces.get_mut(&id) {
            space.is_private = is_private;
        }
    }

    /// Overrides [`RenderSpaceState::root_transform`] for a seeded space (unit tests only).
    pub(crate) fn test_set_space_root_transform(
        &mut self,
        id: RenderSpaceId,
        root_transform: RenderTransform,
    ) {
        if let Some(space) = self.spaces.get_mut(&id) {
            space.root_transform = root_transform;
        }
    }

    /// Inserts a render space and solves world matrices from the given locals (for unit tests).
    pub(crate) fn test_seed_space_identity_worlds(
        &mut self,
        id: RenderSpaceId,
        nodes: Vec<RenderTransform>,
        node_parents: Vec<i32>,
    ) {
        assert_eq!(
            nodes.len(),
            node_parents.len(),
            "nodes and node_parents length must match"
        );
        self.spaces.insert(
            id,
            RenderSpaceState {
                id,
                is_active: true,
                nodes,
                node_parents,
                ..Default::default()
            },
        );
        let space = self.spaces.get(&id).expect("inserted space");
        let mut cache = WorldTransformCache::default();
        let _ =
            compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache);
        self.world_caches.insert(id, cache);
    }

    /// Appends a layer assignment to a seeded render space (unit tests only).
    pub(crate) fn test_push_layer_assignment(
        &mut self,
        id: RenderSpaceId,
        node_id: i32,
        layer: LayerType,
    ) {
        let space = self.spaces.get_mut(&id).expect("seeded space");
        space
            .layer_assignments
            .push(LayerAssignmentEntry { node_id, layer });
        space.layer_index_dirty = true;
    }

    /// Appends a scale-only render transform override to a seeded render space (unit tests only).
    pub(crate) fn test_push_scale_render_transform_override(
        &mut self,
        id: RenderSpaceId,
        node_id: i32,
        context: RenderingContext,
        scale: glam::Vec3,
    ) {
        let space = self.spaces.get_mut(&id).expect("seeded space");
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id,
                context,
                scale_override: Some(scale),
                ..Default::default()
            });
    }

    /// Inserts a render space with static mesh renderers (unit tests only).
    pub(crate) fn test_insert_static_mesh_renderers(
        &mut self,
        id: RenderSpaceId,
        renderers: Vec<StaticMeshRenderer>,
    ) {
        self.spaces.insert(
            id,
            RenderSpaceState {
                id,
                static_mesh_renderers: renderers,
                ..Default::default()
            },
        );
    }

    /// Inserts a render space with skinned mesh renderers (unit tests only).
    pub(crate) fn test_insert_skinned_mesh_renderers(
        &mut self,
        id: RenderSpaceId,
        renderers: Vec<SkinnedMeshRenderer>,
    ) {
        self.spaces.insert(
            id,
            RenderSpaceState {
                id,
                skinned_mesh_renderers: renderers,
                ..Default::default()
            },
        );
    }
}
