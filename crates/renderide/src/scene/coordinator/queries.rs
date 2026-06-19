//! World-matrix queries with render-context overrides and overlay re-rooting.

use glam::{Mat4, Vec3};

use crate::shared::{LayerType, RenderingContext};

use super::super::ids::RenderSpaceId;
use super::super::math::{
    render_matrix_has_degenerate_scale, render_transform_has_degenerate_scale,
};
use super::super::render_space::RenderSpaceState;
use super::super::render_transform_to_matrix;
use super::SceneCoordinator;

impl SceneCoordinator {
    /// Hierarchy world matrix with active render-context-local transform overrides applied.
    pub fn world_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        if transform_index >= space.nodes.len() {
            return None;
        }
        if !space.has_transform_overrides_in_context(context) {
            return self.world_matrix(id, transform_index);
        }

        let mut cursor = transform_index;
        let mut reached_terminal = false;
        let mut any_override = false;
        let mut world = Mat4::IDENTITY;
        for _ in 0..space.nodes.len() {
            let (local, overridden) = local_transform_for_context(space, cursor, context);
            any_override |= overridden;
            world = render_transform_to_matrix(&local) * world;
            let parent = *space.node_parents.get(cursor).unwrap_or(&-1);
            if parent < 0 || parent as usize >= space.nodes.len() || parent == cursor as i32 {
                reached_terminal = true;
                break;
            }
            cursor = parent as usize;
        }
        if !reached_terminal || !any_override {
            return self.world_matrix(id, transform_index);
        }
        Some(world)
    }

    /// Hierarchy world matrix prepared for actual rendering.
    ///
    /// Overlay spaces are re-rooted against the current `HeadOutput.transform` before drawing
    /// (`RenderSpace.UpdateOverlayPositioning` on the host side).
    pub fn world_matrix_for_render_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
        head_output_transform: Mat4,
    ) -> Option<Mat4> {
        let local = self.world_matrix_for_context(id, transform_index, context)?;
        let space = self.spaces.get(&id)?;
        if !space.is_overlay {
            return Some(local);
        }
        Some(overlay_space_root_matrix(space, head_output_transform) * local)
    }

    /// Hierarchy matrix for an overlay-layer draw relative to its nearest overlay-layer ancestor.
    ///
    /// Screen-space overlay meshes inherit [`LayerType::Overlay`] from a parent slot such as the
    /// host `OverlayManager.OverlayRoot`. Rendering should keep the local chain starting at that
    /// overlay ancestor but ignore world-space ancestors above it, otherwise desktop overlay
    /// visuals continue to follow the userspace/world transform instead of the screen.
    pub fn overlay_layer_model_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        if transform_index >= space.nodes.len() {
            return None;
        }

        let anchor = nearest_overlay_layer_ancestor(space, transform_index)?;
        matrix_from_ancestor_for_context(space, transform_index, context, anchor)
    }

    /// Returns the nearest inherited special layer (`Overlay` / `Hidden`) for this transform.
    pub fn transform_special_layer(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<LayerType> {
        self.spaces
            .get(&id)
            .and_then(|space| nearest_special_layer_ancestor(space, transform_index))
            .map(|(_, layer)| layer)
    }

    /// Returns whether `transform_index` is a descendant of an active overlay-layer ancestor.
    pub fn transform_is_in_overlay_layer(&self, id: RenderSpaceId, transform_index: usize) -> bool {
        matches!(
            self.transform_special_layer(id, transform_index),
            Some(LayerType::Overlay)
        )
    }

    /// Returns whether the cached hierarchy for `transform_index` contains degenerate object scale.
    ///
    /// Missing spaces or transforms return `false` so callers preserve existing draw fallbacks when
    /// the world cache is unavailable for reasons unrelated to scale.
    pub fn transform_has_degenerate_scale(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> bool {
        self.world_caches
            .get(&id)
            .and_then(|cache| cache.degenerate_scales.get(transform_index))
            .copied()
            .unwrap_or(false)
    }

    /// Returns whether the effective render-context transform chain collapses object scale.
    ///
    /// Render-transform overrides replace local TRS components for the selected context. Overlay
    /// head-output re-rooting is intentionally ignored because the visibility decision belongs to
    /// the object transform chain, not the camera/root transform used for view placement.
    pub fn transform_has_degenerate_scale_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> bool {
        let Some(space) = self.spaces.get(&id) else {
            return false;
        };
        if transform_index >= space.nodes.len() {
            return false;
        }
        if !space.has_transform_overrides_in_context(context) {
            return self.transform_has_degenerate_scale(id, transform_index);
        }

        let mut cursor = transform_index;
        let mut reached_terminal = false;
        let mut any_override = false;
        let mut any_degenerate_scale = false;
        let mut world = Mat4::IDENTITY;
        for _ in 0..space.nodes.len() {
            let (local, overridden) = local_transform_for_context(space, cursor, context);
            any_override |= overridden;
            any_degenerate_scale |= render_transform_has_degenerate_scale(&local);
            world = render_transform_to_matrix(&local) * world;
            let parent = *space.node_parents.get(cursor).unwrap_or(&-1);
            if parent < 0 || parent as usize >= space.nodes.len() || parent == cursor as i32 {
                reached_terminal = true;
                break;
            }
            cursor = parent as usize;
        }
        if !reached_terminal || !any_override {
            return self.transform_has_degenerate_scale(id, transform_index);
        }
        any_degenerate_scale || render_matrix_has_degenerate_scale(world)
    }
}

fn overlay_space_root_matrix(space: &RenderSpaceState, head_output_transform: Mat4) -> Mat4 {
    let (scale, rotation, position) = head_output_transform.to_scale_rotation_translation();
    let scale = filter_overlay_scale(scale);
    let position = position - space.root_transform.position;
    let rotation = rotation * space.root_transform.rotation;
    Mat4::from_scale_rotation_translation(scale, rotation, position)
}

fn filter_overlay_scale(scale: Vec3) -> Vec3 {
    if scale.x.min(scale.y).min(scale.z) <= 1e-8 {
        Vec3::ONE
    } else {
        scale
    }
}

fn layer_assignment_for_node(space: &RenderSpaceState, node_id: i32) -> Option<LayerType> {
    if node_id < 0 {
        return None;
    }
    if !space.layer_index_dirty {
        return space.layer_index.get(&node_id).copied();
    }
    space
        .layer_assignments
        .iter()
        .rev()
        .find(|entry| entry.node_id == node_id)
        .map(|entry| entry.layer)
}

fn nearest_overlay_layer_ancestor(
    space: &RenderSpaceState,
    transform_index: usize,
) -> Option<usize> {
    match nearest_special_layer_ancestor(space, transform_index) {
        Some((anchor, LayerType::Overlay)) => Some(anchor),
        _ => None,
    }
}

fn nearest_special_layer_ancestor(
    space: &RenderSpaceState,
    transform_index: usize,
) -> Option<(usize, LayerType)> {
    let mut cursor = transform_index as i32;
    for _ in 0..space.nodes.len() {
        if let Some(layer) = layer_assignment_for_node(space, cursor) {
            return Some((cursor as usize, layer));
        }
        let parent = *space.node_parents.get(cursor as usize)?;
        if parent < 0 || parent == cursor || parent as usize >= space.nodes.len() {
            return None;
        }
        cursor = parent;
    }
    None
}

/// Builds the matrix that maps points from `transform_index`'s local frame into
/// `ancestor_index`'s local frame. Multiplies every intermediate node's local transform but
/// stops before multiplying the ancestor's own local transform.
///
/// The ancestor's own local transform is the ancestor's pose relative to its parent, not part of
/// expressing children in ancestor-local coordinates. For overlay-layer items this keeps
/// `OverlayRoot` placement from shifting every screen-space overlay draw.
fn matrix_from_ancestor_for_context(
    space: &RenderSpaceState,
    transform_index: usize,
    context: RenderingContext,
    ancestor_index: usize,
) -> Option<Mat4> {
    if transform_index >= space.nodes.len() || ancestor_index >= space.nodes.len() {
        return None;
    }

    let mut cursor = transform_index;
    let mut matrix = Mat4::IDENTITY;
    for _ in 0..space.nodes.len() {
        if cursor == ancestor_index {
            return Some(matrix);
        }
        let (local, _) = local_transform_for_context(space, cursor, context);
        matrix = render_transform_to_matrix(&local) * matrix;
        let parent = *space.node_parents.get(cursor).unwrap_or(&-1);
        if parent < 0 || parent as usize >= space.nodes.len() || parent == cursor as i32 {
            break;
        }
        cursor = parent as usize;
    }
    None
}

fn local_transform_for_context(
    space: &RenderSpaceState,
    node_id: usize,
    context: RenderingContext,
) -> (crate::shared::RenderTransform, bool) {
    match space.overridden_local_transform(node_id as i32, context) {
        Some(local) => (local, true),
        None => (space.nodes[node_id], false),
    }
}
