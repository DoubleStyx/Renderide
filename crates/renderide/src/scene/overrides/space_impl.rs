//! [`super::super::render_space::RenderSpaceState`] query helpers for render-context overrides.

use crate::shared::{RenderTransform, RenderingContext};

use super::super::render_space::RenderSpaceState;
use super::types::MeshRendererOverrideTarget;

impl RenderSpaceState {
    /// Primary rendering context for this space (user view vs external mirror).
    pub(in crate::scene) fn main_render_context(&self) -> RenderingContext {
        if self.view_position_is_external {
            RenderingContext::ExternalView
        } else {
            RenderingContext::UserView
        }
    }

    /// Returns whether any transform override rows exist for `context`.
    pub(in crate::scene) fn has_transform_overrides_in_context(
        &self,
        context: RenderingContext,
    ) -> bool {
        self.render_transform_overrides
            .iter()
            .any(|entry| entry.context == context && entry.node_id >= 0)
    }

    /// Returns whether any material override rows exist for `context`.
    pub(in crate::scene) fn has_material_overrides_in_context(
        &self,
        context: RenderingContext,
    ) -> bool {
        self.render_material_overrides
            .iter()
            .any(|entry| entry.context == context && entry.node_id >= 0)
    }

    /// Returns whether draw preparation must be specialized for `context`.
    pub(in crate::scene) fn has_draw_prep_overrides_in_context(
        &self,
        context: RenderingContext,
    ) -> bool {
        self.has_transform_overrides_in_context(context)
            || self.has_material_overrides_in_context(context)
    }

    /// Applies transform overrides for `node_id` in `context` atop the dense local transform.
    pub(in crate::scene) fn overridden_local_transform(
        &self,
        node_id: i32,
        context: RenderingContext,
    ) -> Option<RenderTransform> {
        let base = *self.nodes.get(node_id as usize)?;
        let mut local = base;
        let mut matched = false;
        for entry in self
            .render_transform_overrides
            .iter()
            .filter(|entry| entry.node_id == node_id && entry.context == context)
        {
            if let Some(position) = entry.position_override {
                local.position = position;
            }
            if let Some(rotation) = entry.rotation_override {
                local.rotation = rotation;
            }
            if let Some(scale) = entry.scale_override {
                local.scale = scale;
            }
            matched = true;
        }
        matched.then_some(local)
    }

    /// Resolves a material override for `target` and slot, if any, in `context`.
    pub(in crate::scene) fn overridden_material_asset_id(
        &self,
        context: RenderingContext,
        target: MeshRendererOverrideTarget,
        slot_index: usize,
    ) -> Option<i32> {
        let mut replacement = None;
        for entry in self.render_material_overrides.iter().filter(|entry| {
            entry.context == context && entry.target == target && entry.node_id >= 0
        }) {
            for material in &entry.material_overrides {
                if material.material_slot_index == slot_index as i32 {
                    replacement = Some(material.material_asset_id);
                }
            }
        }
        replacement
    }
}
