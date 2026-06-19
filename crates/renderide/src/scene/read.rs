//! Narrow read-only scene contracts for render-facing systems.

use glam::Mat4;

use super::{RenderSpaceId, RenderSpaceView, ResolvedLight};
use crate::shared::{LayerType, RenderSH2, RenderTransform, RenderingContext};

/// Read-only access to one host render-space snapshot.
pub(crate) trait RenderSpaceRead<'a>: Copy {
    /// Returns whether the host render space is active.
    fn is_active(self) -> bool;
    /// Returns whether this render space is an overlay space.
    fn is_overlay(self) -> bool;
    /// Returns whether this render space is private.
    fn is_private(self) -> bool;
    /// Space root transform from the host.
    fn root_transform(self) -> &'a RenderTransform;
    /// Resolved eye/root transform used for view construction.
    fn view_transform(self) -> &'a RenderTransform;
    /// Local transforms indexed by dense transform id.
    fn local_transforms(self) -> &'a [RenderTransform];
    /// Parent ids indexed by dense transform id.
    fn node_parents(self) -> &'a [i32];
}

/// Read-only render-space lookup and iteration.
pub(crate) trait SceneSpaceRead {
    /// Read-only render-space view returned by this scene source.
    type Space<'a>: RenderSpaceRead<'a> + 'a
    where
        Self: 'a;
    /// Stable render-space id iterator.
    type RenderSpaceIds<'a>: Iterator<Item = RenderSpaceId> + 'a
    where
        Self: 'a;

    /// Render space ids currently present, ordered by host id for deterministic traversal.
    fn render_space_ids(&self) -> Self::RenderSpaceIds<'_>;
    /// Read-only access for render-facing systems.
    fn space(&self, id: RenderSpaceId) -> Option<Self::Space<'_>>;
    /// Main non-overlay render space, matching the host's single active main-space expectation.
    fn active_main_space(&self) -> Option<Self::Space<'_>>;
    /// Ambient SH2 from the active non-overlay render space.
    fn active_main_ambient_light(&self) -> RenderSH2;
    /// Current head-output render context for the main view.
    fn active_main_render_context(&self) -> RenderingContext;
    /// Returns whether `context` has transform or material overrides that affect retained draw prep.
    fn render_context_affects_draw_prep(&self, context: RenderingContext) -> bool;
}

/// Read-only transform and material-override queries used by render preparation.
pub(crate) trait SceneTransformRead: SceneSpaceRead {
    /// Hierarchy world matrix with active render-context-local transform overrides applied.
    fn world_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4>;
    /// Hierarchy world matrix prepared for actual rendering.
    fn world_matrix_for_render_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
        head_output_transform: Mat4,
    ) -> Option<Mat4>;
    /// Hierarchy matrix for an overlay-layer draw relative to its nearest overlay-layer ancestor.
    fn overlay_layer_model_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4>;
    /// Returns the nearest inherited special layer for this transform.
    fn transform_special_layer(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<LayerType>;
    /// Returns whether `transform_index` is a descendant of an active overlay-layer ancestor.
    fn transform_is_in_overlay_layer(&self, id: RenderSpaceId, transform_index: usize) -> bool;
    /// Returns whether the effective render-context transform chain collapses object scale.
    fn transform_has_degenerate_scale_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> bool;
    /// Material override for the given renderer + slot in the given render context.
    fn overridden_material_asset_id(
        &self,
        space_id: RenderSpaceId,
        context: RenderingContext,
        skinned: bool,
        renderable_index: usize,
        slot_index: usize,
    ) -> Option<i32>;
}

/// Read-only light queries used by frame resource preparation.
pub(crate) trait SceneLightRead: SceneTransformRead + Send + Sync {
    /// Appends render-context-aware world-space lights for `id` into `out`.
    fn resolve_lights_for_render_context_into(
        &self,
        id: RenderSpaceId,
        context: RenderingContext,
        head_output_transform: Mat4,
        out: &mut Vec<ResolvedLight>,
    );
    /// Estimates cached light rows visible to a view's optional render-space filter.
    fn candidate_light_count_for_render_space_filter(
        &self,
        render_space_filter: Option<RenderSpaceId>,
    ) -> usize;
}

impl<'a> RenderSpaceRead<'a> for RenderSpaceView<'a> {
    fn is_active(self) -> bool {
        RenderSpaceView::is_active(self)
    }

    fn is_overlay(self) -> bool {
        RenderSpaceView::is_overlay(self)
    }

    fn is_private(self) -> bool {
        RenderSpaceView::is_private(self)
    }

    fn root_transform(self) -> &'a RenderTransform {
        RenderSpaceView::root_transform(self)
    }

    fn view_transform(self) -> &'a RenderTransform {
        RenderSpaceView::view_transform(self)
    }

    fn local_transforms(self) -> &'a [RenderTransform] {
        RenderSpaceView::local_transforms(self)
    }

    fn node_parents(self) -> &'a [i32] {
        RenderSpaceView::node_parents(self)
    }
}
