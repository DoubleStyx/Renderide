//! View-local render-space and layer visibility policy for world-mesh collection.

use crate::scene::RenderSpaceId;
use crate::shared::LayerType;

/// Unity layer visibility behavior applied while collecting draws for one view.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ViewLayerPolicy {
    /// Regular world rendering. Hidden and overlay roots are excluded unless a selective camera
    /// transform filter explicitly exposes them.
    #[default]
    MainView,
    /// Host camera rendering with camera culling masks and private-UI opt-in.
    Camera {
        /// Whether private render spaces are visible to a non-selective camera.
        render_private_ui: bool,
    },
    /// Desktop overlay camera rendering. Only overlay roots are included.
    DesktopOverlay,
}

impl ViewLayerPolicy {
    /// Builds a camera layer policy from the host camera's `renderPrivateUI` flag.
    pub const fn camera(render_private_ui: bool) -> Self {
        Self::Camera { render_private_ui }
    }

    /// Returns whether this layer policy includes private render spaces.
    pub(crate) const fn shows_private_render_space(self, has_selective_roots: bool) -> bool {
        match self {
            Self::MainView => true,
            Self::Camera { render_private_ui } => has_selective_roots || render_private_ui,
            Self::DesktopOverlay => true,
        }
    }

    /// Returns whether a renderer under `special_layer` is visible for this view.
    pub(super) fn shows_special_layer(
        self,
        special_layer: Option<LayerType>,
        has_selective_roots: bool,
    ) -> bool {
        match self {
            Self::MainView => match special_layer {
                Some(LayerType::Hidden | LayerType::Overlay) => has_selective_roots,
                _ => true,
            },
            Self::Camera { .. } => match special_layer {
                Some(LayerType::Hidden | LayerType::Overlay) => has_selective_roots,
                _ => true,
            },
            Self::DesktopOverlay => matches!(special_layer, Some(LayerType::Overlay)),
        }
    }

    /// Returns the overlay draw flag to emit for a visible renderer in this view.
    pub(super) fn effective_overlay(self, is_overlay: bool) -> bool {
        is_overlay && !matches!(self, Self::Camera { .. })
    }
}

/// Render-space scope applied while collecting draws for one planned view.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ViewRenderSpaceScope {
    /// Draw all active render spaces that pass the view's layer policy.
    #[default]
    AllActive,
    /// Draw only the specified render space if it passes the view's layer policy.
    Single(RenderSpaceId),
}

impl ViewRenderSpaceScope {
    /// Builds a scope for one render space.
    pub const fn single(space_id: RenderSpaceId) -> Self {
        Self::Single(space_id)
    }

    /// Returns whether `space_id` is inside this scope before layer-policy filtering.
    pub(crate) const fn includes(self, space_id: RenderSpaceId) -> bool {
        match self {
            Self::AllActive => true,
            Self::Single(scoped_id) => scoped_id.0 == space_id.0,
        }
    }

    /// Returns the single scoped render space, when this scope is fixed to one space.
    pub(crate) const fn single_space(self) -> Option<RenderSpaceId> {
        match self {
            Self::AllActive => None,
            Self::Single(space_id) => Some(space_id),
        }
    }
}
