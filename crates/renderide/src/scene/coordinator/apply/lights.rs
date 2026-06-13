//! Phase C: borrowed views of the still-serial light update payloads.
//!
//! Light updates mutate the shared [`crate::scene::LightCache`], so they stay serial after the
//! parallel Phase B apply completes.

use crate::shared::{LightRenderablesUpdate, LightsBufferRendererUpdate, RenderSpaceUpdate};

/// Borrowed view of the still-serial light-update payloads for a [`RenderSpaceUpdate`].
///
/// Carried alongside the parallel-applied per-space payloads so the post-parallel light pass can
/// re-walk the host updates without re-scanning [`crate::shared::FrameSubmitData::render_spaces`].
pub(in crate::scene::coordinator) struct LightUpdateView<'a> {
    /// Render space identity (mirrors [`RenderSpaceUpdate::id`]).
    pub space_id: i32,
    /// Optional [`crate::shared::LightRenderablesUpdate`] payload (regular [`crate::shared::LightState`] rows).
    pub lights_update: Option<&'a LightRenderablesUpdate>,
    /// Optional [`crate::shared::LightsBufferRendererUpdate`] payload (buffer-based lights).
    pub lights_buffer_renderers_update: Option<&'a LightsBufferRendererUpdate>,
}

/// Borrows the still-serial light update fields from a [`RenderSpaceUpdate`].
pub(in crate::scene::coordinator) fn light_updates_view(
    update: &RenderSpaceUpdate,
) -> LightUpdateView<'_> {
    LightUpdateView {
        space_id: update.id,
        lights_update: update.lights_update.as_ref(),
        lights_buffer_renderers_update: update.lights_buffer_renderers_update.as_ref(),
    }
}
