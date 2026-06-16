//! Frame-global light-cookie atlases and GPU blit support.

/// Cookie request assignment into atlas slots.
mod assignment;
/// CPU-authored atlas texture initialization.
mod atlas;
/// GPU copy pipelines from host textures into atlas layers.
mod blit;
/// Texture format and sampler compatibility helpers.
mod format;
/// CPU-side packed atlas layout.
mod packing;
/// Render-graph pass wrapper for atlas updates.
mod pass;
/// Persistent GPU resources for light-cookie atlases.
mod resources;
#[cfg(test)]
/// Light-cookie atlas unit tests.
mod tests;

/// Cubemap face count.
pub(super) const POINT_COOKIE_FACE_COUNT: u32 = 6;

pub(crate) use pass::{LIGHT_COOKIE_ATLAS_PASS_NAME, LightCookieAtlasPass};
pub(super) use resources::LightCookieAtlasResources;
