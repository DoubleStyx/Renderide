//! Borrowed view of resident texture pools for embedded [`super::EmbeddedMaterialBindResources`] `@group(1)` resolution.

use crate::gpu_pools::{
    CubemapPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};

/// References to the GPU pool tables used when hashing bind signatures and resolving texture views/samplers.
///
/// Passed into [`super::EmbeddedMaterialBindResources`] methods so callers (e.g. [`crate::backend::RenderBackend`])
/// do not thread separate pool parameters through every embedded draw path.
pub struct EmbeddedTexturePools<'a> {
    /// Resident [`TexturePool`] (2D textures).
    pub texture: &'a TexturePool,
    /// Resident [`Texture3dPool`].
    pub texture3d: &'a Texture3dPool,
    /// Resident [`CubemapPool`].
    pub cubemap: &'a CubemapPool,
    /// Host render-texture targets ([`RenderTexturePool`]).
    pub render_texture: &'a RenderTexturePool,
    /// Resident [`VideoTexturePool`].
    pub video_texture: &'a VideoTexturePool,
}
