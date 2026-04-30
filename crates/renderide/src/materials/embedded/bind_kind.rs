//! Texture-shape discriminator shared by embedded `@group(1)` placeholder and resolve paths.
//!
//! Sampler hashing, sampler descriptor building, sampler creation, and white-texture upload all
//! switch on the same three texture shapes (2D, 3D, cubemap). [`TextureBindKind`] lets those code
//! paths fold what used to be three near-identical helper triplets into one match per concern.

/// Texture shape selector for embedded `@group(1)` bindings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TextureBindKind {
    /// Standard 2D texture.
    Tex2D,
    /// Volumetric 3D texture.
    Tex3D,
    /// Cubemap (six 2D faces).
    Cube,
}
