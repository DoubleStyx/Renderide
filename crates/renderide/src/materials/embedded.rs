//! Embedded raster materials: WGSL reflection, texture resolution, uniform packing, and `@group(1)` bind groups.

mod bind_kind;
mod embedded_material_bind_error;
mod layout;
mod material_bind;
mod texture_pools;
pub(crate) mod texture_resolve;
mod uniform_pack;

pub use embedded_material_bind_error::EmbeddedMaterialBindError;
pub use material_bind::EmbeddedMaterialBindResources;
pub use texture_pools::EmbeddedTexturePools;
