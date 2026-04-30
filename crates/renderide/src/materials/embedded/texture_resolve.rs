//! Texture asset id resolution and bind signature hashing for embedded material bind groups.

mod bind_signature;
mod lookup;
mod sampler;

pub(crate) use super::bind_kind::TextureBindKind;
pub(crate) use bind_signature::texture_bind_signature;
pub(crate) use lookup::{
    ResolvedTextureBinding, primary_texture_2d_asset_id, resolved_texture_binding_for_host,
    texture_property_ids_for_binding,
};
pub(crate) use sampler::{create_sampler, default_embedded_sampler};

#[cfg(test)]
pub(crate) use lookup::should_fallback_to_primary_texture;
