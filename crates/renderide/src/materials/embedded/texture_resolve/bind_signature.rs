//! Hashes the per-frame signature used to invalidate the embedded `@group(1)` bind cache.
//!
//! [`texture_bind_signature`] visits every reflected texture entry and folds in the resolved
//! binding plus the live texture-pool state (mip residency, sampler state, V-flip flag). When the
//! resulting hash changes, the cached `wgpu::BindGroup` for the host material has to be rebuilt.

use std::hash::{Hash, Hasher};

use ahash::AHasher;

use crate::gpu_pools::SamplerState;
use crate::materials::ReflectedRasterLayout;
use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};

use super::super::layout::StemEmbeddedPropertyIds;
use super::super::texture_pools::EmbeddedTexturePools;
use super::lookup::{
    ResolvedTextureBinding, resolved_texture_binding_for_host, texture_property_ids_for_binding,
};

/// Hashes the mode-affecting fields of a unified [`SamplerState`] for bind-cache invalidation.
pub(crate) fn hash_sampler_state(state: &SamplerState, h: &mut impl Hasher) {
    (state.filter_mode as i32).hash(h);
    state.aniso_level.hash(h);
    (state.wrap_u as i32).hash(h);
    (state.wrap_v as i32).hash(h);
    (state.wrap_w as i32).hash(h);
    state.mipmap_bias.to_bits().hash(h);
}

/// Fingerprint for bind cache invalidation when texture views or residency change.
///
/// When `offscreen_write_render_texture_asset_id` is [`Some`], that render-texture id is treated as
/// non-resident (offscreen color target; self-sampling is masked).
pub(crate) fn texture_bind_signature(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pools: &EmbeddedTexturePools<'_>,
    primary_texture_2d: i32,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> u64 {
    let mut h = AHasher::default();
    offscreen_write_render_texture_asset_id.hash(&mut h);
    for entry in &reflected.material_entries {
        if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            continue;
        }
        let Some(name) = reflected.material_group1_names.get(&entry.binding) else {
            continue;
        };
        let texture_pids = texture_property_ids_for_binding(ids, entry.binding);
        if texture_pids.is_empty() {
            continue;
        }
        let binding = resolved_texture_binding_for_host(
            name.as_str(),
            texture_pids,
            primary_texture_2d,
            store,
            lookup,
        );
        entry.binding.hash(&mut h);
        name.hash(&mut h);
        binding.hash_for_signature(&mut h);
        match binding {
            ResolvedTextureBinding::None => false.hash(&mut h),
            ResolvedTextureBinding::Texture2D { asset_id } => {
                if let Some(t) = pools.texture.get(asset_id) {
                    let resident = t.mip_levels_resident > 0;
                    resident.hash(&mut h);
                    t.mip_levels_resident.hash(&mut h);
                    t.storage_v_inverted.hash(&mut h);
                    hash_sampler_state(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::Texture3D { asset_id } => {
                if let Some(t) = pools.texture3d.get(asset_id) {
                    let resident = t.mip_levels_resident > 0;
                    resident.hash(&mut h);
                    t.mip_levels_resident.hash(&mut h);
                    hash_sampler_state(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::Cubemap { asset_id } => {
                if let Some(t) = pools.cubemap.get(asset_id) {
                    let resident = t.mip_levels_resident > 0;
                    resident.hash(&mut h);
                    t.mip_levels_resident.hash(&mut h);
                    t.storage_v_inverted.hash(&mut h);
                    hash_sampler_state(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                if offscreen_write_render_texture_asset_id == Some(asset_id) {
                    false.hash(&mut h);
                } else if let Some(t) = pools.render_texture.get(asset_id) {
                    t.is_sampleable().hash(&mut h);
                    hash_sampler_state(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::VideoTexture { asset_id } => {
                if let Some(t) = pools.video_texture.get(asset_id) {
                    t.is_sampleable().hash(&mut h);
                    hash_sampler_state(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::shared::{TextureFilterMode, TextureWrapMode};

    /// Hashes the same sampler fields used by `texture_bind_signature` for 2D/render textures.
    fn sampler_signature_for(state: &SamplerState) -> u64 {
        let mut hasher = AHasher::default();
        hash_sampler_state(state, &mut hasher);
        hasher.finish()
    }

    /// Builds a 2D sampler state with the supplied U/V wrap modes.
    fn texture2d_sampler_state(wrap_u: TextureWrapMode, wrap_v: TextureWrapMode) -> SamplerState {
        SamplerState {
            filter_mode: TextureFilterMode::Bilinear,
            aniso_level: 8,
            wrap_u,
            wrap_v,
            wrap_w: TextureWrapMode::default(),
            mipmap_bias: 0.0,
        }
    }

    /// Changing U wrap changes the sampler portion of the material bind signature.
    #[test]
    fn bind_signature_sampler_hash_distinguishes_render_texture_wrap_u() {
        let repeat = texture2d_sampler_state(TextureWrapMode::Repeat, TextureWrapMode::Clamp);
        let clamp = texture2d_sampler_state(TextureWrapMode::Clamp, TextureWrapMode::Clamp);

        assert_ne!(
            sampler_signature_for(&repeat),
            sampler_signature_for(&clamp)
        );
    }

    /// Changing V wrap changes the sampler portion of the material bind signature.
    #[test]
    fn bind_signature_sampler_hash_distinguishes_render_texture_wrap_v() {
        let repeat = texture2d_sampler_state(TextureWrapMode::Clamp, TextureWrapMode::Repeat);
        let clamp = texture2d_sampler_state(TextureWrapMode::Clamp, TextureWrapMode::Clamp);

        assert_ne!(
            sampler_signature_for(&repeat),
            sampler_signature_for(&clamp)
        );
    }
}
