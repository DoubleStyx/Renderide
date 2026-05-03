//! Texture-state hash that drives uniform-buffer refresh independently of the property store.
//!
//! The store's mutation generation only fires on host property writes, but the embedded uniform
//! block consumes texture-pool state too (`_<Tex>_LodBias`, `_<Tex>_StorageVInverted`). When that
//! state changes without a property write, the signature here detects the change and forces the
//! cached uniform buffer to refresh.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ahash::AHasher;

use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};
use crate::shared::ColorProfile;

use super::super::layout::StemMaterialLayout;
use super::super::texture_pools::EmbeddedTexturePools;
use super::super::texture_resolve::{
    ResolvedTextureBinding, resolved_texture_binding_for_host, texture_property_ids_for_binding,
};

/// Hashes texture-pool metadata read by the reflected material uniform block.
pub(super) fn compute_uniform_texture_state_signature(
    layout: &Arc<StemMaterialLayout>,
    pools: &EmbeddedTexturePools<'_>,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    primary_texture_2d: i32,
) -> u64 {
    let mut h = AHasher::default();
    for entry in &layout.reflected.material_entries {
        if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            continue;
        }
        let Some(name) = layout.reflected.material_group1_names.get(&entry.binding) else {
            continue;
        };
        let pids = texture_property_ids_for_binding(layout.ids.as_ref(), entry.binding);
        if pids.is_empty() {
            continue;
        }
        let binding = resolved_texture_binding_for_host(
            name.as_str(),
            pids,
            primary_texture_2d,
            store,
            lookup,
        );
        entry.binding.hash(&mut h);
        let (bias, storage_v_inverted, color_profile) = texture_uniform_state(binding, pools);
        bias.to_bits().hash(&mut h);
        storage_v_inverted.hash(&mut h);
        color_profile.hash(&mut h);
    }
    h.finish()
}

fn texture_uniform_state(
    binding: ResolvedTextureBinding,
    pools: &EmbeddedTexturePools<'_>,
) -> (f32, bool, i32) {
    match binding {
        ResolvedTextureBinding::Texture2D { asset_id } => {
            pools.texture.get(asset_id).map_or((0.0, false, -1), |t| {
                (
                    t.sampler.mipmap_bias,
                    t.storage_v_inverted,
                    color_profile_signature_value(Some(t.color_profile)),
                )
            })
        }
        ResolvedTextureBinding::Texture3D { asset_id } => pools
            .texture3d
            .get(asset_id)
            .map_or((0.0, false, -1), |t| (t.sampler.mipmap_bias, false, -1)),
        ResolvedTextureBinding::Cubemap { asset_id } => {
            pools.cubemap.get(asset_id).map_or((0.0, false, -1), |t| {
                (t.sampler.mipmap_bias, t.storage_v_inverted, -1)
            })
        }
        ResolvedTextureBinding::RenderTexture { .. } => (0.0, true, -1),
        ResolvedTextureBinding::VideoTexture { .. } => (0.0, false, -1),
        ResolvedTextureBinding::None => (0.0, false, -1),
    }
}

fn color_profile_signature_value(profile: Option<ColorProfile>) -> i32 {
    profile.map_or(-1, |profile| profile as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture2d_color_profile_signature_distinguishes_text_modes() {
        let missing = color_profile_signature_value(None);
        let linear = color_profile_signature_value(Some(ColorProfile::Linear));
        let srgb = color_profile_signature_value(Some(ColorProfile::SRGB));
        let srgb_alpha = color_profile_signature_value(Some(ColorProfile::SRGBAlpha));

        assert_ne!(missing, linear);
        assert_ne!(linear, srgb);
        assert_eq!(srgb, 1);
        assert_eq!(srgb_alpha, 2);
    }
}
