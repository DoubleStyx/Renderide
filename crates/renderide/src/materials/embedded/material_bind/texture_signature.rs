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
        let (bias, storage_v_inverted) = lod_bias_and_v_inversion(binding, pools);
        bias.to_bits().hash(&mut h);
        storage_v_inverted.hash(&mut h);
    }
    h.finish()
}

fn lod_bias_and_v_inversion(
    binding: ResolvedTextureBinding,
    pools: &EmbeddedTexturePools<'_>,
) -> (f32, bool) {
    match binding {
        ResolvedTextureBinding::Texture2D { asset_id } => {
            pools.texture.get(asset_id).map_or((0.0, false), |t| {
                (t.sampler.mipmap_bias, t.storage_v_inverted)
            })
        }
        ResolvedTextureBinding::Texture3D { asset_id } => pools
            .texture3d
            .get(asset_id)
            .map_or((0.0, false), |t| (t.sampler.mipmap_bias, false)),
        ResolvedTextureBinding::Cubemap { asset_id } => {
            pools.cubemap.get(asset_id).map_or((0.0, false), |t| {
                (t.sampler.mipmap_bias, t.storage_v_inverted)
            })
        }
        ResolvedTextureBinding::RenderTexture { .. } => (0.0, true),
        ResolvedTextureBinding::VideoTexture { .. } => (0.0, false),
        ResolvedTextureBinding::None => (0.0, false),
    }
}
