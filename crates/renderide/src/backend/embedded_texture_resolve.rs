//! Texture asset id resolution and bind signature hashing for embedded material bind groups.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::assets::texture::texture2d_asset_id_from_packed;
use crate::materials::ReflectedRasterLayout;
use crate::resources::{Texture2dSamplerState, TexturePool};

use super::embedded_material_layout::StemEmbeddedPropertyIds;

/// Resolves primary 2D texture asset id from reflected material entries.
pub(crate) fn primary_texture_2d_asset_id(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> i32 {
    for entry in &reflected.material_entries {
        if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            let Some(pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
                continue;
            };
            if let Some(MaterialPropertyValue::Texture(packed)) = store.get_merged(lookup, *pid) {
                return texture2d_asset_id_from_packed(*packed).unwrap_or(-1);
            }
        }
    }
    -1
}

pub(crate) fn should_fallback_to_primary_texture(host_name: &str) -> bool {
    matches!(host_name, "_MainTex" | "_Tex" | "_TEXTURE")
}

fn texture_property_asset_id_by_pid(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> i32 {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Texture(packed)) => {
            texture2d_asset_id_from_packed(*packed).unwrap_or(-1)
        }
        _ => -1,
    }
}

/// Resolves resident texture asset id for a host property name, with primary-texture fallback.
pub(crate) fn resolved_texture_asset_id_for_host(
    host_name: &str,
    texture_property_id: i32,
    primary_texture_2d: i32,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> i32 {
    let tid = texture_property_asset_id_by_pid(store, lookup, texture_property_id);
    if tid >= 0 {
        return tid;
    }
    if should_fallback_to_primary_texture(host_name) {
        return primary_texture_2d;
    }
    -1
}

/// Fingerprint for bind cache invalidation when texture views or residency change.
pub(crate) fn texture_bind_signature(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    texture_pool: &TexturePool,
    primary_texture_2d: i32,
) -> u64 {
    let mut h = DefaultHasher::new();
    for entry in &reflected.material_entries {
        if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            continue;
        }
        let Some(name) = reflected.material_group1_names.get(&entry.binding) else {
            continue;
        };
        let Some(&texture_pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
            continue;
        };
        let texture_asset_id = resolved_texture_asset_id_for_host(
            name.as_str(),
            texture_pid,
            primary_texture_2d,
            store,
            lookup,
        );
        entry.binding.hash(&mut h);
        name.hash(&mut h);
        texture_asset_id.hash(&mut h);
        texture_pool
            .get_texture(texture_asset_id)
            .is_some_and(|t| t.mip_levels_resident > 0)
            .hash(&mut h);
    }
    h.finish()
}

pub(crate) fn sampler_from_state(
    device: &wgpu::Device,
    state: &Texture2dSamplerState,
) -> wgpu::Sampler {
    let address_mode_u = match state.wrap_u {
        crate::shared::TextureWrapMode::repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::mirror_once => wgpu::AddressMode::ClampToEdge,
    };
    let address_mode_v = match state.wrap_v {
        crate::shared::TextureWrapMode::repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::mirror_once => wgpu::AddressMode::ClampToEdge,
    };
    let (mag, min, mipmap) = match state.filter_mode {
        crate::shared::TextureFilterMode::point => (
            wgpu::FilterMode::Nearest,
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Nearest,
        ),
        crate::shared::TextureFilterMode::bilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::trilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::anisotropic => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("embedded_texture_sampler"),
        address_mode_u,
        address_mode_v,
        address_mode_w: address_mode_u,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mipmap,
        ..Default::default()
    })
}
