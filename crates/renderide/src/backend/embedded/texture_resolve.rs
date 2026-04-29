//! Texture asset id resolution and bind signature hashing for embedded material bind groups.

use std::hash::{Hash, Hasher};

use ahash::AHasher;

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::assets::texture::{
    texture2d_asset_id_from_packed, unpack_host_texture_packed, HostTextureAssetKind,
};
use crate::materials::ReflectedRasterLayout;
use crate::resources::{CubemapSamplerState, Texture2dSamplerState, Texture3dSamplerState};
use crate::shared::{TextureFilterMode, TextureWrapMode};

use super::layout::{shader_writer_unescaped_property_name, StemEmbeddedPropertyIds};
use super::texture_pools::EmbeddedTexturePools;

/// Resolved GPU texture binding for a material property (packed host id or primary fallback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedTextureBinding {
    /// No texture or unsupported packed type.
    None,
    /// [`crate::resources::TexturePool`] entry (unpacked 2D asset id).
    Texture2D { asset_id: i32 },
    /// [`crate::resources::Texture3dPool`] entry (unpacked 3D asset id).
    Texture3D { asset_id: i32 },
    /// [`crate::resources::CubemapPool`] entry (unpacked cubemap asset id).
    Cubemap { asset_id: i32 },
    /// [`crate::resources::RenderTexturePool`] entry (unpacked render-texture asset id).
    RenderTexture { asset_id: i32 },
    /// [`crate::resources::VideoTexturePool`] entry (unpacked 2D asset id).
    VideoTexture { asset_id: i32 },
}

impl ResolvedTextureBinding {
    fn hash_for_signature(self, hasher: &mut impl Hasher) {
        match self {
            ResolvedTextureBinding::None => {
                0u8.hash(hasher);
            }
            ResolvedTextureBinding::Texture2D { asset_id } => {
                1u8.hash(hasher);
                asset_id.hash(hasher);
            }
            ResolvedTextureBinding::Texture3D { asset_id } => {
                3u8.hash(hasher);
                asset_id.hash(hasher);
            }
            ResolvedTextureBinding::Cubemap { asset_id } => {
                4u8.hash(hasher);
                asset_id.hash(hasher);
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                2u8.hash(hasher);
                asset_id.hash(hasher);
            }
            ResolvedTextureBinding::VideoTexture { asset_id } => {
                5u8.hash(hasher);
                asset_id.hash(hasher);
            }
        }
    }
}

/// Property ids to try for one reflected texture binding, in priority order:
/// the exact WGSL global name first, followed by host-side aliases such as
/// `_MaskTex` -> `MaskTexture`.
pub(crate) fn texture_property_ids_for_binding(
    ids: &StemEmbeddedPropertyIds,
    binding: u32,
) -> &[i32] {
    //perf xlinka: aliases are built once per stem, no tiny Vec per texture bind.
    ids.texture_binding_property_ids
        .get(&binding)
        .map_or(&[], |pids| pids.as_ref())
}

fn first_material_texture_binding(reflected: &ReflectedRasterLayout) -> Option<u32> {
    reflected
        .material_entries
        .iter()
        .find(|entry| matches!(entry.ty, wgpu::BindingType::Texture { .. }))
        .map(|entry| entry.binding)
}

/// Resolves the primary 2D texture asset id from the first reflected material texture slot.
pub(crate) fn primary_texture_2d_asset_id(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> i32 {
    let Some(binding) = first_material_texture_binding(reflected) else {
        return -1;
    };
    for &pid in texture_property_ids_for_binding(ids, binding) {
        if let Some(MaterialPropertyValue::Texture(packed)) = store.get_merged(lookup, pid) {
            return texture2d_asset_id_from_packed(*packed).unwrap_or(-1);
        }
    }
    -1
}

/// Whether `host_name` is the canonical primary-texture name for which we should fall back to
/// the bound primary texture when no explicit binding is present.
///
/// Only `_MainTex` and `_Tex` are accepted: the host writes one of these from every primary
/// texture call (`_MainTex` everywhere except `UnlitMaterial` which uses `_Tex`). The
/// previously-carried `_TEXTURE` and `Texture` arms were confirmed dead by auditing the host's
/// `MaterialProperty("…")` declarations and were removed.
pub(crate) fn should_fallback_to_primary_texture(host_name: &str) -> bool {
    let host_name = shader_writer_unescaped_property_name(host_name);
    matches!(host_name, "_MainTex" | "_Tex")
}

fn texture_property_binding(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> ResolvedTextureBinding {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Texture(packed)) => match unpack_host_texture_packed(*packed) {
            Some((id, HostTextureAssetKind::Texture2D)) => {
                ResolvedTextureBinding::Texture2D { asset_id: id }
            }
            Some((id, HostTextureAssetKind::Texture3D)) => {
                ResolvedTextureBinding::Texture3D { asset_id: id }
            }
            Some((id, HostTextureAssetKind::Cubemap)) => {
                ResolvedTextureBinding::Cubemap { asset_id: id }
            }
            Some((id, HostTextureAssetKind::RenderTexture)) => {
                ResolvedTextureBinding::RenderTexture { asset_id: id }
            }
            Some((id, HostTextureAssetKind::VideoTexture)) => {
                ResolvedTextureBinding::VideoTexture { asset_id: id }
            }
            _ => ResolvedTextureBinding::None,
        },
        _ => ResolvedTextureBinding::None,
    }
}

/// Resolves resident texture binding for a host property name, with primary-texture fallback for 2D-only slots.
pub(crate) fn resolved_texture_binding_for_host(
    host_name: &str,
    texture_property_ids: &[i32],
    primary_texture_2d: i32,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> ResolvedTextureBinding {
    for &texture_property_id in texture_property_ids {
        let b = texture_property_binding(store, lookup, texture_property_id);
        if !matches!(b, ResolvedTextureBinding::None) {
            return b;
        }
    }
    if should_fallback_to_primary_texture(host_name) && primary_texture_2d >= 0 {
        return ResolvedTextureBinding::Texture2D {
            asset_id: primary_texture_2d,
        };
    }
    ResolvedTextureBinding::None
}

fn hash_texture2d_sampler(state: &Texture2dSamplerState, h: &mut impl Hasher) {
    (state.filter_mode as i32).hash(h);
    state.aniso_level.hash(h);
    (state.wrap_u as i32).hash(h);
    (state.wrap_v as i32).hash(h);
    state.mipmap_bias.to_bits().hash(h);
}

fn hash_texture3d_sampler(state: &Texture3dSamplerState, h: &mut impl Hasher) {
    (state.filter_mode as i32).hash(h);
    state.aniso_level.hash(h);
    (state.wrap_u as i32).hash(h);
    (state.wrap_v as i32).hash(h);
    (state.wrap_w as i32).hash(h);
    state.mipmap_bias.to_bits().hash(h);
}

fn hash_cubemap_sampler(state: &CubemapSamplerState, h: &mut impl Hasher) {
    (state.filter_mode as i32).hash(h);
    state.aniso_level.hash(h);
    state.mipmap_bias.to_bits().hash(h);
    (state.wrap_u as i32).hash(h);
    (state.wrap_v as i32).hash(h);
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
                if let Some(t) = pools.texture.get_texture(asset_id) {
                    let resident = t.mip_levels_resident > 0;
                    resident.hash(&mut h);
                    t.mip_levels_resident.hash(&mut h);
                    t.storage_v_inverted.hash(&mut h);
                    hash_texture2d_sampler(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::Texture3D { asset_id } => {
                if let Some(t) = pools.texture3d.get_texture(asset_id) {
                    let resident = t.mip_levels_resident > 0;
                    resident.hash(&mut h);
                    t.mip_levels_resident.hash(&mut h);
                    hash_texture3d_sampler(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::Cubemap { asset_id } => {
                if let Some(t) = pools.cubemap.get_texture(asset_id) {
                    let resident = t.mip_levels_resident > 0;
                    resident.hash(&mut h);
                    t.mip_levels_resident.hash(&mut h);
                    t.storage_v_inverted.hash(&mut h);
                    hash_cubemap_sampler(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                if offscreen_write_render_texture_asset_id == Some(asset_id) {
                    false.hash(&mut h);
                } else if let Some(t) = pools.render_texture.get(asset_id) {
                    t.is_sampleable().hash(&mut h);
                    hash_texture2d_sampler(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
            ResolvedTextureBinding::VideoTexture { asset_id } => {
                if let Some(t) = pools.video_texture.get(asset_id) {
                    t.is_sampleable().hash(&mut h);
                    hash_texture2d_sampler(&t.sampler, &mut h);
                } else {
                    false.hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

/// Wgpu filter triplet derived from the host texture filter mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSamplerFilter {
    /// Magnification filter.
    pub(crate) mag_filter: wgpu::FilterMode,
    /// Minification filter.
    pub(crate) min_filter: wgpu::FilterMode,
    /// Mip-level selection filter.
    pub(crate) mipmap_filter: wgpu::MipmapFilterMode,
}

/// Texture address modes for a sampler descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSamplerAddress {
    /// U/S address mode.
    pub(crate) u: wgpu::AddressMode,
    /// V/T address mode.
    pub(crate) v: wgpu::AddressMode,
    /// W/R address mode.
    pub(crate) w: wgpu::AddressMode,
}

/// Builds a sampler descriptor from host texture settings and current mip residency.
pub(crate) fn sampler_descriptor_from_parts(
    label: &'static str,
    address: ResolvedSamplerAddress,
    filter_mode: TextureFilterMode,
    aniso_level: i32,
    mip_levels_resident: u32,
) -> wgpu::SamplerDescriptor<'static> {
    let filter = filter_mode_to_wgpu(filter_mode);
    wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address.u,
        address_mode_v: address.v,
        address_mode_w: address.w,
        mag_filter: filter.mag_filter,
        min_filter: filter.min_filter,
        mipmap_filter: filter.mipmap_filter,
        lod_min_clamp: 0.0,
        lod_max_clamp: mip_levels_resident.saturating_sub(1) as f32,
        anisotropy_clamp: anisotropy_clamp(filter_mode, aniso_level, filter),
        ..Default::default()
    }
}

/// Builds a sampler descriptor for [`Texture2dSamplerState`].
pub(crate) fn sampler_descriptor_from_state(
    state: &Texture2dSamplerState,
    mip_levels_resident: u32,
) -> wgpu::SamplerDescriptor<'static> {
    let address_mode_u = wrap_to_address(state.wrap_u);
    let address_mode_v = wrap_to_address(state.wrap_v);
    sampler_descriptor_from_parts(
        "embedded_texture_sampler",
        ResolvedSamplerAddress {
            u: address_mode_u,
            v: address_mode_v,
            w: address_mode_u,
        },
        state.filter_mode,
        state.aniso_level,
        mip_levels_resident,
    )
}

/// Builds a sampler descriptor for [`Texture3dSamplerState`].
pub(crate) fn sampler_descriptor_from_texture3d_state(
    state: &Texture3dSamplerState,
    mip_levels_resident: u32,
) -> wgpu::SamplerDescriptor<'static> {
    sampler_descriptor_from_parts(
        "embedded_texture3d_sampler",
        ResolvedSamplerAddress {
            u: wrap_to_address(state.wrap_u),
            v: wrap_to_address(state.wrap_v),
            w: wrap_to_address(state.wrap_w),
        },
        state.filter_mode,
        state.aniso_level,
        mip_levels_resident,
    )
}

/// Builds a sampler descriptor for [`CubemapSamplerState`].
pub(crate) fn sampler_descriptor_from_cubemap_state(
    state: &CubemapSamplerState,
    mip_levels_resident: u32,
) -> wgpu::SamplerDescriptor<'static> {
    let address_mode_u = wrap_to_address(state.wrap_u);
    sampler_descriptor_from_parts(
        "embedded_cubemap_sampler",
        ResolvedSamplerAddress {
            u: address_mode_u,
            v: wrap_to_address(state.wrap_v),
            w: address_mode_u,
        },
        state.filter_mode,
        state.aniso_level,
        mip_levels_resident,
    )
}

/// Builds the fallback sampler used with the embedded white placeholder textures.
pub(crate) fn default_embedded_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    let descriptor = sampler_descriptor_from_parts(
        "embedded_default_sampler",
        ResolvedSamplerAddress {
            u: wgpu::AddressMode::Repeat,
            v: wgpu::AddressMode::Repeat,
            w: wgpu::AddressMode::Repeat,
        },
        TextureFilterMode::Trilinear,
        1,
        1,
    );
    device.create_sampler(&descriptor)
}

/// Builds a sampler for [`Texture3dSamplerState`] (three address modes).
pub(crate) fn sampler_from_texture3d_state(
    device: &wgpu::Device,
    state: &Texture3dSamplerState,
    mip_levels_resident: u32,
) -> wgpu::Sampler {
    let descriptor = sampler_descriptor_from_texture3d_state(state, mip_levels_resident);
    device.create_sampler(&descriptor)
}

/// Builds a sampler for [`CubemapSamplerState`].
pub(crate) fn sampler_from_cubemap_state(
    device: &wgpu::Device,
    state: &CubemapSamplerState,
    mip_levels_resident: u32,
) -> wgpu::Sampler {
    let descriptor = sampler_descriptor_from_cubemap_state(state, mip_levels_resident);
    device.create_sampler(&descriptor)
}

/// Converts a host wrap mode to a wgpu address mode.
fn wrap_to_address(w: TextureWrapMode) -> wgpu::AddressMode {
    match w {
        TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        TextureWrapMode::Clamp => wgpu::AddressMode::ClampToEdge,
        TextureWrapMode::Mirror => wgpu::AddressMode::MirrorRepeat,
        TextureWrapMode::MirrorOnce => wgpu::AddressMode::ClampToEdge,
    }
}

/// Converts a host filter mode to wgpu filter fields without changing host semantics.
fn filter_mode_to_wgpu(filter_mode: TextureFilterMode) -> ResolvedSamplerFilter {
    match filter_mode {
        TextureFilterMode::Point => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        },
        TextureFilterMode::Bilinear => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        },
        TextureFilterMode::Trilinear => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
        },
        TextureFilterMode::Anisotropic => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
        },
    }
}

/// Returns the wgpu anisotropy clamp for a resolved host sampler mode.
fn anisotropy_clamp(
    filter_mode: TextureFilterMode,
    aniso_level: i32,
    filter: ResolvedSamplerFilter,
) -> u16 {
    if matches!(filter_mode, TextureFilterMode::Anisotropic)
        && filter.mag_filter == wgpu::FilterMode::Linear
        && filter.min_filter == wgpu::FilterMode::Linear
        && filter.mipmap_filter == wgpu::MipmapFilterMode::Linear
    {
        aniso_level.clamp(1, 16) as u16
    } else {
        1
    }
}

/// Builds a sampler for [`Texture2dSamplerState`] (two address modes).
pub(crate) fn sampler_from_state(
    device: &wgpu::Device,
    state: &Texture2dSamplerState,
    mip_levels_resident: u32,
) -> wgpu::Sampler {
    let descriptor = sampler_descriptor_from_state(state, mip_levels_resident);
    device.create_sampler(&descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use hashbrown::HashMap;

    use crate::assets::material::PropertyIdRegistry;
    use crate::backend::embedded::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
    use crate::resources::Texture2dSamplerState;
    use crate::shared::{TextureFilterMode, TextureWrapMode};

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    fn reflected_with_textures(
        names: &[(u32, &str)],
    ) -> (
        ReflectedRasterLayout,
        StemEmbeddedPropertyIds,
        PropertyIdRegistry,
    ) {
        let registry = PropertyIdRegistry::new();
        let mut texture_binding_property_ids = HashMap::new();
        let mut material_group1_names = HashMap::new();
        let mut material_entries = Vec::new();
        for &(binding, name) in names {
            let pid = registry.intern(name);
            texture_binding_property_ids.insert(binding, Arc::from(vec![pid].into_boxed_slice()));
            material_group1_names.insert(binding, name.to_string());
            material_entries.push(texture_entry(binding));
        }
        (
            ReflectedRasterLayout {
                layout_fingerprint: 0,
                material_entries,
                per_draw_entries: Vec::new(),
                material_uniform: None,
                material_group1_names,
                vs_max_vertex_location: None,
                uses_scene_depth_snapshot: false,
                uses_scene_color_snapshot: false,
                requires_intersection_pass: false,
                requires_grab_pass: false,
            },
            StemEmbeddedPropertyIds {
                shared: Arc::new(EmbeddedSharedKeywordIds::new(&registry)),
                uniform_field_ids: HashMap::new(),
                texture_binding_property_ids,
                keyword_field_probe_ids: HashMap::new(),
            },
            registry,
        )
    }

    /// Hashes the same sampler fields used by `texture_bind_signature` for 2D/render textures.
    fn sampler_signature_for(state: &Texture2dSamplerState) -> u64 {
        let mut hasher = AHasher::default();
        hash_texture2d_sampler(state, &mut hasher);
        hasher.finish()
    }

    /// Builds a 2D sampler state with the supplied U/V wrap modes.
    fn texture2d_sampler_state(
        wrap_u: TextureWrapMode,
        wrap_v: TextureWrapMode,
    ) -> Texture2dSamplerState {
        Texture2dSamplerState {
            filter_mode: TextureFilterMode::Bilinear,
            aniso_level: 8,
            wrap_u,
            wrap_v,
            mipmap_bias: 0.0,
        }
    }

    /// Packs a host texture id using the same bit layout as `IdPacker<TextureAssetType>`.
    fn pack_host_texture(asset_id: i32, kind: HostTextureAssetKind) -> i32 {
        ((asset_id as u32) | ((kind as u32) << 29)) as i32
    }

    #[test]
    fn sampler_filter_modes_preserve_host_semantics() {
        let address = ResolvedSamplerAddress {
            u: wgpu::AddressMode::Repeat,
            v: wgpu::AddressMode::Repeat,
            w: wgpu::AddressMode::Repeat,
        };

        let point = sampler_descriptor_from_parts("test", address, TextureFilterMode::Point, 16, 4);
        assert_eq!(point.mag_filter, wgpu::FilterMode::Nearest);
        assert_eq!(point.min_filter, wgpu::FilterMode::Nearest);
        assert_eq!(point.mipmap_filter, wgpu::MipmapFilterMode::Nearest);
        assert_eq!(point.anisotropy_clamp, 1);
        assert_eq!(point.lod_max_clamp, 3.0);

        let bilinear =
            sampler_descriptor_from_parts("test", address, TextureFilterMode::Bilinear, 16, 4);
        assert_eq!(bilinear.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(bilinear.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(bilinear.mipmap_filter, wgpu::MipmapFilterMode::Nearest);
        assert_eq!(bilinear.anisotropy_clamp, 1);

        let trilinear =
            sampler_descriptor_from_parts("test", address, TextureFilterMode::Trilinear, 16, 4);
        assert_eq!(trilinear.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(trilinear.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(trilinear.mipmap_filter, wgpu::MipmapFilterMode::Linear);
        assert_eq!(trilinear.anisotropy_clamp, 1);

        let anisotropic =
            sampler_descriptor_from_parts("test", address, TextureFilterMode::Anisotropic, 64, 4);
        assert_eq!(anisotropic.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(anisotropic.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(anisotropic.mipmap_filter, wgpu::MipmapFilterMode::Linear);
        assert_eq!(anisotropic.anisotropy_clamp, 16);
    }

    #[test]
    fn sampler_descriptors_apply_wrap_anisotropy_and_lod_clamps_for_all_texture_kinds() {
        let texture2d = Texture2dSamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 8,
            wrap_u: TextureWrapMode::Mirror,
            wrap_v: TextureWrapMode::Clamp,
            mipmap_bias: 0.0,
        };
        let texture2d_desc = sampler_descriptor_from_state(&texture2d, 6);
        assert_eq!(
            texture2d_desc.address_mode_u,
            wgpu::AddressMode::MirrorRepeat
        );
        assert_eq!(
            texture2d_desc.address_mode_v,
            wgpu::AddressMode::ClampToEdge
        );
        assert_eq!(
            texture2d_desc.address_mode_w,
            wgpu::AddressMode::MirrorRepeat
        );
        assert_eq!(texture2d_desc.anisotropy_clamp, 8);
        assert_eq!(texture2d_desc.lod_max_clamp, 5.0);

        let texture3d = Texture3dSamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 12,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Mirror,
            wrap_w: TextureWrapMode::Clamp,
            mipmap_bias: 0.0,
        };
        let texture3d_desc = sampler_descriptor_from_texture3d_state(&texture3d, 3);
        assert_eq!(texture3d_desc.address_mode_u, wgpu::AddressMode::Repeat);
        assert_eq!(
            texture3d_desc.address_mode_v,
            wgpu::AddressMode::MirrorRepeat
        );
        assert_eq!(
            texture3d_desc.address_mode_w,
            wgpu::AddressMode::ClampToEdge
        );
        assert_eq!(texture3d_desc.anisotropy_clamp, 12);
        assert_eq!(texture3d_desc.lod_max_clamp, 2.0);

        let cubemap = CubemapSamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 4,
            mipmap_bias: 0.0,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Clamp,
        };
        let cubemap_desc = sampler_descriptor_from_cubemap_state(&cubemap, 1);
        assert_eq!(cubemap_desc.address_mode_u, wgpu::AddressMode::Repeat);
        assert_eq!(cubemap_desc.address_mode_v, wgpu::AddressMode::ClampToEdge);
        assert_eq!(cubemap_desc.address_mode_w, wgpu::AddressMode::Repeat);
        assert_eq!(cubemap_desc.anisotropy_clamp, 4);
        assert_eq!(cubemap_desc.lod_max_clamp, 0.0);
    }

    #[test]
    fn resolved_texture_binding_uses_alias_property_id() {
        let mut store = MaterialPropertyStore::new();
        let exact_mask_tex_pid = 10;
        let alias_mask_texture_pid = 11;
        store.set_material(
            4,
            alias_mask_texture_pid,
            MaterialPropertyValue::Texture(123),
        );

        assert_eq!(
            resolved_texture_binding_for_host(
                "_MaskTex",
                &[exact_mask_tex_pid, alias_mask_texture_pid],
                -1,
                &store,
                lookup(4),
            ),
            ResolvedTextureBinding::Texture2D { asset_id: 123 }
        );
    }

    #[test]
    fn resolved_texture_binding_prefers_exact_property_id_over_alias() {
        let mut store = MaterialPropertyStore::new();
        let exact_tex_pid = 20;
        let alias_texture_pid = 21;
        store.set_material(5, exact_tex_pid, MaterialPropertyValue::Texture(200));
        store.set_material(5, alias_texture_pid, MaterialPropertyValue::Texture(201));

        assert_eq!(
            resolved_texture_binding_for_host(
                "_Tex",
                &[exact_tex_pid, alias_texture_pid],
                -1,
                &store,
                lookup(5),
            ),
            ResolvedTextureBinding::Texture2D { asset_id: 200 }
        );
    }

    #[test]
    fn resolved_texture_binding_accepts_video_texture_property() {
        let mut store = MaterialPropertyStore::new();
        let main_tex_pid = 30;
        store.set_material(
            8,
            main_tex_pid,
            MaterialPropertyValue::Texture(pack_host_texture(
                44,
                HostTextureAssetKind::VideoTexture,
            )),
        );

        assert_eq!(
            resolved_texture_binding_for_host("_MainTex", &[main_tex_pid], -1, &store, lookup(8)),
            ResolvedTextureBinding::VideoTexture { asset_id: 44 }
        );
    }

    #[test]
    fn primary_texture_fallback_strips_naga_oil_suffix() {
        assert!(should_fallback_to_primary_texture(
            "_MainTexX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ4GSZLYMU5DU5DPN5XDEX"
        ));
    }

    #[test]
    fn primary_texture_ignores_later_non_primary_maps() {
        let (reflected, ids, registry) =
            reflected_with_textures(&[(1, "_MainTex"), (9, "_OcclusionMap")]);
        let mut store = MaterialPropertyStore::new();
        let occlusion = registry.intern("_OcclusionMap");
        store.set_material(6, occlusion, MaterialPropertyValue::Texture(77));

        assert_eq!(
            primary_texture_2d_asset_id(&reflected, &ids, &store, lookup(6)),
            -1
        );
        assert_eq!(
            resolved_texture_binding_for_host(
                "_MainTex",
                texture_property_ids_for_binding(&ids, 1),
                primary_texture_2d_asset_id(&reflected, &ids, &store, lookup(6)),
                &store,
                lookup(6),
            ),
            ResolvedTextureBinding::None
        );

        let main = registry.intern("_MainTex");
        store.set_material(6, main, MaterialPropertyValue::Texture(88));
        assert_eq!(
            primary_texture_2d_asset_id(&reflected, &ids, &store, lookup(6)),
            88
        );
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
