//! Uniform byte packing for embedded `@group(1)` material blocks (reflection-driven defaults
//! plus renderer-reserved fields).

use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::materials::{ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind};

use crate::embedded_shaders::{EmbeddedMaterialDefaultKind, EmbeddedMaterialDefaultValue};

use super::layout::StemEmbeddedPropertyIds;
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::{
    ResolvedTextureBinding, resolved_texture_binding_for_host, texture_property_ids_for_binding,
};
use super::wrap_mode_bits::{WRAP_MODE_BITS_SUFFIX, sampler_wrap_mode_bits};

mod color_space;
mod helpers;
mod tables;

pub(crate) use crate::color_space::srgb_f32x4_rgb_to_linear as srgb_vec4_rgb_to_linear;
pub(crate) use color_space::MaterialUniformValueSpaces;
use helpers::shader_writer_unescaped_field_name;
use tables::inferred_shader_variant_bits_u32;

/// Material uniform fallback values keyed by reflected WGSL field name.
pub(crate) type MaterialUniformDefaults = hashbrown::HashMap<String, EmbeddedMaterialDefaultValue>;

/// Source metadata used while packing material uniform bytes.
pub(crate) struct MaterialUniformPackMetadata<'a> {
    /// Per-field color-space metadata for reflected material uniforms.
    pub(crate) value_spaces: &'a MaterialUniformValueSpaces,
    /// Per-field fallback values declared by shader source metadata.
    pub(crate) material_defaults: &'a MaterialUniformDefaults,
}

/// Suffix convention that opts a uniform field in to host `mipmap_bias` population.
const LOD_BIAS_SUFFIX: &str = "_LodBias";
/// Suffix convention that opts a uniform field in to storage V-inversion population.
const STORAGE_V_INVERTED_SUFFIX: &str = "_StorageVInverted";
/// Suffix convention Unity uses for texture scale/offset uniforms.
const TEXTURE_TRANSFORM_SUFFIX: &str = "_ST";
/// Unity's implicit identity scale/offset value for unwritten texture transform uniforms.
const UNITY_TEXTURE_TRANSFORM_IDENTITY: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

fn write_f32_at(buf: &mut [u8], field: &ReflectedUniformField, v: f32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_u32_at(buf: &mut [u8], field: &ReflectedUniformField, v: u32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_f32x4_at(buf: &mut [u8], field: &ReflectedUniformField, v: &[f32; 4]) {
    let off = field.offset as usize;
    if off + 16 <= buf.len() && field.size >= 16 {
        for (i, c) in v.iter().enumerate() {
            let o = off + i * 4;
            buf[o..o + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
}

/// Writes a host `float4[]` material property into a reflected uniform array field.
fn write_f32x4_array_at(buf: &mut [u8], field: &ReflectedUniformField, values: &[[f32; 4]]) {
    let off = field.offset as usize;
    let max_values = (field.size as usize) / 16;
    for (i, value) in values.iter().take(max_values).enumerate() {
        let elem_off = off + i * 16;
        if elem_off + 16 > buf.len() {
            return;
        }
        for (component, v) in value.iter().enumerate() {
            let component_off = elem_off + component * 4;
            buf[component_off..component_off + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Writes a host `float[]` material property into a reflected uniform scalar array field.
fn write_f32_array_at(buf: &mut [u8], field: &ReflectedUniformField, values: &[f32]) {
    let off = field.offset as usize;
    let max_values = (field.size as usize) / 16;
    for (i, value) in values.iter().take(max_values).enumerate() {
        let elem_off = off + i * 16;
        if elem_off + 4 > buf.len() {
            return;
        }
        buf[elem_off..elem_off + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_srgb_f32x4_array_at(buf: &mut [u8], field: &ReflectedUniformField, values: &[[f32; 4]]) {
    let off = field.offset as usize;
    let max_values = (field.size as usize) / 16;
    for (i, value) in values.iter().take(max_values).enumerate() {
        let elem_off = off + i * 16;
        if elem_off + 16 > buf.len() {
            return;
        }
        for (component, v) in srgb_vec4_rgb_to_linear(*value).iter().enumerate() {
            let component_off = elem_off + component * 4;
            buf[component_off..component_off + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Returns the reflected `vec4<f32>` fallback value for a missing host property.
fn missing_vec4_uniform_default(
    field_name: &str,
    material_defaults: &MaterialUniformDefaults,
) -> [f32; 4] {
    if let Some(default) = material_defaults.get(field_name)
        && default.kind == EmbeddedMaterialDefaultKind::Vec4
    {
        return default.values;
    }
    if shader_writer_unescaped_field_name(field_name).ends_with(TEXTURE_TRANSFORM_SUFFIX) {
        UNITY_TEXTURE_TRANSFORM_IDENTITY
    } else {
        [0.0; 4]
    }
}

/// Returns the reflected `f32` fallback value for a missing host property.
fn missing_f32_uniform_default(
    field_name: &str,
    material_defaults: &MaterialUniformDefaults,
) -> f32 {
    if let Some(default) = material_defaults.get(field_name)
        && default.kind == EmbeddedMaterialDefaultKind::Float
    {
        default.values[0]
    } else {
        0.0
    }
}

/// Auxiliary inputs required to populate texture-sourced uniform fields.
///
/// Threads resident texture pools into the packer so f32 fields following texture suffix
/// conventions can resolve their bound texture and read sampler/orientation metadata.
pub(crate) struct UniformPackTextureContext<'a> {
    /// Resident texture pools (2D / 3D / cubemap / render-texture).
    pub pools: &'a EmbeddedTexturePools<'a>,
    /// Primary 2D texture asset id for `_MainTex` / `_Tex` fallback (from [`crate::materials::embedded::texture_resolve::primary_texture_2d_asset_id`]).
    pub primary_texture_2d: i32,
}

/// Builds CPU bytes for the reflected material uniform block.
///
/// Every value comes from one of several sources, in priority order: texture storage-orientation
/// flags for fields following the [`STORAGE_V_INVERTED_SUFFIX`] convention, host-sourced sampler
/// state for fields following the [`LOD_BIAS_SUFFIX`] convention (`_<Tex>_LodBias`), the host's
/// property store (for host-declared properties), source-declared material defaults, Unity's
/// identity texture transform for missing `*_ST` vec4 fields, texture wrap bits for fields
/// following the [`WRAP_MODE_BITS_SUFFIX`] convention, or the renderer-reserved
/// `_RenderideVariantBits` variant bitfield. Other missing fields fall through to zero.
#[cfg(test)]
pub(crate) fn build_embedded_uniform_bytes(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> Option<Vec<u8>> {
    build_embedded_uniform_bytes_with_value_spaces(
        reflected,
        ids,
        &MaterialUniformValueSpaces::default(),
        store,
        lookup,
        tex_ctx,
        None,
    )
}

/// Builds CPU bytes for the reflected material uniform block using explicit per-field value-space metadata.
#[cfg(test)]
pub(crate) fn build_embedded_uniform_bytes_with_value_spaces(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    value_spaces: &MaterialUniformValueSpaces,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
    shader_variant_bits: Option<u32>,
) -> Option<Vec<u8>> {
    build_embedded_uniform_bytes_with_material_defaults(
        reflected,
        ids,
        &MaterialUniformPackMetadata {
            value_spaces,
            material_defaults: &MaterialUniformDefaults::default(),
        },
        store,
        lookup,
        tex_ctx,
        shader_variant_bits,
    )
}

/// Builds CPU bytes for the reflected material uniform block using explicit value-space and default metadata.
pub(crate) fn build_embedded_uniform_bytes_with_material_defaults(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    metadata: &MaterialUniformPackMetadata<'_>,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
    shader_variant_bits: Option<u32>,
) -> Option<Vec<u8>> {
    profiling::scope!("materials::embedded_uniform_pack");
    let u = reflected.material_uniform.as_ref()?;
    let mut buf = vec![0u8; u.total_size as usize];

    for (field_name, field) in &u.fields {
        let pid = *ids.uniform_field_ids.get(field_name)?;
        match field.kind {
            ReflectedUniformScalarKind::Vec4 => {
                let mut v =
                    if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, pid) {
                        *c
                    } else {
                        missing_vec4_uniform_default(field_name, metadata.material_defaults)
                    };
                if metadata.value_spaces.is_srgb_vec4(field_name) {
                    v = srgb_vec4_rgb_to_linear(v);
                }
                write_f32x4_at(&mut buf, field, &v);
            }
            ReflectedUniformScalarKind::F32 => {
                let v = if let Some(storage_v_inverted) =
                    storage_v_inverted_for_field(field_name, reflected, ids, store, lookup, tex_ctx)
                {
                    storage_v_inverted
                } else if let Some(bias) =
                    lod_bias_for_field(field_name, reflected, ids, store, lookup, tex_ctx)
                {
                    bias
                } else if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, pid)
                {
                    *f
                } else {
                    missing_f32_uniform_default(field_name, metadata.material_defaults)
                };
                write_f32_at(&mut buf, field, v);
            }
            ReflectedUniformScalarKind::U32 => {
                let v = if let Some(wrap_mode_bits) =
                    wrap_mode_bits_for_field(field_name, reflected, ids, store, lookup, tex_ctx)
                {
                    wrap_mode_bits
                } else {
                    inferred_shader_variant_bits_u32(
                        shader_writer_unescaped_field_name(field_name),
                        shader_variant_bits,
                        store,
                        lookup,
                        ids,
                    )
                    .unwrap_or(0)
                };
                write_u32_at(&mut buf, field, v);
            }
            ReflectedUniformScalarKind::Unsupported => {
                if let Some(MaterialPropertyValue::FloatArray(values)) =
                    store.get_merged(lookup, pid)
                {
                    write_f32_array_at(&mut buf, field, values);
                } else if let Some(MaterialPropertyValue::Float4Array(values)) =
                    store.get_merged(lookup, pid)
                {
                    if metadata.value_spaces.is_srgb_vec4_array(field_name) {
                        write_srgb_f32x4_array_at(&mut buf, field, values);
                    } else {
                        write_f32x4_array_at(&mut buf, field, values);
                    }
                }
            }
        }
    }

    Some(buf)
}

/// Resolves the texture binding for a reflected group-1 texture name.
fn resolved_texture_binding_for_texture_name(
    texture_name: &str,
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    primary_texture_2d: i32,
) -> Option<ResolvedTextureBinding> {
    let (&binding, host_name) = reflected
        .material_group1_names
        .iter()
        .find(|(_, name)| name.as_str() == texture_name)?;
    let tex_pids = texture_property_ids_for_binding(ids, binding);
    if tex_pids.is_empty() {
        return Some(ResolvedTextureBinding::None);
    }
    Some(resolved_texture_binding_for_host(
        host_name.as_str(),
        tex_pids,
        primary_texture_2d,
        store,
        lookup,
    ))
}

/// Resolves the texture binding associated with a field following a texture-name suffix convention.
fn resolved_texture_binding_for_field_suffix(
    field_name: &str,
    suffix: &str,
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> Option<ResolvedTextureBinding> {
    let unescaped = shader_writer_unescaped_field_name(field_name);
    let tex_name = unescaped.strip_suffix(suffix)?;
    resolved_texture_binding_for_texture_name(
        tex_name,
        reflected,
        ids,
        store,
        lookup,
        tex_ctx.primary_texture_2d,
    )
}

/// Returns whether a resolved texture binding is a host-uploaded texture with V-inverted storage.
fn binding_storage_v_inverted_from_metadata(
    resolved: ResolvedTextureBinding,
    texture2d_storage_v_inverted: Option<bool>,
    cubemap_storage_v_inverted: Option<bool>,
) -> bool {
    match resolved {
        ResolvedTextureBinding::Texture2D { .. } => texture2d_storage_v_inverted.unwrap_or(false),
        ResolvedTextureBinding::Cubemap { .. } => cubemap_storage_v_inverted.unwrap_or(false),
        ResolvedTextureBinding::None
        | ResolvedTextureBinding::Texture3D { .. }
        | ResolvedTextureBinding::RenderTexture { .. }
        | ResolvedTextureBinding::VideoTexture { .. } => false,
    }
}

/// Returns whether a resolved texture binding is a host-uploaded texture with V-inverted storage.
fn binding_storage_v_inverted(
    resolved: ResolvedTextureBinding,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> bool {
    let texture2d_storage_v_inverted = match resolved {
        ResolvedTextureBinding::Texture2D { asset_id } => tex_ctx
            .pools
            .texture
            .get(asset_id)
            .map(|t| t.storage_v_inverted),
        _ => None,
    };
    let cubemap_storage_v_inverted = match resolved {
        ResolvedTextureBinding::Cubemap { asset_id } => tex_ctx
            .pools
            .cubemap
            .get(asset_id)
            .map(|t| t.storage_v_inverted),
        _ => None,
    };
    binding_storage_v_inverted_from_metadata(
        resolved,
        texture2d_storage_v_inverted,
        cubemap_storage_v_inverted,
    )
}

/// Converts a storage V-inversion flag into the f32 convention used by explicit shader uniforms.
fn storage_v_inverted_flag_value(storage_v_inverted: bool) -> f32 {
    if storage_v_inverted { 1.0 } else { 0.0 }
}

/// Host storage-orientation flag for `_<Tex>_StorageVInverted` fields.
fn storage_v_inverted_for_field(
    field_name: &str,
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> Option<f32> {
    let resolved = resolved_texture_binding_for_field_suffix(
        field_name,
        STORAGE_V_INVERTED_SUFFIX,
        reflected,
        ids,
        store,
        lookup,
        tex_ctx,
    )?;
    Some(storage_v_inverted_flag_value(binding_storage_v_inverted(
        resolved, tex_ctx,
    )))
}

/// Returns texture wrap bits for texture kinds whose sampler state exposes wrap modes.
fn binding_wrap_mode_bits_from_metadata(
    resolved: ResolvedTextureBinding,
    texture2d_wrap_mode_bits: Option<u32>,
    texture3d_wrap_mode_bits: Option<u32>,
    render_texture_wrap_mode_bits: Option<u32>,
    video_texture_wrap_mode_bits: Option<u32>,
) -> u32 {
    match resolved {
        ResolvedTextureBinding::Texture2D { .. } => texture2d_wrap_mode_bits.unwrap_or(0),
        ResolvedTextureBinding::Texture3D { .. } => texture3d_wrap_mode_bits.unwrap_or(0),
        ResolvedTextureBinding::RenderTexture { .. } => render_texture_wrap_mode_bits.unwrap_or(0),
        ResolvedTextureBinding::VideoTexture { .. } => video_texture_wrap_mode_bits.unwrap_or(0),
        ResolvedTextureBinding::None | ResolvedTextureBinding::Cubemap { .. } => 0,
    }
}

/// Returns texture wrap bits for a resolved binding from the resident texture pools.
fn binding_wrap_mode_bits(
    resolved: ResolvedTextureBinding,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> u32 {
    let texture2d_wrap_mode_bits = match resolved {
        ResolvedTextureBinding::Texture2D { asset_id } => tex_ctx
            .pools
            .texture
            .get(asset_id)
            .map(|t| sampler_wrap_mode_bits(&t.sampler)),
        _ => None,
    };
    let texture3d_wrap_mode_bits = match resolved {
        ResolvedTextureBinding::Texture3D { asset_id } => tex_ctx
            .pools
            .texture3d
            .get(asset_id)
            .map(|t| sampler_wrap_mode_bits(&t.sampler)),
        _ => None,
    };
    let render_texture_wrap_mode_bits = match resolved {
        ResolvedTextureBinding::RenderTexture { asset_id } => tex_ctx
            .pools
            .render_texture
            .get(asset_id)
            .map(|t| sampler_wrap_mode_bits(&t.sampler)),
        _ => None,
    };
    let video_texture_wrap_mode_bits = match resolved {
        ResolvedTextureBinding::VideoTexture { asset_id } => tex_ctx
            .pools
            .video_texture
            .get(asset_id)
            .map(|t| sampler_wrap_mode_bits(&t.sampler)),
        _ => None,
    };
    binding_wrap_mode_bits_from_metadata(
        resolved,
        texture2d_wrap_mode_bits,
        texture3d_wrap_mode_bits,
        render_texture_wrap_mode_bits,
        video_texture_wrap_mode_bits,
    )
}

/// Host wrap bits for `_<Tex>_WrapModeBits` fields, or [`None`] if `field_name` is not a
/// wrap-bit field.
fn wrap_mode_bits_for_field(
    field_name: &str,
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> Option<u32> {
    let resolved = resolved_texture_binding_for_field_suffix(
        field_name,
        WRAP_MODE_BITS_SUFFIX,
        reflected,
        ids,
        store,
        lookup,
        tex_ctx,
    )?;
    Some(binding_wrap_mode_bits(resolved, tex_ctx))
}

/// Returns the shader LOD bias for texture kinds whose wire properties expose mip bias.
fn binding_lod_bias_from_metadata(
    resolved: ResolvedTextureBinding,
    texture2d_mipmap_bias: Option<f32>,
    cubemap_mipmap_bias: Option<f32>,
) -> f32 {
    match resolved {
        ResolvedTextureBinding::Texture2D { .. } => texture2d_mipmap_bias.unwrap_or(0.0),
        ResolvedTextureBinding::Cubemap { .. } => cubemap_mipmap_bias.unwrap_or(0.0),
        ResolvedTextureBinding::None
        | ResolvedTextureBinding::Texture3D { .. }
        | ResolvedTextureBinding::RenderTexture { .. }
        | ResolvedTextureBinding::VideoTexture { .. } => 0.0,
    }
}

/// Returns the shader LOD bias for a resolved binding from the resident texture pools.
fn binding_lod_bias(
    resolved: ResolvedTextureBinding,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> f32 {
    let texture2d_mipmap_bias = match resolved {
        ResolvedTextureBinding::Texture2D { asset_id } => tex_ctx
            .pools
            .texture
            .get(asset_id)
            .map(|t| t.sampler.mipmap_bias),
        _ => None,
    };
    let cubemap_mipmap_bias = match resolved {
        ResolvedTextureBinding::Cubemap { asset_id } => tex_ctx
            .pools
            .cubemap
            .get(asset_id)
            .map(|t| t.sampler.mipmap_bias),
        _ => None,
    };
    binding_lod_bias_from_metadata(resolved, texture2d_mipmap_bias, cubemap_mipmap_bias)
}

/// Host `mipmap_bias` for `_<Tex>_LodBias` fields, or [`None`] if `field_name` is not a LOD-bias
/// field or no texture is currently bound to the matching `_<Tex>` slot.
///
/// Fields not following the convention fall through to the store / keyword / default path.
fn lod_bias_for_field(
    field_name: &str,
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> Option<f32> {
    let resolved = resolved_texture_binding_for_field_suffix(
        field_name,
        LOD_BIAS_SUFFIX,
        reflected,
        ids,
        store,
        lookup,
        tex_ctx,
    )?;
    Some(binding_lod_bias(resolved, tex_ctx))
}

#[cfg(test)]
mod tests;
