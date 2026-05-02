//! Uniform byte packing for embedded `@group(1)` material blocks (reflection-driven defaults and keywords).

use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::materials::{ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind};

use super::layout::StemEmbeddedPropertyIds;
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::{
    ResolvedTextureBinding, resolved_texture_binding_for_host, texture_property_ids_for_binding,
};

mod helpers;
mod tables;

use helpers::{default_f32_for_field, default_vec4_for_field, shader_writer_unescaped_field_name};
use tables::inferred_keyword_float_f32;

/// Suffix convention that opts a uniform field in to host `mipmap_bias` population.
const LOD_BIAS_SUFFIX: &str = "_LodBias";
/// Suffix convention that opts a uniform field in to storage V-inversion population.
const STORAGE_V_INVERTED_SUFFIX: &str = "_StorageVInverted";

/// Converts one sRGB channel to linear for color-like vec4 uniforms.
///
/// Values above `1.0` are treated as already-linear HDR and pass through unchanged.
fn srgb_channel_to_linear(v: f32) -> f32 {
    if v > 1.0 {
        return v;
    }
    if v <= 0.04045 {
        return v / 12.92;
    }
    ((v + 0.055) / 1.055).powf(2.4)
}

/// Applies sRGB->linear conversion to the RGB channels while keeping alpha unchanged.
fn srgb_vec4_rgb_to_linear(v: [f32; 4]) -> [f32; 4] {
    [
        srgb_channel_to_linear(v[0]),
        srgb_channel_to_linear(v[1]),
        srgb_channel_to_linear(v[2]),
        v[3],
    ]
}

/// Heuristic for vec4 uniforms that represent display-authored colors/tints.
fn should_linearize_vec4_uniform(field_name: &str) -> bool {
    let unescaped = shader_writer_unescaped_field_name(field_name)
        .strip_prefix('_')
        .unwrap_or(shader_writer_unescaped_field_name(field_name));
    let lower = unescaped.to_ascii_lowercase();
    lower.contains("color") || lower.contains("tint") || lower.ends_with("rim")
}

fn write_f32_at(buf: &mut [u8], field: &ReflectedUniformField, v: f32) {
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
/// Every value comes from one of six sources, in priority order: texture storage-orientation
/// flags for fields following the [`STORAGE_V_INVERTED_SUFFIX`] convention, host-sourced sampler
/// state for fields following the [`LOD_BIAS_SUFFIX`] convention (`_<Tex>_LodBias`), the host's
/// property store (for host-declared properties), explicit-only zero-default fields such as UI
/// text controls, [`inferred_keyword_float_f32`] for multi-compile keyword fields (`_NORMALMAP`,
/// `_ALPHATEST_ON`, …) the host cannot write because FrooxEngine routes them through the
/// `ShaderKeywords.Variant` bitmask the renderer never receives, or the scalar/vector default
/// tables / a zero for the unobservable pre-first-batch window.
pub(crate) fn build_embedded_uniform_bytes(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
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
                        default_vec4_for_field(shader_writer_unescaped_field_name(field_name))
                    };
                if should_linearize_vec4_uniform(field_name) {
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
                } else if let Some(default_value) = default_f32_for_field(field_name) {
                    default_value
                } else if explicit_zero_default_f32_field(field_name) {
                    0.0
                } else {
                    inferred_keyword_float_f32(
                        shader_writer_unescaped_field_name(field_name),
                        store,
                        lookup,
                        ids,
                    )
                    .unwrap_or(0.0)
                };
                write_f32_at(&mut buf, field, v);
            }
            ReflectedUniformScalarKind::U32 => {}
            ReflectedUniformScalarKind::Unsupported => {
                if let Some(MaterialPropertyValue::Float4Array(values)) =
                    store.get_merged(lookup, pid)
                {
                    write_f32x4_array_at(&mut buf, field, values);
                }
            }
        }
    }

    Some(buf)
}

/// Returns whether a scalar field must use only its canonical host property and otherwise default to zero.
fn explicit_zero_default_f32_field(field_name: &str) -> bool {
    matches!(
        shader_writer_unescaped_field_name(field_name),
        "_TextMode" | "_RectClip" | "_OVERLAY"
    )
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
    let (&binding, host_name) = reflected
        .material_group1_names
        .iter()
        .find(|(_, name)| name.as_str() == tex_name)?;
    let tex_pids = texture_property_ids_for_binding(ids, binding);
    if tex_pids.is_empty() {
        return Some(ResolvedTextureBinding::None);
    }
    Some(resolved_texture_binding_for_host(
        host_name.as_str(),
        tex_pids,
        tex_ctx.primary_texture_2d,
        store,
        lookup,
    ))
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
