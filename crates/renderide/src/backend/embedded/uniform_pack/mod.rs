//! Uniform byte packing for embedded `@group(1)` material blocks (reflection-driven defaults and keywords).

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::materials::{ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind};

use super::layout::StemEmbeddedPropertyIds;
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::{
    resolved_texture_binding_for_host, texture_property_ids_for_binding, ResolvedTextureBinding,
};

mod helpers;
mod tables;

use helpers::{default_vec4_for_field, shader_writer_unescaped_field_name};
use tables::inferred_keyword_float_f32;

/// Suffix convention that opts a uniform field in to host `mipmap_bias` population.
const LOD_BIAS_SUFFIX: &str = "_LodBias";

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

/// Auxiliary inputs required to populate host-sourced uniform fields (currently `_<Tex>_LodBias`).
///
/// Threads resident texture pools into the packer so an f32 field following the
/// [`LOD_BIAS_SUFFIX`] convention can resolve its bound texture and read the host-supplied
/// [`crate::resources::Texture2dSamplerState::mipmap_bias`] (and cubemap / 3D analogues).
pub(crate) struct UniformPackTextureContext<'a> {
    /// Resident texture pools (2D / 3D / cubemap / render-texture).
    pub pools: &'a EmbeddedTexturePools<'a>,
    /// Primary 2D texture asset id for `_MainTex` / `_Tex` fallback (from [`crate::backend::embedded::texture_resolve::primary_texture_2d_asset_id`]).
    pub primary_texture_2d: i32,
}

/// Builds CPU bytes for the reflected material uniform block.
///
/// Every value comes from one of four sources, in priority order: host-sourced sampler state for
/// fields following the [`LOD_BIAS_SUFFIX`] convention (`_<Tex>_LodBias`), the host's property
/// store (for host-declared properties), [`inferred_keyword_float_f32`] for multi-compile keyword
/// fields (`_NORMALMAP`, `_ALPHATEST_ON`, …) the host cannot write because FrooxEngine routes
/// them through the `ShaderKeywords.Variant` bitmask the renderer never receives, or the
/// `default_vec4_for_field` table / a zero for the unobservable pre-first-batch window.
pub(crate) fn build_embedded_uniform_bytes(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    tex_ctx: &UniformPackTextureContext<'_>,
) -> Option<Vec<u8>> {
    let u = reflected.material_uniform.as_ref()?;
    let mut buf = vec![0u8; u.total_size as usize];

    for (field_name, field) in &u.fields {
        let pid = *ids.uniform_field_ids.get(field_name)?;
        match field.kind {
            ReflectedUniformScalarKind::Vec4 => {
                let v =
                    if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, pid) {
                        *c
                    } else {
                        default_vec4_for_field(shader_writer_unescaped_field_name(field_name))
                    };
                write_f32x4_at(&mut buf, field, &v);
            }
            ReflectedUniformScalarKind::F32 => {
                let v = if let Some(bias) =
                    lod_bias_for_field(field_name, reflected, ids, store, lookup, tex_ctx)
                {
                    bias
                } else if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, pid)
                {
                    *f
                } else if field_name == "_Cutoff" {
                    // Unity-convention cutoff fallback for the pre-first-batch window.
                    0.5
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
            ReflectedUniformScalarKind::U32 | ReflectedUniformScalarKind::Unsupported => {}
        }
    }

    Some(buf)
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
    let unescaped = shader_writer_unescaped_field_name(field_name);
    let tex_name = unescaped.strip_suffix(LOD_BIAS_SUFFIX)?;
    let (&binding, host_name) = reflected
        .material_group1_names
        .iter()
        .find(|(_, name)| name.as_str() == tex_name)?;
    let tex_pids = texture_property_ids_for_binding(ids, binding);
    if tex_pids.is_empty() {
        return Some(0.0);
    }
    let resolved = resolved_texture_binding_for_host(
        host_name.as_str(),
        tex_pids,
        tex_ctx.primary_texture_2d,
        store,
        lookup,
    );
    match resolved {
        ResolvedTextureBinding::Texture2D { asset_id } => tex_ctx
            .pools
            .texture
            .get_texture(asset_id)
            .map(|t| t.sampler.mipmap_bias)
            .or(Some(0.0)),
        ResolvedTextureBinding::Texture3D { asset_id } => tex_ctx
            .pools
            .texture3d
            .get_texture(asset_id)
            .map(|t| t.sampler.mipmap_bias)
            .or(Some(0.0)),
        ResolvedTextureBinding::Cubemap { asset_id } => tex_ctx
            .pools
            .cubemap
            .get_texture(asset_id)
            .map(|t| t.sampler.mipmap_bias)
            .or(Some(0.0)),
        ResolvedTextureBinding::None | ResolvedTextureBinding::RenderTexture { .. } => Some(0.0),
    }
}

#[cfg(test)]
mod text_uniform_packing_tests {
    use super::tables::inferred_keyword_float_f32;
    use super::*;
    use crate::assets::material::PropertyIdRegistry;
    use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};
    use crate::backend::embedded::layout::StemEmbeddedPropertyIds;

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    #[test]
    fn cutout_blend_mode_infers_alpha_clip_from_canonical_blend_mode() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_BlendMode");
        store.set_material(12, pid, MaterialPropertyValue::Float(1.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(12), &ids),
                Some(1.0),
                "{field_name} should enable for cutout _BlendMode"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(12), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn inferred_pbs_keyword_enables_from_texture_presence() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_SpecularMap");
        store.set_material(4, pid, MaterialPropertyValue::Texture(123));
        assert_eq!(
            inferred_keyword_float_f32("_SPECULARMAP", &store, lookup(4), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(4), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn vec4_defaults_match_documented_unity_conventions() {
        // Spot-check a few entries in the generic vec4 default table that DO need a non-zero
        // value because the relevant WGSL shaders rely on them prior to host writes.
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_SpecularColor"),
            [1.0, 1.0, 1.0, 0.5]
        );
        assert_eq!(default_vec4_for_field("_Rect"), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(default_vec4_for_field("_Point"), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(default_vec4_for_field("_OverlayTint"), [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(
            default_vec4_for_field("_BehindFarColor"),
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(default_vec4_for_field("_Tint0_"), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn only_main_texture_bindings_fallback_to_primary_texture() {
        use crate::backend::embedded::texture_resolve::should_fallback_to_primary_texture;
        assert!(should_fallback_to_primary_texture("_MainTex"));
        assert!(!should_fallback_to_primary_texture("_MainTex1"));
        assert!(!should_fallback_to_primary_texture("_SpecularMap"));
    }

    /// `_ALBEDOTEX` keyword inference must treat a packed [`HostTextureAssetKind::RenderTexture`] like a
    /// bound texture (parity with 2D-only `texture_property_asset_id_by_pid`).
    #[test]
    fn albedo_keyword_infers_from_render_texture_packed_id() {
        use crate::assets::texture::{unpack_host_texture_packed, HostTextureAssetKind};

        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let main_tex = reg.intern("_MainTex");
        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        let asset_id = 7i32;
        let packed = asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift);
        assert_eq!(
            unpack_host_texture_packed(packed),
            Some((asset_id, HostTextureAssetKind::RenderTexture))
        );
        store.set_material(6, main_tex, MaterialPropertyValue::Texture(packed));
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(6), &ids),
            Some(1.0)
        );
    }
}
