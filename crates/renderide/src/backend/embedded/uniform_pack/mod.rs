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
    use std::sync::Arc;

    use hashbrown::HashMap;

    use super::tables::inferred_keyword_float_f32;
    use super::*;
    use crate::assets::material::PropertyIdRegistry;
    use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};
    use crate::backend::embedded::layout::StemEmbeddedPropertyIds;
    use crate::materials::{ReflectedMaterialUniformBlock, ReflectedUniformScalarKind};
    use crate::resources::{CubemapPool, RenderTexturePool, Texture3dPool, TexturePool};

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    /// Builds an empty texture-pool set for uniform-packer tests that only need binding metadata.
    fn empty_texture_pools() -> (TexturePool, Texture3dPool, CubemapPool, RenderTexturePool) {
        (
            TexturePool::default_pool(),
            Texture3dPool::default_pool(),
            CubemapPool::default_pool(),
            RenderTexturePool::new(),
        )
    }

    /// Extracts a packed f32x4 uniform from `bytes`.
    fn read_f32x4(bytes: &[u8], offset: usize) -> [f32; 4] {
        let mut out = [0.0; 4];
        for (i, value) in out.iter_mut().enumerate() {
            let start = offset + i * 4;
            *value = f32::from_le_bytes(
                bytes[start..start + 4]
                    .try_into()
                    .expect("uniform f32 component bytes"),
            );
        }
        out
    }

    /// Packs an asset id as a host render-texture material property.
    fn packed_render_texture(asset_id: i32) -> i32 {
        use crate::assets::texture::HostTextureAssetKind;

        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift)
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

    /// `MaterialRenderType::TransparentCutout` (1) on the wire enables the alpha-test keyword
    /// family even when the host never sends `_Mode` / `_BlendMode` (the FrooxEngine path).
    #[test]
    fn transparent_cutout_render_type_infers_alpha_test_family() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        store.set_material(7, render_type_pid, MaterialPropertyValue::Float(1.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(7), &ids),
                Some(1.0),
                "{field_name} should enable for TransparentCutout render type"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(7), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(7), &ids),
            Some(0.0)
        );
    }

    /// `MaterialRenderType::Opaque` (0) — neither alpha-test nor alpha-blend keyword fires.
    /// This is the case that previously bit Unlit: default `_Cutoff = 0.98` lit up the
    /// `_Cutoff ∈ (0, 1)` heuristic even though the host had selected Opaque.
    #[test]
    fn opaque_render_type_disables_all_alpha_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        store.set_material(8, render_type_pid, MaterialPropertyValue::Float(0.0));

        for field_name in [
            "_ALPHATEST_ON",
            "_ALPHATEST",
            "_ALPHACLIP",
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
        ] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(8), &ids),
                Some(0.0),
                "{field_name} should be disabled for Opaque render type"
            );
        }
    }

    /// `MaterialRenderType::Transparent` (2) with FrooxEngine `BlendMode.Alpha` factors
    /// (`_SrcBlend = SrcAlpha (5)`, `_DstBlend = OneMinusSrcAlpha (10)`) maps to
    /// `_ALPHABLEND_ON`, not `_ALPHAPREMULTIPLY_ON`.
    #[test]
    fn transparent_render_type_with_alpha_factors_infers_alpha_blend() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(9, render_type_pid, MaterialPropertyValue::Float(2.0));
        store.set_material(9, src_blend_pid, MaterialPropertyValue::Float(5.0));
        store.set_material(9, dst_blend_pid, MaterialPropertyValue::Float(10.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(9), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(9), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHATEST_ON", &store, lookup(9), &ids),
            Some(0.0)
        );
    }

    /// `MaterialRenderType::Transparent` (2) with FrooxEngine `BlendMode.Transparent`
    /// (premultiplied) factors `_SrcBlend = One (1)`, `_DstBlend = OneMinusSrcAlpha (10)`
    /// maps to `_ALPHAPREMULTIPLY_ON`, not `_ALPHABLEND_ON`.
    #[test]
    fn transparent_render_type_with_premultiplied_factors_infers_premultiply() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_type_pid = reg.intern("_RenderType");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(11, render_type_pid, MaterialPropertyValue::Float(2.0));
        store.set_material(11, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(11, dst_blend_pid, MaterialPropertyValue::Float(10.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(11), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(11), &ids),
            Some(0.0)
        );
    }

    /// PBS materials (`PBS_DualSidedMaterial.cs` and friends) bypass `SetBlendMode` and
    /// only signal `AlphaHandling.AlphaClip` by writing render queue 2450 plus the
    /// `_ALPHACLIP` shader keyword (which is not on the wire). Queue 2450 alone must
    /// enable the alpha-test family.
    #[test]
    fn render_queue_alpha_test_range_enables_alpha_test_family() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        store.set_material(20, render_queue_pid, MaterialPropertyValue::Float(2450.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(20), &ids),
                Some(1.0),
                "{field_name} should enable for queue 2450 (AlphaTest range)"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(20), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(20), &ids),
            Some(0.0)
        );
    }

    /// Queue 2000 (Geometry / Opaque) must leave every alpha keyword off — this is the
    /// PBS `AlphaHandling.Opaque` default.
    #[test]
    fn render_queue_opaque_range_disables_all_alpha_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        store.set_material(21, render_queue_pid, MaterialPropertyValue::Float(2000.0));

        for field_name in [
            "_ALPHATEST_ON",
            "_ALPHATEST",
            "_ALPHACLIP",
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
        ] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(21), &ids),
                Some(0.0),
                "{field_name} should be disabled for queue 2000 (Opaque range)"
            );
        }
    }

    /// Queue 3000 (Transparent) without premultiplied blend factors enables `_ALPHABLEND_ON`.
    #[test]
    fn render_queue_transparent_range_enables_alpha_blend() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        store.set_material(22, render_queue_pid, MaterialPropertyValue::Float(3000.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(22), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(22), &ids),
            Some(0.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHATEST_ON", &store, lookup(22), &ids),
            Some(0.0)
        );
    }

    /// Queue 3000 (Transparent) with premultiplied factors `_SrcBlend = 1`,
    /// `_DstBlend = 10` is `BlendMode.Transparent` — enables `_ALPHAPREMULTIPLY_ON`.
    #[test]
    fn render_queue_transparent_with_premultiplied_factors_infers_premultiply() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let render_queue_pid = reg.intern("_RenderQueue");
        let src_blend_pid = reg.intern("_SrcBlend");
        let dst_blend_pid = reg.intern("_DstBlend");
        store.set_material(23, render_queue_pid, MaterialPropertyValue::Float(3000.0));
        store.set_material(23, src_blend_pid, MaterialPropertyValue::Float(1.0));
        store.set_material(23, dst_blend_pid, MaterialPropertyValue::Float(10.0));

        assert_eq!(
            inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(23), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(23), &ids),
            Some(0.0)
        );
    }

    /// Render-texture bindings must not rewrite Unity `_ST` values behind the shader's back.
    #[test]
    fn render_texture_binding_leaves_st_uniform_unchanged() {
        let mut fields = HashMap::new();
        fields.insert(
            "_MainTex_ST".to_string(),
            ReflectedUniformField {
                offset: 0,
                size: 16,
                kind: ReflectedUniformScalarKind::Vec4,
            },
        );
        let mut material_group1_names = HashMap::new();
        material_group1_names.insert(1, "_MainTex".to_string());
        let reflected = ReflectedRasterLayout {
            layout_fingerprint: 0,
            material_entries: Vec::new(),
            per_draw_entries: Vec::new(),
            material_uniform: Some(ReflectedMaterialUniformBlock {
                binding: 0,
                total_size: 16,
                fields,
            }),
            material_group1_names,
            vs_max_vertex_location: None,
            requires_intersection_pass: false,
            requires_grab_pass: false,
        };

        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let mut ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let main_tex_st = reg.intern("_MainTex_ST");
        let main_tex = reg.intern("_MainTex");
        ids.uniform_field_ids
            .insert("_MainTex_ST".to_string(), main_tex_st);
        ids.texture_binding_property_ids
            .insert(1, Arc::from(vec![main_tex].into_boxed_slice()));
        store.set_material(
            24,
            main_tex,
            MaterialPropertyValue::Texture(packed_render_texture(9)),
        );
        store.set_material(
            24,
            main_tex_st,
            MaterialPropertyValue::Float4([2.0, 3.0, 0.25, 0.75]),
        );

        let (texture, texture3d, cubemap, render_texture) = empty_texture_pools();
        let pools = EmbeddedTexturePools {
            texture: &texture,
            texture3d: &texture3d,
            cubemap: &cubemap,
            render_texture: &render_texture,
        };
        let tex_ctx = UniformPackTextureContext {
            pools: &pools,
            primary_texture_2d: -1,
        };

        let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(24), &tex_ctx)
            .expect("uniform bytes");

        assert_eq!(read_f32x4(&bytes, 0), [2.0, 3.0, 0.25, 0.75]);
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
