//! Embedded-material keyword inference tests.

use super::super::tables::inferred_keyword_float_f32;
use super::super::*;
use super::common::*;

use std::sync::Arc;

use hashbrown::HashMap;

use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{
    MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::{ReflectedMaterialUniformBlock, ReflectedUniformScalarKind};
use crate::shared::ColorProfile;

#[test]
fn unlit_texture_presence_infers_observable_keywords() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        50,
        reg.intern("_Tex"),
        MaterialPropertyValue::Texture(packed_render_texture(1)),
    );
    store.set_material(
        50,
        reg.intern("_MaskTex"),
        MaterialPropertyValue::Texture(packed_render_texture(2)),
    );
    store.set_material(
        50,
        reg.intern("_OffsetTex"),
        MaterialPropertyValue::Texture(packed_render_texture(3)),
    );

    assert_eq!(
        inferred_keyword_float_f32("_TEXTURE", &store, lookup(50), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_MASK_TEXTURE_MUL", &store, lookup(50), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_MASK_TEXTURE_CLIP", &store, lookup(50), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_OFFSET_TEXTURE", &store, lookup(50), &ids),
        Some(1.0)
    );
}

#[test]
fn right_eye_keyword_infers_from_right_eye_st_presence() {
    let (_reflected, ids, registry) = reflected_with_f32_fields(&[("_RightEye_ST", 0)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        51,
        registry.intern("_RightEye_ST"),
        MaterialPropertyValue::Float4([0.5, 1.0, 0.5, 0.0]),
    );

    assert_eq!(
        inferred_keyword_float_f32("_RIGHT_EYE_ST", &store, lookup(51), &ids),
        Some(1.0)
    );
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

/// `MaterialRenderType::Opaque` (0) -- neither alpha-test nor alpha-blend keyword fires.
/// This is the case that previously bit Unlit: default `_Cutoff = 0.98` lit up the
/// `_Cutoff in (0, 1)` heuristic even though the host had selected Opaque.
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

/// `BlendMode.Additive` writes Transparent render type with `_SrcBlend = One` and
/// `_DstBlend = One`; Unlit uses that signal to enable `_MUL_RGB_BY_ALPHA`.
#[test]
fn transparent_render_type_with_additive_factors_infers_mul_rgb_by_alpha() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(13, render_type_pid, MaterialPropertyValue::Float(2.0));
    store.set_material(13, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(13, dst_blend_pid, MaterialPropertyValue::Float(1.0));

    assert_eq!(
        inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(13), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(13), &ids),
        Some(0.0)
    );
}

/// FrooxEngine drives the LUT `LERP` keyword from `_Lerp > 0` rather than sending a
/// standalone keyword material property.
#[test]
fn lut_lerp_keyword_infers_from_lerp_uniform() {
    let (_reflected, ids, reg) = reflected_with_f32_fields(&[("_Lerp", 0), ("LERP", 4)]);
    let mut store = MaterialPropertyStore::new();
    let lerp_pid = reg.intern("_Lerp");

    assert_eq!(
        inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
        Some(0.0)
    );

    store.set_material(24, lerp_pid, MaterialPropertyValue::Float(0.0));
    assert_eq!(
        inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
        Some(0.0)
    );

    store.set_material(24, lerp_pid, MaterialPropertyValue::Float(0.25));
    assert_eq!(
        inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
        Some(1.0)
    );
}

/// Base LUT materials default to the Unity/FrooxEngine `SRGB` variant unless the reflected
/// keyword field is explicitly supplied as false.
#[test]
fn lut_srgb_keyword_defaults_on() {
    let (_reflected, ids, reg) = reflected_with_f32_fields(&[("SRGB", 0)]);
    let mut store = MaterialPropertyStore::new();

    assert_eq!(
        inferred_keyword_float_f32("SRGB", &store, lookup(25), &ids),
        Some(1.0)
    );

    let srgb_pid = reg.intern("SRGB");
    store.set_material(25, srgb_pid, MaterialPropertyValue::Float(0.0));
    assert_eq!(
        inferred_keyword_float_f32("SRGB", &store, lookup(25), &ids),
        Some(0.0)
    );
}

/// Additive blend factors alone are not enough; the material must also be in a transparent
/// render type or queue range.
#[test]
fn opaque_render_type_with_additive_factors_does_not_infer_mul_rgb_by_alpha() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(14, render_type_pid, MaterialPropertyValue::Float(0.0));
    store.set_material(14, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(14, dst_blend_pid, MaterialPropertyValue::Float(1.0));

    assert_eq!(
        inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(14), &ids),
        Some(0.0)
    );
}

/// Render queue inference covers materials that signal transparency through queue state rather
/// than `MaterialRenderType`.
#[test]
fn render_queue_transparent_with_additive_factors_infers_mul_rgb_by_alpha() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_queue_pid = reg.intern("_RenderQueue");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(15, render_queue_pid, MaterialPropertyValue::Float(3000.0));
    store.set_material(15, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(15, dst_blend_pid, MaterialPropertyValue::Float(1.0));

    assert_eq!(
        inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(15), &ids),
        Some(1.0)
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

/// Queue 2000 (Geometry / Opaque) must leave every alpha keyword off -- this is the
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
/// `_DstBlend = 10` is `BlendMode.Transparent` -- enables `_ALPHAPREMULTIPLY_ON`.
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
        vs_vertex_inputs: Vec::new(),
        vs_max_vertex_location: None,
        uses_scene_depth_snapshot: false,
        uses_scene_color_snapshot: false,
        requires_intersection_pass: false,
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

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
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
fn explicit_ui_text_control_fields_pack_canonical_values() {
    let (reflected, ids, registry) =
        reflected_with_f32_fields(&[("_TextMode", 0), ("_RectClip", 4), ("_OVERLAY", 8)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        25,
        registry.intern("_TextMode"),
        MaterialPropertyValue::Float(2.0),
    );
    store.set_material(
        25,
        registry.intern("_RectClip"),
        MaterialPropertyValue::Float(1.0),
    );
    store.set_material(
        25,
        registry.intern("_OVERLAY"),
        MaterialPropertyValue::Float(1.0),
    );
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(25), &tex_ctx)
        .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 2.0);
    assert_eq!(read_f32_at(&bytes, 4), 1.0);
    assert_eq!(read_f32_at(&bytes, 8), 1.0);
}

#[test]
fn font_atlas_profile_metadata_infers_text_mode() {
    let binding = ResolvedTextureBinding::Texture2D { asset_id: 42 };

    assert_eq!(
        binding_text_mode_from_metadata(binding, Some(ColorProfile::Linear)),
        Some(0.0)
    );
    assert_eq!(
        binding_text_mode_from_metadata(binding, Some(ColorProfile::SRGB)),
        Some(1.0)
    );
    assert_eq!(
        binding_text_mode_from_metadata(binding, Some(ColorProfile::SRGBAlpha)),
        Some(1.0)
    );
    assert_eq!(binding_text_mode_from_metadata(binding, None), None);
    assert_eq!(
        binding_text_mode_from_metadata(
            ResolvedTextureBinding::RenderTexture { asset_id: 42 },
            Some(ColorProfile::SRGB)
        ),
        None
    );
}

#[test]
fn explicit_ui_text_control_fields_ignore_keyword_aliases() {
    let (reflected, ids, registry) =
        reflected_with_f32_fields(&[("_TextMode", 0), ("_RectClip", 4), ("_OVERLAY", 8)]);
    let mut store = MaterialPropertyStore::new();
    for property_name in [
        "TextMode", "textmode", "RectClip", "rectclip", "OVERLAY", "overlay",
    ] {
        store.set_material(
            26,
            registry.intern(property_name),
            MaterialPropertyValue::Float(1.0),
        );
    }
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(26), &tex_ctx)
        .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 0.0);
    assert_eq!(read_f32_at(&bytes, 4), 0.0);
    assert_eq!(read_f32_at(&bytes, 8), 0.0);
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
        inferred_keyword_float_f32("_SPECGLOSSMAP", &store, lookup(4), &ids),
        Some(0.0)
    );
    let spec_gloss_pid = reg.intern("_SpecGlossMap");
    store.set_material(4, spec_gloss_pid, MaterialPropertyValue::Texture(456));
    assert_eq!(
        inferred_keyword_float_f32("_SPECGLOSSMAP", &store, lookup(4), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(4), &ids),
        Some(0.0)
    );
}

#[test]
fn fresnel_texture_keyword_infers_from_far_or_near_textures() {
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);

    for (i, property_name) in [
        "_FarTex",
        "_NearTex",
        "_FarTex0",
        "_NearTex0",
        "_FarTex1",
        "_NearTex1",
    ]
    .iter()
    .enumerate()
    {
        let material_id = 50 + i as i32;
        let mut store = MaterialPropertyStore::new();
        store.set_material(
            material_id,
            reg.intern(property_name),
            MaterialPropertyValue::Texture(packed_texture2d(100 + i as i32)),
        );
        assert_eq!(
            inferred_keyword_float_f32("_TEXTURE", &store, lookup(material_id), &ids),
            Some(1.0),
            "{property_name} should enable _TEXTURE"
        );
    }
}

#[test]
fn inferred_pbs_splat_keywords_enable_from_texture_presence() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    for property_name in [
        "_PackedHeightMap",
        "_PackedNormalMap23",
        "_PackedEmissionMap",
        "_MetallicGloss23",
        "_SpecularMap3",
    ] {
        store.set_material(
            54,
            reg.intern(property_name),
            MaterialPropertyValue::Texture(packed_texture2d(123)),
        );
    }

    for field_name in [
        "_HEIGHTMAP",
        "_PACKED_NORMALMAP",
        "_PACKED_EMISSIONTEX",
        "_METALLICMAP",
        "_SPECULARMAP",
    ] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(54), &ids),
            Some(1.0),
            "{field_name} should infer from its selected texture family"
        );
    }
}

#[test]
fn gradient_keyword_infers_from_gradient_texture() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        60,
        reg.intern("_Gradient"),
        MaterialPropertyValue::Texture(packed_texture2d(14)),
    );

    assert_eq!(
        inferred_keyword_float_f32("GRADIENT", &store, lookup(60), &ids),
        Some(1.0)
    );
}

#[test]
fn normalmap_keyword_infers_from_normal_map_zero() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        61,
        reg.intern("_NormalMap0"),
        MaterialPropertyValue::Texture(packed_texture2d(15)),
    );

    assert_eq!(
        inferred_keyword_float_f32("_NORMALMAP", &store, lookup(61), &ids),
        Some(1.0)
    );
}

#[test]
fn clip_keyword_stays_off_from_clip_range_properties_only() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        62,
        reg.intern("_ClipMin"),
        MaterialPropertyValue::Float(0.0),
    );
    store.set_material(
        62,
        reg.intern("_ClipMax"),
        MaterialPropertyValue::Float(10.0),
    );

    assert_eq!(
        inferred_keyword_float_f32("CLIP", &store, lookup(62), &ids),
        Some(0.0)
    );
}

#[test]
fn inferred_pbs_displace_keywords_follow_host_keyword_predicates() {
    let (_reflected, ids, reg) =
        reflected_with_f32_fields(&[("_VertexOffsetBias", 0), ("_UVOffsetBias", 4)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        55,
        reg.intern("_VertexOffsetMap"),
        MaterialPropertyValue::Texture(packed_texture2d(123)),
    );
    store.set_material(
        55,
        reg.intern("_UVOffsetBias"),
        MaterialPropertyValue::Float(0.25),
    );
    store.set_material(
        55,
        reg.intern("_PositionOffsetMap"),
        MaterialPropertyValue::Texture(packed_texture2d(456)),
    );

    assert_eq!(
        inferred_keyword_float_f32("VERTEX_OFFSET", &store, lookup(55), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("UV_OFFSET", &store, lookup(55), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("OBJECT_POS_OFFSET", &store, lookup(55), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("VERTEX_POS_OFFSET", &store, lookup(55), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("VERTEX_OFFSET", &store, lookup(56), &ids),
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
        default_vec4_for_field("_BackgroundColor"),
        [0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(default_vec4_for_field("_Range"), [0.001, 0.001, 0.0, 0.0]);
    assert_eq!(
        default_vec4_for_field("_BehindFarColor"),
        [0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(default_vec4_for_field("_Tint0_"), [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn only_main_texture_bindings_fallback_to_primary_texture() {
    use crate::materials::embedded::texture_resolve::should_fallback_to_primary_texture;
    assert!(should_fallback_to_primary_texture("_MainTex"));
    assert!(!should_fallback_to_primary_texture("_MainTex1"));
    assert!(!should_fallback_to_primary_texture("_SpecularMap"));
}

/// `_ALBEDOTEX` keyword inference must treat a packed [`HostTextureAssetKind::RenderTexture`] like a
/// bound texture (parity with 2D-only `texture_property_asset_id_by_pid`).
#[test]
fn albedo_keyword_infers_from_render_texture_packed_id() {
    use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};

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
