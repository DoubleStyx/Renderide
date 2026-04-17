//! Uniform byte packing for embedded `@group(1)` material blocks (reflection-driven defaults and keywords).

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::assets::texture::unpack_host_texture_packed;
use crate::materials::{ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind};

use super::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};

/// True when the host material has a `set_float` for `property_id` with value ≥ 0.5 (Unity shader keyword pattern).
fn keyword_float_enabled_by_pid(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> bool {
    matches!(
        store.get_merged(lookup, property_id),
        Some(MaterialPropertyValue::Float(f)) if *f >= 0.5
    )
}

fn keyword_float_enabled_any_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32; 3],
) -> bool {
    pids.iter()
        .any(|&pid| keyword_float_enabled_by_pid(store, lookup, pid))
}

fn first_float_by_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> Option<f32> {
    pids.iter()
        .find_map(|&pid| match store.get_merged(lookup, pid) {
            Some(MaterialPropertyValue::Float(f)) => Some(*f),
            Some(MaterialPropertyValue::Float4(v)) => Some(v[0]),
            _ => None,
        })
}

fn shader_writer_unescaped_field_name(field_name: &str) -> &str {
    let Some(stripped) = field_name.strip_suffix('_') else {
        return field_name;
    };
    if stripped
        .chars()
        .next_back()
        .is_some_and(|c| c.is_ascii_digit())
    {
        stripped
    } else {
        field_name
    }
}

#[allow(clippy::approx_constant)]
fn default_vec4_for_field(field_name: &str) -> [f32; 4] {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if field_name.ends_with("_ST") {
        return [1.0, 1.0, 0.0, 0.0];
    }
    match field_name {
        "_Rect" => [0.0, 0.0, 1.0, 1.0],
        "_FOV" => [6.2831855, 3.1415927, 0.0, 0.0],
        "_SecondTexOffset" => [0.0, 0.0, 0.0, 0.0],
        "_OffsetMagnitude" => [0.1, 0.1, 0.0, 0.0],
        "_PerspectiveFOV" => [0.785398, 0.785398, 0.0, 0.0],
        "_Tint0" => [1.0, 0.0, 0.0, 1.0],
        "_Tint1" => [0.0, 1.0, 0.0, 1.0],
        "_OverlayTint" => [1.0, 1.0, 1.0, 0.5],
        "_EmissionColor"
        | "_EmissionColor1"
        | "_IntersectEmissionColor"
        | "_OutsideColor"
        | "_OcclusionColor"
        | "_SSColor" => [0.0, 0.0, 0.0, 0.0],
        "_OutlineColor" => [0.0, 0.0, 0.0, 1.0],
        "_RimColor" | "_ShadowRim" | "_MatcapTint" => [1.0, 1.0, 1.0, 1.0],
        "_BehindFarColor" | "_FrontFarColor" | "_FarColor" | "_FarColor0" => [0.0, 0.0, 0.0, 1.0],
        "_FarColor1" => [0.2, 0.2, 0.2, 1.0],
        "_NearColor1" => [0.8, 0.8, 0.8, 0.8],
        "_SpecularColor" | "_SpecularColor1" => [1.0, 1.0, 1.0, 0.5],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

fn is_keyword_like_field(field_name: &str) -> bool {
    let field_name = shader_writer_unescaped_field_name(field_name);
    let stripped = field_name.strip_prefix('_').unwrap_or(field_name);
    !stripped.is_empty()
        && stripped
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// `true` when the property has a packed texture that unpacks to any supported host kind (2D, RT, …).
fn texture_property_any_kind_present_by_pid(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> bool {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Texture(packed)) => {
            unpack_host_texture_packed(*packed).is_some()
        }
        _ => false,
    }
}

fn texture_property_present_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> bool {
    pids.iter()
        .any(|&pid| texture_property_any_kind_present_by_pid(store, lookup, pid))
}

fn inferred_keyword_float_f32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(probes) = ids.keyword_field_probe_ids.get(field_name) {
        if keyword_float_enabled_any_pids(store, lookup, probes) {
            return Some(1.0);
        }
    }

    let kw = ids.shared.as_ref();
    match field_name {
        "_ALPHATEST_ON" => {
            let mode = first_float_by_pids(store, lookup, &[kw.mode]).map(|v| v.round() as i32);
            let blend = first_float_by_pids(store, lookup, &[kw.blend_mode, kw.blend_mode_alt])
                .map(|v| v.round() as i32);
            return Some(if mode == Some(1) || blend == Some(1) {
                1.0
            } else {
                0.0
            });
        }
        "_ALPHABLEND_ON" => {
            let mode = first_float_by_pids(store, lookup, &[kw.mode]).map(|v| v.round() as i32);
            let blend = first_float_by_pids(store, lookup, &[kw.blend_mode, kw.blend_mode_alt])
                .map(|v| v.round() as i32);
            return Some(if mode == Some(2) || blend == Some(2) {
                1.0
            } else {
                0.0
            });
        }
        "_ALPHAPREMULTIPLY_ON" => {
            let mode = first_float_by_pids(store, lookup, &[kw.mode]).map(|v| v.round() as i32);
            let blend = first_float_by_pids(store, lookup, &[kw.blend_mode, kw.blend_mode_alt])
                .map(|v| v.round() as i32);
            return Some(if mode == Some(3) || blend == Some(3) {
                1.0
            } else {
                0.0
            });
        }
        _ => {}
    }

    let inferred = match field_name {
        "_LERPTEX" => texture_property_present_pids(store, lookup, &[kw.lerp_tex]),
        "_ALBEDOTEX" => texture_property_present_pids(store, lookup, &[kw.main_tex, kw.main_tex1]),
        "_EMISSION" | "_EMISSIONTEX" => {
            texture_property_present_pids(store, lookup, &[kw.emission_map, kw.emission_map1])
        }
        "_NORMALMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.normal_map, kw.normal_map1, kw.bump_map],
        ),
        "_SPECULARMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.specular_map, kw.specular_map1, kw.spec_gloss_map],
        ),
        "_METALLICGLOSSMAP" => {
            texture_property_present_pids(store, lookup, &[kw.metallic_gloss_map])
        }
        "_METALLICMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.metallic_map, kw.metallic_map1, kw.metallic_gloss_map],
        ),
        "_DETAIL_MULX2" => texture_property_present_pids(
            store,
            lookup,
            &[kw.detail_albedo_map, kw.detail_normal_map, kw.detail_mask],
        ),
        "_PARALLAXMAP" => texture_property_present_pids(store, lookup, &[kw.parallax_map]),
        "_OCCLUSION" => texture_property_present_pids(
            store,
            lookup,
            &[kw.occlusion, kw.occlusion1, kw.occlusion_map],
        ),
        _ if is_keyword_like_field(field_name) => false,
        _ => return None,
    };
    Some(if inferred { 1.0 } else { 0.0 })
}

fn default_f32_for_field(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> f32 {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(v) = inferred_keyword_float_f32(field_name, store, lookup, ids) {
        return v;
    }
    match field_name {
        "_Lerp"
        | "_TextureLerp"
        | "_ProjectionLerp"
        | "_CubeLOD"
        | "_Metallic"
        | "_Metallic1"
        | "_UVSec"
        | "_Mode"
        | "_OffsetFactor"
        | "_OffsetUnits"
        | "_Stencil"
        | "_StencilOp"
        | "_RimIntensity"
        | "_RimAlbedoTint"
        | "_RimCubemapTint"
        | "_SpecularIntensity"
        | "_ShadowRimAlbedoTint"
        | "_OutlineAlbedoTint"
        | "_OutlineLighting"
        | "_OutlineEmissive"
        | "_OutlineEmissiveues"
        | "_FadeDither"
        | "_FadeDitherDistance"
        | "_VertexColorAlbedo"
        | "_TilingMode"
        | "_UVSetAlbedo"
        | "_UVSetNormal"
        | "_UVSetDetNormal"
        | "_UVSetDetMask"
        | "_UVSetMetallic"
        | "_UVSetSpecular"
        | "_UVSetReflectivity"
        | "_UVSetThickness"
        | "_UVSetOcclusion"
        | "_UVSetEmission"
        | "_ClearCoat"
        | "_ReflectionBlendMode"
        | "_EmissionToDiffuse"
        | "_SpecMode"
        | "_SpecularStyle"
        | "_Offset" => 0.0,
        "_NormalScale"
        | "_NormalScale1"
        | "_BumpScale"
        | "_DetailNormalMapScale"
        | "_GlossMapScale"
        | "_OcclusionStrength"
        | "_SpecularHighlights"
        | "_GlossyReflections"
        | "_Exposure"
        | "_Gamma"
        | "_ZWrite"
        | "_Saturation"
        | "_Reflectivity"
        | "_ClearcoatStrength"
        | "_ScaleWithLight"
        | "_ScaleWithLightSensitivity"
        | "_RimAttenEffect"
        | "_SpecularAlbedoTint"
        | "_OutlineWidth"
        | "_SSDistortion"
        | "_SSPower"
        | "_SSScale"
        | "_SrcBlendBase"
        | "_SrcBlendAdd" => 1.0,
        "_Exp" | "_Exp0" | "_Exp1" | "_PolarPow" | "_LerpPolarPow" => 1.0,
        "_MaxIntensity" => 4.0,
        "_Parallax" => 0.02,
        "_GammaCurve" => 2.2,
        "_SrcBlend" => 1.0,
        "_DstBlend" | "_DstBlendBase" => 0.0,
        "_DstBlendAdd" => 1.0,
        "_ZTest" | "_Cull" | "_Culling" => 2.0,
        "_StencilComp" => 8.0,
        "_StencilWriteMask" | "_StencilReadMask" => 255.0,
        "_ColorMask" => 15.0,
        "_colormask" => 15.0,
        "_ReflectionMode" => 3.0,
        "_ClearcoatSmoothness" => 0.8,
        "_RimRange" | "_ShadowRimRange" => 0.7,
        "_RimThreshold" | "_ShadowRimThreshold" => 0.1,
        "_RimSharpness" => 0.1,
        "_ShadowRimSharpness" => 0.3,
        "_AnisotropicAX" => 0.25,
        "_AnisotropicAY" => 0.75,
        "_HalftoneDotSize" => 1.7,
        "_HalftoneDotAmount" => 50.0,
        "_HalftoneLineAmount" => 150.0,
        "_Cutoff" | "_AlphaClip" | "_Glossiness" | "_Glossiness1" => 0.5,
        _ => 0.5,
    }
}

/// Packs `UI_TextUnlit`-style `_TextMode`: explicit `0`/`1`/`2`, else keyword floats `MSDF` / `SDF` / `RASTER`, else `0` (MSDF default).
fn packed_text_mode_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> f32 {
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, kw.text_mode) {
        return *f;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.msdf) {
        return 0.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.sdf) {
        return 2.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.raster) {
        return 1.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.msdf_lower) {
        return 0.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.sdf_lower) {
        return 2.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.raster_lower) {
        return 1.0;
    }
    0.0
}

/// Packs `_RectClip`: explicit value, else `RECTCLIP` / `rectclip` keyword floats, else `0`.
fn packed_rect_clip_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> f32 {
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, kw.rect_clip) {
        return *f;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.rectclip) {
        return 1.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.rectclip_lower) {
        return 1.0;
    }
    0.0
}

/// Packs `flags` from `_Flags` when present; otherwise derives bits from texture presence,
/// `_Cutoff`, and Unity `#pragma multi_compile` keyword floats.
fn pack_flags_u32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    primary_texture_any_kind_present: bool,
    cutoff: f32,
) -> u32 {
    if field_name != "flags" {
        return 0;
    }
    if let Some(v) = store.get_merged(lookup, kw.flags) {
        match v {
            MaterialPropertyValue::Float(f) => return (*f).max(0.0) as u32,
            MaterialPropertyValue::Float4(a) => return a[0].max(0.0) as u32,
            _ => {}
        }
    }

    let mut flags = 0u32;

    if primary_texture_any_kind_present {
        flags |= 0x01;
    }
    if cutoff > 0.0 && cutoff < 1.0 {
        flags |= 0x02;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.offset_texture)
        || keyword_float_enabled_by_pid(store, lookup, kw.offset_texture_alt)
    {
        flags |= 0x04;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_mul)
        || keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_mul_alt)
    {
        flags |= 0x08;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_clip)
        || keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_clip_alt)
    {
        flags |= 0x10;
    }
    if let Some(mask_mode) = first_float_by_pids(store, lookup, &[kw.mask_mode, kw.mask_mode_alt]) {
        match mask_mode.round() as i32 {
            // Resonite MaskTextureMode: MultiplyAlpha = 0, Cutoff = 1.
            0 => flags |= 0x08,
            1 => flags |= 0x10,
            // Tolerate future/combined modes by enabling both paths.
            v if v > 1 => flags |= 0x18,
            _ => {}
        }
    }
    if let Some(blend_mode) =
        first_float_by_pids(store, lookup, &[kw.blend_mode, kw.blend_mode_alt])
    {
        // Resonite BlendMode.Cutout = 1.
        if blend_mode.round() as i32 == 1 {
            flags |= 0x02;
        }
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mul_rgb_by_alpha)
        || keyword_float_enabled_by_pid(store, lookup, kw.mul_rgb_by_alpha_alt)
    {
        flags |= 0x20;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mul_alpha_intensity)
        || keyword_float_enabled_by_pid(store, lookup, kw.mul_alpha_intensity_alt)
    {
        flags |= 0x40;
    }

    flags
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

fn write_u32_at(buf: &mut [u8], field: &ReflectedUniformField, v: u32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Builds CPU bytes for the reflected material uniform block.
pub(crate) fn build_embedded_uniform_bytes(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    primary_texture_any_kind_present: bool,
) -> Option<Vec<u8>> {
    let u = reflected.material_uniform.as_ref()?;
    let mut buf = vec![0u8; u.total_size as usize];

    let mut cutoff = 0.5f32;
    for (field_name, field) in &u.fields {
        let pid = *ids.uniform_field_ids.get(field_name)?;
        match field.kind {
            ReflectedUniformScalarKind::Vec4 => {
                let default = default_vec4_for_field(field_name);
                let mut v = default;
                if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, pid) {
                    v = *c;
                }
                write_f32x4_at(&mut buf, field, &v);
            }
            ReflectedUniformScalarKind::F32 => {
                let v = if field_name == "_TextMode" {
                    packed_text_mode_f32(store, lookup, ids.shared.as_ref())
                } else if field_name == "_RectClip" {
                    packed_rect_clip_f32(store, lookup, ids.shared.as_ref())
                } else if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, pid)
                {
                    *f
                } else if field_name == "_Cutoff" {
                    first_float_by_pids(
                        store,
                        lookup,
                        &[ids.shared.alpha_cutoff, ids.shared.alpha_cutoff_alt],
                    )
                    .unwrap_or_else(|| default_f32_for_field(field_name, store, lookup, ids))
                } else {
                    default_f32_for_field(field_name, store, lookup, ids)
                };
                if field_name == "_Cutoff" {
                    cutoff = v;
                }
                write_f32_at(&mut buf, field, v);
            }
            ReflectedUniformScalarKind::U32 => {
                if field_name == "flags" {
                    let flags = pack_flags_u32(
                        field_name,
                        store,
                        lookup,
                        ids.shared.as_ref(),
                        primary_texture_any_kind_present,
                        cutoff,
                    );
                    write_u32_at(&mut buf, field, flags);
                }
            }
            ReflectedUniformScalarKind::Unsupported => {}
        }
    }

    Some(buf)
}

#[cfg(test)]
mod text_uniform_packing_tests {
    use super::*;
    use crate::assets::material::PropertyIdRegistry;
    use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};
    use crate::backend::embedded::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    #[test]
    fn packed_text_mode_defaults_to_msdf_when_empty() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        assert_eq!(
            packed_text_mode_f32(&store, lookup(1), &kw),
            0.0,
            "FrooxEngine default GlyphRenderMethod is MSDF"
        );
    }

    #[test]
    fn packed_text_mode_explicit_overrides_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid_tm = reg.intern("_TextMode");
        let pid_msdf = reg.intern("MSDF");
        store.set_material(1, pid_msdf, MaterialPropertyValue::Float(1.0));
        store.set_material(1, pid_tm, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(1), &kw), 1.0);
    }

    #[test]
    fn packed_text_mode_raster_from_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid = reg.intern("RASTER");
        store.set_material(2, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(2), &kw), 1.0);
    }

    #[test]
    fn packed_rect_clip_from_rectclip_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid = reg.intern("RECTCLIP");
        store.set_material(3, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_rect_clip_f32(&store, lookup(3), &kw), 1.0);
    }

    #[test]
    fn unlit_mask_mode_derives_mask_flags() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid = reg.intern("MaskMode");

        store.set_material(7, pid, MaterialPropertyValue::Float(0.0));
        assert_eq!(
            pack_flags_u32("flags", &store, lookup(7), &kw, false, 0.5) & 0x18,
            0x08
        );

        store.set_material(7, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(
            pack_flags_u32("flags", &store, lookup(7), &kw, false, 0.5) & 0x18,
            0x10
        );
    }

    #[test]
    fn unlit_blend_mode_cutout_enables_alpha_test_flag() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid = reg.intern("BlendMode");
        store.set_material(8, pid, MaterialPropertyValue::Float(1.0));
        assert_ne!(
            pack_flags_u32("flags", &store, lookup(8), &kw, false, 0.5) & 0x02,
            0
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
    fn default_pbs_uniforms_match_unity_style_defaults() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        assert_eq!(default_f32_for_field("_Lerp", &store, lookup(5), &ids), 0.0);
        assert_eq!(
            default_f32_for_field("_NormalScale", &store, lookup(5), &ids),
            1.0
        );
        assert_eq!(default_f32_for_field("_Cull", &store, lookup(5), &ids), 2.0);
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_SpecularColor"),
            [1.0, 1.0, 1.0, 0.5]
        );
        assert_eq!(default_vec4_for_field("_Rect"), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(default_vec4_for_field("_OverlayTint"), [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(
            default_vec4_for_field("_BehindFarColor"),
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            default_f32_for_field("_ZTest", &store, lookup(5), &ids),
            2.0
        );
        assert_eq!(default_f32_for_field("_Exp", &store, lookup(5), &ids), 1.0);
        assert_eq!(
            default_f32_for_field("_GammaCurve", &store, lookup(5), &ids),
            2.2
        );
        assert_eq!(default_vec4_for_field("_Tint0_"), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            default_f32_for_field("_Metallic1_", &store, lookup(5), &ids),
            0.0
        );
        assert_eq!(
            default_f32_for_field("_StencilComp", &store, lookup(5), &ids),
            8.0
        );
        assert_eq!(
            default_f32_for_field("_ColorMask", &store, lookup(5), &ids),
            15.0
        );
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
