//! Stem-level reflection cache for embedded raster materials: composed WGSL, [`wgpu::BindGroupLayout`],
//! and interned property ids per [`crate::materials::ReflectedRasterLayout`].
//!
//! Per-frame uniform bytes and [`wgpu::BindGroup`] instances are built in [`crate::backend::embedded::material_bind`].

use hashbrown::HashMap;
use std::sync::Arc;

use crate::assets::material::PropertyIdRegistry;
use crate::embedded_shaders;
use crate::materials::{reflect_raster_material_wgsl, ReflectedRasterLayout};

/// Cached reflection and layout for one composed shader stem.
pub(crate) struct StemMaterialLayout {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) reflected: ReflectedRasterLayout,
    pub(crate) ids: Arc<StemEmbeddedPropertyIds>,
}

/// Pre-interned property ids for static keyword names (`MSDF`, `_Flags`, texture probe lists), shared by all embedded stems.
pub(crate) struct EmbeddedSharedKeywordIds {
    pub(crate) text_mode: i32,
    pub(crate) rect_clip: i32,
    pub(crate) msdf: i32,
    pub(crate) sdf: i32,
    pub(crate) raster: i32,
    pub(crate) msdf_lower: i32,
    pub(crate) sdf_lower: i32,
    pub(crate) raster_lower: i32,
    pub(crate) rectclip: i32,
    pub(crate) rectclip_lower: i32,
    pub(crate) flags: i32,
    pub(crate) offset_texture: i32,
    pub(crate) offset_texture_alt: i32,
    pub(crate) mask_texture_mul: i32,
    pub(crate) mask_texture_mul_alt: i32,
    pub(crate) mask_texture_clip: i32,
    pub(crate) mask_texture_clip_alt: i32,
    pub(crate) mask_mode: i32,
    pub(crate) mask_mode_alt: i32,
    pub(crate) blend_mode: i32,
    pub(crate) blend_mode_alt: i32,
    pub(crate) alpha_cutoff: i32,
    pub(crate) alpha_cutoff_alt: i32,
    pub(crate) mode: i32,
    pub(crate) mul_rgb_by_alpha: i32,
    pub(crate) mul_rgb_by_alpha_alt: i32,
    pub(crate) mul_alpha_intensity: i32,
    pub(crate) mul_alpha_intensity_alt: i32,
    pub(crate) lerp_tex: i32,
    pub(crate) main_tex: i32,
    pub(crate) main_tex1: i32,
    pub(crate) emission_map: i32,
    pub(crate) emission_map1: i32,
    pub(crate) normal_map: i32,
    pub(crate) normal_map1: i32,
    pub(crate) bump_map: i32,
    pub(crate) specular_map: i32,
    pub(crate) specular_map1: i32,
    pub(crate) spec_gloss_map: i32,
    pub(crate) metallic_map: i32,
    pub(crate) metallic_map1: i32,
    pub(crate) metallic_gloss_map: i32,
    pub(crate) detail_albedo_map: i32,
    pub(crate) detail_normal_map: i32,
    pub(crate) detail_mask: i32,
    pub(crate) parallax_map: i32,
    pub(crate) occlusion: i32,
    pub(crate) occlusion1: i32,
    pub(crate) occlusion_map: i32,
}

impl EmbeddedSharedKeywordIds {
    pub(crate) fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            text_mode: registry.intern("_TextMode"),
            rect_clip: registry.intern("_RectClip"),
            msdf: registry.intern("MSDF"),
            sdf: registry.intern("SDF"),
            raster: registry.intern("RASTER"),
            msdf_lower: registry.intern("msdf"),
            sdf_lower: registry.intern("sdf"),
            raster_lower: registry.intern("raster"),
            rectclip: registry.intern("RECTCLIP"),
            rectclip_lower: registry.intern("rectclip"),
            flags: registry.intern("_Flags"),
            offset_texture: registry.intern("_OFFSET_TEXTURE"),
            offset_texture_alt: registry.intern("_OffsetTexture"),
            mask_texture_mul: registry.intern("_MASK_TEXTURE_MUL"),
            mask_texture_mul_alt: registry.intern("_MaskTextureMul"),
            mask_texture_clip: registry.intern("_MASK_TEXTURE_CLIP"),
            mask_texture_clip_alt: registry.intern("_MaskTextureClip"),
            mask_mode: registry.intern("_MaskMode"),
            mask_mode_alt: registry.intern("MaskMode"),
            blend_mode: registry.intern("_BlendMode"),
            blend_mode_alt: registry.intern("BlendMode"),
            alpha_cutoff: registry.intern("_AlphaCutoff"),
            alpha_cutoff_alt: registry.intern("AlphaCutoff"),
            mode: registry.intern("_Mode"),
            mul_rgb_by_alpha: registry.intern("_MUL_RGB_BY_ALPHA"),
            mul_rgb_by_alpha_alt: registry.intern("_MulRgbByAlpha"),
            mul_alpha_intensity: registry.intern("_MUL_ALPHA_INTENSITY"),
            mul_alpha_intensity_alt: registry.intern("_MulAlphaIntensity"),
            lerp_tex: registry.intern("_LerpTex"),
            main_tex: registry.intern("_MainTex"),
            main_tex1: registry.intern("_MainTex1"),
            emission_map: registry.intern("_EmissionMap"),
            emission_map1: registry.intern("_EmissionMap1"),
            normal_map: registry.intern("_NormalMap"),
            normal_map1: registry.intern("_NormalMap1"),
            bump_map: registry.intern("_BumpMap"),
            specular_map: registry.intern("_SpecularMap"),
            specular_map1: registry.intern("_SpecularMap1"),
            spec_gloss_map: registry.intern("_SpecGlossMap"),
            metallic_map: registry.intern("_MetallicMap"),
            metallic_map1: registry.intern("_MetallicMap1"),
            metallic_gloss_map: registry.intern("_MetallicGlossMap"),
            detail_albedo_map: registry.intern("_DetailAlbedoMap"),
            detail_normal_map: registry.intern("_DetailNormalMap"),
            detail_mask: registry.intern("_DetailMask"),
            parallax_map: registry.intern("_ParallaxMap"),
            occlusion: registry.intern("_Occlusion"),
            occlusion1: registry.intern("_Occlusion1"),
            occlusion_map: registry.intern("_OcclusionMap"),
        }
    }
}

/// Per-stem stable property ids from WGSL reflection (uniform members and `@group(1)` texture globals), built once when the stem layout loads.
pub(crate) struct StemEmbeddedPropertyIds {
    pub(crate) shared: Arc<EmbeddedSharedKeywordIds>,
    pub(crate) uniform_field_ids: HashMap<String, i32>,
    pub(crate) texture_binding_property_ids: HashMap<u32, Arc<[i32]>>,
    pub(crate) keyword_field_probe_ids: HashMap<String, [i32; 3]>,
}

fn texture_property_aliases(name: &str) -> &'static [&'static str] {
    match name {
        "_Tex" => &["Texture", "_MainTex"],
        "_MaskTex" => &["MaskTexture"],
        "_OffsetTex" => &["OffsetTexture"],
        "_MainTex" => &["Texture", "_Tex"],
        _ => &[],
    }
}

pub(crate) fn shader_writer_unescaped_property_name(name: &str) -> &str {
    let name = name
        .split_once("X_naga_oil_mod_")
        .map_or(name, |(base, _)| base);
    let Some(stripped) = name.strip_suffix('_') else {
        return name;
    };
    if stripped
        .chars()
        .next_back()
        .is_some_and(|c| c.is_ascii_digit())
    {
        stripped
    } else {
        name
    }
}

impl StemEmbeddedPropertyIds {
    pub(crate) fn build(
        shared: Arc<EmbeddedSharedKeywordIds>,
        registry: &PropertyIdRegistry,
        reflected: &ReflectedRasterLayout,
    ) -> Self {
        let mut uniform_field_ids = HashMap::new();
        let mut keyword_field_probe_ids = HashMap::new();
        if let Some(u) = reflected.material_uniform.as_ref() {
            for field_name in u.fields.keys() {
                let host_field_name = shader_writer_unescaped_property_name(field_name);
                let pid = registry.intern(host_field_name);
                uniform_field_ids.insert(field_name.clone(), pid);
                let stripped = host_field_name.strip_prefix('_').unwrap_or(host_field_name);
                let lowercase = stripped.to_ascii_lowercase();
                let pid_strip = registry.intern(stripped);
                let pid_lower = registry.intern(lowercase.as_str());
                keyword_field_probe_ids.insert(field_name.clone(), [pid, pid_strip, pid_lower]);
            }
        }

        let mut texture_binding_property_ids = HashMap::new();
        for entry in &reflected.material_entries {
            if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
                if let Some(name) = reflected.material_group1_names.get(&entry.binding) {
                    let host_name = shader_writer_unescaped_property_name(name.as_str());
                    let pid = registry.intern(host_name);

                    let mut pids =
                        Vec::with_capacity(1 + texture_property_aliases(host_name).len());
                    pids.push(pid);
                    for alias in texture_property_aliases(host_name) {
                        let alias_pid = registry.intern(alias);
                        if !pids.contains(&alias_pid) {
                            pids.push(alias_pid);
                        }
                    }
                    texture_binding_property_ids.insert(entry.binding, Arc::from(pids));
                }
            }
        }

        Self {
            shared,
            uniform_field_ids,
            texture_binding_property_ids,
            keyword_field_probe_ids,
        }
    }
}

#[cfg(test)]
impl StemEmbeddedPropertyIds {
    /// Shared keyword ids only (no per-stem uniform/texture reflection); for unit tests.
    pub fn minimal_for_tests(registry: &PropertyIdRegistry) -> Self {
        Self {
            shared: Arc::new(EmbeddedSharedKeywordIds::new(registry)),
            uniform_field_ids: HashMap::new(),
            texture_binding_property_ids: HashMap::new(),
            keyword_field_probe_ids: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        shader_writer_unescaped_property_name, EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds,
    };
    use crate::assets::material::PropertyIdRegistry;
    use crate::materials::reflect_raster_material_wgsl;

    #[test]
    fn shader_writer_escape_strips_digit_suffix_underscore() {
        assert_eq!(shader_writer_unescaped_property_name("_Tint0_"), "_Tint0");
        assert_eq!(shader_writer_unescaped_property_name("_Color1_"), "_Color1");
        assert_eq!(
            shader_writer_unescaped_property_name("_MainTex_ST"),
            "_MainTex_ST"
        );
    }

    #[test]
    fn shader_writer_escape_strips_naga_oil_module_suffix() {
        assert_eq!(
            shader_writer_unescaped_property_name(
                "_MainTexX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ4GSZLYMU5DU5DPN5XDEX"
            ),
            "_MainTex"
        );
        assert_eq!(
            shader_writer_unescaped_property_name(
                "_Tint0_X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ4GSZLYMU5DU5DPN5XDEX"
            ),
            "_Tint0"
        );
    }

    #[test]
    fn xiexe_module_textures_resolve_to_unmangled_property_ids() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xiexe_xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let registry = PropertyIdRegistry::new();
        let shared = Arc::new(EmbeddedSharedKeywordIds::new(&registry));

        let ids = StemEmbeddedPropertyIds::build(shared, &registry, &reflected);

        assert_eq!(
            ids.texture_binding_property_ids.get(&1).map(|p| &**p),
            Some(
                [
                    registry.intern("_MainTex"),
                    registry.intern("Texture"),
                    registry.intern("_Tex"),
                ]
                .as_slice()
            )
        );
    }
}

/// Stable hash for stem strings (uniform/bind cache keys).
pub(crate) fn stem_hash(stem: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    stem.hash(&mut h);
    h.finish()
}

/// Reflects embedded WGSL for `stem`, builds the `@group(1)` layout, and interns property ids.
pub(crate) fn build_stem_material_layout(
    device: &wgpu::Device,
    stem: &str,
    shared_keyword_ids: &Arc<EmbeddedSharedKeywordIds>,
    property_registry: &PropertyIdRegistry,
) -> Result<Arc<StemMaterialLayout>, String> {
    let wgsl = embedded_shaders::embedded_target_wgsl(stem)
        .ok_or_else(|| format!("embedded WGSL missing for stem {stem}"))?;
    let reflected =
        reflect_raster_material_wgsl(wgsl).map_err(|e| format!("reflect {stem}: {e}"))?;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("embedded_raster_material"),
        entries: &reflected.material_entries,
    });

    let ids = Arc::new(StemEmbeddedPropertyIds::build(
        Arc::clone(shared_keyword_ids),
        property_registry,
        &reflected,
    ));

    Ok(Arc::new(StemMaterialLayout {
        bind_group_layout,
        reflected,
        ids,
    }))
}
