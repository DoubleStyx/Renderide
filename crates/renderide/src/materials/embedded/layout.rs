//! Stem-level reflection cache for embedded raster materials: composed WGSL, [`wgpu::BindGroupLayout`],
//! and interned property ids per [`crate::materials::ReflectedRasterLayout`].
//!
//! Per-frame uniform bytes and [`wgpu::BindGroup`] instances are built in [`crate::materials::embedded::material_bind`].

use hashbrown::HashMap;
use std::sync::Arc;

use crate::embedded_shaders;
use crate::materials::host_data::PropertyIdRegistry;
use crate::materials::{ReflectedRasterLayout, reflect_raster_material_wgsl};

/// Cached reflection and layout for one composed shader stem.
pub(crate) struct StemMaterialLayout {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) reflected: ReflectedRasterLayout,
    pub(crate) ids: Arc<StemEmbeddedPropertyIds>,
}

/// Pre-interned property ids used by [`super::uniform_pack::inferred_keyword_float_f32`] to
/// probe texture presence (PBS `_NORMALMAP` / `_EMISSION` / `_SPECULARMAP` / ... flags) and by
/// the `_ALPHATEST_ON` / `_ALPHABLEND_ON` / `_MUL_RGB_BY_ALPHA` inference path that reads on-wire signals,
/// each captured as a synthetic property by
/// [`crate::materials::host_data::parse_materials_update_batch_into_store`]:
///
/// 1. The [`crate::shared::MaterialRenderType`] tag at `_RenderType`
///    (`MaterialProvider.SetBlendMode` family -- `Unlit`, `Toon`, etc.).
/// 2. The Unity render queue at `_RenderQueue` (PBS `AlphaHandling` family --
///    `PBS_DualSidedMaterial.cs` and friends bypass `SetBlendMode` and the `_ALPHACLIP`
///    keyword bitmask, signaling AlphaClip via queue 2450 and Opaque via queue 2000).
/// 3. The `_SrcBlend` / `_DstBlend` factors for distinguishing alpha-blend, additive, and
///    premultiplied alpha within the Transparent range.
///
/// FrooxEngine's `ShaderKeywords.Variant` bitmask is never sent over IPC, so every
/// multi-compile-keyword toggle has to be inferred from what the host does send: texture
/// bindings, numeric material properties, render-type tag, render queue, and blend factors.
pub(crate) struct EmbeddedSharedKeywordIds {
    pub(crate) blend_mode: i32,
    pub(crate) mode: i32,
    pub(crate) render_type: i32,
    pub(crate) render_queue: i32,
    pub(crate) src_blend: i32,
    pub(crate) dst_blend: i32,
    pub(crate) lerp_tex: i32,
    pub(crate) tex: i32,
    pub(crate) far_tex: i32,
    pub(crate) near_tex: i32,
    pub(crate) far_tex0: i32,
    pub(crate) near_tex0: i32,
    pub(crate) far_tex1: i32,
    pub(crate) near_tex1: i32,
    pub(crate) gradient: i32,
    pub(crate) main_tex: i32,
    pub(crate) main_tex1: i32,
    pub(crate) mask_tex: i32,
    pub(crate) emission_map: i32,
    pub(crate) emission_map1: i32,
    pub(crate) emission_map2: i32,
    pub(crate) emission_map3: i32,
    pub(crate) normal_map: i32,
    pub(crate) normal_map0: i32,
    pub(crate) normal_map1: i32,
    pub(crate) bump_map: i32,
    pub(crate) specular_map: i32,
    pub(crate) specular_map1: i32,
    pub(crate) specular_map2: i32,
    pub(crate) specular_map3: i32,
    pub(crate) spec_gloss_map: i32,
    pub(crate) metallic_map: i32,
    pub(crate) metallic_map1: i32,
    pub(crate) metallic_gloss_map: i32,
    pub(crate) metallic_gloss01: i32,
    pub(crate) metallic_gloss23: i32,
    pub(crate) detail_albedo_map: i32,
    pub(crate) detail_normal_map: i32,
    pub(crate) detail_mask: i32,
    pub(crate) parallax_map: i32,
    pub(crate) occlusion: i32,
    pub(crate) occlusion1: i32,
    pub(crate) occlusion_map: i32,
    /// `PBSColorSplat` height blend texture.
    pub(crate) packed_height_map: i32,
    /// `PBSColorSplat` packed layer-normal texture for layers 0 and 1.
    pub(crate) packed_normal_map01: i32,
    /// `PBSColorSplat` packed layer-normal texture for layers 2 and 3.
    pub(crate) packed_normal_map23: i32,
    /// `PBSColorSplat` packed emission texture.
    pub(crate) packed_emission_map: i32,
    /// `PBSDisplace` vertex-height texture.
    pub(crate) vertex_offset_map: i32,
    /// `PBSDisplace` fragment UV-offset texture.
    pub(crate) uv_offset_map: i32,
    /// `PBSDisplace` world/object-position texture used to offset vertex-offset sampling UVs.
    pub(crate) position_offset_map: i32,
    /// `Projection360` secondary equirectangular texture binding.
    pub(crate) second_tex: i32,
    /// `Projection360` tint-modulation texture binding.
    pub(crate) tint_tex: i32,
    /// `Projection360` direction-perturbation texture binding.
    pub(crate) offset_tex: i32,
    /// `Projection360` primary cubemap texture binding.
    pub(crate) main_cube: i32,
    /// `Projection360` secondary cubemap texture binding.
    pub(crate) second_cube: i32,
    /// FrooxEngine `FogBoxVolumeMaterial` / Unity `Volume/FogBox`: Froox sends `AccumulationMode` as this float
    /// enum (0=Linear, 1=Exp, 2=Exp2), not the `FOG_*` shader keyword floats until a UI toggle.
    pub(crate) accumulation_mode: i32,
    pub(crate) accumulation_mode_legacy: i32,
}

impl EmbeddedSharedKeywordIds {
    pub(crate) fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            blend_mode: registry.intern("_BlendMode"),
            mode: registry.intern("_Mode"),
            render_type: registry.intern("_RenderType"),
            render_queue: registry.intern("_RenderQueue"),
            src_blend: registry.intern("_SrcBlend"),
            dst_blend: registry.intern("_DstBlend"),
            lerp_tex: registry.intern("_LerpTex"),
            tex: registry.intern("_Tex"),
            far_tex: registry.intern("_FarTex"),
            near_tex: registry.intern("_NearTex"),
            far_tex0: registry.intern("_FarTex0"),
            near_tex0: registry.intern("_NearTex0"),
            far_tex1: registry.intern("_FarTex1"),
            near_tex1: registry.intern("_NearTex1"),
            gradient: registry.intern("_Gradient"),
            main_tex: registry.intern("_MainTex"),
            main_tex1: registry.intern("_MainTex1"),
            mask_tex: registry.intern("_MaskTex"),
            emission_map: registry.intern("_EmissionMap"),
            emission_map1: registry.intern("_EmissionMap1"),
            emission_map2: registry.intern("_EmissionMap2"),
            emission_map3: registry.intern("_EmissionMap3"),
            normal_map: registry.intern("_NormalMap"),
            normal_map0: registry.intern("_NormalMap0"),
            normal_map1: registry.intern("_NormalMap1"),
            bump_map: registry.intern("_BumpMap"),
            specular_map: registry.intern("_SpecularMap"),
            specular_map1: registry.intern("_SpecularMap1"),
            specular_map2: registry.intern("_SpecularMap2"),
            specular_map3: registry.intern("_SpecularMap3"),
            spec_gloss_map: registry.intern("_SpecGlossMap"),
            metallic_map: registry.intern("_MetallicMap"),
            metallic_map1: registry.intern("_MetallicMap1"),
            metallic_gloss_map: registry.intern("_MetallicGlossMap"),
            metallic_gloss01: registry.intern("_MetallicGloss01"),
            metallic_gloss23: registry.intern("_MetallicGloss23"),
            detail_albedo_map: registry.intern("_DetailAlbedoMap"),
            detail_normal_map: registry.intern("_DetailNormalMap"),
            detail_mask: registry.intern("_DetailMask"),
            parallax_map: registry.intern("_ParallaxMap"),
            occlusion: registry.intern("_Occlusion"),
            occlusion1: registry.intern("_Occlusion1"),
            occlusion_map: registry.intern("_OcclusionMap"),
            packed_height_map: registry.intern("_PackedHeightMap"),
            packed_normal_map01: registry.intern("_PackedNormalMap01"),
            packed_normal_map23: registry.intern("_PackedNormalMap23"),
            packed_emission_map: registry.intern("_PackedEmissionMap"),
            vertex_offset_map: registry.intern("_VertexOffsetMap"),
            uv_offset_map: registry.intern("_UVOffsetMap"),
            position_offset_map: registry.intern("_PositionOffsetMap"),
            second_tex: registry.intern("_SecondTex"),
            tint_tex: registry.intern("_TintTex"),
            offset_tex: registry.intern("_OffsetTex"),
            main_cube: registry.intern("_MainCube"),
            second_cube: registry.intern("_SecondCube"),
            accumulation_mode: registry.intern("_AccumulationMode"),
            accumulation_mode_legacy: registry.intern("AccumulationMode"),
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

/// Returns alternate host property names for a canonical texture binding name.
///
/// Only the `_Tex` <-> `_MainTex` cross-alias is live (`UnlitMaterial` uses `_Tex`; PBS/Toon
/// materials use `_MainTex`). A host-side audit confirmed that the no-underscore forms `Texture`,
/// `MaskTexture`, `OffsetTexture` are never declared as `MaterialProperty` and thus never sent;
/// they were removed.
fn texture_property_aliases(name: &str) -> &'static [&'static str] {
    match name {
        "_Tex" => &["_MainTex"],
        "_MainTex" => &["_Tex"],
        _ => &[],
    }
}

pub(crate) use crate::materials::shader_writer::unescape_property_name as shader_writer_unescaped_property_name;

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
            if matches!(entry.ty, wgpu::BindingType::Texture { .. })
                && let Some(name) = reflected.material_group1_names.get(&entry.binding)
            {
                let host_name = shader_writer_unescaped_property_name(name.as_str());
                let pid = registry.intern(host_name);

                let mut pids = Vec::with_capacity(1 + texture_property_aliases(host_name).len());
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds, shader_writer_unescaped_property_name,
    };
    use crate::materials::host_data::PropertyIdRegistry;
    use crate::materials::reflect_raster_material_wgsl;

    #[test]
    fn xiexe_module_textures_resolve_to_unmangled_property_ids() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let registry = PropertyIdRegistry::new();
        let shared = Arc::new(EmbeddedSharedKeywordIds::new(&registry));

        let ids = StemEmbeddedPropertyIds::build(shared, &registry, &reflected);

        assert_eq!(
            ids.texture_binding_property_ids.get(&1).map(|p| &**p),
            Some([registry.intern("_MainTex"), registry.intern("_Tex"),].as_slice())
        );
    }

    #[test]
    fn xiexe_outline_emissive_typo_alias_is_preserved() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let uniform = reflected
            .material_uniform
            .as_ref()
            .expect("xiexe material uniform block");
        assert!(
            uniform.fields.contains_key("_OutlineEmissiveues"),
            "the deliberate `_OutlineEmissiveues` Unity-property typo must remain in the reflected uniform block; XSToon2.0.shader sets this name and removing it would break outline-mode lookup"
        );
    }

    #[test]
    fn xiexe_baked_cubemap_binding_is_reflected() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let unmangled: Vec<String> = reflected
            .material_group1_names
            .values()
            .map(|n| shader_writer_unescaped_property_name(n).to_string())
            .collect();
        assert!(
            unmangled.iter().any(|n| n == "_BakedCubemap"),
            "missing `_BakedCubemap` cube binding in xstoon2 layout: {unmangled:?}"
        );
        assert!(
            unmangled.iter().any(|n| n == "_BakedCubemap_sampler"),
            "missing `_BakedCubemap_sampler` in xstoon2 layout: {unmangled:?}"
        );
    }
}
