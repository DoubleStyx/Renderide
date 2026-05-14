//! Stem-level reflection cache for embedded raster materials: composed WGSL, [`wgpu::BindGroupLayout`],
//! and interned property ids per [`crate::materials::ReflectedRasterLayout`].
//!
//! Per-frame uniform bytes and [`wgpu::BindGroup`] instances are built in [`crate::materials::embedded::material_bind`].

use hashbrown::HashMap;
use std::sync::Arc;

use crate::embedded_shaders;
use crate::embedded_shaders::EmbeddedTextureDefaultKind;
use crate::materials::host_data::PropertyIdRegistry;
use crate::materials::{ReflectedRasterLayout, reflect_raster_material_wgsl};

use super::uniform_pack::MaterialUniformValueSpaces;

/// Cached reflection and layout for one composed shader stem.
pub(crate) struct StemMaterialLayout {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) reflected: ReflectedRasterLayout,
    pub(crate) ids: Arc<StemEmbeddedPropertyIds>,
    pub(crate) uniform_value_spaces: MaterialUniformValueSpaces,
    pub(crate) texture_default_by_binding: HashMap<u32, EmbeddedTextureDefaultKind>,
}

/// Per-stem stable property ids from WGSL reflection (uniform members and `@group(1)` texture globals), built once when the stem layout loads.
pub(crate) struct StemEmbeddedPropertyIds {
    pub(crate) uniform_field_ids: HashMap<String, i32>,
    pub(crate) texture_binding_property_ids: HashMap<u32, Arc<[i32]>>,
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

pub(crate) use crate::materials::wgsl_reflect::identifier_names::unescape_property_name as shader_writer_unescaped_property_name;

impl StemEmbeddedPropertyIds {
    pub(crate) fn build(registry: &PropertyIdRegistry, reflected: &ReflectedRasterLayout) -> Self {
        let mut uniform_field_ids = HashMap::new();
        if let Some(u) = reflected.material_uniform.as_ref() {
            for field_name in u.fields.keys() {
                let host_field_name = shader_writer_unescaped_property_name(field_name);
                let pid = registry.intern(host_field_name);
                uniform_field_ids.insert(field_name.clone(), pid);
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
            uniform_field_ids,
            texture_binding_property_ids,
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

fn reflected_texture_bindings_by_name(reflected: &ReflectedRasterLayout) -> HashMap<String, u32> {
    let mut bindings = HashMap::new();
    for entry in &reflected.material_entries {
        if matches!(entry.ty, wgpu::BindingType::Texture { .. })
            && let Some(name) = reflected.material_group1_names.get(&entry.binding)
        {
            bindings.insert(
                shader_writer_unescaped_property_name(name.as_str()).to_string(),
                entry.binding,
            );
        }
    }
    bindings
}

fn texture_default_bindings_for_stem(
    stem: &str,
    reflected: &ReflectedRasterLayout,
) -> Result<HashMap<u32, EmbeddedTextureDefaultKind>, String> {
    let reflected_bindings = reflected_texture_bindings_by_name(reflected);
    let mut defaults = HashMap::new();
    for default in embedded_shaders::embedded_target_texture_defaults(stem) {
        let property = shader_writer_unescaped_property_name(default.property);
        let binding = reflected_bindings.get(property).copied().ok_or_else(|| {
            format!(
                "texture default `{property}` for stem `{stem}` has no reflected texture binding"
            )
        })?;
        if let Some(previous) = defaults.insert(binding, default.kind) {
            return Err(format!(
                "texture default for stem `{stem}` binding {binding} was declared twice ({previous:?}, {:?})",
                default.kind
            ));
        }
    }
    Ok(defaults)
}

/// Reflects embedded WGSL for `stem`, builds the `@group(1)` layout, and interns property ids.
pub(crate) fn build_stem_material_layout(
    device: &wgpu::Device,
    stem: &str,
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
        property_registry,
        &reflected,
    ));
    let uniform_value_spaces = MaterialUniformValueSpaces::for_stem(stem, &reflected);
    let texture_default_by_binding = texture_default_bindings_for_stem(stem, &reflected)?;

    Ok(Arc::new(StemMaterialLayout {
        bind_group_layout,
        reflected,
        ids,
        uniform_value_spaces,
        texture_default_by_binding,
    }))
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;

    use super::{
        StemEmbeddedPropertyIds, shader_writer_unescaped_property_name,
        texture_default_bindings_for_stem,
    };
    use crate::embedded_shaders;
    use crate::materials::host_data::PropertyIdRegistry;
    use crate::materials::reflect_raster_material_wgsl;

    #[test]
    fn xiexe_module_textures_resolve_to_unmangled_property_ids() {
        let wgsl =
            embedded_shaders::embedded_target_wgsl("xstoon2.0_default").expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let registry = PropertyIdRegistry::new();

        let ids = StemEmbeddedPropertyIds::build(&registry, &reflected);

        assert_eq!(
            ids.texture_binding_property_ids.get(&1).map(|p| &**p),
            Some([registry.intern("_MainTex"), registry.intern("_Tex"),].as_slice())
        );
    }

    #[test]
    fn xiexe_outline_emissive_typo_alias_is_preserved() {
        let wgsl =
            embedded_shaders::embedded_target_wgsl("xstoon2.0_default").expect("xiexe target WGSL");
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
    fn xiexe_layout_drops_xstoon3_extension_bindings() {
        let wgsl =
            embedded_shaders::embedded_target_wgsl("xstoon2.0_default").expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let unmangled: Vec<String> = reflected
            .material_group1_names
            .values()
            .map(|n| shader_writer_unescaped_property_name(n).to_string())
            .collect();
        for forbidden in [
            "_BakedCubemap",
            "_BakedCubemap_sampler",
            "_DetailNormalMap",
            "_DetailNormalMap_sampler",
            "_DetailMask",
            "_DetailMask_sampler",
            "_SpecularMap",
            "_SpecularMap_sampler",
        ] {
            assert!(
                !unmangled.iter().any(|n| n == forbidden),
                "xstoon 2.0 must not bind XSToon3 extension `{forbidden}`: {unmangled:?}"
            );
        }
    }

    #[test]
    fn all_embedded_material_textures_declare_default_directives() {
        for stem in embedded_shaders::COMPILED_MATERIAL_STEMS {
            let wgsl = embedded_shaders::embedded_target_wgsl(stem).expect("embedded target WGSL");
            let reflected = reflect_raster_material_wgsl(wgsl).expect("embedded reflection");
            let reflected_texture_names = reflected
                .material_entries
                .iter()
                .filter_map(|entry| {
                    matches!(entry.ty, wgpu::BindingType::Texture { .. })
                        .then(|| reflected.material_group1_names.get(&entry.binding))
                        .flatten()
                        .map(|name| shader_writer_unescaped_property_name(name).to_string())
                })
                .collect::<HashSet<_>>();
            let default_names = embedded_shaders::embedded_target_texture_defaults(stem)
                .iter()
                .map(|default| shader_writer_unescaped_property_name(default.property).to_string())
                .collect::<HashSet<_>>();

            assert_eq!(
                default_names, reflected_texture_names,
                "{stem} texture defaults must match reflected texture bindings"
            );
            texture_default_bindings_for_stem(stem, &reflected)
                .expect("texture defaults map to reflected texture bindings");
        }
    }
}
