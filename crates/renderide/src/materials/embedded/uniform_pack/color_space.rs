//! Explicit material uniform value-space metadata.

use hashbrown::HashMap;

use crate::materials::{ReflectedRasterLayout, ReflectedUniformScalarKind};

use super::helpers::shader_writer_unescaped_field_name;

/// Host value space expected for a reflected material uniform field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaterialUniformValueSpace {
    Raw,
    SrgbColor,
    SrgbColorArray,
}

/// Per-stem material uniform value-space overrides.
#[derive(Clone, Debug, Default)]
pub(crate) struct MaterialUniformValueSpaces {
    fields: HashMap<String, MaterialUniformValueSpace>,
}

impl MaterialUniformValueSpaces {
    /// Builds the color-space overrides for one reflected embedded shader target.
    pub(crate) fn for_stem(stem: &str, reflected: &ReflectedRasterLayout) -> Self {
        let Some(uniform) = reflected.material_uniform.as_ref() else {
            return Self::default();
        };
        let mut fields = HashMap::new();
        for (field_name, field) in &uniform.fields {
            let host_field_name = shader_writer_unescaped_field_name(field_name);
            let value_space = match field.kind {
                ReflectedUniformScalarKind::Vec4 if srgb_vec4_uniform_field(host_field_name) => {
                    MaterialUniformValueSpace::SrgbColor
                }
                ReflectedUniformScalarKind::Unsupported
                    if srgb_vec4_array_uniform_field(stem, host_field_name) =>
                {
                    MaterialUniformValueSpace::SrgbColorArray
                }
                _ => MaterialUniformValueSpace::Raw,
            };
            if value_space != MaterialUniformValueSpace::Raw {
                fields.insert(field_name.clone(), value_space);
            }
        }
        Self { fields }
    }

    /// Returns whether the reflected `vec4<f32>` field should be converted from sRGB to linear RGB.
    pub(crate) fn is_srgb_vec4(&self, field_name: &str) -> bool {
        self.fields
            .get(field_name)
            .is_some_and(|space| *space == MaterialUniformValueSpace::SrgbColor)
    }

    /// Returns whether the reflected `array<vec4<f32>, N>` field should be converted from sRGB to linear RGB.
    pub(crate) fn is_srgb_vec4_array(&self, field_name: &str) -> bool {
        self.fields
            .get(field_name)
            .is_some_and(|space| *space == MaterialUniformValueSpace::SrgbColorArray)
    }
}

fn srgb_vec4_uniform_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "_BackgroundColor"
            | "_BaseColor"
            | "_BehindColor"
            | "_BehindFarColor"
            | "_BehindNearColor"
            | "_Blend"
            | "_Color"
            | "_Color0"
            | "_Color1"
            | "_Color2"
            | "_Color3"
            | "_ColorTint"
            | "_EdgeColor"
            | "_EdgeEmission"
            | "_EdgeEmissionColor"
            | "_EmissionColor"
            | "_EmissionColor1"
            | "_EmissionColor2"
            | "_EmissionColor3"
            | "_EmissionColorFrom"
            | "_EmissionColorTo"
            | "_FarColor"
            | "_FarColor0"
            | "_FarColor1"
            | "_FillTint"
            | "_FresnelTint"
            | "_FrontColor"
            | "_FrontFarColor"
            | "_FrontNearColor"
            | "_GroundColor"
            | "_InnerColor"
            | "_IntersectColor"
            | "_IntersectEmissionColor"
            | "_MatcapTint"
            | "_NearColor"
            | "_NearColor0"
            | "_NearColor1"
            | "_OcclusionColor"
            | "_OuterColor"
            | "_OutlineColor"
            | "_OutlineTint"
            | "_OutsideColor"
            | "_OverlayTint"
            | "_RimColor"
            | "_SSColor"
            | "_SecondaryEmissionColor"
            | "_ShadowRim"
            | "_SkyTint"
            | "_SpecColor"
            | "_SpecularColor"
            | "_SpecularColor1"
            | "_SpecularColor2"
            | "_SpecularColor3"
            | "_SunColor"
            | "_Tint"
            | "_Tint0"
            | "_Tint1"
            | "_TintColor"
            | "_node_2829"
    )
}

fn srgb_vec4_array_uniform_field(stem: &str, field_name: &str) -> bool {
    field_name == "_TintColors"
        && matches!(
            source_stem_from_target_stem(stem),
            "pbsdistancelerp"
                | "pbsdistancelerpspecular"
                | "pbsdistancelerptransparent"
                | "pbsdistancelerpspeculartransparent"
        )
}

fn source_stem_from_target_stem(stem: &str) -> &str {
    stem.strip_suffix("_default")
        .or_else(|| stem.strip_suffix("_multiview"))
        .unwrap_or(stem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedded_shaders;
    use crate::materials::reflect_raster_material_wgsl;

    fn reflected_material_value_spaces(stem: &str) -> MaterialUniformValueSpaces {
        let wgsl = embedded_shaders::embedded_target_wgsl(stem).expect("embedded WGSL target");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("reflect embedded WGSL target");
        MaterialUniformValueSpaces::for_stem(stem, &reflected)
    }

    #[test]
    fn srgb_conversion_matches_elements_material_profile_rules() {
        let linear = crate::color_space::srgb_f32x4_rgb_to_linear([-0.5, 0.04045, 1.25, 0.33]);

        assert!((linear[0] - -0.214_041_14).abs() < 0.000_001);
        assert!((linear[1] - (0.04045 / 12.92)).abs() < 0.000_001);
        assert_eq!(linear[2], 1.25);
        assert_eq!(linear[3], 0.33);
    }

    #[test]
    fn metadata_marks_known_color_vec4_uniforms() {
        let overlay = reflected_material_value_spaces("overlay_default");
        assert!(overlay.is_srgb_vec4("_Blend"));

        let voronoi = reflected_material_value_spaces("pbsvoronoicrystal_default");
        assert!(voronoi.is_srgb_vec4("_EdgeEmission"));

        let pbs = reflected_material_value_spaces("pbsmetallic_default");
        assert!(pbs.is_srgb_vec4("_Color"));
        assert!(pbs.is_srgb_vec4("_EmissionColor"));
    }

    #[test]
    fn metadata_keeps_color_named_transform_uniforms_raw() {
        let color_splat = reflected_material_value_spaces("pbscolorsplat_default");
        assert!(!color_splat.is_srgb_vec4("_ColorMap_ST"));

        let color_mask = reflected_material_value_spaces("pbscolormask_default");
        assert!(!color_mask.is_srgb_vec4("_ColorMask_ST"));

        let projection = reflected_material_value_spaces("projection360_default");
        assert!(!projection.is_srgb_vec4("_TintTex_ST"));
    }

    #[test]
    fn metadata_marks_only_srgb_authored_color_arrays() {
        let distance = reflected_material_value_spaces("pbsdistancelerp_default");
        assert!(distance.is_srgb_vec4_array("_TintColors"));
        assert!(!distance.is_srgb_vec4_array("_Points"));

        let gradient = reflected_material_value_spaces("gradientskybox_default");
        assert!(!gradient.is_srgb_vec4_array("_Color0"));
        assert!(!gradient.is_srgb_vec4_array("_Color1"));
    }

    #[test]
    fn metadata_does_not_classify_st_fields_by_name_heuristics() {
        for stem in embedded_shaders::COMPILED_MATERIAL_STEMS {
            let wgsl = embedded_shaders::embedded_target_wgsl(stem).expect("embedded WGSL target");
            let reflected =
                reflect_raster_material_wgsl(wgsl).expect("reflect embedded WGSL target");
            let spaces = MaterialUniformValueSpaces::for_stem(stem, &reflected);
            let Some(uniform) = reflected.material_uniform.as_ref() else {
                continue;
            };
            for field_name in uniform.fields.keys() {
                let host_field_name = shader_writer_unescaped_field_name(field_name);
                if host_field_name.ends_with("_ST") {
                    assert!(
                        !spaces.is_srgb_vec4(field_name),
                        "{stem}:{field_name} must remain raw transform data"
                    );
                }
            }
        }
    }
}
