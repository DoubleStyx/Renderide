//! Small property-store probes used by uniform packing and keyword inference.

use crate::assets::texture::unpack_host_texture_packed;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};

/// True when the host material has a `set_float` for `property_id` with value >= 0.5 (Unity shader keyword pattern).
pub(super) fn keyword_float_enabled_by_pid(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> bool {
    matches!(
        store.get_merged(lookup, property_id),
        Some(MaterialPropertyValue::Float(f)) if *f >= 0.5
    )
}

pub(super) fn keyword_float_enabled_any_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32; 3],
) -> bool {
    pids.iter()
        .any(|&pid| keyword_float_enabled_by_pid(store, lookup, pid))
}

pub(super) fn first_float_by_pids(
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

pub(super) use crate::materials::shader_writer::strip_writer_digit_escape as shader_writer_unescaped_field_name;

pub(super) fn default_f32_for_field(field_name: &str) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    match field_name {
        "_Cutoff" => Some(0.5),
        "_Glossiness" => Some(0.5),
        "_GlossMapScale" => Some(1.0),
        "_BumpScale" | "_NormalScale" | "_DetailNormalMapScale" => Some(1.0),
        "_OcclusionStrength" => Some(1.0),
        "_Parallax" => Some(0.02),
        "_Metallic" | "_UVSec" => Some(0.0),
        _ => None,
    }
}

pub(super) fn default_vec4_for_field(field_name: &str) -> [f32; 4] {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if field_name.ends_with("_ST") {
        return [1.0, 1.0, 0.0, 0.0];
    }
    match field_name {
        "_Point" => [0.0, 0.0, 0.0, 0.0],
        "_Rect" => [0.0, 0.0, 1.0, 1.0],
        "_FOV" => [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0],
        "_SecondTexOffset" => [0.0, 0.0, 0.0, 0.0],
        "_OffsetMagnitude" => [0.1, 0.1, 0.0, 0.0],
        "_PointSize" => [0.1, 0.1, 0.0, 0.0],
        "_PerspectiveFOV" => [
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
            0.0,
        ],
        "_Tint0" => [1.0, 0.0, 0.0, 1.0],
        "_Tint1" => [0.0, 1.0, 0.0, 1.0],
        "_OverlayTint" => [1.0, 1.0, 1.0, 0.5],
        "_BackgroundColor" => [0.0, 0.0, 0.0, 0.0],
        "_Range" => [0.001, 0.001, 0.0, 0.0],
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

pub(super) fn is_keyword_like_field(field_name: &str) -> bool {
    let field_name = shader_writer_unescaped_field_name(field_name);
    let stripped = field_name.strip_prefix('_').unwrap_or(field_name);
    !stripped.is_empty()
        && stripped
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// `true` when the property has a packed texture that unpacks to any supported host kind (2D, RT, ...).
pub(super) fn texture_property_any_kind_present_by_pid(
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

pub(super) fn texture_property_present_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> bool {
    pids.iter()
        .any(|&pid| texture_property_any_kind_present_by_pid(store, lookup, pid))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(mat: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: mat,
            mesh_property_block_slot0: None,
        }
    }

    #[test]
    fn default_vec4_known_names() {
        assert_eq!(default_vec4_for_field("_Rect"), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(default_vec4_for_field("_MainTex_ST"), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_BackgroundColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(default_vec4_for_field("_Range"), [0.001, 0.001, 0.0, 0.0]);
        assert_eq!(default_vec4_for_field("_Tint0"), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn default_vec4_unknown_falls_back_to_white() {
        assert_eq!(default_vec4_for_field("_Unknown"), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn default_f32_known_standard_names() {
        assert_eq!(default_f32_for_field("_GlossMapScale"), Some(1.0));
        assert_eq!(default_f32_for_field("_OcclusionStrength"), Some(1.0));
        assert_eq!(default_f32_for_field("_UVSec"), Some(0.0));
        assert_eq!(default_f32_for_field("_UnknownScalar"), None);
    }

    #[test]
    fn default_vec4_unescaped_digit_field_resolves_to_known_default() {
        // `_Tex0_` unescapes to `_Tex0`; not a known special case, fallback is white.
        assert_eq!(default_vec4_for_field("_Tex0_"), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn is_keyword_like_recognizes_upper_and_digit_tokens() {
        assert!(is_keyword_like_field("_MSDF"));
        assert!(is_keyword_like_field("_USE_LOD_0"));
        assert!(!is_keyword_like_field("_MainTex"));
        assert!(!is_keyword_like_field("_"));
        assert!(!is_keyword_like_field(""));
    }

    #[test]
    fn keyword_float_enabled_thresholds_on_half() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 100, MaterialPropertyValue::Float(0.4));
        store.set_material(1, 101, MaterialPropertyValue::Float(0.5));
        store.set_material(1, 102, MaterialPropertyValue::Float(1.0));
        assert!(!keyword_float_enabled_by_pid(&store, lookup(1), 100));
        assert!(keyword_float_enabled_by_pid(&store, lookup(1), 101));
        assert!(keyword_float_enabled_by_pid(&store, lookup(1), 102));
        assert!(!keyword_float_enabled_by_pid(&store, lookup(1), 999));
    }

    #[test]
    fn keyword_float_any_pids_short_circuits() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 101, MaterialPropertyValue::Float(1.0));
        assert!(keyword_float_enabled_any_pids(
            &store,
            lookup(1),
            &[50, 60, 101],
        ));
        assert!(!keyword_float_enabled_any_pids(
            &store,
            lookup(1),
            &[50, 60, 70],
        ));
    }

    #[test]
    fn first_float_by_pids_accepts_float4_x_component() {
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, 200, MaterialPropertyValue::Float4([3.5, 0.0, 0.0, 0.0]));
        store.set_material(1, 201, MaterialPropertyValue::Float(7.0));
        assert_eq!(
            first_float_by_pids(&store, lookup(1), &[999, 200, 201]),
            Some(3.5)
        );
        assert_eq!(
            first_float_by_pids(&store, lookup(1), &[999, 201]),
            Some(7.0)
        );
        assert_eq!(first_float_by_pids(&store, lookup(1), &[999]), None);
    }
}
