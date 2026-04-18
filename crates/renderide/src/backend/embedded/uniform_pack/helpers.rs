//! Small property-store probes used by uniform packing and keyword inference.

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::assets::texture::unpack_host_texture_packed;

/// True when the host material has a `set_float` for `property_id` with value ≥ 0.5 (Unity shader keyword pattern).
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

pub(super) fn shader_writer_unescaped_field_name(field_name: &str) -> &str {
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

pub(super) fn default_vec4_for_field(field_name: &str) -> [f32; 4] {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if field_name.ends_with("_ST") {
        return [1.0, 1.0, 0.0, 0.0];
    }
    match field_name {
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

/// `true` when the property has a packed texture that unpacks to any supported host kind (2D, RT, …).
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
