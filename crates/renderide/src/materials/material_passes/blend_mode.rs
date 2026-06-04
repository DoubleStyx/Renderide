//! Material blend mode reconstructed from `_SrcBlend` / `_DstBlend` host properties.

use crate::materials::host_data::MaterialPropertyValue;
#[cfg(test)]
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyLookupIds};

use super::property_ids::MaterialPipelinePropertyIds;
use super::wire_tables::unity_blend_factor;

/// Resonite/Froox material blend mode, or the shader stem's default when no material field is present.
///
/// Reconstructed from the `_SrcBlend` / `_DstBlend` blend-factor floats that FrooxEngine
/// writes for every material (see [`MaterialBlendMode::from_unity_blend_factors`]). The host
/// never sends a named `BlendMode` enum value on the wire -- `MaterialProvider.SetBlendMode(Alpha)`
/// on the C# side simply translates to `SrcBlend=SrcAlpha` / `DstBlend=OneMinusSrcAlpha` floats --
/// so only three shapes are observable here: no override, the `(1, 0)` opaque canonical form, and
/// every other valid `(src, dst)` pair.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialBlendMode {
    /// No material-level override; use the stem's normal static behavior.
    #[default]
    StemDefault,
    /// Canonical `Blend One Zero` -- opaque, no color blend.
    Opaque,
    /// Direct `Blend[src][dst], One One` factors from `_SrcBlend` / `_DstBlend`.
    UnityBlend {
        /// Source blend factor enum value.
        src: u8,
        /// Destination blend factor enum value.
        dst: u8,
    },
}

impl MaterialBlendMode {
    /// Returns the `(src, dst)` Unity blend factor pair, or `None` when the mode is `StemDefault`.
    #[inline]
    pub(crate) fn unity_blend_factors(self) -> Option<(u8, u8)> {
        match self {
            Self::StemDefault => None,
            Self::Opaque => Some((1, 0)),
            Self::UnityBlend { src, dst } => Some((src, dst)),
        }
    }

    /// Converts Unity `BlendMode` factor property values (`_SrcBlend`, `_DstBlend`).
    #[inline]
    pub fn from_unity_blend_factors(src: f32, dst: f32) -> Self {
        let src = src.round().clamp(0.0, 255.0) as u8;
        let dst = dst.round().clamp(0.0, 255.0) as u8;
        match (src, dst) {
            // UnityEngine.Rendering.BlendMode.One / Zero.
            (1, 0) => Self::Opaque,
            _ if unity_blend_factor(src).is_some() && unity_blend_factor(dst).is_some() => {
                Self::UnityBlend { src, dst }
            }
            _ => Self::StemDefault,
        }
    }

    /// Returns true when the mode must be sorted/drawn as transparent.
    #[inline]
    pub fn is_transparent(self) -> bool {
        matches!(self, Self::UnityBlend { .. })
    }
}

/// One side of a [`MaterialPropertyLookupIds`] fetched via
/// [`MaterialDictionary::fetch_property_maps`]: the inner `property_id -> value` map for either the
/// material or the property block. `None` when no properties have been stored for that id.
pub(crate) type PropertyMapRef<'a> = Option<&'a hashbrown::HashMap<i32, MaterialPropertyValue>>;

/// Iterates `pids` against pre-fetched material / property-block inner maps, matching the
/// [`crate::materials::host_data::MaterialPropertyStore::get_merged`] "property block overrides
/// material" semantics across all aliases.
#[inline]
pub(crate) fn first_float_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    pids: &[i32],
) -> Option<f32> {
    first_float_from_map(property_block_map, pids)
        .or_else(|| first_float_from_map(material_map, pids))
}

/// Like [`first_float_from_maps`] but reads `Float4` (vec4) values for `pids`. Used by the UI
/// rect-mask CPU cull to read `_Rect` from the merged material/property-block view.
#[inline]
pub(crate) fn first_vec4_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    pids: &[i32],
) -> Option<[f32; 4]> {
    first_vec4_from_map(property_block_map, pids)
        .or_else(|| first_vec4_from_map(material_map, pids))
}

#[inline]
fn first_float_from_map(map: PropertyMapRef<'_>, pids: &[i32]) -> Option<f32> {
    let map = map?;
    pids.iter().find_map(|&pid| match map.get(&pid)? {
        MaterialPropertyValue::Float(f) => Some(*f),
        MaterialPropertyValue::Float4(v4) => Some(v4[0]),
        _ => None,
    })
}

#[inline]
fn first_vec4_from_map(map: PropertyMapRef<'_>, pids: &[i32]) -> Option<[f32; 4]> {
    let map = map?;
    pids.iter().find_map(|&pid| match map.get(&pid)? {
        MaterialPropertyValue::Float4(v4) => Some(*v4),
        _ => None,
    })
}

/// Resolves a material/property-block `BlendMode` override using pre-fetched inner maps. Prefer
/// this in hot paths that also call [`crate::materials::material_render_state_from_maps`] for
/// the same lookup -- the two outer-map probes are amortised across both calls.
#[inline]
pub fn material_blend_mode_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    if let (Some(src), Some(dst)) = (
        first_float_from_maps(material_map, property_block_map, &ids.src_blend),
        first_float_from_maps(material_map, property_block_map, &ids.dst_blend),
    ) {
        return MaterialBlendMode::from_unity_blend_factors(src, dst);
    }
    MaterialBlendMode::StemDefault
}

/// Resolves a material/property-block `BlendMode` override.
#[cfg(test)]
pub fn material_blend_mode_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup);
    material_blend_mode_from_maps(mat_map, pb_map, ids)
}
