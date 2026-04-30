//! Resolves a [`MaterialRenderState`] from a host material / property block lookup.
//!
//! [`material_render_state_from_maps`] takes the two pre-fetched inner property maps and reads
//! every Unity render-state property in one pass; [`material_render_state_for_lookup`] is the
//! convenience entry point that fetches those maps from a [`MaterialDictionary`].

use crate::materials::host_data::{MaterialDictionary, MaterialPropertyLookupIds};

use super::super::material_passes::{
    MaterialPipelinePropertyIds, PropertyMapRef, first_float_from_maps,
};
use super::types::{
    MaterialCullOverride, MaterialDepthOffsetState, MaterialRenderState, MaterialStencilState,
};
use super::unity_mapping::{unity_mask, unity_offset_units, unity_u8};

/// Resolves Unity color, stencil, and depth properties using pre-fetched inner maps. Prefer this
/// in hot paths that also call [`crate::materials::material_blend_mode_from_maps`] for the same
/// lookup — the two outer-map probes are amortised across both calls.
pub fn material_render_state_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialRenderState {
    // Shorthand to keep the per-field lookups readable.
    let get = |pids: &[i32]| first_float_from_maps(material_map, property_block_map, pids);

    let stencil = resolve_stencil(get, ids);
    let color_mask = get(&ids.color_mask).map(unity_u8);
    let depth_write = get(&ids.z_write).map(|v| v.round().clamp(0.0, 1.0) >= 0.5);
    let depth_compare = get(&ids.z_test).map(unity_u8);
    let cull_override = resolve_cull_override(get(&ids.cull));
    let depth_offset = resolve_depth_offset(get(&ids.offset_factor), get(&ids.offset_units));

    MaterialRenderState {
        stencil,
        color_mask,
        depth_write,
        depth_compare,
        depth_offset,
        cull_override,
    }
}

/// Resolves Unity color, stencil, and depth properties for a material/property-block pair.
pub fn material_render_state_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialRenderState {
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup);
    material_render_state_from_maps(mat_map, pb_map, ids)
}

fn resolve_stencil(
    mut get: impl FnMut(&[i32]) -> Option<f32>,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialStencilState {
    let stencil_ref = get(&ids.stencil_ref);
    let stencil_comp = get(&ids.stencil_comp);
    let stencil_op = get(&ids.stencil_op);
    let stencil_fail_op = get(&ids.stencil_fail_op);
    let stencil_depth_fail_op = get(&ids.stencil_depth_fail_op);
    let stencil_read_mask = get(&ids.stencil_read_mask);
    let stencil_write_mask = get(&ids.stencil_write_mask);

    let stencil_present = stencil_ref.is_some()
        || stencil_comp.is_some()
        || stencil_op.is_some()
        || stencil_fail_op.is_some()
        || stencil_depth_fail_op.is_some()
        || stencil_read_mask.is_some()
        || stencil_write_mask.is_some();
    let compare = stencil_comp.map_or(8, unity_u8);
    MaterialStencilState {
        enabled: stencil_present && compare != 0,
        reference: stencil_ref.map_or(0, unity_mask),
        compare,
        pass_op: stencil_op.map_or(0, unity_u8),
        fail_op: stencil_fail_op.map_or(0, unity_u8),
        depth_fail_op: stencil_depth_fail_op.map_or(0, unity_u8),
        read_mask: stencil_read_mask.map_or(0xff, unity_mask),
        write_mask: stencil_write_mask.map_or(0xff, unity_mask),
    }
}

fn resolve_cull_override(value: Option<f32>) -> MaterialCullOverride {
    match value.map(unity_u8) {
        None => MaterialCullOverride::Unspecified,
        // UnityEngine.Rendering.CullMode: Off / Front / Back
        Some(0) => MaterialCullOverride::Off,
        Some(1) => MaterialCullOverride::Front,
        Some(2) => MaterialCullOverride::Back,
        Some(_) => MaterialCullOverride::Unspecified,
    }
}

fn resolve_depth_offset(
    factor: Option<f32>,
    units: Option<f32>,
) -> Option<MaterialDepthOffsetState> {
    if factor.is_none() && units.is_none() {
        return None;
    }
    MaterialDepthOffsetState::new(factor.unwrap_or(0.0), units.map_or(0, unity_offset_units))
}
