//! Parse composed WGSL with naga and derive [`wgpu::BindGroupLayoutEntry`] lists for `@group(1)` and
//! `@group(2)`, and a [`ReflectedRasterLayout::layout_fingerprint`] for tests and diagnostics.
//!
//! Validates `@group(0)` against [`crate::backend::FrameGpuResources`] buffer sizes and optional
//! scene-depth snapshot bindings.

mod bind_layout;
mod fingerprint;
mod frame_group0;
mod resource;
mod types;
mod uniform_vertex;

pub use types::{
    ReflectError, ReflectedMaterialUniformBlock, ReflectedRasterLayout, ReflectedUniformField,
    ReflectedUniformScalarKind,
};

use std::collections::BTreeMap;

use naga::front::wgsl::parse_str;
use naga::proc::Layouter;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;

use self::bind_layout::global_to_layout_entry;
use self::fingerprint::fingerprint_layout;
use self::frame_group0::validate_frame_group0;
use self::uniform_vertex::{
    material_uniform_requires_intersection_subpass, reflect_first_group1_uniform_struct,
    reflect_group1_global_binding_names, reflect_vs_main_max_vertex_location,
};

/// Parses and validates WGSL, checks frame globals, and builds layout entries for groups 1 and 2.
pub fn reflect_raster_material_wgsl(source: &str) -> Result<ReflectedRasterLayout, ReflectError> {
    let module = parse_str(source).map_err(|e| ReflectError::Parse(e.to_string()))?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all());
    validator
        .validate(&module)
        .map_err(|e| ReflectError::Validate(e.to_string()))?;

    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|e| ReflectError::Layout(e.to_string()))?;

    validate_frame_group0(&module, &layouter)?;

    let mut g1: BTreeMap<u32, wgpu::BindGroupLayoutEntry> = BTreeMap::new();
    let mut g2: BTreeMap<u32, wgpu::BindGroupLayoutEntry> = BTreeMap::new();

    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group > 2 {
            return Err(ReflectError::InvalidBindGroup(rb.group));
        }
        if rb.group == 0 {
            continue;
        }
        let entry = global_to_layout_entry(&module, &layouter, gv, rb.group, rb.binding)?;
        match rb.group {
            1 => {
                g1.insert(rb.binding, entry);
            }
            2 => {
                g2.insert(rb.binding, entry);
            }
            _ => {}
        }
    }

    let material_entries: Vec<_> = g1.into_values().collect();
    let per_draw_entries: Vec<_> = g2.into_values().collect();

    let material_uniform = reflect_first_group1_uniform_struct(&module, &layouter);
    let material_group1_names = reflect_group1_global_binding_names(&module);
    let vs_max_vertex_location = reflect_vs_main_max_vertex_location(&module);

    let layout_fingerprint = fingerprint_layout(
        &material_entries,
        &per_draw_entries,
        vs_max_vertex_location,
        &material_group1_names,
    );

    let requires_intersection_pass =
        material_uniform_requires_intersection_subpass(&material_uniform);

    Ok(ReflectedRasterLayout {
        layout_fingerprint,
        material_entries,
        per_draw_entries,
        material_uniform,
        material_group1_names,
        vs_max_vertex_location,
        requires_intersection_pass,
    })
}

/// Returns true when `vs_main` uses vertex `@location` 2 or higher (UV0 stream).
pub fn reflect_vertex_shader_needs_uv0_stream(wgsl_source: &str) -> bool {
    reflect_raster_material_wgsl(wgsl_source)
        .ok()
        .and_then(|r| r.vs_max_vertex_location)
        .map(|m| m >= 2)
        .unwrap_or(false)
}

/// Returns true when `vs_main` uses vertex `@location` 3 or higher (extra color stream).
pub fn reflect_vertex_shader_needs_color_stream(wgsl_source: &str) -> bool {
    reflect_raster_material_wgsl(wgsl_source)
        .ok()
        .and_then(|r| r.vs_max_vertex_location)
        .map(|m| m >= 3)
        .unwrap_or(false)
}

/// `true` when reflection reports an intersect-style material (uniform field `_IntersectColor`).
pub fn reflect_raster_material_requires_intersection_pass(wgsl_source: &str) -> bool {
    reflect_raster_material_wgsl(wgsl_source)
        .ok()
        .is_some_and(|r| r.requires_intersection_pass)
}

/// Validates that `@group(2)` matches the dynamic per-draw uniform slab (single binding, 256-byte stride).
pub fn validate_per_draw_group2(
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<(), ReflectError> {
    if entries.len() != 1 {
        return Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: 0,
            reason: format!(
                "expected exactly one per-draw binding, got {}",
                entries.len()
            ),
        });
    }
    let e = &entries[0];
    if e.binding != 0 {
        return Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: e.binding,
            reason: "per-draw binding must be @binding(0)".into(),
        });
    }
    match e.ty {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(n),
        } if n.get() == PER_DRAW_UNIFORM_STRIDE as u64 => Ok(()),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: 0,
            reason: "expected var<uniform> dynamic offset min_binding_size 256".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_debug_world_normals_default_embedded() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("debug_world_normals_default")
            .expect("stem");
        let r = reflect_raster_material_wgsl(wgsl).expect("reflect");
        assert!(r.material_entries.is_empty());
        validate_per_draw_group2(&r.per_draw_entries).expect("per_draw");
        assert_ne!(r.layout_fingerprint, 0);
        assert_eq!(
            r.vs_max_vertex_location,
            Some(1),
            "debug normals: position + normal only"
        );
    }

    /// Every composed `shaders/target/*.wgsl` must declare the full [`crate::backend::FrameGpuResources`] `@group(0)`
    /// contract; naga-oil strips unused imports, so a material that omits cluster buffer references
    /// can fail at runtime during pipeline creation unless this test catches it.
    #[test]
    fn reflect_all_embedded_material_targets_match_frame_group0() {
        for stem in crate::embedded_shaders::COMPILED_MATERIAL_STEMS {
            let wgsl = crate::embedded_shaders::embedded_target_wgsl(stem)
                .unwrap_or_else(|| panic!("embedded_target_wgsl missing for {stem}"));
            reflect_raster_material_wgsl(wgsl).unwrap_or_else(|e| {
                panic!("reflect_raster_material_wgsl failed for {stem}: {e}");
            });
        }
    }
}
