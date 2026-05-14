//! Parse composed WGSL with naga and derive [`wgpu::BindGroupLayoutEntry`] lists for `@group(1)` and
//! `@group(2)`, and a [`ReflectedRasterLayout::layout_fingerprint`] for tests and diagnostics.
//!
//! Validates `@group(0)` against the frame GPU ABI in [`crate::gpu`] and optional scene-depth
//! snapshot bindings.

mod bind_layout;
#[cfg(test)]
mod fingerprint;
mod frame_group0;
pub(in crate::materials) mod identifier_names;
mod resource;
mod types;
mod uniform_vertex;

pub use types::{
    ReflectError, ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind,
    ReflectedVertexInputFormat,
};
#[cfg(test)]
pub(crate) use types::{ReflectedMaterialUniformBlock, ReflectedVertexInput};

use std::collections::BTreeMap;

use naga::front::wgsl::parse_str;
use naga::proc::Layouter;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;

use self::bind_layout::global_to_layout_entry;
#[cfg(test)]
use self::fingerprint::fingerprint_layout;
use self::frame_group0::{reflect_frame_snapshot_usage, validate_frame_group0};
use self::uniform_vertex::{
    material_uniform_requires_intersection_subpass, reflect_first_group1_uniform_struct,
    reflect_group1_global_binding_names, reflect_vertex_entry_inputs,
    reflect_vs_main_vertex_inputs,
};

/// Parses and validates WGSL, checks frame globals, and builds layout entries for groups 1 and 2.
pub fn reflect_raster_material_wgsl(source: &str) -> Result<ReflectedRasterLayout, ReflectError> {
    reflect_raster_material_wgsl_inner(source, None)
}

/// Parses and validates WGSL using the material pass vertex entries for vertex stream reflection.
pub(in crate::materials) fn reflect_raster_material_wgsl_with_vertex_entries(
    source: &str,
    vertex_entries: &[&str],
) -> Result<ReflectedRasterLayout, ReflectError> {
    reflect_raster_material_wgsl_inner(source, Some(vertex_entries))
}

fn reflect_raster_material_wgsl_inner(
    source: &str,
    vertex_entries: Option<&[&str]>,
) -> Result<ReflectedRasterLayout, ReflectError> {
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
    let vs_vertex_inputs = if let Some(vertex_entries) = vertex_entries {
        reflect_vertex_entry_inputs(&module, vertex_entries)?
    } else {
        reflect_vs_main_vertex_inputs(&module)
    };
    let snapshot_usage = reflect_frame_snapshot_usage(&module);

    #[cfg(test)]
    let vs_max_vertex_location = vs_vertex_inputs.iter().map(|input| input.location).max();
    #[cfg(test)]
    let layout_fingerprint = fingerprint_layout(
        &material_entries,
        &per_draw_entries,
        vs_max_vertex_location,
        &vs_vertex_inputs,
        &material_group1_names,
    );

    let requires_intersection_pass =
        material_uniform_requires_intersection_subpass(material_uniform.as_ref());

    Ok(ReflectedRasterLayout {
        #[cfg(test)]
        layout_fingerprint,
        material_entries,
        per_draw_entries,
        material_uniform,
        material_group1_names,
        vs_vertex_inputs,
        #[cfg(test)]
        vs_max_vertex_location,
        uses_scene_depth_snapshot: snapshot_usage.depth,
        uses_scene_color_snapshot: snapshot_usage.color,
        requires_intersection_pass,
    })
}

/// Validates a reflected raster layout against device caps from [`crate::gpu::GpuLimits`].
///
/// Checks per-group entry count vs `max_bindings_per_bind_group`, per-stage sampler / sampled
/// texture counts across the full frame/material/per-draw pipeline layout, and uniform / storage
/// `min_binding_size` against the matching device cap. Used at pipeline build time so a material
/// that exceeds an effective device cap fails with a clear [`ReflectError`] instead of triggering a
/// downstream wgpu validation panic.
pub fn validate_layout_against_limits(
    layout: &ReflectedRasterLayout,
    frame_entries: &[wgpu::BindGroupLayoutEntry],
    limits: &crate::gpu::GpuLimits,
) -> Result<(), ReflectError> {
    validate_group_against_limits(0, frame_entries, limits)?;
    validate_group_against_limits(1, &layout.material_entries, limits)?;
    validate_group_against_limits(2, &layout.per_draw_entries, limits)?;
    validate_pipeline_stage_resources(frame_entries, layout, limits)
}

fn validate_pipeline_stage_resources(
    frame_entries: &[wgpu::BindGroupLayoutEntry],
    layout: &ReflectedRasterLayout,
    limits: &crate::gpu::GpuLimits,
) -> Result<(), ReflectError> {
    let max_samplers = limits.max_samplers_per_shader_stage();
    let max_textures = limits.max_sampled_textures_per_shader_stage();
    for (stage, stage_name) in [
        (wgpu::ShaderStages::VERTEX, "vertex"),
        (wgpu::ShaderStages::FRAGMENT, "fragment"),
    ] {
        let (samplers, textures) = count_stage_sampled_resources(
            stage,
            frame_entries
                .iter()
                .chain(layout.material_entries.iter())
                .chain(layout.per_draw_entries.iter()),
        );
        if samplers > max_samplers {
            return Err(ReflectError::ExceedsSamplersPerStage {
                stage: stage_name,
                count: samplers,
                max: max_samplers,
            });
        }
        if textures > max_textures {
            return Err(ReflectError::ExceedsSampledTexturesPerStage {
                stage: stage_name,
                count: textures,
                max: max_textures,
            });
        }
    }
    Ok(())
}

fn count_stage_sampled_resources<'a>(
    stage: wgpu::ShaderStages,
    entries: impl Iterator<Item = &'a wgpu::BindGroupLayoutEntry>,
) -> (u32, u32) {
    let mut samplers = 0u32;
    let mut textures = 0u32;
    for entry in entries {
        if !entry.visibility.contains(stage) {
            continue;
        }
        let count = entry.count.map_or(1, |count| count.get());
        match entry.ty {
            wgpu::BindingType::Sampler(_) => samplers = samplers.saturating_add(count),
            wgpu::BindingType::Texture { .. } => textures = textures.saturating_add(count),
            _ => {}
        }
    }
    (samplers, textures)
}

fn validate_group_against_limits(
    group: u32,
    entries: &[wgpu::BindGroupLayoutEntry],
    limits: &crate::gpu::GpuLimits,
) -> Result<(), ReflectError> {
    let count = entries.len() as u32;
    let max_bindings = limits.max_bindings_per_bind_group();
    if count > max_bindings {
        return Err(ReflectError::ExceedsBindingsPerGroup {
            group,
            count,
            max: max_bindings,
        });
    }
    for e in entries {
        if let wgpu::BindingType::Buffer {
            ty,
            min_binding_size: Some(min_size),
            ..
        } = e.ty
        {
            let n = min_size.get();
            match ty {
                wgpu::BufferBindingType::Uniform => {
                    if !limits.uniform_binding_fits(n) {
                        return Err(ReflectError::UniformBindingExceedsLimit {
                            group,
                            binding: e.binding,
                            size: n,
                            max: limits.max_uniform_buffer_binding_size(),
                        });
                    }
                }
                wgpu::BufferBindingType::Storage { .. } => {
                    if !limits.storage_binding_fits(n) {
                        return Err(ReflectError::StorageBindingExceedsLimit {
                            group,
                            binding: e.binding,
                            size: n,
                            max: limits.max_storage_buffer_binding_size(),
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validates a built vertex layout against device caps. Counts every attribute across all buffers.
pub fn validate_vertex_layout_against_limits(
    buffers: &[wgpu::VertexBufferLayout<'_>],
    limits: &crate::gpu::GpuLimits,
) -> Result<(), ReflectError> {
    let buffer_count = buffers.len() as u32;
    let attribute_count: u32 = buffers
        .iter()
        .map(|b| b.attributes.len() as u32)
        .sum::<u32>();
    let max_buffers = limits.max_vertex_buffers();
    let max_attributes = limits.max_vertex_attributes();
    if buffer_count > max_buffers || attribute_count > max_attributes {
        return Err(ReflectError::VertexLayoutExceedsLimit {
            buffers: buffer_count,
            attributes: attribute_count,
            max_buffers,
            max_attributes,
        });
    }
    Ok(())
}

/// Validates that `@group(2)` matches the per-draw storage slab (single binding, 256-byte element stride).
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
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: true,
            min_binding_size: Some(n),
        } if n.get() == PER_DRAW_UNIFORM_STRIDE as u64 => Ok(()),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: 0,
            reason:
                "expected var<storage, read> array with dynamic offset and min_binding_size 256"
                    .into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::HashMap;

    fn synthetic_limits(max_samplers: u32, max_textures: u32) -> crate::gpu::GpuLimits {
        crate::gpu::GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_bindings_per_bind_group: 100,
                max_samplers_per_shader_stage: max_samplers,
                max_sampled_textures_per_shader_stage: max_textures,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    fn reflected_layout_with_material_entries(
        material_entries: Vec<wgpu::BindGroupLayoutEntry>,
    ) -> ReflectedRasterLayout {
        ReflectedRasterLayout {
            layout_fingerprint: 0,
            material_entries,
            per_draw_entries: Vec::new(),
            material_uniform: None,
            material_group1_names: HashMap::new(),
            vs_vertex_inputs: Vec::new(),
            vs_max_vertex_location: None,
            uses_scene_depth_snapshot: false,
            uses_scene_color_snapshot: false,
            requires_intersection_pass: false,
        }
    }

    fn fragment_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }

    fn fragment_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    fn fragment_sampler_entries(count: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        (0..count).map(fragment_sampler_entry).collect()
    }

    fn fragment_texture_entries(count: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        (0..count).map(fragment_texture_entry).collect()
    }

    #[test]
    fn reflect_null_default_embedded() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("null_default").expect("stem");
        let r = reflect_raster_material_wgsl(wgsl).expect("reflect");
        assert!(r.material_entries.is_empty());
        validate_per_draw_group2(&r.per_draw_entries).expect("per_draw");
        assert_ne!(r.layout_fingerprint, 0);
        assert_eq!(
            r.vs_max_vertex_location,
            Some(1),
            "null fallback: position + normal only"
        );
    }

    #[test]
    fn reflects_embedded_pass_vertex_entries_without_vs_main() -> Result<(), ReflectError> {
        let stem = "furfx-3.0-10layer_default";
        let wgsl = crate::embedded_shaders::embedded_target_wgsl(stem)
            .ok_or(ReflectError::EmbeddedTargetMissing(stem))?;
        let passes = crate::embedded_shaders::embedded_target_passes(stem);
        let vertex_entries = passes
            .iter()
            .map(|pass| pass.vertex_entry)
            .collect::<Vec<_>>();
        let reflected = reflect_raster_material_wgsl_with_vertex_entries(wgsl, &vertex_entries)?;

        assert!(reflected.vs_vertex_inputs.contains(&ReflectedVertexInput {
            location: 2,
            format: ReflectedVertexInputFormat::Float32x2,
        }));
        assert!(reflected.vs_vertex_inputs.contains(&ReflectedVertexInput {
            location: 4,
            format: ReflectedVertexInputFormat::Float32x4,
        }));
        Ok(())
    }

    #[test]
    fn full_pipeline_sampler_count_includes_frame_group0() {
        let frame_entries = crate::gpu::frame_bind_group_layout_entries();
        let limits = synthetic_limits(16, 64);
        let layout = reflected_layout_with_material_entries(fragment_sampler_entries(14));

        validate_layout_against_limits(&layout, &frame_entries, &limits).expect("14 + 2 samplers");
    }

    #[test]
    fn full_pipeline_sampler_count_rejects_frame_group0_overflow() {
        let frame_entries = crate::gpu::frame_bind_group_layout_entries();
        let limits = synthetic_limits(16, 64);
        let layout = reflected_layout_with_material_entries(fragment_sampler_entries(15));

        let err = validate_layout_against_limits(&layout, &frame_entries, &limits)
            .expect_err("15 + 2 samplers should exceed a 16-sampler stage limit");
        assert!(matches!(
            err,
            ReflectError::ExceedsSamplersPerStage {
                stage: "fragment",
                count: 17,
                max: 16
            }
        ));
    }

    #[test]
    fn full_pipeline_sampled_texture_count_includes_frame_group0() {
        let frame_entries = crate::gpu::frame_bind_group_layout_entries();
        let limits = synthetic_limits(64, 8);
        let layout = reflected_layout_with_material_entries(fragment_texture_entries(2));

        validate_layout_against_limits(&layout, &frame_entries, &limits)
            .expect("2 + 6 sampled textures");
    }

    #[test]
    fn full_pipeline_sampled_texture_count_rejects_frame_group0_overflow() {
        let frame_entries = crate::gpu::frame_bind_group_layout_entries();
        let limits = synthetic_limits(64, 8);
        let layout = reflected_layout_with_material_entries(fragment_texture_entries(3));

        let err = validate_layout_against_limits(&layout, &frame_entries, &limits)
            .expect_err("3 + 6 sampled textures should exceed an 8-texture stage limit");
        assert!(matches!(
            err,
            ReflectError::ExceedsSampledTexturesPerStage {
                stage: "fragment",
                count: 9,
                max: 8
            }
        ));
    }

    /// Every composed `shaders/target/*.wgsl` must declare the full frame globals `@group(0)`
    /// contract; naga-oil strips unused imports, so a material that omits cluster buffer references
    /// can fail at runtime during pipeline creation unless this test catches it.
    #[test]
    fn reflect_all_embedded_material_targets_match_frame_group0() -> Result<(), ReflectError> {
        for stem in crate::embedded_shaders::COMPILED_MATERIAL_STEMS {
            let wgsl = crate::embedded_shaders::embedded_target_wgsl(stem)
                .ok_or(ReflectError::EmbeddedTargetMissing(stem))?;
            reflect_raster_material_wgsl(wgsl)?;
        }
        Ok(())
    }

    #[test]
    fn reflect_group1_material_uniforms_use_dynamic_offsets() -> Result<(), ReflectError> {
        for stem in crate::embedded_shaders::COMPILED_MATERIAL_STEMS {
            let wgsl = crate::embedded_shaders::embedded_target_wgsl(stem)
                .ok_or(ReflectError::EmbeddedTargetMissing(stem))?;
            let reflected = reflect_raster_material_wgsl(wgsl)?;
            let Some(uniform) = reflected.material_uniform.as_ref() else {
                continue;
            };
            let entry = reflected
                .material_entries
                .iter()
                .find(|entry| entry.binding == uniform.binding)
                .expect("reflected material uniform entry");
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    ..
                } => {}
                _ => panic!(
                    "{stem}: group(1) binding({}) material uniform must use a dynamic offset",
                    uniform.binding
                ),
            }
        }
        Ok(())
    }
}
