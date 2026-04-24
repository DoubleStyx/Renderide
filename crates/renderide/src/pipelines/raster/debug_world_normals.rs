//! Debug mesh material: world-space normals as RGB (`shaders/target/debug_world_normals_*.wgsl`).

use crate::embedded_shaders;
use crate::materials::raster_pipeline::{
    create_reflective_raster_mesh_forward_pipeline, ReflectiveRasterMeshForwardPipelineDesc,
};
use crate::materials::PipelineBuildError;
use crate::materials::{
    reflect_raster_material_wgsl, validate_per_draw_group2, MaterialPipelineDesc,
    MaterialRenderState,
};
use crate::pipelines::ShaderPermutation;

/// [`ShaderPermutation`] for multiview WGSL (`debug_world_normals_multiview` target stem).
pub const SHADER_PERM_MULTIVIEW_STEREO: ShaderPermutation = ShaderPermutation(1);

/// World-normal debug visualization for decomposed position/normal vertex streams.
pub struct DebugWorldNormalsFamily;

impl DebugWorldNormalsFamily {
    /// `@group(2)` per-draw storage layout for [`crate::backend::PerDrawResources`].
    ///
    /// Matches naga reflection of the embedded `debug_world_normals_default` target (same `@group(2)`
    /// as the multiview variant).
    pub fn per_draw_bind_group_layout(
        device: &wgpu::Device,
    ) -> Result<wgpu::BindGroupLayout, PipelineBuildError> {
        let wgsl = embedded_shaders::embedded_target_wgsl("debug_world_normals_default").ok_or(
            PipelineBuildError::MissingEmbeddedShader("debug_world_normals_default".to_string()),
        )?;
        let r = reflect_raster_material_wgsl(wgsl)?;
        validate_per_draw_group2(&r.per_draw_entries)?;
        Ok(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("debug_world_normals_per_draw"),
                entries: &r.per_draw_entries,
            }),
        )
    }

    fn target_stem(permutation: ShaderPermutation) -> &'static str {
        if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
            "debug_world_normals_multiview"
        } else {
            "debug_world_normals_default"
        }
    }
}

pub(crate) fn build_debug_world_normals_wgsl(
    permutation: ShaderPermutation,
) -> Result<String, PipelineBuildError> {
    let stem = DebugWorldNormalsFamily::target_stem(permutation);
    let wgsl = embedded_shaders::embedded_target_wgsl(stem)
        .ok_or_else(|| PipelineBuildError::MissingEmbeddedShader(stem.to_string()))?;
    Ok(wgsl.to_string())
}

pub(crate) fn create_debug_world_normals_render_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
) -> Result<wgpu::RenderPipeline, PipelineBuildError> {
    create_reflective_raster_mesh_forward_pipeline(
        device,
        module,
        desc,
        wgsl_source,
        "debug_world_normals_material",
        ReflectiveRasterMeshForwardPipelineDesc {
            include_uv_vertex_buffer: false,
            include_color_vertex_buffer: false,
            use_alpha_blending: false,
            depth_write_enabled: true,
            render_state: MaterialRenderState::default(),
        },
    )
}

#[cfg(test)]
mod wgsl_dispatch_tests {
    use super::{
        build_debug_world_normals_wgsl, DebugWorldNormalsFamily, SHADER_PERM_MULTIVIEW_STEREO,
    };
    use crate::pipelines::ShaderPermutation;

    /// Default permutation picks the `debug_world_normals_default` embedded stem and yields a
    /// non-empty WGSL source.
    #[test]
    fn default_permutation_selects_default_stem() {
        assert_eq!(
            DebugWorldNormalsFamily::target_stem(ShaderPermutation(0)),
            "debug_world_normals_default"
        );
        let wgsl = build_debug_world_normals_wgsl(ShaderPermutation(0)).expect("default wgsl");
        assert!(!wgsl.is_empty());
    }

    /// Multiview permutation picks the `debug_world_normals_multiview` stem and differs from the
    /// default permutation's WGSL.
    #[test]
    fn multiview_permutation_selects_multiview_stem() {
        assert_eq!(
            DebugWorldNormalsFamily::target_stem(SHADER_PERM_MULTIVIEW_STEREO),
            "debug_world_normals_multiview"
        );
        let default_wgsl =
            build_debug_world_normals_wgsl(ShaderPermutation(0)).expect("default wgsl");
        let multiview_wgsl =
            build_debug_world_normals_wgsl(SHADER_PERM_MULTIVIEW_STEREO).expect("multiview wgsl");
        assert_ne!(default_wgsl, multiview_wgsl);
    }

    /// Unknown permutation bits fall through to the default stem.
    #[test]
    fn unknown_permutation_falls_back_to_default_stem() {
        assert_eq!(
            DebugWorldNormalsFamily::target_stem(ShaderPermutation(0xDEAD_BEEF)),
            "debug_world_normals_default"
        );
    }
}
