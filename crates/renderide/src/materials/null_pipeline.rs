//! Null fallback raster material: object-space black/grey checkerboard (`shaders/target/null_*.wgsl`).
//!
//! Used when the host shader has no embedded target or when pipeline build fails for the requested kind.
//! Object-space projection is robust against missing or malformed mesh UVs, mirroring the role of
//! `Null.shader` in `Resonite.UnityShaders`.

use crate::embedded_shaders;
use crate::materials::PipelineBuildError;
use crate::materials::raster_pipeline::{
    ReflectiveRasterMeshForwardPipelineDesc, create_reflective_raster_mesh_forward_pipeline,
};
use crate::materials::shader_permutation::{SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation};
use crate::materials::{
    MaterialPipelineDesc, MaterialRenderState, RasterFrontFace, RasterPrimitiveTopology,
    reflect_raster_material_wgsl, validate_per_draw_group2,
};

/// Null/fallback material family for decomposed position/normal vertex streams.
pub struct NullFamily;

impl NullFamily {
    /// `@group(2)` per-draw storage layout for [`crate::backend::PerDrawResources`].
    ///
    /// Matches naga reflection of the embedded `null_default` target (same `@group(2)` as the
    /// multiview variant).
    pub fn per_draw_bind_group_layout(
        device: &wgpu::Device,
    ) -> Result<wgpu::BindGroupLayout, PipelineBuildError> {
        let wgsl = embedded_shaders::embedded_target_wgsl("null_default").ok_or(
            PipelineBuildError::MissingEmbeddedShader("null_default".to_string()),
        )?;
        let r = reflect_raster_material_wgsl(wgsl)?;
        validate_per_draw_group2(&r.per_draw_entries)?;
        Ok(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("null_per_draw"),
                entries: &r.per_draw_entries,
            }),
        )
    }

    /// Pick the embedded WGSL stem for the given permutation (default vs. multiview).
    fn target_stem(permutation: ShaderPermutation) -> &'static str {
        if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
            "null_multiview"
        } else {
            "null_default"
        }
    }
}

/// Build the WGSL source for the null material in the requested permutation.
pub(crate) fn build_null_wgsl(
    permutation: ShaderPermutation,
) -> Result<String, PipelineBuildError> {
    let stem = NullFamily::target_stem(permutation);
    let wgsl = embedded_shaders::embedded_target_wgsl(stem)
        .ok_or_else(|| PipelineBuildError::MissingEmbeddedShader(stem.to_string()))?;
    Ok(wgsl.to_string())
}

/// Construct the wgpu render pipeline for the null material from a precompiled shader module.
pub(crate) fn create_null_render_pipeline(
    device: &wgpu::Device,
    limits: &crate::gpu::GpuLimits,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
) -> Result<wgpu::RenderPipeline, PipelineBuildError> {
    create_reflective_raster_mesh_forward_pipeline(
        device,
        limits,
        module,
        desc,
        wgsl_source,
        "null_material",
        ReflectiveRasterMeshForwardPipelineDesc {
            include_uv_vertex_buffer: false,
            include_color_vertex_buffer: false,
            include_uv1_vertex_buffer: false,
            use_alpha_blending: false,
            depth_write_enabled: true,
            render_state: MaterialRenderState::default(),
            front_face,
            primitive_topology,
        },
    )
}

#[cfg(test)]
mod wgsl_dispatch_tests {
    use super::{NullFamily, SHADER_PERM_MULTIVIEW_STEREO, build_null_wgsl};
    use crate::materials::shader_permutation::ShaderPermutation;

    /// Default permutation picks the `null_default` embedded stem and yields a non-empty WGSL source.
    #[test]
    fn default_permutation_selects_default_stem() {
        assert_eq!(
            NullFamily::target_stem(ShaderPermutation(0)),
            "null_default"
        );
        let wgsl = build_null_wgsl(ShaderPermutation(0)).expect("default wgsl");
        assert!(!wgsl.is_empty());
    }

    /// Multiview permutation picks the `null_multiview` stem and differs from the default permutation's WGSL.
    #[test]
    fn multiview_permutation_selects_multiview_stem() {
        assert_eq!(
            NullFamily::target_stem(SHADER_PERM_MULTIVIEW_STEREO),
            "null_multiview"
        );
        let default_wgsl = build_null_wgsl(ShaderPermutation(0)).expect("default wgsl");
        let multiview_wgsl = build_null_wgsl(SHADER_PERM_MULTIVIEW_STEREO).expect("multiview wgsl");
        assert_ne!(default_wgsl, multiview_wgsl);
    }

    /// Unknown permutation bits fall through to the default stem.
    #[test]
    fn unknown_permutation_falls_back_to_default_stem() {
        assert_eq!(
            NullFamily::target_stem(ShaderPermutation(0xDEAD_BEEF)),
            "null_default"
        );
    }
}
