//! Pipeline classification and construction for the world-mesh forward skybox.
//!
//! Pipelines themselves are cached by [`super::SkyboxRenderer`] in `material_pipelines` and
//! `clear_pipelines` maps; this module owns the *keys* used for those caches and the wgpu
//! `RenderPipeline` factory invoked on cache miss.

use super::super::state::WorldMeshForwardPipelineState;
use crate::gpu::MAIN_FORWARD_DEPTH_COMPARE;
use crate::materials::MaterialShaderSpecializationKey;

/// Skybox material family supported by the dedicated background draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum SkyboxFamily {
    /// Froox `Projection360Material`.
    Projection360,
    /// Froox `GradientSkyMaterial`.
    Gradient,
    /// Froox `ProceduralSkyMaterial`.
    Procedural,
}

impl SkyboxFamily {
    /// Resolves the supported family from an embedded material stem.
    pub(super) fn from_stem(stem: &str) -> Option<Self> {
        let base = stem
            .strip_suffix("_default")
            .or_else(|| stem.strip_suffix("_multiview"))
            .unwrap_or(stem);
        match base.to_ascii_lowercase().as_str() {
            "projection360" => Some(Self::Projection360),
            "gradientskybox" => Some(Self::Gradient),
            "proceduralskybox" | "proceduralsky" => Some(Self::Procedural),
            _ => None,
        }
    }

    /// Embedded backend shader target for this family and view permutation.
    pub(super) fn shader_target(self, multiview: bool) -> &'static str {
        match (self, multiview) {
            (Self::Projection360, false) => "skybox_projection360_default",
            (Self::Projection360, true) => "skybox_projection360_multiview",
            (Self::Gradient, false) => "skybox_gradientskybox_default",
            (Self::Gradient, true) => "skybox_gradientskybox_multiview",
            (Self::Procedural, false) => "skybox_proceduralskybox_default",
            (Self::Procedural, true) => "skybox_proceduralskybox_multiview",
        }
    }

    /// Vertex buffer layouts required by the skybox family.
    pub(super) const fn vertex_buffer_layouts(
        self,
    ) -> &'static [wgpu::VertexBufferLayout<'static>] {
        match self {
            Self::Procedural => &PROCEDURAL_SKYBOX_VERTEX_BUFFER_LAYOUTS,
            Self::Projection360 | Self::Gradient => &[],
        }
    }
}

/// Depth state used by a fullscreen skybox pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SkyboxDepthState {
    /// Whether the skybox updates the depth buffer.
    pub(super) write_enabled: bool,
    /// Depth compare used for the skybox draw.
    pub(super) compare: wgpu::CompareFunction,
}

impl SkyboxDepthState {
    /// Depth state used by fixed-background skyboxes.
    pub(super) const fn fixed_background() -> Self {
        Self {
            write_enabled: false,
            compare: MAIN_FORWARD_DEPTH_COMPARE,
        }
    }
}

impl Default for SkyboxDepthState {
    fn default() -> Self {
        Self::fixed_background()
    }
}

/// Render-target state that must match the containing skybox render pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SkyboxPipelineTarget {
    /// HDR scene-color format.
    color_format: wgpu::TextureFormat,
    /// Depth-stencil attachment format used by the containing world pass.
    depth_stencil_format: Option<wgpu::TextureFormat>,
    /// Raster sample count.
    sample_count: u32,
    /// Whether the target uses stereo multiview.
    pub(super) multiview: bool,
}

impl SkyboxPipelineTarget {
    /// Builds the target descriptor from the prepared world-mesh forward pipeline state.
    pub(super) fn from_forward_state(pipeline_state: &WorldMeshForwardPipelineState) -> Self {
        Self {
            color_format: pipeline_state.pass_desc.surface_format,
            depth_stencil_format: pipeline_state.pass_desc.depth_stencil_format,
            sample_count: pipeline_state.pass_desc.sample_count,
            multiview: pipeline_state.use_multiview,
        }
    }
}

/// Cached material skybox pipeline key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SkyboxPipelineKey {
    /// Supported sky material family.
    pub(super) family: SkyboxFamily,
    /// Render-target state required by wgpu pipeline/pass compatibility.
    pub(super) target: SkyboxPipelineTarget,
    /// Renderer-local shader specialization constants for material keyword branches.
    pub(super) shader_specialization: MaterialShaderSpecializationKey,
}

/// Cached solid-color background pipeline key.
pub(super) type ClearPipelineKey = SkyboxPipelineTarget;

/// Inputs for constructing a fullscreen skybox/background render pipeline.
pub(super) struct SkyboxPipelineBuildDesc<'a> {
    /// Pipeline and shader label.
    pub(super) label: &'a str,
    /// Compiled shader module.
    pub(super) shader: &'a wgpu::ShaderModule,
    /// Full composed WGSL source for source-mangled pipeline constants.
    pub(super) wgsl_source: &'a str,
    /// Pipeline layout matching the active skybox family.
    pub(super) layout: &'a wgpu::PipelineLayout,
    /// Vertex buffer layouts required by the skybox family.
    pub(super) vertex_buffer_layouts: &'static [wgpu::VertexBufferLayout<'static>],
    /// Render target state required by wgpu pipeline/pass compatibility.
    pub(super) target: SkyboxPipelineTarget,
    /// Depth state used for the skybox draw.
    pub(super) depth: SkyboxDepthState,
    /// Renderer-local shader specialization constants for material keyword branches.
    pub(super) shader_specialization: MaterialShaderSpecializationKey,
}

/// Creates a fullscreen skybox/background render pipeline compatible with the world pass.
pub(super) fn create_skybox_pipeline(
    device: &wgpu::Device,
    desc: SkyboxPipelineBuildDesc<'_>,
) -> wgpu::RenderPipeline {
    let SkyboxPipelineBuildDesc {
        label,
        shader,
        wgsl_source,
        layout,
        vertex_buffer_layouts,
        target,
        depth,
        shader_specialization,
    } = desc;
    let shader_specialization_constants =
        shader_specialization.pipeline_constants_for_wgsl_source(wgsl_source);
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: shader_specialization_constants.as_slice(),
                ..Default::default()
            },
            buffers: vertex_buffer_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: shader_specialization_constants.as_slice(),
                ..Default::default()
            },
            targets: &[Some(wgpu::ColorTargetState {
                format: target.color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: target
            .depth_stencil_format
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(depth.write_enabled),
                depth_compare: Some(depth.compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        multisample: wgpu::MultisampleState {
            count: target.sample_count.max(1),
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: target
            .multiview
            .then(|| std::num::NonZeroU32::new(3))
            .flatten(),
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "passes::skybox_pipeline");
    pipeline
}

const PROCEDURAL_SKYBOX_VERTEX_BUFFER_LAYOUTS: [wgpu::VertexBufferLayout<'static>; 1] =
    [wgpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        }],
    }];

#[cfg(test)]
mod tests {
    use super::{SkyboxDepthState, SkyboxFamily};
    use crate::gpu::MAIN_FORWARD_DEPTH_COMPARE;
    use crate::materials::MaterialShaderSpecializationKey;

    #[test]
    fn skybox_family_resolves_supported_stems() {
        assert_eq!(
            SkyboxFamily::from_stem("projection360_default"),
            Some(SkyboxFamily::Projection360)
        );
        assert_eq!(
            SkyboxFamily::from_stem("gradientskybox_default"),
            Some(SkyboxFamily::Gradient)
        );
        assert_eq!(
            SkyboxFamily::from_stem("proceduralskybox_multiview"),
            Some(SkyboxFamily::Procedural)
        );
        assert_eq!(SkyboxFamily::from_stem("pbsmetallic_default"), None);
    }

    #[test]
    fn material_skybox_uses_multiview_shader_targets() {
        assert_eq!(
            SkyboxFamily::Gradient.shader_target(false),
            "skybox_gradientskybox_default"
        );
        assert_eq!(
            SkyboxFamily::Gradient.shader_target(true),
            "skybox_gradientskybox_multiview"
        );
    }

    #[test]
    fn only_procedural_skybox_uses_a_vertex_buffer_layout() {
        assert!(
            SkyboxFamily::Projection360
                .vertex_buffer_layouts()
                .is_empty()
        );
        assert!(SkyboxFamily::Gradient.vertex_buffer_layouts().is_empty());
        assert_eq!(SkyboxFamily::Procedural.vertex_buffer_layouts().len(), 1);
    }

    #[test]
    fn fixed_background_skybox_depth_uses_reverse_z_compare_without_writes() {
        let depth = SkyboxDepthState::fixed_background();

        assert!(!depth.write_enabled);
        assert_eq!(depth.compare, MAIN_FORWARD_DEPTH_COMPARE);
    }

    #[test]
    fn material_skybox_pipeline_key_is_independent_of_depth_state() {
        let target = super::SkyboxPipelineTarget {
            color_format: wgpu::TextureFormat::Rgba16Float,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
            multiview: false,
        };

        let key = super::SkyboxPipelineKey {
            family: SkyboxFamily::Projection360,
            target,
            shader_specialization: MaterialShaderSpecializationKey::disabled(),
        };

        assert_eq!(key.family, SkyboxFamily::Projection360);
        assert_eq!(key.target, target);
    }
}
