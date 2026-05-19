//! Pipeline classification and construction for the world-mesh forward skybox.
//!
//! Pipelines themselves are cached by [`super::SkyboxRenderer`] in `material_pipelines` and
//! `clear_pipelines` maps; this module owns the *keys* used for those caches and the wgpu
//! `RenderPipeline` factory invoked on cache miss.

use super::super::state::WorldMeshForwardPipelineState;
use crate::gpu::MAIN_FORWARD_DEPTH_COMPARE;
use crate::materials::MaterialRenderState;

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

    /// Vertex count used by the fullscreen background draw.
    pub(super) const fn draw_vertex_count(self) -> u32 {
        match self {
            Self::Projection360 | Self::Gradient => 3,
            Self::Procedural => 3,
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

    /// Resolves skybox depth state for a material family.
    pub(super) fn for_family(family: SkyboxFamily, render_state: MaterialRenderState) -> Self {
        match family {
            SkyboxFamily::Projection360 => Self {
                write_enabled: render_state.depth_write(false),
                compare: render_state.depth_compare(MAIN_FORWARD_DEPTH_COMPARE),
            },
            SkyboxFamily::Gradient | SkyboxFamily::Procedural => Self::fixed_background(),
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
    /// Depth state selected for this skybox material.
    pub(super) depth: SkyboxDepthState,
}

/// Cached solid-color background pipeline key.
pub(super) type ClearPipelineKey = SkyboxPipelineTarget;

/// Creates a fullscreen skybox/background render pipeline compatible with the world pass.
pub(super) fn create_skybox_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    target: SkyboxPipelineTarget,
    depth: SkyboxDepthState,
) -> wgpu::RenderPipeline {
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
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

#[cfg(test)]
mod tests {
    use super::{SkyboxDepthState, SkyboxFamily};
    use crate::gpu::MAIN_FORWARD_DEPTH_COMPARE;
    use crate::materials::{MaterialDepthCompareOverride, MaterialRenderState};

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
    fn procedural_skybox_draws_fullscreen_quad() {
        assert_eq!(SkyboxFamily::Projection360.draw_vertex_count(), 3);
        assert_eq!(SkyboxFamily::Gradient.draw_vertex_count(), 3);
        assert_eq!(SkyboxFamily::Procedural.draw_vertex_count(), 6);
    }

    #[test]
    fn fixed_background_skybox_depth_uses_reverse_z_compare_without_writes() {
        let depth = SkyboxDepthState::fixed_background();

        assert!(!depth.write_enabled);
        assert_eq!(depth.compare, MAIN_FORWARD_DEPTH_COMPARE);
    }

    #[test]
    fn projection360_skybox_honors_zwrite_override() {
        let write_on = SkyboxDepthState::for_family(
            SkyboxFamily::Projection360,
            MaterialRenderState {
                depth_write: Some(true),
                ..MaterialRenderState::default()
            },
        );
        let write_off = SkyboxDepthState::for_family(
            SkyboxFamily::Projection360,
            MaterialRenderState {
                depth_write: Some(false),
                ..MaterialRenderState::default()
            },
        );

        assert!(write_on.write_enabled);
        assert!(!write_off.write_enabled);
    }

    #[test]
    fn projection360_skybox_honors_ztest_shaderlab_mapping() {
        let depth = SkyboxDepthState::for_family(
            SkyboxFamily::Projection360,
            MaterialRenderState {
                depth_compare: Some(MaterialDepthCompareOverride::HostValue(2)),
                ..MaterialRenderState::default()
            },
        );

        assert_eq!(depth.compare, wgpu::CompareFunction::Greater);
    }

    #[test]
    fn analytic_skyboxes_ignore_material_depth_state() {
        let render_state = MaterialRenderState {
            depth_write: Some(true),
            depth_compare: Some(MaterialDepthCompareOverride::HostValue(6)),
            ..MaterialRenderState::default()
        };

        for family in [SkyboxFamily::Gradient, SkyboxFamily::Procedural] {
            let depth = SkyboxDepthState::for_family(family, render_state);
            assert_eq!(depth, SkyboxDepthState::fixed_background());
        }
    }
}
