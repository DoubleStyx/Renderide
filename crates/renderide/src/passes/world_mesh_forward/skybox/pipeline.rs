//! Pipeline classification and construction for the world-mesh forward skybox.
//!
//! Pipelines themselves are cached by [`super::SkyboxRenderer`] in `material_pipelines` and
//! `clear_pipelines` maps; this module owns the *keys* used for those caches and the wgpu
//! `RenderPipeline` factory invoked on cache miss.

use super::super::state::WorldMeshForwardPipelineState;

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
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
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
    })
}

#[cfg(test)]
mod tests {
    use super::SkyboxFamily;

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
}
