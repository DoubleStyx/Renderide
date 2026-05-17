//! Compute pipeline + sampler cache shared by the bake cache and the standalone convolver.

use std::sync::Arc;

use super::errors::SkyboxIblBakeError;
use super::pipeline::{
    ComputePipeline, analytic_layout_entries, downsample_layout_entries, ensure_pipeline,
    mip0_input_layout_entries,
};
use super::sampler::create_linear_clamp_sampler;

/// Identifier for one of the IBL compute pipeline slots.
#[derive(Clone, Copy, Debug)]
pub(super) enum PipelineSlot {
    /// Parameter-driven mip-0 producer used for constant-color probe sources.
    Analytic,
    /// Cubemap mip-0 producer.
    Cube,
    /// Source-pyramid downsample pass.
    Downsample,
    /// Seam stitch pass applied after each generated mip.
    Stitch,
    /// GGX convolve pass.
    Convolve,
}

impl PipelineSlot {
    /// Shader stem used by [`ensure_pipeline`] for this slot.
    fn shader_stem(self) -> &'static str {
        match self {
            Self::Analytic => "skybox_bake_params",
            Self::Cube => "skybox_mip0_cube_params",
            Self::Downsample => "skybox_ibl_downsample",
            Self::Stitch => "skybox_ibl_stitch",
            Self::Convolve => "skybox_ibl_convolve_params",
        }
    }
}

/// Lazily-built pipelines and cached sampler used by every IBL bake and convolve path.
#[derive(Default)]
pub(super) struct PipelineStore {
    analytic: Option<ComputePipeline>,
    cube: Option<ComputePipeline>,
    downsample: Option<ComputePipeline>,
    stitch: Option<ComputePipeline>,
    convolve: Option<ComputePipeline>,
    sampler: Option<Arc<wgpu::Sampler>>,
}

impl PipelineStore {
    /// Builds (or returns the cached) compute pipeline for one slot.
    pub(super) fn ensure(
        &mut self,
        slot: PipelineSlot,
        device: &wgpu::Device,
    ) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        let stem = slot.shader_stem();
        match slot {
            PipelineSlot::Analytic => {
                ensure_pipeline(&mut self.analytic, device, stem, &analytic_layout_entries())
            }
            PipelineSlot::Cube => ensure_pipeline(
                &mut self.cube,
                device,
                stem,
                &mip0_input_layout_entries(wgpu::TextureViewDimension::D2Array),
            ),
            PipelineSlot::Downsample => ensure_pipeline(
                &mut self.downsample,
                device,
                stem,
                &downsample_layout_entries(),
            ),
            PipelineSlot::Stitch => {
                ensure_pipeline(&mut self.stitch, device, stem, &downsample_layout_entries())
            }
            PipelineSlot::Convolve => ensure_pipeline(
                &mut self.convolve,
                device,
                stem,
                &mip0_input_layout_entries(wgpu::TextureViewDimension::D2Array),
            ),
        }
    }

    /// Eagerly builds every pipeline used by [`SkyboxIblCache`](super::cache::SkyboxIblCache) bakes.
    pub(super) fn ensure_all(&mut self, device: &wgpu::Device) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::ensure_pipelines");
        for slot in [
            PipelineSlot::Analytic,
            PipelineSlot::Cube,
            PipelineSlot::Downsample,
            PipelineSlot::Stitch,
            PipelineSlot::Convolve,
        ] {
            let _ = self.ensure(slot, device)?;
        }
        Ok(())
    }

    /// Returns the cached pipeline for a slot, or a missing-shader error.
    pub(super) fn get(&self, slot: PipelineSlot) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        let stem = slot.shader_stem();
        let slot_ref = match slot {
            PipelineSlot::Analytic => self.analytic.as_ref(),
            PipelineSlot::Cube => self.cube.as_ref(),
            PipelineSlot::Downsample => self.downsample.as_ref(),
            PipelineSlot::Stitch => self.stitch.as_ref(),
            PipelineSlot::Convolve => self.convolve.as_ref(),
        };
        slot_ref.ok_or(SkyboxIblBakeError::MissingShader(stem))
    }

    /// Returns the cached linear/clamp sampler used for every IBL input, building it on first use.
    pub(super) fn ensure_sampler(&mut self, device: &wgpu::Device) -> Arc<wgpu::Sampler> {
        self.sampler
            .get_or_insert_with(|| {
                Arc::new(create_linear_clamp_sampler(
                    device,
                    "skybox_ibl_input_sampler",
                ))
            })
            .clone()
    }
}
