//! Standalone GGX convolver for caller-owned cubemap textures.

use std::sync::Arc;

use crate::gpu::GpuContext;
use crate::profiling::GpuProfilerHandle;

use super::encode::{
    ConvolveEncodeContext, DownsampleEncodeContext, StitchEncodeContext, encode_convolve_mips,
    encode_downsample_mips, encode_stitch_mip,
};
use super::errors::SkyboxIblConvolveError;
use super::key::source_max_lod;
use super::pipeline_store::{PipelineSlot, PipelineStore};
use super::resources::{
    PendingBakeResources, copy_cube_mip0, create_full_array_sample_view, create_ibl_cube,
};

/// Resources produced while encoding convolve passes and retained until submit completion.
pub(crate) struct SkyboxIblConvolveResources {
    _resources: PendingBakeResources,
    _source_sample_view: Arc<wgpu::TextureView>,
    _sampler: Arc<wgpu::Sampler>,
}

/// Texture set used while convolving a caller-owned cubemap.
struct ConvolverTextures {
    /// Stitched source radiance mip pyramid.
    source_cube: super::resources::IblCubeTexture,
    /// Scratch target for source mip generation before stitching.
    source_scratch_cube: super::resources::IblCubeTexture,
    /// Scratch target for filtered mip generation before stitching.
    filtered_scratch_cube: super::resources::IblCubeTexture,
    /// Full-mip 2D-array view of [`Self::source_cube`].
    source_sample_view: Arc<wgpu::TextureView>,
}

impl ConvolverTextures {
    /// Allocates transient textures and a sample view for one convolve operation.
    fn create(device: &wgpu::Device, face_size: u32, mip_levels: u32) -> Self {
        let source_cube = create_ibl_cube(
            device,
            "skybox_ibl_existing_source_cube",
            face_size,
            mip_levels,
        );
        let source_scratch_cube = create_ibl_cube(
            device,
            "skybox_ibl_existing_source_scratch_cube",
            face_size,
            mip_levels,
        );
        let filtered_scratch_cube = create_ibl_cube(
            device,
            "skybox_ibl_existing_filtered_scratch_cube",
            face_size,
            mip_levels,
        );
        let source_sample_view = Arc::new(create_full_array_sample_view(
            source_cube.texture.as_ref(),
            mip_levels,
        ));
        Self {
            source_cube,
            source_scratch_cube,
            filtered_scratch_cube,
            source_sample_view,
        }
    }

    /// Retains transient textures and views until the caller's submit completes.
    fn retain_transient(&self, resources: &mut PendingBakeResources) {
        resources.textures.push(self.source_cube.texture.clone());
        resources
            .textures
            .push(self.source_scratch_cube.texture.clone());
        resources
            .textures
            .push(self.filtered_scratch_cube.texture.clone());
        resources.source_sample_view = Some(self.source_sample_view.clone());
    }
}

/// Minimal GGX convolver for caller-owned cubemap textures.
#[derive(Default)]
pub(crate) struct SkyboxIblConvolver {
    pipelines: PipelineStore,
}

impl SkyboxIblConvolver {
    /// Creates an empty convolver with lazily-built GPU resources.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Encodes GGX convolve passes for mips `1..mip_levels` of `texture`.
    pub(crate) fn encode_existing_cube_mips(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        face_size: u32,
        mip_levels: u32,
        profiler: Option<&GpuProfilerHandle>,
    ) -> Result<SkyboxIblConvolveResources, SkyboxIblConvolveError> {
        profiling::scope!("skybox_ibl::encode_existing_cube_mips");
        let sampler = self.pipelines.ensure_sampler(gpu.device());
        self.pipelines
            .ensure(PipelineSlot::Convolve, gpu.device())
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_convolve_params"))?;
        self.pipelines
            .ensure(PipelineSlot::Downsample, gpu.device())
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_downsample"))?;
        self.pipelines
            .ensure(PipelineSlot::Stitch, gpu.device())
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_stitch"))?;
        let convolve_pipeline = self
            .pipelines
            .get(PipelineSlot::Convolve)
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_convolve_params"))?;
        let downsample_pipeline = self
            .pipelines
            .get(PipelineSlot::Downsample)
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_downsample"))?;
        let stitch_pipeline = self
            .pipelines
            .get(PipelineSlot::Stitch)
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_stitch"))?;

        let textures = ConvolverTextures::create(gpu.device(), face_size, mip_levels);
        let mut resources = PendingBakeResources::default();
        textures.retain_transient(&mut resources);
        copy_cube_mip0(
            encoder,
            texture,
            textures.source_scratch_cube.texture.as_ref(),
            face_size,
            profiler,
        );
        encode_stitch_mip(
            StitchEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: stitch_pipeline,
                src_texture: textures.source_scratch_cube.texture.as_ref(),
                dst_texture: textures.source_cube.texture.as_ref(),
                mip: 0,
                dst_size: face_size,
                profiler,
                label: "skybox_ibl stitch existing source mip0",
                profiler_label: "skybox_ibl::stitch_existing_source_mip0".to_string(),
            },
            &mut resources,
        );
        encode_stitch_mip(
            StitchEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: stitch_pipeline,
                src_texture: textures.source_cube.texture.as_ref(),
                dst_texture: texture,
                mip: 0,
                dst_size: face_size,
                profiler,
                label: "skybox_ibl stitch existing output mip0",
                profiler_label: "skybox_ibl::stitch_existing_output_mip0".to_string(),
            },
            &mut resources,
        );
        encode_downsample_mips(
            DownsampleEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: downsample_pipeline,
                stitch_pipeline,
                texture: textures.source_cube.texture.as_ref(),
                scratch_texture: textures.source_scratch_cube.texture.as_ref(),
                face_size,
                mip_levels,
                profiler,
            },
            &mut resources,
        );
        encode_convolve_mips(
            ConvolveEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: convolve_pipeline,
                stitch_pipeline,
                texture,
                scratch_texture: textures.filtered_scratch_cube.texture.as_ref(),
                src_view: textures.source_sample_view.as_ref(),
                sampler: sampler.as_ref(),
                face_size,
                mip_levels,
                src_max_lod: source_max_lod(mip_levels),
                profiler,
            },
            &mut resources,
        );
        Ok(SkyboxIblConvolveResources {
            _resources: resources,
            _source_sample_view: textures.source_sample_view,
            _sampler: sampler,
        })
    }
}
