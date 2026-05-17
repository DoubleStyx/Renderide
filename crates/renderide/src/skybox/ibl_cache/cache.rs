//! Owns IBL bakes for prefiltered specular reflection cubemaps.

use std::sync::Arc;

use hashbrown::HashMap;

use crate::backend::gpu_jobs::{GpuJobResources, GpuSubmitJobTracker, SubmittedGpuJob};
use crate::gpu::GpuContext;
use crate::profiling::GpuProfilerHandle;
use crate::skybox::specular::{SkyboxIblSource, solid_color_params};

use super::encode::{
    AnalyticEncodeContext, ConvolveEncodeContext, CubeEncodeContext, DownsampleEncodeContext,
    RuntimeCubeEncodeContext, StitchEncodeContext, encode_analytic_mip0, encode_convolve_mips,
    encode_cube_mip0, encode_downsample_mips, encode_runtime_cube_mip0, encode_stitch_mip,
};
use super::errors::SkyboxIblBakeError;
use super::key::{SkyboxIblKey, mip_levels_for_edge, source_max_lod};
use super::pipeline_store::{PipelineSlot, PipelineStore};
use super::resources::{
    PendingBake, PendingBakeResources, PrefilteredCube, copy_cube_mip0,
    create_full_array_sample_view, create_ibl_cube,
};

/// Maximum concurrent in-flight bakes; matches the analytic-only ceiling we used previously.
const MAX_IN_FLIGHT_IBL_BAKES: usize = 2;
/// Tick budget after which a missing submit-completion callback is treated as lost.
const MAX_PENDING_IBL_BAKE_AGE_FRAMES: u32 = 120;

/// Resources required to encode a mip-0 producer for any source variant.
struct SourceMip0EncodeContext<'a> {
    gpu: &'a GpuContext,
    encoder: &'a mut wgpu::CommandEncoder,
    texture: &'a wgpu::Texture,
    face_size: u32,
    sampler: &'a wgpu::Sampler,
    profiler: Option<&'a GpuProfilerHandle>,
}

/// Texture set used by one asynchronous IBL bake.
struct BakeTextures {
    /// Stitched source radiance mip pyramid.
    source_cube: super::resources::IblCubeTexture,
    /// Scratch target for source mip generation before stitching.
    source_scratch_cube: super::resources::IblCubeTexture,
    /// Final prefiltered specular cubemap.
    filtered_cube: super::resources::IblCubeTexture,
    /// Scratch target for filtered mip generation before stitching.
    filtered_scratch_cube: super::resources::IblCubeTexture,
    /// Full-mip 2D-array view of [`Self::source_cube`].
    source_sample_view: Arc<wgpu::TextureView>,
}

impl BakeTextures {
    /// Allocates the textures and source sample view needed by one bake.
    fn create(device: &wgpu::Device, face_size: u32, mip_levels: u32) -> Self {
        let source_cube = create_ibl_cube(device, "skybox_ibl_source_cube", face_size, mip_levels);
        let source_scratch_cube = create_ibl_cube(
            device,
            "skybox_ibl_source_scratch_cube",
            face_size,
            mip_levels,
        );
        let filtered_cube =
            create_ibl_cube(device, "skybox_ibl_filtered_cube", face_size, mip_levels);
        let filtered_scratch_cube = create_ibl_cube(
            device,
            "skybox_ibl_filtered_scratch_cube",
            face_size,
            mip_levels,
        );
        let source_sample_view = Arc::new(create_full_array_sample_view(
            &source_cube.texture,
            mip_levels,
        ));
        Self {
            source_cube,
            source_scratch_cube,
            filtered_cube,
            filtered_scratch_cube,
            source_sample_view,
        }
    }

    /// Retains transient textures and views until the submitted bake completes.
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

/// Owns IBL bakes for prefiltered specular reflection cubemaps.
pub(crate) struct SkyboxIblCache {
    /// Submit-completion tracker for in-flight bakes.
    jobs: GpuSubmitJobTracker<SkyboxIblKey>,
    /// In-flight prefiltered cubes retained until their submit callback fires.
    pending: HashMap<SkyboxIblKey, PendingBake>,
    /// Completed prefiltered cubes for the active skybox key.
    completed: HashMap<SkyboxIblKey, PrefilteredCube>,
    /// Lazily-built compute pipelines and cached input sampler.
    pipelines: PipelineStore,
}

impl Default for SkyboxIblCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SkyboxIblCache {
    /// Creates an empty IBL cache.
    pub(crate) fn new() -> Self {
        Self {
            jobs: GpuSubmitJobTracker::new(MAX_PENDING_IBL_BAKE_AGE_FRAMES),
            pending: HashMap::new(),
            completed: HashMap::new(),
            pipelines: PipelineStore::default(),
        }
    }

    /// Drains submit-completed bakes.
    pub(crate) fn maintain_completed_jobs(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        self.drain_completed_jobs();
    }

    /// Removes completed cubes whose keys are not retained by the caller.
    pub(crate) fn prune_completed_except(&mut self, retain: &hashbrown::HashSet<SkyboxIblKey>) {
        self.completed.retain(|key, _| retain.contains(key));
    }

    /// Removes pending and completed IBL bakes whose keys match `predicate`.
    pub(crate) fn purge_where(
        &mut self,
        mut predicate: impl FnMut(&SkyboxIblKey) -> bool,
    ) -> usize {
        let pending_before = self.pending.len();
        let completed_before = self.completed.len();
        self.pending.retain(|key, _| !predicate(key));
        self.completed.retain(|key, _| !predicate(key));
        self.jobs.retain(|key| !predicate(key));
        pending_before.saturating_sub(self.pending.len())
            + completed_before.saturating_sub(self.completed.len())
    }

    /// Ensures one arbitrary IBL source is scheduled for baking.
    pub(crate) fn ensure_source(
        &mut self,
        gpu: &mut GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
    ) {
        if self.completed.contains_key(&key)
            || self.pending.contains_key(&key)
            || self.jobs.contains_key(&key)
            || self.jobs.len() >= MAX_IN_FLIGHT_IBL_BAKES
        {
            return;
        }
        if let Err(e) = self.schedule_bake(gpu, key, source) {
            logger::warn!("skybox_ibl: bake failed: {e}");
        }
    }

    /// Returns a completed prefiltered cube by key.
    pub(crate) fn completed_cube(&self, key: &SkyboxIblKey) -> Option<&PrefilteredCube> {
        self.completed.get(key)
    }

    /// Promotes submit-completed bakes into the completed cache.
    fn drain_completed_jobs(&mut self) {
        let outcomes = self.jobs.maintain();
        for key in outcomes.completed {
            if let Some(pending) = self.pending.remove(&key) {
                self.completed.insert(key, pending.cube);
            }
        }
        for key in outcomes.failed {
            self.pending.remove(&key);
            logger::warn!("skybox_ibl: bake expired before submit completion (key {key:?})");
        }
    }

    /// Encodes one IBL bake (mip-0 producer + per-mip GGX convolves) and submits it.
    fn schedule_bake(
        &mut self,
        gpu: &mut GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
    ) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::schedule_bake");
        let mut profiler = gpu.take_gpu_profiler();
        let result = self.schedule_bake_with_profiler(gpu, key, source, profiler.as_mut());
        gpu.restore_gpu_profiler(profiler);
        result
    }

    fn schedule_bake_with_profiler(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
        mut profiler: Option<&mut GpuProfilerHandle>,
    ) -> Result<(), SkyboxIblBakeError> {
        self.pipelines.ensure_all(gpu.device())?;
        let input_sampler = self.pipelines.ensure_sampler(gpu.device());
        let face_size = key.face_size();
        let mip_levels = mip_levels_for_edge(face_size);
        let textures = BakeTextures::create(gpu.device(), face_size, mip_levels);
        let mut resources = PendingBakeResources::default();
        textures.retain_transient(&mut resources);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("skybox_ibl bake encoder"),
            });
        self.encode_source_mip0(
            SourceMip0EncodeContext {
                gpu,
                encoder: &mut encoder,
                texture: textures.source_scratch_cube.texture.as_ref(),
                face_size,
                sampler: input_sampler.as_ref(),
                profiler: profiler.as_deref(),
            },
            source,
            &mut resources,
        )?;
        let downsample_pipeline = self.pipelines.get(PipelineSlot::Downsample)?;
        let stitch_pipeline = self.pipelines.get(PipelineSlot::Stitch)?;
        let convolve_pipeline = self.pipelines.get(PipelineSlot::Convolve)?;
        encode_stitch_mip(
            StitchEncodeContext {
                device: gpu.device(),
                encoder: &mut encoder,
                pipeline: stitch_pipeline,
                src_texture: textures.source_scratch_cube.texture.as_ref(),
                dst_texture: textures.source_cube.texture.as_ref(),
                mip: 0,
                dst_size: face_size,
                profiler: profiler.as_deref(),
                label: "skybox_ibl stitch source mip0",
                profiler_label: "skybox_ibl::stitch_source_mip0".to_string(),
            },
            &mut resources,
        );
        copy_cube_mip0(
            &mut encoder,
            textures.source_cube.texture.as_ref(),
            textures.filtered_cube.texture.as_ref(),
            face_size,
            profiler.as_deref(),
        );
        encode_downsample_mips(
            DownsampleEncodeContext {
                device: gpu.device(),
                encoder: &mut encoder,
                pipeline: downsample_pipeline,
                stitch_pipeline,
                texture: textures.source_cube.texture.as_ref(),
                scratch_texture: textures.source_scratch_cube.texture.as_ref(),
                face_size,
                mip_levels,
                profiler: profiler.as_deref(),
            },
            &mut resources,
        );
        encode_convolve_mips(
            ConvolveEncodeContext {
                device: gpu.device(),
                encoder: &mut encoder,
                pipeline: convolve_pipeline,
                stitch_pipeline,
                texture: textures.filtered_cube.texture.as_ref(),
                scratch_texture: textures.filtered_scratch_cube.texture.as_ref(),
                src_view: textures.source_sample_view.as_ref(),
                sampler: input_sampler.as_ref(),
                face_size,
                mip_levels,
                src_max_lod: source_max_lod(mip_levels),
                profiler: profiler.as_deref(),
            },
            &mut resources,
        );
        if let Some(profiler) = profiler.as_mut() {
            profiling::scope!("skybox_ibl::resolve_profiler_queries");
            profiler.resolve_queries(&mut encoder);
        }
        let pending = PendingBake {
            cube: PrefilteredCube {
                texture: textures.filtered_cube.texture.clone(),
                mip_levels,
            },
            _resources: resources,
        };
        self.submit_pending_bake(gpu, key, encoder, pending);
        Ok(())
    }

    /// Dispatches the variant-specific mip-0 producer for one source.
    fn encode_source_mip0(
        &self,
        ctx: SourceMip0EncodeContext<'_>,
        source: SkyboxIblSource,
        resources: &mut PendingBakeResources,
    ) -> Result<(), SkyboxIblBakeError> {
        match source {
            SkyboxIblSource::Cubemap(src) => {
                let pipeline = self.pipelines.get(PipelineSlot::Cube)?;
                encode_cube_mip0(
                    CubeEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        src,
                        sampler: ctx.sampler,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
            SkyboxIblSource::SolidColor(src) => {
                let params = solid_color_params(src.color);
                let pipeline = self.pipelines.get(PipelineSlot::Analytic)?;
                encode_analytic_mip0(
                    AnalyticEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        params: &params,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
            SkyboxIblSource::RuntimeCubemap(src) => {
                let pipeline = self.pipelines.get(PipelineSlot::Cube)?;
                encode_runtime_cube_mip0(
                    RuntimeCubeEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        src,
                        sampler: ctx.sampler,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
        }
        Ok(())
    }

    /// Tracks and submits an encoded bake, retaining transient resources until completion.
    fn submit_pending_bake(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        encoder: wgpu::CommandEncoder,
        pending: PendingBake,
    ) {
        profiling::scope!("skybox_ibl::submit_bake");
        let tx = self.jobs.submit_done_sender();
        let callback_key = key.clone();
        self.jobs.insert(
            key.clone(),
            SubmittedGpuJob {
                resources: GpuJobResources::new(),
            },
        );
        self.pending.insert(key, pending);
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::skybox_ibl");
            encoder.finish()
        };
        gpu.submit_frame_batch_with_callbacks(
            vec![command_buffer],
            None,
            None,
            vec![Box::new(move || {
                let _ = tx.send(callback_key);
            })],
        );
    }
}
