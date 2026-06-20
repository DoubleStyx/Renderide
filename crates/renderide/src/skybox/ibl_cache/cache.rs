//! Owns IBL bakes for prefiltered specular reflection cubemaps.

use std::collections::VecDeque;
use std::sync::Arc;

use hashbrown::HashMap;

use crate::gpu::{FrameSubmitKind, GpuContext};
use crate::gpu_jobs::{GpuJobResources, GpuSubmitJobTracker, SubmittedGpuJob};
use crate::profiling::GpuProfilerHandle;
use crate::skybox::specular::{SkyboxIblSource, solid_color_params};

use super::encode::{
    AnalyticEncodeContext, ConvolveEncodeContext, ConvolveMipEncodeContext, CubeEncodeContext,
    DownsampleEncodeContext, DownsampleMipEncodeContext, RuntimeCubeEncodeContext,
    StitchEncodeContext, encode_analytic_mip0, encode_convolve_mip, encode_convolve_mips,
    encode_cube_mip0, encode_downsample_mip, encode_downsample_mips, encode_runtime_cube_mip0,
    encode_stitch_mip,
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
/// Unity spends eight frames filtering after runtime reflection-probe face capture.
const UNITY_RUNTIME_FILTER_SLICES: u32 = 8;
/// Tick budget after which a missing submit-completion callback is treated as lost.
const MAX_PENDING_IBL_BAKE_AGE_FRAMES: u32 = 120;

/// Scheduling policy for one IBL bake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IblBakePolicy {
    /// Encode and submit every mip in one GPU batch.
    Immediate,
    /// Spread runtime reflection-probe filtering across Unity's post-face update window.
    UnityTimeSliced,
}

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

/// Runtime IBL bake retained while its filter work is spread over multiple ticks.
struct ActiveIblBake {
    /// Source consumed during the first slice.
    source: Option<SkyboxIblSource>,
    /// Textures retained across every slice.
    textures: BakeTextures,
    /// GPU resources retained until the final submit callback promotes the bake.
    resources: PendingBakeResources,
    /// Linear clamp sampler used by source and convolve passes.
    sampler: Arc<wgpu::Sampler>,
    /// Destination face size in texels.
    face_size: u32,
    /// Destination mip count.
    mip_levels: u32,
    /// Number of filter slices in the logical update window.
    filter_slices: u32,
    /// Next filter slice to encode.
    next_slice: u32,
}

impl ActiveIblBake {
    /// Allocates resources for one sliced runtime IBL bake.
    fn new(
        device: &wgpu::Device,
        source: SkyboxIblSource,
        sampler: Arc<wgpu::Sampler>,
        face_size: u32,
        filter_slices: u32,
    ) -> Self {
        let mip_levels = mip_levels_for_edge(face_size);
        let textures = BakeTextures::create(device, face_size, mip_levels);
        let mut resources = PendingBakeResources::default();
        textures.retain_transient(&mut resources);
        Self {
            source: Some(source),
            textures,
            resources,
            sampler,
            face_size,
            mip_levels,
            filter_slices: filter_slices.max(1),
            next_slice: 0,
        }
    }

    /// Encodes the next slice and returns true when this was the final slice.
    fn encode_next_slice(
        &mut self,
        gpu: &GpuContext,
        pipelines: &PipelineStore,
        profiler: Option<&mut GpuProfilerHandle>,
    ) -> Result<(wgpu::CommandEncoder, bool), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::encode_sliced_bake_step");
        let mut profiler = profiler;
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("skybox_ibl sliced bake encoder"),
            });
        let downsample_pipeline = pipelines.get(PipelineSlot::Downsample)?;
        let stitch_pipeline = pipelines.get(PipelineSlot::Stitch)?;
        let convolve_pipeline = pipelines.get(PipelineSlot::Convolve)?;
        if self.next_slice == 0 {
            let source = self
                .source
                .take()
                .ok_or(SkyboxIblBakeError::InvalidSlicedBakeState(
                    "missing first-slice source",
                ))?;
            encode_source_mip0(
                pipelines,
                SourceMip0EncodeContext {
                    gpu,
                    encoder: &mut encoder,
                    texture: self.textures.source_scratch_cube.texture.as_ref(),
                    face_size: self.face_size,
                    sampler: self.sampler.as_ref(),
                    profiler: profiler.as_deref(),
                },
                source,
                &mut self.resources,
            )?;
            encode_stitch_mip(
                StitchEncodeContext {
                    device: gpu.device(),
                    encoder: &mut encoder,
                    pipeline: stitch_pipeline,
                    src_texture: self.textures.source_scratch_cube.texture.as_ref(),
                    dst_texture: self.textures.source_cube.texture.as_ref(),
                    mip: 0,
                    dst_size: self.face_size,
                    profiler: profiler.as_deref(),
                    label: "skybox_ibl stitch source mip0",
                    profiler_label: "skybox_ibl::stitch_source_mip0".to_string(),
                },
                &mut self.resources,
            );
            copy_cube_mip0(
                &mut encoder,
                self.textures.source_cube.texture.as_ref(),
                self.textures.filtered_cube.texture.as_ref(),
                self.face_size,
                profiler.as_deref(),
            );
            for mip in 1..self.mip_levels {
                encode_downsample_mip(
                    DownsampleMipEncodeContext {
                        device: gpu.device(),
                        encoder: &mut encoder,
                        pipeline: downsample_pipeline,
                        stitch_pipeline,
                        texture: self.textures.source_cube.texture.as_ref(),
                        scratch_texture: self.textures.source_scratch_cube.texture.as_ref(),
                        face_size: self.face_size,
                        mip,
                        profiler: profiler.as_deref(),
                    },
                    &mut self.resources,
                );
            }
        }
        for mip in convolve_mips_for_slice(self.mip_levels, self.filter_slices, self.next_slice) {
            encode_convolve_mip(
                ConvolveMipEncodeContext {
                    device: gpu.device(),
                    encoder: &mut encoder,
                    pipeline: convolve_pipeline,
                    stitch_pipeline,
                    texture: self.textures.filtered_cube.texture.as_ref(),
                    scratch_texture: self.textures.filtered_scratch_cube.texture.as_ref(),
                    src_view: self.textures.source_sample_view.as_ref(),
                    sampler: self.sampler.as_ref(),
                    face_size: self.face_size,
                    mip_levels: self.mip_levels,
                    src_max_lod: source_max_lod(self.mip_levels),
                    mip,
                    profiler: profiler.as_deref(),
                },
                &mut self.resources,
            );
        }
        if let Some(profiler) = profiler.as_mut() {
            profiling::scope!("skybox_ibl::resolve_sliced_profiler_queries");
            profiler.resolve_queries(&mut encoder);
        }
        self.next_slice = self.next_slice.saturating_add(1);
        Ok((encoder, self.next_slice >= self.filter_slices))
    }

    /// Converts a finished sliced bake into a pending submit-completion record.
    fn into_pending(self) -> PendingBake {
        PendingBake {
            cube: PrefilteredCube {
                texture: self.textures.filtered_cube.texture.clone(),
                mip_levels: self.mip_levels,
            },
            _resources: self.resources,
        }
    }
}

/// Owns IBL bakes for prefiltered specular reflection cubemaps.
pub(crate) struct SkyboxIblCache {
    /// Submit-completion tracker for in-flight bakes.
    jobs: GpuSubmitJobTracker<SkyboxIblKey>,
    /// In-flight prefiltered cubes retained until their submit callback fires.
    pending: HashMap<SkyboxIblKey, PendingBake>,
    /// Runtime IBL bakes whose filtering is spread across several maintenance ticks.
    active_sliced: HashMap<SkyboxIblKey, ActiveIblBake>,
    /// Round-robin queue for active sliced IBL bake keys.
    sliced_queue: VecDeque<SkyboxIblKey>,
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
            active_sliced: HashMap::new(),
            sliced_queue: VecDeque::new(),
            completed: HashMap::new(),
            pipelines: PipelineStore::default(),
        }
    }

    /// Drains submit-completed bakes and advances one sliced runtime bake step.
    pub(crate) fn maintain_gpu_jobs(&mut self, gpu: &mut GpuContext) {
        {
            profiling::scope!("skybox_ibl::poll_completed_jobs");
            let _ = gpu.device().poll(wgpu::PollType::Poll);
        }
        self.drain_completed_jobs();
        self.advance_sliced_bakes(gpu);
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
        let active_before = self.active_sliced.len();
        let completed_before = self.completed.len();
        self.pending.retain(|key, _| !predicate(key));
        self.active_sliced.retain(|key, _| !predicate(key));
        self.sliced_queue
            .retain(|key| self.active_sliced.contains_key(key));
        self.completed.retain(|key, _| !predicate(key));
        self.jobs.retain(|key| !predicate(key));
        pending_before.saturating_sub(self.pending.len())
            + active_before.saturating_sub(self.active_sliced.len())
            + completed_before.saturating_sub(self.completed.len())
    }

    /// Ensures one arbitrary IBL source is scheduled with the requested policy.
    pub(crate) fn ensure_source_with_policy(
        &mut self,
        gpu: &mut GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
        policy: IblBakePolicy,
    ) -> bool {
        if self.completed.contains_key(&key)
            || self.pending.contains_key(&key)
            || self.jobs.contains_key(&key)
            || self.active_sliced.contains_key(&key)
            || self.in_flight_len() >= MAX_IN_FLIGHT_IBL_BAKES
        {
            return false;
        }
        let result = match policy {
            IblBakePolicy::Immediate => self.schedule_bake(gpu, key, source),
            IblBakePolicy::UnityTimeSliced => {
                if matches!(
                    source,
                    SkyboxIblSource::Cubemap(_) | SkyboxIblSource::SolidColor(_)
                ) {
                    self.schedule_bake(gpu, key, source)
                } else {
                    self.schedule_sliced_bake(gpu, key, source, UNITY_RUNTIME_FILTER_SLICES)
                }
            }
        };
        if let Err(e) = result {
            logger::warn!("skybox_ibl: bake failed: {e}");
            return false;
        }
        true
    }

    /// Returns the number of IBL bakes waiting for submit-completion callbacks.
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len() + self.active_sliced.len()
    }

    /// Returns the number of completed filtered cubes currently retained.
    pub(crate) fn completed_len(&self) -> usize {
        self.completed.len()
    }

    /// Returns the number of runtime bakes currently being filtered across multiple ticks.
    pub(crate) fn active_sliced_len(&self) -> usize {
        self.active_sliced.len()
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

    /// Returns total in-flight IBL jobs, including sliced bakes not yet on the submit tracker.
    fn in_flight_len(&self) -> usize {
        self.jobs.len() + self.active_sliced.len()
    }

    /// Starts a sliced runtime IBL bake without submitting work immediately.
    fn schedule_sliced_bake(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
        filter_slices: u32,
    ) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::schedule_sliced_bake");
        self.pipelines.ensure_all(gpu.device())?;
        let sampler = self.pipelines.ensure_sampler(gpu.device());
        let face_size = key.face_size();
        let bake = ActiveIblBake::new(gpu.device(), source, sampler, face_size, filter_slices);
        self.active_sliced.insert(key.clone(), bake);
        self.sliced_queue.push_back(key);
        Ok(())
    }

    /// Advances every active sliced runtime IBL bake by one Unity filter slice.
    fn advance_sliced_bakes(&mut self, gpu: &mut GpuContext) {
        if self.active_sliced.is_empty() {
            return;
        }
        profiling::scope!("skybox_ibl::advance_sliced_bakes");
        let mut bakes_to_advance = self.active_sliced.len();
        while bakes_to_advance > 0 {
            let Some(key) = self.sliced_queue.pop_front() else {
                return;
            };
            let Some(mut bake) = self.active_sliced.remove(&key) else {
                continue;
            };
            bakes_to_advance -= 1;
            let mut profiler = gpu.take_gpu_profiler();
            let result = bake.encode_next_slice(gpu, &self.pipelines, profiler.as_mut());
            gpu.restore_gpu_profiler(profiler);
            match result {
                Ok((encoder, true)) => {
                    let pending = bake.into_pending();
                    self.submit_pending_bake(gpu, key, encoder, pending);
                }
                Ok((encoder, false)) => {
                    self.submit_sliced_bake_step(gpu, encoder);
                    self.active_sliced.insert(key.clone(), bake);
                    self.sliced_queue.push_back(key);
                }
                Err(error) => {
                    logger::warn!("skybox_ibl: sliced bake failed for key {key:?}: {error}");
                }
            }
        }
    }

    /// Submits a non-final sliced bake step without installing a completion callback.
    fn submit_sliced_bake_step(&self, gpu: &GpuContext, encoder: wgpu::CommandEncoder) {
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::skybox_ibl_sliced");
            encoder.finish()
        };
        {
            profiling::scope!("skybox_ibl::submit_sliced_step_enqueue");
            gpu.submit_frame_batch(
                FrameSubmitKind::BackgroundGpuWork,
                vec![command_buffer],
                None,
                None,
            );
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
        encode_source_mip0(
            &self.pipelines,
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
        {
            profiling::scope!("skybox_ibl::submit_bake_enqueue");
            gpu.submit_frame_batch_with_callbacks(
                FrameSubmitKind::BackgroundGpuWork,
                vec![command_buffer],
                None,
                None,
                vec![Box::new(move || {
                    let _ = tx.send(callback_key);
                })],
            );
        }
    }
}

/// Dispatches the variant-specific mip-0 producer for one source.
fn encode_source_mip0(
    pipelines: &PipelineStore,
    ctx: SourceMip0EncodeContext<'_>,
    source: SkyboxIblSource,
    resources: &mut PendingBakeResources,
) -> Result<(), SkyboxIblBakeError> {
    match source {
        SkyboxIblSource::Cubemap(src) => {
            let pipeline = pipelines.get(PipelineSlot::Cube)?;
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
            let pipeline = pipelines.get(PipelineSlot::Analytic)?;
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
            let pipeline = pipelines.get(PipelineSlot::Cube)?;
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

/// Returns the filtered mip range assigned to one Unity filter slice.
fn convolve_mips_for_slice(
    mip_levels: u32,
    filter_slices: u32,
    slice: u32,
) -> std::ops::Range<u32> {
    let total_mips = mip_levels.saturating_sub(1);
    let slices = filter_slices.max(1);
    if total_mips == 0 || slice >= slices {
        return 1..1;
    }
    let base = total_mips / slices;
    let extra = total_mips % slices;
    let count = base + u32::from(slice < extra);
    let start_offset = slice * base + slice.min(extra);
    let start = 1 + start_offset;
    start..start + count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_sliced_default_256_assigns_one_filtered_mip_per_slice() {
        let ranges = (0..UNITY_RUNTIME_FILTER_SLICES)
            .map(|slice| {
                convolve_mips_for_slice(
                    mip_levels_for_edge(256),
                    UNITY_RUNTIME_FILTER_SLICES,
                    slice,
                )
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            ranges,
            vec![
                vec![1],
                vec![2],
                vec![3],
                vec![4],
                vec![5],
                vec![6],
                vec![7],
                vec![8],
            ]
        );
    }

    #[test]
    fn unity_sliced_smaller_probe_keeps_eight_slice_cadence() {
        let assigned = (0..UNITY_RUNTIME_FILTER_SLICES)
            .flat_map(|slice| {
                convolve_mips_for_slice(
                    mip_levels_for_edge(128),
                    UNITY_RUNTIME_FILTER_SLICES,
                    slice,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(assigned, vec![1, 2, 3, 4, 5, 6, 7]);
        assert!(convolve_mips_for_slice(1, UNITY_RUNTIME_FILTER_SLICES, 7).is_empty());
    }

    #[test]
    fn unity_sliced_larger_probe_groups_extra_mips_without_duplicates() {
        let assigned = (0..UNITY_RUNTIME_FILTER_SLICES)
            .flat_map(|slice| {
                convolve_mips_for_slice(
                    mip_levels_for_edge(512),
                    UNITY_RUNTIME_FILTER_SLICES,
                    slice,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(assigned, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
