//! Nonblocking GPU SH2 projection for reflection-probe host tasks.

use std::collections::{HashMap, HashSet, VecDeque};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::SceneCoordinator;
use crate::shared::{ComputeResult, FrameSubmitData, ReflectionProbeSH2Tasks, RenderSH2};

mod projection_pipeline;
mod readback_jobs;
mod source_resolution;
mod task_rows;

use projection_pipeline::{
    encode_projection_job, ensure_projection_pipeline, ProjectionBinding, ProjectionPipeline,
};
use readback_jobs::{Sh2ReadbackJobs, SubmittedGpuSh2Job};
use source_resolution::{resolve_task_source, Sh2ResolvedSource};
use task_rows::{
    debug_assert_no_scheduled_rows, read_task_header, task_stride, write_task_answer, TaskAnswer,
    TaskHeader,
};

#[cfg(test)]
use crate::shared::ReflectionProbeSH2Task;
#[cfg(test)]
use glam::Vec4;
#[cfg(test)]
use task_rows::read_i32_le;

/// Skybox projection sample resolution per cube face.
const DEFAULT_SAMPLE_SIZE: u32 = 64;
/// Maximum pending GPU jobs kept alive at once.
const MAX_IN_FLIGHT_JOBS: usize = 6;
/// Number of renderer ticks before a pending GPU readback is treated as failed.
const MAX_PENDING_JOB_AGE_FRAMES: u32 = 120;
/// Bytes copied back from the compute output buffer.
const SH2_OUTPUT_BYTES: u64 = (9 * 16) as u64;
/// Default `Projection360` field of view used by Unity material defaults.
const PROJECTION360_DEFAULT_FOV: [f32; 4] = [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];
/// Default texture scale/offset used by Unity `_MainTex_ST` properties.
const DEFAULT_MAIN_TEX_ST: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
/// Uniform payload shared by SH2 projection compute kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Sh2ProjectParams {
    /// Sample grid edge per cube face.
    sample_size: u32,
    /// Projection evaluator mode for parameter-only sky sources.
    mode: u32,
    /// Number of active gradient lobes.
    gradient_count: u32,
    /// Reserved alignment slot.
    _pad0: u32,
    /// Generic color slot 0.
    color0: [f32; 4],
    /// Generic color slot 1.
    color1: [f32; 4],
    /// Generic direction and scalar slot.
    direction: [f32; 4],
    /// Generic scalar slot.
    scalars: [f32; 4],
    /// Gradient direction/spread rows.
    dirs_spread: [[f32; 4]; 16],
    /// Gradient color rows A.
    gradient_color0: [[f32; 4]; 16],
    /// Gradient color rows B.
    gradient_color1: [[f32; 4]; 16],
    /// Gradient parameter rows.
    gradient_params: [[f32; 4]; 16],
}

impl Sh2ProjectParams {
    /// Creates a parameter block with the default sample grid.
    fn empty(mode: SkyParamMode) -> Self {
        Self {
            sample_size: DEFAULT_SAMPLE_SIZE,
            mode: mode as u32,
            gradient_count: 0,
            _pad0: 0,
            color0: [0.0; 4],
            color1: [0.0; 4],
            direction: [0.0, 1.0, 0.0, 0.0],
            scalars: [1.0, 0.0, 0.0, 0.0],
            dirs_spread: [[0.0; 4]; 16],
            gradient_color0: [[0.0; 4]; 16],
            gradient_color1: [[0.0; 4]; 16],
            gradient_params: [[0.0; 4]; 16],
        }
    }
}

/// Hashable `Projection360` equirectangular sampling state used by SH2 cache keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct Projection360EquirectKey {
    /// `_FOV` bit pattern.
    fov_bits: [u32; 4],
    /// `_MainTex_ST` bit pattern.
    main_tex_st_bits: [u32; 4],
    /// `_MainTex_StorageVInverted` bit pattern.
    storage_v_inverted_bits: u32,
}

impl Projection360EquirectKey {
    /// Builds a cache-key fragment from the packed projection parameters.
    fn from_params(params: &Sh2ProjectParams) -> Self {
        Self {
            fov_bits: f32x4_bits(params.color0),
            main_tex_st_bits: f32x4_bits(params.color1),
            storage_v_inverted_bits: params.scalars[0].to_bits(),
        }
    }
}

/// Parameter-only sky evaluator mode used by `sh2_project_sky_params`.
#[derive(Clone, Copy, Debug)]
enum SkyParamMode {
    /// Procedural sky approximation from material scalar/color properties.
    Procedural = 1,
    /// Gradient sky approximation from material array properties.
    Gradient = 2,
}

/// Hashable description of the source projected into SH2.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Sh2SourceKey {
    /// Analytic constant-color source.
    ConstantColor {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// RGBA color bit pattern.
        color_bits: [u32; 4],
    },
    /// Resident cubemap source.
    Cubemap {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Cubemap asset id.
        asset_id: i32,
        /// Face size.
        size: u32,
        /// Contiguous resident mip count.
        resident_mips: u32,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
        /// Host material generation mixed into skybox sources.
        material_generation: u64,
    },
    /// Resident equirectangular texture source.
    EquirectTexture2D {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Texture asset id.
        asset_id: i32,
        /// Mip0 width.
        width: u32,
        /// Mip0 height.
        height: u32,
        /// Contiguous resident mip count.
        resident_mips: u32,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
        /// Host material generation.
        material_generation: u64,
        /// Projection360 equirectangular sampling state.
        projection: Projection360EquirectKey,
    },
    /// Parameter-only sky material source.
    SkyParams {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Skybox material asset id.
        material_asset_id: i32,
        /// Host material generation.
        material_generation: u64,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
        /// Shader route discriminator.
        route_hash: u64,
    },
}

/// GPU-projected source payload queued for scheduling.
#[derive(Clone, Debug)]
enum GpuSh2Source {
    /// Cubemap sampled from the cubemap pool.
    Cubemap { asset_id: i32 },
    /// Equirectangular 2D texture sampled from the texture pool.
    EquirectTexture2D {
        /// Texture asset id.
        asset_id: i32,
        /// Projection360 sampling parameters.
        params: Box<Sh2ProjectParams>,
    },
    /// Parameter-only sky material evaluator.
    SkyParams { params: Box<Sh2ProjectParams> },
}

/// Nonblocking SH2 projection cache and GPU-job scheduler.
pub struct ReflectionProbeSh2System {
    /// Completed projection results keyed by source identity.
    completed: HashMap<Sh2SourceKey, RenderSH2>,
    /// In-flight GPU readback jobs keyed by source identity.
    readback_jobs: Sh2ReadbackJobs,
    /// Sources that failed recently.
    failed: HashSet<Sh2SourceKey>,
    /// Source payloads awaiting an in-flight slot.
    queued_sources: HashMap<Sh2SourceKey, GpuSh2Source>,
    /// FIFO ordering for [`Self::queued_sources`].
    queue_order: VecDeque<Sh2SourceKey>,
    /// Lazily-created cubemap pipeline.
    cubemap_pipeline: Option<ProjectionPipeline>,
    /// Lazily-created equirectangular 2D pipeline.
    equirect_pipeline: Option<ProjectionPipeline>,
    /// Lazily-created parameter sky pipeline.
    sky_params_pipeline: Option<ProjectionPipeline>,
    /// Source keys touched by the current task pass.
    touched_this_pass: HashSet<Sh2SourceKey>,
}

impl Default for ReflectionProbeSh2System {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionProbeSh2System {
    /// Creates an empty SH2 system.
    pub fn new() -> Self {
        Self {
            completed: HashMap::new(),
            readback_jobs: Sh2ReadbackJobs::new(),
            failed: HashSet::new(),
            queued_sources: HashMap::new(),
            queue_order: VecDeque::new(),
            cubemap_pipeline: None,
            equirect_pipeline: None,
            sky_params_pipeline: None,
            touched_this_pass: HashSet::new(),
        }
    }

    /// Answers every SH2 task row in a frame submit without blocking for GPU readback.
    pub fn answer_frame_submit_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        materials: &crate::backend::MaterialSystem,
        assets: &crate::backend::AssetTransferQueue,
        data: &FrameSubmitData,
    ) {
        profiling::scope!("reflection_probe_sh2::answer_frame_submit_tasks");
        self.touched_this_pass.clear();
        for update in &data.render_spaces {
            let Some(tasks) = update.reflection_probe_sh2_taks.as_ref() else {
                continue;
            };
            self.answer_task_buffer(shm, scene, materials, assets, update.id, tasks);
        }
        self.prune_untouched_failures();
    }

    /// Advances GPU callbacks, maps completed buffers, and schedules queued work.
    pub fn maintain_gpu_jobs(
        &mut self,
        gpu: &GpuContext,
        assets: &crate::backend::AssetTransferQueue,
    ) {
        profiling::scope!("reflection_probe_sh2::maintain_gpu_jobs");
        let _ = gpu.device().poll(wgpu::PollType::Poll);
        let outcomes = self.readback_jobs.maintain();
        for (key, sh) in outcomes.completed {
            self.failed.remove(&key);
            self.completed.insert(key, sh);
        }
        for key in outcomes.failed {
            self.failed.insert(key);
        }
        self.schedule_queued_sources(gpu, assets);
    }

    /// Answers all rows in one shared-memory task descriptor.
    fn answer_task_buffer(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        materials: &crate::backend::MaterialSystem,
        assets: &crate::backend::AssetTransferQueue,
        render_space_id: i32,
        tasks: &ReflectionProbeSH2Tasks,
    ) {
        if tasks.tasks.length <= 0 {
            return;
        }

        let ok = shm.access_mut_bytes(&tasks.tasks, |bytes| {
            let mut offset = 0usize;
            while offset + task_stride() <= bytes.len() {
                let Some(task) = read_task_header(bytes, offset) else {
                    break;
                };
                if task.renderable_index < 0 {
                    break;
                }
                let answer = self.answer_for_task(scene, materials, assets, render_space_id, task);
                write_task_answer(bytes, offset, answer);
                offset += task_stride();
            }
            debug_assert_no_scheduled_rows(bytes);
        });

        if !ok {
            logger::warn!(
                "reflection_probe_sh2: could not write SH2 task results (shared memory buffer)"
            );
        }
    }

    /// Resolves one host task into an immediate answer.
    fn answer_for_task(
        &mut self,
        scene: &SceneCoordinator,
        materials: &crate::backend::MaterialSystem,
        assets: &crate::backend::AssetTransferQueue,
        render_space_id: i32,
        task: TaskHeader,
    ) -> TaskAnswer {
        let Some((key, source)) =
            resolve_task_source(scene, materials, assets, render_space_id, task)
        else {
            return TaskAnswer::status(ComputeResult::Failed);
        };

        self.touched_this_pass.insert(key.clone());
        if let Some(sh) = self.completed.get(&key) {
            return TaskAnswer::computed(*sh);
        }
        if self.readback_jobs.contains_key(&key) {
            return TaskAnswer::status(ComputeResult::Postpone);
        }
        if self.failed.contains(&key) {
            return TaskAnswer::status(ComputeResult::Failed);
        }
        match source {
            Sh2ResolvedSource::Cpu(sh) => {
                let sh = *sh;
                self.completed.insert(key, sh);
                TaskAnswer::computed(sh)
            }
            Sh2ResolvedSource::Gpu(gpu_source) => {
                self.queue_source(key, gpu_source);
                TaskAnswer::status(ComputeResult::Postpone)
            }
            Sh2ResolvedSource::Postpone => TaskAnswer::status(ComputeResult::Postpone),
        }
    }

    /// Queues a source for later GPU scheduling.
    fn queue_source(&mut self, key: Sh2SourceKey, source: GpuSh2Source) {
        if self.queued_sources.contains_key(&key) {
            return;
        }
        self.queue_order.push_back(key.clone());
        self.queued_sources.insert(key, source);
    }

    /// Drops failed keys that are no longer present in host task rows.
    fn prune_untouched_failures(&mut self) {
        self.failed
            .retain(|key| self.touched_this_pass.contains(key));
    }

    /// Schedules queued sources until the in-flight cap is reached.
    fn schedule_queued_sources(
        &mut self,
        gpu: &GpuContext,
        assets: &crate::backend::AssetTransferQueue,
    ) {
        while self.readback_jobs.len() < MAX_IN_FLIGHT_JOBS {
            let Some(key) = self.queue_order.pop_front() else {
                break;
            };
            let Some(source) = self.queued_sources.remove(&key) else {
                continue;
            };
            if self.completed.contains_key(&key)
                || self.readback_jobs.contains_key(&key)
                || self.failed.contains(&key)
            {
                continue;
            }
            match self.schedule_source(gpu, assets, key.clone(), source) {
                Ok(job) => {
                    self.readback_jobs.insert(key, job);
                }
                Err(e) => {
                    logger::warn!("reflection_probe_sh2: GPU SH2 schedule failed: {e}");
                    self.failed.insert(key);
                }
            }
        }
    }

    /// Encodes and submits one source projection.
    fn schedule_source(
        &mut self,
        gpu: &GpuContext,
        assets: &crate::backend::AssetTransferQueue,
        key: Sh2SourceKey,
        source: GpuSh2Source,
    ) -> Result<SubmittedGpuSh2Job, String> {
        match source {
            GpuSh2Source::Cubemap { asset_id } => {
                let tex = assets
                    .cubemap_pool
                    .get_texture(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .ok_or_else(|| format!("cubemap {asset_id} not resident"))?;
                let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("SH2 cubemap sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                let view = tex.view.clone();
                let submit_done_tx = self.readback_jobs.submit_done_sender();
                let pipeline = ensure_projection_pipeline(
                    &mut self.cubemap_pipeline,
                    gpu.device(),
                    "sh2_project_cubemap",
                )?;
                encode_projection_job(
                    gpu,
                    key,
                    pipeline,
                    &[
                        ProjectionBinding::TextureView(view.as_ref()),
                        ProjectionBinding::Sampler(&sampler),
                    ],
                    &Sh2ProjectParams::empty(SkyParamMode::Procedural),
                    &submit_done_tx,
                )
            }
            GpuSh2Source::EquirectTexture2D { asset_id, params } => {
                let tex = assets
                    .texture_pool
                    .get_texture(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .ok_or_else(|| format!("texture2d {asset_id} not resident"))?;
                let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("SH2 equirect sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                let view = tex.view.clone();
                let submit_done_tx = self.readback_jobs.submit_done_sender();
                let pipeline = ensure_projection_pipeline(
                    &mut self.equirect_pipeline,
                    gpu.device(),
                    "sh2_project_equirect",
                )?;
                encode_projection_job(
                    gpu,
                    key,
                    pipeline,
                    &[
                        ProjectionBinding::TextureView(view.as_ref()),
                        ProjectionBinding::Sampler(&sampler),
                    ],
                    params.as_ref(),
                    &submit_done_tx,
                )
            }
            GpuSh2Source::SkyParams { params } => {
                let submit_done_tx = self.readback_jobs.submit_done_sender();
                let pipeline = ensure_projection_pipeline(
                    &mut self.sky_params_pipeline,
                    gpu.device(),
                    "sh2_project_sky_params",
                )?;
                encode_projection_job(gpu, key, pipeline, &[], params.as_ref(), &submit_done_tx)
            }
        }
    }
}

/// Converts a storage-orientation boolean to the shader keyword float convention.
fn storage_v_inverted_flag(storage_v_inverted: bool) -> f32 {
    if storage_v_inverted {
        1.0
    } else {
        0.0
    }
}

/// Bit pattern for a packed float4.
fn f32x4_bits(v: [f32; 4]) -> [u32; 4] {
    [
        v[0].to_bits(),
        v[1].to_bits(),
        v[2].to_bits(),
        v[3].to_bits(),
    ]
}

/// Analytic SH2 coefficients for a constant radiance color.
pub fn constant_color_sh2(color: Vec3) -> RenderSH2 {
    let c = color * (4.0 * std::f32::consts::PI * SH_C0);
    RenderSH2 {
        sh0: c,
        ..RenderSH2::default()
    }
}

/// Zeroth-order SH basis constant.
pub const SH_C0: f32 = 0.282_094_8;

/// First-order SH basis constant.
#[cfg(test)]
pub const SH_C1: f32 = 0.488_602_52;

/// Second-order `xy`, `yz`, and `xz` SH basis constant.
#[cfg(test)]
pub const SH_C2: f32 = 1.092_548_5;

/// Second-order `3z²-1` SH basis constant.
#[cfg(test)]
pub const SH_C3: f32 = 0.315_391_57;

/// Second-order `x²-y²` SH basis constant.
#[cfg(test)]
pub const SH_C4: f32 = 0.546_274_24;

/// Evaluates raw RenderSH2 coefficients for a world-space normal.
#[cfg(test)]
pub fn evaluate_sh2(sh: &RenderSH2, n: Vec3) -> Vec3 {
    sh.sh0 * SH_C0
        + sh.sh1 * (SH_C1 * n.y)
        + sh.sh2 * (SH_C1 * n.z)
        + sh.sh3 * (SH_C1 * n.x)
        + sh.sh4 * (SH_C2 * n.x * n.y)
        + sh.sh5 * (SH_C2 * n.y * n.z)
        + sh.sh6 * (SH_C3 * (3.0 * n.z * n.z - 1.0))
        + sh.sh7 * (SH_C2 * n.x * n.z)
        + sh.sh8 * (SH_C4 * (n.x * n.x - n.y * n.y))
}

/// Applies WGSL-style positive modulo for Projection360 angle wrapping.
#[cfg(test)]
fn positive_fmod_scalar(v: f32, wrap: f32) -> f32 {
    let mut r = v - (v / wrap).trunc() * wrap;
    r += wrap;
    r - (r / wrap).trunc() * wrap
}

/// Converts a raw texture-space direction to the pre-ST equirectangular UV convention.
#[cfg(test)]
fn raw_equirect_uv_for_dir(dir: Vec3) -> [f32; 2] {
    [
        dir.x.atan2(dir.z) / std::f32::consts::TAU + 0.5,
        dir.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI,
    ]
}

/// Converts a Projection360 view direction to pre-ST UVs using the visible shader formula.
#[cfg(test)]
fn projection360_dir_to_uv_for_test(view_dir: Vec3, params: &Sh2ProjectParams) -> [f32; 2] {
    let angle_x = view_dir.x.atan2(view_dir.z) + params.color0[0] * 0.5 + params.color0[2];
    let angle_y = view_dir.y.clamp(-1.0, 1.0).acos() - std::f32::consts::FRAC_PI_2
        + params.color0[1] * 0.5
        + params.color0[3];
    [
        positive_fmod_scalar(angle_x, std::f32::consts::TAU)
            / params.color0[0].abs().max(0.000_001),
        positive_fmod_scalar(angle_y, std::f32::consts::PI) / params.color0[1].abs().max(0.000_001),
    ]
}

/// Applies the visible shader's `_MainTex_ST` and storage-orientation handling.
#[cfg(test)]
fn projection360_main_tex_uv_for_test(uv: [f32; 2], params: &Sh2ProjectParams) -> [f32; 2] {
    let u = uv[0].clamp(0.0, 1.0) * params.color1[0] + params.color1[2];
    let v = uv[1].clamp(0.0, 1.0) * params.color1[1] + params.color1[3];
    if params.scalars[0] > 0.5 {
        [u, v]
    } else {
        [u, 1.0 - v]
    }
}

/// Returns the texture UV that visible Projection360 equirectangular skybox sampling uses.
#[cfg(test)]
fn projection360_equirect_uv_for_world_dir(world_dir: Vec3, params: &Sh2ProjectParams) -> [f32; 2] {
    projection360_main_tex_uv_for_test(
        projection360_dir_to_uv_for_test(-world_dir.normalize(), params),
        params,
    )
}

/// Returns the cubemap direction used by the visible Projection360 cubemap path.
#[cfg(test)]
fn projection360_cubemap_sample_dir_for_world_dir(world_dir: Vec3) -> Vec3 {
    let view_dir = -world_dir.normalize();
    (-view_dir).normalize()
}

/// Evaluates the GradientSkybox color using the visible shader formula.
#[cfg(test)]
fn gradient_sky_visible_color_for_dir(dir: Vec3, params: &Sh2ProjectParams) -> Vec3 {
    let mut color = Vec3::from_array([params.color0[0], params.color0[1], params.color0[2]]);
    let count = params.gradient_count.min(16) as usize;
    for i in 0..count {
        let dirs_spread = params.dirs_spread[i];
        let gradient_params = params.gradient_params[i];
        let axis = Vec3::new(dirs_spread[0], dirs_spread[1], dirs_spread[2]).normalize();
        let spread = dirs_spread[3].abs().max(0.000_001);
        let expv = gradient_params[1].max(0.000_001);
        let fromv = gradient_params[2];
        let tov = gradient_params[3];
        let denom = (tov - fromv).abs().max(0.000_001);
        let mut r = (0.5 - dir.dot(axis) * 0.5) / spread;
        if r <= 1.0 {
            r = r.max(0.0).powf(expv);
            r = ((r - fromv) / denom).clamp(0.0, 1.0);
            let c0 = Vec4::from_array(params.gradient_color0[i]);
            let c1 = Vec4::from_array(params.gradient_color1[i]);
            let c = c0.lerp(c1, r);
            if gradient_params[0].abs() <= f32::EPSILON {
                color = color * (1.0 - c.w) + c.truncate() * c.w;
            } else {
                color += c.truncate() * c.w;
            }
        }
    }
    color
}

/// Computes the cubemap texel solid-angle helper used by the GPU SH kernels.
#[cfg(test)]
fn sh2_area_element(x: f32, y: f32) -> f32 {
    (x * y).atan2((x * x + y * y + 1.0).sqrt())
}

/// Computes a cube-face texel solid angle for CPU SH regression tests.
#[cfg(test)]
fn sh2_texel_solid_angle(x: u32, y: u32, n: u32) -> f32 {
    let inv = 1.0 / n as f32;
    let x0 = (x as f32 * inv) * 2.0 - 1.0;
    let y0 = (y as f32 * inv) * 2.0 - 1.0;
    let x1 = ((x + 1) as f32 * inv) * 2.0 - 1.0;
    let y1 = ((y + 1) as f32 * inv) * 2.0 - 1.0;
    (sh2_area_element(x0, y0) - sh2_area_element(x0, y1) - sh2_area_element(x1, y0)
        + sh2_area_element(x1, y1))
    .abs()
}

/// Returns the Unity cube-face direction for one sample location.
#[cfg(test)]
fn sh2_cube_dir(face: u32, x: u32, y: u32, n: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / n as f32;
    let v = (y as f32 + 0.5) / n as f32;
    match face {
        0 => Vec3::new(1.0, v * -2.0 + 1.0, u * -2.0 + 1.0).normalize(),
        1 => Vec3::new(-1.0, v * -2.0 + 1.0, u * 2.0 - 1.0).normalize(),
        2 => Vec3::new(u * 2.0 - 1.0, 1.0, v * 2.0 - 1.0).normalize(),
        3 => Vec3::new(u * 2.0 - 1.0, -1.0, v * -2.0 + 1.0).normalize(),
        4 => Vec3::new(u * 2.0 - 1.0, v * -2.0 + 1.0, 1.0).normalize(),
        _ => Vec3::new(u * -2.0 + 1.0, v * -2.0 + 1.0, -1.0).normalize(),
    }
}

/// Accumulates one weighted radiance sample into RenderSH2 coefficients.
#[cfg(test)]
fn add_weighted_sh2_sample(sh: &mut RenderSH2, c: Vec3, dir: Vec3, weight: f32) {
    sh.sh0 += c * (SH_C0 * weight);
    sh.sh1 += c * (SH_C1 * dir.y * weight);
    sh.sh2 += c * (SH_C1 * dir.z * weight);
    sh.sh3 += c * (SH_C1 * dir.x * weight);
    sh.sh4 += c * (SH_C2 * dir.x * dir.y * weight);
    sh.sh5 += c * (SH_C2 * dir.y * dir.z * weight);
    sh.sh6 += c * (SH_C3 * (3.0 * dir.z * dir.z - 1.0) * weight);
    sh.sh7 += c * (SH_C2 * dir.x * dir.z * weight);
    sh.sh8 += c * (SH_C4 * (dir.x * dir.x - dir.y * dir.y) * weight);
}

/// Projects a directional equirectangular lobe through the Projection360 `_VIEW` convention.
#[cfg(test)]
fn project_projection360_equirect_lobe(sample_size: u32, bright_texture_dir: Vec3) -> RenderSH2 {
    let n = sample_size.max(1);
    let bright_texture_dir = bright_texture_dir.normalize();
    let mut sh = RenderSH2::default();
    for face in 0..6 {
        for y in 0..n {
            for x in 0..n {
                let world_dir = sh2_cube_dir(face, x, y, n);
                let texture_dir = -world_dir;
                let intensity = texture_dir.dot(bright_texture_dir).max(0.0).powf(16.0);
                if intensity > 0.0 {
                    add_weighted_sh2_sample(
                        &mut sh,
                        Vec3::splat(intensity),
                        world_dir,
                        sh2_texel_solid_angle(x, y, n),
                    );
                }
            }
        }
    }
    sh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_color_evaluates_back_to_color() {
        let color = Vec3::new(0.25, 0.5, 1.0);
        let sh = constant_color_sh2(color);
        let evaluated = evaluate_sh2(&sh, Vec3::Y);
        assert!((evaluated - color).length() < 1e-5);
    }

    #[test]
    fn basis_constants_match_unity_values() {
        assert!((SH_C0 - 0.282_094_8).abs() < 1e-7);
        assert!((SH_C1 - 0.488_602_52).abs() < 1e-7);
        assert!((SH_C2 - 1.092_548_5).abs() < 1e-7);
        assert!((SH_C3 - 0.315_391_57).abs() < 1e-7);
        assert!((SH_C4 - 0.546_274_24).abs() < 1e-7);
    }

    #[test]
    fn projection360_equirect_view_sampling_uses_opposite_world_direction() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        params.color0 = PROJECTION360_DEFAULT_FOV;
        params.color1 = DEFAULT_MAIN_TEX_ST;
        params.scalars = [1.0, 0.0, 0.0, 0.0];

        let world_dir = Vec3::X;
        let visible_uv = projection360_equirect_uv_for_world_dir(world_dir, &params);
        let opposite_uv = raw_equirect_uv_for_dir(-world_dir);
        let direct_uv = raw_equirect_uv_for_dir(world_dir);

        assert!((visible_uv[0] - opposite_uv[0]).abs() < 1e-6);
        assert!((visible_uv[1] - opposite_uv[1]).abs() < 1e-6);
        assert!((visible_uv[0] - direct_uv[0]).abs() > 0.25);
    }

    #[test]
    fn projection360_fov_st_and_storage_affect_equirect_source_key() {
        let mut base = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        base.color0 = PROJECTION360_DEFAULT_FOV;
        base.color1 = DEFAULT_MAIN_TEX_ST;
        base.scalars = [0.0, 0.0, 0.0, 0.0];
        let base_key = Projection360EquirectKey::from_params(&base);

        let mut fov = base;
        fov.color0[2] = 0.125;
        let mut st = base;
        st.color1[2] = 0.25;
        let mut storage = base;
        storage.scalars[0] = 1.0;

        assert_ne!(base_key, Projection360EquirectKey::from_params(&fov));
        assert_ne!(base_key, Projection360EquirectKey::from_params(&st));
        assert_ne!(base_key, Projection360EquirectKey::from_params(&storage));
    }

    #[test]
    fn projection360_cubemap_path_keeps_world_direction() {
        let world_dir = Vec3::new(0.25, 0.5, -1.0).normalize();
        let sample_dir = projection360_cubemap_sample_dir_for_world_dir(world_dir);
        assert!((sample_dir - world_dir).length() < 1e-6);
    }

    #[test]
    fn gradient_sky_sampling_matches_visible_axes() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
        params.color0 = [0.0, 0.0, 0.0, 1.0];
        params.gradient_count = 1;
        params.dirs_spread[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color1[0] = [0.0, 0.0, 1.0, 1.0];
        params.gradient_params[0] = [0.0, 1.0, 0.0, 1.0];

        let plus_x = gradient_sky_visible_color_for_dir(Vec3::X, &params);
        let minus_x = gradient_sky_visible_color_for_dir(-Vec3::X, &params);
        let plus_y = gradient_sky_visible_color_for_dir(Vec3::Y, &params);
        let plus_z = gradient_sky_visible_color_for_dir(Vec3::Z, &params);

        assert!((plus_x - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((minus_x - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
        assert!((plus_y - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
        assert!((plus_z - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
    }

    #[test]
    fn projection360_equirect_lobe_evaluates_strongest_in_visible_world_direction() {
        let sh = project_projection360_equirect_lobe(24, -Vec3::X);
        let visible_direction = evaluate_sh2(&sh, Vec3::X).x;
        let opposite_direction = evaluate_sh2(&sh, -Vec3::X).x;
        assert!(visible_direction > opposite_direction);
    }

    #[test]
    fn task_answer_postpone_leaves_no_scheduled_row() {
        const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
        let mut row = vec![0u8; task_stride()];
        row[0..4].copy_from_slice(&0i32.to_le_bytes());
        row[4..8].copy_from_slice(&0i32.to_le_bytes());
        row[RESULT_OFFSET..RESULT_OFFSET + 4]
            .copy_from_slice(&(ComputeResult::Scheduled as i32).to_le_bytes());

        write_task_answer(&mut row, 0, TaskAnswer::status(ComputeResult::Postpone));
        debug_assert_no_scheduled_rows(&row);

        let result = read_i32_le(&row[RESULT_OFFSET..RESULT_OFFSET + 4]);
        assert_eq!(result, Some(ComputeResult::Postpone as i32));
    }

    #[test]
    fn computed_task_answer_writes_data_before_result_slot() {
        const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
        const DATA_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result_data);
        let mut row = vec![0u8; task_stride()];
        row[0..4].copy_from_slice(&0i32.to_le_bytes());
        row[4..8].copy_from_slice(&0i32.to_le_bytes());
        let sh = RenderSH2 {
            sh0: Vec3::new(1.0, 2.0, 3.0),
            ..RenderSH2::default()
        };

        write_task_answer(&mut row, 0, TaskAnswer::computed(sh));
        debug_assert_no_scheduled_rows(&row);

        let result = read_i32_le(&row[RESULT_OFFSET..RESULT_OFFSET + 4]);
        let first_component = f32::from_le_bytes(
            row[DATA_OFFSET..DATA_OFFSET + 4]
                .try_into()
                .expect("four-byte f32"),
        );
        assert_eq!(result, Some(ComputeResult::Computed as i32));
        assert_eq!(first_component, 1.0);
    }
}
