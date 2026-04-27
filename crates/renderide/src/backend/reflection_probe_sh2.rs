//! Nonblocking GPU SH2 projection for reflection-probe host tasks.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use crossbeam_channel as mpsc;
use glam::{Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::assets::texture::{unpack_host_texture_packed, HostTextureAssetKind};
use crate::embedded_shaders;
use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::{reflection_probe_skybox_only, RenderSpaceId, SceneCoordinator};
use crate::shared::memory_packable::MemoryPackable;
use crate::shared::memory_packer::MemoryPacker;
use crate::shared::{
    ComputeResult, FrameSubmitData, ReflectionProbeClear, ReflectionProbeSH2Task,
    ReflectionProbeSH2Tasks, ReflectionProbeType, RenderSH2, RENDER_SH2_HOST_ROW_BYTES,
};

/// Skybox projection sample resolution per cube face.
const DEFAULT_SAMPLE_SIZE: u32 = 64;
/// Maximum pending GPU jobs kept alive at once.
const MAX_IN_FLIGHT_JOBS: usize = 6;
/// Number of renderer ticks before a pending GPU readback is treated as failed.
const MAX_PENDING_JOB_AGE_FRAMES: u32 = 120;
/// Bytes copied back from the compute output buffer.
const SH2_OUTPUT_BYTES: u64 = (9 * 16) as u64;
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
    EquirectTexture2D { asset_id: i32 },
    /// Parameter-only sky material evaluator.
    SkyParams { params: Box<Sh2ProjectParams> },
}

/// A GPU job whose commands have been submitted and whose readback may complete later.
struct PendingGpuJob {
    /// Staging buffer copied from the compute output.
    staging: wgpu::Buffer,
    /// Compute output buffer kept alive until readback finishes.
    _output: wgpu::Buffer,
    /// Bind group kept alive until the queued command has completed.
    _bind_group: wgpu::BindGroup,
    /// Uniform/parameter buffers kept alive until the queued command has completed.
    _buffers: Vec<wgpu::Buffer>,
    /// Whether the submit-done callback has fired.
    submit_done: bool,
    /// Pending `map_async` result receiver.
    map_recv: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Age in renderer ticks.
    age_frames: u32,
}

/// Nonblocking SH2 projection cache and GPU-job scheduler.
pub struct ReflectionProbeSh2System {
    /// Completed projection results keyed by source identity.
    completed: HashMap<Sh2SourceKey, RenderSH2>,
    /// In-flight GPU jobs keyed by source identity.
    pending: HashMap<Sh2SourceKey, PendingGpuJob>,
    /// Sources that failed recently.
    failed: HashSet<Sh2SourceKey>,
    /// Source payloads awaiting an in-flight slot.
    queued_sources: HashMap<Sh2SourceKey, GpuSh2Source>,
    /// FIFO ordering for [`Self::queued_sources`].
    queue_order: VecDeque<Sh2SourceKey>,
    /// Submit-done channel sender captured by queue callbacks.
    submit_done_tx: mpsc::Sender<Sh2SourceKey>,
    /// Submit-done channel receiver drained on the main thread.
    submit_done_rx: mpsc::Receiver<Sh2SourceKey>,
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
        let (submit_done_tx, submit_done_rx) = mpsc::unbounded();
        Self {
            completed: HashMap::new(),
            pending: HashMap::new(),
            failed: HashSet::new(),
            queued_sources: HashMap::new(),
            queue_order: VecDeque::new(),
            submit_done_tx,
            submit_done_rx,
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
        self.drain_submit_done();
        self.start_ready_maps();
        self.drain_completed_maps();
        self.age_pending_jobs();
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
        if self.pending.contains_key(&key) {
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

    /// Marks jobs whose queue submit has completed.
    fn drain_submit_done(&mut self) {
        while let Ok(key) = self.submit_done_rx.try_recv() {
            if let Some(job) = self.pending.get_mut(&key) {
                job.submit_done = true;
            }
        }
    }

    /// Starts `map_async` for submitted jobs on the main thread.
    fn start_ready_maps(&mut self) {
        for job in self.pending.values_mut() {
            if !job.submit_done || job.map_recv.is_some() {
                continue;
            }
            let slice = job.staging.slice(..);
            let (tx, rx) = mpsc::bounded::<Result<(), wgpu::BufferAsyncError>>(1);
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            job.map_recv = Some(rx);
        }
    }

    /// Moves completed mapped buffers into the result cache.
    fn drain_completed_maps(&mut self) {
        let mut completed = Vec::new();
        let mut failed = Vec::new();
        for (key, job) in &mut self.pending {
            let Some(recv) = job.map_recv.as_ref() else {
                continue;
            };
            match recv.try_recv() {
                Ok(Ok(())) => match read_sh2_from_staging(&job.staging) {
                    Some(sh) => completed.push((key.clone(), sh)),
                    None => failed.push(key.clone()),
                },
                Ok(Err(_)) => failed.push(key.clone()),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => failed.push(key.clone()),
            }
        }
        for (key, sh) in completed {
            self.pending.remove(&key);
            self.failed.remove(&key);
            self.completed.insert(key, sh);
        }
        for key in failed {
            if let Some(job) = self.pending.remove(&key) {
                job.staging.unmap();
            }
            self.failed.insert(key);
        }
    }

    /// Ages in-flight jobs and fails sources that never map back.
    fn age_pending_jobs(&mut self) {
        let mut expired = Vec::new();
        for (key, job) in &mut self.pending {
            job.age_frames = job.age_frames.saturating_add(1);
            if job.age_frames > MAX_PENDING_JOB_AGE_FRAMES {
                expired.push(key.clone());
            }
        }
        for key in expired {
            if let Some(job) = self.pending.remove(&key) {
                job.staging.unmap();
            }
            self.failed.insert(key);
        }
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
        while self.pending.len() < MAX_IN_FLIGHT_JOBS {
            let Some(key) = self.queue_order.pop_front() else {
                break;
            };
            let Some(source) = self.queued_sources.remove(&key) else {
                continue;
            };
            if self.completed.contains_key(&key)
                || self.pending.contains_key(&key)
                || self.failed.contains(&key)
            {
                continue;
            }
            match self.schedule_source(gpu, assets, key.clone(), source) {
                Ok(job) => {
                    self.pending.insert(key, job);
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
    ) -> Result<PendingGpuJob, String> {
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
                let submit_done_tx = self.submit_done_tx.clone();
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
            GpuSh2Source::EquirectTexture2D { asset_id } => {
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
                let submit_done_tx = self.submit_done_tx.clone();
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
                    &Sh2ProjectParams::empty(SkyParamMode::Procedural),
                    &submit_done_tx,
                )
            }
            GpuSh2Source::SkyParams { params } => {
                let submit_done_tx = self.submit_done_tx.clone();
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

/// Lazily-created compute pipeline and bind-group layout.
struct ProjectionPipeline {
    /// Compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind-group layout for one projection source.
    layout: wgpu::BindGroupLayout,
}

/// Extra binding resource for texture-backed projection kernels.
enum ProjectionBinding<'a> {
    /// Sampled texture view.
    TextureView(&'a wgpu::TextureView),
    /// Sampler paired with the texture view.
    Sampler(&'a wgpu::Sampler),
}

/// Ensures a projection pipeline exists for an embedded compute shader.
fn ensure_projection_pipeline<'a>(
    slot: &'a mut Option<ProjectionPipeline>,
    device: &wgpu::Device,
    stem: &str,
) -> Result<&'a ProjectionPipeline, String> {
    if slot.is_none() {
        let source = embedded_shaders::embedded_target_wgsl(stem)
            .ok_or_else(|| format!("embedded shader {stem} not found"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(stem),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        let layout_entries = projection_layout_entries(stem);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{stem} bind group layout")),
            entries: &layout_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{stem} pipeline layout")),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(stem),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        *slot = Some(ProjectionPipeline { pipeline, layout });
    }
    slot.as_ref()
        .ok_or_else(|| format!("projection pipeline {stem} missing after creation"))
}

/// Returns bind-group layout entries for a projection shader.
fn projection_layout_entries(stem: &str) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    match stem {
        "sh2_project_cubemap" => {
            entries.push(texture_layout_entry(1, wgpu::TextureViewDimension::Cube));
            entries.push(sampler_layout_entry(2));
        }
        "sh2_project_equirect" => {
            entries.push(texture_layout_entry(1, wgpu::TextureViewDimension::D2));
            entries.push(sampler_layout_entry(2));
        }
        _ => {}
    }
    entries.sort_by_key(|entry| entry.binding);
    entries
}

/// Texture bind-group layout entry for projection kernels.
fn texture_layout_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// Sampler bind-group layout entry for projection kernels.
fn sampler_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Encodes one projection dispatch and queues it through the GPU driver thread.
fn encode_projection_job(
    gpu: &GpuContext,
    key: Sh2SourceKey,
    pipeline: &ProjectionPipeline,
    extra_bindings: &[ProjectionBinding<'_>],
    params: &Sh2ProjectParams,
    submit_done_tx: &mpsc::Sender<Sh2SourceKey>,
) -> Result<PendingGpuJob, String> {
    let params_buffer = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SH2 projection params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let output = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("SH2 projection output"),
        size: SH2_OUTPUT_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("SH2 projection readback"),
        size: SH2_OUTPUT_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: output.as_entire_binding(),
        },
    ];
    for (i, binding) in extra_bindings.iter().enumerate() {
        let binding_index = i as u32 + 1;
        let resource = match binding {
            ProjectionBinding::TextureView(view) => wgpu::BindingResource::TextureView(view),
            ProjectionBinding::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
        };
        entries.push(wgpu::BindGroupEntry {
            binding: binding_index,
            resource,
        });
    }
    entries.sort_by_key(|entry| entry.binding);
    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SH2 projection bind group"),
        layout: &pipeline.layout,
        entries: &entries,
    });

    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SH2 projection encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SH2 projection"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, SH2_OUTPUT_BYTES);

    let tx = submit_done_tx.clone();
    let key_for_callback = key;
    gpu.submit_frame_batch_with_callbacks(
        vec![encoder.finish()],
        None,
        None,
        vec![Box::new(move || {
            let _ = tx.send(key_for_callback);
        })],
    );

    Ok(PendingGpuJob {
        staging,
        _output: output,
        _bind_group: bind_group,
        _buffers: vec![params_buffer],
        submit_done: false,
        map_recv: None,
        age_frames: 0,
    })
}

/// Either a synchronous CPU result or a GPU source to project.
enum Sh2ResolvedSource {
    /// CPU-computed SH2.
    Cpu(Box<RenderSH2>),
    /// GPU-computed SH2 source.
    Gpu(GpuSh2Source),
    /// Source is expected to become available later.
    Postpone,
}

/// Compact task header parsed out of a shared-memory row.
#[derive(Clone, Copy, Debug)]
struct TaskHeader {
    /// Host renderable index for the SH2 component.
    renderable_index: i32,
    /// Reflection-probe renderable index referenced by this SH2 task.
    reflection_probe_renderable_index: i32,
}

/// Immediate task answer to write into shared memory.
struct TaskAnswer {
    /// Result status.
    result: ComputeResult,
    /// Optional SH2 payload for computed rows.
    data: Option<RenderSH2>,
}

impl TaskAnswer {
    /// Creates a status-only answer.
    fn status(result: ComputeResult) -> Self {
        Self { result, data: None }
    }

    /// Creates a computed answer with SH2 data.
    fn computed(data: RenderSH2) -> Self {
        Self {
            result: ComputeResult::Computed,
            data: Some(data),
        }
    }
}

/// Resolves a host task into a cache key and source payload.
fn resolve_task_source(
    scene: &SceneCoordinator,
    materials: &crate::backend::MaterialSystem,
    assets: &crate::backend::AssetTransferQueue,
    render_space_id: i32,
    task: TaskHeader,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if task.renderable_index < 0 || task.reflection_probe_renderable_index < 0 {
        return None;
    }
    let space = scene.space(RenderSpaceId(render_space_id))?;
    let probe = space
        .reflection_probes
        .get(task.reflection_probe_renderable_index as usize)?;
    let state = probe.state;
    if state.clear_flags == ReflectionProbeClear::Color {
        let color = state.background_color * state.intensity.max(0.0);
        let key = Sh2SourceKey::ConstantColor {
            render_space_id,
            color_bits: vec4_bits(color),
        };
        return Some((
            key,
            Sh2ResolvedSource::Cpu(Box::new(constant_color_sh2(color.truncate()))),
        ));
    }

    if state.r#type == ReflectionProbeType::Baked {
        if state.cubemap_asset_id < 0 {
            return None;
        }
        let Some(cubemap) = assets.cubemap_pool.get_texture(state.cubemap_asset_id) else {
            return Some((
                Sh2SourceKey::Cubemap {
                    render_space_id,
                    asset_id: state.cubemap_asset_id,
                    size: 0,
                    resident_mips: 0,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    material_generation: 0,
                },
                Sh2ResolvedSource::Postpone,
            ));
        };
        if cubemap.mip_levels_resident == 0 {
            return Some((
                Sh2SourceKey::Cubemap {
                    render_space_id,
                    asset_id: state.cubemap_asset_id,
                    size: cubemap.size,
                    resident_mips: 0,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    material_generation: 0,
                },
                Sh2ResolvedSource::Postpone,
            ));
        }
        let key = Sh2SourceKey::Cubemap {
            render_space_id,
            asset_id: state.cubemap_asset_id,
            size: cubemap.size,
            resident_mips: cubemap.mip_levels_resident,
            sample_size: DEFAULT_SAMPLE_SIZE,
            material_generation: 0,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
                asset_id: state.cubemap_asset_id,
            }),
        ));
    }

    if !reflection_probe_skybox_only(state.flags) {
        return None;
    }
    resolve_skybox_source(
        render_space_id,
        space.skybox_material_asset_id,
        materials,
        assets,
    )
}

/// Resolves an active skybox material into a source payload.
fn resolve_skybox_source(
    render_space_id: i32,
    material_asset_id: i32,
    materials: &crate::backend::MaterialSystem,
    assets: &crate::backend::AssetTransferQueue,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if material_asset_id < 0 {
        return None;
    }
    let store = materials.material_property_store();
    let generation = store.material_generation(material_asset_id);
    let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
    let route_name = shader_route_name(materials, shader_asset_id);
    let route_hash = hash_route_name(route_name.as_deref().unwrap_or(""));
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };
    let registry = materials.property_id_registry();

    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("projection360"))
    {
        return resolve_projection360_source(
            render_space_id,
            store,
            registry,
            assets,
            lookup,
            generation,
        );
    }
    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("gradient"))
    {
        let params = gradient_sky_params(store, registry, lookup);
        let key = Sh2SourceKey::SkyParams {
            render_space_id,
            material_asset_id,
            material_generation: generation,
            sample_size: DEFAULT_SAMPLE_SIZE,
            route_hash,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                params: Box::new(params),
            }),
        ));
    }
    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("procedural"))
    {
        let params = procedural_sky_params(store, registry, lookup);
        let key = Sh2SourceKey::SkyParams {
            render_space_id,
            material_asset_id,
            material_generation: generation,
            sample_size: DEFAULT_SAMPLE_SIZE,
            route_hash,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                params: Box::new(params),
            }),
        ));
    }
    None
}

/// Returns a shader route name or stem for a shader asset id.
fn shader_route_name(
    materials: &crate::backend::MaterialSystem,
    shader_asset_id: i32,
) -> Option<String> {
    let registry = materials.material_registry()?;
    if let Some(stem) = registry.stem_for_shader_asset(shader_asset_id) {
        return Some(stem.to_string());
    }
    registry
        .shader_routes_for_hud()
        .into_iter()
        .find(|(id, _, _)| *id == shader_asset_id)
        .and_then(|(_, _, name)| name)
}

/// Resolves a `Projection360` material to a texture-backed source.
fn resolve_projection360_source(
    render_space_id: i32,
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &crate::backend::AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    generation: u64,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    let main_cube = property_texture(store, registry, lookup, "_MainCube")
        .or_else(|| property_texture(store, registry, lookup, "_Cube"));
    if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
        let Some(cubemap) = assets.cubemap_pool.get_texture(asset_id) else {
            return Some((
                Sh2SourceKey::Cubemap {
                    render_space_id,
                    asset_id,
                    size: 0,
                    resident_mips: 0,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    material_generation: generation,
                },
                Sh2ResolvedSource::Postpone,
            ));
        };
        let key = Sh2SourceKey::Cubemap {
            render_space_id,
            asset_id,
            size: cubemap.size,
            resident_mips: cubemap.mip_levels_resident,
            sample_size: DEFAULT_SAMPLE_SIZE,
            material_generation: generation,
        };
        if cubemap.mip_levels_resident == 0 {
            return Some((key, Sh2ResolvedSource::Postpone));
        }
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap { asset_id }),
        ));
    }

    let main_tex = property_texture(store, registry, lookup, "_MainTex")
        .or_else(|| property_texture(store, registry, lookup, "_Tex"));
    match main_tex {
        Some((asset_id, HostTextureAssetKind::Texture2D)) => {
            let Some(tex) = assets.texture_pool.get_texture(asset_id) else {
                return Some((
                    Sh2SourceKey::EquirectTexture2D {
                        render_space_id,
                        asset_id,
                        width: 0,
                        height: 0,
                        resident_mips: 0,
                        sample_size: DEFAULT_SAMPLE_SIZE,
                        material_generation: generation,
                    },
                    Sh2ResolvedSource::Postpone,
                ));
            };
            let key = Sh2SourceKey::EquirectTexture2D {
                render_space_id,
                asset_id,
                width: tex.width,
                height: tex.height,
                resident_mips: tex.mip_levels_resident,
                sample_size: DEFAULT_SAMPLE_SIZE,
                material_generation: generation,
            };
            if tex.mip_levels_resident == 0 {
                return Some((key, Sh2ResolvedSource::Postpone));
            }
            Some((
                key,
                Sh2ResolvedSource::Gpu(GpuSh2Source::EquirectTexture2D { asset_id }),
            ))
        }
        Some((asset_id, HostTextureAssetKind::Cubemap)) => {
            let Some(cubemap) = assets.cubemap_pool.get_texture(asset_id) else {
                return Some((
                    Sh2SourceKey::Cubemap {
                        render_space_id,
                        asset_id,
                        size: 0,
                        resident_mips: 0,
                        sample_size: DEFAULT_SAMPLE_SIZE,
                        material_generation: generation,
                    },
                    Sh2ResolvedSource::Postpone,
                ));
            };
            let key = Sh2SourceKey::Cubemap {
                render_space_id,
                asset_id,
                size: cubemap.size,
                resident_mips: cubemap.mip_levels_resident,
                sample_size: DEFAULT_SAMPLE_SIZE,
                material_generation: generation,
            };
            if cubemap.mip_levels_resident == 0 {
                return Some((key, Sh2ResolvedSource::Postpone));
            }
            Some((
                key,
                Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap { asset_id }),
            ))
        }
        _ => None,
    }
}

/// Reads a packed texture property by host name.
fn property_texture(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> Option<(i32, HostTextureAssetKind)> {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Texture(packed)) => unpack_host_texture_packed(*packed),
        _ => None,
    }
}

/// Builds parameter payload for a procedural sky material.
fn procedural_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> Sh2ProjectParams {
    let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
    params.color0 = property_float4(store, registry, lookup, "_SkyTint", [0.5, 0.5, 0.5, 1.0]);
    params.color1 = property_float4(
        store,
        registry,
        lookup,
        "_GroundColor",
        [0.35, 0.35, 0.35, 1.0],
    );
    params.direction = property_float4(
        store,
        registry,
        lookup,
        "_SunDirection",
        [0.0, 1.0, 0.0, 0.0],
    );
    let exposure = property_float(store, registry, lookup, "_Exposure", 1.0);
    let sun_size = property_float(store, registry, lookup, "_SunSize", 0.04);
    params.scalars = [exposure, sun_size, 0.0, 0.0];
    params.gradient_color0[0] =
        property_float4(store, registry, lookup, "_SunColor", [1.0, 0.95, 0.85, 1.0]);
    params
}

/// Builds parameter payload for a gradient sky material.
fn gradient_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> Sh2ProjectParams {
    let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
    params.color0 = property_float4(store, registry, lookup, "_BaseColor", [0.0, 0.0, 0.0, 1.0]);
    params.dirs_spread = array16_from_property(store, registry, lookup, "_DirsSpread");
    params.gradient_color0 = array16_from_property(store, registry, lookup, "_Color0");
    params.gradient_color1 = array16_from_property(store, registry, lookup, "_Color1");
    params.gradient_params = array16_from_property(store, registry, lookup, "_Params");
    params.gradient_count = property_float(store, registry, lookup, "_Gradients", 0.0)
        .round()
        .clamp(0.0, 16.0) as u32;
    if params.gradient_count == 0 {
        params.gradient_count = params
            .dirs_spread
            .iter()
            .position(|v| v.iter().all(|c| c.abs() < 1e-6))
            .unwrap_or(16) as u32;
    }
    params
}

/// Reads a float material property.
fn property_float(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: f32,
) -> f32 {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float(v)) => *v,
        _ => fallback,
    }
}

/// Reads a float4 material property.
fn property_float4(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: [f32; 4],
) -> [f32; 4] {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float4(v)) => *v,
        _ => fallback,
    }
}

/// Reads up to sixteen float4 rows from a material property.
fn array16_from_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> [[f32; 4]; 16] {
    let pid = registry.intern(name);
    let mut out = [[0.0; 4]; 16];
    if let Some(MaterialPropertyValue::Float4Array(values)) = store.get_merged(lookup, pid) {
        for (dst, src) in out.iter_mut().zip(values.iter()) {
            *dst = *src;
        }
    }
    out
}

/// Bit pattern for a `Vec4`.
fn vec4_bits(v: Vec4) -> [u32; 4] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), v.w.to_bits()]
}

/// Hashes a route name into a stable source-key discriminator.
fn hash_route_name(route: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    route.hash(&mut h);
    h.finish()
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

/// Stride of a host SH2 task row.
fn task_stride() -> usize {
    std::mem::size_of::<ReflectionProbeSH2Task>()
}

/// Reads the two index fields from one SH2 task row.
fn read_task_header(bytes: &[u8], offset: usize) -> Option<TaskHeader> {
    let renderable_index = read_i32_le(bytes.get(offset..offset + 4)?)?;
    let probe_index = read_i32_le(bytes.get(offset + 4..offset + 8)?)?;
    Some(TaskHeader {
        renderable_index,
        reflection_probe_renderable_index: probe_index,
    })
}

/// Reads a little-endian `i32` from a four-byte slice.
fn read_i32_le(bytes: &[u8]) -> Option<i32> {
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(i32::from_le_bytes(arr))
}

/// Writes a task answer into a shared-memory row.
fn write_task_answer(bytes: &mut [u8], offset: usize, answer: TaskAnswer) {
    const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
    const DATA_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result_data);
    if let Some(mut data) = answer.data {
        let Some(slot) =
            bytes.get_mut(offset + DATA_OFFSET..offset + DATA_OFFSET + RENDER_SH2_HOST_ROW_BYTES)
        else {
            return;
        };
        let mut packer = MemoryPacker::new(slot);
        data.pack(&mut packer);
    }
    if let Some(slot) = bytes.get_mut(offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4) {
        slot.copy_from_slice(&(answer.result as i32).to_le_bytes());
    }
}

/// Debug helper that asserts every active row has been moved out of `Scheduled`.
fn debug_assert_no_scheduled_rows(bytes: &[u8]) {
    #[cfg(debug_assertions)]
    {
        const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
        let mut offset = 0usize;
        while offset + task_stride() <= bytes.len() {
            let Some(task) = read_task_header(bytes, offset) else {
                break;
            };
            if task.renderable_index < 0 {
                break;
            }
            let Some(result_bytes) = bytes.get(offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4)
            else {
                break;
            };
            let Some(result) = read_i32_le(result_bytes) else {
                break;
            };
            debug_assert_ne!(result, ComputeResult::Scheduled as i32);
            offset += task_stride();
        }
    }
}

/// Reads a mapped staging buffer into a RenderSH2 payload.
fn read_sh2_from_staging(staging: &wgpu::Buffer) -> Option<RenderSH2> {
    let mapped = staging.slice(..).get_mapped_range();
    let mut coeffs = [[0.0f32; 4]; 9];
    for (i, chunk) in mapped.chunks_exact(16).take(9).enumerate() {
        coeffs[i] = [
            f32::from_le_bytes(chunk.get(0..4)?.try_into().ok()?),
            f32::from_le_bytes(chunk.get(4..8)?.try_into().ok()?),
            f32::from_le_bytes(chunk.get(8..12)?.try_into().ok()?),
            f32::from_le_bytes(chunk.get(12..16)?.try_into().ok()?),
        ];
    }
    drop(mapped);
    staging.unmap();
    Some(RenderSH2 {
        sh0: Vec3::new(coeffs[0][0], coeffs[0][1], coeffs[0][2]),
        sh1: Vec3::new(coeffs[1][0], coeffs[1][1], coeffs[1][2]),
        sh2: Vec3::new(coeffs[2][0], coeffs[2][1], coeffs[2][2]),
        sh3: Vec3::new(coeffs[3][0], coeffs[3][1], coeffs[3][2]),
        sh4: Vec3::new(coeffs[4][0], coeffs[4][1], coeffs[4][2]),
        sh5: Vec3::new(coeffs[5][0], coeffs[5][1], coeffs[5][2]),
        sh6: Vec3::new(coeffs[6][0], coeffs[6][1], coeffs[6][2]),
        sh7: Vec3::new(coeffs[7][0], coeffs[7][1], coeffs[7][2]),
        sh8: Vec3::new(coeffs[8][0], coeffs[8][1], coeffs[8][2]),
    })
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
