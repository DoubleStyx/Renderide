//! Host camera render-task queue, offscreen render, GPU readback, and IPC writeback.

use std::sync::Arc;
use std::time::Duration;

use hashbrown::HashSet;

use crate::backend::RenderBackend;
use crate::camera::{ViewId, camera_render_task_world_matrix, host_camera_frame_for_render_task};
use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::render_graph::{
    FrameViewClear, GraphExecuteError, OffscreenWriteTarget, RenderPathProfile, ViewPostProcessing,
};
use crate::scene::{RenderSpaceId, RenderSpaceView, SceneCoordinator};
use crate::shared::{CameraRenderParameters, CameraRenderTask, RenderingContext, TextureFormat};
use crate::world_mesh::{
    CameraTransformDrawFilter, ViewLayerPolicy, ViewRenderSpaceScope,
    WorldMeshDrawCollectParallelism,
};

use super::super::RendererRuntime;
use super::super::frame::schedule::{
    CpuRenderSchedule, RenderScheduleKind, execute_one_shot_view_plans,
};
use super::super::frame::view_plan::{
    FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget, OffscreenTargetHandles,
};
use super::readback::{
    AwaitBufferMapError, LinearTextureReadbackPlan, LinearTextureReadbackPlanError,
    await_buffer_map,
};

mod alpha_coverage;
mod camera360;
mod result_write;
#[cfg(test)]
mod tests;

pub(in crate::runtime) use result_write::zero_camera_render_task_results;
use result_write::{output_byte_count, write_camera_task_result, zero_task_result};

/// Maximum time to wait for the blocking task readback callback after `device.poll`.
const CAMERA_READBACK_TIMEOUT: Duration = Duration::from_secs(5);
/// Color attachment format used for CPU camera readback tasks.
const CAMERA_TASK_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Bytes per texel copied from the readback color target.
pub(super) const RGBA8_BYTES_PER_PIXEL: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CameraTaskOutputFormat {
    Argb32,
    Rgba32,
    Bgra32,
    Rgb24,
}

impl CameraTaskOutputFormat {
    pub(super) fn from_texture_format(format: TextureFormat) -> Option<Self> {
        match format {
            TextureFormat::ARGB32 => Some(Self::Argb32),
            TextureFormat::RGBA32 => Some(Self::Rgba32),
            TextureFormat::BGRA32 => Some(Self::Bgra32),
            TextureFormat::RGB24 => Some(Self::Rgb24),
            _ => None,
        }
    }

    pub(super) const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Argb32 | Self::Rgba32 | Self::Bgra32 => 4,
            Self::Rgb24 => 3,
        }
    }

    pub(super) const fn needs_alpha_coverage_repair(self) -> bool {
        matches!(self, Self::Argb32 | Self::Rgba32 | Self::Bgra32)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CameraTaskExtent {
    pub(super) width: u32,
    pub(super) height: u32,
}

impl CameraTaskExtent {
    fn from_parameters(parameters: &CameraRenderParameters) -> Result<Self, CameraReadbackError> {
        let width = u32::try_from(parameters.resolution.x).map_err(|_err| {
            CameraReadbackError::InvalidExtent {
                width: parameters.resolution.x,
                height: parameters.resolution.y,
            }
        })?;
        let height = u32::try_from(parameters.resolution.y).map_err(|_err| {
            CameraReadbackError::InvalidExtent {
                width: parameters.resolution.x,
                height: parameters.resolution.y,
            }
        })?;
        if width == 0 || height == 0 {
            return Err(CameraReadbackError::InvalidExtent {
                width: parameters.resolution.x,
                height: parameters.resolution.y,
            });
        }
        Ok(Self { width, height })
    }

    const fn tuple(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum CameraReadbackError {
    #[error("CameraRenderTask missing parameters")]
    MissingParameters,
    #[error("CameraRenderTask render space {0} is missing")]
    MissingRenderSpace(i32),
    #[error("CameraRenderTask render space {0} is inactive")]
    InactiveRenderSpace(i32),
    #[error("CameraRenderTask extent {width}x{height} is invalid")]
    InvalidExtent { width: i32, height: i32 },
    #[error("CameraRenderTask extent {width}x{height} exceeds max_texture_dimension_2d={max}")]
    ExtentExceedsLimit { width: u32, height: u32, max: u32 },
    #[error("CameraRenderTask format {0:?} is not supported for readback")]
    UnsupportedFormat(TextureFormat),
    #[error(
        "CameraRenderTask Camera360 requires 6 texture array layers but max_texture_array_layers={max}"
    )]
    CubemapArrayLayersUnsupported { max: u32 },
    #[error("CameraRenderTask readback buffer {size} bytes exceeds device max_buffer_size={max}")]
    ReadbackBufferTooLarge { size: u64, max: u64 },
    #[error("CameraRenderTask output byte count overflow")]
    OutputByteCountOverflow,
    #[error("CameraRenderTask mapped readback is too small: need {required} bytes, got {actual}")]
    MappedReadbackTooSmall { required: usize, actual: usize },
    #[error(
        "CameraRenderTask result shared-memory descriptor is too small: need {required} bytes, got {actual}"
    )]
    ResultDescriptorTooSmall { required: usize, actual: usize },
    #[error("CameraRenderTask result shared-memory descriptor could not be mapped")]
    SharedMemoryMapFailed,
    #[error("CameraRenderTask render graph failed: {0}")]
    Graph(#[from] GraphExecuteError),
    #[error("device lost during CameraRenderTask readback poll: {0}")]
    DeviceLost(String),
    #[error("CameraRenderTask map_async timed out")]
    ReadbackTimeout,
    #[error("CameraRenderTask map_async failed: {0}")]
    Map(String),
}

impl From<AwaitBufferMapError> for CameraReadbackError {
    fn from(err: AwaitBufferMapError) -> Self {
        match err {
            AwaitBufferMapError::DeviceLost(s) => Self::DeviceLost(s),
            AwaitBufferMapError::Timeout => Self::ReadbackTimeout,
            AwaitBufferMapError::Map(s) => Self::Map(s),
        }
    }
}

impl From<LinearTextureReadbackPlanError> for CameraReadbackError {
    fn from(err: LinearTextureReadbackPlanError) -> Self {
        match err {
            LinearTextureReadbackPlanError::InvalidExtent { width, height } => {
                Self::InvalidExtent {
                    width: i32::try_from(width).unwrap_or(i32::MAX),
                    height: i32::try_from(height).unwrap_or(i32::MAX),
                }
            }
            LinearTextureReadbackPlanError::InvalidBytesPerPixel
            | LinearTextureReadbackPlanError::OutputByteCountOverflow => {
                Self::OutputByteCountOverflow
            }
            LinearTextureReadbackPlanError::ReadbackBufferTooLarge { size, max } => {
                Self::ReadbackBufferTooLarge { size, max }
            }
            LinearTextureReadbackPlanError::MappedReadbackTooSmall { required, actual } => {
                Self::MappedReadbackTooSmall { required, actual }
            }
        }
    }
}

#[derive(Default)]
struct CameraReadbackDrainStats {
    completed: u64,
    failed: u64,
}

struct CameraTaskTargets {
    color_texture: Arc<wgpu::Texture>,
    color_view: Arc<wgpu::TextureView>,
    depth_texture: Arc<wgpu::Texture>,
    depth_view: Arc<wgpu::TextureView>,
    color_format: wgpu::TextureFormat,
    extent: CameraTaskExtent,
}

impl CameraTaskTargets {
    fn create(gpu: &GpuContext, extent: CameraTaskExtent) -> Result<Self, CameraReadbackError> {
        let max_dim = gpu.limits().max_texture_dimension_2d();
        if extent.width > max_dim || extent.height > max_dim {
            return Err(CameraReadbackError::ExtentExceedsLimit {
                width: extent.width,
                height: extent.height,
                max: max_dim,
            });
        }
        let size = wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: 1,
        };
        let color_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-camera-task-color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAMERA_TASK_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let color_view =
            Arc::new(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        crate::profiling::note_resource_churn!(TextureView, "runtime::camera_task_color_view");

        let depth_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-camera-task-depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::gpu::main_forward_depth_stencil_format(gpu.device().features()),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let depth_view =
            Arc::new(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        crate::profiling::note_resource_churn!(TextureView, "runtime::camera_task_depth_view");

        Ok(Self {
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            color_format: CAMERA_TASK_COLOR_FORMAT,
            extent,
        })
    }

    fn to_offscreen_handles(&self) -> OffscreenTargetHandles {
        OffscreenTargetHandles::new(
            OffscreenWriteTarget::Untracked,
            self.color_texture.as_ref().clone(),
            self.color_view.as_ref().clone(),
            self.depth_texture.as_ref().clone(),
            self.depth_view.as_ref().clone(),
            self.extent.tuple(),
            self.color_format,
        )
    }
}

impl RendererRuntime {
    /// Appends host camera render tasks to the pre-begin-frame GPU readback queue.
    pub(in crate::runtime) fn queue_camera_render_tasks(&mut self, tasks: &[CameraRenderTask]) {
        profiling::scope!("camera_task::queue");
        if tasks.is_empty() {
            return;
        }
        self.tick_state
            .submit_completion_work
            .queue_camera_tasks(tasks);
        self.set_pending_camera_readbacks(self.tick_state.submit_completion_work.camera_count());
    }

    /// Drains queued camera readback tasks before the next host begin-frame is sent.
    pub fn drain_camera_render_tasks(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("camera_task::drain");
        let tasks = self.tick_state.submit_completion_work.take_camera_tasks();
        if tasks.is_empty() {
            self.set_pending_camera_readbacks(0);
            return;
        }
        self.sync_master_msaa(gpu);
        self.set_pending_camera_readbacks(tasks.len());
        let frame_time_seconds = self.tick_state.frame_time_seconds();
        let mut stats = CameraReadbackDrainStats::default();
        let RendererRuntime {
            frontend,
            backend,
            scene,
            host_camera,
            diagnostics,
            ..
        } = self;
        let Some(shm) = frontend.shared_memory_mut() else {
            logger::warn!(
                "dropping {} CameraRenderTask readback(s): shared memory is unavailable",
                tasks.len()
            );
            stats.failed = tasks.len() as u64;
            diagnostics.set_pending_camera_readbacks(0);
            diagnostics.note_camera_readback_results(stats.completed, stats.failed);
            return;
        };

        let base_camera = &*host_camera;
        let total = tasks.len();
        for (index, task) in tasks.into_iter().enumerate() {
            let task_index = i32::try_from(index).unwrap_or(i32::MAX);
            match render_camera_task(CameraTaskRenderCtx {
                gpu: &mut *gpu,
                backend: &mut *backend,
                scene,
                base_camera,
                shm: &mut *shm,
                task_index,
                task: &task,
                frame_time_seconds,
            }) {
                Ok(()) => stats.completed = stats.completed.saturating_add(1),
                Err(error) => {
                    stats.failed = stats.failed.saturating_add(1);
                    logger::warn!(
                        "CameraRenderTask readback failed for render_space_id={} task_index={task_index}: {error}",
                        task.render_space_id
                    );
                    zero_task_result(shm, &task);
                }
            }
            let remaining = total.saturating_sub(index.saturating_add(1));
            diagnostics.set_pending_camera_readbacks(remaining);
        }
        diagnostics.set_pending_camera_readbacks(0);
        diagnostics.note_camera_readback_results(stats.completed, stats.failed);
    }
}

struct CameraTaskRenderCtx<'a> {
    gpu: &'a mut GpuContext,
    backend: &'a mut RenderBackend,
    scene: &'a SceneCoordinator,
    base_camera: &'a crate::camera::HostCameraFrame,
    shm: &'a mut SharedMemoryAccessor,
    task_index: i32,
    task: &'a CameraRenderTask,
    frame_time_seconds: f32,
}

fn render_camera_task(ctx: CameraTaskRenderCtx<'_>) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::render_one");
    if ctx
        .task
        .parameters
        .as_ref()
        .is_some_and(camera360::camera_render_parameters_request_camera360)
    {
        return camera360::render_camera360_task(ctx);
    }
    let planned = plan_camera_task(
        ctx.gpu,
        ctx.scene,
        ctx.base_camera,
        ctx.task_index,
        ctx.task,
        ctx.frame_time_seconds,
    )?;
    render_camera_task_offscreen(ctx.gpu, ctx.backend, ctx.scene, planned.plan)?;
    if planned.output_format.needs_alpha_coverage_repair() {
        alpha_coverage::apply_camera_task_alpha_coverage(ctx.gpu, &planned.targets);
    }
    let rgba = readback_camera_task_texture(ctx.gpu, planned.targets.color_texture.as_ref())?;
    write_camera_task_result(
        ctx.shm,
        ctx.task,
        planned.output_format,
        planned.targets.extent,
        &rgba,
    )
}

struct PlannedCameraTask {
    plan: FrameViewPlan<'static>,
    targets: CameraTaskTargets,
    output_format: CameraTaskOutputFormat,
}

fn active_camera_task_space(
    scene: &SceneCoordinator,
    render_space_id: i32,
) -> Result<RenderSpaceView<'_>, CameraReadbackError> {
    let id = RenderSpaceId(render_space_id);
    let Some(space) = scene.space(id) else {
        return Err(CameraReadbackError::MissingRenderSpace(render_space_id));
    };
    if !space.is_active() {
        return Err(CameraReadbackError::InactiveRenderSpace(render_space_id));
    }
    Ok(space)
}

fn plan_camera_task(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    base_camera: &crate::camera::HostCameraFrame,
    task_index: i32,
    task: &CameraRenderTask,
    frame_time_seconds: f32,
) -> Result<PlannedCameraTask, CameraReadbackError> {
    profiling::scope!("camera_task::plan");
    let parameters = task
        .parameters
        .as_ref()
        .ok_or(CameraReadbackError::MissingParameters)?;
    let output_format = CameraTaskOutputFormat::from_texture_format(parameters.texture_format)
        .ok_or(CameraReadbackError::UnsupportedFormat(
            parameters.texture_format,
        ))?;
    let extent = CameraTaskExtent::from_parameters(parameters)?;
    let required = output_byte_count(extent, output_format)?;
    let actual = usize::try_from(task.result_data.length).unwrap_or(0);
    if actual < required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall { required, actual });
    }
    let render_space_id = RenderSpaceId(task.render_space_id);
    active_camera_task_space(scene, task.render_space_id)?;
    let targets = CameraTaskTargets::create(gpu, extent)?;
    let camera_world_matrix = camera_render_task_world_matrix(task.position, task.rotation);
    let host_camera = host_camera_frame_for_render_task(
        base_camera,
        parameters,
        extent.tuple(),
        camera_world_matrix,
    );
    let filter = draw_filter_from_camera_render_task(task);
    let mut plan = FrameViewPlan::new(
        &host_camera,
        FrameViewPlanParams {
            render_context: RenderingContext::RenderToAsset,
            frame_time_seconds,
            view_id: ViewId::camera_render_task(render_space_id, task_index),
            viewport_px: extent.tuple(),
            clear: FrameViewClear::from_camera_render_parameters(parameters),
            profile: RenderPathProfile::camera_readback(camera_render_task_post_processing(
                parameters,
            )),
            target: FrameViewPlanTarget::offscreen(targets.to_offscreen_handles()),
        },
    );
    plan.draw_filter = Some(filter);
    plan.transform_filter_space = Some(render_space_id);
    plan.render_space_scope = camera_render_task_scope(task, render_space_id);
    plan.layer_policy = camera_render_task_layer_policy(parameters);
    Ok(PlannedCameraTask {
        plan,
        targets,
        output_format,
    })
}

fn camera_render_task_post_processing(parameters: &CameraRenderParameters) -> ViewPostProcessing {
    ViewPostProcessing::from_camera_render_parameters(parameters)
}

fn camera_render_task_layer_policy(parameters: &CameraRenderParameters) -> ViewLayerPolicy {
    ViewLayerPolicy::camera(parameters.render_private_ui)
}

fn camera_render_task_scope(
    task: &CameraRenderTask,
    render_space_id: RenderSpaceId,
) -> ViewRenderSpaceScope {
    if task.only_render_list.is_empty() {
        ViewRenderSpaceScope::AllActive
    } else {
        ViewRenderSpaceScope::single(render_space_id)
    }
}

fn draw_filter_from_camera_render_task(task: &CameraRenderTask) -> CameraTransformDrawFilter {
    if task.only_render_list.is_empty() {
        CameraTransformDrawFilter {
            only: None,
            exclude: task.exclude_render_list.iter().copied().collect(),
        }
    } else {
        CameraTransformDrawFilter {
            only: Some(
                task.only_render_list
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>(),
            ),
            exclude: task.exclude_render_list.iter().copied().collect(),
        }
    }
}

fn render_camera_task_offscreen(
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    plan: FrameViewPlan<'static>,
) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::offscreen_render");
    execute_one_shot_view_plans(
        CpuRenderSchedule::new(RenderScheduleKind::CameraTask),
        gpu,
        backend,
        scene,
        vec![plan],
        WorldMeshDrawCollectParallelism::Full,
    )
    .map_err(CameraReadbackError::Graph)
}

fn readback_camera_task_texture(
    gpu: &mut GpuContext,
    color_texture: &wgpu::Texture,
) -> Result<Vec<u8>, CameraReadbackError> {
    profiling::scope!("camera_task::gpu_copy_and_map");
    let plan = LinearTextureReadbackPlan::new(
        color_texture.size(),
        RGBA8_BYTES_PER_PIXEL as u32,
        gpu.limits().max_buffer_size(),
    )?;
    let readback = create_readback_buffer(gpu, plan);
    submit_texture_to_buffer_copy(gpu, color_texture, plan, &readback);
    let slice = readback.slice(..);
    {
        profiling::scope!("camera_task::map_readback");
        await_buffer_map(slice, gpu.device(), CAMERA_READBACK_TIMEOUT)?;
    }
    let tight = {
        profiling::scope!("camera_task::copy_padded_rows");
        let view = slice.get_mapped_range();
        plan.copy_mapped_rows_to_tight(&view)?
    };
    readback.unmap();
    Ok(tight)
}

fn create_readback_buffer(gpu: &GpuContext, plan: LinearTextureReadbackPlan) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("renderide-camera-task-readback"),
        size: plan.buffer_size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(Buffer, "runtime::camera_task_readback_buffer");
    buffer
}

fn submit_texture_to_buffer_copy(
    gpu: &mut GpuContext,
    color_texture: &wgpu::Texture,
    plan: LinearTextureReadbackPlan,
    readback: &wgpu::Buffer,
) {
    profiling::scope!("camera_task::gpu_copy");
    gpu.flush_driver();
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("renderide-camera-task-readback"),
        });
    let copy_query = gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_query("camera_task::texture_to_buffer_copy", &mut encoder));
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: readback,
            layout: plan.copy_buffer_layout(),
        },
        plan.copy_extent(),
    );
    if let Some(query) = copy_query
        && let Some(prof) = gpu.gpu_profiler_mut()
    {
        prof.end_query(&mut encoder, query);
        prof.resolve_queries(&mut encoder);
    }
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::camera_task_readback");
        encoder.finish()
    };
    gpu.queue().submit(std::iter::once(command_buffer));
}
