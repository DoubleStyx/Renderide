//! Host reflection-probe cubemap bake tasks, offscreen face rendering, readback, and IPC results.

use std::sync::Arc;

use hashbrown::HashSet;

use crate::backend::RenderBackend;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::{CUBEMAP_ARRAY_LAYERS, GpuContext};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::render_graph::{GraphExecuteError, OffscreenSampleCountPolicy};
use crate::scene::{
    ReflectionProbeOnChangesRenderRequest, RenderSpaceId, SceneCoordinator,
    reflection_probe_skybox_only,
};
use crate::shared::{
    FrameSubmitData, ReflectionProbeRenderResult, ReflectionProbeRenderTask, ReflectionProbeState,
    RendererCommand, RenderingContext,
};
use crate::skybox::ibl_cache::{SkyboxIblConvolver, mip_levels_for_edge};
use crate::world_mesh::{CameraTransformDrawFilter, WorldMeshDrawCollectParallelism};

mod face;
mod onchanges;
mod readback;
#[cfg(test)]
mod tests;

pub(in crate::runtime) use onchanges::ActiveOnChangesReflectionProbeCapture;
use onchanges::same_onchanges_probe;
use readback::{
    compute_probe_readback_layout, readback_reflection_probe_cube, write_probe_task_result,
    zero_probe_task_result,
};

use face::{
    CUBE_FACE_COUNT, ProbeCubeFace, clear_from_reflection_probe_state,
    draw_filter_from_reflection_probe_state, face_view_desc, host_camera_frame_for_probe_face,
    reflection_probe_bake_post_processing,
};

use super::super::RendererRuntime;
use super::super::frame::extract::{ExtractedFrame, PreparedViews};
use super::super::frame::view_plan::{FrameViewPlan, FrameViewPlanTarget, OffscreenRtHandles};
use super::super::state::tick::QueuedReflectionProbeRenderTask;

const RGBA16F_BYTES_PER_PIXEL: usize = 8;
const RGBA8_BYTES_PER_PIXEL: usize = 4;
const PROBE_TASK_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// MSAA policy used by reflection-probe utility captures.
const REFLECTION_PROBE_SAMPLE_COUNT_POLICY: OffscreenSampleCountPolicy =
    OffscreenSampleCountPolicy::SingleSample;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeOutputFormat {
    Rgba8,
    Rgba16Float,
}

impl ProbeOutputFormat {
    const fn from_hdr(hdr: bool) -> Self {
        if hdr { Self::Rgba16Float } else { Self::Rgba8 }
    }

    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 => RGBA8_BYTES_PER_PIXEL,
            Self::Rgba16Float => RGBA16F_BYTES_PER_PIXEL,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProbeTaskExtent {
    size: u32,
    mip_levels: u32,
}

impl ProbeTaskExtent {
    fn from_task(task: &ReflectionProbeRenderTask) -> Result<Self, ReflectionProbeBakeError> {
        Self::from_size(task.size)
    }

    fn from_size(size: i32) -> Result<Self, ReflectionProbeBakeError> {
        let size =
            u32::try_from(size).map_err(|_err| ReflectionProbeBakeError::InvalidSize { size })?;
        if size == 0 {
            return Err(ReflectionProbeBakeError::InvalidSize { size: 0 });
        }
        Ok(Self {
            size,
            mip_levels: mip_levels_for_edge(size),
        })
    }

    const fn tuple(self) -> (u32, u32) {
        (self.size, self.size)
    }
}

#[derive(Clone, Debug)]
struct ProbeMipReadback {
    face: ProbeCubeFace,
    mip: u32,
    extent: u32,
    bytes_per_row_tight: u32,
    bytes_per_row_padded: u32,
    buffer_offset: u64,
    host_origin: usize,
    host_byte_count: usize,
}

#[derive(Clone, Debug)]
struct ProbeReadbackLayout {
    subresources: Vec<ProbeMipReadback>,
    buffer_size: u64,
    output_format: ProbeOutputFormat,
}

#[derive(Debug, thiserror::Error)]
enum ReflectionProbeBakeError {
    #[error("ReflectionProbeRenderTask render space {0} is missing")]
    MissingRenderSpace(i32),
    #[error("ReflectionProbeRenderTask render space {0} is inactive")]
    InactiveRenderSpace(i32),
    #[error("ReflectionProbeRenderTask renderable_index {0} is invalid")]
    InvalidRenderableIndex(i32),
    #[error("ReflectionProbeRenderTask renderable_index {0} was not found")]
    MissingProbe(i32),
    #[error("ReflectionProbeRenderTask probe transform_id {0} is invalid")]
    InvalidProbeTransform(i32),
    #[error("ReflectionProbeRenderTask probe transform_id {0} has no world matrix")]
    MissingProbeTransform(i32),
    #[error("ReflectionProbeRenderTask size {size} is invalid")]
    InvalidSize { size: i32 },
    #[error("ReflectionProbeRenderTask size {size} exceeds max_texture_dimension_2d={max}")]
    SizeExceedsLimit { size: u32, max: u32 },
    #[error(
        "ReflectionProbeRenderTask requires 6 texture array layers but max_texture_array_layers={max}"
    )]
    CubemapArrayLayersUnsupported { max: u32 },
    #[error("ReflectionProbeRenderTask mip_origins has {actual} faces; expected 6")]
    InvalidMipOriginFaces { actual: usize },
    #[error("ReflectionProbeRenderTask mip_origins[{face}] has {actual} mips; expected {expected}")]
    InvalidMipOriginCount {
        face: usize,
        expected: usize,
        actual: usize,
    },
    #[error("ReflectionProbeRenderTask mip_origins[{face}][{mip}] is negative: {origin}")]
    NegativeMipOrigin {
        face: usize,
        mip: usize,
        origin: i32,
    },
    #[error(
        "ReflectionProbeRenderTask result shared-memory descriptor is too small: need {required} bytes, got {actual}"
    )]
    ResultDescriptorTooSmall { required: usize, actual: usize },
    #[error("ReflectionProbeRenderTask byte count overflow")]
    OutputByteCountOverflow,
    #[error(
        "ReflectionProbeRenderTask readback buffer {size} bytes exceeds device max_buffer_size={max}"
    )]
    ReadbackBufferTooLarge { size: u64, max: u64 },
    #[error(
        "ReflectionProbeRenderTask mapped readback is too small: need {required} bytes, got {actual}"
    )]
    MappedReadbackTooSmall { required: usize, actual: usize },
    #[error("ReflectionProbeRenderTask result shared-memory descriptor could not be mapped")]
    SharedMemoryMapFailed,
    #[error("ReflectionProbeRenderTask render graph failed: {0}")]
    Graph(#[from] GraphExecuteError),
    #[error("ReflectionProbeRenderTask IBL convolve failed: {0}")]
    Convolve(String),
    #[error("device lost during ReflectionProbeRenderTask readback poll: {0}")]
    DeviceLost(String),
    #[error("ReflectionProbeRenderTask map_async timed out")]
    ReadbackTimeout,
    #[error("ReflectionProbeRenderTask map_async failed: {0}")]
    Map(String),
}

struct ProbeTaskTargets {
    cube_texture: Arc<wgpu::Texture>,
    face_color_views: [Arc<wgpu::TextureView>; CUBE_FACE_COUNT],
    face_depth_textures: [Arc<wgpu::Texture>; CUBE_FACE_COUNT],
    face_depth_views: [Arc<wgpu::TextureView>; CUBE_FACE_COUNT],
    color_format: wgpu::TextureFormat,
    extent: ProbeTaskExtent,
}

impl ProbeTaskTargets {
    fn create(gpu: &GpuContext, extent: ProbeTaskExtent) -> Result<Self, ReflectionProbeBakeError> {
        let max_dim = gpu.limits().max_texture_dimension_2d();
        if extent.size > max_dim {
            return Err(ReflectionProbeBakeError::SizeExceedsLimit {
                size: extent.size,
                max: max_dim,
            });
        }
        if !gpu.limits().array_layers_fit(CUBEMAP_ARRAY_LAYERS) {
            return Err(ReflectionProbeBakeError::CubemapArrayLayersUnsupported {
                max: gpu.limits().max_texture_array_layers(),
            });
        }
        let size = wgpu::Extent3d {
            width: extent.size,
            height: extent.size,
            depth_or_array_layers: CUBEMAP_ARRAY_LAYERS,
        };
        let cube_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-reflection-probe-task-cube"),
            size,
            mip_level_count: extent.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PROBE_TASK_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let face_color_views = std::array::from_fn(|i| {
            Arc::new(cube_texture.create_view(&face_view_desc(
                "renderide-reflection-probe-task-face-color",
                i as u32,
                PROBE_TASK_COLOR_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            )))
        });
        crate::profiling::note_resource_churn!(
            TextureView,
            "runtime::reflection_probe_task_face_color_views"
        );

        let depth_format = crate::gpu::main_forward_depth_stencil_format(gpu.device().features());
        let depth_size = wgpu::Extent3d {
            width: extent.size,
            height: extent.size,
            depth_or_array_layers: 1,
        };
        let face_depth_textures = std::array::from_fn(|_i| {
            Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-reflection-probe-task-face-depth"),
                size: depth_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }))
        });
        let face_depth_views = std::array::from_fn(|i| {
            Arc::new(
                face_depth_textures[i].create_view(&wgpu::TextureViewDescriptor {
                    label: Some("renderide-reflection-probe-task-face-depth"),
                    format: Some(depth_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    aspect: wgpu::TextureAspect::All,
                    ..Default::default()
                }),
            )
        });
        crate::profiling::note_resource_churn!(
            TextureView,
            "runtime::reflection_probe_task_face_depth_views"
        );

        Ok(Self {
            cube_texture,
            face_color_views,
            face_depth_textures,
            face_depth_views,
            color_format: PROBE_TASK_COLOR_FORMAT,
            extent,
        })
    }

    fn to_offscreen_handles(&self, face: ProbeCubeFace) -> OffscreenRtHandles {
        OffscreenRtHandles {
            rt_id: -1,
            color_view: Arc::clone(&self.face_color_views[face.index()]),
            depth_texture: Arc::clone(&self.face_depth_textures[face.index()]),
            depth_view: Arc::clone(&self.face_depth_views[face.index()]),
            color_format: self.color_format,
            sample_count_policy: REFLECTION_PROBE_SAMPLE_COUNT_POLICY,
        }
    }

    fn cube_sample_view(&self) -> Arc<wgpu::TextureView> {
        Arc::new(self.cube_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("renderide-reflection-probe-onchanges-cube-view"),
            format: Some(PROBE_TASK_COLOR_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBEMAP_ARRAY_LAYERS),
        }))
    }

    fn array_sample_view(&self) -> Arc<wgpu::TextureView> {
        Arc::new(self.cube_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("renderide-reflection-probe-onchanges-array-view"),
            format: Some(PROBE_TASK_COLOR_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(self.extent.mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(CUBEMAP_ARRAY_LAYERS),
        }))
    }
}

struct PlannedReflectionProbeTask {
    plans: Vec<FrameViewPlan<'static>>,
    targets: ProbeTaskTargets,
    readback_layout: ProbeReadbackLayout,
}

impl RendererRuntime {
    /// Queues host OnChanges reflection-probe render requests that need scene capture.
    pub(in crate::runtime) fn queue_onchanges_reflection_probe_requests(
        &mut self,
        requests: Vec<ReflectionProbeOnChangesRenderRequest>,
    ) {
        profiling::scope!("reflection_probe_onchanges::queue");
        for request in requests {
            if let Some(active) = self
                .tick_state
                .active_onchanges_reflection_probe_captures
                .iter_mut()
                .find(|active| same_onchanges_probe(active.request, request))
            {
                active.queued_unique_id = Some(request.unique_id);
                continue;
            }
            if let Some(pending) = self
                .tick_state
                .pending_onchanges_reflection_probe_requests
                .iter_mut()
                .find(|pending| {
                    pending.render_space_id == request.render_space_id
                        && pending.renderable_index == request.renderable_index
                })
            {
                pending.unique_id = request.unique_id;
                continue;
            }
            self.tick_state
                .pending_onchanges_reflection_probe_requests
                .push(request);
        }
    }

    /// Appends host reflection-probe cubemap bake tasks to the pre-begin-frame GPU readback queue.
    pub(in crate::runtime) fn queue_reflection_probe_render_tasks(
        &mut self,
        data: &FrameSubmitData,
    ) {
        profiling::scope!("reflection_probe_task::queue");
        let initial = self.tick_state.pending_reflection_probe_render_tasks.len();
        for space in &data.render_spaces {
            let render_space_id = RenderSpaceId(space.id);
            self.tick_state
                .pending_reflection_probe_render_tasks
                .extend(
                    space
                        .reflection_probe_render_tasks
                        .iter()
                        .cloned()
                        .map(|task| QueuedReflectionProbeRenderTask {
                            render_space_id,
                            task,
                        }),
                );
        }
        let added = self
            .tick_state
            .pending_reflection_probe_render_tasks
            .len()
            .saturating_sub(initial);
        if added > 0 {
            logger::debug!(
                "queued {} ReflectionProbeRenderTask bake(s); pending={}",
                added,
                self.tick_state.pending_reflection_probe_render_tasks.len()
            );
        }
    }

    /// Queues failure results for every reflection-probe bake task in a rejected frame submit.
    pub(in crate::runtime) fn queue_failed_reflection_probe_render_task_results(
        &mut self,
        data: &FrameSubmitData,
    ) {
        profiling::scope!("reflection_probe_task::queue_failed_results");
        for space in &data.render_spaces {
            self.tick_state
                .pending_reflection_probe_render_results
                .extend(space.reflection_probe_render_tasks.iter().map(|task| {
                    ReflectionProbeRenderResult {
                        render_task_id: task.render_task_id,
                        success: false,
                    }
                }));
        }
    }

    /// Attempts to flush queued reflection-probe bake results to the background IPC queue.
    pub(in crate::runtime) fn flush_reflection_probe_render_results(&mut self) {
        profiling::scope!("reflection_probe_task::flush_results");
        let RendererRuntime {
            frontend,
            tick_state,
            ..
        } = self;
        let mut ipc = frontend.ipc_mut();
        flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
    }

    /// Drains queued reflection-probe bake tasks before the next host begin-frame is sent.
    pub fn drain_reflection_probe_render_tasks(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("reflection_probe_task::drain");
        let mut tasks = std::mem::take(&mut self.tick_state.pending_reflection_probe_render_tasks);
        self.flush_reflection_probe_render_results();
        if !tasks.is_empty() {
            let RendererRuntime {
                frontend,
                backend,
                scene,
                tick_state,
                host_camera,
                ..
            } = self;
            let base_camera = &*host_camera;
            let (shm, mut ipc) = frontend.transport_pair_mut();
            flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
            if let Some(shm) = shm {
                let mut convolver = SkyboxIblConvolver::new();
                let mut completed = 0u64;
                let mut failed = 0u64;
                for queued in tasks.drain(..) {
                    match render_reflection_probe_task(ReflectionProbeTaskRenderCtx {
                        gpu: &mut *gpu,
                        backend: &mut *backend,
                        scene,
                        base_camera,
                        shm: &mut *shm,
                        convolver: &mut convolver,
                        queued: &queued,
                    }) {
                        Ok(()) => {
                            completed = completed.saturating_add(1);
                            tick_state.pending_reflection_probe_render_results.push(
                                ReflectionProbeRenderResult {
                                    render_task_id: queued.task.render_task_id,
                                    success: true,
                                },
                            );
                        }
                        Err(error) => {
                            failed = failed.saturating_add(1);
                            logger::warn!(
                                "ReflectionProbeRenderTask bake failed for render_space_id={} render_task_id={}: {error}",
                                queued.render_space_id.0,
                                queued.task.render_task_id
                            );
                            zero_probe_task_result(shm, &queued.task);
                            tick_state.pending_reflection_probe_render_results.push(
                                ReflectionProbeRenderResult {
                                    render_task_id: queued.task.render_task_id,
                                    success: false,
                                },
                            );
                        }
                    }
                    flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
                }
                logger::debug!(
                    "drained ReflectionProbeRenderTask bakes: completed={} failed={}",
                    completed,
                    failed
                );
            } else {
                logger::warn!(
                    "dropping {} ReflectionProbeRenderTask bake(s): shared memory is unavailable",
                    tasks.len()
                );
                queue_reflection_probe_failures(
                    tick_state,
                    tasks.iter().map(|queued| &queued.task),
                );
                flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
            }
        }
        self.drain_onchanges_reflection_probe_captures(gpu);
    }
}

struct ReflectionProbeTaskRenderCtx<'a> {
    gpu: &'a mut GpuContext,
    backend: &'a mut RenderBackend,
    scene: &'a SceneCoordinator,
    base_camera: &'a HostCameraFrame,
    shm: &'a mut SharedMemoryAccessor,
    convolver: &'a mut SkyboxIblConvolver,
    queued: &'a QueuedReflectionProbeRenderTask,
}

fn render_reflection_probe_task(
    ctx: ReflectionProbeTaskRenderCtx<'_>,
) -> Result<(), ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::render_one");
    let planned = plan_reflection_probe_task(ctx.gpu, ctx.scene, ctx.base_camera, ctx.queued)?;
    let view_ids = planned
        .plans
        .iter()
        .map(|plan| plan.view_id)
        .collect::<Vec<_>>();
    let render_result =
        render_reflection_probe_faces_offscreen(ctx.gpu, ctx.backend, ctx.scene, planned.plans);
    if let Err(error) = render_result {
        ctx.backend.retire_one_shot_views(&view_ids);
        return Err(error);
    }
    let mapped = match readback_reflection_probe_cube(
        ctx.gpu,
        ctx.convolver,
        planned.targets.cube_texture.as_ref(),
        planned.targets.extent,
        &planned.readback_layout,
    ) {
        Ok(mapped) => mapped,
        Err(error) => {
            ctx.backend.retire_one_shot_views(&view_ids);
            return Err(error);
        }
    };
    ctx.backend.retire_one_shot_views(&view_ids);
    write_probe_task_result(ctx.shm, &ctx.queued.task, &planned.readback_layout, &mapped)
}

fn plan_reflection_probe_task(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    base_camera: &HostCameraFrame,
    queued: &QueuedReflectionProbeRenderTask,
) -> Result<PlannedReflectionProbeTask, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::plan");
    let task = &queued.task;
    let extent = ProbeTaskExtent::from_task(task)?;
    let output_format = ProbeOutputFormat::from_hdr(task.hdr);
    let readback_layout =
        compute_probe_readback_layout(task, extent, output_format, gpu.limits().max_buffer_size())?;
    let space =
        scene
            .space(queued.render_space_id)
            .ok_or(ReflectionProbeBakeError::MissingRenderSpace(
                queued.render_space_id.0,
            ))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            queued.render_space_id.0,
        ));
    }
    let probe_index = usize::try_from(task.renderable_index)
        .map_err(|_err| ReflectionProbeBakeError::InvalidRenderableIndex(task.renderable_index))?;
    let probe = space.reflection_probes().get(probe_index).ok_or(
        ReflectionProbeBakeError::MissingProbe(task.renderable_index),
    )?;
    let transform_index = usize::try_from(probe.transform_id)
        .map_err(|_err| ReflectionProbeBakeError::InvalidProbeTransform(probe.transform_id))?;
    let probe_world = scene
        .world_matrix_for_render_context(
            queued.render_space_id,
            transform_index,
            RenderingContext::RenderToAsset,
            base_camera.head_output_transform,
        )
        .ok_or(ReflectionProbeBakeError::MissingProbeTransform(
            probe.transform_id,
        ))?;
    let targets = ProbeTaskTargets::create(gpu, extent)?;
    let filter = draw_filter_from_reflection_probe_task(task, &probe.state);
    let probe_position = probe_world.col(3).truncate();
    let plans = ProbeCubeFace::ALL
        .iter()
        .copied()
        .map(|face| FrameViewPlan {
            host_camera: host_camera_frame_for_probe_face(
                base_camera,
                probe.state,
                extent.tuple(),
                probe_position,
                face,
            ),
            render_context: RenderingContext::RenderToAsset,
            draw_filter: Some(filter.clone()),
            render_space_filter: Some(queued.render_space_id),
            view_id: ViewId::reflection_probe_render_task(
                queued.render_space_id,
                task.render_task_id,
                face.view_id_face_index(),
            ),
            viewport_px: extent.tuple(),
            clear: clear_from_reflection_probe_state(probe.state),
            post_processing: reflection_probe_bake_post_processing(),
            target: FrameViewPlanTarget::SecondaryRt(targets.to_offscreen_handles(face)),
        })
        .collect();
    Ok(PlannedReflectionProbeTask {
        plans,
        targets,
        readback_layout,
    })
}

fn draw_filter_from_reflection_probe_task(
    task: &ReflectionProbeRenderTask,
    state: &ReflectionProbeState,
) -> CameraTransformDrawFilter {
    if reflection_probe_skybox_only(state.flags) {
        CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: task.exclude_transform_ids.iter().copied().collect(),
        }
    }
}

fn render_reflection_probe_faces_offscreen(
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    plans: Vec<FrameViewPlan<'static>>,
) -> Result<(), ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::offscreen_render");
    let prepared_views = PreparedViews::new(plans, None);
    backend.prepare_lights_for_views(
        scene,
        prepared_views
            .plans()
            .iter()
            .map(FrameViewPlan::light_view_desc),
    );
    let view_perms = prepared_views
        .plans()
        .iter()
        .map(|plan| (plan.render_context(), plan.shader_permutation()))
        .collect::<Vec<_>>();
    let shared =
        backend.extract_frame_shared(scene, WorldMeshDrawCollectParallelism::Full, &view_perms);
    let submit_frame = ExtractedFrame::new(prepared_views, shared)
        .prepare_draws()
        .into_submit_frame();
    submit_frame.execute(gpu, scene, backend)?;
    Ok(())
}

fn queue_reflection_probe_failures<'a>(
    tick_state: &mut super::super::state::tick::RuntimeTickState,
    tasks: impl IntoIterator<Item = &'a ReflectionProbeRenderTask>,
) {
    tick_state
        .pending_reflection_probe_render_results
        .extend(tasks.into_iter().map(|task| ReflectionProbeRenderResult {
            render_task_id: task.render_task_id,
            success: false,
        }));
}

fn flush_reflection_probe_render_results_to_ipc(
    tick_state: &mut super::super::state::tick::RuntimeTickState,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    if tick_state
        .pending_reflection_probe_render_results
        .is_empty()
    {
        return;
    }
    let Some(ipc) = ipc.as_mut() else {
        return;
    };
    let pending = std::mem::take(&mut tick_state.pending_reflection_probe_render_results);
    let mut sent = 0usize;
    for result in &pending {
        if !ipc
            .send_background_reliable(RendererCommand::ReflectionProbeRenderResult(result.clone()))
        {
            break;
        }
        sent = sent.saturating_add(1);
    }
    tick_state.pending_reflection_probe_render_results = pending.into_iter().skip(sent).collect();
}

pub(in crate::runtime) fn reflection_probe_render_task_count(data: &FrameSubmitData) -> usize {
    data.render_spaces
        .iter()
        .map(|space| space.reflection_probe_render_tasks.len())
        .sum()
}
