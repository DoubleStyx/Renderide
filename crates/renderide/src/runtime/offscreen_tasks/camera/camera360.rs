//! Camera360 photo capture through shared cubemap faces and equirectangular projection.

use std::mem::size_of;
use std::num::NonZeroU64;
use std::sync::LazyLock;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::camera::{ViewId, camera_render_task_clip};
use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::GpuContext;
use crate::gpu::bind_layout::{
    sampler_layout_entry, texture_layout_entry, uniform_buffer_layout_entry,
};
use crate::gpu_resource::{OnceGpu, RenderPipelineMap};
use crate::render_graph::gpu_cache::{
    FullscreenRenderPipelineDesc, create_fullscreen_render_pipeline, create_linear_clamp_sampler,
    create_wgsl_shader_module,
};
use crate::render_graph::{FrameViewClear, RenderPathProfile};
use crate::scene::RenderSpaceId;
use crate::shared::{CameraRenderParameters, RenderingContext};

use super::super::super::frame::view_plan::{FrameViewPlan, FrameViewPlanTarget};
use super::super::cube_capture::{
    CUBE_FACE_COUNT, CubeCaptureBasisMode, CubeCaptureExtent, CubeCaptureFace,
    CubeCaptureTargetError, CubeCaptureTargets, host_camera_frame_for_cube_face_with_basis,
    render_cube_capture_faces_offscreen,
};
use super::{
    CAMERA_TASK_COLOR_FORMAT, CameraReadbackError, CameraTaskExtent, CameraTaskOutputFormat,
    CameraTaskRenderCtx, CameraTaskTargets, alpha_coverage, camera_render_task_post_processing,
    draw_filter_from_camera_render_task, output_byte_count, readback_camera_task_texture,
    write_camera_task_result,
};

/// Cubemap orientation mode used by Camera360 before equirectangular projection.
pub(super) const CAMERA360_CUBE_BASIS_MODE: CubeCaptureBasisMode =
    CubeCaptureBasisMode::Camera360Copied;

/// Uniforms consumed by the Camera360 cubemap-to-equirect projection pass.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera360ProjectionUniform {
    /// World-space rotation applied while sampling the captured cubemap.
    rotation: [[f32; 4]; 4],
    /// Storage-orientation parameters; `.x` is the cubemap V-inversion flag.
    storage: [f32; 4],
}

impl Camera360ProjectionUniform {
    /// Builds projection uniforms from the host task rotation.
    fn from_task_rotation(rotation: glam::Quat) -> Self {
        Self {
            rotation: glam::Mat4::from_quat(rotation).to_cols_array_2d(),
            storage: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Returns whether camera render parameters request Camera360 equirectangular output.
pub(super) fn camera_render_parameters_request_camera360(
    parameters: &CameraRenderParameters,
) -> bool {
    parameters.fov >= 180.0
}

/// Computes the square cubemap face size used for a Camera360 output extent.
pub(super) fn camera360_face_size_for_extent(
    extent: CameraTaskExtent,
) -> Result<u32, CameraReadbackError> {
    let total_texels = u64::from(extent.width)
        .checked_mul(u64::from(extent.height))
        .ok_or(CameraReadbackError::OutputByteCountOverflow)?;
    let texels_per_face = total_texels as f64 / CUBE_FACE_COUNT as f64;
    let root = texels_per_face.sqrt() as u32;
    Ok(root.max(1).next_power_of_two())
}

/// Renders and writes one Camera360 photo task.
pub(super) fn render_camera360_task(
    ctx: CameraTaskRenderCtx<'_>,
) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera360_task::render_one");
    let planned = plan_camera360_task(&ctx)?;
    let view_ids = planned
        .plans
        .iter()
        .map(|plan| plan.view_id)
        .collect::<Vec<_>>();
    let render_result =
        render_cube_capture_faces_offscreen(ctx.gpu, ctx.backend, ctx.scene, planned.plans);
    if let Err(error) = render_result {
        ctx.backend.retire_one_shot_views(&view_ids);
        return Err(CameraReadbackError::Graph(error));
    }
    if planned.output_format.needs_alpha_coverage_repair() {
        apply_camera360_alpha_coverage(ctx.gpu, &planned.cube_targets);
    }
    project_camera360_to_equirect(
        ctx.gpu,
        &planned.cube_targets,
        &planned.output_targets,
        ctx.task.rotation,
    );
    let rgba = match readback_camera_task_texture(
        ctx.gpu,
        planned.output_targets.color_texture.as_ref(),
    ) {
        Ok(rgba) => rgba,
        Err(error) => {
            ctx.backend.retire_one_shot_views(&view_ids);
            return Err(error);
        }
    };
    ctx.backend.retire_one_shot_views(&view_ids);
    write_camera_task_result(
        ctx.shm,
        ctx.task,
        planned.output_format,
        planned.output_targets.extent,
        &rgba,
    )
}

/// Fully planned Camera360 render task.
struct PlannedCamera360Task {
    /// Per-face cubemap render plans.
    plans: Vec<FrameViewPlan<'static>>,
    /// Captured cubemap face targets.
    cube_targets: CubeCaptureTargets,
    /// Final equirectangular output target.
    output_targets: CameraTaskTargets,
    /// Host-requested output packing format.
    output_format: CameraTaskOutputFormat,
}

/// Validates and prepares one Camera360 task.
fn plan_camera360_task(
    ctx: &CameraTaskRenderCtx<'_>,
) -> Result<PlannedCamera360Task, CameraReadbackError> {
    profiling::scope!("camera360_task::plan");
    let parameters = ctx
        .task
        .parameters
        .as_ref()
        .ok_or(CameraReadbackError::MissingParameters)?;
    let output_format = CameraTaskOutputFormat::from_texture_format(parameters.texture_format)
        .ok_or(CameraReadbackError::UnsupportedFormat(
            parameters.texture_format,
        ))?;
    let output_extent = CameraTaskExtent::from_parameters(parameters)?;
    let required = output_byte_count(output_extent, output_format)?;
    let actual = usize::try_from(ctx.task.result_data.length).unwrap_or(0);
    if actual < required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall { required, actual });
    }
    let render_space_id = RenderSpaceId(ctx.task.render_space_id);
    let Some(space) = ctx.scene.space(render_space_id) else {
        return Err(CameraReadbackError::MissingRenderSpace(
            ctx.task.render_space_id,
        ));
    };
    if !space.is_active() {
        return Err(CameraReadbackError::InactiveRenderSpace(
            ctx.task.render_space_id,
        ));
    }

    let face_size = camera360_face_size_for_extent(output_extent)?;
    let cube_targets = create_camera360_cube_targets(ctx.gpu, face_size)?;
    let output_targets = CameraTaskTargets::create(ctx.gpu, output_extent)?;
    let face_viewport = cube_targets.extent.viewport();
    let filter = draw_filter_from_camera_render_task(ctx.task);
    let clip = camera_render_task_clip(parameters);
    let plans = CubeCaptureFace::ALL
        .iter()
        .copied()
        .map(|face| FrameViewPlan {
            host_camera: host_camera_frame_for_cube_face_with_basis(
                ctx.base_camera,
                clip,
                face_viewport,
                ctx.task.position,
                face,
                CAMERA360_CUBE_BASIS_MODE,
            ),
            render_context: RenderingContext::RenderToAsset,
            draw_filter: Some(filter.clone()),
            render_space_filter: Some(render_space_id),
            view_id: ViewId::camera360_render_task_face(
                render_space_id,
                ctx.task_index,
                face.view_id_face_index(),
            ),
            viewport_px: face_viewport,
            clear: FrameViewClear::from_camera_render_parameters(parameters),
            profile: RenderPathProfile::cube_capture(camera_render_task_post_processing(
                parameters,
            )),
            target: FrameViewPlanTarget::SecondaryRt(cube_targets.to_offscreen_handles(face)),
        })
        .collect();

    Ok(PlannedCamera360Task {
        plans,
        cube_targets,
        output_targets,
        output_format,
    })
}

/// Allocates Camera360 cubemap capture targets.
fn create_camera360_cube_targets(
    gpu: &GpuContext,
    face_size: u32,
) -> Result<CubeCaptureTargets, CameraReadbackError> {
    CubeCaptureTargets::create(
        gpu,
        CubeCaptureExtent::new(face_size, 1),
        CAMERA_TASK_COLOR_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        "renderide-camera360-task-cube",
    )
    .map_err(camera360_target_error)
}

/// Converts shared cubemap allocation failures to CameraRenderTask errors.
fn camera360_target_error(error: CubeCaptureTargetError) -> CameraReadbackError {
    match error {
        CubeCaptureTargetError::SizeExceedsLimit { size, max } => {
            CameraReadbackError::ExtentExceedsLimit {
                width: size,
                height: size,
                max,
            }
        }
        CubeCaptureTargetError::CubemapArrayLayersUnsupported { max } => {
            CameraReadbackError::CubemapArrayLayersUnsupported { max }
        }
    }
}

/// Repairs alpha coverage on every rendered cubemap face before projection.
fn apply_camera360_alpha_coverage(gpu: &mut GpuContext, targets: &CubeCaptureTargets) {
    profiling::scope!("camera360_task::alpha_coverage");
    for face in CubeCaptureFace::ALL {
        alpha_coverage::apply_alpha_coverage_to_target(
            gpu,
            targets.face_color_views[face.index()].as_ref(),
            targets.face_depth_textures[face.index()].as_ref(),
            targets.color_format,
            "camera360_task_alpha_coverage",
        );
    }
}

/// Projects the captured cubemap to the final equirectangular output texture.
fn project_camera360_to_equirect(
    gpu: &GpuContext,
    cube_targets: &CubeCaptureTargets,
    output_targets: &CameraTaskTargets,
    rotation: glam::Quat,
) {
    profiling::scope!("camera360_task::project_equirect");
    let pipelines = projection_pipeline_cache();
    let uniform = Camera360ProjectionUniform::from_task_rotation(rotation);
    let uniform_buffer = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera360_projection_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    crate::profiling::note_resource_churn!(Buffer, "runtime::camera360_projection_uniform");
    let cube_view = cube_targets.cube_sample_view("renderide-camera360-task-cube-view");
    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera360_projection"),
        layout: pipelines.bind_group_layout(gpu.device()),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(cube_view.as_ref()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(pipelines.sampler(gpu.device())),
            },
        ],
    });
    crate::profiling::note_resource_churn!(BindGroup, "runtime::camera360_projection_bind_group");
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("camera360_projection"),
        });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("camera360_projection"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_targets.color_view.as_ref(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(
            pipelines
                .pipeline(gpu.device(), output_targets.color_format)
                .as_ref(),
        );
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::camera360_projection");
        encoder.finish()
    };
    gpu.queue().submit(std::iter::once(command_buffer));
}

/// Returns the process-wide Camera360 projection pipeline cache.
fn projection_pipeline_cache() -> &'static Camera360ProjectionPipelineCache {
    static CACHE: LazyLock<Camera360ProjectionPipelineCache> =
        LazyLock::new(Camera360ProjectionPipelineCache::default);
    &CACHE
}

/// Cached GPU resources for Camera360 equirectangular projection.
#[derive(Default)]
struct Camera360ProjectionPipelineCache {
    /// Projection bind-group layout.
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear clamp sampler used to sample the cubemap.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Render pipelines keyed by final output format.
    pipelines: RenderPipelineMap<wgpu::TextureFormat>,
}

impl Camera360ProjectionPipelineCache {
    /// Returns the projection bind-group layout.
    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera360_projection"),
                entries: &[
                    uniform_buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        NonZeroU64::new(size_of::<Camera360ProjectionUniform>() as u64),
                    ),
                    texture_layout_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: true },
                        wgpu::TextureViewDimension::Cube,
                        false,
                    ),
                    sampler_layout_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::SamplerBindingType::Filtering,
                    ),
                ],
            })
        })
    }

    /// Returns the projection sampler.
    fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "camera360_projection"))
    }

    /// Returns the projection render pipeline for an output format.
    fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
    ) -> std::sync::Arc<wgpu::RenderPipeline> {
        self.pipelines.get_or_create(output_format, |format| {
            logger::debug!(
                "camera360_projection: building pipeline (dst format = {:?})",
                format
            );
            create_camera360_projection_pipeline(device, self.bind_group_layout(device), *format)
        })
    }
}

/// Builds a Camera360 projection render pipeline.
fn create_camera360_projection_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = create_wgsl_shader_module(
        device,
        "camera360_equirect",
        embedded_wgsl!("camera360_equirect"),
    );
    create_fullscreen_render_pipeline(
        device,
        FullscreenRenderPipelineDesc {
            label: "camera360_projection",
            bind_group_layouts: &[Some(bind_group_layout)],
            shader: &shader,
            fragment_entry: "fs_main",
            output_format,
            blend: None,
            multiview_stereo: false,
        },
    )
}
