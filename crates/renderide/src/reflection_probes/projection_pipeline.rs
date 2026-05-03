//! Projection compute pipelines and dispatch encoding for reflection-probe SH2 jobs.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use super::readback_jobs::SubmittedGpuSh2Job;
use super::{SH2_OUTPUT_BYTES, Sh2ProjectParams, Sh2SourceKey};
use crate::embedded_shaders;
use crate::gpu::GpuContext;
use crate::profiling::{GpuProfilerHandle, compute_pass_timestamp_writes};

/// Lazily-created compute pipeline and bind-group layout.
pub(super) struct ProjectionPipeline {
    /// Compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind-group layout for one projection source.
    layout: wgpu::BindGroupLayout,
}

/// Extra binding resource for texture-backed projection kernels.
pub(super) enum ProjectionBinding<'a> {
    /// Sampled texture view.
    TextureView(&'a wgpu::TextureView),
    /// Sampler paired with the texture view.
    Sampler(&'a wgpu::Sampler),
}

/// GPU buffers used by one SH2 projection job.
struct ProjectionJobBuffers {
    /// Uniform parameters consumed by the projection shader.
    params: wgpu::Buffer,
    /// Storage-buffer output written by the projection shader.
    output: wgpu::Buffer,
    /// CPU-readable staging buffer receiving the copied output.
    staging: wgpu::Buffer,
}

/// Ensures a projection pipeline exists for an embedded compute shader.
pub(super) fn ensure_projection_pipeline<'a>(
    slot: &'a mut Option<ProjectionPipeline>,
    device: &wgpu::Device,
    stem: &str,
) -> Result<&'a ProjectionPipeline, String> {
    if slot.is_none() {
        profiling::scope!("reflection_probe_sh2::create_projection_pipeline", stem);
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
pub(super) fn encode_projection_job(
    gpu: &mut GpuContext,
    key: Sh2SourceKey,
    pipeline: &ProjectionPipeline,
    extra_bindings: &[ProjectionBinding<'_>],
    params: &Sh2ProjectParams,
    submit_done_tx: &crossbeam_channel::Sender<Sh2SourceKey>,
    profile_label: &'static str,
) -> Result<SubmittedGpuSh2Job, String> {
    let mut profiler = gpu.take_gpu_profiler();
    let request = ProjectionJobRequest {
        key,
        pipeline,
        extra_bindings,
        params,
        submit_done_tx,
        profile_label,
    };
    let result = encode_projection_job_with_profiler(gpu, request, profiler.as_mut());
    gpu.restore_gpu_profiler(profiler);
    result
}

struct ProjectionJobRequest<'request, 'resource> {
    key: Sh2SourceKey,
    pipeline: &'request ProjectionPipeline,
    extra_bindings: &'request [ProjectionBinding<'resource>],
    params: &'request Sh2ProjectParams,
    submit_done_tx: &'request crossbeam_channel::Sender<Sh2SourceKey>,
    profile_label: &'static str,
}

fn encode_projection_job_with_profiler(
    gpu: &GpuContext,
    request: ProjectionJobRequest<'_, '_>,
    mut profiler: Option<&mut GpuProfilerHandle>,
) -> Result<SubmittedGpuSh2Job, String> {
    profiling::scope!("reflection_probe_sh2::encode_projection_job");
    let buffers = create_projection_buffers(gpu, request.params);
    let bind_group =
        create_projection_bind_group(gpu, request.pipeline, &buffers, request.extra_bindings);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SH2 projection encoder"),
        });
    record_projection_dispatch(
        &mut encoder,
        request.pipeline,
        &bind_group,
        request.profile_label,
        profiler.as_deref(),
    );
    record_projection_readback_copy(&mut encoder, &buffers);
    if let Some(profiler) = profiler.as_mut() {
        profiling::scope!("reflection_probe_sh2::resolve_profiler_queries");
        profiler.resolve_queries(&mut encoder);
    }
    submit_projection_job(gpu, encoder, request.key, request.submit_done_tx);

    Ok(SubmittedGpuSh2Job {
        staging: buffers.staging,
        output: buffers.output,
        bind_group,
        buffers: vec![buffers.params],
    })
}

/// Creates the uniform, storage output, and staging buffers for one projection job.
fn create_projection_buffers(gpu: &GpuContext, params: &Sh2ProjectParams) -> ProjectionJobBuffers {
    profiling::scope!("reflection_probe_sh2::projection_buffers");
    let params = gpu
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
    ProjectionJobBuffers {
        params,
        output,
        staging,
    }
}

/// Creates the projection bind group for the source-specific shader layout.
fn create_projection_bind_group(
    gpu: &GpuContext,
    pipeline: &ProjectionPipeline,
    buffers: &ProjectionJobBuffers,
    extra_bindings: &[ProjectionBinding<'_>],
) -> wgpu::BindGroup {
    profiling::scope!("reflection_probe_sh2::projection_bind_group");
    let mut entries = projection_bind_entries(buffers, extra_bindings);
    entries.sort_by_key(|entry| entry.binding);
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SH2 projection bind group"),
        layout: &pipeline.layout,
        entries: &entries,
    })
}

/// Builds bind entries for the projection bind group.
fn projection_bind_entries<'a>(
    buffers: &'a ProjectionJobBuffers,
    extra_bindings: &[ProjectionBinding<'a>],
) -> Vec<wgpu::BindGroupEntry<'a>> {
    profiling::scope!("reflection_probe_sh2::projection_bind_entries");
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: buffers.params.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: buffers.output.as_entire_binding(),
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
    entries
}

/// Records the projection compute pass with optional GPU timestamp writes.
fn record_projection_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ProjectionPipeline,
    bind_group: &wgpu::BindGroup,
    profile_label: &'static str,
    profiler: Option<&GpuProfilerHandle>,
) {
    profiling::scope!("reflection_probe_sh2::record_projection_pass");
    let pass_query = profiler.map(|profiler| profiler.begin_pass_query(profile_label, encoder));
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SH2 projection"),
            timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    };
    if let (Some(profiler), Some(query)) = (profiler, pass_query) {
        profiler.end_query(encoder, query);
    }
}

/// Records the GPU-to-staging copy used by the async readback tracker.
fn record_projection_readback_copy(
    encoder: &mut wgpu::CommandEncoder,
    buffers: &ProjectionJobBuffers,
) {
    profiling::scope!("reflection_probe_sh2::record_staging_copy");
    encoder.copy_buffer_to_buffer(&buffers.output, 0, &buffers.staging, 0, SH2_OUTPUT_BYTES);
}

/// Submits the projection command buffer through the renderer driver thread.
fn submit_projection_job(
    gpu: &GpuContext,
    encoder: wgpu::CommandEncoder,
    key: Sh2SourceKey,
    submit_done_tx: &crossbeam_channel::Sender<Sh2SourceKey>,
) {
    profiling::scope!("reflection_probe_sh2::submit_projection_job");
    let tx = submit_done_tx.clone();
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::reflection_probe_sh2");
        encoder.finish()
    };
    gpu.submit_frame_batch_with_callbacks(
        vec![command_buffer],
        None,
        None,
        vec![Box::new(move || {
            let _ = tx.send(key);
        })],
    );
}
