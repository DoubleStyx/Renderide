//! Projection compute pipelines and dispatch encoding for reflection-probe SH2 jobs.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use super::readback_jobs::SubmittedGpuSh2Job;
use super::{Sh2ProjectParams, Sh2SourceKey, SH2_OUTPUT_BYTES};
use crate::embedded_shaders;
use crate::gpu::GpuContext;

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

/// Ensures a projection pipeline exists for an embedded compute shader.
pub(super) fn ensure_projection_pipeline<'a>(
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
pub(super) fn encode_projection_job(
    gpu: &GpuContext,
    key: Sh2SourceKey,
    pipeline: &ProjectionPipeline,
    extra_bindings: &[ProjectionBinding<'_>],
    params: &Sh2ProjectParams,
    submit_done_tx: &crossbeam_channel::Sender<Sh2SourceKey>,
) -> Result<SubmittedGpuSh2Job, String> {
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

    Ok(SubmittedGpuSh2Job {
        staging,
        output,
        bind_group,
        buffers: vec![params_buffer],
    })
}
