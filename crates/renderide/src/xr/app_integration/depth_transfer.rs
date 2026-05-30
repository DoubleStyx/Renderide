//! Multiview transfer from renderer-owned HMD depth into an OpenXR depth swapchain image.

use std::num::NonZeroU32;
use std::sync::OnceLock;

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::GpuContext;

/// GPU resources for copying HMD depth into an OpenXR depth swapchain.
#[derive(Default)]
pub(super) struct XrDepthTransferResources {
    depth32_pipeline: Option<wgpu::RenderPipeline>,
    depth16_pipeline: Option<wgpu::RenderPipeline>,
}

impl XrDepthTransferResources {
    /// Creates an empty depth-transfer pipeline cache.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Encodes a single multiview depth transfer into the acquired OpenXR depth target.
    pub(super) fn encode_hmd_depth_to_openxr(
        &mut self,
        gpu: &mut GpuContext,
        eye_extent: (u32, u32),
        source_depth: &wgpu::TextureView,
        target_depth: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::CommandBuffer {
        let device_arc = gpu.device().clone();
        let device = device_arc.as_ref();
        let bind_group = create_depth_transfer_bind_group(device, source_depth);
        let pipeline = self.pipeline_for_format(device, target_format);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("xr_depth_transfer"),
        });
        let query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_pass_query("graph::xr.depth_transfer", &mut encoder));
        let timestamp_writes = crate::profiling::render_pass_timestamp_writes(query.as_ref());
        encode_depth_transfer_pass(
            &mut encoder,
            eye_extent,
            target_depth,
            pipeline,
            &bind_group,
            timestamp_writes,
        );
        if let Some(query) = query
            && let Some(prof) = gpu.gpu_profiler_mut()
        {
            prof.end_query(&mut encoder, query);
            prof.resolve_queries(&mut encoder);
        }
        profiling::scope!("CommandEncoder::finish::xr_depth_transfer");
        encoder.finish()
    }

    fn pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        match format {
            wgpu::TextureFormat::Depth32Float => self
                .depth32_pipeline
                .get_or_insert_with(|| create_depth_transfer_pipeline(device, format)),
            wgpu::TextureFormat::Depth16Unorm => self
                .depth16_pipeline
                .get_or_insert_with(|| create_depth_transfer_pipeline(device, format)),
            _ => {
                logger::warn!("xr depth transfer requested unsupported format {format:?}");
                self.depth32_pipeline
                    .get_or_insert_with(|| create_depth_transfer_pipeline(device, format))
            }
        }
    }
}

fn create_depth_transfer_bind_group(
    device: &wgpu::Device,
    source_depth: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("xr_depth_transfer"),
        layout: depth_transfer_bind_group_layout(device),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(source_depth),
        }],
    });
    crate::profiling::note_resource_churn!(BindGroup, "xr::depth_transfer");
    bind_group
}

fn depth_transfer_bind_group_layout(device: &wgpu::Device) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xr_depth_transfer"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            }],
        })
    })
}

fn create_depth_transfer_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xr_depth_transfer"),
        source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("xr_depth_transfer_multiview").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("xr_depth_transfer"),
        bind_group_layouts: &[Some(depth_transfer_bind_group_layout(device))],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("xr_depth_transfer"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: stereo_multiview_mask(),
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "xr::depth_transfer_pipeline");
    pipeline
}

fn encode_depth_transfer_pass(
    encoder: &mut wgpu::CommandEncoder,
    eye_extent: (u32, u32),
    target_depth: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("xr_depth_transfer"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: target_depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(crate::gpu::MAIN_FORWARD_DEPTH_CLEAR),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        timestamp_writes,
        multiview_mask: stereo_multiview_mask(),
    });
    pass.set_viewport(0.0, 0.0, eye_extent.0 as f32, eye_extent.1 as f32, 0.0, 1.0);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn stereo_multiview_mask() -> Option<NonZeroU32> {
    NonZeroU32::new(3)
}
