//! Compute + fullscreen depth blit used to resolve multisampled depth to single-sample [`wgpu::TextureFormat::Depth32Float`]
//! (no storage writes on depth in core WebGPU).

use crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR;

const COMPUTE_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/msaa_depth_resolve_to_r32.wgsl"
));
const BLIT_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/backend/depth_blit_r32_to_depth.wgsl"
));
const BLIT_STEREO_WGSL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/backend/depth_blit_r32_to_depth_stereo.wgsl"
));

/// Pipelines and layouts for MSAA depth → R32F compute → depth blit.
///
/// Exposes both the desktop (`D2`) path via [`Self::encode_resolve`] and the stereo (OpenXR 2-layer
/// `D2Array`) path via [`Self::encode_resolve_stereo`]. The stereo path reuses the same compute
/// pipeline by dispatching once per eye on single-layer `D2` views, then runs one **multiview**
/// blit pass (`multiview_mask = 0b11`) that writes both depth layers via `@builtin(view_index)`.
///
/// This indirection exists because WGSL `texture_depth_multisampled_2d_array` is not yet available
/// in current `wgpu`, so we keep the compute shader as `texture_depth_multisampled_2d` and issue
/// two dispatches from per-layer views produced by the stereo MSAA depth attachment.
pub struct MsaaDepthResolveResources {
    compute_pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,
    compute_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    /// Multiview blit pipeline for the stereo path; `None` when `MULTIVIEW` is unavailable
    /// (disables stereo MSAA depth resolve but allows the rest of the engine to run).
    blit_stereo_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind-group layout for the stereo blit (`texture_2d_array<f32>` source).
    blit_stereo_bgl: Option<wgpu::BindGroupLayout>,
}

impl MsaaDepthResolveResources {
    /// Builds compute and blit pipelines; returns [`None`] if shader creation fails.
    pub fn try_new(device: &wgpu::Device) -> Option<Self> {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("msaa_depth_resolve_cs"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_WGSL.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("msaa_depth_resolve_blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("msaa_depth_resolve_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: true,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("msaa_depth_blit_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("msaa_depth_resolve_compute_pl"),
            bind_group_layouts: &[Some(&compute_bgl)],
            ..Default::default()
        });

        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("msaa_depth_blit_pl"),
            bind_group_layouts: &[Some(&blit_bgl)],
            ..Default::default()
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("msaa_depth_resolve_compute"),
            layout: Some(&compute_layout),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("msaa_depth_blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (blit_stereo_pipeline, blit_stereo_bgl) =
            if device.features().contains(wgpu::Features::MULTIVIEW) {
                let blit_stereo_shader =
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("msaa_depth_resolve_blit_stereo"),
                        source: wgpu::ShaderSource::Wgsl(BLIT_STEREO_WGSL.into()),
                    });
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("msaa_depth_blit_stereo_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    }],
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("msaa_depth_blit_stereo_pl"),
                    bind_group_layouts: &[Some(&bgl)],
                    ..Default::default()
                });
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("msaa_depth_blit_stereo"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &blit_stereo_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &blit_stereo_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Always),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    // `3 = 0b11` selects both stereo eyes.
                    multiview_mask: std::num::NonZeroU32::new(3),
                    cache: None,
                });
                (Some(pipeline), Some(bgl))
            } else {
                (None, None)
            };

        Some(Self {
            compute_pipeline,
            blit_pipeline,
            compute_bgl,
            blit_bgl,
            blit_stereo_pipeline,
            blit_stereo_bgl,
        })
    }

    /// Resolves `msaa_depth_view` into `dst_depth_view` via R32F intermediate `r32_view`.
    pub fn encode_resolve(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extent: (u32, u32),
        msaa_depth_view: &wgpu::TextureView,
        r32_view: &wgpu::TextureView,
        dst_depth_view: &wgpu::TextureView,
    ) {
        let (w, h) = (extent.0.max(1), extent.1.max(1));

        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("msaa_depth_resolve_compute_bg"),
            layout: &self.compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(msaa_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(r32_view),
                },
            ],
        });

        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("msaa_depth_blit_bg"),
            layout: &self.blit_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(r32_view),
            }],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("msaa-depth-resolve-r32"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &compute_bg, &[]);
            let gx = w.div_ceil(8);
            let gy = h.div_ceil(8);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("msaa-depth-blit-r32-to-depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: dst_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &blit_bg, &[]);
            rpass.draw(0..3, 0..1);
        }
    }

    /// Stereo (OpenXR multiview) MSAA depth resolve.
    ///
    /// - `msaa_depth_layer_views`: two `D2`, single-layer views of the multisampled depth texture,
    ///   sourced from the graph-owned `forward_msaa_depth` transient's per-layer views.
    /// - `r32_layer_views`: two `D2`, single-layer storage views of the intermediate `R32Float` texture.
    /// - `r32_array_view`: `D2Array` sampled view of the same intermediate, used by the multiview blit.
    /// - `dst_depth_view`: `D2Array` (2 layers) view of the single-sample `Depth32Float` attachment.
    ///
    /// Issues two compute dispatches (one per eye) because WGSL lacks
    /// `texture_depth_multisampled_2d_array` today, then one multiview blit pass
    /// (`multiview_mask = 0b11`) that writes both depth layers via `@builtin(view_index)`.
    ///
    /// Does nothing when [`wgpu::Features::MULTIVIEW`] was unavailable at construction
    /// (stereo MSAA is implicitly off in that case via the feature mask in the XR bootstrap).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_resolve_stereo(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extent: (u32, u32),
        msaa_depth_layer_views: [&wgpu::TextureView; 2],
        r32_layer_views: [&wgpu::TextureView; 2],
        r32_array_view: &wgpu::TextureView,
        dst_depth_view: &wgpu::TextureView,
    ) {
        let (Some(blit_stereo_pipeline), Some(blit_stereo_bgl)) =
            (&self.blit_stereo_pipeline, &self.blit_stereo_bgl)
        else {
            return;
        };
        let (w, h) = (extent.0.max(1), extent.1.max(1));

        for eye in 0..2 {
            let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("msaa_depth_resolve_compute_bg_stereo"),
                layout: &self.compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(msaa_depth_layer_views[eye]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(r32_layer_views[eye]),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("msaa-depth-resolve-r32-stereo"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &compute_bg, &[]);
            let gx = w.div_ceil(8);
            let gy = h.div_ceil(8);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("msaa_depth_blit_bg_stereo"),
            layout: blit_stereo_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(r32_array_view),
            }],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("msaa-depth-blit-r32-to-depth-stereo"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: dst_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: std::num::NonZeroU32::new(3),
        });
        rpass.set_pipeline(blit_stereo_pipeline);
        rpass.set_bind_group(0, &blit_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
