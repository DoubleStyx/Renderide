//! Compute + fullscreen depth blit used to resolve multisampled depth to the single-sample forward depth target
//! (no storage writes on depth in core WebGPU).

use super::limits::GpuLimits;
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

/// Single-view (desktop) MSAA depth resolve: sampled views and destination depth format.
pub struct MsaaDepthResolveMonoTargets<'a> {
    /// Multisampled depth texture view (compute input).
    pub msaa_depth_view: &'a wgpu::TextureView,
    /// R32Float intermediate (compute output, blit source).
    pub r32_view: &'a wgpu::TextureView,
    /// Single-sample depth attachment to clear and blit into.
    pub dst_depth_view: &'a wgpu::TextureView,
    /// Format of `dst_depth_view` (selects blit pipeline).
    pub dst_depth_format: wgpu::TextureFormat,
}

/// Stereo multiview MSAA depth resolve: per-eye views, array sample view, and destination.
pub struct MsaaDepthResolveStereoTargets<'a> {
    /// Two single-layer MSAA depth views (one dispatch each).
    pub msaa_depth_layer_views: [&'a wgpu::TextureView; 2],
    /// Two single-layer R32Float storage views.
    pub r32_layer_views: [&'a wgpu::TextureView; 2],
    /// `D2Array` view of the R32 intermediate for the multiview blit.
    pub r32_array_view: &'a wgpu::TextureView,
    /// Two-layer depth attachment view.
    pub dst_depth_view: &'a wgpu::TextureView,
    /// Format of `dst_depth_view` (selects stereo blit pipeline).
    pub dst_depth_format: wgpu::TextureFormat,
}

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
    blit_pipeline_depth32: wgpu::RenderPipeline,
    blit_pipeline_depth24_stencil8: wgpu::RenderPipeline,
    blit_pipeline_depth32_stencil8: Option<wgpu::RenderPipeline>,
    compute_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    /// Multiview blit pipeline for the stereo path; `None` when `MULTIVIEW` is unavailable
    /// (disables stereo MSAA depth resolve but allows the rest of the engine to run).
    blit_stereo_pipeline_depth32: Option<wgpu::RenderPipeline>,
    blit_stereo_pipeline_depth24_stencil8: Option<wgpu::RenderPipeline>,
    blit_stereo_pipeline_depth32_stencil8: Option<wgpu::RenderPipeline>,
    /// Bind-group layout for the stereo blit (`texture_2d_array<f32>` source).
    blit_stereo_bgl: Option<wgpu::BindGroupLayout>,
}

/// Desktop (non-multiview) depth blit pipelines for each depth/stencil format variant.
struct DesktopBlitPipelines {
    depth32: wgpu::RenderPipeline,
    depth24_stencil8: wgpu::RenderPipeline,
    depth32_stencil8: Option<wgpu::RenderPipeline>,
}

/// Optional multiview stereo blit pipelines and bind-group layout.
struct StereoMultiviewBlitPipelines {
    depth32: Option<wgpu::RenderPipeline>,
    depth24_stencil8: Option<wgpu::RenderPipeline>,
    depth32_stencil8: Option<wgpu::RenderPipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
}

fn create_desktop_blit_pipelines(
    device: &wgpu::Device,
    blit_shader: &wgpu::ShaderModule,
    blit_layout: &wgpu::PipelineLayout,
) -> DesktopBlitPipelines {
    DesktopBlitPipelines {
        depth32: create_depth_blit_pipeline(
            device,
            blit_shader,
            blit_layout,
            "msaa_depth_blit_depth32",
            wgpu::TextureFormat::Depth32Float,
            None,
        ),
        depth24_stencil8: create_depth_blit_pipeline(
            device,
            blit_shader,
            blit_layout,
            "msaa_depth_blit_depth24_stencil8",
            wgpu::TextureFormat::Depth24PlusStencil8,
            None,
        ),
        depth32_stencil8: device
            .features()
            .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
            .then(|| {
                create_depth_blit_pipeline(
                    device,
                    blit_shader,
                    blit_layout,
                    "msaa_depth_blit_depth32_stencil8",
                    wgpu::TextureFormat::Depth32FloatStencil8,
                    None,
                )
            }),
    }
}

fn create_stereo_multiview_blit_pipelines(device: &wgpu::Device) -> StereoMultiviewBlitPipelines {
    if !device.features().contains(wgpu::Features::MULTIVIEW) {
        return StereoMultiviewBlitPipelines {
            depth32: None,
            depth24_stencil8: None,
            depth32_stencil8: None,
            bgl: None,
        };
    }
    let blit_stereo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
    let multiview_mask = std::num::NonZeroU32::new(3);
    let depth32 = create_depth_blit_pipeline(
        device,
        &blit_stereo_shader,
        &layout,
        "msaa_depth_blit_stereo_depth32",
        wgpu::TextureFormat::Depth32Float,
        multiview_mask,
    );
    let depth24_stencil8 = create_depth_blit_pipeline(
        device,
        &blit_stereo_shader,
        &layout,
        "msaa_depth_blit_stereo_depth24_stencil8",
        wgpu::TextureFormat::Depth24PlusStencil8,
        multiview_mask,
    );
    let depth32_stencil8 = device
        .features()
        .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8)
        .then(|| {
            create_depth_blit_pipeline(
                device,
                &blit_stereo_shader,
                &layout,
                "msaa_depth_blit_stereo_depth32_stencil8",
                wgpu::TextureFormat::Depth32FloatStencil8,
                multiview_mask,
            )
        });
    StereoMultiviewBlitPipelines {
        depth32: Some(depth32),
        depth24_stencil8: Some(depth24_stencil8),
        depth32_stencil8,
        bgl: Some(bgl),
    }
}

fn create_depth_blit_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    label: &str,
    format: wgpu::TextureFormat,
    multiview_mask: Option<std::num::NonZeroU32>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
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
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask,
        cache: None,
    })
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

        let desktop = create_desktop_blit_pipelines(device, &blit_shader, &blit_layout);
        let stereo = create_stereo_multiview_blit_pipelines(device);

        Some(Self {
            compute_pipeline,
            blit_pipeline_depth32: desktop.depth32,
            blit_pipeline_depth24_stencil8: desktop.depth24_stencil8,
            blit_pipeline_depth32_stencil8: desktop.depth32_stencil8,
            compute_bgl,
            blit_bgl,
            blit_stereo_pipeline_depth32: stereo.depth32,
            blit_stereo_pipeline_depth24_stencil8: stereo.depth24_stencil8,
            blit_stereo_pipeline_depth32_stencil8: stereo.depth32_stencil8,
            blit_stereo_bgl: stereo.bgl,
        })
    }

    fn blit_pipeline_for_format(
        &self,
        format: wgpu::TextureFormat,
    ) -> Option<&wgpu::RenderPipeline> {
        match format {
            wgpu::TextureFormat::Depth24PlusStencil8 => Some(&self.blit_pipeline_depth24_stencil8),
            wgpu::TextureFormat::Depth32FloatStencil8 => {
                self.blit_pipeline_depth32_stencil8.as_ref()
            }
            _ => Some(&self.blit_pipeline_depth32),
        }
    }

    fn stereo_blit_pipeline_for_format(
        &self,
        format: wgpu::TextureFormat,
    ) -> Option<&wgpu::RenderPipeline> {
        match format {
            wgpu::TextureFormat::Depth24PlusStencil8 => {
                self.blit_stereo_pipeline_depth24_stencil8.as_ref()
            }
            wgpu::TextureFormat::Depth32FloatStencil8 => {
                self.blit_stereo_pipeline_depth32_stencil8.as_ref()
            }
            _ => self.blit_stereo_pipeline_depth32.as_ref(),
        }
    }

    /// Resolves `targets.msaa_depth_view` into `targets.dst_depth_view` via R32F intermediate `targets.r32_view`.
    ///
    /// When the 8×8-tiled compute dispatch would exceed [`GpuLimits::compute_dispatch_fits`], logs a
    /// warning and skips compute and blit (degraded depth for intersection / Hi-Z vs invalid GPU work).
    pub fn encode_resolve(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extent: (u32, u32),
        targets: MsaaDepthResolveMonoTargets<'_>,
        limits: &GpuLimits,
    ) {
        let MsaaDepthResolveMonoTargets {
            msaa_depth_view,
            r32_view,
            dst_depth_view,
            dst_depth_format,
        } = targets;
        let (w, h) = (extent.0.max(1), extent.1.max(1));
        let gx = w.div_ceil(8);
        let gy = h.div_ceil(8);
        if !limits.compute_dispatch_fits(gx, gy, 1) {
            logger::warn!(
                "MSAA depth resolve: dispatch {}×{}×1 exceeds max_compute_workgroups_per_dimension ({})",
                gx,
                gy,
                limits.max_compute_workgroups_per_dimension()
            );
            return;
        }

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
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        let Some(blit_pipeline) = self.blit_pipeline_for_format(dst_depth_format) else {
            logger::warn!(
                "MSAA depth resolve: mono blit pipeline missing for {:?} (DEPTH32FLOAT_STENCIL8 feature unavailable?)",
                dst_depth_format
            );
            return;
        };
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
            rpass.set_pipeline(blit_pipeline);
            rpass.set_bind_group(0, &blit_bg, &[]);
            rpass.draw(0..3, 0..1);
        }
    }

    /// Stereo (OpenXR multiview) MSAA depth resolve.
    ///
    /// See [`MsaaDepthResolveStereoTargets`] for the view layout. Issues two compute dispatches
    /// (one per eye) because WGSL lacks `texture_depth_multisampled_2d_array` today, then one
    /// multiview blit pass (`multiview_mask = 0b11`) that writes both depth layers via
    /// `@builtin(view_index)`.
    ///
    /// Does nothing when [`wgpu::Features::MULTIVIEW`] was unavailable at construction
    /// (stereo MSAA is implicitly off in that case via the feature mask in the XR bootstrap).
    ///
    /// See [`Self::encode_resolve`] for compute dispatch limit handling.
    pub fn encode_resolve_stereo(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extent: (u32, u32),
        targets: MsaaDepthResolveStereoTargets<'_>,
        limits: &GpuLimits,
    ) {
        let MsaaDepthResolveStereoTargets {
            msaa_depth_layer_views,
            r32_layer_views,
            r32_array_view,
            dst_depth_view,
            dst_depth_format,
        } = targets;
        let Some(blit_stereo_pipeline) = self.stereo_blit_pipeline_for_format(dst_depth_format)
        else {
            return;
        };
        let Some(blit_stereo_bgl) = &self.blit_stereo_bgl else {
            return;
        };
        let (w, h) = (extent.0.max(1), extent.1.max(1));
        let gx = w.div_ceil(8);
        let gy = h.div_ceil(8);
        if !limits.compute_dispatch_fits(gx, gy, 1) {
            logger::warn!(
                "MSAA depth resolve (stereo): dispatch {}×{}×1 exceeds max_compute_workgroups_per_dimension ({})",
                gx,
                gy,
                limits.max_compute_workgroups_per_dimension()
            );
            return;
        }

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
