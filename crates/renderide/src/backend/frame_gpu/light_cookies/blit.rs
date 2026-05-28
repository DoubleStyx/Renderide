use super::format::{
    LightCookieAtlasFormat, LightCookieSource, LightCookieSourceChannel, LightCookieSourceSampling,
};

/// Embedded WGSL target for copying 2D source cookies into atlas layers.
pub(super) const LIGHT_COOKIE_BLIT_2D_STEM: &str = "light_cookie_blit_2d";
/// Source WGSL used only if embedded shader metadata is unexpectedly missing.
const LIGHT_COOKIE_BLIT_2D_SOURCE: &str =
    include_str!("../../../../shaders/passes/backend/light_cookie_blit_2d.wgsl");

pub(super) struct LightCookieBlitPipelines {
    /// Filterable 2D texture source bind-group layout.
    source_filter_layout: wgpu::BindGroupLayout,
    /// Non-filterable 2D texture source bind-group layout.
    source_non_filter_layout: wgpu::BindGroupLayout,
    /// Alpha-channel filterable source blit pipeline.
    alpha_filter_pipeline: wgpu::RenderPipeline,
    /// Red-channel filterable source blit pipeline.
    red_filter_pipeline: wgpu::RenderPipeline,
    /// Alpha-channel non-filterable source blit pipeline.
    alpha_non_filter_pipeline: wgpu::RenderPipeline,
    /// Red-channel non-filterable source blit pipeline.
    red_non_filter_pipeline: wgpu::RenderPipeline,
    /// Nearest sampler used with non-filterable float source textures.
    source_nearest_sampler: wgpu::Sampler,
}

impl LightCookieBlitPipelines {
    /// Creates blit pipelines for light-cookie atlas updates.
    pub(super) fn new(device: &wgpu::Device, atlas_format: LightCookieAtlasFormat) -> Self {
        let source_filter_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("light_cookie_source_2d_filter_bgl"),
                entries: &[
                    sampled_texture_entry(0, wgpu::TextureViewDimension::D2, true),
                    sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                ],
            });
        let source_non_filter_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("light_cookie_source_2d_non_filter_bgl"),
                entries: &[
                    sampled_texture_entry(0, wgpu::TextureViewDimension::D2, false),
                    sampler_entry(1, wgpu::SamplerBindingType::NonFiltering),
                ],
            });
        let alpha_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_alpha_filter",
            light_cookie_blit_2d_wgsl(),
            &source_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Alpha, atlas_format),
            atlas_format,
        );
        let red_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_red_filter",
            light_cookie_blit_2d_wgsl(),
            &source_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Red, atlas_format),
            atlas_format,
        );
        let alpha_non_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_alpha_non_filter",
            light_cookie_blit_2d_wgsl(),
            &source_non_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Alpha, atlas_format),
            atlas_format,
        );
        let red_non_filter_pipeline = create_blit_pipeline(
            device,
            "light_cookie_blit_red_non_filter",
            light_cookie_blit_2d_wgsl(),
            &source_non_filter_layout,
            blit_fragment_entry(LightCookieSourceChannel::Red, atlas_format),
            atlas_format,
        );
        let source_nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("light_cookie_source_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            source_filter_layout,
            source_non_filter_layout,
            alpha_filter_pipeline,
            red_filter_pipeline,
            alpha_non_filter_pipeline,
            red_non_filter_pipeline,
            source_nearest_sampler,
        }
    }

    /// Returns the bind-group layout for `sampling`.
    fn layout(&self, sampling: LightCookieSourceSampling) -> &wgpu::BindGroupLayout {
        match sampling {
            LightCookieSourceSampling::Filtering => &self.source_filter_layout,
            LightCookieSourceSampling::NonFiltering => &self.source_non_filter_layout,
        }
    }

    /// Returns the render pipeline for `channel` and `sampling`.
    pub(super) fn pipeline(
        &self,
        channel: LightCookieSourceChannel,
        sampling: LightCookieSourceSampling,
    ) -> &wgpu::RenderPipeline {
        match (channel, sampling) {
            (LightCookieSourceChannel::Alpha, LightCookieSourceSampling::Filtering) => {
                &self.alpha_filter_pipeline
            }
            (LightCookieSourceChannel::Red, LightCookieSourceSampling::Filtering) => {
                &self.red_filter_pipeline
            }
            (LightCookieSourceChannel::Alpha, LightCookieSourceSampling::NonFiltering) => {
                &self.alpha_non_filter_pipeline
            }
            (LightCookieSourceChannel::Red, LightCookieSourceSampling::NonFiltering) => {
                &self.red_non_filter_pipeline
            }
        }
    }

    /// Returns the sampler used for source blits.
    fn sampler<'a>(
        &'a self,
        sampling: LightCookieSourceSampling,
        filtering_sampler: &'a wgpu::Sampler,
    ) -> &'a wgpu::Sampler {
        match sampling {
            LightCookieSourceSampling::Filtering => filtering_sampler,
            LightCookieSourceSampling::NonFiltering => &self.source_nearest_sampler,
        }
    }
}

/// Returns the fragment entry point matching the atlas target channel count.
fn blit_fragment_entry(
    channel: LightCookieSourceChannel,
    atlas_format: LightCookieAtlasFormat,
) -> &'static str {
    match (channel, atlas_format) {
        (LightCookieSourceChannel::Alpha, LightCookieAtlasFormat::Rgba16Float) => "fs_alpha_rgba",
        (LightCookieSourceChannel::Red, LightCookieAtlasFormat::Rgba16Float) => "fs_red_rgba",
        (LightCookieSourceChannel::Alpha, _) => "fs_alpha_scalar",
        (LightCookieSourceChannel::Red, _) => "fs_red_scalar",
    }
}

/// Builds a source bind group for one blit.
pub(super) fn create_source_bind_group(
    device: &wgpu::Device,
    blit: &LightCookieBlitPipelines,
    source: LightCookieSource<'_>,
    filtering_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light_cookie_source_bg"),
        layout: blit.layout(source.sampling),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(
                    blit.sampler(source.sampling, filtering_sampler),
                ),
            },
        ],
    })
}

/// Returns the composed 2D light-cookie blit shader.
fn light_cookie_blit_2d_wgsl() -> &'static str {
    let Some(source) = crate::embedded_shaders::embedded_target_wgsl(LIGHT_COOKIE_BLIT_2D_STEM)
    else {
        logger::warn!(
            "embedded WGSL target `{LIGHT_COOKIE_BLIT_2D_STEM}` missing; using raw source fallback"
        );
        return LIGHT_COOKIE_BLIT_2D_SOURCE;
    };
    source
}

/// Builds a sampled texture binding layout entry.
fn sampled_texture_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
    filterable: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// Builds a sampler binding layout entry.
fn sampler_entry(
    binding: u32,
    sampler_type: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}

/// Creates a fullscreen scalar-cookie render pipeline.
fn create_blit_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    fragment_entry: &'static str,
    atlas_format: LightCookieAtlasFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: atlas_format.wgpu(),
                blend: None,
                write_mask: wgpu::ColorWrites::RED,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "backend::light_cookie_blit_pipeline");
    pipeline
}

/// Clears a cookie layer to white.
pub(super) fn clear_cookie_layer(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    label: &'static str,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let pass_query = profiler.map(|p| p.begin_pass_query(label, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    if let (Some(p), Some(q)) = (profiler, pass_query) {
        p.end_query(encoder, q);
    }
}

/// Draws one fullscreen blit into a cookie layer.
pub(super) fn blit_cookie_layer(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    label: &'static str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let pass_query = profiler.map(|p| p.begin_pass_query(label, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    if let (Some(p), Some(q)) = (profiler, pass_query) {
        p.end_query(encoder, q);
    }
}
