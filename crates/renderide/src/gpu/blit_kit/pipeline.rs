//! Color-blit pipeline builder, per-format pipeline slot, and lazy UV uniform buffer.
//!
//! All three primitives are shared by the desktop display blit and the VR mirror surface blit.
//! Each builds a vertex-less triangle-list pipeline against a single color target, no MSAA, no
//! depth-stencil.

/// Builds a vertex-less triangle-list color-target render pipeline.
///
/// `shader` must expose `vs_main` and `fs_main` entry points and read all bindings through the
/// pipeline's single bind-group layout.
pub(crate) fn color_blit_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    label: &'static str,
    color_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    color_blit_pipeline_with_multiview_mask(device, shader, layout, label, color_format, None)
}

/// Builds a vertex-less triangle-list color-target render pipeline with optional multiview.
///
/// `shader` must expose `vs_main` and `fs_main` entry points and read all bindings through the
/// pipeline's single bind-group layout.
pub(crate) fn color_blit_pipeline_with_multiview_mask(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    label: &'static str,
    color_format: wgpu::TextureFormat,
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
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask,
        cache: None,
    })
}

/// Single-slot per-format pipeline cache that rebuilds on format change.
#[derive(Debug, Default)]
pub(crate) struct ColorBlitPipelineSlot {
    entry: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

impl ColorBlitPipelineSlot {
    /// Empty slot; the pipeline is built lazily on the first [`Self::get_or_build`] call.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the cached pipeline when it matches `format`, otherwise rebuilds it via `build`
    /// (which must produce a pipeline compatible with `format`).
    pub(crate) fn get_or_build(
        &mut self,
        format: wgpu::TextureFormat,
        build: impl FnOnce(wgpu::TextureFormat) -> wgpu::RenderPipeline,
    ) -> &wgpu::RenderPipeline {
        if let Some((cached, pipeline)) = self.entry.take()
            && cached == format
        {
            let entry = self.entry.insert((cached, pipeline));
            return &entry.1;
        }
        let entry = self.entry.insert((format, build(format)));
        &entry.1
    }
}

/// Lazy 16-byte UV uniform buffer (`UNIFORM | COPY_DST`) shared by color-blit subsystems.
#[derive(Debug, Default)]
pub(crate) struct UvUniformBuffer {
    buf: Option<wgpu::Buffer>,
}

impl UvUniformBuffer {
    /// Empty buffer; allocated on first [`Self::ensure`].
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Allocates the underlying buffer on first call. Subsequent calls are no-ops.
    pub(crate) fn ensure(&mut self, device: &wgpu::Device, label: &'static str) {
        if self.buf.is_some() {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "blit_kit::uv_uniform");
        self.buf = Some(buf);
    }

    /// Returns the buffer once [`Self::ensure`] has run.
    pub(crate) fn get(&self) -> Option<&wgpu::Buffer> {
        self.buf.as_ref()
    }

    /// Uploads `bytes` (expected length 16) to the buffer via `queue.write_buffer`.
    pub(crate) fn write(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        if let Some(buf) = self.buf.as_ref() {
            queue.write_buffer(buf, 0, bytes);
        }
    }
}
