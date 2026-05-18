//! Cached GPU state for motion-vector and motion-blur passes.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::bind_layout::{
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry, texture_layout_entry,
    uniform_buffer_layout_entry,
};
use crate::gpu_resource::{OnceGpu, RenderPipelineMap};
use crate::render_graph::gpu_cache::{
    FullscreenRenderPipelineDesc, create_fullscreen_render_pipeline, create_linear_clamp_sampler,
    create_uniform_buffer, create_wgsl_shader_module,
};

/// GPU velocity-pass uniforms matching `MotionVectorUniforms` in `motion_blur.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct MotionVectorParamsGpu {
    /// Matrix transforming current clip position to previous clip position for the left eye.
    pub current_clip_to_prev_clip_left: [[f32; 4]; 4],
    /// Matrix transforming current clip position to previous clip position for the right eye.
    pub current_clip_to_prev_clip_right: [[f32; 4]; 4],
    /// Viewport size in pixels.
    pub viewport_px: [f32; 2],
    /// `1.0` when previous camera history is valid, otherwise `0.0`.
    pub history_valid: f32,
    /// Alignment padding.
    pub _pad0: f32,
}

/// GPU blur-pass uniforms matching `MotionBlurUniforms` in `motion_blur.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct MotionBlurParamsGpu {
    /// Shutter opening as a fraction of the frame interval.
    pub shutter_angle: f32,
    /// Maximum blur radius in pixels.
    pub max_velocity_pixels: f32,
    /// Number of samples along the motion vector.
    pub sample_count: u32,
    /// `1` when the blur resolve should sample velocity, otherwise it copies input.
    pub enabled: u32,
    /// Viewport size in pixels.
    pub viewport_px: [f32; 2],
    /// Alignment padding.
    pub _pad0: [f32; 2],
}

/// Motion blur render pipeline kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum MotionBlurPipelineKind {
    /// Camera-depth velocity generation into `Rg16Float`.
    MotionVectors,
    /// HDR scene-color blur resolve.
    BlurResolve,
}

impl MotionBlurPipelineKind {
    /// Fragment entry point for this pipeline kind.
    fn entry_point(self) -> &'static str {
        match self {
            Self::MotionVectors => "fs_motion_vectors",
            Self::BlurResolve => "fs_motion_blur",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    /// Pipeline variant.
    kind: MotionBlurPipelineKind,
    /// Color target format.
    output_format: wgpu::TextureFormat,
    /// Whether the pipeline records as stereo multiview.
    multiview_stereo: bool,
}

/// GPU state shared by every motion blur pass instance.
#[derive(Default)]
pub(super) struct MotionBlurPipelineCache {
    /// Linear clamp sampler used by the blur resolve.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Bind-group layout for mono depth reprojection.
    motion_vectors_bgl_mono: OnceGpu<wgpu::BindGroupLayout>,
    /// Bind-group layout for stereo multiview depth reprojection.
    motion_vectors_bgl_multiview: OnceGpu<wgpu::BindGroupLayout>,
    /// Bind-group layout for HDR color, velocity, and blur parameters.
    blur_bgl: OnceGpu<wgpu::BindGroupLayout>,
    /// Compiled shader module for the mono variant.
    shader_mono: OnceGpu<wgpu::ShaderModule>,
    /// Compiled shader module for the stereo multiview variant.
    shader_multiview: OnceGpu<wgpu::ShaderModule>,
    /// Lazily created render pipelines keyed by pass kind, target format, and view mode.
    pipelines: RenderPipelineMap<PipelineKey>,
}

impl MotionBlurPipelineCache {
    /// Linear clamp sampler shared by blur resolve texture inputs.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "motion_blur"))
    }

    /// Allocates a per-view motion-vector uniform buffer.
    pub(super) fn create_motion_vector_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        create_uniform_buffer(
            device,
            "motion-blur-vector-params",
            size_of::<MotionVectorParamsGpu>() as u64,
        )
    }

    /// Allocates a per-view blur uniform buffer.
    pub(super) fn create_blur_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        create_uniform_buffer(
            device,
            "motion-blur-params",
            size_of::<MotionBlurParamsGpu>() as u64,
        )
    }

    /// Returns the bind-group layout used by the velocity pass for the active view mode.
    fn motion_vectors_bind_group_layout(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> &wgpu::BindGroupLayout {
        let slot = if multiview_stereo {
            &self.motion_vectors_bgl_multiview
        } else {
            &self.motion_vectors_bgl_mono
        };
        slot.get_or_create(|| {
            let view_dimension = if multiview_stereo {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion-blur-vectors-group0"),
                entries: &[
                    texture_layout_entry(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Depth,
                        view_dimension,
                        false,
                    ),
                    uniform_buffer_layout_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferSize::new(size_of::<MotionVectorParamsGpu>() as u64),
                    ),
                ],
            })
        })
    }

    /// Returns the bind-group layout used by the blur resolve pass.
    fn blur_bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.blur_bgl.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion-blur-resolve-group1"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                    fragment_filterable_d2_array_entry(2),
                    uniform_buffer_layout_entry(
                        3,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferSize::new(size_of::<MotionBlurParamsGpu>() as u64),
                    ),
                ],
            })
        })
    }

    /// Returns the compiled shader module for the active view mode.
    fn shader_module(&self, device: &wgpu::Device, multiview_stereo: bool) -> &wgpu::ShaderModule {
        let slot = if multiview_stereo {
            &self.shader_multiview
        } else {
            &self.shader_mono
        };
        slot.get_or_create(|| {
            let (label, source) = if multiview_stereo {
                (
                    "motion_blur_multiview",
                    embedded_wgsl!("motion_blur_multiview"),
                )
            } else {
                ("motion_blur_default", embedded_wgsl!("motion_blur_default"))
            };
            create_wgsl_shader_module(device, label, source)
        })
    }

    /// Returns or builds a motion-blur render pipeline.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        kind: MotionBlurPipelineKind,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let key = PipelineKey {
            kind,
            output_format,
            multiview_stereo,
        };
        self.pipelines.get_or_create(key, |key| {
            let shader = self.shader_module(device, key.multiview_stereo).clone();
            let motion_vectors_bgl =
                self.motion_vectors_bind_group_layout(device, key.multiview_stereo);
            let blur_bgl = self.blur_bind_group_layout(device);
            let layouts: &[Option<&wgpu::BindGroupLayout>] = match key.kind {
                MotionBlurPipelineKind::MotionVectors => &[Some(motion_vectors_bgl)],
                MotionBlurPipelineKind::BlurResolve => &[None, Some(blur_bgl)],
            };
            let label = format!("motion-blur-{:?}", key.kind);
            create_fullscreen_render_pipeline(
                device,
                FullscreenRenderPipelineDesc {
                    label: &label,
                    bind_group_layouts: layouts,
                    shader: &shader,
                    fragment_entry: key.kind.entry_point(),
                    output_format: key.output_format,
                    blend: None,
                    multiview_stereo: key.multiview_stereo,
                },
            )
        })
    }

    /// Builds a velocity-pass bind group.
    pub(super) fn motion_vectors_bind_group(
        &self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion-blur-vectors-group0"),
            layout: self.motion_vectors_bind_group_layout(device, multiview_stereo),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "passes::motion_vectors_bind_group");
        bind_group
    }

    /// Builds a blur-resolve bind group.
    pub(super) fn blur_bind_group(
        &self,
        device: &wgpu::Device,
        scene_color: &wgpu::TextureView,
        velocity: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion-blur-resolve-group1"),
            layout: self.blur_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(velocity),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        crate::profiling::note_resource_churn!(BindGroup, "passes::motion_blur_bind_group");
        bind_group
    }
}
