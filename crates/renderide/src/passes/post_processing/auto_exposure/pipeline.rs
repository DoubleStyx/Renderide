//! Cached pipelines, bind layouts, and per-view GPU state for auto-exposure.

use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::config::AutoExposureSettings;
use crate::embedded_shaders::{
    AUTO_EXPOSURE_APPLY_DEFAULT_WGSL, AUTO_EXPOSURE_APPLY_MULTIVIEW_WGSL,
    AUTO_EXPOSURE_HISTOGRAM_WGSL,
};
use crate::gpu::bind_layout::{
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry, texture_layout_entry,
    uniform_buffer_layout_entry,
};
use crate::render_graph::gpu_cache::{
    FullscreenPipelineVariantDesc, FullscreenShaderVariants, OnceGpu, RenderPipelineMap,
    create_d2_array_view, create_linear_clamp_sampler, create_wgsl_shader_module,
    fullscreen_pipeline_variant,
};

/// Number of histogram bins used by the auto-exposure compute pass.
pub(super) const HISTOGRAM_BIN_COUNT: u64 = 64;
/// Workgroup width used by `compute_histogram`.
pub(super) const HISTOGRAM_WORKGROUP_WIDTH: u32 = 16;
/// Workgroup height used by `compute_histogram`.
pub(super) const HISTOGRAM_WORKGROUP_HEIGHT: u32 = 16;

const HISTOGRAM_BUFFER_SIZE: u64 = HISTOGRAM_BIN_COUNT * size_of::<u32>() as u64;
const EXPOSURE_BUFFER_SIZE: u64 = size_of::<f32>() as u64;

/// CPU mirror of the WGSL `AutoExposureParams` uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct AutoExposureParamsGpu {
    min_log_lum: f32,
    inv_log_lum_range: f32,
    log_lum_range: f32,
    low_percent: f32,
    high_percent: f32,
    speed_brighten: f32,
    speed_darken: f32,
    exponential_transition_distance: f32,
    compensation_ev: f32,
    delta_time_seconds: f32,
    layer_count: u32,
    _pad: u32,
}

impl AutoExposureParamsGpu {
    pub(super) fn from_settings(
        settings: AutoExposureSettings,
        delta_seconds: f32,
        layer_count: u32,
    ) -> Self {
        let (min_log_lum, max_log_lum) = settings.resolved_ev_range();
        let log_lum_range = max_log_lum - min_log_lum;
        let (low_percent, high_percent) = settings.resolved_filter();
        Self {
            min_log_lum,
            inv_log_lum_range: 1.0 / log_lum_range,
            log_lum_range,
            low_percent,
            high_percent,
            speed_brighten: settings.resolved_speed_brighten(),
            speed_darken: settings.resolved_speed_darken(),
            exponential_transition_distance: settings.resolved_exponential_transition_distance(),
            compensation_ev: settings.resolved_compensation_ev(),
            delta_time_seconds: delta_seconds.max(0.0),
            layer_count: layer_count.max(1),
            _pad: 0,
        }
    }
}

/// Per-view GPU buffers retained by the auto-exposure effect.
pub(super) struct ViewAutoExposureGpuState {
    /// Per-view settings uniform buffer.
    pub(super) settings: wgpu::Buffer,
    /// Per-view histogram storage buffer.
    pub(super) histogram: wgpu::Buffer,
    /// Per-view persistent exposure EV storage buffer.
    pub(super) exposure: wgpu::Buffer,
}

impl ViewAutoExposureGpuState {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            settings: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("auto_exposure_settings"),
                size: size_of::<AutoExposureParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            histogram: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("auto_exposure_histogram"),
                size: HISTOGRAM_BUFFER_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            exposure: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("auto_exposure_ev"),
                size: EXPOSURE_BUFFER_SIZE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

/// Process-wide pipeline cache for the auto-exposure compute and apply passes.
#[derive(Default)]
pub(super) struct AutoExposurePipelineCache {
    compute_bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    apply_bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    sampler: OnceGpu<wgpu::Sampler>,
    histogram_pipeline: OnceGpu<wgpu::ComputePipeline>,
    average_pipeline: OnceGpu<wgpu::ComputePipeline>,
    mono_apply: RenderPipelineMap<wgpu::TextureFormat>,
    multiview_apply: RenderPipelineMap<wgpu::TextureFormat>,
}

impl AutoExposurePipelineCache {
    fn compute_bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.compute_bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("auto_exposure_compute"),
                entries: &[
                    uniform_buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        NonZeroU64::new(size_of::<AutoExposureParamsGpu>() as u64),
                    ),
                    texture_layout_entry(
                        1,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::TextureSampleType::Float { filterable: false },
                        wgpu::TextureViewDimension::D2Array,
                        false,
                    ),
                    storage_buffer_layout_entry(
                        2,
                        wgpu::ShaderStages::COMPUTE,
                        false,
                        HISTOGRAM_BUFFER_SIZE,
                    ),
                    storage_buffer_layout_entry(
                        3,
                        wgpu::ShaderStages::COMPUTE,
                        false,
                        EXPOSURE_BUFFER_SIZE,
                    ),
                ],
            })
        })
    }

    fn apply_bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.apply_bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("auto_exposure_apply"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                    storage_buffer_layout_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        true,
                        EXPOSURE_BUFFER_SIZE,
                    ),
                ],
            })
        })
    }

    pub(super) fn histogram_pipeline(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.histogram_pipeline.get_or_create(|| {
            create_auto_exposure_compute_pipeline(
                device,
                self.compute_bind_group_layout(device),
                "compute_histogram",
            )
        })
    }

    pub(super) fn average_pipeline(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        self.average_pipeline.get_or_create(|| {
            create_auto_exposure_compute_pipeline(
                device,
                self.compute_bind_group_layout(device),
                "compute_average",
            )
        })
    }

    pub(super) fn apply_pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let bind_group_layout = self.apply_bind_group_layout(device);
        fullscreen_pipeline_variant(
            device,
            FullscreenPipelineVariantDesc {
                output_format,
                multiview_stereo,
                mono: &self.mono_apply,
                multiview: &self.multiview_apply,
                shader: FullscreenShaderVariants {
                    mono_label: "auto_exposure_apply_default",
                    mono_source: AUTO_EXPOSURE_APPLY_DEFAULT_WGSL,
                    multiview_label: "auto_exposure_apply_multiview",
                    multiview_source: AUTO_EXPOSURE_APPLY_MULTIVIEW_WGSL,
                },
                bind_group_layouts: &[Some(bind_group_layout)],
                log_name: "auto_exposure_apply",
            },
        )
    }

    pub(super) fn compute_bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
        state: &ViewAutoExposureGpuState,
    ) -> wgpu::BindGroup {
        let view = create_d2_array_view(
            scene_color_texture,
            "auto_exposure_histogram_src",
            multiview_stereo,
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("auto_exposure_compute"),
            layout: self.compute_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.settings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.histogram.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.exposure.as_entire_binding(),
                },
            ],
        })
    }

    pub(super) fn apply_bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
        state: &ViewAutoExposureGpuState,
    ) -> wgpu::BindGroup {
        let view = create_d2_array_view(
            scene_color_texture,
            "auto_exposure_apply_src",
            multiview_stereo,
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("auto_exposure_apply"),
            layout: self.apply_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.exposure.as_entire_binding(),
                },
            ],
        })
    }

    fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "auto_exposure_apply"))
    }
}

fn storage_buffer_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
    min_binding_size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_binding_size),
        },
        count: None,
    }
}

fn create_auto_exposure_compute_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    logger::debug!("auto_exposure: building compute pipeline {entry_point}");
    let shader = create_wgsl_shader_module(
        device,
        "auto_exposure_histogram",
        AUTO_EXPOSURE_HISTOGRAM_WGSL,
    );
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("auto_exposure_histogram"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{AutoExposureParamsGpu, HISTOGRAM_BIN_COUNT};

    #[test]
    fn auto_exposure_params_are_uniform_aligned() {
        assert_eq!(size_of::<AutoExposureParamsGpu>(), 48);
        assert_eq!(size_of::<AutoExposureParamsGpu>() % 16, 0);
    }

    #[test]
    fn histogram_has_expected_bin_count() {
        assert_eq!(HISTOGRAM_BIN_COUNT, 64);
    }
}
