//! GTAO shader parameter packing and shared uniform buffer.

use bytemuck::{Pod, Zeroable};

use crate::config::GtaoSettings;
use crate::gpu_resource::OnceGpu;
use crate::render_graph::gpu_cache::create_uniform_buffer;

/// AO term and packed-edges target format. Both intermediates use `R8Unorm` so wgpu can
/// render-attach them and the shaders can sample with floating-point math throughout.
pub(in crate::passes::post_processing::gtao) const AO_TERM_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::R8Unorm;
/// Packed-edges target format (mirrors the AO term).
pub(in crate::passes::post_processing::gtao) const EDGES_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::R8Unorm;
/// View-space depth prefilter format.
pub(in crate::passes::post_processing::gtao) const VIEW_DEPTH_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::R32Float;
/// Number of view-space depth mips generated for the horizon search.
pub(in crate::passes::post_processing::gtao) const VIEW_DEPTH_MIP_COUNT: u32 = 5;

/// CPU mirror of the WGSL `GtaoParams` uniform (64 bytes, 16-byte aligned).
///
/// Rewritten every record from the live [`crate::config::GtaoSettings`] (with `final_apply`
/// and `denoise_blur_beta` adjusted per-stage).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(in crate::passes::post_processing::gtao) struct GtaoParamsGpu {
    /// World-space search radius (meters).
    pub radius_world: f32,
    /// Radius scale used to compensate for screen-space bias.
    pub radius_multiplier: f32,
    /// Cap on the horizon search in pixels.
    pub max_pixel_radius: f32,
    /// AO strength exponent applied to the raw visibility factor.
    pub intensity: f32,
    /// Distance-falloff range as a fraction of `radius_world`.
    pub falloff_range: f32,
    /// Step-distribution power; higher values bias samples toward the center pixel.
    pub sample_distribution_power: f32,
    /// Depth thickness compensation for thin occluders.
    pub thin_occluder_compensation: f32,
    /// Final visibility power applied after slice averaging.
    pub final_value_power: f32,
    /// Bias for selecting the prefiltered depth mip used by horizon samples.
    pub depth_mip_sampling_offset: f32,
    /// Gray-albedo proxy for the multi-bounce fit.
    pub albedo_multibounce: f32,
    /// Bilateral blur strength for the active denoise stage.
    pub denoise_blur_beta: f32,
    /// Number of slice directions selected from the quality preset.
    pub slice_count: u32,
    /// Number of steps per slice selected from the quality preset.
    pub steps_per_slice: u32,
    /// Set to `1` on the apply stage, `0` on production and intermediate denoise.
    pub final_apply: u32,
    /// Number of valid view-depth mips bound for the production shader.
    pub view_depth_mip_count: u32,
    /// Padding to keep the uniform block size 16-byte aligned.
    pub _pad1: u32,
}

impl GtaoParamsGpu {
    /// Builds stage-specific GPU parameters from live settings.
    pub(in crate::passes::post_processing::gtao) fn from_settings(
        settings: GtaoSettings,
        denoise_blur_beta: f32,
        final_apply: bool,
    ) -> Self {
        let preset = GtaoQualityPreset::from_level(settings.quality_level, settings.step_count);
        Self {
            radius_world: settings.radius_meters.max(0.0),
            radius_multiplier: settings.radius_multiplier.clamp(0.3, 3.0),
            max_pixel_radius: settings.max_pixel_radius.max(1.0),
            intensity: settings.intensity.max(0.0),
            falloff_range: settings.falloff_range.clamp(0.05, 1.0),
            sample_distribution_power: settings.sample_distribution_power.clamp(1.0, 3.0),
            thin_occluder_compensation: settings.thin_occluder_compensation.clamp(0.0, 0.7),
            final_value_power: settings.final_value_power.clamp(0.5, 5.0),
            depth_mip_sampling_offset: settings.depth_mip_sampling_offset.clamp(0.0, 30.0),
            albedo_multibounce: settings.albedo_multibounce.clamp(0.0, 0.99),
            denoise_blur_beta,
            slice_count: preset.slice_count,
            steps_per_slice: preset.steps_per_slice,
            final_apply: u32::from(final_apply),
            view_depth_mip_count: VIEW_DEPTH_MIP_COUNT,
            _pad1: 0,
        }
    }

    /// Returns a copy with the view-depth mip count clamped to the shader's supported range.
    pub(in crate::passes::post_processing::gtao) fn with_view_depth_mip_count(
        mut self,
        mip_count: u32,
    ) -> Self {
        self.view_depth_mip_count = mip_count.clamp(1, VIEW_DEPTH_MIP_COUNT);
        self
    }
}

/// Sampling preset selected by `GtaoSettings::quality_level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::passes::post_processing::gtao) struct GtaoQualityPreset {
    /// Slice directions evaluated by the horizon search.
    pub(in crate::passes::post_processing::gtao) slice_count: u32,
    /// Steps evaluated per slice direction.
    pub(in crate::passes::post_processing::gtao) steps_per_slice: u32,
}

impl GtaoQualityPreset {
    /// Maps quality levels to slice/step layouts.
    pub(in crate::passes::post_processing::gtao) fn from_level(
        level: u32,
        step_count_floor: u32,
    ) -> Self {
        let preset = match level.min(3) {
            0 => Self {
                slice_count: 1,
                steps_per_slice: 2,
            },
            1 => Self {
                slice_count: 2,
                steps_per_slice: 2,
            },
            2 => Self {
                slice_count: 3,
                steps_per_slice: 3,
            },
            _ => Self {
                slice_count: 9,
                steps_per_slice: 3,
            },
        };
        Self {
            slice_count: preset.slice_count,
            steps_per_slice: preset.steps_per_slice.max(step_count_floor.clamp(1, 8)),
        }
    }
}

/// Process-wide `GtaoParams` uniform buffer, shared across the pipeline caches.
#[derive(Default)]
pub(in crate::passes::post_processing::gtao) struct GtaoParamsBuffer {
    buffer: OnceGpu<wgpu::Buffer>,
}

impl GtaoParamsBuffer {
    /// Returns the resident GTAO params buffer, creating it on first use.
    pub(in crate::passes::post_processing::gtao) fn get(
        &self,
        device: &wgpu::Device,
    ) -> &wgpu::Buffer {
        self.buffer.get_or_create(|| {
            create_uniform_buffer(device, "gtao-params", size_of::<GtaoParamsGpu>() as u64)
        })
    }
}
