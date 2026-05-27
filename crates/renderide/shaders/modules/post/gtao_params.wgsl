//! Shared GTAO pass parameter layout.

#define_import_path renderide::post::gtao_params

struct GtaoParams {
    radius_world: f32,
    radius_multiplier: f32,
    max_pixel_radius: f32,
    intensity: f32,
    falloff_range: f32,
    sample_distribution_power: f32,
    thin_occluder_compensation: f32,
    final_value_power: f32,
    depth_mip_sampling_offset: f32,
    albedo_multibounce: f32,
    denoise_blur_beta: f32,
    slice_count: u32,
    steps_per_slice: u32,
    final_apply: u32,
    view_depth_mip_count: u32,
    resolution_divisor: u32,
}
