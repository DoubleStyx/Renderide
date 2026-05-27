//! Post-processing renderer-config HUD controls.

use crate::config::{
    AutoExposureSettings, BloomCompositeMode, GTAO_MAX_DENOISE_PASSES, GTAO_MAX_QUALITY_LEVEL,
    GTAO_MAX_RESOLUTION_DIVISOR, GTAO_MAX_SLICE_COUNT, GTAO_MAX_STEPS_PER_SLICE, GtaoSettings,
    MotionBlurSettings, RendererSettings, TonemapMode,
};

/// Master toggle, GTAO, bloom, motion blur, auto-exposure, and tonemap settings.
pub(super) fn post_processing_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Post-Processing");
    ui.indent();
    post_processing_master(ui, g, dirty);
    ui.separator();
    post_processing_gtao(ui, g, dirty);
    ui.separator();
    post_processing_bloom(ui, g, dirty);
    ui.separator();
    post_processing_motion_blur(ui, g, dirty);
    ui.separator();
    post_processing_auto_exposure(ui, g, dirty);
    ui.separator();
    post_processing_tonemap(ui, g, dirty);
    ui.unindent();
}

/// Edits the master post-processing stack toggle.
fn post_processing_master(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("master");
    if ui.checkbox(
        "Enable post-processing stack",
        &mut g.post_processing.enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled(
        "Master toggle for the post-processing chain (HDR scene color -> display target). \
         Applied on the next frame (the render graph is rebuilt automatically when the chain \
         topology changes).",
    );
}

/// Edits GTAO enablement and quality controls.
fn post_processing_gtao(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("gtao");
    ui.text_disabled(
        "GTAO (Ground-Truth Ambient Occlusion): samples the forward view-normal prepass \
         and modulates HDR scene color by a physical visibility factor. Runs pre-tonemap.",
    );
    if ui.checkbox("Enable GTAO", &mut g.post_processing.gtao.enabled) {
        *dirty = true;
    }
    let gtao = &mut g.post_processing.gtao;
    gtao_quality_controls(ui, gtao, dirty);
    gtao_sampling_controls(ui, gtao, dirty);
    gtao_denoise_controls(ui, gtao, dirty);
}

/// Edits coarse GTAO quality controls.
fn gtao_quality_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Quality");
    ui.indent();
    if ui
        .slider_config("Quality level", 0_u32, GTAO_MAX_QUALITY_LEVEL)
        .build(&mut gtao.quality_level)
    {
        *dirty = true;
    }
    ui.text_disabled("0 = low, 1 = medium, 2 = high, 3 = ultra, 4 = experimental high.");
    if ui
        .slider_config("Slice override", 0_u32, GTAO_MAX_SLICE_COUNT)
        .build(&mut gtao.slice_count_override)
    {
        *dirty = true;
    }
    ui.text_disabled("0 uses the selected preset.");
    if ui
        .slider_config("Steps override", 0_u32, GTAO_MAX_STEPS_PER_SLICE)
        .build(&mut gtao.step_count)
    {
        *dirty = true;
    }
    ui.text_disabled("0 uses the selected preset.");
    if ui
        .slider_config("Resolution divisor", 1_u32, GTAO_MAX_RESOLUTION_DIVISOR)
        .build(&mut gtao.resolution_divisor)
    {
        *dirty = true;
    }
    let (effective_slices, effective_steps) = gtao.effective_sample_counts();
    let effective_divisor = gtao.effective_resolution_divisor();
    ui.text_disabled(format!(
        "Effective: {effective_slices} slices x {effective_steps} steps, divisor {effective_divisor}, ~{} depth taps per AO pixel.",
        gtao.approximate_depth_samples_per_pixel(),
    ));
    ui.unindent();
}

/// Edits GTAO sampling radius, falloff, distribution, and intensity controls.
fn gtao_sampling_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Sampling");
    ui.indent();
    if ui
        .slider_config("Radius (m)", 0.01_f32, 10.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.radius_meters)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Radius multiplier", 0.1_f32, 8.0_f32)
        .display_format("%.3f")
        .build(&mut gtao.radius_multiplier)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Intensity", 0.0_f32, 8.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.intensity)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Max pixel radius", 1.0_f32, 4096.0_f32)
        .display_format("%.0f")
        .build(&mut gtao.max_pixel_radius)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Falloff range", 0.01_f32, 2.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.falloff_range)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Sample distribution power", 0.25_f32, 6.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.sample_distribution_power)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Thin occluder compensation", 0.0_f32, 2.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.thin_occluder_compensation)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Final value power", 0.1_f32, 12.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.final_value_power)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Depth mip sampling offset", -8.0_f32, 30.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.depth_mip_sampling_offset)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Multi-bounce albedo", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.albedo_multibounce)
    {
        *dirty = true;
    }
    ui.unindent();
}

/// Edits GTAO denoise pass controls.
fn gtao_denoise_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Denoise");
    ui.indent();
    if ui
        .slider_config("Denoise passes", 0_u32, GTAO_MAX_DENOISE_PASSES)
        .build(&mut gtao.denoise_passes)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Denoise blur beta", 0.0_f32, 16.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.denoise_blur_beta)
    {
        *dirty = true;
    }
    ui.unindent();
}

/// Edits bloom enablement, thresholds, mip policy, and composite mode.
fn post_processing_bloom(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("bloom");
    ui.text_disabled(
        "Bloom (dual-filter): HDR-linear scatter via a \
         mip-chain downsample/upsample pyramid with Karis firefly reduction on mip 0. Runs \
         pre-tonemap. Energy-conserving mode redistributes the bloom source term. Changing \
         `max mip dimension` rebuilds the render graph; other knobs take effect next frame via the \
         shared params UBO / per-mip blend constant.",
    );
    if ui.checkbox("Enable bloom", &mut g.post_processing.bloom.enabled) {
        *dirty = true;
    }
    let bloom = &mut g.post_processing.bloom;
    if ui
        .slider_config("Intensity", 0.0_f32, 1.0_f32)
        .display_format("%.3f")
        .build(&mut bloom.intensity)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Low-frequency boost", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.low_frequency_boost)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Low-frequency boost curvature", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.low_frequency_boost_curvature)
    {
        *dirty = true;
    }
    if ui
        .slider_config("High-pass frequency", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.high_pass_frequency)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Prefilter threshold (HDR)", 0.0_f32, 8.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.prefilter_threshold)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Prefilter threshold softness", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.prefilter_threshold_softness)
    {
        *dirty = true;
    }
    ui.text("Composite mode");
    for (i, &mode) in BloomCompositeMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(0x1000 + i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(bloom.composite_mode == mode)
            .build()
        {
            bloom.composite_mode = mode;
            *dirty = true;
        }
    }
    if ui
        .slider_config("Max mip dimension (px)", 64_u32, 2048_u32)
        .build(&mut bloom.max_mip_dimension)
    {
        *dirty = true;
    }
    let effective_max_mip_dimension = bloom.effective_max_mip_dimension();
    if effective_max_mip_dimension != bloom.max_mip_dimension {
        ui.text_disabled(format!(
            "Effective max mip dimension: {effective_max_mip_dimension} px (rounded down to power of two)."
        ));
    }
}

/// Edits motion-blur enablement and velocity/shutter controls.
fn post_processing_motion_blur(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("motion_blur");
    ui.text_disabled(
        "Motion blur: derives screen-space velocity only while enabled, then blurs HDR scene \
         color after bloom and before tonemapping. VR views are opt-in.",
    );
    if ui.checkbox(
        "Enable motion blur",
        &mut g.post_processing.motion_blur.enabled,
    ) {
        *dirty = true;
    }
    if ui.checkbox("Allow in VR", &mut g.post_processing.motion_blur.allow_vr) {
        *dirty = true;
    }
    let motion_blur = &mut g.post_processing.motion_blur;
    if ui
        .slider_config("Shutter angle", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut motion_blur.shutter_angle)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Samples", 0_u32, MotionBlurSettings::MAX_SAMPLE_COUNT)
        .build(&mut motion_blur.sample_count)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Max velocity (px)", 0.0_f32, 512.0_f32)
        .display_format("%.0f")
        .build(&mut motion_blur.max_velocity_pixels)
    {
        *dirty = true;
    }
    if !motion_blur.is_effectively_enabled() {
        ui.text_disabled("Effective state: disabled by zero samples, shutter, or velocity clamp.");
    }
}

/// Edits auto-exposure histogram, percentile, and adaptation controls.
fn post_processing_auto_exposure(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("auto_exposure");
    ui.text_disabled(
        "Auto-exposure: builds a scene-linear log-luminance histogram, targets middle gray, ignores dark/bright percentile tails, and adapts exposure before tonemapping.",
    );
    if ui.checkbox(
        "Enable auto-exposure",
        &mut g.post_processing.auto_exposure.enabled,
    ) {
        *dirty = true;
    }
    let auto = &mut g.post_processing.auto_exposure;
    if ui
        .slider_config("Min EV", -16.0_f32, 16.0_f32)
        .display_format("%.2f")
        .build(&mut auto.min_ev)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Max EV", -16.0_f32, 16.0_f32)
        .display_format("%.2f")
        .build(&mut auto.max_ev)
    {
        *dirty = true;
    }
    let (min_ev, max_ev) = auto.resolved_ev_range();
    if (min_ev, max_ev) != (auto.min_ev, auto.max_ev) {
        ui.text_disabled(format!("Effective EV range: {min_ev:.2} to {max_ev:.2}."));
    }
    if ui
        .slider_config("Low percentile", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut auto.low_percent)
    {
        *dirty = true;
    }
    if ui
        .slider_config("High percentile", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut auto.high_percent)
    {
        *dirty = true;
    }
    let (low, high) = auto.resolved_filter();
    if (low, high) != (auto.low_percent, auto.high_percent) {
        ui.text_disabled(format!(
            "Effective percentile filter: {low:.2} to {high:.2}."
        ));
    }
    if ui
        .slider_config("Brighten speed (EV/s)", 0.0_f32, 12.0_f32)
        .display_format("%.2f")
        .build(&mut auto.speed_brighten)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Darken speed (EV/s)", 0.0_f32, 12.0_f32)
        .display_format("%.2f")
        .build(&mut auto.speed_darken)
    {
        *dirty = true;
    }
    if ui
        .slider_config(
            "Transition distance (EV)",
            AutoExposureSettings::MIN_TRANSITION_DISTANCE,
            8.0_f32,
        )
        .display_format("%.2f")
        .build(&mut auto.exponential_transition_distance)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Middle-gray compensation (EV)", -8.0_f32, 8.0_f32)
        .display_format("%.2f")
        .build(&mut auto.compensation_ev)
    {
        *dirty = true;
    }
}

/// Edits tonemap operator selection.
fn post_processing_tonemap(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("tonemap");
    ui.text_disabled("Tonemap (HDR linear -> display-referred 0..1 linear).");
    for (i, &mode) in TonemapMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.post_processing.tonemap.mode == mode)
            .build()
        {
            g.post_processing.tonemap.mode = mode;
            *dirty = true;
        }
    }
    ui.text_disabled(
        "ACES Fitted is filmic with stronger hue shifts. AgX is more neutral. `None` skips \
         tonemapping (HDR pass-through; values >1 will clip in the swapchain).",
    );
}
