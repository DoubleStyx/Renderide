//! Post-processing renderer-config HUD controls.

use crate::config::{
    AutoExposureSettings, BloomCompositeMode, GTAO_MAX_DENOISE_PASSES, GTAO_MAX_QUALITY_LEVEL,
    GTAO_MAX_RESOLUTION_DIVISOR, GTAO_MAX_SLICE_COUNT, GTAO_MAX_STEPS_PER_SLICE, GtaoSettings,
    MotionBlurSettings, RendererSettings, TonemapMode,
};

use super::controls::{drag_f32_slider_setting, drag_u32_slider_setting};

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
    if ui.slider(
        "Quality level",
        0,
        GTAO_MAX_QUALITY_LEVEL,
        &mut gtao.quality_level,
    ) {
        *dirty = true;
    }
    ui.text_disabled("0 = low, 1 = medium, 2 = high, 3 = ultra, 4 = experimental high.");
    if drag_u32_slider_setting(
        ui,
        "Slice override",
        &mut gtao.slice_count_override,
        0,
        GTAO_MAX_SLICE_COUNT,
    ) {
        *dirty = true;
    }
    ui.text_disabled("0 uses the selected preset.");
    if drag_u32_slider_setting(
        ui,
        "Steps override",
        &mut gtao.step_count,
        0,
        GTAO_MAX_STEPS_PER_SLICE,
    ) {
        *dirty = true;
    }
    ui.text_disabled("0 uses the selected preset.");
    if drag_u32_slider_setting(
        ui,
        "Resolution divisor",
        &mut gtao.resolution_divisor,
        1,
        GTAO_MAX_RESOLUTION_DIVISOR,
    ) {
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
    if drag_f32_slider_setting(
        ui,
        "Radius (m)",
        &mut gtao.radius_meters,
        0.01,
        10.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Radius multiplier",
        &mut gtao.radius_multiplier,
        0.1,
        8.0,
        Some("%.3f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(ui, "Intensity", &mut gtao.intensity, 0.0, 8.0, Some("%.2f")) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Max pixel radius",
        &mut gtao.max_pixel_radius,
        1.0,
        4096.0,
        Some("%.0f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Falloff range",
        &mut gtao.falloff_range,
        0.01,
        2.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Sample distribution power",
        &mut gtao.sample_distribution_power,
        0.25,
        6.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Thin occluder compensation",
        &mut gtao.thin_occluder_compensation,
        0.0,
        2.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Final value power",
        &mut gtao.final_value_power,
        0.1,
        12.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Depth mip sampling offset",
        &mut gtao.depth_mip_sampling_offset,
        -8.0,
        30.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Multi-bounce albedo",
        &mut gtao.albedo_multibounce,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    ui.unindent();
}

/// Edits GTAO denoise pass controls.
fn gtao_denoise_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Denoise");
    ui.indent();
    if drag_u32_slider_setting(
        ui,
        "Denoise passes",
        &mut gtao.denoise_passes,
        0,
        GTAO_MAX_DENOISE_PASSES,
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Denoise blur beta",
        &mut gtao.denoise_blur_beta,
        0.0,
        16.0,
        Some("%.2f"),
    ) {
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
    if drag_f32_slider_setting(
        ui,
        "Intensity",
        &mut bloom.intensity,
        0.0,
        1.0,
        Some("%.3f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Low-frequency boost",
        &mut bloom.low_frequency_boost,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Low-frequency boost curvature",
        &mut bloom.low_frequency_boost_curvature,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "High-pass frequency",
        &mut bloom.high_pass_frequency,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Prefilter threshold (HDR)",
        &mut bloom.prefilter_threshold,
        0.0,
        8.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Prefilter threshold softness",
        &mut bloom.prefilter_threshold_softness,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
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
    if drag_u32_slider_setting(
        ui,
        "Max mip dimension (px)",
        &mut bloom.max_mip_dimension,
        64,
        2048,
    ) {
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
    if drag_f32_slider_setting(
        ui,
        "Shutter angle",
        &mut motion_blur.shutter_angle,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Samples",
        &mut motion_blur.sample_count,
        0,
        MotionBlurSettings::MAX_SAMPLE_COUNT,
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Max velocity (px)",
        &mut motion_blur.max_velocity_pixels,
        0.0,
        512.0,
        Some("%.0f"),
    ) {
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
    if drag_f32_slider_setting(ui, "Min EV", &mut auto.min_ev, -16.0, 16.0, Some("%.2f")) {
        *dirty = true;
    }
    if drag_f32_slider_setting(ui, "Max EV", &mut auto.max_ev, -16.0, 16.0, Some("%.2f")) {
        *dirty = true;
    }
    let (min_ev, max_ev) = auto.resolved_ev_range();
    if (min_ev, max_ev) != (auto.min_ev, auto.max_ev) {
        ui.text_disabled(format!("Effective EV range: {min_ev:.2} to {max_ev:.2}."));
    }
    if drag_f32_slider_setting(
        ui,
        "Low percentile",
        &mut auto.low_percent,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "High percentile",
        &mut auto.high_percent,
        0.0,
        1.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    let (low, high) = auto.resolved_filter();
    if (low, high) != (auto.low_percent, auto.high_percent) {
        ui.text_disabled(format!(
            "Effective percentile filter: {low:.2} to {high:.2}."
        ));
    }
    if drag_f32_slider_setting(
        ui,
        "Brighten speed (EV/s)",
        &mut auto.speed_brighten,
        0.0,
        12.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Darken speed (EV/s)",
        &mut auto.speed_darken,
        0.0,
        12.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Transition distance (EV)",
        &mut auto.exponential_transition_distance,
        AutoExposureSettings::MIN_TRANSITION_DISTANCE,
        8.0,
        Some("%.2f"),
    ) {
        *dirty = true;
    }
    if drag_f32_slider_setting(
        ui,
        "Middle-gray compensation (EV)",
        &mut auto.compensation_ev,
        -8.0,
        8.0,
        Some("%.2f"),
    ) {
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
