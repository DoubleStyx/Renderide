//! Post-processing graph cache rebuild tests for [`super::super::RenderBackend`].

use std::sync::{Arc, RwLock};

use super::*;
use crate::config::{
    GtaoSettings, PostProcessingSettings, RendererSettings, TonemapMode, TonemapSettings,
};
use crate::render_graph::{
    GraphCacheKey, RenderPathProfile, ViewFamilyGraphRequirements, ViewPostProcessing,
    post_process_chain::PostProcessChainSignature,
};
use hashbrown::HashMap;

fn settings_handle(post: PostProcessingSettings) -> RendererSettingsHandle {
    Arc::new(RwLock::new(RendererSettings {
        post_processing: post,
        ..Default::default()
    }))
}

/// Returns the current cached graph key.
fn cached_graph_key(backend: &RenderBackend) -> GraphCacheKey {
    backend
        .graph_state
        .frame_graph_cache
        .last_key()
        .expect("graph key should exist after sync")
}

fn desktop_requirements() -> ViewFamilyGraphRequirements {
    ViewFamilyGraphRequirements::from_profile(RenderPathProfile::desktop_main(), false)
}

fn xr_requirements() -> ViewFamilyGraphRequirements {
    ViewFamilyGraphRequirements::from_profile(RenderPathProfile::xr_hmd(), true)
}

fn headless_requirements() -> ViewFamilyGraphRequirements {
    ViewFamilyGraphRequirements::from_profile(RenderPathProfile::headless_main(), false)
}

fn camera_readback_without_motion_blur_requirements() -> ViewFamilyGraphRequirements {
    ViewFamilyGraphRequirements::from_profile(
        RenderPathProfile::camera_readback(ViewPostProcessing::new(true, true, false)),
        false,
    )
}

fn limits_with_format_usage(
    format: wgpu::TextureFormat,
    allowed_usages: wgpu::TextureUsages,
) -> GpuLimits {
    let mut format_features = HashMap::new();
    format_features.insert(
        format,
        wgpu::TextureFormatFeatures {
            allowed_usages,
            flags: wgpu::TextureFormatFeatureFlags::empty(),
        },
    );
    GpuLimits::synthetic_for_tests(
        wgpu::Limits {
            max_texture_dimension_2d: 4096,
            max_storage_buffer_binding_size: 256 * 1024,
            ..Default::default()
        },
        wgpu::Features::empty(),
        format_features,
    )
}

/// First sync builds the graph and stores the live signature.
#[test]
fn first_sync_builds_graph_and_records_signature() {
    let mut backend = RenderBackend::new();
    let handle = settings_handle(PostProcessingSettings {
        enabled: true,
        auto_exposure: crate::config::AutoExposureSettings {
            enabled: true,
            ..Default::default()
        },
        tonemap: TonemapSettings {
            mode: TonemapMode::AcesFitted,
        },
        ..Default::default()
    });
    backend.renderer_settings = Some(handle);
    backend.ensure_frame_graph_in_sync(desktop_requirements());
    assert!(
        backend.frame_graph_pass_count() > 0,
        "graph should be built"
    );
    assert_eq!(
        cached_graph_key(&backend).post_processing,
        PostProcessChainSignature {
            aces_tonemap: true,
            agx_tonemap: false,
            auto_exposure: true,
            bloom: true,
            bloom_max_mip_dimension: 512,
            gtao: true,
            gtao_denoise_passes: GtaoSettings::default().effective_denoise_passes(),
            gtao_resolution_divisor: GtaoSettings::default().effective_resolution_divisor(),
            motion_blur: true,
        }
    );
}

/// Toggling the master enable flips the signature and rebuilds the graph with an extra pass.
#[test]
fn signature_change_triggers_rebuild() {
    let mut backend = RenderBackend::new();
    let handle = settings_handle(PostProcessingSettings {
        enabled: false,
        ..Default::default()
    });
    backend.renderer_settings = Some(Arc::clone(&handle));
    backend.ensure_frame_graph_in_sync(desktop_requirements());
    let initial_passes = backend.frame_graph_pass_count();
    let initial_signature = cached_graph_key(&backend).post_processing;

    if let Ok(mut g) = handle.write() {
        g.post_processing.enabled = true;
        g.post_processing.tonemap.mode = TonemapMode::AcesFitted;
    }
    backend.ensure_frame_graph_in_sync(desktop_requirements());

    assert_ne!(
        cached_graph_key(&backend).post_processing,
        initial_signature,
        "signature must update after rebuild"
    );
    assert!(
        backend.frame_graph_pass_count() > initial_passes,
        "enabling ACES should add a graph pass"
    );
}

/// Repeat sync without HUD edits is a no-op (no rebuild, signature and pass count unchanged).
#[test]
fn unchanged_signature_does_not_rebuild() {
    let mut backend = RenderBackend::new();
    let handle = settings_handle(PostProcessingSettings {
        enabled: true,
        tonemap: TonemapSettings {
            mode: TonemapMode::AcesFitted,
        },
        ..Default::default()
    });
    backend.renderer_settings = Some(handle);
    backend.ensure_frame_graph_in_sync(desktop_requirements());
    let signature = cached_graph_key(&backend).post_processing;
    let pass_count = backend.frame_graph_pass_count();

    backend.ensure_frame_graph_in_sync(desktop_requirements());
    assert_eq!(cached_graph_key(&backend).post_processing, signature);
    assert_eq!(backend.frame_graph_pass_count(), pass_count);
}

/// Switching between mono and stereo multiview should flip the graph key in one place so the
/// runtime does not rely on implicit backend assumptions when VR starts or stops.
#[test]
fn multiview_change_updates_graph_key() {
    let mut backend = RenderBackend::new();
    backend.renderer_settings = Some(settings_handle(PostProcessingSettings::default()));

    backend.ensure_frame_graph_in_sync(desktop_requirements());
    let mono_key = cached_graph_key(&backend);
    backend.ensure_frame_graph_in_sync(xr_requirements());
    let stereo_key = cached_graph_key(&backend);

    assert!(!mono_key.multiview_stereo);
    assert!(stereo_key.multiview_stereo);
    assert_ne!(mono_key, stereo_key);
    assert!(mono_key.post_processing.motion_blur);
    assert!(
        !stereo_key.post_processing.motion_blur,
        "default VR/multiview graph should omit motion blur unless explicitly allowed"
    );
}

#[test]
fn multiview_motion_blur_is_opt_in() {
    let mut backend = RenderBackend::new();
    let mut settings = PostProcessingSettings::default();
    settings.motion_blur.allow_vr = true;
    backend.renderer_settings = Some(settings_handle(settings));

    backend.ensure_frame_graph_in_sync(xr_requirements());

    assert!(cached_graph_key(&backend).post_processing.motion_blur);
}

#[test]
fn post_processing_camera_without_motion_blur_omits_motion_blur_topology() {
    let mut backend = RenderBackend::new();
    backend.renderer_settings = Some(settings_handle(PostProcessingSettings::default()));

    backend.ensure_frame_graph_in_sync(camera_readback_without_motion_blur_requirements());

    let key = cached_graph_key(&backend);
    assert!(key.post_processing.active_count() > 0);
    assert!(
        !key.post_processing.motion_blur,
        "camera/readback graph should omit motion blur when every view disables it"
    );
}

#[test]
fn mixed_view_family_retains_motion_blur_when_any_view_can_use_it() {
    let mut backend = RenderBackend::new();
    backend.renderer_settings = Some(settings_handle(PostProcessingSettings::default()));
    let mut requirements = camera_readback_without_motion_blur_requirements();
    requirements.include_profile(RenderPathProfile::desktop_main(), false);

    backend.ensure_frame_graph_in_sync(requirements);

    assert!(cached_graph_key(&backend).post_processing.motion_blur);
}

/// Keeps upload arena slots alive when toggling back to a cached graph variant.
#[test]
fn cached_multiview_variant_switch_does_not_reset_upload_arena() {
    let mut backend = RenderBackend::new();
    backend.renderer_settings = Some(settings_handle(PostProcessingSettings::default()));

    let initial_generation = backend.upload_arena_generation_for_tests();
    backend.ensure_frame_graph_in_sync(desktop_requirements());
    let after_mono_build = backend.upload_arena_generation_for_tests();
    backend.ensure_frame_graph_in_sync(xr_requirements());
    let after_stereo_build = backend.upload_arena_generation_for_tests();
    backend.ensure_frame_graph_in_sync(desktop_requirements());
    let after_mono_cache_hit = backend.upload_arena_generation_for_tests();

    assert!(
        after_mono_build > initial_generation,
        "building the first graph variant should reset stale persistent upload slots"
    );
    assert!(
        after_stereo_build > after_mono_build,
        "building the stereo variant should reset stale persistent upload slots"
    );
    assert_eq!(
        after_mono_cache_hit, after_stereo_build,
        "switching to a cached mono variant must not reset upload staging"
    );
    assert!(
        !cached_graph_key(&backend).multiview_stereo,
        "final active graph should be the cached mono variant"
    );
}

#[test]
fn headless_profile_forces_empty_post_processing_signature() {
    let backend = RenderBackend::new();
    let settings = PostProcessingSettings {
        enabled: true,
        auto_exposure: crate::config::AutoExposureSettings {
            enabled: true,
            ..Default::default()
        },
        tonemap: TonemapSettings {
            mode: TonemapMode::AcesFitted,
        },
        ..Default::default()
    };

    let effective =
        backend.effective_post_processing_settings_for_graph(&settings, headless_requirements());

    assert!(
        !effective.enabled,
        "headless graph policy must not mutate individual effects; it should disable the master gate"
    );
    assert!(PostProcessChainSignature::from_settings(&effective).is_empty());
    assert!(settings.enabled, "caller settings must stay unchanged");
}

#[test]
fn scene_color_format_falls_back_when_requested_format_is_not_renderable() {
    let limits = limits_with_format_usage(
        wgpu::TextureFormat::Rg11b10Ufloat,
        wgpu::TextureUsages::TEXTURE_BINDING,
    );

    assert_eq!(
        effective_scene_color_format(wgpu::TextureFormat::Rg11b10Ufloat, &limits, false),
        wgpu::TextureFormat::Rgba16Float
    );
}

#[test]
fn scene_color_format_promotes_unsigned_when_signed_rgb_is_required() {
    let limits = limits_with_format_usage(
        wgpu::TextureFormat::Rg11b10Ufloat,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );

    assert_eq!(
        effective_scene_color_format(wgpu::TextureFormat::Rg11b10Ufloat, &limits, true),
        wgpu::TextureFormat::Rgba16Float
    );
    assert_eq!(
        effective_scene_color_format(wgpu::TextureFormat::Rg11b10Ufloat, &limits, false),
        wgpu::TextureFormat::Rg11b10Ufloat
    );
}
