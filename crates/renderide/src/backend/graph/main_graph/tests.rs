//! Tests for the parent module.

use wgpu::TextureFormat;

use super::*;
use crate::config::{
    BloomSettings, GtaoSettings, PostProcessingSettings, TonemapMode, TonemapSettings,
};
use crate::render_graph::error::GraphBuildError;
use crate::render_graph::post_process_chain::PostProcessChainSignature;
use crate::render_graph::resources::TransientArrayLayers;
use crate::render_graph::{GraphCache, GraphCacheEnsureResult};

fn smoke_key() -> GraphCacheKey {
    GraphCacheKey {
        surface_extent: (1280, 720),
        msaa_sample_count: 1,
        multiview_stereo: false,
        surface_format: TextureFormat::Bgra8UnormSrgb,
        scene_color_format: TextureFormat::Rgba16Float,
        post_processing: PostProcessChainSignature::default(),
    }
}

fn no_post() -> PostProcessingSettings {
    PostProcessingSettings {
        enabled: false,
        ..Default::default()
    }
}

fn aces_enabled_post() -> PostProcessingSettings {
    PostProcessingSettings {
        enabled: true,
        gtao: GtaoSettings {
            enabled: false,
            ..Default::default()
        },
        bloom: BloomSettings {
            enabled: false,
            ..Default::default()
        },
        auto_exposure: crate::config::AutoExposureSettings {
            enabled: false,
            ..Default::default()
        },
        tonemap: TonemapSettings {
            mode: TonemapMode::AcesFitted,
        },
    }
}

fn agx_enabled_post() -> PostProcessingSettings {
    PostProcessingSettings {
        enabled: true,
        gtao: GtaoSettings {
            enabled: false,
            ..Default::default()
        },
        bloom: BloomSettings {
            enabled: false,
            ..Default::default()
        },
        auto_exposure: crate::config::AutoExposureSettings {
            enabled: false,
            ..Default::default()
        },
        tonemap: TonemapSettings {
            mode: TonemapMode::AgX,
        },
    }
}

fn gtao_enabled_post() -> PostProcessingSettings {
    PostProcessingSettings {
        enabled: true,
        gtao: GtaoSettings {
            enabled: true,
            ..Default::default()
        },
        bloom: BloomSettings {
            enabled: false,
            ..Default::default()
        },
        auto_exposure: crate::config::AutoExposureSettings {
            enabled: false,
            ..Default::default()
        },
        tonemap: TonemapSettings {
            mode: TonemapMode::None,
        },
    }
}

#[test]
fn default_main_needs_surface_and_ten_passes() {
    let g = build_main_graph(smoke_key(), &no_post()).expect("default graph");
    assert!(g.needs_surface_acquire());
    assert_eq!(g.pass_count(), 10);
    assert_eq!(g.compile_stats.topo_levels, 10);
    assert_eq!(g.compile_stats.transient_texture_count, 4);
    assert!(
        !g.pass_info
            .iter()
            .any(|p| p.name.as_str() == "WorldMeshForwardPrepare")
    );
    let pass_names: Vec<&str> = g.pass_info.iter().map(|p| p.name.as_str()).collect();
    let depth_prepass_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardDepthPrepass")
        .expect("depth prepass");
    let opaque_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardOpaque")
        .expect("opaque pass");
    assert!(depth_prepass_pos < opaque_pos);
}

#[test]
fn msaa_main_graph_uses_transparent_sequence_for_grab_resolves() {
    let mut key = smoke_key();
    key.msaa_sample_count = 4;
    let g = build_main_graph(key, &no_post()).expect("MSAA graph");
    let pass_names: Vec<&str> = g.pass_info.iter().map(|p| p.name.as_str()).collect();
    let intersect_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardIntersect")
        .expect("intersect pass");
    let sequence_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardTransparentSequence")
        .expect("transparent sequence pass");
    let depth_resolve_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardDepthResolve")
        .expect("depth resolve pass");

    assert!(intersect_pos < sequence_pos);
    assert!(sequence_pos < depth_resolve_pos);
    assert!(!pass_names.contains(&"WorldMeshForwardColorResolvePreGrab"));
    assert!(!pass_names.contains(&"WorldMeshColorSnapshot"));
    assert!(!pass_names.contains(&"WorldMeshForwardTransparent"));
    assert!(!pass_names.contains(&"WorldMeshForwardColorResolveFinal"));
    assert_eq!(g.pass_count(), 10);
    assert_eq!(g.compile_stats.topo_levels, 10);
}

#[test]
fn enabling_aces_adds_a_pass_and_a_transient() {
    let g_off = build_main_graph(smoke_key(), &no_post()).expect("default graph");
    let mut key_on = smoke_key();
    key_on.post_processing = PostProcessChainSignature::from_settings(&aces_enabled_post());
    let g_on = build_main_graph(key_on, &aces_enabled_post()).expect("aces graph");
    assert_eq!(g_on.pass_count(), g_off.pass_count() + 1);
    assert!(g_on.needs_surface_acquire());
    assert!(
        g_on.compile_stats.transient_texture_count >= g_off.compile_stats.transient_texture_count
    );
}

#[test]
fn enabling_agx_adds_a_pass_and_a_transient() {
    let g_off = build_main_graph(smoke_key(), &no_post()).expect("default graph");
    let post = agx_enabled_post();
    let mut key_on = smoke_key();
    key_on.post_processing = PostProcessChainSignature::from_settings(&post);
    let g_on = build_main_graph(key_on, &post).expect("agx graph");
    let pass_names: Vec<&str> = g_on.pass_info.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(g_on.pass_count(), g_off.pass_count() + 1);
    assert!(pass_names.contains(&"AgxTonemap"));
    assert!(!pass_names.contains(&"AcesTonemap"));
    assert!(g_on.needs_surface_acquire());
    assert!(
        g_on.compile_stats.transient_texture_count >= g_off.compile_stats.transient_texture_count
    );
}

#[test]
fn mono_gtao_graph_declares_single_layer_view_depth_mips() {
    let post = gtao_enabled_post();
    let mut key = smoke_key();
    key.multiview_stereo = false;
    key.post_processing = PostProcessChainSignature::from_settings(&post);
    let graph = build_main_graph(key, &post).expect("mono gtao graph");

    assert!(
        graph
            .subresources
            .iter()
            .any(|desc| desc.label == "gtao_view_depth_mip4" && desc.array_layer_count == 1)
    );
    assert!(
        !graph
            .subresources
            .iter()
            .any(|desc| desc.label.starts_with("gtao_view_depth_")
                && (desc.label.ends_with("_l0") || desc.label.ends_with("_l1")))
    );
    let view_depth = graph
        .transient_textures
        .iter()
        .find(|texture| texture.desc.label == "gtao_view_depth")
        .expect("gtao view depth transient");
    assert_eq!(view_depth.desc.array_layers, TransientArrayLayers::Frame);
}

#[test]
fn stereo_gtao_graph_declares_layered_view_depth_mips() {
    let post = gtao_enabled_post();
    let mut key = smoke_key();
    key.multiview_stereo = true;
    key.post_processing = PostProcessChainSignature::from_settings(&post);
    let graph = build_main_graph(key, &post).expect("stereo gtao graph");

    assert!(
        graph
            .subresources
            .iter()
            .any(|desc| desc.label == "gtao_view_depth_mip4" && desc.array_layer_count == 2)
    );
    let view_depth = graph
        .transient_textures
        .iter()
        .find(|texture| texture.desc.label == "gtao_view_depth")
        .expect("gtao view depth transient");
    assert_eq!(view_depth.desc.array_layers, TransientArrayLayers::Fixed(2));
}

#[test]
fn full_post_processing_orders_exposure_before_bloom_and_tonemap_last() {
    let mut post = PostProcessingSettings {
        tonemap: TonemapSettings {
            mode: TonemapMode::AgX,
        },
        ..Default::default()
    };
    post.auto_exposure.enabled = true;
    let mut key = smoke_key();
    key.post_processing = PostProcessChainSignature::from_settings(&post);
    let graph = build_main_graph(key, &post).expect("full post-processing graph");
    let pass_names: Vec<&str> = graph.pass_info.iter().map(|p| p.name.as_str()).collect();
    let auto_compute_pos = pass_names
        .iter()
        .position(|name| *name == "AutoExposureCompute")
        .expect("auto-exposure compute pass");
    let auto_apply_pos = pass_names
        .iter()
        .position(|name| *name == "AutoExposureApply")
        .expect("auto-exposure apply pass");
    let bloom_downsample_pos = pass_names
        .iter()
        .position(|name| *name == "BloomDownsampleFirst")
        .expect("first bloom downsample pass");
    let bloom_composite_pos = pass_names
        .iter()
        .position(|name| *name == "BloomComposite")
        .expect("bloom composite pass");
    let agx_tonemap_pos = pass_names
        .iter()
        .position(|name| *name == "AgxTonemap")
        .expect("AgX tonemap pass");

    assert!(auto_compute_pos < auto_apply_pos);
    assert!(auto_apply_pos < bloom_downsample_pos);
    assert!(bloom_downsample_pos < bloom_composite_pos);
    assert!(bloom_composite_pos < agx_tonemap_pos);
}

#[test]
fn agx_post_processing_orders_exposure_before_bloom_and_tonemap_last() {
    let mut post = PostProcessingSettings {
        tonemap: TonemapSettings {
            mode: TonemapMode::AgX,
        },
        ..Default::default()
    };
    post.gtao.enabled = false;
    post.auto_exposure.enabled = true;
    let mut key = smoke_key();
    key.post_processing = PostProcessChainSignature::from_settings(&post);
    let graph = build_main_graph(key, &post).expect("agx post-processing graph");
    let pass_names: Vec<&str> = graph.pass_info.iter().map(|p| p.name.as_str()).collect();
    let auto_apply_pos = pass_names
        .iter()
        .position(|name| *name == "AutoExposureApply")
        .expect("auto-exposure apply pass");
    let bloom_composite_pos = pass_names
        .iter()
        .position(|name| *name == "BloomComposite")
        .expect("bloom composite pass");
    let agx_tonemap_pos = pass_names
        .iter()
        .position(|name| *name == "AgxTonemap")
        .expect("AgX tonemap pass");

    assert!(auto_apply_pos < bloom_composite_pos);
    assert!(bloom_composite_pos < agx_tonemap_pos);
    assert!(!pass_names.contains(&"AcesTonemap"));
}

#[test]
fn enabling_gtao_adds_normal_prepass_before_gtao_main() {
    let post = gtao_enabled_post();
    let mut key = smoke_key();
    key.post_processing = PostProcessChainSignature::from_settings(&post);
    let g = build_main_graph(key, &post).expect("gtao graph");
    let pass_names: Vec<&str> = g.pass_info.iter().map(|p| p.name.as_str()).collect();
    let normal_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardNormals")
        .expect("GTAO normal prepass");
    let depth_prepass_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardDepthPrepass")
        .expect("depth prepass");
    let opaque_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshForwardOpaque")
        .expect("opaque pass");
    let depth_snapshot_pos = pass_names
        .iter()
        .position(|name| *name == "WorldMeshDepthSnapshot")
        .expect("depth snapshot pass");
    let gtao_main_pos = pass_names
        .iter()
        .position(|name| *name == "GtaoMain")
        .expect("GTAO main pass");

    assert!(depth_prepass_pos < opaque_pos);
    assert!(opaque_pos < normal_pos);
    assert!(normal_pos < depth_snapshot_pos);
    assert!(depth_snapshot_pos < gtao_main_pos);
    assert!(
        g.transient_textures
            .iter()
            .any(|t| t.desc.label == "gtao_view_normals")
    );
    assert!(
        g.transient_textures
            .iter()
            .any(|t| t.desc.label == "gtao_view_normals_msaa")
    );
}

#[test]
fn graph_cache_reuses_when_key_unchanged() {
    let key = smoke_key();
    let post = no_post();
    let mut cache = GraphCache::default();
    assert_eq!(
        cache
            .ensure(key, || build_main_graph(key, &post))
            .expect("first build"),
        GraphCacheEnsureResult::Built
    );
    let n = cache.pass_count();
    let mut build_called = false;
    assert_eq!(
        cache
            .ensure(key, || {
                build_called = true;
                build_main_graph(key, &post)
            })
            .expect("second ensure"),
        GraphCacheEnsureResult::Hit
    );
    assert!(!build_called);
    assert_eq!(cache.pass_count(), n);
}

/// Reuses the cached mono graph after switching to and from a stereo graph variant.
#[test]
fn graph_cache_reuses_previous_variant_after_multiview_switch() {
    let mono_key = smoke_key();
    let mut stereo_key = smoke_key();
    stereo_key.multiview_stereo = true;
    let post = no_post();
    let mut cache = GraphCache::default();

    assert_eq!(
        cache
            .ensure(mono_key, || build_main_graph(mono_key, &post))
            .expect("mono build"),
        GraphCacheEnsureResult::Built
    );
    let mono_pass_count = cache.pass_count();
    assert_eq!(
        cache
            .ensure(stereo_key, || build_main_graph(stereo_key, &post))
            .expect("stereo build"),
        GraphCacheEnsureResult::Built
    );
    let mut build_called = false;
    assert_eq!(
        cache
            .ensure(mono_key, || {
                build_called = true;
                build_main_graph(mono_key, &post)
            })
            .expect("mono cache hit after stereo"),
        GraphCacheEnsureResult::Hit
    );

    assert!(!build_called);
    assert_eq!(cache.last_key(), Some(mono_key));
    assert_eq!(cache.pass_count(), mono_pass_count);
    assert_eq!(cache.variant_count_for_tests(), 2);
}

/// Keeps previously built variants available when building a new graph variant fails.
#[test]
fn graph_cache_build_failure_preserves_cached_variants() {
    let cached_key = smoke_key();
    let mut failing_key = smoke_key();
    failing_key.scene_color_format = TextureFormat::Rg11b10Ufloat;
    let post = no_post();
    let mut cache = GraphCache::default();

    assert_eq!(
        cache
            .ensure(cached_key, || build_main_graph(cached_key, &post))
            .expect("cached build"),
        GraphCacheEnsureResult::Built
    );
    let cached_pass_count = cache.pass_count();
    assert!(matches!(
        cache.ensure(failing_key, || Err(GraphBuildError::CycleDetected)),
        Err(GraphBuildError::CycleDetected)
    ));
    assert_eq!(cache.last_key(), None);
    assert_eq!(cache.pass_count(), 0);

    let mut build_called = false;
    assert_eq!(
        cache
            .ensure(cached_key, || {
                build_called = true;
                build_main_graph(cached_key, &post)
            })
            .expect("cached key after failed build"),
        GraphCacheEnsureResult::Hit
    );

    assert!(!build_called);
    assert_eq!(cache.pass_count(), cached_pass_count);
}

/// Evicts the least-recently-used inactive variant while preserving the active graph.
#[test]
fn graph_cache_evicts_lru_without_dropping_active_variant() {
    let post = no_post();
    let mut cache = GraphCache::default();
    let keys: [GraphCacheKey; 5] = std::array::from_fn(|index| {
        let mut key = smoke_key();
        key.surface_extent = (1280 + index as u32, 720);
        key
    });

    for key in keys {
        assert_eq!(
            cache
                .ensure(key, || build_main_graph(key, &post))
                .expect("variant build"),
            GraphCacheEnsureResult::Built
        );
        assert_eq!(cache.last_key(), Some(key));
        assert!(cache.contains_key(key));
    }

    assert_eq!(cache.variant_count_for_tests(), 4);
    assert_eq!(cache.last_key(), Some(keys[4]));
    assert!(cache.contains_key(keys[4]));
    assert!(!cache.contains_key(keys[0]));
}

#[test]
fn graph_cache_rebuilds_when_scene_color_format_changes() {
    let mut a = smoke_key();
    a.scene_color_format = TextureFormat::Rgba16Float;
    let mut b = smoke_key();
    b.scene_color_format = TextureFormat::Rg11b10Ufloat;
    let post = no_post();
    let mut cache = GraphCache::default();
    cache
        .ensure(a, || build_main_graph(a, &post))
        .expect("first build");
    let mut build_called = false;
    assert_eq!(
        cache
            .ensure(b, || {
                build_called = true;
                build_main_graph(b, &post)
            })
            .expect("second ensure"),
        GraphCacheEnsureResult::Built
    );
    assert!(build_called);
}

/// MSAA depth transients must follow [`TransientArrayLayers::Frame`] so stereo execution matches
/// HDR color even when [`GraphCacheKey::multiview_stereo`] was `false` at compile time.
#[test]
fn forward_msaa_depth_uses_frame_array_layers_with_mono_cache_key() {
    let mut key = smoke_key();
    key.multiview_stereo = false;
    let g = build_main_graph(key, &no_post()).expect("default graph");
    let forward_depth = g
        .transient_textures
        .iter()
        .find(|t| t.desc.label == "forward_msaa_depth")
        .expect("forward_msaa_depth transient");
    assert_eq!(forward_depth.desc.array_layers, TransientArrayLayers::Frame);
    let r32 = g
        .transient_textures
        .iter()
        .find(|t| t.desc.label == "forward_msaa_depth_r32")
        .expect("forward_msaa_depth_r32 transient");
    assert_eq!(r32.desc.array_layers, TransientArrayLayers::Frame);
}

#[test]
fn graph_cache_rebuilds_when_post_processing_signature_changes() {
    let mut a = smoke_key();
    a.post_processing = PostProcessChainSignature::default();
    let mut b = smoke_key();
    b.post_processing = PostProcessChainSignature::from_settings(&aces_enabled_post());
    let mut cache = GraphCache::default();
    cache
        .ensure(a, || build_main_graph(a, &no_post()))
        .expect("first build");
    let mut build_called = false;
    cache
        .ensure(b, || {
            build_called = true;
            build_main_graph(b, &aces_enabled_post())
        })
        .expect("second ensure");
    assert!(build_called);
}
