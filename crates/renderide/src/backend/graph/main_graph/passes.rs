//! Pass registration for the main render graph.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;

use super::gtao::{GtaoNode, add_gtao_if_active};
use super::handles::MainGraphHandles;

/// Pass ids registered by [`register_main_graph_passes`]; consumed by edge wiring.
pub(super) struct MainGraphPassIds {
    pub(super) light_cookies: PassId,
    pub(super) deform: PassId,
    pub(super) shadows: PassId,
    pub(super) clustered: PassId,
    pub(super) depth_prepass: PassId,
    pub(super) forward_opaque: PassId,
    pub(super) gtao: Option<GtaoNode>,
    pub(super) depth_snapshot: PassId,
    pub(super) forward_intersect: PassId,
    pub(super) occlusion_depth_resolve: Option<PassId>,
    pub(super) hiz: PassId,
    pub(super) forward_transparent_sequence: PassId,
    pub(super) depth_resolve: Option<PassId>,
}

fn main_forward_resources(h: &MainGraphHandles) -> crate::passes::WorldMeshForwardGraphResources {
    crate::passes::WorldMeshForwardGraphResources {
        scene_color_hdr: h.scene_color_hdr,
        depth: h.depth,
        msaa: h.msaa.map(|msaa| crate::passes::ForwardMsaaResources {
            scene_color_hdr: msaa.scene_color_hdr,
            depth: msaa.forward_depth,
            depth_r32: msaa.forward_depth_r32,
        }),
        cluster_light_counts: h.cluster_light_counts,
        cluster_light_indices: h.cluster_light_indices,
        lights: h.lights,
        per_draw_slab: h.per_draw_slab,
        frame_uniforms: h.frame_uniforms,
    }
}

fn main_depth_prepass_resources(
    h: &MainGraphHandles,
) -> crate::passes::WorldMeshForwardDepthPrepassGraphResources {
    crate::passes::WorldMeshForwardDepthPrepassGraphResources {
        depth: h.depth,
        msaa_depth: h.msaa.map(|msaa| msaa.forward_depth),
        per_draw_slab: h.per_draw_slab,
    }
}

pub(super) fn main_desktop_overlay_resources(
    h: &MainGraphHandles,
) -> crate::passes::WorldMeshDesktopOverlayGraphResources {
    crate::passes::WorldMeshDesktopOverlayGraphResources {
        frame_color: h.color,
        depth: h.depth,
        cluster_light_counts: h.cluster_light_counts,
        cluster_light_indices: h.cluster_light_indices,
        lights: h.lights,
        per_draw_slab: h.per_draw_slab,
        frame_uniforms: h.frame_uniforms,
    }
}

fn add_world_mesh_depth_prepass(builder: &mut GraphBuilder, h: &MainGraphHandles) -> PassId {
    builder.add_raster_pass(Box::new(crate::passes::WorldMeshForwardDepthPrepass::new(
        main_depth_prepass_resources(h),
    )))
}

/// Registers every pre-post-processing pass for the main render graph and returns their ids.
///
/// Order matches execution: mesh deform compute, clustered lights, world-mesh depth prepass and
/// forward opaque raster, optional opaque-only GTAO subchain, depth snapshot, forward intersect,
/// pre-transparent Hi-Z build, transparent sequence, and final MSAA depth resolve. Edge wiring is performed
/// separately by [`super::edges::add_main_graph_edges`].
pub(super) fn register_main_graph_passes(
    builder: &mut GraphBuilder,
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    msaa_sample_count: u8,
    multiview_stereo: bool,
) -> MainGraphPassIds {
    let msaa_enabled = msaa_sample_count > 1;
    let light_cookies = builder.add_encoder_pass(Box::new(
        crate::backend::frame_gpu::LightCookieAtlasPass::new(),
    ));
    let deform = builder.add_compute_pass(Box::new(crate::passes::MeshDeformPass::new()));
    let shadows =
        builder.add_encoder_pass(Box::new(crate::backend::frame_gpu::ShadowAtlasPass::new()));
    let clustered = builder.add_compute_pass(Box::new(crate::passes::ClusteredLightPass::new(
        crate::passes::ClusteredLightGraphResources {
            lights: h.lights,
            cluster_light_counts: h.cluster_light_counts,
            cluster_light_indices: h.cluster_light_indices,
            params: h.cluster_params,
        },
    )));
    let forward_resources = main_forward_resources(h);
    let depth_prepass = add_world_mesh_depth_prepass(builder, h);
    let forward_opaque = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardOpaquePass::new(forward_resources),
    ));
    let gtao = add_gtao_if_active(
        builder,
        forward_resources,
        post_processing_settings,
        multiview_stereo,
    );
    let depth_snapshot = builder.add_encoder_pass(Box::new(
        crate::passes::WorldMeshDepthSnapshotPass::new(forward_resources),
    ));
    let forward_intersect = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardIntersectPass::new(forward_resources),
    ));
    let occlusion_depth_resolve = if msaa_enabled && forward_resources.msaa_enabled() {
        Some(builder.add_encoder_pass(Box::new(
            crate::passes::WorldMeshForwardOcclusionDepthResolvePass::new(forward_resources),
        )))
    } else {
        builder.note_skipped_pass();
        None
    };
    let hiz = builder.add_compute_pass(Box::new(crate::passes::HiZBuildPass::new(
        crate::passes::HiZBuildGraphResources {
            depth: h.depth,
            hi_z_current: h.hi_z_current,
        },
    )));
    let forward_transparent_sequence = builder.add_encoder_pass(Box::new(
        crate::passes::WorldMeshForwardTransparentSequencePass::new(forward_resources),
    ));
    let depth_resolve = if msaa_enabled && forward_resources.msaa_enabled() {
        Some(builder.add_encoder_pass(Box::new(
            crate::passes::WorldMeshForwardDepthResolvePass::new(forward_resources),
        )))
    } else {
        builder.note_skipped_pass();
        None
    };
    MainGraphPassIds {
        light_cookies,
        deform,
        shadows,
        clustered,
        depth_prepass,
        forward_opaque,
        gtao,
        depth_snapshot,
        forward_intersect,
        occlusion_depth_resolve,
        hiz,
        forward_transparent_sequence,
        depth_resolve,
    }
}
