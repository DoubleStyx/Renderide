//! Ground-Truth Ambient Occlusion (Jimenez et al. 2016) opaque-only screen-space effect with
//! depth-aware bilateral denoise.
//!
//! Registers a GTAO chain on the main graph after opaque/cutout world-mesh rendering and before
//! transparent/intersection draws:
//!
//! 1. [`depth_prefilter_pass::GtaoDepthPrefilterPass`] -- converts raw depth to view-space
//!    depth and builds the five-level depth chain sampled by the horizon search.
//! 2. [`main_pass::GtaoMainPass`] -- produces the AO term (scaled by
//!    `1 / OCCLUSION_TERM_SCALE` to leave denoise headroom) and packed depth-edge
//!    weights from the prefiltered depth chain plus the forward view-normal prepass.
//! 3. [`denoise_pass::GtaoDenoisePass`] -- 3x3 edge-preserving bilateral filter. Registered
//!    `denoise_passes - 1` times when denoise asks for intermediate ping-pong iterations.
//! 4. [`apply_pass::GtaoOpaqueCompositePass`] -- final denoise iteration that outputs recovered
//!    visibility. Multiplicative blending modulates the existing opaque HDR scene color in place.
//!    Always registered. The shader short-circuits the kernel when `denoise_blur_beta <= 0`, so
//!    `denoise_passes == 0` collapses to a "modulate by raw AO" path without re-binding a
//!    different pipeline.
//!
//! Multiview (stereo) is handled by per-stage pipeline variants. Raster stages use
//! `multiview_mask_override` with `#ifdef MULTIVIEW` selecting `@builtin(view_index)`, while
//! the compute depth prefilter dispatches one workgroup layer per eye against array subresources.

mod apply_pass;
mod denoise_pass;
mod depth_prefilter_pass;
mod main_pass;
mod pipeline;

use std::sync::LazyLock;

use apply_pass::{GtaoOpaqueCompositePass, GtaoOpaqueCompositeResources};
use denoise_pass::{GtaoDenoisePass, GtaoDenoiseResources};
use depth_prefilter_pass::{GtaoDepthPrefilterPass, GtaoDepthPrefilterResources};
use main_pass::{GtaoMainPass, GtaoMainResources};
use pipeline::{
    AO_TERM_FORMAT, EDGES_FORMAT, GtaoPipelines, VIEW_DEPTH_FORMAT, VIEW_DEPTH_MIP_COUNT,
};

use crate::config::GtaoSettings;
use crate::graph_inputs::GraphPassFrameView;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::{
    ImportedBufferHandle, ImportedTextureHandle, SubresourceHandle, TextureHandle,
    TransientArrayLayers, TransientExtent, TransientSampleCount, TransientSubresourceDesc,
    TransientTextureDesc, TransientTextureFormat,
};

const GTAO_VIEW_DEPTH_MIP_LABELS: [&str; VIEW_DEPTH_MIP_COUNT as usize] = [
    "gtao_view_depth_mip0",
    "gtao_view_depth_mip1",
    "gtao_view_depth_mip2",
    "gtao_view_depth_mip3",
    "gtao_view_depth_mip4",
];

/// Graph resources consumed and produced by the opaque GTAO subchain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GtaoGraphResources {
    /// Imported forward depth texture sampled by the view-depth prefilter.
    pub(crate) depth: ImportedTextureHandle,
    /// Smooth view-space normals produced from the visible opaque/cutout surface.
    pub(crate) view_normals: TextureHandle,
    /// Imported frame-uniform buffer used by depth reconstruction.
    pub(crate) frame_uniforms: ImportedBufferHandle,
    /// Single-sample HDR scene color target.
    pub(crate) scene_color_hdr: TextureHandle,
    /// Multisampled HDR scene color target used when MSAA is active.
    pub(crate) scene_color_hdr_msaa: Option<TextureHandle>,
    /// Whether this graph is compiled for OpenXR multiview stereo.
    pub(crate) multiview_stereo: bool,
}

/// First and last pass ids registered by the opaque GTAO subchain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GtaoPassRange {
    /// First pass in the subchain.
    pub(crate) first: PassId,
    /// Last pass in the subchain.
    pub(crate) last: PassId,
}

/// Effect descriptor that contributes the opaque GTAO pass chain to the main graph.
pub(crate) struct GtaoEffect {
    /// Snapshot of the GTAO settings used when building the chain for this frame. Live edits
    /// after graph build flow in via
    /// [`crate::passes::post_processing::settings_slots::GtaoSettingsSlot`] for non-topology
    /// fields; topology fields (`enabled`, `denoise_passes`, `resolution_divisor`) trigger a graph rebuild via
    /// [`crate::render_graph::post_process_chain::PostProcessChainSignature`].
    pub(crate) settings: GtaoSettings,
    /// Graph resources used by this effect.
    pub(crate) resources: GtaoGraphResources,
}

/// Returns whether this view has visible opaque world-mesh color for GTAO to sample and composite.
pub(super) fn gtao_view_recording_needed(
    blackboard: &Blackboard,
    view: &GraphPassFrameView<'_>,
) -> bool {
    super::view_post_processing_enabled(view)
        && blackboard
            .get::<crate::passes::WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| prepared.opaque_recorded)
}

impl GtaoEffect {
    /// Registers the opaque-only GTAO subchain and returns its first and last pass ids.
    pub(crate) fn register(&self, builder: &mut GraphBuilder) -> GtaoPassRange {
        let pipelines = gtao_pipelines();
        let denoise_passes = self.settings.effective_denoise_passes();
        let resolution_divisor = self.settings.effective_resolution_divisor();

        let view_depth_mips = create_view_depth_textures(
            builder,
            self.resources.multiview_stereo,
            resolution_divisor,
        );
        let prefilter = add_view_depth_prefilter(builder, view_depth_mips, self, pipelines);

        let ao_term_a =
            builder.create_texture(ao_buffer_desc("gtao_ao_term_a", resolution_divisor));
        let edges = builder.create_texture(ao_buffer_desc_format(
            "gtao_edges",
            TransientTextureFormat::Fixed(EDGES_FORMAT),
            resolution_divisor,
        ));
        let ao_term_b = (denoise_passes >= 2)
            .then(|| builder.create_texture(ao_buffer_desc("gtao_ao_term_b", resolution_divisor)));

        let main = builder.add_raster_pass(Box::new(GtaoMainPass::new(
            GtaoMainResources {
                view_depth_mips: view_depth_mips.textures,
                view_depth_views: view_depth_mips.storage_views,
                view_normals: self.resources.view_normals,
                frame_uniforms: self.resources.frame_uniforms,
                ao_term: ao_term_a,
                edges,
            },
            self.settings,
            pipelines,
        )));
        builder.add_edge(prefilter, main);

        let mut last = main;
        let mut ao_for_apply = ao_term_a;
        if let Some(ao_term_b) = ao_term_b {
            let mut ao_in = ao_term_a;
            for iteration in 0..denoise_passes.saturating_sub(1) {
                let ao_out = if iteration % 2 == 0 {
                    ao_term_b
                } else {
                    ao_term_a
                };
                let denoise = builder.add_raster_pass(Box::new(GtaoDenoisePass::new(
                    GtaoDenoiseResources {
                        ao_in,
                        edges,
                        ao_out,
                    },
                    self.settings,
                    pipelines,
                )));
                builder.add_edge(last, denoise);
                last = denoise;
                ao_in = ao_out;
                ao_for_apply = ao_out;
            }
        }

        let (target_color, target_is_msaa) =
            if let Some(scene_color_hdr_msaa) = self.resources.scene_color_hdr_msaa {
                (scene_color_hdr_msaa, true)
            } else {
                (self.resources.scene_color_hdr, false)
            };
        let apply = builder.add_raster_pass(Box::new(GtaoOpaqueCompositePass::new(
            GtaoOpaqueCompositeResources {
                target_color,
                target_is_msaa,
                ao_in: ao_for_apply,
                edges,
            },
            self.settings,
            pipelines,
        )));
        builder.add_edge(last, apply);

        GtaoPassRange {
            first: prefilter,
            last: apply,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ViewDepthTextures {
    /// View-depth textures, one per GTAO depth level.
    textures: [TextureHandle; VIEW_DEPTH_MIP_COUNT as usize],
    /// Writable views for the view-depth textures.
    storage_views: [SubresourceHandle; VIEW_DEPTH_MIP_COUNT as usize],
}

fn create_view_depth_textures(
    builder: &mut GraphBuilder,
    multiview_stereo: bool,
    resolution_divisor: u32,
) -> ViewDepthTextures {
    let array_layer_count = stereo_array_layer_count(multiview_stereo);
    let pairs = std::array::from_fn(|mip| {
        let texture = builder.create_texture(view_depth_desc(
            GTAO_VIEW_DEPTH_MIP_LABELS[mip],
            multiview_stereo,
            resolution_divisor,
            mip as u32,
        ));
        let storage_view = builder.create_subresource(TransientSubresourceDesc {
            parent: texture,
            label: GTAO_VIEW_DEPTH_MIP_LABELS[mip],
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count,
        });
        (texture, storage_view)
    });
    ViewDepthTextures {
        textures: pairs.map(|(texture, _)| texture),
        storage_views: pairs.map(|(_, storage_view)| storage_view),
    }
}

fn add_view_depth_prefilter(
    builder: &mut GraphBuilder,
    view_depth_mips: ViewDepthTextures,
    effect: &GtaoEffect,
    pipelines: &'static GtaoPipelines,
) -> PassId {
    let resources = GtaoDepthPrefilterResources {
        depth: effect.resources.depth,
        frame_uniforms: effect.resources.frame_uniforms,
        output_textures: view_depth_mips.textures,
        output_mips: view_depth_mips.storage_views,
    };
    builder.add_compute_pass(Box::new(GtaoDepthPrefilterPass::new(
        resources,
        effect.settings,
        pipelines,
        effect.resources.multiview_stereo,
    )))
}

/// Process-wide pipeline + UBO singleton shared across every GTAO chain rebuild.
fn gtao_pipelines() -> &'static GtaoPipelines {
    static CACHE: LazyLock<GtaoPipelines> = LazyLock::new(GtaoPipelines::default);
    &CACHE
}

fn stereo_array_layer_count(multiview_stereo: bool) -> u32 {
    if multiview_stereo { 2 } else { 1 }
}

/// Returns `true` when the active device supports GTAO's transient texture formats and usages.
pub(crate) fn gpu_supports_gtao(limits: &crate::gpu::GpuLimits) -> bool {
    let sampled_render_target =
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    limits.texture_usage_supported(
        VIEW_DEPTH_FORMAT,
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    ) && limits.texture_usage_supported(AO_TERM_FORMAT, sampled_render_target)
        && limits.texture_usage_supported(EDGES_FORMAT, sampled_render_target)
        && limits.texture_usage_supported(
            crate::passes::GTAO_VIEW_NORMAL_FORMAT,
            sampled_render_target,
        )
}

/// Transient texture descriptor for the AO term ping-pong buffers (`R8Unorm`, frame array
/// layers).
fn ao_buffer_desc(label: &'static str, resolution_divisor: u32) -> TransientTextureDesc {
    ao_buffer_desc_format(
        label,
        TransientTextureFormat::Fixed(AO_TERM_FORMAT),
        resolution_divisor,
    )
}

/// Transient texture descriptor for an `R8Unorm` GTAO buffer with a custom format slot.
fn ao_buffer_desc_format(
    label: &'static str,
    format: TransientTextureFormat,
    resolution_divisor: u32,
) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format,
        extent: TransientExtent::BackbufferDivisor {
            divisor: resolution_divisor.max(1),
        },
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}

fn view_depth_desc(
    label: &'static str,
    multiview_stereo: bool,
    resolution_divisor: u32,
    mip: u32,
) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::Fixed(VIEW_DEPTH_FORMAT),
        extent: TransientExtent::BackbufferDivisorMip {
            divisor: resolution_divisor.max(1),
            mip,
        },
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: if multiview_stereo {
            TransientArrayLayers::Fixed(2)
        } else {
            TransientArrayLayers::Frame
        },
        base_usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_effect(multiview_stereo: bool) -> GtaoEffect {
        GtaoEffect {
            settings: GtaoSettings::default(),
            resources: GtaoGraphResources {
                depth: ImportedTextureHandle(0),
                view_normals: TextureHandle(0),
                frame_uniforms: ImportedBufferHandle(0),
                scene_color_hdr: TextureHandle(1),
                scene_color_hdr_msaa: None,
                multiview_stereo,
            },
        }
    }

    /// The WGSL `GtaoParams` struct is 64 bytes (16 x 4); changes here require updating
    /// `gtao_main.wgsl`, `gtao_denoise.wgsl`, and `gtao_apply.wgsl` simultaneously.
    #[test]
    fn gtao_params_gpu_size_is_64_bytes() {
        assert_eq!(size_of::<pipeline::GtaoParamsGpu>(), 64);
    }

    #[test]
    fn gtao_params_clamp_runtime_view_depth_mips() {
        assert_eq!(
            pipeline::GtaoParamsGpu::from_settings(GtaoSettings::default(), 0.0, false)
                .with_view_depth_mip_count(0)
                .view_depth_mip_count,
            1
        );
        assert_eq!(
            pipeline::GtaoParamsGpu::from_settings(GtaoSettings::default(), 0.0, false)
                .with_view_depth_mip_count(3)
                .view_depth_mip_count,
            3
        );
        assert_eq!(
            pipeline::GtaoParamsGpu::from_settings(GtaoSettings::default(), 0.0, false)
                .with_view_depth_mip_count(99)
                .view_depth_mip_count,
            VIEW_DEPTH_MIP_COUNT
        );
    }

    #[test]
    fn gtao_quality_levels_match_expected_presets() {
        /// Builds a GTAO settings value with explicit overrides disabled.
        fn preset_settings(quality_level: u32) -> GtaoSettings {
            GtaoSettings {
                quality_level,
                slice_count_override: 0,
                step_count: 0,
                ..Default::default()
            }
        }

        assert_eq!(preset_settings(0).effective_sample_counts(), (1, 2));
        assert_eq!(preset_settings(1).effective_sample_counts(), (2, 2));
        assert_eq!(preset_settings(2).effective_sample_counts(), (3, 3));
        assert_eq!(preset_settings(3).effective_sample_counts(), (9, 3));
        assert_eq!(preset_settings(4).effective_sample_counts(), (12, 4));
        assert_eq!(
            GtaoSettings::default().effective_sample_counts(),
            (crate::config::GTAO_MAX_SLICE_COUNT, 2)
        );
    }

    #[test]
    fn gtao_quality_overrides_clamp_to_supported_range() {
        let settings = GtaoSettings {
            slice_count_override: 99,
            step_count: 99,
            resolution_divisor: 99,
            denoise_passes: 99,
            ..Default::default()
        };

        assert_eq!(settings.effective_sample_counts(), (16, 8));
        assert_eq!(settings.effective_resolution_divisor(), 4);
        assert_eq!(settings.effective_denoise_passes(), 6);
    }

    /// Verifies the bundle of caches constructs (which exercises the manual `Default`
    /// implementations in `pipeline.rs` that pick bounded bind-group caches).
    #[test]
    fn pipeline_caches_default_construct() {
        let _ = GtaoPipelines::default();
    }

    #[test]
    fn mono_view_depth_declares_only_layer_zero() {
        let mut builder = GraphBuilder::new();
        let _mips = create_view_depth_textures(&mut builder, false, 1);

        assert_eq!(builder.textures.len(), VIEW_DEPTH_MIP_COUNT as usize);
        assert!(
            builder
                .textures
                .iter()
                .all(|desc| desc.array_layers == TransientArrayLayers::Frame)
        );
        assert!(
            builder
                .textures
                .iter()
                .enumerate()
                .all(|(mip, desc)| desc.extent
                    == (TransientExtent::BackbufferDivisorMip {
                        divisor: 1,
                        mip: mip as u32,
                    }))
        );
    }

    #[test]
    fn stereo_view_depth_declares_shared_layered_mips() {
        let mut builder = GraphBuilder::new();
        let _mips = create_view_depth_textures(&mut builder, true, 2);

        assert_eq!(builder.textures.len(), VIEW_DEPTH_MIP_COUNT as usize);
        assert!(
            builder
                .textures
                .iter()
                .all(|desc| desc.array_layers == TransientArrayLayers::Fixed(2))
        );
        assert!(builder.textures.iter().all(|desc| matches!(
            desc.extent,
            TransientExtent::BackbufferDivisorMip { divisor: 2, .. }
        )));
    }

    #[test]
    fn stereo_depth_prefilter_registers_one_combined_pass() {
        let mut builder = GraphBuilder::new();
        let mips = create_view_depth_textures(&mut builder, true, 1);
        let effect = test_effect(true);
        let before = builder.passes.len();

        let first = add_view_depth_prefilter(&mut builder, mips, &effect, gtao_pipelines());

        assert_eq!(builder.passes.len() - before, 1);
        assert_eq!(first.0, before);
    }
}
