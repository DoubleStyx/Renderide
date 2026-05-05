//! Ground-Truth Ambient Occlusion (Jimenez et al. 2016) post-processing effect with
//! XeGTAO-style depth-aware bilateral denoise.
//!
//! Registers an XeGTAO-style chain on the post-processing graph builder:
//!
//! 1. [`depth_prefilter_pass::GtaoDepthPrefilterPass`] -- converts raw depth to view-space
//!    depth and builds the five-mip depth chain sampled by the horizon search.
//! 2. [`main_pass::GtaoMainPass`] -- produces the AO term (scaled by
//!    `1 / OCCLUSION_TERM_SCALE` per XeGTAO's headroom convention) and packed depth-edge
//!    weights from the prefiltered depth chain plus the forward view-normal prepass. The HDR
//!    scene-color input is *not* read here; modulation is deferred to the apply stage so the
//!    bilateral denoiser can act on the AO term first.
//! 3. [`denoise_pass::GtaoDenoisePass`] -- XeGTAO 3x3 edge-preserving bilateral filter.
//!    Registered once when [`crate::config::GtaoSettings::denoise_passes`] is `>= 2`, and
//!    twice when it is `>= 3`.
//! 4. [`apply_pass::GtaoApplyPass`] -- final denoise iteration that multiplies the AO term by
//!    `OCCLUSION_TERM_SCALE` to recover the true visibility, then modulates HDR scene color
//!    and writes the chain's HDR output. Always registered. The shader short-circuits the
//!    kernel when `denoise_blur_beta <= 0`, so `denoise_passes == 0` collapses to a
//!    "modulate by raw AO" path without re-binding a different pipeline.
//!
//! Multiview (stereo) is handled by per-stage pipeline variants (mono / multiview-stereo)
//! picked via a `multiview_mask_override` of `NonZeroU32::new(3)` in stereo, with
//! `#ifdef MULTIVIEW` in each shader selecting `@builtin(view_index)` and the array depth
//! sample path.

mod apply_pass;
mod denoise_pass;
mod depth_prefilter_pass;
mod main_pass;
mod pipeline;

use std::sync::LazyLock;

use apply_pass::{GtaoApplyPass, GtaoApplyResources};
use denoise_pass::{GtaoDenoisePass, GtaoDenoiseResources};
use depth_prefilter_pass::{GtaoDepthPrefilterPass, GtaoDepthPrefilterResources};
use main_pass::{GtaoMainPass, GtaoMainResources};
use pipeline::{
    AO_TERM_FORMAT, EDGES_FORMAT, GtaoPipelines, VIEW_DEPTH_FORMAT, VIEW_DEPTH_MIP_COUNT,
};

use crate::config::{GtaoSettings, PostProcessingSettings};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::post_processing::{EffectPasses, PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{
    ImportedBufferHandle, ImportedTextureHandle, SubresourceHandle, TextureHandle,
    TransientArrayLayers, TransientExtent, TransientSampleCount, TransientSubresourceDesc,
    TransientTextureDesc, TransientTextureFormat,
};

const GTAO_VIEW_DEPTH_LAYER0_LABELS: [&str; VIEW_DEPTH_MIP_COUNT as usize] = [
    "gtao_view_depth_mip0_l0",
    "gtao_view_depth_mip1_l0",
    "gtao_view_depth_mip2_l0",
    "gtao_view_depth_mip3_l0",
    "gtao_view_depth_mip4_l0",
];
const GTAO_VIEW_DEPTH_LAYER1_LABELS: [&str; VIEW_DEPTH_MIP_COUNT as usize] = [
    "gtao_view_depth_mip0_l1",
    "gtao_view_depth_mip1_l1",
    "gtao_view_depth_mip2_l1",
    "gtao_view_depth_mip3_l1",
    "gtao_view_depth_mip4_l1",
];

/// Effect descriptor that contributes the GTAO pass chain to the post-processing chain.
pub struct GtaoEffect {
    /// Snapshot of the GTAO settings used when building the chain for this frame. Live edits
    /// after chain build flow in via
    /// [`crate::passes::post_processing::settings_slot::GtaoSettingsSlot`] for non-topology
    /// fields; topology fields (`enabled`, `denoise_passes`) trigger a graph rebuild via
    /// [`crate::render_graph::post_processing::PostProcessChainSignature`].
    pub settings: GtaoSettings,
    /// Imported depth texture handle (declared as a sampled read for scheduling).
    pub depth: ImportedTextureHandle,
    /// Smooth view-space normal target produced after opaque forward rendering.
    pub view_normals: TextureHandle,
    /// Imported frame-uniforms buffer handle (fallback / scheduling; actual bind sources from
    /// [`crate::render_graph::frame_params::PerViewFramePlanSlot`] at record time).
    pub frame_uniforms: ImportedBufferHandle,
    /// Whether this graph is compiled for OpenXR multiview stereo.
    pub multiview_stereo: bool,
}

impl PostProcessEffect for GtaoEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::Gtao
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.gtao.enabled
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let pipelines = gtao_pipelines();
        let denoise_passes = self.settings.denoise_passes.min(3);

        let view_depth =
            builder.create_texture(view_depth_desc("gtao_view_depth", self.multiview_stereo));
        let view_depth_mips =
            create_view_depth_subresources(builder, view_depth, self.multiview_stereo);
        let (first_prefilter, last_prefilter) =
            add_view_depth_prefilter(builder, view_depth_mips, self, pipelines);

        let ao_term_a = builder.create_texture(ao_buffer_desc("gtao_ao_term_a"));
        let edges = builder.create_texture(ao_buffer_desc_format(
            "gtao_edges",
            TransientTextureFormat::Fixed(EDGES_FORMAT),
        ));
        let ao_term_b =
            (denoise_passes >= 2).then(|| builder.create_texture(ao_buffer_desc("gtao_ao_term_b")));

        let main = builder.add_raster_pass(Box::new(GtaoMainPass::new(
            GtaoMainResources {
                view_depth,
                view_normals: self.view_normals,
                frame_uniforms: self.frame_uniforms,
                ao_term: ao_term_a,
                edges,
            },
            self.settings,
            pipelines,
        )));
        builder.add_edge(last_prefilter, main);

        let mut last = main;
        let mut ao_for_apply = ao_term_a;
        if let Some(ao_term_b) = ao_term_b {
            let denoise_1 = builder.add_raster_pass(Box::new(GtaoDenoisePass::new(
                GtaoDenoiseResources {
                    ao_in: ao_term_a,
                    edges,
                    ao_out: ao_term_b,
                },
                self.settings,
                pipelines,
            )));
            builder.add_edge(last, denoise_1);
            last = denoise_1;
            ao_for_apply = ao_term_b;

            if denoise_passes >= 3 {
                let denoise_2 = builder.add_raster_pass(Box::new(GtaoDenoisePass::new(
                    GtaoDenoiseResources {
                        ao_in: ao_term_b,
                        edges,
                        ao_out: ao_term_a,
                    },
                    self.settings,
                    pipelines,
                )));
                builder.add_edge(last, denoise_2);
                last = denoise_2;
                ao_for_apply = ao_term_a;
            }
        }

        let apply = builder.add_raster_pass(Box::new(GtaoApplyPass::new(
            GtaoApplyResources {
                hdr_input: input,
                ao_in: ao_for_apply,
                edges,
                hdr_output: output,
            },
            self.settings,
            pipelines,
        )));
        builder.add_edge(last, apply);

        EffectPasses {
            first: first_prefilter,
            last: apply,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ViewDepthSubresources {
    layer0: [SubresourceHandle; VIEW_DEPTH_MIP_COUNT as usize],
    layer1: Option<[SubresourceHandle; VIEW_DEPTH_MIP_COUNT as usize]>,
}

impl ViewDepthSubresources {
    fn layer_count(self) -> u32 {
        if self.layer1.is_some() { 2 } else { 1 }
    }

    fn layer(self, layer: u32) -> Option<[SubresourceHandle; VIEW_DEPTH_MIP_COUNT as usize]> {
        match layer {
            0 => Some(self.layer0),
            1 => self.layer1,
            _ => None,
        }
    }
}

fn create_view_depth_subresources(
    builder: &mut GraphBuilder,
    view_depth: TextureHandle,
    multiview_stereo: bool,
) -> ViewDepthSubresources {
    let layer0 = std::array::from_fn(|mip| {
        builder.create_subresource(TransientSubresourceDesc {
            parent: view_depth,
            label: GTAO_VIEW_DEPTH_LAYER0_LABELS[mip],
            base_mip_level: mip as u32,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        })
    });
    let layer1 = multiview_stereo.then(|| {
        std::array::from_fn(|mip| {
            builder.create_subresource(TransientSubresourceDesc {
                parent: view_depth,
                label: GTAO_VIEW_DEPTH_LAYER1_LABELS[mip],
                base_mip_level: mip as u32,
                mip_level_count: 1,
                base_array_layer: 1,
                array_layer_count: 1,
            })
        })
    });
    ViewDepthSubresources { layer0, layer1 }
}

fn add_view_depth_prefilter(
    builder: &mut GraphBuilder,
    view_depth_mips: ViewDepthSubresources,
    effect: &GtaoEffect,
    pipelines: &'static GtaoPipelines,
) -> (
    crate::render_graph::ids::PassId,
    crate::render_graph::ids::PassId,
) {
    let first_resources = GtaoDepthPrefilterResources {
        depth: effect.depth,
        frame_uniforms: effect.frame_uniforms,
        source_mip: None,
        output_mip: view_depth_mips.layer0[0],
    };
    let first = builder.add_compute_pass(Box::new(GtaoDepthPrefilterPass::mip0(
        first_resources,
        effect.settings,
        pipelines,
        0,
    )));
    let mut last = first;
    for layer in 0..view_depth_mips.layer_count() {
        let Some(layer_mips) = view_depth_mips.layer(layer) else {
            continue;
        };
        for mip in 0..VIEW_DEPTH_MIP_COUNT {
            if mip == 0 && layer == 0 {
                continue;
            }
            let output_mip = layer_mips[mip as usize];
            let source_mip = (mip > 0).then(|| layer_mips[mip as usize - 1]);
            let resources = GtaoDepthPrefilterResources {
                depth: effect.depth,
                frame_uniforms: effect.frame_uniforms,
                source_mip,
                output_mip,
            };
            let pass = if mip == 0 {
                GtaoDepthPrefilterPass::mip0(resources, effect.settings, pipelines, layer)
            } else {
                GtaoDepthPrefilterPass::downsample(
                    resources,
                    effect.settings,
                    pipelines,
                    mip,
                    layer,
                )
            };
            let id = builder.add_compute_pass(Box::new(pass));
            builder.add_edge(last, id);
            last = id;
        }
    }
    (first, last)
}

/// Process-wide pipeline + UBO singleton shared across every GTAO chain rebuild.
fn gtao_pipelines() -> &'static GtaoPipelines {
    static CACHE: LazyLock<GtaoPipelines> = LazyLock::new(GtaoPipelines::default);
    &CACHE
}

fn should_record_depth_prefilter_layer(layer: u32, multiview_stereo: bool) -> bool {
    layer == 0 || (layer == 1 && multiview_stereo)
}

/// Transient texture descriptor for the AO term ping-pong buffers (`R8Unorm`, frame array
/// layers).
fn ao_buffer_desc(label: &'static str) -> TransientTextureDesc {
    ao_buffer_desc_format(label, TransientTextureFormat::Fixed(AO_TERM_FORMAT))
}

/// Transient texture descriptor for an `R8Unorm` GTAO buffer with a custom format slot.
fn ao_buffer_desc_format(
    label: &'static str,
    format: TransientTextureFormat,
) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format,
        extent: TransientExtent::Backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}

fn view_depth_desc(label: &'static str, multiview_stereo: bool) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::Fixed(VIEW_DEPTH_FORMAT),
        extent: TransientExtent::Backbuffer,
        mip_levels: VIEW_DEPTH_MIP_COUNT,
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

    #[test]
    fn gtao_effect_id_label() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
            view_normals: TextureHandle(0),
            frame_uniforms: ImportedBufferHandle(0),
            multiview_stereo: false,
        };
        assert_eq!(e.id(), PostProcessEffectId::Gtao);
        assert_eq!(e.id().label(), "GTAO");
    }

    #[test]
    fn gtao_effect_is_gated_by_master_and_per_effect_enable() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
            view_normals: TextureHandle(0),
            frame_uniforms: ImportedBufferHandle(0),
            multiview_stereo: false,
        };
        let mut s = PostProcessingSettings {
            enabled: false,
            ..Default::default()
        };
        assert!(!e.is_enabled(&s), "master off gates GTAO");
        s.enabled = true;
        assert!(e.is_enabled(&s), "master on + default GTAO on");
        s.gtao.enabled = false;
        assert!(!e.is_enabled(&s), "master on but GTAO off");
        s.gtao.enabled = true;
        s.enabled = false;
        assert!(!e.is_enabled(&s), "master off disables even if gtao on");
    }

    /// The WGSL `GtaoParams` struct is 64 bytes (16 x 4); changes here require updating
    /// `gtao_main.wgsl`, `gtao_denoise.wgsl`, and `gtao_apply.wgsl` simultaneously.
    #[test]
    fn gtao_params_gpu_size_is_64_bytes() {
        assert_eq!(size_of::<pipeline::GtaoParamsGpu>(), 64);
    }

    #[test]
    fn gtao_quality_levels_match_xegtao_presets() {
        assert_eq!(
            pipeline::GtaoQualityPreset::from_level(0, 1),
            pipeline::GtaoQualityPreset {
                slice_count: 1,
                steps_per_slice: 2,
            }
        );
        assert_eq!(
            pipeline::GtaoQualityPreset::from_level(1, 1),
            pipeline::GtaoQualityPreset {
                slice_count: 2,
                steps_per_slice: 2,
            }
        );
        assert_eq!(
            pipeline::GtaoQualityPreset::from_level(2, 1),
            pipeline::GtaoQualityPreset {
                slice_count: 3,
                steps_per_slice: 3,
            }
        );
        assert_eq!(
            pipeline::GtaoQualityPreset::from_level(3, 1),
            pipeline::GtaoQualityPreset {
                slice_count: 9,
                steps_per_slice: 3,
            }
        );
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
        let texture = builder.create_texture(view_depth_desc("gtao_view_depth", false));
        let mips = create_view_depth_subresources(&mut builder, texture, false);

        assert_eq!(mips.layer_count(), 1);
        assert!(mips.layer(1).is_none());
        assert!(
            builder
                .subresources
                .iter()
                .all(|desc| desc.base_array_layer == 0)
        );
        assert!(
            builder
                .subresources
                .iter()
                .all(|desc| !desc.label.ends_with("_l1"))
        );
    }

    #[test]
    fn stereo_view_depth_declares_layer_one() {
        let mut builder = GraphBuilder::new();
        let texture = builder.create_texture(view_depth_desc("gtao_view_depth", true));
        let mips = create_view_depth_subresources(&mut builder, texture, true);

        assert_eq!(mips.layer_count(), 2);
        assert!(mips.layer(1).is_some());
        assert!(
            builder
                .subresources
                .iter()
                .any(|desc| desc.label == "gtao_view_depth_mip4_l1")
        );
        assert_eq!(
            builder.textures[texture.index()].array_layers,
            TransientArrayLayers::Fixed(2)
        );
    }

    #[test]
    fn depth_prefilter_record_gate_matches_view_layers() {
        assert!(should_record_depth_prefilter_layer(0, false));
        assert!(should_record_depth_prefilter_layer(0, true));
        assert!(!should_record_depth_prefilter_layer(1, false));
        assert!(should_record_depth_prefilter_layer(1, true));
    }
}
