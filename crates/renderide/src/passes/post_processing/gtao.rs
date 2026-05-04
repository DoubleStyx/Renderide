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
    ImportedBufferHandle, ImportedTextureHandle, TextureHandle, TransientArrayLayers,
    TransientExtent, TransientSampleCount, TransientSubresourceDesc, TransientTextureDesc,
    TransientTextureFormat,
};

const GTAO_VIEW_DEPTH_LABELS: [[&str; 2]; VIEW_DEPTH_MIP_COUNT as usize] = [
    ["gtao_view_depth_mip0_l0", "gtao_view_depth_mip0_l1"],
    ["gtao_view_depth_mip1_l0", "gtao_view_depth_mip1_l1"],
    ["gtao_view_depth_mip2_l0", "gtao_view_depth_mip2_l1"],
    ["gtao_view_depth_mip3_l0", "gtao_view_depth_mip3_l1"],
    ["gtao_view_depth_mip4_l0", "gtao_view_depth_mip4_l1"],
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

        let view_depth = builder.create_texture(view_depth_desc("gtao_view_depth"));
        let view_depth_mips = create_view_depth_subresources(builder, view_depth);
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

fn create_view_depth_subresources(
    builder: &mut GraphBuilder,
    view_depth: TextureHandle,
) -> [[crate::render_graph::resources::SubresourceHandle; 2]; VIEW_DEPTH_MIP_COUNT as usize] {
    std::array::from_fn(|mip| {
        std::array::from_fn(|layer| {
            builder.create_subresource(TransientSubresourceDesc {
                parent: view_depth,
                label: GTAO_VIEW_DEPTH_LABELS[mip][layer],
                base_mip_level: mip as u32,
                mip_level_count: 1,
                base_array_layer: layer as u32,
                array_layer_count: 1,
            })
        })
    })
}

fn add_view_depth_prefilter(
    builder: &mut GraphBuilder,
    view_depth_mips: [[crate::render_graph::resources::SubresourceHandle; 2];
        VIEW_DEPTH_MIP_COUNT as usize],
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
        output_mip: view_depth_mips[0][0],
    };
    let first = builder.add_compute_pass(Box::new(GtaoDepthPrefilterPass::mip0(
        first_resources,
        effect.settings,
        pipelines,
        0,
    )));
    let mut last = first;
    for mip in 0..VIEW_DEPTH_MIP_COUNT {
        for layer in 0..2 {
            if mip == 0 && layer == 0 {
                continue;
            }
            let layer_idx = layer as usize;
            let output_mip = view_depth_mips[mip as usize][layer_idx];
            let source_mip = (mip > 0).then(|| view_depth_mips[mip as usize - 1][layer_idx]);
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

fn view_depth_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::Fixed(VIEW_DEPTH_FORMAT),
        extent: TransientExtent::Backbuffer,
        mip_levels: VIEW_DEPTH_MIP_COUNT,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
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
}
