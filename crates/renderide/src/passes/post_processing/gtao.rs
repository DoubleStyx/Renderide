//! Ground-Truth Ambient Occlusion (Jimenez et al. 2016) post-processing effect with
//! XeGTAO-style depth-aware bilateral denoise.
//!
//! Registers a three-stage chain on the post-processing graph builder:
//!
//! 1. [`main_pass::GtaoMainPass`] — produces the AO term (scaled by
//!    `1 / OCCLUSION_TERM_SCALE` per XeGTAO's headroom convention) and packed depth-edge
//!    weights from the imported scene depth. The HDR scene-color input is *not* read here;
//!    modulation is deferred to the apply stage so the bilateral denoiser can act on the AO
//!    term first.
//! 2. [`denoise_pass::GtaoDenoisePass`] — XeGTAO 3×3 edge-preserving bilateral filter.
//!    Only registered when [`crate::config::GtaoSettings::denoise_passes`] is `>= 2`.
//! 3. [`apply_pass::GtaoApplyPass`] — final denoise iteration that multiplies the AO term by
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
mod main_pass;
mod pipeline;

use std::sync::LazyLock;

use apply_pass::{GtaoApplyPass, GtaoApplyResources};
use denoise_pass::{GtaoDenoisePass, GtaoDenoiseResources};
use main_pass::{GtaoMainPass, GtaoMainResources};
use pipeline::{AO_TERM_FORMAT, EDGES_FORMAT, GtaoPipelines};

use crate::config::{GtaoSettings, PostProcessingSettings};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::post_processing::{EffectPasses, PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{
    ImportedBufferHandle, ImportedTextureHandle, TextureHandle, TransientArrayLayers,
    TransientExtent, TransientSampleCount, TransientTextureDesc, TransientTextureFormat,
};

/// Effect descriptor that contributes the GTAO three-pass chain to the post-processing
/// chain.
pub struct GtaoEffect {
    /// Snapshot of the GTAO settings used when building the chain for this frame. Live edits
    /// after chain build flow in via
    /// [`crate::passes::post_processing::settings_slot::GtaoSettingsSlot`] for non-topology
    /// fields; topology fields (`enabled`, `denoise_passes`) trigger a graph rebuild via
    /// [`crate::render_graph::post_processing::PostProcessChainSignature`].
    pub settings: GtaoSettings,
    /// Imported depth texture handle (declared as a sampled read for scheduling).
    pub depth: ImportedTextureHandle,
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
        let denoise_passes = self.settings.denoise_passes.min(2);

        let ao_term_a = builder.create_texture(ao_buffer_desc("gtao_ao_term_a"));
        let edges = builder.create_texture(ao_buffer_desc_format(
            "gtao_edges",
            TransientTextureFormat::Fixed(EDGES_FORMAT),
        ));
        let ao_term_b =
            (denoise_passes >= 2).then(|| builder.create_texture(ao_buffer_desc("gtao_ao_term_b")));

        let main = builder.add_raster_pass(Box::new(GtaoMainPass::new(
            GtaoMainResources {
                depth: self.depth,
                frame_uniforms: self.frame_uniforms,
                ao_term: ao_term_a,
                edges,
            },
            self.settings,
            pipelines,
        )));

        let (intermediate, ao_for_apply) = if let Some(ao_term_b) = ao_term_b {
            let intermediate = builder.add_raster_pass(Box::new(GtaoDenoisePass::new(
                GtaoDenoiseResources {
                    ao_in: ao_term_a,
                    edges,
                    ao_out: ao_term_b,
                },
                self.settings,
                pipelines,
            )));
            builder.add_edge(main, intermediate);
            (Some(intermediate), ao_term_b)
        } else {
            (None, ao_term_a)
        };

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
        builder.add_edge(intermediate.unwrap_or(main), apply);

        EffectPasses {
            first: main,
            last: apply,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gtao_effect_id_label() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
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

    /// The WGSL `GtaoParams` struct is 32 bytes (8 × 4); changes here require updating
    /// `gtao_main.wgsl`, `gtao_denoise.wgsl`, and `gtao_apply.wgsl` simultaneously.
    #[test]
    fn gtao_params_gpu_size_is_32_bytes() {
        assert_eq!(size_of::<pipeline::GtaoParamsGpu>(), 32);
    }

    /// Verifies the bundle of caches constructs (which exercises the manual `Default`
    /// implementations in `pipeline.rs` that pick bounded bind-group caches).
    #[test]
    fn pipeline_caches_default_construct() {
        let _ = GtaoPipelines::default();
    }
}
