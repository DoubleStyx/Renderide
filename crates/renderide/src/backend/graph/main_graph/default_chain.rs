//! Default post-processing chain shipped with the renderer.

use crate::render_graph::post_process_chain;
use crate::render_graph::resources::TextureHandle;

use super::handles::{MainGraphHandles, MainGraphPostProcessingResources};

/// Builds the canonical post-processing chain.
///
/// Execution order is GTAO -> auto-exposure -> bloom -> motion blur -> selected tonemap. GTAO runs first so
/// ambient occlusion modulates linear HDR light before metering; auto-exposure meters and scales
/// the HDR scene before bloom; bloom scatters exposed HDR light; motion blur filters HDR scene
/// color from camera velocity; then the selected tonemap curve compresses the final exposed HDR signal
/// to display-referred `[0, 1]`. Each effect gates itself
/// via [`post_process_chain::PostProcessEffect::is_enabled`] against the live
/// [`crate::config::PostProcessingSettings`].
///
/// `GtaoEffect` is parameterised with the current [`crate::config::GtaoSettings`] snapshot and the
/// imported `frame_uniforms` handle (used to access per-eye projection coefficients and the frame
/// index at record time). It is registered only when the graph also created the matching
/// view-normal texture. `BloomEffect` captures a [`crate::config::BloomSettings`] snapshot for its
/// shared params UBO and per-mip blend constants.
pub(super) fn build_default_post_processing_chain(
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    multiview_stereo: bool,
    post_processing_resources: &MainGraphPostProcessingResources,
    gtao_view_normals: Option<TextureHandle>,
) -> post_process_chain::PostProcessChain {
    let mut chain = post_process_chain::PostProcessChain::new();
    if let Some(view_normals) = gtao_view_normals {
        chain.push(Box::new(crate::passes::GtaoEffect {
            settings: post_processing_settings.gtao,
            depth: h.depth,
            view_normals,
            frame_uniforms: h.frame_uniforms,
            multiview_stereo,
        }));
    }
    chain.push(Box::new(crate::passes::AutoExposureEffect::new(
        post_processing_resources.auto_exposure_state_cache(),
    )));
    chain.push(Box::new(crate::passes::BloomEffect {
        settings: post_processing_settings.bloom,
    }));
    chain.push(Box::new(crate::passes::MotionBlurEffect::new(
        h.depth,
        post_processing_resources.motion_blur_state_cache(),
    )));
    chain.push(Box::new(crate::passes::AcesTonemapEffect));
    chain.push(Box::new(crate::passes::AgxTonemapEffect));
    chain
}
