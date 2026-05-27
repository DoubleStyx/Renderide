//! Default post-processing chain shipped with the renderer.

use super::handles::{MainGraphHandles, MainGraphPostProcessingResources};
use crate::render_graph::post_process_chain;

/// Builds the canonical post-processing chain.
///
/// Execution order is auto-exposure -> bloom -> motion blur -> selected tonemap. GTAO is not part
/// of this chain; the main graph applies it earlier, directly after opaque/cutout rendering, so
/// later transparent pixels are not darkened by opaque screen-space occlusion. Auto-exposure
/// meters and scales the HDR scene before bloom; bloom scatters exposed HDR light; motion blur
/// filters HDR scene color from camera velocity; then the selected tonemap curve compresses the
/// final exposed HDR signal to display-referred `[0, 1]`. Each effect gates itself
/// via [`post_process_chain::PostProcessEffect::is_enabled`] against the live
/// [`crate::config::PostProcessingSettings`].
///
/// `BloomEffect` captures a [`crate::config::BloomSettings`] snapshot for its shared params UBO
/// and per-mip blend constants.
pub(super) fn build_default_post_processing_chain(
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    post_processing_resources: &MainGraphPostProcessingResources,
) -> post_process_chain::PostProcessChain {
    let mut chain = post_process_chain::PostProcessChain::new();
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
