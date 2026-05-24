//! Constant-color analytic IBL source builders.

use crate::skybox::params::{SkyboxEvaluatorParams, SkyboxParamMode};

use super::source::{SkyboxIblSource, SolidColorIblSource};

/// Builds a constant-color IBL source.
pub(crate) fn solid_color_ibl_source(identity: u64, color: [f32; 4]) -> SkyboxIblSource {
    SkyboxIblSource::SolidColor(SolidColorIblSource { identity, color })
}

/// Builds analytic evaluator params for a constant-color source.
pub(crate) fn solid_color_params(color: [f32; 4]) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Gradient);
    params.color0 = color;
    params.sample_size = 1;
    params.gradient_count = 0;
    params
}
