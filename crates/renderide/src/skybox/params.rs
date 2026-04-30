//! Shared CPU-side parameter packing for analytic skybox evaluators.

use bytemuck::{Pod, Zeroable};

use crate::backend::material_property_reader::{
    float_property, float4_array16_property, float4_property,
};
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

/// Default sky parameter sample grid used by SH2 projection.
pub(crate) const DEFAULT_SKYBOX_SAMPLE_SIZE: u32 = 64;
/// Default generated cubemap face size for analytic skybox baking.
pub(crate) const DEFAULT_GENERATED_SKYBOX_FACE_SIZE: u32 = 128;
/// Default `Projection360` field of view used by Unity material defaults.
pub(crate) const PROJECTION360_DEFAULT_FOV: [f32; 4] =
    [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];
/// Default texture scale/offset used by Unity `_MainTex_ST` properties.
pub(crate) const DEFAULT_MAIN_TEX_ST: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

/// Parameter-only sky evaluator mode used by skybox compute shaders.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SkyboxParamMode {
    /// Procedural sky approximation from material scalar/color properties.
    Procedural = 1,
    /// Gradient sky approximation from material array properties.
    Gradient = 2,
}

/// Uniform payload shared by analytic skybox compute kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct SkyboxEvaluatorParams {
    /// Sample grid edge for projection or generated cubemap face edge for baking.
    pub(crate) sample_size: u32,
    /// Evaluator mode from [`SkyboxParamMode`].
    pub(crate) mode: u32,
    /// Number of active gradient lobes.
    pub(crate) gradient_count: u32,
    /// Reserved alignment slot.
    pub(crate) _pad0: u32,
    /// Generic color slot 0.
    pub(crate) color0: [f32; 4],
    /// Generic color slot 1.
    pub(crate) color1: [f32; 4],
    /// Generic direction and scalar slot.
    pub(crate) direction: [f32; 4],
    /// Generic scalar slot.
    pub(crate) scalars: [f32; 4],
    /// Gradient direction/spread rows.
    pub(crate) dirs_spread: [[f32; 4]; 16],
    /// Gradient color rows A.
    pub(crate) gradient_color0: [[f32; 4]; 16],
    /// Gradient color rows B.
    pub(crate) gradient_color1: [[f32; 4]; 16],
    /// Gradient parameter rows.
    pub(crate) gradient_params: [[f32; 4]; 16],
}

impl SkyboxEvaluatorParams {
    /// Creates a parameter block with the default projection sample grid.
    pub(crate) fn empty(mode: SkyboxParamMode) -> Self {
        Self {
            sample_size: DEFAULT_SKYBOX_SAMPLE_SIZE,
            mode: mode as u32,
            gradient_count: 0,
            _pad0: 0,
            color0: [0.0; 4],
            color1: [0.0; 4],
            direction: [0.0, 1.0, 0.0, 0.0],
            scalars: [1.0, 0.0, 0.0, 0.0],
            dirs_spread: [[0.0; 4]; 16],
            gradient_color0: [[0.0; 4]; 16],
            gradient_color1: [[0.0; 4]; 16],
            gradient_params: [[0.0; 4]; 16],
        }
    }

    /// Returns a copy with the sample or face edge set.
    pub(crate) fn with_sample_size(mut self, sample_size: u32) -> Self {
        self.sample_size = sample_size.max(1);
        self
    }
}

/// Converts a storage-orientation boolean to the shader keyword float convention.
pub(crate) fn storage_v_inverted_flag(storage_v_inverted: bool) -> f32 {
    if storage_v_inverted { 1.0 } else { 0.0 }
}

/// Builds the `Projection360` equirectangular sampling payload shared with compute shaders.
pub(crate) fn projection360_equirect_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    storage_v_inverted: bool,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Procedural);
    params.color0 = float4_property(store, registry, lookup, "_FOV", PROJECTION360_DEFAULT_FOV);
    params.color1 = float4_property(store, registry, lookup, "_MainTex_ST", DEFAULT_MAIN_TEX_ST);
    params.scalars = [storage_v_inverted_flag(storage_v_inverted), 0.0, 0.0, 0.0];
    params
}

/// Builds parameter payload for a procedural sky material.
pub(crate) fn procedural_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Procedural);
    params.color0 = float4_property(store, registry, lookup, "_SkyTint", [0.5, 0.5, 0.5, 1.0]);
    params.color1 = float4_property(
        store,
        registry,
        lookup,
        "_GroundColor",
        [0.35, 0.35, 0.35, 1.0],
    );
    params.direction = float4_property(
        store,
        registry,
        lookup,
        "_SunDirection",
        [0.0, 1.0, 0.0, 0.0],
    );
    let exposure = float_property(store, registry, lookup, "_Exposure", 1.0);
    let sun_size = float_property(store, registry, lookup, "_SunSize", 0.04);
    let atmosphere = float_property(store, registry, lookup, "_AtmosphereThickness", 1.0);
    let sun_disk_mode = procedural_sun_disk_mode(store, registry, lookup);
    params.scalars = [exposure, sun_size, atmosphere, sun_disk_mode];
    params.gradient_color0[0] =
        float4_property(store, registry, lookup, "_SunColor", [1.0, 0.95, 0.85, 1.0]);
    params
}

/// Builds parameter payload for a gradient sky material.
pub(crate) fn gradient_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Gradient);
    params.color0 = float4_property(store, registry, lookup, "_BaseColor", [0.0, 0.0, 0.0, 1.0]);
    params.dirs_spread = float4_array16_property(store, registry, lookup, "_DirsSpread");
    params.gradient_color0 = float4_array16_property(store, registry, lookup, "_Color0");
    params.gradient_color1 = float4_array16_property(store, registry, lookup, "_Color1");
    params.gradient_params = float4_array16_property(store, registry, lookup, "_Params");
    params.gradient_count = float_property(store, registry, lookup, "_Gradients", 0.0)
        .round()
        .clamp(0.0, 16.0) as u32;
    if params.gradient_count == 0 {
        params.gradient_count = params
            .dirs_spread
            .iter()
            .position(|v| v.iter().all(|c| c.abs() < 1e-6))
            .unwrap_or(16) as u32;
    }
    params
}

/// Encodes the procedural sun disk keyword state as a scalar for WGSL.
fn procedural_sun_disk_mode(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> f32 {
    let none = float_property(store, registry, lookup, "_SUNDISK_NONE", 0.0);
    if none.abs() > f32::EPSILON {
        return 0.0;
    }
    let high_quality = float_property(store, registry, lookup, "_SUNDISK_HIGH_QUALITY", 0.0);
    if high_quality.abs() > f32::EPSILON {
        return 2.0;
    }
    1.0
}
