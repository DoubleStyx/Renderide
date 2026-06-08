//! Host-owned realtime shadow quality state.

use crate::shared::{LightType, QualityConfig, ShadowCascadeMode, ShadowResolutionMode};

/// Default Froox/Renderite local-light shadow cap.
const DEFAULT_PER_PIXEL_LIGHTS: u32 = 12;
/// Default Froox/Renderite directional shadow distance.
const DEFAULT_SHADOW_DISTANCE: f32 = 175.0;
/// Maximum local shadowed lights Renderide assigns per view from host quality.
const MAX_LOCAL_SHADOWED_LIGHTS: u32 = 32;
/// Minimum sane finite shadow distance.
const MIN_SHADOW_DISTANCE: f32 = 0.001;
/// Maximum quality-derived directional shadow resolution.
const MAX_DIRECTIONAL_SHADOW_RESOLUTION: u32 = 4096;
/// Maximum quality-derived spot shadow resolution.
const MAX_SPOT_SHADOW_RESOLUTION: u32 = 2048;
/// Maximum quality-derived point face shadow resolution.
const MAX_POINT_SHADOW_RESOLUTION: u32 = 1024;

/// Renderer-side shadow quality derived from the host `QualityConfig`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HostShadowQuality {
    /// Local shadowed point/spot light cap.
    pub(crate) per_pixel_lights: u32,
    /// Directional-light cascade count.
    pub(crate) cascade_count: u32,
    /// Requested shadow tile edge in pixels.
    pub(crate) tile_resolution: u32,
    /// Maximum directional shadow distance.
    pub(crate) shadow_distance: f32,
}

impl HostShadowQuality {
    /// Converts host quality settings into renderer shadow planning values.
    pub(crate) fn from_quality_config(cfg: &QualityConfig) -> Self {
        Self {
            per_pixel_lights: sanitize_per_pixel_lights(cfg.per_pixel_lights),
            cascade_count: cascade_count(cfg.shadow_cascades),
            tile_resolution: tile_resolution(cfg.shadow_resolution),
            shadow_distance: sanitize_shadow_distance(cfg.shadow_distance),
        }
    }

    /// Returns the quality-derived shadow tile edge for `light_type`.
    pub(crate) fn tile_resolution_for_light_type(self, light_type: LightType) -> u32 {
        match light_type {
            LightType::Directional => self.tile_resolution.min(MAX_DIRECTIONAL_SHADOW_RESOLUTION),
            LightType::Spot => self.tile_resolution.min(MAX_SPOT_SHADOW_RESOLUTION),
            LightType::Point => self.tile_resolution.min(MAX_POINT_SHADOW_RESOLUTION),
        }
    }
}

impl Default for HostShadowQuality {
    fn default() -> Self {
        Self {
            per_pixel_lights: DEFAULT_PER_PIXEL_LIGHTS,
            cascade_count: 4,
            tile_resolution: 2048,
            shadow_distance: DEFAULT_SHADOW_DISTANCE,
        }
    }
}

fn sanitize_per_pixel_lights(raw: i32) -> u32 {
    if raw <= 0 {
        return DEFAULT_PER_PIXEL_LIGHTS;
    }
    (raw as u32).min(MAX_LOCAL_SHADOWED_LIGHTS)
}

fn cascade_count(mode: ShadowCascadeMode) -> u32 {
    match mode {
        ShadowCascadeMode::None => 1,
        ShadowCascadeMode::TwoCascades => 2,
        ShadowCascadeMode::FourCascades => 4,
    }
}

fn tile_resolution(mode: ShadowResolutionMode) -> u32 {
    match mode {
        ShadowResolutionMode::Low => 512,
        ShadowResolutionMode::Medium => 1024,
        ShadowResolutionMode::High => 2048,
        ShadowResolutionMode::Ultra => 4096,
    }
}

fn sanitize_shadow_distance(raw: f32) -> f32 {
    if raw.is_finite() && raw > 0.0 {
        raw.max(MIN_SHADOW_DISTANCE)
    } else {
        DEFAULT_SHADOW_DISTANCE
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::{LightType, QualityConfig, ShadowCascadeMode, ShadowResolutionMode};

    use super::HostShadowQuality;

    #[test]
    fn maps_host_shadow_quality() {
        let quality = HostShadowQuality::from_quality_config(&QualityConfig {
            per_pixel_lights: 8,
            shadow_cascades: ShadowCascadeMode::TwoCascades,
            shadow_resolution: ShadowResolutionMode::Medium,
            shadow_distance: 42.0,
            ..Default::default()
        });

        assert_eq!(quality.per_pixel_lights, 8);
        assert_eq!(quality.cascade_count, 2);
        assert_eq!(quality.tile_resolution, 1024);
        assert_eq!(quality.shadow_distance, 42.0);
    }

    #[test]
    fn defaults_invalid_host_shadow_values_to_froox_scalar_defaults() {
        let quality = HostShadowQuality::from_quality_config(&QualityConfig {
            per_pixel_lights: -1,
            shadow_distance: f32::NAN,
            ..Default::default()
        });

        assert_eq!(
            quality.per_pixel_lights,
            HostShadowQuality::default().per_pixel_lights
        );
        assert_eq!(
            quality.shadow_distance,
            HostShadowQuality::default().shadow_distance
        );
    }

    #[test]
    fn ultra_quality_uses_light_type_shadow_resolution_caps() {
        let quality = HostShadowQuality::from_quality_config(&QualityConfig {
            shadow_resolution: ShadowResolutionMode::Ultra,
            ..Default::default()
        });

        assert_eq!(
            quality.tile_resolution_for_light_type(LightType::Directional),
            4096
        );
        assert_eq!(
            quality.tile_resolution_for_light_type(LightType::Spot),
            2048
        );
        assert_eq!(
            quality.tile_resolution_for_light_type(LightType::Point),
            1024
        );
    }
}
