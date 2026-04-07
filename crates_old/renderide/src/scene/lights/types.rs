//! Cached and resolved light types for the scene light pipeline.

use glam::Vec3;

use crate::shared::{LightData, LightType, LightsBufferRendererState, ShadowType};

/// Cached light entry combining pose data from submission with state from updates.
#[derive(Clone, Debug)]
pub struct CachedLight {
    /// Local-space pose and color from LightsBufferRendererSubmission.
    pub data: LightData,
    /// Renderable index, type, and shadow params from LightsBufferRendererUpdate.
    pub state: LightsBufferRendererState,
    /// Transform index for world matrix lookup. Set from additions (host sends transform indices).
    pub transform_id: usize,
}

impl CachedLight {
    /// Creates a new cached light with default state when only data is available.
    pub fn from_data(data: LightData) -> Self {
        Self {
            data,
            state: LightsBufferRendererState::default(),
            transform_id: 0,
        }
    }
}

/// Resolved light in world space, ready for the render loop.
#[derive(Clone, Debug)]
pub struct ResolvedLight {
    /// World-space position.
    pub world_position: Vec3,
    /// World-space propagation direction (normalized): local **+Z** (host `transform.forward`)
    /// rotated by [`LightData::orientation`] and the light’s world matrix. PBR shaders use
    /// `normalize(-world_direction)` for directional **to-light** and the same axis for spot cones.
    pub world_direction: Vec3,
    /// RGB color.
    pub color: Vec3,
    /// Light intensity.
    pub intensity: f32,
    /// Attenuation range (point/spot).
    pub range: f32,
    /// Spot angle in degrees (spot only).
    pub spot_angle: f32,
    /// Light type: point, directional, or spot.
    pub light_type: LightType,
    /// Global unique ID for consumed feedback.
    pub global_unique_id: i32,
    /// Shadow mode from the host (`LightsBufferRendererState` / [`LightState`]).
    pub shadow_type: ShadowType,
    /// Shadow strength multiplier (0 = no shadow contribution).
    pub shadow_strength: f32,
    /// Near plane for shadow projection volumes (host units).
    pub shadow_near_plane: f32,
    /// Depth bias applied when sampling shadow maps (host value; unused until map shadows).
    pub shadow_bias: f32,
    /// Normal bias for shadow receivers (host value; unused until map shadows).
    pub shadow_normal_bias: f32,
}

/// Returns whether `resolved` should participate in shadow casting for RT (and matches the
/// per-light guard used in PBR ray-query WGSL).
///
/// Aligns with Renderite-style rules: [`ShadowType::none`] or non-positive
/// [`ResolvedLight::shadow_strength`] means no shadow rays for that light.
pub fn light_casts_shadows(resolved: &ResolvedLight) -> bool {
    resolved.shadow_type != ShadowType::none && resolved.shadow_strength > 0.0
}
