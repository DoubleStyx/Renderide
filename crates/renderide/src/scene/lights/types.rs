//! Cached and resolved light types for the scene light pipeline (CPU-side host mirror).

use glam::Vec3;

use crate::shared::{LightData, LightType, LightsBufferRendererState, ShadowType};

/// Cached light entry combining pose data from submission with state from updates.
#[derive(Clone, Debug)]
pub struct CachedLight {
    /// Local-space pose and color from [`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) payload rows.
    pub data: LightData,
    /// Renderable index, type, and shadow params from frame updates.
    pub state: LightsBufferRendererState,
    /// Dense transform index for world matrix lookup (from host additions batch).
    pub transform_id: usize,
}

impl CachedLight {
    /// Creates a cached light with default buffer state when only [`LightData`] is available.
    pub fn from_data(data: LightData) -> Self {
        Self {
            data,
            state: LightsBufferRendererState::default(),
            transform_id: 0,
        }
    }
}

/// Resolved light in world space, ready for GPU packing and shading.
#[derive(Clone, Debug)]
pub struct ResolvedLight {
    /// World-space position.
    pub world_position: Vec3,
    /// World-space propagation direction (normalized): local **+Z** after transform.
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
    /// Buffer global unique id, or `-1` for regular [`crate::shared::LightState`] lights.
    pub global_unique_id: i32,
    /// Shadow mode from the host.
    pub shadow_type: ShadowType,
    /// Shadow strength multiplier (0 = no shadow contribution).
    pub shadow_strength: f32,
    /// Near plane for shadow volumes (host units).
    pub shadow_near_plane: f32,
    /// Depth bias for shadow maps (host value).
    pub shadow_bias: f32,
    /// Normal bias for shadow receivers (host value).
    pub shadow_normal_bias: f32,
}

/// Whether `resolved` should cast shadows (ray-traced path guard).
///
/// [`ShadowType::none`] or non-positive [`ResolvedLight::shadow_strength`] disables shadow rays.
pub fn light_casts_shadows(resolved: &ResolvedLight) -> bool {
    resolved.shadow_type != ShadowType::none && resolved.shadow_strength > 0.0
}
