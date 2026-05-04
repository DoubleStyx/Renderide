//! Unity Built-in Render Pipeline render queue constants and material queue resolution.

use super::host_data::MaterialPropertyValue;
use super::material_passes::{MaterialPipelinePropertyIds, PropertyMapRef};

/// Unity `Background` render queue.
pub const UNITY_RENDER_QUEUE_BACKGROUND: i32 = 1000;
/// Unity `Geometry` render queue and the default queue for opaque materials.
pub const UNITY_RENDER_QUEUE_GEOMETRY: i32 = 2000;
/// Unity `AlphaTest` render queue.
pub const UNITY_RENDER_QUEUE_ALPHA_TEST: i32 = 2450;
/// Highest Unity queue value still sorted as opaque by the Built-in Render Pipeline.
pub const UNITY_OPAQUE_RENDER_QUEUE_MAX: i32 = 2500;
/// First Unity queue value sorted as transparent by the Built-in Render Pipeline.
pub const UNITY_TRANSPARENT_RENDER_QUEUE_MIN: i32 = 2501;
/// Unity `Transparent` render queue and the fallback queue for unqueued alpha-blended materials.
pub const UNITY_RENDER_QUEUE_TRANSPARENT: i32 = 3000;
/// Unity `Overlay` render queue.
pub const UNITY_RENDER_QUEUE_OVERLAY: i32 = 4000;
/// Fog volume proxies (`FogBoxVolume`): draw after typical `Transparent` (3000+) PBS meshes so
/// alpha-blended fog composites on top instead of being replaced by nearer opaque-cutout draws.
pub const UNITY_RENDER_QUEUE_VOLUME_FOG: i32 = UNITY_RENDER_QUEUE_OVERLAY - 10;

/// Returns whether a Unity render queue uses transparent-style sorting.
#[inline]
pub fn render_queue_is_transparent(render_queue: i32) -> bool {
    render_queue >= UNITY_TRANSPARENT_RENDER_QUEUE_MIN
}

/// Returns the compatibility fallback queue when the host did not send a material queue.
#[inline]
pub(crate) fn fallback_render_queue_for_material(alpha_blended: bool) -> i32 {
    if alpha_blended {
        UNITY_RENDER_QUEUE_TRANSPARENT
    } else {
        UNITY_RENDER_QUEUE_GEOMETRY
    }
}

/// Resolves the material-side `_RenderQueue` override, falling back when absent or negative.
///
/// The property-block map is intentionally ignored: Unity render queue is material state, not a
/// per-renderer property-block override.
pub(crate) fn material_render_queue_from_maps(
    material_map: PropertyMapRef<'_>,
    _property_block_map: PropertyMapRef<'_>,
    ids: &MaterialPipelinePropertyIds,
    fallback_render_queue: i32,
) -> i32 {
    first_material_float(material_map, &ids.render_queue)
        .and_then(sanitized_render_queue_override)
        .unwrap_or(fallback_render_queue)
}

fn first_material_float(material_map: PropertyMapRef<'_>, pids: &[i32]) -> Option<f32> {
    pids.iter().find_map(|&pid| {
        let value = material_map?.get(&pid)?;
        match value {
            MaterialPropertyValue::Float(value) => Some(*value),
            MaterialPropertyValue::Float4(value) => Some(value[0]),
            _ => None,
        }
    })
}

fn sanitized_render_queue_override(raw: f32) -> Option<i32> {
    if !raw.is_finite() {
        return None;
    }
    let queue = raw.round() as i32;
    (queue >= 0).then_some(queue)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::materials::host_data::{MaterialPropertyValue, PropertyIdRegistry};

    #[test]
    fn transparent_sorting_starts_after_opaque_cutoff() {
        assert!(!render_queue_is_transparent(UNITY_OPAQUE_RENDER_QUEUE_MAX));
        assert!(render_queue_is_transparent(
            UNITY_TRANSPARENT_RENDER_QUEUE_MIN
        ));
    }

    #[test]
    fn volume_fog_queue_is_transparent_and_before_overlay() {
        assert!(render_queue_is_transparent(UNITY_RENDER_QUEUE_VOLUME_FOG));
        const { assert!(UNITY_RENDER_QUEUE_VOLUME_FOG < UNITY_RENDER_QUEUE_OVERLAY) };
        const { assert!(UNITY_RENDER_QUEUE_VOLUME_FOG > UNITY_RENDER_QUEUE_TRANSPARENT) };
    }

    #[test]
    fn material_render_queue_uses_material_override() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut material = HashMap::new();
        material.insert(ids.render_queue[0], MaterialPropertyValue::Float(2450.0));

        assert_eq!(
            material_render_queue_from_maps(
                Some(&material),
                None,
                &ids,
                UNITY_RENDER_QUEUE_GEOMETRY,
            ),
            UNITY_RENDER_QUEUE_ALPHA_TEST
        );
    }

    #[test]
    fn negative_render_queue_falls_back() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut material = HashMap::new();
        material.insert(ids.render_queue[0], MaterialPropertyValue::Float(-1.0));

        assert_eq!(
            material_render_queue_from_maps(
                Some(&material),
                None,
                &ids,
                UNITY_RENDER_QUEUE_TRANSPARENT,
            ),
            UNITY_RENDER_QUEUE_TRANSPARENT
        );
    }

    #[test]
    fn property_block_does_not_override_render_queue() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut material = HashMap::new();
        let mut property_block = HashMap::new();
        material.insert(ids.render_queue[0], MaterialPropertyValue::Float(2000.0));
        property_block.insert(ids.render_queue[0], MaterialPropertyValue::Float(4000.0));

        assert_eq!(
            material_render_queue_from_maps(
                Some(&material),
                Some(&property_block),
                &ids,
                UNITY_RENDER_QUEUE_TRANSPARENT,
            ),
            UNITY_RENDER_QUEUE_GEOMETRY
        );
    }
}
