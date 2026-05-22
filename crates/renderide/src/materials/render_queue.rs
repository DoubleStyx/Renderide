//! Unity Built-in Render Pipeline render queue constants and material queue resolution.

use super::host_data::MaterialPropertyValue;
use super::material_passes::{MaterialPipelinePropertyIds, PropertyMapRef};

/// Unity `Geometry` render queue and the default queue for opaque materials.
pub const UNITY_RENDER_QUEUE_GEOMETRY: i32 = 2000;
/// Unity `AlphaTest` render queue.
pub const UNITY_RENDER_QUEUE_ALPHA_TEST: i32 = 2450;
/// First Unity queue value sorted as transparent by the Built-in Render Pipeline.
pub const UNITY_TRANSPARENT_RENDER_QUEUE_MIN: i32 = 2501;
/// Unity `Transparent` render queue and the fallback queue for unqueued alpha-blended materials.
pub const UNITY_RENDER_QUEUE_TRANSPARENT: i32 = 3000;
/// Unity `Overlay` render queue.
#[cfg(test)]
pub const UNITY_RENDER_QUEUE_OVERLAY: i32 = 4000;

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
    (queue >= 0).then(|| unity_render_queue_conversion(queue))
}

/// Applies the render queue mechanism that Unity uses for very large values.
/// Queues behave exactly as expected until reaching 2^15.
/// After that point, they seem to behave like slightly non opaque geometry (2501),
/// until reaching 2^16+2501, at which point they start behaving
/// as if they wrapped around to 2501.
/// Then the cycle continues: as expected of (x-2^16)
/// until they reach 2^16+2^15, at which point they return to 2000,
/// and so on... Repeating every 2^16. Tested manually up to 163840 (2^17 + 2^15).
fn unity_render_queue_conversion(queue: i32) -> i32 {
    if queue < 0x8000 {
        return queue;
    }
    let truncated = queue & 0xFFFF;
    if (UNITY_TRANSPARENT_RENDER_QUEUE_MIN..0x8000).contains(&truncated) {
        return truncated;
    }
    UNITY_TRANSPARENT_RENDER_QUEUE_MIN
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::host_data::{MaterialPropertyValue, PropertyIdRegistry};
    use hashbrown::HashMap;

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
    fn render_queue_pins_value_between_2_15_and_2_16() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut material = HashMap::new();
        material.insert(ids.render_queue[0], MaterialPropertyValue::Float(42_000.0));

        assert_eq!(
            material_render_queue_from_maps(
                Some(&material),
                None,
                &ids,
                UNITY_RENDER_QUEUE_GEOMETRY,
            ),
            UNITY_TRANSPARENT_RENDER_QUEUE_MIN
        );
    }

    #[test]
    fn render_queue_wraps_around_after_2_16() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut material = HashMap::new();
        material.insert(ids.render_queue[0], MaterialPropertyValue::Float(69_536.0));

        assert_eq!(
            material_render_queue_from_maps(
                Some(&material),
                None,
                &ids,
                UNITY_RENDER_QUEUE_GEOMETRY,
            ),
            UNITY_RENDER_QUEUE_OVERLAY
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
