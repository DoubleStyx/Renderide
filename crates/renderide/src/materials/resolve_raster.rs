//! Host shader asset → raster material family for mesh draws.

use super::router::MaterialRouter;
use super::MaterialFamilyId;

/// Resolves the material family used for **mesh rasterization** for a host shader asset id.
///
/// Uses [`MaterialRouter::family_for_shader_asset`], populated when the host sends
/// [`crate::shared::ShaderUpload`] (see [`crate::assets::shader::resolve_shader_upload`]).
pub fn resolve_raster_family(shader_asset_id: i32, router: &MaterialRouter) -> MaterialFamilyId {
    router.family_for_shader_asset(shader_asset_id)
}

#[cfg(test)]
mod tests {
    use super::resolve_raster_family;
    use crate::materials::{MaterialRouter, DEBUG_WORLD_NORMALS_FAMILY_ID, SOLID_COLOR_FAMILY_ID};

    #[test]
    fn unknown_shader_uses_router_fallback() {
        let r = MaterialRouter::new(SOLID_COLOR_FAMILY_ID);
        assert_eq!(resolve_raster_family(999, &r), SOLID_COLOR_FAMILY_ID);
    }

    #[test]
    fn registered_shader_uses_route_family() {
        let mut r = MaterialRouter::new(SOLID_COLOR_FAMILY_ID);
        r.set_shader_family(7, DEBUG_WORLD_NORMALS_FAMILY_ID);
        assert_eq!(resolve_raster_family(7, &r), DEBUG_WORLD_NORMALS_FAMILY_ID);
    }
}
