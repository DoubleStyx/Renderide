//! Host shader asset → [`RasterPipelineKind`] for mesh draws.

use super::pipeline_kind::RasterPipelineKind;
use super::router::MaterialRouter;

/// Resolves the raster pipeline kind used for **mesh rasterization** for a host shader asset id.
///
/// Uses [`MaterialRouter::pipeline_for_shader_asset`], populated when the host sends
/// [`crate::shared::ShaderUpload`] (see [`crate::assets::shader::resolve_shader_upload`]).
pub fn resolve_raster_pipeline(
    shader_asset_id: i32,
    router: &MaterialRouter,
) -> RasterPipelineKind {
    router.pipeline_for_shader_asset(shader_asset_id)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::resolve_raster_pipeline;
    use crate::materials::{MaterialRouter, RasterPipelineKind};

    const FALLBACK: RasterPipelineKind = RasterPipelineKind::DebugWorldNormals;

    #[test]
    fn unknown_shader_uses_router_fallback() {
        let r = MaterialRouter::new(FALLBACK);
        assert_eq!(resolve_raster_pipeline(999, &r), FALLBACK);
    }

    #[test]
    fn registered_shader_uses_route_pipeline() {
        let mut r = MaterialRouter::new(FALLBACK);
        let route = RasterPipelineKind::EmbeddedStem(Arc::from("test_embedded_default"));
        r.set_shader_pipeline(7, route.clone());
        assert_eq!(resolve_raster_pipeline(7, &r), route);
    }
}
