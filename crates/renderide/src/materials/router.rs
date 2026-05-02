//! Maps host shader asset ids from `set_shader` to [`RasterPipelineKind`].
//!
//! Populated from [`crate::assets::shader::resolve_shader_upload`] when the host sends
//! [`crate::shared::ShaderUpload`]. Unknown ids use [`MaterialRouter::fallback`].

use std::collections::HashMap;

use super::RasterPipelineKind;

/// Host shader route: raster pipeline kind plus optional AssetBundle shader asset name for the debug HUD.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderRouteEntry {
    /// Pipeline kind for this host shader asset id.
    pub pipeline: RasterPipelineKind,
    /// Shader asset filename or stem from the AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
}

/// Shader asset id -> route; unknown ids use the fallback pipeline set via [`Self::new`] /
/// [`Self::set_fallback`].
#[derive(Debug)]
pub struct MaterialRouter {
    routes: HashMap<i32, ShaderRouteEntry>,
    /// Optional composed WGSL stem (`shaders/target/<stem>.wgsl`) when an embedded `{key}_default` target exists.
    shader_stem: HashMap<i32, String>,
    /// Default when `routes` has no entry.
    fallback: RasterPipelineKind,
    /// Monotonic counter bumped on every mutation; read by
    /// [`crate::world_mesh::FrameMaterialBatchCache`]
    /// to invalidate resolved entries when any route, stem, or fallback changes.
    generation: u64,
}

impl MaterialRouter {
    /// Builds a router with only a fallback pipeline kind.
    pub fn new(fallback: RasterPipelineKind) -> Self {
        Self {
            routes: HashMap::new(),
            shader_stem: HashMap::new(),
            fallback,
            generation: 0,
        }
    }

    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Monotonic generation counter bumped on any route / stem / fallback mutation.
    ///
    /// Persistent resolved-material caches compare a snapshot of this value against the current
    /// value to detect when a re-resolve is required.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Replaces the fallback pipeline kind used when no route is registered for a shader id.
    pub fn set_fallback(&mut self, fallback: RasterPipelineKind) {
        self.fallback = fallback;
        self.bump();
    }

    /// Returns the current fallback pipeline kind.
    pub fn fallback(&self) -> &RasterPipelineKind {
        &self.fallback
    }

    /// Inserts or replaces a host shader route (pipeline kind and optional shader asset name).
    pub fn set_shader_route(
        &mut self,
        shader_asset_id: i32,
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
    ) {
        self.routes.insert(
            shader_asset_id,
            ShaderRouteEntry {
                pipeline,
                shader_asset_name,
            },
        );
        self.bump();
    }

    /// Inserts a host shader -> pipeline mapping with no AssetBundle shader asset name.
    pub fn set_shader_pipeline(&mut self, shader_asset_id: i32, pipeline: RasterPipelineKind) {
        self.set_shader_route(shader_asset_id, pipeline, None);
    }

    /// Resolves the pipeline kind for a host shader asset id.
    pub fn pipeline_for_shader_asset(&self, shader_asset_id: i32) -> RasterPipelineKind {
        self.routes
            .get(&shader_asset_id)
            .map_or_else(|| self.fallback.clone(), |e| e.pipeline.clone())
    }

    /// Records a target WGSL stem for `shader_asset_id` (from embedded Unity name resolution).
    pub fn set_shader_stem(&mut self, shader_asset_id: i32, stem: String) {
        self.shader_stem.insert(shader_asset_id, stem);
        self.bump();
    }

    /// Clears [`Self::stem_for_shader_asset`] for `shader_asset_id`.
    pub fn remove_shader_stem(&mut self, shader_asset_id: i32) {
        if self.shader_stem.remove(&shader_asset_id).is_some() {
            self.bump();
        }
    }

    /// Composed material stem when the host shader name matched an embedded target.
    pub fn stem_for_shader_asset(&self, shader_asset_id: i32) -> Option<&str> {
        self.shader_stem.get(&shader_asset_id).map(String::as_str)
    }

    /// Drops a host shader id mapping after [`crate::shared::ShaderUnload`].
    pub fn remove_shader_route(&mut self, shader_asset_id: i32) {
        let had_route = self.routes.remove(&shader_asset_id).is_some();
        let had_stem = self.shader_stem.remove(&shader_asset_id).is_some();
        if had_route || had_stem {
            self.bump();
        }
    }

    /// Returns the mapped pipeline kind when the host id was registered via [`Self::set_shader_route`].
    pub fn get_shader_pipeline(&self, shader_asset_id: i32) -> Option<RasterPipelineKind> {
        self.routes
            .get(&shader_asset_id)
            .map(|e| e.pipeline.clone())
    }

    /// Host shader asset ids, pipeline kinds, and optional AssetBundle shader asset names, sorted by id.
    pub fn routes_sorted_for_hud(&self) -> Vec<(i32, RasterPipelineKind, Option<String>)> {
        let mut v: Vec<_> = self
            .routes
            .iter()
            .map(|(&k, e)| (k, e.pipeline.clone(), e.shader_asset_name.clone()))
            .collect();
        v.sort_by_key(|(k, _, _)| *k);
        v
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::MaterialRouter;
    use crate::materials::RasterPipelineKind;

    /// Arbitrary embedded stem for router tests.
    fn test_route_pipeline() -> RasterPipelineKind {
        RasterPipelineKind::EmbeddedStem(Arc::from("test_route_default"))
    }

    #[test]
    fn remove_shader_route_clears_entry() {
        let mut r = MaterialRouter::new(RasterPipelineKind::Null);
        r.set_shader_pipeline(7, test_route_pipeline());
        assert_eq!(r.get_shader_pipeline(7), Some(test_route_pipeline()));
        r.remove_shader_route(7);
        assert_eq!(r.get_shader_pipeline(7), None);
        assert_eq!(r.pipeline_for_shader_asset(7), RasterPipelineKind::Null);
    }

    #[test]
    fn remove_shader_route_clears_stem() {
        let mut r = MaterialRouter::new(RasterPipelineKind::Null);
        r.set_shader_route(1, test_route_pipeline(), Some("x".to_string()));
        r.set_shader_stem(1, "null_default".to_string());
        assert_eq!(r.stem_for_shader_asset(1), Some("null_default"));
        r.remove_shader_route(1);
        assert_eq!(r.stem_for_shader_asset(1), None);
    }

    #[test]
    fn set_shader_route_stores_shader_asset_name_for_hud() {
        let mut r = MaterialRouter::new(RasterPipelineKind::Null);
        r.set_shader_route(3, test_route_pipeline(), Some("ExampleShader".to_string()));
        assert_eq!(
            r.routes_sorted_for_hud(),
            vec![(3, test_route_pipeline(), Some("ExampleShader".to_string()))]
        );
        assert_eq!(r.get_shader_pipeline(3), Some(test_route_pipeline()));
    }

    #[test]
    fn routes_sorted_for_hud_sorted_by_id() {
        let mut r = MaterialRouter::new(RasterPipelineKind::Null);
        r.set_shader_route(10, test_route_pipeline(), None);
        r.set_shader_route(2, test_route_pipeline(), Some("a".to_string()));
        assert_eq!(
            r.routes_sorted_for_hud()
                .into_iter()
                .map(|(id, _, _)| id)
                .collect::<Vec<_>>(),
            vec![2, 10]
        );
    }

    #[test]
    fn generation_bumps_on_mutations() {
        let mut r = MaterialRouter::new(RasterPipelineKind::Null);
        let g0 = r.generation();
        r.set_shader_pipeline(1, test_route_pipeline());
        let g1 = r.generation();
        assert_ne!(g0, g1);
        // Reads don't bump.
        let _ = r.pipeline_for_shader_asset(1);
        assert_eq!(r.generation(), g1);
        r.set_shader_stem(1, "foo_default".to_string());
        let g2 = r.generation();
        assert_ne!(g1, g2);
        r.remove_shader_route(1);
        let g3 = r.generation();
        assert_ne!(g2, g3);
        r.set_fallback(RasterPipelineKind::Null);
        assert_ne!(r.generation(), g3);
    }

    #[test]
    fn remove_without_effect_does_not_bump() {
        let mut r = MaterialRouter::new(RasterPipelineKind::Null);
        r.set_shader_pipeline(1, test_route_pipeline());
        let g = r.generation();
        r.remove_shader_stem(999);
        r.remove_shader_route(999);
        assert_eq!(r.generation(), g);
    }
}
