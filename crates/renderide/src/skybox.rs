//! Skybox rendering: unified IBL prefilter cache, sky evaluator params, and active-main resolution.

pub(crate) mod ibl_cache;
pub(crate) mod params;
mod prepared;
pub(crate) mod specular;

pub use prepared::{
    PreparedClearColorSkybox, PreparedMaterialSkybox, PreparedMaterialSkyboxGeometry,
    PreparedMaterialSkyboxMesh, PreparedSkybox,
};
