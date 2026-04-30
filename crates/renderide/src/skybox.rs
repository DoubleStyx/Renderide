//! Skybox rendering: environment cube cache, IBL specular params, and active-main resolution.

mod environment;
pub(crate) mod params;
mod prepared;
mod specular;

pub(crate) use environment::SkyboxEnvironmentCache;
pub use prepared::{PreparedClearColorSkybox, PreparedMaterialSkybox, PreparedSkybox};
pub(crate) use specular::resolve_active_main_skybox_specular_environment;
