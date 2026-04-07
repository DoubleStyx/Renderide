//! Raster pipeline family builders (mesh materials, UI, etc.).

mod debug_world_normals;

pub use debug_world_normals::{
    DebugWorldNormalsFamily, DEBUG_WORLD_NORMALS_FAMILY_ID, SHADER_PERM_MULTIVIEW_STEREO,
};
