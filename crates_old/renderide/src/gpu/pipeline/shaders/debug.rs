//! WGSL shader sources for debug visualization pipelines, sourced from `RENDERIDESHADERS`.

pub(crate) const NORMAL_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/debug/normal_debug.wgsl"
));
pub(crate) const UV_DEBUG_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/debug/uv_debug.wgsl"
));
pub(crate) const NORMAL_DEBUG_MRT_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/debug/normal_debug_mrt.wgsl"
));
pub(crate) const UV_DEBUG_MRT_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/debug/uv_debug_mrt.wgsl"
));
