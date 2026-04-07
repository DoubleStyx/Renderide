//! WGSL shader sources for skinned mesh pipelines, sourced from `RENDERIDESHADERS`.

pub(crate) const SKINNED_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/skinned/skinned.wgsl"
));
pub(crate) const SKINNED_MRT_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/skinned/skinned_mrt.wgsl"
));
