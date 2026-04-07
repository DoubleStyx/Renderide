//! WGSL shader sources for non-ray-query PBR pipelines, sourced from `RENDERIDESHADERS`.

pub(crate) const PBR_SHADER_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/pbs_metallic.wgsl"));
pub(crate) const PBR_MRT_SHADER_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/pbs_metallic_mrt.wgsl"));
pub(crate) const SKINNED_PBR_SHADER_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/pbs_metallic_skinned.wgsl"));
pub(crate) const SKINNED_PBR_MRT_SHADER_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/pbs_metallic_skinned_mrt.wgsl"));
