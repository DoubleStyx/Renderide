//! Forward PBR with host `_MainTex` multiply sourced from `RENDERIDESHADERS`.

pub(crate) const PBR_HOST_ALBEDO_SHADER_SRC: &str =
    include_str!(concat!(env!("OUT_DIR"), "/pbs_metallic_host_albedo.wgsl"));
