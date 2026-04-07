//! WGSL shader source strings for all builtin pipelines, organized by type.
//!
//! Each submodule owns the shaders for one pipeline family:
//! - [`debug`]: normal-debug and UV-debug, both single-target and MRT variants.
//! - [`skinned`]: bone-skinned shader, single-target and MRT.
//! - [`pbr`]: Cook-Torrance PBR, non-skinned and skinned, single-target and MRT.
//! - [`pbr_ray_query`]: same PBR families with `wgpu_ray_query` and TLAS for RT shadows.
//! - [`overlay`]: overlay stencil shader with optional rect-clip discard.
//!
//! World/UI native shaders are built from `RENDERIDESHADERS/` with **naga_oil** in `build.rs`.
//! Debug and overlay variants stay inline until migrated.

mod debug;
mod overlay;
mod pbr;
mod pbr_host_albedo;
mod pbr_ray_query;
mod skinned;

pub(crate) use debug::{
    NORMAL_DEBUG_MRT_SHADER_SRC, NORMAL_SHADER_SRC, UV_DEBUG_MRT_SHADER_SRC, UV_DEBUG_SHADER_SRC,
};
pub(crate) use overlay::OVERLAY_STENCIL_SHADER_SRC;
pub(crate) use pbr::{
    PBR_MRT_SHADER_SRC, PBR_SHADER_SRC, SKINNED_PBR_MRT_SHADER_SRC, SKINNED_PBR_SHADER_SRC,
};
pub(crate) use pbr_host_albedo::PBR_HOST_ALBEDO_SHADER_SRC;
pub(crate) use pbr_ray_query::{
    PBR_MRT_RAY_QUERY_SHADER_SRC, PBR_RAY_QUERY_SHADER_SRC, SKINNED_PBR_MRT_RAY_QUERY_SHADER_SRC,
    SKINNED_PBR_RAY_QUERY_SHADER_SRC,
};
pub(crate) use skinned::{SKINNED_MRT_SHADER_SRC, SKINNED_SHADER_SRC};
