//! WGSL shader source strings for all builtin pipelines, organized by type.
//!
//! Each submodule owns the shaders for one pipeline family:
//! - [`debug`]: normal-debug and UV-debug, both single-target and MRT variants.
//! - [`skinned`]: bone-skinned shader, single-target and MRT.
//! - [`pbr`]: Cook-Torrance PBR, non-skinned and skinned, single-target and MRT.
//! - [`overlay`]: overlay stencil shader with optional rect-clip discard.
//!
//! # Why are WGSL sources embedded as strings?
//! There is currently no offline shader compilation (SPIR-V pre-compilation, naga offline, etc.)
//! or runtime pipeline assembly (composable WGSL modules). Each pipeline variant therefore
//! contains its full WGSL source inline. When a shader preprocessor or `wgsl-import` step is
//! added, these constants can be replaced with compiled artifacts and the per-variant copies of
//! shared code (e.g. the PBR BRDF functions) can be deduplicated.

mod debug;
mod overlay;
mod pbr;
mod skinned;

pub(crate) use debug::{
    NORMAL_DEBUG_MRT_SHADER_SRC, NORMAL_SHADER_SRC, UV_DEBUG_MRT_SHADER_SRC, UV_DEBUG_SHADER_SRC,
};
pub(crate) use overlay::OVERLAY_STENCIL_SHADER_SRC;
pub(crate) use pbr::{
    PBR_MRT_SHADER_SRC, PBR_SHADER_SRC, SKINNED_PBR_MRT_SHADER_SRC, SKINNED_PBR_SHADER_SRC,
};
pub(crate) use skinned::{SKINNED_MRT_SHADER_SRC, SKINNED_SHADER_SRC};
