//! PBR pipelines that bind a TLAS on scene group 1 (`@binding(4)`) and use `wgpu_ray_query` in
//! the fragment shader for shadow rays.
//!
//! Used only when [`crate::gpu::GpuState::ray_tracing_available`] and a frame TLAS exists; otherwise
//! the standard [`super::pbr::PbrPipeline`] family is used (no acceleration structure binding).
//!
//! [`RenderPipeline::set_mesh_buffers`] and [`RenderPipeline::draw_mesh_indexed`] must be
//! implemented (not left as trait defaults): [`crate::render::pass::mesh_draw::record_non_skinned_draws`]
//! binds VB/IB once per mesh then issues indexed draws separately, including instanced runs.
//!
//! # Structure
//! - [`scene`]: scene bind group layout and creation (TLAS, lights, RT shadow atlas).
//! - [`non_skinned`]: `VertexPosNormal` pipelines, single-target and MRT.
//! - [`skinned`]: skinned draw bind groups and bone pipelines, single-target and MRT.

mod non_skinned;
mod scene;
mod skinned;

pub use non_skinned::{PbrMrtRayQueryPipeline, PbrRayQueryPipeline};
pub use skinned::{SkinnedPbrMrtRayQueryPipeline, SkinnedPbrRayQueryPipeline};

use super::core::{MAX_INSTANCE_RUN, UNIFORM_ALIGNMENT};

const _: () = {
    let _ = MAX_INSTANCE_RUN;
    let _ = UNIFORM_ALIGNMENT;
};
