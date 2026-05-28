//! PhotonDust render-buffer decoding and generated mesh helpers.

/// Bounding-volume helpers for generated particle meshes.
mod bounds;
/// Generated mesh asset id helpers.
mod ids;
/// Point-particle decoding and billboard mesh generation.
mod point;
#[cfg(test)]
/// Particle mesh unit tests.
mod tests;
/// Trail decoding and trail mesh generation.
mod trail;
/// Shared particle asset types and decode helpers.
mod types;
/// Generated mesh upload packing helpers.
pub(crate) mod upload;

pub(crate) use ids::{
    billboard_render_buffer_mesh_asset_id, is_generated_billboard_mesh_asset_id,
    is_generated_particle_mesh_asset_id, point_render_buffer_generated_mesh_ids,
    trail_render_buffer_generated_mesh_ids, trail_render_buffer_mesh_asset_id,
};
pub(crate) use point::{PointRenderBufferBuild, build_point_render_buffer_cpu};
pub(crate) use trail::{TrailRenderBufferBuild, build_trail_render_buffer_cpu};
pub(crate) use types::{
    ParticleRenderBufferError, PointParticle, PointRenderBufferAsset, TrailRenderBufferAsset,
};
pub(crate) use upload::upload_generated_mesh;
