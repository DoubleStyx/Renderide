//! PBS / PBR **per-material** bind-group helpers (strangler seam).
//!
//! [`crate::gpu::pipeline::PbrHostAlbedoPipeline`] binds host `_MainTex` in a small bind group
//! (group 0, dynamic slot for albedo texture). Cache keys use `(material_asset_id, texture2d_asset_id)`
//! so materials stay addressable independently even when they share a texture (future sampler /
//! scale divergence).

/// Cache key for [`crate::gpu::state::GpuState::pbr_host_albedo_bind_cache`]: host material id and
/// bound `Texture2D` asset id.
pub type PbrHostAlbedoMaterialBindKey = (i32, i32);
