//! Plain-data backend snapshot consumed by [`crate::diagnostics::FrameDiagnosticsSnapshot`] and
//! [`crate::diagnostics::RendererInfoSnapshot`].
//!
//! Captured by [`crate::backend::RenderBackend::snapshot_for_diagnostics`] before the diagnostics
//! HUD layer runs, so `diagnostics/` never borrows `&RenderBackend` directly.

use crate::materials::RasterPipelineKind;
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

/// One host-shader -> renderer-pipeline routing row captured for the **Shader routes** HUD tab.
#[derive(Clone, Debug)]
pub struct ShaderRouteSnapshot {
    /// Host-assigned shader asset id.
    pub shader_asset_id: i32,
    /// Resolved renderer pipeline kind for the route.
    pub pipeline: RasterPipelineKind,
    /// Shader asset filename if known.
    pub shader_asset_name: Option<String>,
}

/// Plain-data view of `RenderBackend` fields the diagnostics HUD reads.
///
/// This breaks the diagnostics-to-backend borrow: `diagnostics/` consumes this snapshot rather
/// than borrowing `&RenderBackend` directly, which keeps backend internals private and lets the
/// HUD layer evolve independently of backend visibility.
#[derive(Clone, Debug)]
pub struct BackendDiagSnapshot {
    /// CPU-side host texture format registrations.
    pub texture_format_registration_count: usize,
    /// CPU-side host textures with mip 0 ready.
    pub texture_mip0_ready_count: usize,
    /// Resident GPU textures in the renderer's [`crate::gpu_pools::TexturePool`].
    pub texture_pool_resident_count: usize,
    /// Resident host render textures in the renderer's [`crate::gpu_pools::RenderTexturePool`].
    pub render_texture_pool_len: usize,
    /// Resident mesh entries in the renderer's [`crate::gpu_pools::MeshPool`].
    pub mesh_pool_entry_count: usize,
    /// Host-shader -> pipeline routing rows for the **Shader routes** HUD tab.
    pub shader_routes: Vec<ShaderRouteSnapshot>,
    /// Latest world-mesh draw stats published by the previous frame.
    pub last_world_mesh_draw_stats: WorldMeshDrawStats,
    /// Latest world-mesh draw-state rows published by the previous frame.
    pub last_world_mesh_draw_state_rows: Vec<WorldMeshDrawStateRow>,
    /// Allocated material property uniform slots.
    pub material_property_slots: usize,
    /// Allocated material property block slots.
    pub property_block_slots: usize,
    /// Distinct shader binding sets registered for materials.
    pub material_shader_bindings: usize,
    /// Pass count in the compiled main render graph.
    pub frame_graph_pass_count: usize,
    /// Kahn-style DAG wave count from the compiled main render graph.
    pub frame_graph_topo_levels: usize,
    /// Packed lights after the latest [`crate::backend::RenderBackend::prepare_lights_from_scene`].
    pub gpu_light_count: usize,
}
