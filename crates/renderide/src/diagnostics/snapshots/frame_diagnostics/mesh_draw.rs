//! Mesh draw stats and resident-pool counts fragment of [`super::FrameDiagnosticsSnapshot`].

use crate::diagnostics::BackendDiagSnapshot;
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

/// Mesh draw / batching / culling stats plus resident pool counts captured for the **Stats** and
/// **Draw state** tabs.
#[derive(Clone, Debug, Default)]
pub struct MeshDrawFragment {
    /// World mesh forward pass draw batching stats for the frame.
    pub stats: WorldMeshDrawStats,
    /// Sorted draw rows with resolved material pipeline state for the **Draw state** tab.
    pub draw_state_rows: Vec<WorldMeshDrawStateRow>,
    /// Host [`crate::shared::FrameSubmitData::render_tasks`] count from the last applied submit.
    pub last_submit_render_task_count: usize,
    /// Camera readback tasks waiting for GPU processing before the next begin-frame send.
    pub pending_camera_readbacks: usize,
    /// Cumulative camera readback tasks successfully written to host shared memory.
    pub completed_camera_readbacks: u64,
    /// Cumulative camera readback tasks failed and zero-filled when possible.
    pub failed_camera_readbacks: u64,
    /// Textures with a registered [`crate::shared::SetTexture2DFormat`] on the backend.
    pub textures_cpu_registered: usize,
    /// GPU-resident textures with at least mip 0 resident (`mip_levels_resident > 0`).
    pub textures_cpu_mip0_ready: usize,
    /// Resident GPU textures in [`crate::gpu_pools::TexturePool`].
    pub textures_gpu_resident: usize,
    /// GPU-resident host render textures ([`crate::gpu_pools::RenderTexturePool`]).
    pub render_textures_gpu_resident: usize,
    /// Rows in [`crate::gpu_pools::MeshPool`] (resident GPU mesh entries).
    pub mesh_pool_entry_count: usize,
}

impl MeshDrawFragment {
    /// Builds the fragment from the backend snapshot plus camera task diagnostics.
    pub fn capture(
        backend: &BackendDiagSnapshot,
        last_submit_render_task_count: usize,
        pending_camera_readbacks: usize,
        completed_camera_readbacks: u64,
        failed_camera_readbacks: u64,
    ) -> Self {
        Self {
            stats: backend.last_world_mesh_draw_stats,
            draw_state_rows: backend.last_world_mesh_draw_state_rows.clone(),
            last_submit_render_task_count,
            pending_camera_readbacks,
            completed_camera_readbacks,
            failed_camera_readbacks,
            textures_cpu_registered: backend.texture_format_registration_count,
            textures_cpu_mip0_ready: backend.texture_mip0_ready_count,
            textures_gpu_resident: backend.texture_pool_resident_count,
            render_textures_gpu_resident: backend.render_texture_pool_len,
            mesh_pool_entry_count: backend.mesh_pool_entry_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::ShaderRouteSnapshot;
    use crate::materials::RasterPipelineKind;

    use super::*;

    #[test]
    fn capture_copies_backend_counts_and_draw_rows() {
        let backend = BackendDiagSnapshot {
            texture_format_registration_count: 2,
            texture_mip0_ready_count: 3,
            texture_pool_resident_count: 4,
            render_texture_pool_len: 5,
            mesh_pool_entry_count: 6,
            shader_routes: vec![ShaderRouteSnapshot {
                shader_asset_id: 10,
                pipeline: RasterPipelineKind::Null,
                shader_asset_name: Some(String::from("fallback.shader")),
                shader_variant_bits: None,
            }],
            last_world_mesh_draw_stats: WorldMeshDrawStats {
                draws_total: 12,
                ..Default::default()
            },
            last_world_mesh_draw_state_rows: Vec::new(),
            material_property_slots: 7,
            property_block_slots: 8,
            material_shader_bindings: 9,
            material_shader_graph: Default::default(),
            material_pipeline_cache: Default::default(),
            frame_graph_pass_count: 10,
            frame_graph_topo_levels: 11,
            gpu_light_count: 12,
            signed_scene_color_active: true,
        };

        let fragment = MeshDrawFragment::capture(&backend, 13, 14, 15, 16);

        assert_eq!(fragment.stats.draws_total, 12);
        assert_eq!(fragment.last_submit_render_task_count, 13);
        assert_eq!(fragment.pending_camera_readbacks, 14);
        assert_eq!(fragment.completed_camera_readbacks, 15);
        assert_eq!(fragment.failed_camera_readbacks, 16);
        assert_eq!(fragment.textures_cpu_registered, 2);
        assert_eq!(fragment.textures_cpu_mip0_ready, 3);
        assert_eq!(fragment.textures_gpu_resident, 4);
        assert_eq!(fragment.render_textures_gpu_resident, 5);
        assert_eq!(fragment.mesh_pool_entry_count, 6);
        assert!(fragment.draw_state_rows.is_empty());
    }
}
