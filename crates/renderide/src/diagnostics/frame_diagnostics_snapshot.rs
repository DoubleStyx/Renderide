//! Per-frame diagnostics for the **Frame** debug HUD tab (CPU/GPU timing, allocator, draws).

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::render_graph::WorldMeshDrawStats;

/// Host CPU model and memory usage (from `sysinfo`, refreshed periodically).
#[derive(Clone, Debug, Default)]
pub struct HostCpuMemoryHud {
    /// Reported CPU model name (first logical CPU brand string).
    pub cpu_model: String,
    /// Number of logical CPUs.
    pub logical_cpus: usize,
    /// Global CPU usage percentage (0–100).
    pub cpu_usage_percent: f32,
    /// Installed RAM in bytes.
    pub ram_total_bytes: u64,
    /// Used RAM in bytes (OS-defined).
    pub ram_used_bytes: u64,
}

/// Optional wgpu allocator totals when the backend exposes a report.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuAllocatorHud {
    /// Sum of live allocation sizes from the device allocator report.
    pub allocated_bytes: Option<u64>,
    /// Reserved capacity including internal fragmentation.
    pub reserved_bytes: Option<u64>,
}

/// Snapshot assembled after the render graph runs (draw stats, timings, host metrics).
#[derive(Clone, Debug)]
pub struct FrameDiagnosticsSnapshot {
    /// Inter-frame wall time in ms (winit redraw spacing), used for HUD FPS.
    pub wall_frame_time_ms: f64,
    /// Wall-clock time for [`crate::runtime::RendererRuntime::execute_frame_graph`] (CPU-side graph work).
    pub unified_cpu_frame_ms: f64,
    /// GPU mesh raster timing from timestamp queries — not implemented yet.
    pub gpu_mesh_pass_ms: Option<f64>,
    pub gpu_allocator: GpuAllocatorHud,
    pub host: HostCpuMemoryHud,
    pub mesh_draw: WorldMeshDrawStats,
    /// Host [`FrameSubmitData::render_tasks`] count from the last applied frame submit.
    pub last_submit_render_task_count: usize,
    /// Textures with a registered [`crate::shared::SetTexture2DFormat`] on the backend.
    pub textures_cpu_registered: usize,
    /// GPU-resident textures with at least mip 0 resident (`mip_levels_resident > 0`).
    pub textures_cpu_mip0_ready: usize,
    /// Same as [`crate::diagnostics::RendererInfoSnapshot::resident_texture_count`] — GPU pool entries.
    pub textures_gpu_resident: usize,
    /// Rows in [`crate::resources::MeshPool`] (resident GPU mesh entries).
    pub mesh_pool_entry_count: usize,
    /// Lines describing host shader → material family routes (sorted by shader id).
    pub shader_route_lines: Vec<String>,
}

impl FrameDiagnosticsSnapshot {
    /// Builds the snapshot after [`crate::backend::RenderBackend::execute_frame_graph`] completes.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        gpu: &GpuContext,
        wall_frame_time_ms: f64,
        unified_cpu_frame_ms: f64,
        host: HostCpuMemoryHud,
        last_submit_render_task_count: usize,
        backend: &RenderBackend,
    ) -> Self {
        let (alloc, resv) = gpu.gpu_allocator_bytes();
        let gpu_allocator = GpuAllocatorHud {
            allocated_bytes: alloc,
            reserved_bytes: resv,
        };

        let textures_cpu_registered = backend.texture_format_registration_count();
        let textures_cpu_mip0_ready = backend.texture_mip0_ready_count();
        let textures_gpu_resident = backend.texture_pool().resident_texture_count();
        let mesh_pool_entry_count = backend.mesh_pool().meshes().len();

        let shader_route_lines = backend
            .material_registry()
            .map(|reg| {
                reg.shader_routes_for_hud()
                    .into_iter()
                    .map(|(id, fam)| format!("shader_asset_id={id}  →  family {:?}", fam))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            wall_frame_time_ms,
            unified_cpu_frame_ms,
            gpu_mesh_pass_ms: None,
            gpu_allocator,
            host,
            mesh_draw: backend.last_world_mesh_draw_stats(),
            last_submit_render_task_count,
            textures_cpu_registered,
            textures_cpu_mip0_ready,
            textures_gpu_resident,
            mesh_pool_entry_count,
            shader_route_lines,
        }
    }

    /// FPS from wall-clock interval between redraws (matches legacy HUD semantics).
    pub fn fps_from_wall(&self) -> f64 {
        if self.wall_frame_time_ms <= f64::EPSILON {
            0.0
        } else {
            1000.0 / self.wall_frame_time_ms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrameDiagnosticsSnapshot;

    #[test]
    fn fps_from_wall_matches_inverse_ms() {
        let s = FrameDiagnosticsSnapshot {
            wall_frame_time_ms: 16.0,
            unified_cpu_frame_ms: 2.0,
            gpu_mesh_pass_ms: None,
            gpu_allocator: Default::default(),
            host: Default::default(),
            mesh_draw: Default::default(),
            last_submit_render_task_count: 0,
            textures_cpu_registered: 0,
            textures_cpu_mip0_ready: 0,
            textures_gpu_resident: 0,
            mesh_pool_entry_count: 0,
            shader_route_lines: Vec::new(),
        };
        assert!((s.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let s = FrameDiagnosticsSnapshot {
            wall_frame_time_ms: 0.0,
            unified_cpu_frame_ms: 0.0,
            gpu_mesh_pass_ms: None,
            gpu_allocator: Default::default(),
            host: Default::default(),
            mesh_draw: Default::default(),
            last_submit_render_task_count: 0,
            textures_cpu_registered: 0,
            textures_cpu_mip0_ready: 0,
            textures_gpu_resident: 0,
            mesh_pool_entry_count: 0,
            shader_route_lines: Vec::new(),
        };
        assert_eq!(s.fps_from_wall(), 0.0);
    }
}
