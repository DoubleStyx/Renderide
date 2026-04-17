//! Per-frame diagnostics for the **Frame** debug HUD tab (CPU/GPU timing, allocator, draws)
//! and the **GPU memory** tab (throttled full [`wgpu::AllocatorReport`]).

use std::sync::Arc;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::materials::RasterPipelineKind;
use crate::render_graph::{WorldMeshDrawStateRow, WorldMeshDrawStats};

/// One row in the **Shader routes** tab: identifies the host shader, its backing pipeline, and
/// whether the renderer has a real embedded shader for it or falls back to `debug_world_normals`.
#[derive(Clone, Debug)]
pub struct ShaderRouteRow {
    /// Host-assigned shader asset id.
    pub shader_asset_id: i32,
    /// Logical shader name when known (ShaderLab name, WGSL banner, or upload field).
    pub display_name: Option<String>,
    /// Human-readable pipeline label (composed stem, or `debug_world_normals`).
    pub pipeline_label: String,
    /// True when the route resolved to a real embedded shader; false when it fell back to debug.
    pub implemented: bool,
}

/// Full GPU allocator report plus sort order for the **GPU memory** HUD tab.
///
/// The report is refreshed on a timer in [`crate::runtime::RendererRuntime`] (not every frame);
/// [`GpuAllocatorHud`] totals on the **Stats** tab are still sampled each capture via
/// [`GpuContext::gpu_allocator_bytes`].
#[derive(Clone, Debug)]
pub struct GpuAllocatorReportHud {
    /// Live [`wgpu::Device::generate_allocator_report`] payload when the backend supports it.
    pub report: Arc<wgpu::AllocatorReport>,
    /// Indices into [`wgpu::AllocatorReport::allocations`], sorted by descending allocation size.
    pub allocation_indices_by_size: Arc<[usize]>,
}

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

/// Snapshot assembled after the winit frame tick ends (draw stats, timings, host metrics).
#[derive(Clone, Debug)]
pub struct FrameDiagnosticsSnapshot {
    /// Wall-clock time between consecutive redraw ticks (ms): **total frame time** for pacing; FPS = `1000.0 / wall_frame_time_ms`.
    pub wall_frame_time_ms: f64,
    /// Wall-clock from the start of the winit frame tick to the last tracked `Queue::submit` (ms).
    pub cpu_frame_until_submit_ms: Option<f64>,
    /// Wall-clock from submit to GPU idle for the **most recently completed** tracked submission (ms).
    ///
    /// May lag the current frame; matches the Frame timing HUD GPU line.
    pub gpu_frame_after_submit_ms: Option<f64>,
    /// Optional wgpu allocator byte totals when exposed by the device.
    pub gpu_allocator: GpuAllocatorHud,
    /// Throttled full allocator report for the **GPU memory** tab (`None` if unsupported or before first refresh).
    ///
    /// Allocation names reflect wgpu resource `label` values where set.
    pub gpu_allocator_report: Option<GpuAllocatorReportHud>,
    /// Seconds until [`crate::runtime::RendererRuntime`] replaces [`Self::gpu_allocator_report`] on the next capture.
    ///
    /// **Stats** tab totals ([`GpuAllocatorHud`]) are still updated every capture via [`GpuContext::gpu_allocator_bytes`];
    /// this field only governs the heavy full report used by the **GPU memory** tab.
    pub gpu_allocator_report_next_refresh_in_secs: f32,
    /// Host CPU model and memory usage for the HUD.
    pub host: HostCpuMemoryHud,
    /// World mesh forward pass draw batching stats for the frame.
    pub mesh_draw: WorldMeshDrawStats,
    /// Sorted draw rows with resolved material pipeline state for the **Draw state** tab.
    pub draw_state_rows: Vec<WorldMeshDrawStateRow>,
    /// Host [`FrameSubmitData::render_tasks`] count from the last applied frame submit.
    pub last_submit_render_task_count: usize,
    /// Textures with a registered [`crate::shared::SetTexture2DFormat`] on the backend.
    pub textures_cpu_registered: usize,
    /// GPU-resident textures with at least mip 0 resident (`mip_levels_resident > 0`).
    pub textures_cpu_mip0_ready: usize,
    /// Same as [`crate::diagnostics::RendererInfoSnapshot::resident_texture_count`] — GPU pool entries.
    pub textures_gpu_resident: usize,
    /// GPU-resident host render textures ([`crate::resources::RenderTexturePool`]).
    pub render_textures_gpu_resident: usize,
    /// Rows in [`crate::resources::MeshPool`] (resident GPU mesh entries).
    pub mesh_pool_entry_count: usize,
    /// Host shader routes (id, name, pipeline, implemented flag), sorted implemented-first then by id.
    pub shader_routes: Vec<ShaderRouteRow>,
}

impl FrameDiagnosticsSnapshot {
    /// Builds the snapshot after [`crate::gpu::GpuContext::end_frame_timing`] for the tick.
    pub fn capture(
        gpu: &GpuContext,
        wall_frame_time_ms: f64,
        host: HostCpuMemoryHud,
        last_submit_render_task_count: usize,
        backend: &RenderBackend,
        gpu_allocator_report: Option<GpuAllocatorReportHud>,
        gpu_allocator_report_next_refresh_in_secs: f32,
    ) -> Self {
        let (cpu_frame_until_submit_ms, gpu_frame_after_submit_ms) = gpu.frame_cpu_gpu_ms_for_hud();
        let (alloc, resv) = gpu.gpu_allocator_bytes();
        let gpu_allocator = GpuAllocatorHud {
            allocated_bytes: alloc,
            reserved_bytes: resv,
        };

        let textures_cpu_registered = backend.texture_format_registration_count();
        let textures_cpu_mip0_ready = backend.texture_mip0_ready_count();
        let textures_gpu_resident = backend.texture_pool().resident_texture_count();
        let render_textures_gpu_resident = backend.render_texture_pool().len();
        let mesh_pool_entry_count = backend.mesh_pool().meshes().len();

        let mut shader_routes: Vec<ShaderRouteRow> = backend
            .material_registry()
            .map(|reg| {
                reg.shader_routes_for_hud()
                    .into_iter()
                    .map(|(id, pipeline, name)| {
                        let implemented =
                            !matches!(pipeline, RasterPipelineKind::DebugWorldNormals);
                        let pipeline_label = match &pipeline {
                            RasterPipelineKind::EmbeddedStem(stem) => stem.to_string(),
                            RasterPipelineKind::DebugWorldNormals => {
                                "debug_world_normals".to_string()
                            }
                        };
                        ShaderRouteRow {
                            shader_asset_id: id,
                            display_name: name,
                            pipeline_label,
                            implemented,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Implemented routes first, then fallbacks; preserve id-ascending order within each group.
        shader_routes.sort_by(|a, b| {
            b.implemented
                .cmp(&a.implemented)
                .then(a.shader_asset_id.cmp(&b.shader_asset_id))
        });

        Self {
            wall_frame_time_ms,
            cpu_frame_until_submit_ms,
            gpu_frame_after_submit_ms,
            gpu_allocator,
            gpu_allocator_report,
            gpu_allocator_report_next_refresh_in_secs,
            host,
            mesh_draw: backend.last_world_mesh_draw_stats(),
            draw_state_rows: backend.last_world_mesh_draw_state_rows(),
            last_submit_render_task_count,
            textures_cpu_registered,
            textures_cpu_mip0_ready,
            textures_gpu_resident,
            render_textures_gpu_resident,
            mesh_pool_entry_count,
            shader_routes,
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
            cpu_frame_until_submit_ms: Some(2.0),
            gpu_frame_after_submit_ms: Some(1.0),
            gpu_allocator: Default::default(),
            gpu_allocator_report: None,
            gpu_allocator_report_next_refresh_in_secs: 0.0,
            host: Default::default(),
            mesh_draw: Default::default(),
            draw_state_rows: Vec::new(),
            last_submit_render_task_count: 0,
            textures_cpu_registered: 0,
            textures_cpu_mip0_ready: 0,
            textures_gpu_resident: 0,
            render_textures_gpu_resident: 0,
            mesh_pool_entry_count: 0,
            shader_routes: Vec::new(),
        };
        assert!((s.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let s = FrameDiagnosticsSnapshot {
            wall_frame_time_ms: 0.0,
            cpu_frame_until_submit_ms: None,
            gpu_frame_after_submit_ms: None,
            gpu_allocator: Default::default(),
            gpu_allocator_report: None,
            gpu_allocator_report_next_refresh_in_secs: 0.0,
            host: Default::default(),
            mesh_draw: Default::default(),
            draw_state_rows: Vec::new(),
            last_submit_render_task_count: 0,
            textures_cpu_registered: 0,
            textures_cpu_mip0_ready: 0,
            textures_gpu_resident: 0,
            render_textures_gpu_resident: 0,
            mesh_pool_entry_count: 0,
            shader_routes: Vec::new(),
        };
        assert_eq!(s.fps_from_wall(), 0.0);
    }
}
