//! Per-frame diagnostic snapshot types consumed by the debug HUD and built each frame in [`crate::app`].

use crate::render::pass::MeshDrawPrepStats;

/// Optional wgpu allocator totals when the active backend exposes [`wgpu::Device::generate_allocator_report`].
#[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
#[derive(Clone, Debug, Default)]
pub struct GpuAllocatorSnapshot {
    /// Sum of live allocation sizes reported by the allocator.
    pub allocated_bytes: Option<u64>,
    /// Sum of reserved block capacity including internal fragmentation.
    pub reserved_bytes: Option<u64>,
}

/// Host CPU model string and usage plus system RAM (from sysinfo).
#[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
#[derive(Clone, Debug, Default)]
pub struct HostCpuMemorySnapshot {
    /// Reported CPU model name (first logical CPU’s brand string).
    pub cpu_model: String,
    /// Number of logical CPUs in [`sysinfo::System::cpus`].
    pub logical_cpus: usize,
    /// Global CPU usage percentage (0–100), best after a few refresh cycles.
    pub cpu_usage_percent: f32,
    /// Installed RAM in bytes.
    pub ram_total_bytes: u64,
    /// Currently used RAM in bytes (platform-defined; excludes caches where the OS reports them separately).
    pub ram_used_bytes: u64,
}

/// Per-frame diagnostics sample shown in the debug HUD.
#[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct LiveFrameDiagnostics {
    pub frame_index: i32,
    pub viewport: (u32, u32),

    // ── CPU phase timings ────────────────────────────────────────────────────
    pub session_update_us: u64,
    /// IPC batch collection: `MainViewFrameInput::from_session`.
    pub ipc_collect_us: u64,
    /// Mesh-draw culling + GPU buffer upload: `prepare_mesh_draws_for_view`.
    pub mesh_prep_us: u64,
    /// `ipc_collect_us + mesh_prep_us` (sum retained for external consumers and log diagnostics).
    #[allow(dead_code)]
    pub collect_us: u64,
    /// `render_loop.render_frame` wall time (TLAS build + all pass recording + submit).
    pub render_us: u64,
    pub present_us: u64,
    pub total_us: u64,
    /// Wall-clock microseconds since the previous `run_frame()` call (includes sleep time).
    /// Use this for actual FPS; `total_us` only measures active work per call.
    pub wall_interval_us: u64,

    // ── GPU timing ───────────────────────────────────────────────────────────
    /// GPU mesh rasterisation pass time (timestamp query, updated every 60 frames).
    pub gpu_mesh_pass_ms: Option<f64>,

    // ── Draw stats ───────────────────────────────────────────────────────────
    pub batch_count: usize,
    pub overlay_batch_count: usize,
    pub total_draws_in_batches: usize,
    pub overlay_draws_in_batches: usize,
    pub prep_stats: MeshDrawPrepStats,
    pub mesh_cache_count: usize,
    pub pending_render_tasks: usize,
    pub pending_camera_task_readbacks: usize,

    // ── Textures (2D) ───────────────────────────────────────────────────────
    /// Host-registered Texture2D rows in [`crate::assets::AssetRegistry`].
    pub textures_cpu_registered: usize,
    /// Same registry entries with mip0 decoded and sized for GPU upload.
    pub textures_cpu_ready_for_gpu: usize,
    /// Entries in [`crate::gpu::GpuState::texture2d_gpu`] (uploaded wgpu textures).
    pub textures_gpu_resident: usize,

    // ── Lights ───────────────────────────────────────────────────────────────
    /// Active light count uploaded to the GPU by the clustered light pass.
    pub gpu_light_count: u32,

    // ── Ray tracing / RTAO ───────────────────────────────────────────────────
    /// Number of meshes with a built BLAS (acceleration structure).
    pub blas_count: usize,
    /// Whether a TLAS was successfully built for this frame.
    pub tlas_available: bool,
    /// `ao_radius` from render config (world-space AO ray length).
    pub ao_radius: f32,
    /// `rtao_strength` from render config (AO multiplier applied in composite).
    pub ao_strength: f32,
    /// Fixed sample count used by the RTAO compute shader this build.
    pub ao_sample_count: u32,

    // ── Feature flags ────────────────────────────────────────────────────────
    pub frustum_culling_enabled: bool,
    pub rtao_enabled: bool,
    /// [`crate::config::RenderConfig::ray_traced_shadows_enabled`]: request PBR ray-query shadows.
    pub ray_traced_shadows_enabled: bool,
    pub ray_tracing_available: bool,
    /// GPU adapter metadata from wgpu (name, device class, driver, backend).
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub adapter_info: wgpu::AdapterInfo,
    /// Process GPU memory tracked by wgpu’s native allocator, when available.
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub gpu_allocator: GpuAllocatorSnapshot,
    /// Host CPU/RAM snapshot for the HUD.
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub host: HostCpuMemorySnapshot,
    /// Native UI strangler routing counters (last frame) when the feature is enabled in config.
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub native_ui_routing_metrics:
        Option<crate::session::native_ui_routing_metrics::NativeUiRoutingFrameMetrics>,
    /// Material batch wire opcode counts (last frame) when [`crate::config::RenderConfig::material_batch_wire_metrics`] is on.
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub material_batch_wire_metrics:
        Option<crate::assets::material_batch_wire_metrics::MaterialBatchWireFrameMetrics>,
    /// Top per-frame shader-route summaries shown in the HUD.
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub shader_route_lines: Vec<String>,
    /// Per-frame fallback summaries for host shaders that had no native program match.
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub shader_fallback_lines: Vec<String>,
    /// Per-frame material/shader binding warnings (missing property ids, missing textures, etc.).
    #[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
    pub shader_warning_lines: Vec<String>,
}

#[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
impl LiveFrameDiagnostics {
    pub(crate) fn frame_time_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }

    pub(crate) fn fps(&self) -> f64 {
        if self.wall_interval_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.wall_interval_us as f64
        }
    }

    pub(crate) fn bottleneck(&self) -> &'static str {
        match self.gpu_mesh_pass_ms {
            Some(gpu_ms) if gpu_ms > self.frame_time_ms() => "GPU",
            Some(_) => "CPU",
            None => "CPU?",
        }
    }

    pub(crate) fn submitted_overlay_draws(&self) -> usize {
        self.prep_stats
            .submitted_draws()
            .min(self.total_draws_in_batches)
            .saturating_sub(self.submitted_main_draws())
    }

    pub(crate) fn submitted_main_draws(&self) -> usize {
        let main_draws = self
            .total_draws_in_batches
            .saturating_sub(self.overlay_draws_in_batches);
        self.prep_stats.submitted_draws().min(main_draws)
    }
}

#[cfg(test)]
fn test_adapter_info() -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: "Test GPU".into(),
        vendor: 0x10de,
        device: 0x2800,
        device_type: wgpu::DeviceType::DiscreteGpu,
        device_pci_bus_id: String::new(),
        driver: "unit-test".into(),
        driver_info: "0".into(),
        backend: wgpu::Backend::Vulkan,
        subgroup_min_size: wgpu::MINIMUM_SUBGROUP_MIN_SIZE,
        subgroup_max_size: wgpu::MAXIMUM_SUBGROUP_MAX_SIZE,
        transient_saves_memory: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diag(total_us: u64, gpu_ms: Option<f64>) -> LiveFrameDiagnostics {
        LiveFrameDiagnostics {
            frame_index: 12,
            viewport: (1280, 720),
            session_update_us: 1_000,
            ipc_collect_us: 500,
            mesh_prep_us: 1_500,
            collect_us: 2_000,
            render_us: 3_000,
            present_us: 500,
            total_us,
            wall_interval_us: total_us,
            gpu_mesh_pass_ms: gpu_ms,
            batch_count: 4,
            overlay_batch_count: 1,
            total_draws_in_batches: 20,
            overlay_draws_in_batches: 5,
            prep_stats: MeshDrawPrepStats {
                rigid_input_draws: 12,
                skinned_input_draws: 8,
                submitted_rigid_draws: 10,
                submitted_skinned_draws: 8,
                ..MeshDrawPrepStats::default()
            },
            mesh_cache_count: 10,
            pending_render_tasks: 0,
            pending_camera_task_readbacks: 0,
            textures_cpu_registered: 3,
            textures_cpu_ready_for_gpu: 2,
            textures_gpu_resident: 2,
            gpu_light_count: 4,
            blas_count: 10,
            tlas_available: true,
            ao_radius: 1.5,
            ao_strength: 0.85,
            ao_sample_count: 8,
            frustum_culling_enabled: true,
            rtao_enabled: true,
            ray_traced_shadows_enabled: false,
            ray_tracing_available: true,
            adapter_info: test_adapter_info(),
            gpu_allocator: GpuAllocatorSnapshot::default(),
            host: HostCpuMemorySnapshot::default(),
            native_ui_routing_metrics: None,
            material_batch_wire_metrics: None,
            shader_route_lines: Vec::new(),
            shader_fallback_lines: Vec::new(),
            shader_warning_lines: Vec::new(),
        }
    }

    #[test]
    fn bottleneck_prefers_gpu_when_gpu_time_exceeds_cpu_frame_time() {
        assert_eq!(make_diag(4_000, Some(8.0)).bottleneck(), "GPU");
        assert_eq!(make_diag(12_000, Some(4.0)).bottleneck(), "CPU");
    }

    #[test]
    fn submitted_overlay_draws_never_underflow() {
        let s = make_diag(10_000, None);
        assert_eq!(s.submitted_main_draws(), 15);
        assert_eq!(s.submitted_overlay_draws(), 3);
    }
}
