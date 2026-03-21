//! Diagnostic helpers and optional ImGui on-screen HUD.
//!
//! The HUD is enabled by the `debug-hud` Cargo feature (on by default). Disable default features
//! (`cargo build -p renderide --no-default-features`) for lean builds without `imgui` / `imgui-wgpu`.

use std::time::{Duration, Instant};

use crate::render::pass::MeshDrawPrepStats;

#[cfg(feature = "debug-hud")]
use crate::input::WindowInputState;
#[cfg(feature = "debug-hud")]
use crate::render::RenderTarget;

#[cfg(feature = "debug-hud")]
use imgui::{
    Condition, Context, FontConfig, FontSource, Io, MouseButton as ImGuiMouseButton, WindowFlags,
};
#[cfg(feature = "debug-hud")]
use imgui_wgpu::{Renderer, RendererConfig};

// ── ImGui HUD ────────────────────────────────────────────────────────────────

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
}

#[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
impl LiveFrameDiagnostics {
    fn frame_time_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }

    fn fps(&self) -> f64 {
        if self.wall_interval_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.wall_interval_us as f64
        }
    }

    fn bottleneck(&self) -> &'static str {
        match self.gpu_mesh_pass_ms {
            Some(gpu_ms) if gpu_ms > self.frame_time_ms() => "GPU",
            Some(_) => "CPU",
            None => "CPU?",
        }
    }

    fn submitted_overlay_draws(&self) -> usize {
        self.prep_stats
            .submitted_draws()
            .min(self.total_draws_in_batches)
            .saturating_sub(self.submitted_main_draws())
    }

    fn submitted_main_draws(&self) -> usize {
        let main_draws = self
            .total_draws_in_batches
            .saturating_sub(self.overlay_draws_in_batches);
        self.prep_stats.submitted_draws().min(main_draws)
    }
}

/// Maps [`wgpu::DeviceType`] to a short phrase for the HUD.
#[cfg(feature = "debug-hud")]
fn device_type_label(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::Other => "other / unknown",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software / CPU",
    }
}

/// Feeds the current winit-derived pointer state into Dear ImGui before `Context::frame`.
#[cfg(feature = "debug-hud")]
fn apply_window_input_to_imgui(io: &mut Io, input: &WindowInputState) {
    if input.mouse_active && input.window_focused {
        io.add_mouse_pos_event([input.window_position.x, input.window_position.y]);
    } else {
        io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
    }
    io.add_mouse_button_event(ImGuiMouseButton::Left, input.left_held);
    io.add_mouse_button_event(ImGuiMouseButton::Right, input.right_held);
    io.add_mouse_button_event(ImGuiMouseButton::Middle, input.middle_held);
    io.add_mouse_button_event(ImGuiMouseButton::Extra1, input.button4_held);
    io.add_mouse_button_event(ImGuiMouseButton::Extra2, input.button5_held);
}

/// Right-aligned numeric [`format!`] helpers so HUD numeric columns keep a stable width.
#[cfg(feature = "debug-hud")]
mod hud_fmt {
    /// Formats `value` as a right-aligned decimal with `decimals` places and total width `width`.
    pub fn f64_field(width: usize, decimals: usize, value: f64) -> String {
        format!("{value:>w$.d$}", w = width, d = decimals)
    }

    /// Formats `value` as a right-aligned float with `decimals` places and total width `width`.
    pub fn f32_field(width: usize, decimals: usize, value: f32) -> String {
        f64_field(width, decimals, f64::from(value))
    }

    /// Converts microseconds to milliseconds with fixed width.
    pub fn ms_from_us(width: usize, decimals: usize, micros: u64) -> String {
        f64_field(width, decimals, micros as f64 / 1000.0)
    }

    /// Human-readable gibibytes from bytes (numeric part only; caller adds `GiB` suffix).
    pub fn gib_value(width: usize, decimals: usize, bytes: u64) -> String {
        let g = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        f64_field(width, decimals, g)
    }
}

/// ImGui-backed on-screen diagnostics panel when the `debug-hud` feature is enabled; otherwise a
/// zero-cost stub (see [`Self::new`], [`Self::render`], [`Self::update`]).
///
/// The panel is a normal titled window: drag the title bar to move it, and use the title-bar
/// collapse control to fold the content away. Initial placement is top-right on first use each run.
/// Pointer state is taken from [`crate::input::WindowInputState`] each frame so ImGui can hit-test the window.
#[cfg(feature = "debug-hud")]
pub struct DebugHud {
    imgui: Context,
    renderer: Renderer,
    last_frame_at: Instant,
    latest: Option<LiveFrameDiagnostics>,
}

#[cfg(feature = "debug-hud")]
impl DebugHud {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);
        imgui.set_log_filename(None);
        imgui.io_mut().config_windows_move_from_title_bar_only = true;
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                oversample_h: 2,
                pixel_snap_h: true,
                size_pixels: 14.0,
                ..FontConfig::default()
            }),
        }]);

        let renderer_config = RendererConfig {
            texture_format: surface_format,
            ..RendererConfig::default()
        };
        let renderer = Renderer::new(&mut imgui, device, queue, renderer_config);

        Ok(Self {
            imgui,
            renderer,
            last_frame_at: Instant::now(),
            latest: None,
        })
    }

    pub fn update(&mut self, sample: LiveFrameDiagnostics) {
        self.latest = Some(sample);
    }

    /// Renders the HUD over `target`, using `window_input` for ImGui hit-testing and dragging.
    ///
    /// `display_size` is the swapchain extent in physical pixels; `Io::display_framebuffer_scale`
    /// must stay `[1.0, 1.0]` so imgui-wgpu’s `fb_size` (`display_size * framebuffer_scale`) matches
    /// the swapchain view (otherwise scissors exceed the target and wgpu validation errors).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &RenderTarget,
        window_input: &WindowInputState,
    ) -> Result<(), String> {
        let (width, height) = target.dimensions();
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.update_delta_time(delta);
        apply_window_input_to_imgui(io, window_input);

        let ui = self.imgui.frame();
        const PANEL_WIDTH: f32 = 760.0;
        let panel_x = (width as f32 - PANEL_WIDTH - 12.0).max(12.0);
        let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
            | WindowFlags::NO_SCROLLBAR
            | WindowFlags::NO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV;

        ui.window("Render Debug")
            .position([panel_x, 12.0], Condition::FirstUseEver)
            .size_constraints([PANEL_WIDTH, 0.0], [PANEL_WIDTH, 1.0e9])
            .bg_alpha(0.72)
            .flags(window_flags)
            .build(|| {
                if let Some(sample) = self.latest.as_ref() {
                    let ai = &sample.adapter_info;
                    ui.text(format!(
                        "FPS {}  |  {} ms  |  {}",
                        hud_fmt::f64_field(8, 2, sample.fps()),
                        hud_fmt::f64_field(8, 3, sample.frame_time_ms()),
                        sample.bottleneck()
                    ));
                    ui.text(format!(
                        "Frame {:>7}  |  {:>5}x{:>5}",
                        sample.frame_index, sample.viewport.0, sample.viewport.1
                    ));

                    ui.separator();
                    ui.text("GPU (wgpu adapter)");
                    ui.text_wrapped(format!("Name: {}", ai.name));
                    ui.text(format!(
                        "Class: {}  |  backend {:?}",
                        device_type_label(ai.device_type),
                        ai.backend
                    ));
                    ui.text_wrapped(format!("Driver: {}  ({})", ai.driver, ai.driver_info));
                    match (
                        sample.gpu_allocator.allocated_bytes,
                        sample.gpu_allocator.reserved_bytes,
                    ) {
                        (Some(alloc), Some(resv)) => ui.text(format!(
                            "Process GPU memory (wgpu allocator): {} / {} GiB allocated / reserved",
                            hud_fmt::gib_value(7, 2, alloc),
                            hud_fmt::gib_value(7, 2, resv)
                        )),
                        _ => ui.text(
                            "Process GPU memory (wgpu allocator): not reported for this backend",
                        ),
                    }
                    ui.separator();
                    ui.text("CPU / RAM (host)");
                    if sample.host.cpu_model.is_empty() {
                        ui.text("CPU model: (unknown)");
                    } else {
                        ui.text_wrapped(format!("CPU model: {}", sample.host.cpu_model));
                    }
                    let ram_pct = if sample.host.ram_total_bytes > 0 {
                        100.0 * sample.host.ram_used_bytes as f64
                            / sample.host.ram_total_bytes as f64
                    } else {
                        0.0
                    };
                    ui.text(format!(
                        "Logical CPUs: {:>3}  |  usage {}%",
                        sample.host.logical_cpus,
                        hud_fmt::f32_field(6, 2, sample.host.cpu_usage_percent)
                    ));
                    ui.text(format!(
                        "RAM: {} / {} GiB  ({}%)",
                        hud_fmt::gib_value(7, 2, sample.host.ram_used_bytes),
                        hud_fmt::gib_value(7, 2, sample.host.ram_total_bytes),
                        hud_fmt::f64_field(5, 1, ram_pct)
                    ));

                    ui.separator();
                    ui.text("Frame timing (ms)");
                    ui.text(format!(
                        "update {}  ipc {}  mesh-prep {}  render {}  present {}",
                        hud_fmt::ms_from_us(8, 3, sample.session_update_us),
                        hud_fmt::ms_from_us(8, 3, sample.ipc_collect_us),
                        hud_fmt::ms_from_us(8, 3, sample.mesh_prep_us),
                        hud_fmt::ms_from_us(8, 3, sample.render_us),
                        hud_fmt::ms_from_us(8, 3, sample.present_us),
                    ));
                    let phases: [(&str, u64); 5] = [
                        ("update", sample.session_update_us),
                        ("ipc", sample.ipc_collect_us),
                        ("mesh-prep", sample.mesh_prep_us),
                        ("render", sample.render_us),
                        ("present", sample.present_us),
                    ];
                    let worst = phases
                        .iter()
                        .max_by_key(|p| p.1)
                        .map(|p| p.0)
                        .unwrap_or("?");
                    ui.text(format!(
                        "  ↳ dominant CPU phase: {}  (total {} ms)",
                        worst,
                        hud_fmt::f64_field(8, 3, sample.frame_time_ms())
                    ));
                    let gpu_mesh = match sample.gpu_mesh_pass_ms {
                        Some(ms) => format!(
                            "{} ms  (timestamp, ~60-frame lag)",
                            hud_fmt::f64_field(8, 3, ms)
                        ),
                        None => "waiting for timestamp readback...".to_string(),
                    };
                    ui.text(format!("GPU mesh pass: {}", gpu_mesh));

                    ui.separator();
                    ui.text(format!(
                        "Batches {:>5} total  |  {:>5} main  |  {:>5} overlay",
                        sample.batch_count,
                        sample
                            .batch_count
                            .saturating_sub(sample.overlay_batch_count),
                        sample.overlay_batch_count
                    ));
                    ui.text(format!(
                        "Draws {:>5} total  |  {:>5} main  |  {:>5} overlay",
                        sample.total_draws_in_batches,
                        sample
                            .total_draws_in_batches
                            .saturating_sub(sample.overlay_draws_in_batches),
                        sample.overlay_draws_in_batches
                    ));
                    ui.text(format!(
                        "Submitted {:>5} total  |  {:>5} main  |  {:>5} overlay",
                        sample.prep_stats.submitted_draws(),
                        sample.submitted_main_draws(),
                        sample.submitted_overlay_draws()
                    ));

                    ui.separator();
                    ui.text(format!(
                        "Prep rigid {:>5}  skinned {:>5}",
                        sample.prep_stats.rigid_input_draws, sample.prep_stats.skinned_input_draws
                    ));
                    ui.text(format!(
                        "Culled rigid {:>5}  skinned {:>5}  total {:>5}  |  degenerate skip {:>5}",
                        sample.prep_stats.frustum_culled_rigid_draws,
                        sample.prep_stats.frustum_culled_skinned_draws,
                        sample.prep_stats.frustum_culled_rigid_draws
                            + sample.prep_stats.frustum_culled_skinned_draws,
                        sample.prep_stats.skipped_cull_degenerate_bounds
                    ));
                    ui.text(format!(
                        "Missing mesh {:>5}  empty mesh {:>5}  missing GPU {:>5}",
                        sample.prep_stats.skipped_missing_mesh_asset,
                        sample.prep_stats.skipped_empty_mesh,
                        sample.prep_stats.skipped_missing_gpu_buffers
                    ));
                    ui.text(format!(
                        "Skinned skips bind {:>5}  ids {:>5}  mismatch {:>5}  vb {:>5}",
                        sample.prep_stats.skipped_skinned_missing_bind_poses,
                        sample.prep_stats.skipped_skinned_missing_bone_ids,
                        sample.prep_stats.skipped_skinned_id_count_mismatch,
                        sample.prep_stats.skipped_skinned_missing_vertex_buffer
                    ));

                    ui.separator();
                    ui.text(format!(
                        "Lights: {}  active  (GPU clustered buffer)",
                        sample.gpu_light_count
                    ));

                    ui.separator();
                    ui.text(format!(
                        "Mesh cache {:>5}  |  tasks {:>5}  |  readbacks {:>5}",
                        sample.mesh_cache_count,
                        sample.pending_render_tasks,
                        sample.pending_camera_task_readbacks
                    ));

                    ui.separator();
                    let tlas_str = if sample.tlas_available {
                        "built"
                    } else {
                        "NONE"
                    };
                    ui.text(format!(
                        "RT  BLASes {}  |  TLAS {}  |  raytracing={}",
                        sample.blas_count, tlas_str, sample.ray_tracing_available
                    ));
                    let rtao_state = if sample.rtao_enabled { "ON" } else { "OFF" };
                    ui.text(format!(
                        "RTAO {}  radius {:.2}  strength {:.2}  samples {}",
                        rtao_state, sample.ao_radius, sample.ao_strength, sample.ao_sample_count
                    ));

                    ui.separator();
                    ui.text(format!(
                        "Flags  cull={}  rtao={}",
                        sample.frustum_culling_enabled, sample.rtao_enabled
                    ));
                } else {
                    ui.text("Waiting for frame diagnostics...");
                }
            });

        let draw_data = self.imgui.render();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("imgui debug hud encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("imgui debug hud pass"),
                timestamp_writes: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.color_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.renderer
                .render(draw_data, queue, device, &mut pass)
                .map_err(|e| format!("imgui render failed: {e}"))?;
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

#[cfg(not(feature = "debug-hud"))]
/// Stub type when the `debug-hud` feature is disabled.
pub struct DebugHud;

#[cfg(not(feature = "debug-hud"))]
impl DebugHud {
    /// Returns an empty HUD (no ImGui initialization).
    pub fn new(
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _surface_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        Ok(Self)
    }

    /// No-op without ImGui.
    pub fn update(&mut self, _sample: LiveFrameDiagnostics) {}

    /// No-op without ImGui.
    pub fn render(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _target: &crate::render::RenderTarget,
        _window_input: &crate::input::WindowInputState,
    ) -> Result<(), String> {
        Ok(())
    }
}

// ── Diagnostic helpers ───────────────────────────────────────────────────────

/// Event emitted by [`ThrottledDropLog::record_drop`] when a log line should be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropLogEvent {
    /// First drop on this channel in the process lifetime.
    First {
        /// Byte length of the dropped payload.
        bytes: usize,
    },
    /// Additional drops since the last log, after the throttle interval elapsed.
    Burst {
        /// Number of dropped sends in this burst (excluding the separately logged first drop).
        count: u32,
        /// Sum of payload bytes in this burst.
        bytes: u64,
    },
}

/// Aggregates outbound IPC drops: log the first immediately, then at most one summary per interval.
pub struct ThrottledDropLog {
    interval: Duration,
    last_log: Option<Instant>,
    pending_count: u32,
    pending_bytes: u64,
    had_first: bool,
}

impl ThrottledDropLog {
    /// Creates a throttle with the given minimum interval between burst summaries.
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_log: None,
            pending_count: 0,
            pending_bytes: 0,
            had_first: false,
        }
    }

    /// Records one dropped send of `bytes`. Returns an event to log, if any.
    pub fn record_drop(&mut self, bytes: usize) -> Option<DropLogEvent> {
        let now = Instant::now();
        if !self.had_first {
            self.had_first = true;
            self.last_log = Some(now);
            return Some(DropLogEvent::First { bytes });
        }
        self.pending_count = self.pending_count.saturating_add(1);
        self.pending_bytes = self.pending_bytes.saturating_add(bytes as u64);
        let last = self.last_log?;
        if now.duration_since(last) >= self.interval {
            let count = self.pending_count;
            let b = self.pending_bytes;
            self.pending_count = 0;
            self.pending_bytes = 0;
            self.last_log = Some(now);
            return Some(DropLogEvent::Burst { count, bytes: b });
        }
        None
    }
}

/// Remembers the last value and reports whether a new value differs.
#[derive(Debug, Default)]
pub struct LogOnChange<T: Eq> {
    last: Option<T>,
}

impl<T: Eq + Clone> LogOnChange<T> {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self { last: None }
    }

    /// Returns `true` when `value` is different from the previously seen value (including first set).
    pub fn changed(&mut self, value: T) -> bool {
        if self.last.as_ref() == Some(&value) {
            return false;
        }
        self.last = Some(value);
        true
    }

    /// Clears the last value so the next `changed` call compares only against `None`.
    pub fn reset(&mut self) {
        self.last = None;
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
            gpu_light_count: 4,
            blas_count: 10,
            tlas_available: true,
            ao_radius: 1.5,
            ao_strength: 0.85,
            ao_sample_count: 8,
            frustum_culling_enabled: true,
            rtao_enabled: true,
            ray_tracing_available: true,
            adapter_info: test_adapter_info(),
            gpu_allocator: GpuAllocatorSnapshot::default(),
            host: HostCpuMemorySnapshot::default(),
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

    #[test]
    fn throttled_first_drop_always_emits() {
        let mut t = ThrottledDropLog::new(Duration::from_secs(2));
        assert_eq!(t.record_drop(9), Some(DropLogEvent::First { bytes: 9 }));
    }

    #[test]
    fn throttled_second_drop_before_interval_does_not_emit() {
        let mut t = ThrottledDropLog::new(Duration::from_secs(60));
        let _ = t.record_drop(9);
        assert_eq!(t.record_drop(10), None);
        assert_eq!(t.record_drop(11), None);
    }

    #[test]
    fn throttled_burst_after_interval() {
        let mut t = ThrottledDropLog::new(Duration::ZERO);
        let _ = t.record_drop(9);
        let ev = t.record_drop(7).expect("burst");
        match ev {
            DropLogEvent::Burst { count, bytes } => {
                assert_eq!(count, 1);
                assert_eq!(bytes, 7);
            }
            DropLogEvent::First { .. } => panic!("expected burst"),
        }
    }

    #[test]
    fn log_on_change_first_and_repeat() {
        let mut c = LogOnChange::new();
        assert!(c.changed(1u32));
        assert!(!c.changed(1));
        assert!(c.changed(2));
        assert!(!c.changed(2));
    }

    #[test]
    fn log_on_change_reset() {
        let mut c = LogOnChange::new();
        assert!(c.changed(1u8));
        assert!(!c.changed(1));
        c.reset();
        assert!(c.changed(1));
    }

    #[cfg(feature = "debug-hud")]
    #[test]
    fn hud_fmt_produces_stable_field_width() {
        assert_eq!(super::hud_fmt::f64_field(8, 2, 1.0).len(), 8);
        assert_eq!(super::hud_fmt::f64_field(8, 2, 123.456).len(), 8);
    }
}
