//! Diagnostic helpers and optional ImGui on-screen HUD.
//!
//! The HUD is enabled by the `debug-hud` Cargo feature (on by default). Disable default features
//! (`cargo build -p renderide --no-default-features`) for lean builds without `imgui` / `imgui-wgpu`.

use std::time::{Duration, Instant};

use crate::render::pass::MeshDrawPrepStats;

#[cfg(feature = "debug-hud")]
use crate::render::RenderTarget;

#[cfg(feature = "debug-hud")]
use imgui::{Condition, Context, FontConfig, FontSource, WindowFlags};
#[cfg(feature = "debug-hud")]
use imgui_wgpu::{Renderer, RendererConfig};

// ── ImGui HUD ────────────────────────────────────────────────────────────────

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
}

#[cfg_attr(not(feature = "debug-hud"), allow(dead_code))]
impl LiveFrameDiagnostics {
    fn frame_time_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }

    fn fps(&self) -> f64 {
        if self.total_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.total_us as f64
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

/// ImGui-backed on-screen diagnostics panel when the `debug-hud` feature is enabled; otherwise a
/// zero-cost stub (see [`Self::new`], [`Self::render`], [`Self::update`]).
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

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &RenderTarget,
    ) -> Result<(), String> {
        let (width, height) = target.dimensions();
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.update_delta_time(delta);

        let ui = self.imgui.frame();
        let panel_width = 700.0f32;
        let panel_x = (width as f32 - panel_width - 12.0).max(12.0);
        let window_flags = WindowFlags::NO_DECORATION
            | WindowFlags::NO_MOVE
            | WindowFlags::NO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV
            | WindowFlags::NO_INPUTS;

        ui.window("Render Debug")
            .position([panel_x, 12.0], Condition::Always)
            .size([panel_width, 0.0], Condition::Always)
            .bg_alpha(0.72)
            .flags(window_flags)
            .build(|| {
                if let Some(s) = self.latest.as_ref() {
                    // ── Header ──────────────────────────────────────────────
                    ui.text(format!(
                        "FPS {:.1}  |  {:.2} ms  |  {}",
                        s.fps(), s.frame_time_ms(), s.bottleneck()
                    ));
                    ui.text(format!(
                        "Frame {}  |  {}x{}",
                        s.frame_index, s.viewport.0, s.viewport.1
                    ));

                    // ── CPU timings ──────────────────────────────────────────
                    ui.separator();
                    let ms = |us: u64| us as f64 / 1000.0;
                    ui.text(format!(
                        "CPU update {:.2}  ipc {:.2}  mesh-prep {:.2}  render {:.2}  present {:.2}  [ms]",
                        ms(s.session_update_us), ms(s.ipc_collect_us),
                        ms(s.mesh_prep_us), ms(s.render_us), ms(s.present_us)
                    ));
                    // Identify the dominant CPU phase
                    let phases: [(&str, u64); 5] = [
                        ("update",    s.session_update_us),
                        ("ipc",       s.ipc_collect_us),
                        ("mesh-prep", s.mesh_prep_us),
                        ("render",    s.render_us),
                        ("present",   s.present_us),
                    ];
                    let worst = phases.iter().max_by_key(|p| p.1).map(|p| p.0).unwrap_or("?");
                    ui.text(format!("  ↳ dominant CPU phase: {}  (total {:.2} ms)", worst, s.frame_time_ms()));
                    ui.text(match s.gpu_mesh_pass_ms {
                        Some(ms_val) => format!("GPU mesh pass {:.2} ms  (timestamp, ~60-frame lag)", ms_val),
                        None => "GPU mesh pass: waiting for timestamp readback...".to_string(),
                    });

                    // ── Draw batches ─────────────────────────────────────────
                    ui.separator();
                    ui.text(format!(
                        "Batches {} total  |  {} main  |  {} overlay",
                        s.batch_count,
                        s.batch_count.saturating_sub(s.overlay_batch_count),
                        s.overlay_batch_count
                    ));
                    ui.text(format!(
                        "Draws {} total  |  {} main  |  {} overlay",
                        s.total_draws_in_batches,
                        s.total_draws_in_batches.saturating_sub(s.overlay_draws_in_batches),
                        s.overlay_draws_in_batches
                    ));
                    ui.text(format!(
                        "Submitted {} total  |  {} main  |  {} overlay",
                        s.prep_stats.submitted_draws(),
                        s.submitted_main_draws(),
                        s.submitted_overlay_draws()
                    ));

                    // ── Mesh prep detail ─────────────────────────────────────
                    ui.separator();
                    ui.text(format!(
                        "Prep rigid {}  skinned {}",
                        s.prep_stats.rigid_input_draws, s.prep_stats.skinned_input_draws
                    ));
                    ui.text(format!(
                        "Culled rigid {}  skinned {}  total {}  |  degenerate skip {}",
                        s.prep_stats.frustum_culled_rigid_draws,
                        s.prep_stats.frustum_culled_skinned_draws,
                        s.prep_stats.frustum_culled_rigid_draws + s.prep_stats.frustum_culled_skinned_draws,
                        s.prep_stats.skipped_cull_degenerate_bounds
                    ));
                    ui.text(format!(
                        "Missing mesh {}  empty mesh {}  missing GPU {}",
                        s.prep_stats.skipped_missing_mesh_asset,
                        s.prep_stats.skipped_empty_mesh,
                        s.prep_stats.skipped_missing_gpu_buffers
                    ));
                    ui.text(format!(
                        "Skinned skips bind {}  ids {}  mismatch {}  vb {}",
                        s.prep_stats.skipped_skinned_missing_bind_poses,
                        s.prep_stats.skipped_skinned_missing_bone_ids,
                        s.prep_stats.skipped_skinned_id_count_mismatch,
                        s.prep_stats.skipped_skinned_missing_vertex_buffer
                    ));

                    // ── Lights ────────────────────────────────────────────────
                    ui.separator();
                    ui.text(format!(
                        "Lights: {}  active  (GPU clustered buffer)",
                        s.gpu_light_count
                    ));

                    // ── Caches / tasks ───────────────────────────────────────
                    ui.separator();
                    ui.text(format!(
                        "Mesh cache {}  |  tasks {}  |  readbacks {}",
                        s.mesh_cache_count, s.pending_render_tasks, s.pending_camera_task_readbacks
                    ));

                    // ── Ray tracing / RTAO ────────────────────────────────────
                    ui.separator();
                    let tlas_str = if s.tlas_available { "built" } else { "NONE" };
                    ui.text(format!(
                        "RT  BLASes {}  |  TLAS {}  |  raytracing={}",
                        s.blas_count, tlas_str, s.ray_tracing_available
                    ));
                    let rtao_state = if s.rtao_enabled { "ON" } else { "OFF" };
                    ui.text(format!(
                        "RTAO {}  radius {:.2}  strength {:.2}  samples {}",
                        rtao_state, s.ao_radius, s.ao_strength, s.ao_sample_count
                    ));

                    // ── Feature flags ─────────────────────────────────────────
                    ui.separator();
                    ui.text(format!(
                        "Flags  cull={}  rtao={}",
                        s.frustum_culling_enabled, s.rtao_enabled
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
}
