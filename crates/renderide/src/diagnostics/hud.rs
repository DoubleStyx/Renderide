//! Dear ImGui overlay for developer diagnostics.
//!
//! The **Frame timing** window shows FPS, CPU/GPU frame intervals, RAM/VRAM, and a frametime graph.
//! **Feedback / Bug Report** shows quick links for reporting issues and joining discussion.
//! **[`crate::config::DebugSettings::debug_hud_frame_timing`]** toggles the **Frame timing** window (default on).
//! **[`crate::config::DebugSettings::debug_hud_links`]** toggles the **Feedback / Bug Report** links panel.
//! **[`crate::config::DebugSettings::debug_hud_enabled`]** toggles **Renderide debug** (Stats / Shader routes / Draw state / GPU memory / GPU passes).
//! **[`crate::config::DebugSettings::debug_hud_transforms`]** toggles the **Scene transforms** window.
//! **[`crate::config::DebugSettings::debug_hud_textures`]** toggles the **Textures** window.
//!
//! HUD-rendering infrastructure lives in submodules:
//!
//! - [`layout`]: declarative window placement (`Viewport`, `WindowSlot`) plus
//!   the stacked-column constants and helpers used by the anchored HUD windows.
//! - [`state`]: [`state::HudUiState`], the per-tab view state and filters owned by
//!   [`DebugHud`].
//! - [`view`]: rendering algebra (`HudWindow`, `TabView`).
//! - [`registry`]: static-dispatch [`registry::DebugWindow`] enum + [`registry::OverlayFeatureFlags`].
//! - [`fmt`]: right-aligned numeric formatters and byte-compaction helpers.
//! - [`input`]: HUD input transport plus per-frame ImGui IO bridge.
//! - [`windows`]: concrete window/tab impls.

pub mod fmt;
pub mod input;
pub mod layout;
pub mod metrics;
pub(crate) mod persistence;
pub mod registry;
pub mod state;
pub mod view;
pub mod windows;

pub use crate::hud_contract::DebugHudEncodeError;
pub use input::{DebugHudInput, sanitize_input_state_for_imgui_host};
pub use metrics::DebugHudMetricInterest;
pub use state::HudUiState;

/// GPU target and encoder state for one debug-HUD overlay encode.
pub(crate) struct DebugHudOverlayContext<'a, 'encoder> {
    /// WGPU device used by the ImGui renderer.
    pub(crate) device: &'a wgpu::Device,
    /// WGPU queue used by the ImGui renderer for texture updates.
    pub(crate) queue: &'a wgpu::Queue,
    /// Command encoder receiving the HUD render pass.
    pub(crate) encoder: &'encoder mut wgpu::CommandEncoder,
    /// Swapchain or offscreen color view the HUD should composite over.
    pub(crate) backbuffer: &'a wgpu::TextureView,
    /// Pixel extent of the target surface.
    pub(crate) extent: (u32, u32),
    /// Optional GPU profiler for pass timestamp instrumentation.
    pub(crate) profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use imgui::{Condition, Context, FontConfig, FontSource};
use imgui_wgpu::{Renderer as ImguiWgpuRenderer, RendererConfig};

use crate::config::{DebugHudSettings, RendererSettingsHandle, save_renderer_settings};

use self::input::apply_input;
use self::layout::Viewport;
use self::persistence::{
    imgui_ini_path_from_config_save_path, read_nonempty_text, write_nonempty_text_atomic,
};
use self::registry::{DebugWindow, OverlayFeatureFlags};
use self::view::HudWindow;
use self::windows::feedback::FeedbackWindow;
use self::windows::frame_timing::FrameTimingWindow;
use self::windows::main_debug::{MainDebugWindow, MainDebugWindowData};
use self::windows::renderer_config::{RendererConfigData, RendererConfigWindow};
use self::windows::scene_transforms::SceneTransformsWindow;
use self::windows::texture_debug::TextureDebugWindow;
use super::snapshots::frame_diagnostics::FrameDiagnosticsSnapshot;
use super::snapshots::frame_timing::FrameTimingHudSnapshot;
use super::snapshots::renderer_info::RendererInfoSnapshot;
use super::snapshots::scene_transforms::SceneTransformsSnapshot;
use super::snapshots::texture_debug::TextureDebugSnapshot;

const IMGUI_INI_SAVE_RATE_SECS: f32 = 0.25;

/// Dear ImGui overlay: feedback links, frame timing, renderer stats, shader routes, scene transforms, and config UI.
pub struct DebugHud {
    imgui: Context,
    renderer: ImguiWgpuRenderer,
    last_frame_at: Instant,
    /// Lightweight FPS / wall / CPU-submit / GPU-idle metrics ([`FrameTimingHudSnapshot`]).
    frame_timing: Option<FrameTimingHudSnapshot>,
    latest: Option<RendererInfoSnapshot>,
    /// Per-frame timing, draws, host metrics, shader routes, and GPU allocator detail ([`FrameDiagnosticsSnapshot`]).
    frame_diagnostics: Option<FrameDiagnosticsSnapshot>,
    /// Per-frame world transform listing for the **Scene transforms** window.
    scene_transforms: SceneTransformsSnapshot,
    /// Per-frame texture pool listing for the **Textures** window.
    texture_debug: TextureDebugSnapshot,
    /// Per-tab view state and filter toggles.
    ui_state: HudUiState,
    /// Live settings + persistence target for the **Renderer config** window.
    renderer_settings: RendererSettingsHandle,
    config_save_path: PathBuf,
    /// Sidecar path for Dear ImGui's raw window layout settings.
    imgui_ini_path: PathBuf,
    /// When `true`, do not write `config.toml` from the overlay (startup Figment extract failed).
    suppress_renderer_config_disk_writes: bool,
    /// Most recent flattened per-pass GPU timings and query stats for the **GPU passes** tab.
    ///
    /// Empty until the first profiled frame completes; see
    /// [`crate::gpu::GpuContext::latest_gpu_profiler_snapshot_handle`].
    gpu_profiler_snapshot: crate::profiling::GpuProfilerSnapshot,
}

impl DebugHud {
    /// Builds ImGui and the wgpu render backend for the swapchain format.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        renderer_settings: RendererSettingsHandle,
        config_save_path: PathBuf,
        suppress_renderer_config_disk_writes: bool,
    ) -> Self {
        let hud_settings = renderer_settings
            .read()
            .map(|g| g.debug.hud.clone())
            .unwrap_or_else(|_| DebugHudSettings::default());

        let mut imgui = Context::create();
        imgui.set_ini_filename(None);
        imgui.set_log_filename(None);
        {
            let io = imgui.io_mut();
            io.config_windows_move_from_title_bar_only = true;
            io.config_drag_click_to_input_text = true;
            io.font_global_scale = hud_settings.resolved_ui_scale();
            io.ini_saving_rate = IMGUI_INI_SAVE_RATE_SECS;
        }
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                oversample_h: 2,
                pixel_snap_h: true,
                size_pixels: 14.0,
                ..FontConfig::default()
            }),
        }]);

        let imgui_ini_path = imgui_ini_path_from_config_save_path(&config_save_path);
        Self::load_imgui_ini_if_enabled(&mut imgui, &imgui_ini_path, &hud_settings);

        let mut renderer_config = RendererConfig::new();
        renderer_config.texture_format = surface_format;
        let renderer = ImguiWgpuRenderer::new(&mut imgui, device, queue, renderer_config);

        Self {
            imgui,
            renderer,
            last_frame_at: Instant::now(),
            frame_timing: None,
            latest: None,
            frame_diagnostics: None,
            scene_transforms: SceneTransformsSnapshot::default(),
            texture_debug: TextureDebugSnapshot::default(),
            ui_state: HudUiState::from_settings(&hud_settings),
            renderer_settings,
            config_save_path,
            imgui_ini_path,
            suppress_renderer_config_disk_writes,
            gpu_profiler_snapshot: crate::profiling::GpuProfilerSnapshot::default(),
        }
    }

    /// Replaces the GPU profiler snapshot shown in the **GPU passes** tab.
    ///
    /// Called once per winit tick from the HUD update path with the latest snapshot read out of
    /// [`crate::gpu::GpuContext::latest_gpu_profiler_snapshot_handle`].
    pub fn set_gpu_profiler_snapshot(&mut self, snapshot: crate::profiling::GpuProfilerSnapshot) {
        self.gpu_profiler_snapshot = snapshot;
    }

    /// Clears the **GPU passes** tab payload.
    pub fn clear_gpu_profiler_snapshot(&mut self) {
        self.gpu_profiler_snapshot = crate::profiling::GpuProfilerSnapshot::default();
    }

    /// Stores [`FrameTimingHudSnapshot`] for the **Frame timing** window.
    pub fn set_frame_timing(&mut self, sample: FrameTimingHudSnapshot) {
        self.frame_timing = Some(sample);
    }

    /// Clears the **Frame timing** window payload.
    pub fn clear_frame_timing(&mut self) {
        self.frame_timing = None;
    }

    /// Stores [`RendererInfoSnapshot`] for the **Stats** tab (IPC, adapter, scene, materials, graph).
    pub fn set_snapshot(&mut self, sample: RendererInfoSnapshot) {
        self.latest = Some(sample);
    }

    /// Stores [`FrameDiagnosticsSnapshot`] for timing, host/allocator, draws, textures, shader routes, and GPU memory tab data.
    pub fn set_frame_diagnostics(&mut self, sample: FrameDiagnosticsSnapshot) {
        self.frame_diagnostics = Some(sample);
    }

    /// Stores per-render-space world transform rows for the **Scene transforms** window.
    pub fn set_scene_transforms_snapshot(&mut self, sample: SceneTransformsSnapshot) {
        self.scene_transforms = sample;
    }

    /// Clears the **Scene transforms** window payload.
    pub fn clear_scene_transforms_snapshot(&mut self) {
        self.scene_transforms = SceneTransformsSnapshot::default();
    }

    /// Stores texture pool rows for the **Textures** window.
    pub fn set_texture_debug_snapshot(&mut self, sample: TextureDebugSnapshot) {
        self.texture_debug = sample;
    }

    /// Clears main debug tab payloads only (not [`Self::frame_timing`] or scene transforms).
    pub fn clear_stats_hud_payloads(&mut self) {
        self.latest = None;
        self.frame_diagnostics = None;
        self.clear_gpu_profiler_snapshot();
    }

    /// Clears the **Textures** HUD payload.
    pub fn clear_texture_debug_snapshot(&mut self) {
        self.texture_debug = TextureDebugSnapshot::default();
    }

    /// Updates ImGui delta time, display size, and injects [`DebugHudInput`] for this frame.
    fn apply_overlay_frame_io(&mut self, (width, height): (u32, u32), input: &DebugHudInput) {
        profiling::scope!("hud::apply_input");
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();
        let hud_settings = self.current_hud_settings();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.font_global_scale = hud_settings.resolved_ui_scale();
        io.update_delta_time(delta);
        apply_input(io, input);
    }

    fn current_hud_settings(&self) -> DebugHudSettings {
        self.renderer_settings
            .read()
            .map(|g| g.debug.hud.clone())
            .unwrap_or_else(|_| DebugHudSettings::default())
    }

    fn load_imgui_ini_if_enabled(
        imgui: &mut Context,
        imgui_ini_path: &Path,
        hud_settings: &DebugHudSettings,
    ) {
        if !hud_settings.persist_layout {
            return;
        }

        match read_nonempty_text(imgui_ini_path) {
            Ok(Some(data)) => imgui.load_ini_settings(&data),
            Ok(None) => {
                logger::debug!(
                    "Ignoring empty ImGui layout file at {}",
                    imgui_ini_path.display()
                );
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => {
                logger::warn!(
                    "Failed to read ImGui layout from {}: {e}",
                    imgui_ini_path.display()
                );
            }
        }
    }

    fn persist_ui_state_to_config_if_changed(&self) {
        let Ok(mut g) = self.renderer_settings.write() else {
            logger::warn!("Failed to persist HUD state: renderer settings store is unavailable");
            return;
        };

        if !self.ui_state.write_to_settings(&mut g.debug.hud) {
            return;
        }

        if self.suppress_renderer_config_disk_writes {
            logger::error!(
                "Refusing to save renderer config to {}: disk writes suppressed after startup extract failure",
                self.config_save_path.display()
            );
            return;
        }

        if let Err(e) = save_renderer_settings(&self.config_save_path, &g) {
            logger::warn!(
                "Failed to save renderer config to {}: {e}",
                self.config_save_path.display()
            );
        }
    }

    fn save_imgui_ini_now(&mut self) {
        if self.current_hud_settings().persist_layout {
            let mut contents = String::new();
            self.imgui.save_ini_settings(&mut contents);
            match write_nonempty_text_atomic(&self.imgui_ini_path, &contents) {
                Ok(true) => {}
                Ok(false) => {
                    logger::debug!(
                        "Skipping empty ImGui layout save to {}",
                        self.imgui_ini_path.display()
                    );
                }
                Err(e) => {
                    logger::warn!(
                        "Failed to save ImGui layout to {}: {e}",
                        self.imgui_ini_path.display()
                    );
                }
            }
        }

        self.imgui.io_mut().want_save_ini_settings = false;
    }

    fn save_imgui_ini_if_requested(&mut self) {
        if self.imgui.io().want_save_ini_settings {
            self.save_imgui_ini_now();
        }
    }

    /// Returns `true` when at least one HUD window will draw something this frame.
    ///
    /// Used by the render-graph executor to skip the entire HUD command encoder + GPU profiler
    /// query wrap when ImGui is hidden. Skipping is safe even when the HUD has been open in prior
    /// frames: ImGui's per-frame state lives on [`Self::imgui`] and is only consumed when
    /// [`Self::encode_overlay`] runs, so dropping a frame's encode does not corrupt later frames'
    /// drawing. When ImGui is visible, **Renderer config** always draws so configuration remains
    /// reachable.
    pub fn has_visible_content(&self) -> bool {
        let flags = OverlayFeatureFlags::from_settings(&self.renderer_settings);
        flags.imgui_visible
    }

    /// Encodes ImGui draw lists into a load-on-top pass over `backbuffer` and returns want-capture flags.
    fn encode_imgui_wgpu_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(bool, bool), DebugHudEncodeError> {
        profiling::scope!("hud::encode_imgui_wgpu");
        let draw_data = self.imgui.render();
        let pass_query = profiler.map(|p| p.begin_pass_query("hud::imgui_wgpu_pass", encoder));
        let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("imgui-debug-hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: backbuffer,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes,
                multiview_mask: None,
            });
            self.renderer
                .render(draw_data, queue, device, &mut pass)
                .map_err(|e| DebugHudEncodeError::ImguiWgpu(e.to_string()))?;
        };
        if let (Some(p), Some(q)) = (profiler, pass_query) {
            p.end_query(encoder, q);
        }
        let io = self.imgui.io();
        Ok((io.want_capture_mouse, io.want_capture_keyboard))
    }

    /// Records ImGui into `encoder` as a load-on-top pass over `backbuffer`.
    ///
    /// Iterates [`DebugWindow::ALL`] and dispatches to the matching [`HudWindow`] impl per
    /// variant; [`DebugWindow::enabled`] gates each window from
    /// [`OverlayFeatureFlags::from_settings`]. Adding a new window means adding an enum variant
    /// + one match arm; the encode path stays a single for-loop.
    pub fn encode_overlay(
        &mut self,
        target: DebugHudOverlayContext<'_, '_>,
        input: &DebugHudInput,
    ) -> Result<(bool, bool), DebugHudEncodeError> {
        profiling::scope!("hud::encode_overlay");
        let DebugHudOverlayContext {
            device,
            queue,
            encoder,
            backbuffer,
            extent: (width, height),
            profiler,
        } = target;
        self.apply_overlay_frame_io((width, height), input);

        let flags = OverlayFeatureFlags::from_settings(&self.renderer_settings);
        let viewport = Viewport { width, height };

        // `self.imgui.frame()` already holds a mutable borrow on `self.imgui`, so the dispatch
        // loop cannot also take `&mut self`. Field-disjoint borrows let each match arm borrow
        // exactly the snapshot it needs while sharing `&mut self.ui_state`.
        let ui = self.imgui.frame();
        let ui_state = &mut self.ui_state;
        let frame_timing = self.frame_timing.as_ref();
        let renderer_info = self.latest.as_ref();
        let frame_diagnostics = self.frame_diagnostics.as_ref();
        let gpu_profiler_snapshot = &self.gpu_profiler_snapshot;
        let scene_transforms = &self.scene_transforms;
        let texture_debug = &self.texture_debug;
        let renderer_settings = &self.renderer_settings;
        let config_save_path = self.config_save_path.as_path();
        let suppress_renderer_config_disk_writes = self.suppress_renderer_config_disk_writes;

        for &window in DebugWindow::ALL {
            if !window.enabled(flags) {
                continue;
            }
            match window {
                DebugWindow::FrameTiming => {
                    render_window(ui, viewport, &FrameTimingWindow, frame_timing, ui_state);
                }
                DebugWindow::Feedback => {
                    render_window(ui, viewport, &FeedbackWindow, (), ui_state);
                }
                DebugWindow::Main => render_window(
                    ui,
                    viewport,
                    &MainDebugWindow,
                    MainDebugWindowData {
                        renderer_info,
                        frame_diagnostics,
                        gpu_profiler_snapshot,
                    },
                    ui_state,
                ),
                DebugWindow::SceneTransforms => render_window(
                    ui,
                    viewport,
                    &SceneTransformsWindow,
                    scene_transforms,
                    ui_state,
                ),
                DebugWindow::Textures => {
                    render_window(ui, viewport, &TextureDebugWindow, texture_debug, ui_state);
                }
                DebugWindow::RendererConfig => render_window(
                    ui,
                    viewport,
                    &RendererConfigWindow,
                    RendererConfigData {
                        settings: renderer_settings,
                        save_path: config_save_path,
                        suppress_renderer_config_disk_writes,
                    },
                    ui_state,
                ),
            }
        }

        let result = self.encode_imgui_wgpu_pass(device, queue, encoder, backbuffer, profiler);
        self.persist_ui_state_to_config_if_changed();
        self.save_imgui_ini_if_requested();
        result
    }
}

impl Drop for DebugHud {
    fn drop(&mut self) {
        self.save_imgui_ini_now();
    }
}

/// Renders one [`HudWindow`] in the standard ImGui envelope (position, size, flags, bg alpha,
/// body).
fn render_window<W>(
    ui: &imgui::Ui,
    viewport: Viewport,
    window: &W,
    data: W::Data<'_>,
    state: &mut W::State,
) where
    W: HudWindow,
{
    profiling::scope!("hud::render_window");
    let slot = window.anchor(viewport);
    ui.window(window.title())
        .position(slot.position, Condition::FirstUseEver)
        .size(slot.size, Condition::FirstUseEver)
        .size_constraints(slot.size_min, slot.size_max)
        .bg_alpha(window.bg_alpha())
        .flags(window.flags())
        .build(|| window.body(ui, data, state));
}
