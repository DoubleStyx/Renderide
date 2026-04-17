//! Dear ImGui overlay for developer diagnostics.
//!
//! The **Frame timing** window shows FPS and CPU/GPU submit-interval metrics (wall-clock splits around submits).
//! **[`crate::config::DebugSettings::debug_hud_frame_timing`]** toggles the **Frame timing** window (default on).
//! **[`crate::config::DebugSettings::debug_hud_enabled`]** toggles **Renderide debug** (Stats / Shader routes / GPU memory).
//! **[`crate::config::DebugSettings::debug_hud_transforms`]** toggles the **Scene transforms** window.
//!
//! Window bodies live in [`mod@windows`].

mod fmt;
mod layout;
mod windows;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::config::RendererSettingsHandle;
use imgui::{Condition, Context, FontConfig, FontSource, WindowFlags};
use imgui_wgpu::{Renderer as ImguiWgpuRenderer, RendererConfig};

use super::debug_hud_encode_error::DebugHudEncodeError;
use super::frame_diagnostics_snapshot::FrameDiagnosticsSnapshot;
use super::frame_timing_hud_snapshot::FrameTimingHudSnapshot;
use super::hud_input::DebugHudInput;
use super::renderer_info_snapshot::RendererInfoSnapshot;
use super::scene_transforms_snapshot::SceneTransformsSnapshot;

/// Renders the **Frame timing** overlay window when enabled.
fn overlay_frame_timing_window(ui: &imgui::Ui, frame_timing: Option<&FrameTimingHudSnapshot>) {
    DebugHud::frame_timing_window(ui, frame_timing);
}

/// Renders the **Renderide debug** tabbed panel (Stats / Shader routes / GPU memory).
fn overlay_main_debug_window(
    ui: &imgui::Ui,
    width: u32,
    latest: Option<&RendererInfoSnapshot>,
    frame_diagnostics: Option<&FrameDiagnosticsSnapshot>,
) {
    const PANEL_WIDTH: f32 = 760.0;
    let panel_x = (width as f32 - PANEL_WIDTH - layout::MARGIN).max(layout::MARGIN);
    let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
        | WindowFlags::NO_RESIZE
        | WindowFlags::NO_SAVED_SETTINGS
        | WindowFlags::NO_FOCUS_ON_APPEARING
        | WindowFlags::NO_NAV;

    ui.window("Renderide debug")
        .position([panel_x, layout::MARGIN], Condition::FirstUseEver)
        .size_constraints([PANEL_WIDTH, 0.0], [PANEL_WIDTH, 1.0e9])
        .bg_alpha(0.72)
        .flags(window_flags)
        .build(|| {
            if let Some(_tab_bar) = ui.tab_bar("debug_tabs") {
                if let Some(_tab) = ui.tab_item("Stats") {
                    DebugHud::main_debug_panel(ui, latest, frame_diagnostics);
                }
                if let Some(_tab) = ui.tab_item("Shader routes") {
                    DebugHud::shader_mappings_tab(ui, frame_diagnostics);
                }
                if let Some(_tab) = ui.tab_item("GPU memory") {
                    DebugHud::gpu_memory_tab(ui, frame_diagnostics);
                }
            }
        });
}

/// Renders the **Scene transforms** window when enabled.
fn overlay_scene_transforms_window(
    ui: &imgui::Ui,
    scene_transforms: &SceneTransformsSnapshot,
    scene_transforms_open: &mut bool,
) {
    DebugHud::scene_transforms_window(ui, scene_transforms, scene_transforms_open);
}

/// Dear ImGui overlay: frame timing, renderer stats, shader routes, scene transforms, and config UI.
pub struct DebugHud {
    imgui: Context,
    renderer: ImguiWgpuRenderer,
    last_frame_at: Instant,
    /// Lightweight FPS / wall / CPU–submit / GPU-idle metrics ([`FrameTimingHudSnapshot`]).
    frame_timing: Option<FrameTimingHudSnapshot>,
    latest: Option<RendererInfoSnapshot>,
    /// Per-frame timing, draws, host metrics, shader routes, and GPU allocator detail ([`FrameDiagnosticsSnapshot`]).
    frame_diagnostics: Option<FrameDiagnosticsSnapshot>,
    /// Per-frame world transform listing for the **Scene transforms** window.
    scene_transforms: SceneTransformsSnapshot,
    /// Whether the **Scene transforms** window is open (independent of the stats panel).
    scene_transforms_open: bool,
    /// Live settings + persistence target for the **Renderer config** window.
    renderer_settings: RendererSettingsHandle,
    config_save_path: PathBuf,
    /// Whether the **Renderer config** window is open.
    renderer_config_open: bool,
}

impl DebugHud {
    /// Builds ImGui and the wgpu render backend for the swapchain format.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        renderer_settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) -> Self {
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
            scene_transforms_open: true,
            renderer_settings,
            config_save_path,
            renderer_config_open: true,
        }
    }

    /// Stores [`FrameTimingHudSnapshot`] for the **Frame timing** window.
    pub fn set_frame_timing(&mut self, sample: FrameTimingHudSnapshot) {
        self.frame_timing = Some(sample);
    }

    /// Stores [`RendererInfoSnapshot`] for the **Stats** tab (IPC, adapter, scene, materials, graph).
    pub fn set_snapshot(&mut self, sample: RendererInfoSnapshot) {
        self.latest = Some(sample);
    }

    /// Stores [`FrameDiagnosticsSnapshot`] for timing, host/allocator, draws, textures, shader routes, and GPU memory tab data.
    pub fn set_frame_diagnostics(&mut self, sample: FrameDiagnosticsSnapshot) {
        self.frame_diagnostics = Some(sample);
    }

    /// Stores per–render-space world transform rows for the **Scene transforms** window.
    pub fn set_scene_transforms_snapshot(&mut self, sample: SceneTransformsSnapshot) {
        self.scene_transforms = sample;
    }

    /// Clears Stats / Shader routes payloads only (not [`Self::frame_timing`] or scene transforms).
    pub fn clear_stats_hud_payloads(&mut self) {
        self.latest = None;
        self.frame_diagnostics = None;
    }

    /// Clears the **Scene transforms** HUD payload.
    pub fn clear_scene_transforms_snapshot(&mut self) {
        self.scene_transforms = SceneTransformsSnapshot::default();
    }

    /// Clears all HUD payloads (including frame timing).
    pub fn clear_diagnostic_snapshots(&mut self) {
        self.frame_timing = None;
        self.clear_stats_hud_payloads();
        self.clear_scene_transforms_snapshot();
    }

    /// Updates ImGui delta time, display size, and injects [`DebugHudInput`] for this frame.
    fn apply_overlay_frame_io(&mut self, (width, height): (u32, u32), input: &DebugHudInput) {
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.update_delta_time(delta);
        windows::apply_input(io, input);
    }

    /// Reads which optional HUD windows are enabled from live settings.
    fn overlay_feature_flags(&self) -> (bool, bool, bool, bool) {
        let (frame_timing_hud, main_hud, transforms_hud) = self
            .renderer_settings
            .read()
            .map(|g| {
                (
                    g.debug.debug_hud_frame_timing,
                    g.debug.debug_hud_enabled,
                    g.debug.debug_hud_transforms,
                )
            })
            .unwrap_or((true, false, false));
        let any_debug_content = frame_timing_hud || main_hud || transforms_hud;
        (
            frame_timing_hud,
            main_hud,
            transforms_hud,
            any_debug_content,
        )
    }

    /// Encodes ImGui draw lists into a load-on-top pass over `backbuffer` and returns want-capture flags.
    fn encode_imgui_wgpu_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
    ) -> Result<(bool, bool), DebugHudEncodeError> {
        let draw_data = self.imgui.render();
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
                timestamp_writes: None,
                multiview_mask: None,
            });
            self.renderer
                .render(draw_data, queue, device, &mut pass)
                .map_err(|e| DebugHudEncodeError::ImguiWgpu(e.to_string()))?;
        }
        let io = self.imgui.io();
        Ok((io.want_capture_mouse, io.want_capture_keyboard))
    }

    /// Records ImGui into `encoder` as a load-on-top pass over `backbuffer`.
    pub fn encode_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        (width, height): (u32, u32),
        input: &DebugHudInput,
    ) -> Result<(bool, bool), DebugHudEncodeError> {
        self.apply_overlay_frame_io((width, height), input);

        let (frame_timing_hud, main_hud, transforms_hud, any_debug_content) =
            self.overlay_feature_flags();

        let ui = self.imgui.frame();
        if any_debug_content {
            if frame_timing_hud {
                overlay_frame_timing_window(ui, self.frame_timing.as_ref());
            }
            if main_hud {
                overlay_main_debug_window(
                    ui,
                    width,
                    self.latest.as_ref(),
                    self.frame_diagnostics.as_ref(),
                );
            }
            if transforms_hud {
                overlay_scene_transforms_window(
                    ui,
                    &self.scene_transforms,
                    &mut self.scene_transforms_open,
                );
            }
        }

        Self::renderer_config_window(
            ui,
            &self.renderer_settings,
            &self.config_save_path,
            &mut self.renderer_config_open,
        );

        self.encode_imgui_wgpu_pass(device, queue, encoder, backbuffer)
    }
}
