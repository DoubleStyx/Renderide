//! Dear ImGui overlay for developer diagnostics.
//!
//! The **Frame timing** window shows FPS and CPU/GPU submit-interval metrics (wall-clock splits around submits).
//! **[`crate::config::DebugSettings::debug_hud_frame_timing`]** toggles the **Frame timing** window (default on).
//! **[`crate::config::DebugSettings::debug_hud_enabled`]** toggles **Renderide debug** (Stats / Shader routes).
//! **[`crate::config::DebugSettings::debug_hud_transforms`]** toggles the **Scene transforms** window.
//! **[`crate::config::DebugSettings::debug_hud_textures`]** toggles the **Textures** window.
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

use super::frame_diagnostics_snapshot::FrameDiagnosticsSnapshot;
use super::frame_timing_hud_snapshot::FrameTimingHudSnapshot;
use super::hud_input::DebugHudInput;
use super::renderer_info_snapshot::RendererInfoSnapshot;
use super::scene_transforms_snapshot::SceneTransformsSnapshot;
use super::texture_debug_snapshot::TextureDebugSnapshot;

/// Dear ImGui overlay: frame timing, renderer stats, shader routes, scene transforms, textures, and config UI.
pub struct DebugHud {
    imgui: Context,
    renderer: ImguiWgpuRenderer,
    last_frame_at: Instant,
    /// Lightweight FPS / wall / CPU–submit / GPU-idle metrics ([`FrameTimingHudSnapshot`]).
    frame_timing: Option<FrameTimingHudSnapshot>,
    latest: Option<RendererInfoSnapshot>,
    /// Per-frame timing, draws, host metrics, and shader-route strings ([`FrameDiagnosticsSnapshot`]).
    frame_diagnostics: Option<FrameDiagnosticsSnapshot>,
    /// Per-frame world transform listing for the **Scene transforms** window.
    scene_transforms: SceneTransformsSnapshot,
    /// Whether the **Scene transforms** window is open (independent of the stats panel).
    scene_transforms_open: bool,
    /// Per-frame GPU texture pool listing for the **Textures** window.
    texture_debug: TextureDebugSnapshot,
    /// Whether the **Textures** window is open (independent of the stats panel).
    texture_debug_open: bool,
    /// Texture table filter: only rows used by submitted world draws for the current view.
    texture_debug_current_view_only: bool,
    /// **Shader routes** filter: when true, hides implemented routes and shows only the ones
    /// that fall back to `debug_world_normals` (i.e., shaders with no embedded target yet).
    shader_routes_only_fallback: bool,
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
            texture_debug: TextureDebugSnapshot::default(),
            texture_debug_open: true,
            texture_debug_current_view_only: false,
            shader_routes_only_fallback: false,
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

    /// Stores [`FrameDiagnosticsSnapshot`] for timing, host/allocator, draws, textures, and shader routes.
    pub fn set_frame_diagnostics(&mut self, sample: FrameDiagnosticsSnapshot) {
        self.frame_diagnostics = Some(sample);
    }

    /// Stores per–render-space world transform rows for the **Scene transforms** window.
    pub fn set_scene_transforms_snapshot(&mut self, sample: SceneTransformsSnapshot) {
        self.scene_transforms = sample;
    }

    /// Stores per-texture rows for the **Textures** window.
    pub fn set_texture_debug_snapshot(&mut self, sample: TextureDebugSnapshot) {
        self.texture_debug = sample;
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

    /// Clears the **Textures** HUD payload.
    pub fn clear_texture_debug_snapshot(&mut self) {
        self.texture_debug = TextureDebugSnapshot::default();
    }

    /// Clears all HUD payloads (including frame timing).
    pub fn clear_diagnostic_snapshots(&mut self) {
        self.frame_timing = None;
        self.clear_stats_hud_payloads();
        self.clear_scene_transforms_snapshot();
        self.clear_texture_debug_snapshot();
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
    ) -> Result<(bool, bool), String> {
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.update_delta_time(delta);
        windows::apply_input(io, input);

        let (frame_timing_hud, main_hud, transforms_hud, textures_hud) = self
            .renderer_settings
            .read()
            .map(|g| {
                (
                    g.debug.debug_hud_frame_timing,
                    g.debug.debug_hud_enabled,
                    g.debug.debug_hud_transforms,
                    g.debug.debug_hud_textures,
                )
            })
            .unwrap_or((true, false, false, false));

        let any_debug_content = frame_timing_hud || main_hud || transforms_hud || textures_hud;

        let ui = self.imgui.frame();
        if any_debug_content {
            if frame_timing_hud {
                Self::frame_timing_window(ui, self.frame_timing.as_ref());
            }

            if main_hud {
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
                                Self::main_debug_panel(
                                    ui,
                                    self.latest.as_ref(),
                                    self.frame_diagnostics.as_ref(),
                                );
                            }
                            if let Some(_tab) = ui.tab_item("Shader routes") {
                                Self::shader_mappings_tab(
                                    ui,
                                    self.frame_diagnostics.as_ref(),
                                    &mut self.shader_routes_only_fallback,
                                );
                            }
                        }
                    });
            }

            if transforms_hud {
                Self::scene_transforms_window(
                    ui,
                    &self.scene_transforms,
                    &mut self.scene_transforms_open,
                );
            }

            if textures_hud {
                Self::texture_debug_window(
                    ui,
                    &self.texture_debug,
                    &mut self.texture_debug_open,
                    &mut self.texture_debug_current_view_only,
                );
            }
        }

        Self::renderer_config_window(
            ui,
            &self.renderer_settings,
            &self.config_save_path,
            &mut self.renderer_config_open,
        );

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
                .map_err(|e| format!("imgui-wgpu render: {e}"))?;
        }
        let io = self.imgui.io();
        Ok((io.want_capture_mouse, io.want_capture_keyboard))
    }
}
