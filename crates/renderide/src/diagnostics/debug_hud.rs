//! Dear ImGui overlay for developer diagnostics.
//!
//! The **Frame timing** window shows FPS and CPU/GPU submit-interval metrics (wall-clock splits around submits).
//! **[`crate::config::DebugSettings::debug_hud_frame_timing`]** toggles the **Frame timing** window (default on).
//! **[`crate::config::DebugSettings::debug_hud_enabled`]** toggles **Renderide debug** (Stats / Shader routes).
//! **[`crate::config::DebugSettings::debug_hud_transforms`]** toggles the **Scene transforms** window.

use super::frame_diagnostics_snapshot::FrameDiagnosticsSnapshot;
use super::frame_timing_hud_snapshot::FrameTimingHudSnapshot;
use super::renderer_info_snapshot::RendererInfoSnapshot;
use super::scene_transforms_snapshot::RenderSpaceTransformsSnapshot;
use super::DebugHudInput;
use super::SceneTransformsSnapshot;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::config::{save_renderer_settings, PowerPreferenceSetting, RendererSettingsHandle};

use imgui::{
    Condition, Context, Drag, FontConfig, FontSource, Io, ListClipper,
    MouseButton as ImGuiMouseButton, TableFlags, WindowFlags,
};
use imgui_wgpu::{Renderer as ImguiWgpuRenderer, RendererConfig};

use super::debug_hud_fmt as hud_fmt;
use super::debug_hud_layout as overlay_layout;

/// Dear ImGui overlay: frame timing, renderer stats, shader routes, scene transforms, and config UI.
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
    /// Live settings + persistence target for the **Renderer config** window.
    renderer_settings: RendererSettingsHandle,
    config_save_path: PathBuf,
    /// Whether the **Renderer config** window is open.
    renderer_config_open: bool,
}

fn device_type_label(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::Other => "other / unknown",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software / CPU",
    }
}

fn apply_input(io: &mut Io, input: &DebugHudInput) {
    if input.mouse_active && input.window_focused {
        io.add_mouse_pos_event(input.cursor_px);
    } else {
        io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
    }
    io.add_mouse_button_event(ImGuiMouseButton::Left, input.left);
    io.add_mouse_button_event(ImGuiMouseButton::Right, input.right);
    io.add_mouse_button_event(ImGuiMouseButton::Middle, input.middle);
    io.add_mouse_button_event(ImGuiMouseButton::Extra1, input.extra1);
    io.add_mouse_button_event(ImGuiMouseButton::Extra2, input.extra2);
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

    /// Stores [`FrameDiagnosticsSnapshot`] for timing, host/allocator, draws, textures, and shader routes.
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
        apply_input(io, input);

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

        let ui = self.imgui.frame();
        if any_debug_content {
            if frame_timing_hud {
                Self::frame_timing_window(ui, self.frame_timing.as_ref());
            }

            if main_hud {
                const PANEL_WIDTH: f32 = 760.0;
                let panel_x = (width as f32 - PANEL_WIDTH - overlay_layout::MARGIN)
                    .max(overlay_layout::MARGIN);
                let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
                    | WindowFlags::NO_RESIZE
                    | WindowFlags::NO_SAVED_SETTINGS
                    | WindowFlags::NO_FOCUS_ON_APPEARING
                    | WindowFlags::NO_NAV;

                ui.window("Renderide debug")
                    .position([panel_x, overlay_layout::MARGIN], Condition::FirstUseEver)
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
                                Self::shader_mappings_tab(ui, self.frame_diagnostics.as_ref());
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

    /// Wall-clock **FPS**, frame interval, and CPU/GPU submit splits use the same definitions as
    /// [`FrameDiagnosticsSnapshot`] (see **Frame timing** window).
    fn frame_timing_window(ui: &imgui::Ui, timing: Option<&FrameTimingHudSnapshot>) {
        let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV;
        ui.window("Frame timing")
            .position(overlay_layout::frame_timing_xy(), Condition::FirstUseEver)
            .bg_alpha(0.72)
            .flags(window_flags)
            .build(|| {
                let Some(t) = timing else {
                    ui.text("Waiting for snapshot…");
                    return;
                };
                let fps = t.fps_from_wall();
                ui.text(format!("FPS {}", hud_fmt::f64_field(8, 2, fps)));
                ui.text(format!(
                    "Frame time (ms) {}",
                    hud_fmt::f64_field(8, 3, t.wall_frame_time_ms)
                ));
                if let Some(ms) = t.cpu_frame_until_submit_ms {
                    ui.text(format!(
                        "CPU (tick to last submit) {} ms",
                        hud_fmt::f64_field(8, 3, ms)
                    ));
                } else {
                    ui.text_disabled("CPU (tick to last submit): n/a");
                }
                if let Some(ms) = t.gpu_frame_after_submit_ms {
                    ui.text(format!(
                        "GPU (last completed submit→idle) {} ms",
                        hud_fmt::f64_field(8, 3, ms)
                    ));
                } else {
                    ui.text_disabled("GPU (last completed submit→idle): n/a");
                }
            });
    }

    /// Unified IPC, adapter, scene, draws, and resources (FPS / submit timing: **Frame timing** window).
    fn main_debug_panel(
        ui: &imgui::Ui,
        renderer: Option<&RendererInfoSnapshot>,
        frame: Option<&FrameDiagnosticsSnapshot>,
    ) {
        if renderer.is_none() && frame.is_none() {
            ui.text("Waiting for snapshot…");
            return;
        }

        if let Some(r) = renderer {
            ui.text(format!(
                "Frame index {}  |  viewport {}×{}",
                r.last_frame_index, r.viewport_px.0, r.viewport_px.1
            ));
        } else if frame.is_some() {
            ui.text_disabled("Frame index / viewport: (need renderer snapshot)");
        }

        if let Some(r) = renderer {
            ui.separator();
            ui.text("GPU (adapter)");
            ui.text_wrapped(format!("Name: {}", r.adapter_name));
            ui.text(format!(
                "Class: {}  |  backend: {:?}",
                device_type_label(r.adapter_device_type),
                r.adapter_backend
            ));
            ui.text_wrapped(format!(
                "Driver: {} ({})",
                r.adapter_driver, r.adapter_driver_info
            ));
            ui.text(format!(
                "Surface: {:?}  |  present: {:?}",
                r.surface_format, r.present_mode
            ));
        }

        if let Some(f) = frame {
            ui.separator();
            ui.text("Process GPU memory (wgpu allocator)");
            match (
                f.gpu_allocator.allocated_bytes,
                f.gpu_allocator.reserved_bytes,
            ) {
                (Some(alloc), Some(resv)) => ui.text(format!(
                    "{} / {} GiB allocated / reserved",
                    hud_fmt::gib_value(7, 2, alloc),
                    hud_fmt::gib_value(7, 2, resv)
                )),
                _ => ui.text("not reported for this backend"),
            }

            ui.separator();
            ui.text("CPU / RAM (host)");
            if f.host.cpu_model.is_empty() {
                ui.text("CPU model: (unknown)");
            } else {
                ui.text_wrapped(format!("CPU model: {}", f.host.cpu_model));
            }
            ui.text(format!(
                "Logical CPUs: {:>3}  |  usage {}%",
                f.host.logical_cpus,
                hud_fmt::f64_field(6, 2, f64::from(f.host.cpu_usage_percent))
            ));
            let ram_pct = if f.host.ram_total_bytes > 0 {
                100.0 * f.host.ram_used_bytes as f64 / f.host.ram_total_bytes as f64
            } else {
                0.0
            };
            ui.text(format!(
                "RAM: {} / {} GiB  ({}%)",
                hud_fmt::gib_value(7, 2, f.host.ram_used_bytes),
                hud_fmt::gib_value(7, 2, f.host.ram_total_bytes),
                hud_fmt::f64_field(5, 1, ram_pct)
            ));
        }

        if let Some(r) = renderer {
            ui.separator();
            ui.text("IPC / init");
            ui.text(format!(
                "Connected: {}  |  init: {:?}",
                r.ipc_connected, r.init_state
            ));

            ui.separator();
            ui.text("Scene");
            ui.text(format!("Render spaces: {}", r.render_space_count));
            ui.text(format!(
                "Mesh renderables (CPU tables): {}",
                r.mesh_renderable_count
            ));
        }

        if let Some(f) = frame {
            ui.separator();
            ui.text("Batches");
            let m = &f.mesh_draw;
            ui.text(format!(
                "{:>5} total  |  {:>5} main  |  {:>5} overlay",
                m.batch_total, m.batch_main, m.batch_overlay
            ));
            ui.text("Draws");
            ui.text(format!(
                "{:>5} total  |  {:>5} main  |  {:>5} overlay",
                m.draws_total, m.draws_main, m.draws_overlay
            ));
            ui.text(format!(
                "Frustum cull: {:>5} considered  |  {:>5} culled  |  Hi-Z {:>5} culled  |  {:>5} submitted after cull",
                m.draws_pre_cull, m.draws_culled, m.draws_hi_z_culled, m.draws_total
            ));
            ui.text(format!(
                "Prep rigid {:>5}  skinned {:>5}",
                m.rigid_draws, m.skinned_draws
            ));
            ui.text(format!(
                "Last submit render_tasks: {}  |  pending camera readbacks: not implemented",
                f.last_submit_render_task_count
            ));
        }

        if let Some(r) = renderer {
            ui.separator();
            ui.text("Resources");
            if let Some(f) = frame {
                ui.text(format!("Mesh pool: {}", f.mesh_pool_entry_count));
            } else {
                ui.text(format!("Mesh pool: {}", r.resident_mesh_count));
            }
            ui.text(format!("Textures (pool): {}", r.resident_texture_count));
            let render_rt = frame
                .map(|f| f.render_textures_gpu_resident)
                .unwrap_or(r.resident_render_texture_count);
            ui.text(format!("Render textures (pool): {}", render_rt));

            ui.separator();
            ui.text("Materials (property store)");
            ui.text(format!(
                "Material property maps: {}  |  property blocks: {}  |  shader bindings: {}",
                r.material_property_slots, r.property_block_slots, r.material_shader_bindings
            ));

            ui.separator();
            ui.text("Frame graph");
            ui.text(format!(
                "Render graph passes: {}  |  GPU lights (packed): {}",
                r.frame_graph_pass_count, r.gpu_light_count
            ));
        } else if let Some(f) = frame {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", f.mesh_pool_entry_count));
            ui.text(format!("Textures (pool): {}", f.textures_gpu_resident));
            ui.text(format!(
                "Render textures (pool): {}",
                f.render_textures_gpu_resident
            ));
        }
    }

    /// Host shader asset id, logical name (or `<none>`), and material family per line (see **Shader routes** tab).
    fn shader_mappings_tab(ui: &imgui::Ui, frame: Option<&FrameDiagnosticsSnapshot>) {
        let Some(d) = frame else {
            ui.text("Waiting for frame diagnostics…");
            return;
        };
        if d.shader_route_lines.is_empty() {
            ui.text("No shader route data");
        } else {
            for line in &d.shader_route_lines {
                ui.text_wrapped(line);
            }
        }
    }

    /// Third overlay window: editable [`crate::config::RendererSettings`] with immediate disk sync.
    fn renderer_config_window(
        ui: &imgui::Ui,
        settings: &RendererSettingsHandle,
        save_path: &std::path::Path,
        open: &mut bool,
    ) {
        ui.window("Renderer config")
            .opened(open)
            .position([overlay_layout::MARGIN, overlay_layout::MARGIN], Condition::FirstUseEver)
            .size(
                [overlay_layout::RENDERER_CONFIG_W, overlay_layout::RENDERER_CONFIG_H],
                Condition::FirstUseEver,
            )
            .bg_alpha(0.88)
            .build(|| {
                ui.text_wrapped(
                    "This file is owned by the renderer. Do not edit config.toml manually while \
                     the process is running — your changes may be overwritten or lost. Use these \
                     controls instead.",
                );
                ui.separator();

                let Ok(mut g) = settings.write() else {
                    ui.text_colored([1.0, 0.4, 0.4, 1.0], "Settings store is unavailable.");
                    return;
                };

                let mut dirty = false;

                ui.text("Display");
                ui.indent();
                let mut ff = g.display.focused_fps_cap as f32;
                if Drag::new("Focused FPS cap (0 = uncapped)")
                    .range(0.0, 2000.0)
                    .speed(1.0)
                    .build(ui, &mut ff)
                {
                    g.display.focused_fps_cap = ff.round().clamp(0.0, u32::MAX as f32) as u32;
                    dirty = true;
                }
                let mut uf = g.display.unfocused_fps_cap as f32;
                if Drag::new("Unfocused FPS cap (0 = uncapped)")
                    .range(0.0, 2000.0)
                    .speed(1.0)
                    .build(ui, &mut uf)
                {
                    g.display.unfocused_fps_cap = uf.round().clamp(0.0, u32::MAX as f32) as u32;
                    dirty = true;
                }
                ui.unindent();

                ui.text("Rendering");
                ui.indent();
                if ui.checkbox("VSync", &mut g.rendering.vsync) {
                    dirty = true;
                }
                ui.text_disabled("Swapchain present mode; applies immediately (no restart).");
                ui.unindent();

                ui.text("Debug");
                ui.indent();
                if ui.checkbox("Frame timing HUD", &mut g.debug.debug_hud_frame_timing) {
                    dirty = true;
                }
                ui.text_disabled("FPS and CPU/GPU submit intervals; snapshot is cheap.");
                if ui.checkbox("Debug HUD (Stats + Shader routes)", &mut g.debug.debug_hud_enabled) {
                    dirty = true;
                }
                ui.text_disabled(
                    "Main debug panels and per-frame diagnostics capture when enabled.",
                );
                if ui.checkbox("Scene transforms HUD", &mut g.debug.debug_hud_transforms) {
                    dirty = true;
                }
                ui.text_disabled(
                    "Per-space world transform table; separate from main HUD (can be expensive on large scenes).",
                );
                if ui.checkbox("Log verbose", &mut g.debug.log_verbose) {
                    dirty = true;
                }
                if ui.checkbox("GPU validation layers", &mut g.debug.gpu_validation_layers) {
                    dirty = true;
                }
                ui.text_disabled(
                    "Vulkan validation layers significantly reduce performance; enable only when debugging. Restart required to apply (desktop and OpenXR).",
                );
                ui.text_disabled("Power preference (applies on next GPU adapter init)");
                for (i, &pref) in PowerPreferenceSetting::ALL.iter().enumerate() {
                    let _id = ui.push_id_int(i as i32);
                    if ui
                        .selectable_config(pref.label())
                        .selected(g.debug.power_preference == pref)
                        .build()
                    {
                        g.debug.power_preference = pref;
                        dirty = true;
                    }
                }
                ui.unindent();

                if dirty {
                    if let Err(e) = save_renderer_settings(save_path, &g) {
                        logger::warn!(
                            "Failed to save renderer config to {}: {e}",
                            save_path.display()
                        );
                    }
                }

                ui.separator();
                ui.text_disabled(format!("Persist: {}", save_path.display()));
            });
    }

    /// Second overlay window: one tab per render space and a clipped table of world TRS rows.
    fn scene_transforms_window(
        ui: &imgui::Ui,
        snapshot: &SceneTransformsSnapshot,
        open: &mut bool,
    ) {
        const SCENE_W: f32 = 720.0;
        const SCENE_H: f32 = 420.0;
        let scene_y = overlay_layout::scene_transforms_y(ui.io().display_size[1], SCENE_H);
        ui.window("Scene transforms")
            .opened(open)
            .position([overlay_layout::MARGIN, scene_y], Condition::FirstUseEver)
            .size([SCENE_W, SCENE_H], Condition::FirstUseEver)
            .bg_alpha(0.85)
            .build(|| {
                if snapshot.spaces.is_empty() {
                    ui.text("No render spaces.");
                    return;
                }
                if let Some(_bar) = ui.tab_bar("scene_transform_tabs") {
                    for space in &snapshot.spaces {
                        let tab_label =
                            format!("Space {}##tab_space_{}", space.space_id, space.space_id);
                        if let Some(_tab) = ui.tab_item(tab_label) {
                            Self::scene_transform_space_tab(ui, space);
                        }
                    }
                }
            });
    }

    /// Renders space header fields and the transform table for the active tab.
    fn scene_transform_space_tab(ui: &imgui::Ui, space: &RenderSpaceTransformsSnapshot) {
        ui.text(format!(
            "active={}  overlay={}  private={}",
            space.is_active, space.is_overlay, space.is_private
        ));
        let rows = &space.rows;
        let n = rows.len();
        let table_id = format!("transforms##space_{}", space.space_id);
        let table_flags = TableFlags::BORDERS
            | TableFlags::ROW_BG
            | TableFlags::SCROLL_Y
            | TableFlags::RESIZABLE
            | TableFlags::SIZING_STRETCH_PROP;
        if let Some(_table) =
            ui.begin_table_with_sizing(&table_id, 5, table_flags, [0.0, 320.0], 0.0)
        {
            ui.table_setup_column("ID");
            ui.table_setup_column("Parent");
            ui.table_setup_column("Translation (world)");
            ui.table_setup_column("Rotation (xyzw)");
            ui.table_setup_column("Scale (world)");
            ui.table_headers_row();

            let clip = ListClipper::new(n as i32);
            let tok = clip.begin(ui);
            for row_i in tok.iter() {
                let row = &rows[row_i as usize];
                ui.table_next_row();
                ui.table_next_column();
                ui.text(format!("{}", row.transform_id));
                ui.table_next_column();
                ui.text(format!("{}", row.parent_id));
                match &row.world {
                    None => {
                        ui.table_next_column();
                        ui.text_disabled("—");
                        ui.table_next_column();
                        ui.text_disabled("—");
                        ui.table_next_column();
                        ui.text_disabled("—");
                    }
                    Some(w) => {
                        ui.table_next_column();
                        ui.text(format!(
                            "{:.4}  {:.4}  {:.4}",
                            w.translation.x, w.translation.y, w.translation.z
                        ));
                        ui.table_next_column();
                        ui.text(format!(
                            "{:.4}  {:.4}  {:.4}  {:.4}",
                            w.rotation.x, w.rotation.y, w.rotation.z, w.rotation.w
                        ));
                        ui.table_next_column();
                        ui.text(format!(
                            "{:.4}  {:.4}  {:.4}",
                            w.scale.x, w.scale.y, w.scale.z
                        ));
                    }
                }
            }
        }
    }
}
