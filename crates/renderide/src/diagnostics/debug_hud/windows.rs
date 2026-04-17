//! ImGui window bodies for [`super::DebugHud`].

use crate::config::{
    save_renderer_settings, MsaaSampleCount, PowerPreferenceSetting, RendererSettingsHandle,
};
use crate::diagnostics::scene_transforms_snapshot::RenderSpaceTransformsSnapshot;
use crate::diagnostics::{
    FrameDiagnosticsSnapshot, FrameTimingHudSnapshot, RendererInfoSnapshot,
    SceneTransformsSnapshot, TextureDebugSnapshot,
};
use crate::materials::{MaterialBlendMode, RasterPipelineKind};
use crate::render_graph::WorldMeshDrawStateRow;

use imgui::{
    Condition, Drag, Io, ListClipper, MouseButton as ImGuiMouseButton, TableFlags, WindowFlags,
};

use super::fmt as hud_fmt;
use super::layout as overlay_layout;
use super::DebugHud;

fn device_type_label(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::Other => "other / unknown",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software / CPU",
    }
}

/// Frame index line and optional “waiting” stub for the Stats tab.
fn main_debug_panel_frame_line(
    ui: &imgui::Ui,
    renderer: Option<&RendererInfoSnapshot>,
    frame: Option<&FrameDiagnosticsSnapshot>,
) {
    if let Some(r) = renderer {
        ui.text(format!(
            "Frame index {}  |  viewport {}×{}",
            r.last_frame_index, r.viewport_px.0, r.viewport_px.1
        ));
    } else if frame.is_some() {
        ui.text_disabled("Frame index / viewport: (need renderer snapshot)");
    }
}

/// Adapter name, limits, and surface info (Stats tab).
fn main_debug_panel_gpu_adapter(ui: &imgui::Ui, r: &RendererInfoSnapshot) {
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
    ui.text(format!(
        "MSAA: requested {}×  |  effective {}×  |  max {}×",
        r.msaa_requested_samples, r.msaa_effective_samples, r.msaa_max_samples
    ));
    ui.text(format!(
        "MSAA (VR stereo): effective {}×  |  max {}×",
        r.msaa_effective_samples_stereo, r.msaa_max_samples_stereo
    ));
    ui.text(format!(
        "Limits: tex2d≤{}  max_buf={}  storage_bind={}  |  base_instance={}  multiview={}",
        r.gpu_max_texture_dim_2d,
        r.gpu_max_buffer_size,
        r.gpu_max_storage_binding,
        r.gpu_supports_base_instance,
        r.gpu_supports_multiview
    ));
}

/// Allocator and host CPU / RAM lines (Stats tab).
fn main_debug_panel_host_and_allocator(ui: &imgui::Ui, f: &FrameDiagnosticsSnapshot) {
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

/// IPC connection, init state, and coarse scene counts (Stats tab).
fn main_debug_panel_ipc_and_scene(ui: &imgui::Ui, r: &RendererInfoSnapshot) {
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

/// Draw batches, instance batches, and cull stats (Stats tab).
fn main_debug_panel_draw_stats(ui: &imgui::Ui, f: &FrameDiagnosticsSnapshot) {
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
        "GPU instance batches (indexed submits): {:>5}",
        m.instance_batch_total
    ));
    ui.text(format!(
        "Pipeline pass submits: {:>5}",
        m.submitted_pipeline_pass_total
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

/// Pool counts, materials, and render graph summary (Stats tab).
fn main_debug_panel_resources_and_graph(
    ui: &imgui::Ui,
    renderer: Option<&RendererInfoSnapshot>,
    frame: Option<&FrameDiagnosticsSnapshot>,
) {
    match (renderer, frame) {
        (Some(r), Some(f)) => {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", f.mesh_pool_entry_count));
            ui.text(format!("Textures (pool): {}", r.resident_texture_count));
            ui.text(format!(
                "Render textures (pool): {}",
                f.render_textures_gpu_resident
            ));

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
        }
        (Some(r), None) => {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", r.resident_mesh_count));
            ui.text(format!("Textures (pool): {}", r.resident_texture_count));
            ui.text(format!(
                "Render textures (pool): {}",
                r.resident_render_texture_count
            ));

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
        }
        (None, Some(f)) => {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", f.mesh_pool_entry_count));
            ui.text(format!("Textures (pool): {}", f.textures_gpu_resident));
            ui.text(format!(
                "Render textures (pool): {}",
                f.render_textures_gpu_resident
            ));
        }
        (None, None) => {}
    }
}

/// Focused / unfocused FPS caps (Renderer config window).
fn renderer_config_display_section(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Display");
    ui.indent();
    let mut ff = g.display.focused_fps_cap as f32;
    if Drag::new("Focused FPS cap (0 = uncapped)")
        .range(0.0, 2000.0)
        .speed(1.0)
        .build(ui, &mut ff)
    {
        g.display.focused_fps_cap = ff.round().clamp(0.0, u32::MAX as f32) as u32;
        *dirty = true;
    }
    let mut uf = g.display.unfocused_fps_cap as f32;
    if Drag::new("Unfocused FPS cap (0 = uncapped)")
        .range(0.0, 2000.0)
        .speed(1.0)
        .build(ui, &mut uf)
    {
        g.display.unfocused_fps_cap = uf.round().clamp(0.0, u32::MAX as f32) as u32;
        *dirty = true;
    }
    ui.unindent();
}

/// VSync toggle (Renderer config window).
fn renderer_config_rendering_section(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Rendering");
    ui.indent();
    if ui.checkbox("VSync", &mut g.rendering.vsync) {
        *dirty = true;
    }
    ui.text_disabled("Swapchain present mode; applies immediately (no restart).");
    ui.text_disabled("MSAA (main window forward path; clamped to GPU max).");
    for (i, &msaa) in MsaaSampleCount::ALL.iter().enumerate() {
        let _id = ui.push_id_int(i as i32);
        if ui
            .selectable_config(msaa.label())
            .selected(g.rendering.msaa == msaa)
            .build()
        {
            g.rendering.msaa = msaa;
            *dirty = true;
        }
    }
    ui.unindent();
}

/// Debug HUD toggles, logging, validation layers, power preference (Renderer config window).
fn renderer_config_debug_section(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Debug");
    ui.indent();
    if ui.checkbox("Frame timing HUD", &mut g.debug.debug_hud_frame_timing) {
        *dirty = true;
    }
    ui.text_disabled("FPS and CPU/GPU submit intervals; snapshot is cheap.");
    if ui.checkbox(
        "Debug HUD (Stats / Shader routes / Draw state / GPU memory)",
        &mut g.debug.debug_hud_enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled("Main debug panels and per-frame diagnostics capture when enabled.");
    if ui.checkbox("Scene transforms HUD", &mut g.debug.debug_hud_transforms) {
        *dirty = true;
    }
    ui.text_disabled(
        "Per-space world transform table; separate from main HUD (can be expensive on large scenes).",
    );
    if ui.checkbox("Textures HUD", &mut g.debug.debug_hud_textures) {
        *dirty = true;
    }
    ui.text_disabled("Texture pool rows and current-view usage; can be noisy in large scenes.");
    if ui.checkbox("Log verbose", &mut g.debug.log_verbose) {
        *dirty = true;
    }
    if ui.checkbox("GPU validation layers", &mut g.debug.gpu_validation_layers) {
        *dirty = true;
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
            *dirty = true;
        }
    }
    ui.unindent();
}

/// Body of **Renderer config**: grouped controls and immediate save.
fn renderer_config_panel_body(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    save_path: &std::path::Path,
) {
    let mut dirty = false;
    renderer_config_display_section(ui, g, &mut dirty);
    renderer_config_rendering_section(ui, g, &mut dirty);
    renderer_config_debug_section(ui, g, &mut dirty);

    if dirty {
        if let Err(e) = save_renderer_settings(save_path, g) {
            logger::warn!(
                "Failed to save renderer config to {}: {e}",
                save_path.display()
            );
        }
    }

    ui.separator();
    ui.text_disabled(format!("Persist: {}", save_path.display()));
}

/// Feeds winit-derived [`crate::diagnostics::DebugHudInput`] into ImGui `io` before each frame.
pub(super) fn apply_input(io: &mut Io, input: &crate::diagnostics::DebugHudInput) {
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

fn pipeline_label(pipeline: &RasterPipelineKind) -> String {
    match pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => stem.to_string(),
        RasterPipelineKind::DebugWorldNormals => "debug_world_normals".to_string(),
    }
}

fn draw_state_is_uiish(row: &WorldMeshDrawStateRow) -> bool {
    row.is_overlay
        || row.alpha_blended
        || matches!(
            &row.pipeline,
            RasterPipelineKind::EmbeddedStem(stem)
                if stem.starts_with("ui_")
                    || stem.contains("text")
                    || stem.contains("overlay")
        )
}

fn draw_state_has_override(row: &WorldMeshDrawStateRow) -> bool {
    row.depth_write.is_some()
        || row.depth_compare.is_some()
        || row.depth_offset.is_some()
        || row.color_mask.is_some()
        || row.stencil_enabled
}

fn blend_mode_label(mode: MaterialBlendMode) -> String {
    match mode {
        MaterialBlendMode::StemDefault => "stem".to_string(),
        MaterialBlendMode::Opaque => "opaque".to_string(),
        MaterialBlendMode::Cutout => "cutout".to_string(),
        MaterialBlendMode::Alpha => "alpha".to_string(),
        MaterialBlendMode::Transparent => "transparent".to_string(),
        MaterialBlendMode::Additive => "additive".to_string(),
        MaterialBlendMode::Multiply => "multiply".to_string(),
        MaterialBlendMode::UnityBlend { src, dst } => format!("unity {src}/{dst}"),
    }
}

fn color_mask_label(mask: Option<u8>) -> String {
    let Some(mask) = mask else {
        return "pass".to_string();
    };
    let mut out = String::new();
    if mask & 8 != 0 {
        out.push('R');
    }
    if mask & 4 != 0 {
        out.push('G');
    }
    if mask & 2 != 0 {
        out.push('B');
    }
    if mask & 1 != 0 {
        out.push('A');
    }
    if out.is_empty() {
        "none".to_string()
    } else {
        out
    }
}

fn stencil_label(row: &WorldMeshDrawStateRow) -> String {
    if !row.stencil_enabled {
        return "off".to_string();
    }
    format!(
        "ref={} cmp={} pass={} read=0x{:02x} write=0x{:02x}",
        row.stencil_reference,
        row.stencil_compare,
        row.stencil_pass_op,
        row.stencil_read_mask,
        row.stencil_write_mask
    )
}

fn ztest_label(value: Option<u8>) -> &'static str {
    match value {
        Some(0) => "off",
        Some(1) => "never",
        Some(2) => "less",
        Some(3) => "equal",
        Some(4) => "lequal",
        Some(5) => "greater",
        Some(6) => "not-equal",
        Some(7) => "gequal",
        Some(8) => "always",
        Some(_) => "invalid",
        None => "pass",
    }
}

fn offset_label(value: Option<(u32, i32)>) -> String {
    match value {
        Some((factor_bits, units)) => format!("{:.3}/{}", f32::from_bits(factor_bits), units),
        None => "pass".to_string(),
    }
}

impl DebugHud {
    /// Wall-clock **FPS**, frame interval, and CPU/GPU submit splits use the same definitions as
    /// [`FrameDiagnosticsSnapshot`] (see **Frame timing** window).
    pub(super) fn frame_timing_window(ui: &imgui::Ui, timing: Option<&FrameTimingHudSnapshot>) {
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
    pub(super) fn main_debug_panel(
        ui: &imgui::Ui,
        renderer: Option<&RendererInfoSnapshot>,
        frame: Option<&FrameDiagnosticsSnapshot>,
    ) {
        if renderer.is_none() && frame.is_none() {
            ui.text("Waiting for snapshot…");
            return;
        }

        main_debug_panel_frame_line(ui, renderer, frame);

        if let Some(r) = renderer {
            main_debug_panel_gpu_adapter(ui, r);
        }

        if let Some(f) = frame {
            main_debug_panel_host_and_allocator(ui, f);
        }

        if let Some(r) = renderer {
            main_debug_panel_ipc_and_scene(ui, r);
        }

        if let Some(f) = frame {
            main_debug_panel_draw_stats(ui, f);
        }

        main_debug_panel_resources_and_graph(ui, renderer, frame);
    }

    /// Host shader asset id, logical name (or `<none>`), and material family per line (see **Shader routes** tab).
    pub(super) fn shader_mappings_tab(
        ui: &imgui::Ui,
        frame: Option<&FrameDiagnosticsSnapshot>,
        only_fallback: &mut bool,
    ) {
        let Some(d) = frame else {
            ui.text("Waiting for frame diagnostics…");
            return;
        };
        ui.checkbox("Only fallback routes", only_fallback);
        if d.shader_routes.is_empty() {
            ui.text("No shader route data");
        } else {
            for route in &d.shader_routes {
                if *only_fallback && route.implemented {
                    continue;
                }
                ui.text_wrapped(format!(
                    "{}  {}  {}  {}",
                    route.shader_asset_id,
                    route.display_name.as_deref().unwrap_or("<none>"),
                    route.pipeline_label,
                    if route.implemented {
                        "implemented"
                    } else {
                        "fallback"
                    },
                ));
            }
        }
    }

    /// Sorted draw rows with runtime material state.
    pub(super) fn draw_state_tab(
        ui: &imgui::Ui,
        frame: Option<&FrameDiagnosticsSnapshot>,
        ui_only: &mut bool,
        only_overrides: &mut bool,
    ) {
        let Some(d) = frame else {
            ui.text("Waiting for frame diagnostics");
            return;
        };
        ui.checkbox("Only UI / alpha rows", ui_only);
        ui.checkbox("Only render-state overrides", only_overrides);

        let rows: Vec<&WorldMeshDrawStateRow> = d
            .draw_state_rows
            .iter()
            .filter(|row| !*ui_only || draw_state_is_uiish(row))
            .filter(|row| !*only_overrides || draw_state_has_override(row))
            .collect();
        ui.text(format!(
            "{} rows ({} submitted)",
            rows.len(),
            d.draw_state_rows.len()
        ));

        let table_flags = TableFlags::BORDERS
            | TableFlags::ROW_BG
            | TableFlags::SCROLL_Y
            | TableFlags::RESIZABLE
            | TableFlags::SIZING_STRETCH_PROP;
        if let Some(_table) =
            ui.begin_table_with_sizing("draw_state_rows", 11, table_flags, [0.0, 360.0], 0.0)
        {
            ui.table_setup_column("Draw");
            ui.table_setup_column("Node");
            ui.table_setup_column("Mesh");
            ui.table_setup_column("Material");
            ui.table_setup_column("Pipeline");
            ui.table_setup_column("Blend");
            ui.table_setup_column("ZWrite");
            ui.table_setup_column("ZTest");
            ui.table_setup_column("Offset");
            ui.table_setup_column("Color");
            ui.table_setup_column("Stencil");
            ui.table_headers_row();
            let clip = ListClipper::new(rows.len() as i32);
            let tok = clip.begin(ui);
            for row_i in tok.iter() {
                let row = rows[row_i as usize];
                ui.table_next_row();
                ui.table_next_column();
                ui.text(format!("{}", row.draw_index));
                ui.table_next_column();
                ui.text(format!("{}", row.node_id));
                ui.table_next_column();
                ui.text(format!("{}:{}", row.mesh_asset_id, row.slot_index));
                ui.table_next_column();
                ui.text(format!(
                    "{} / {:?}",
                    row.material_asset_id, row.property_block_slot0
                ));
                ui.table_next_column();
                ui.text_wrapped(pipeline_label(&row.pipeline));
                ui.table_next_column();
                ui.text(blend_mode_label(row.blend_mode));
                ui.table_next_column();
                ui.text(match row.depth_write {
                    Some(true) => "on",
                    Some(false) => "off",
                    None => "pass",
                });
                ui.table_next_column();
                ui.text(ztest_label(row.depth_compare));
                ui.table_next_column();
                ui.text(offset_label(row.depth_offset));
                ui.table_next_column();
                ui.text(color_mask_label(row.color_mask));
                ui.table_next_column();
                ui.text_wrapped(stencil_label(row));
            }
        }
    }

    /// Texture pool window with current-view filtering.
    pub(super) fn texture_debug_window(
        ui: &imgui::Ui,
        snapshot: &TextureDebugSnapshot,
        open: &mut bool,
        current_view_only: &mut bool,
    ) {
        ui.window("Textures")
            .opened(open)
            .position(
                [overlay_layout::MARGIN, overlay_layout::MARGIN + 360.0],
                Condition::FirstUseEver,
            )
            .size([860.0, 420.0], Condition::FirstUseEver)
            .bg_alpha(0.85)
            .build(|| {
                ui.checkbox("Only current view", current_view_only);
                ui.text(format!(
                    "{} textures  |  {} current-view  |  {} total",
                    snapshot.rows.len(),
                    snapshot.current_view_texture_count,
                    hud_fmt::bytes_compact(snapshot.total_resident_bytes)
                ));
                let rows: Vec<_> = snapshot
                    .rows
                    .iter()
                    .filter(|row| !*current_view_only || row.used_by_current_view)
                    .collect();
                let table_flags = TableFlags::BORDERS
                    | TableFlags::ROW_BG
                    | TableFlags::SCROLL_Y
                    | TableFlags::RESIZABLE
                    | TableFlags::SIZING_STRETCH_PROP;
                if let Some(_table) = ui.begin_table_with_sizing(
                    "texture_debug_rows",
                    8,
                    table_flags,
                    [0.0, 330.0],
                    0.0,
                ) {
                    ui.table_setup_column("Asset");
                    ui.table_setup_column("Size");
                    ui.table_setup_column("Mips");
                    ui.table_setup_column("Bytes");
                    ui.table_setup_column("Host");
                    ui.table_setup_column("GPU");
                    ui.table_setup_column("Sampler");
                    ui.table_setup_column("View");
                    ui.table_headers_row();
                    let clip = ListClipper::new(rows.len() as i32);
                    let tok = clip.begin(ui);
                    for row_i in tok.iter() {
                        let row = rows[row_i as usize];
                        ui.table_next_row();
                        ui.table_next_column();
                        ui.text(format!("{}", row.asset_id));
                        ui.table_next_column();
                        ui.text(format!("{}x{}", row.width, row.height));
                        ui.table_next_column();
                        ui.text(format!(
                            "{}/{}",
                            row.mip_levels_resident, row.mip_levels_total
                        ));
                        ui.table_next_column();
                        ui.text(hud_fmt::bytes_compact(row.resident_bytes));
                        ui.table_next_column();
                        ui.text(format!("{:?} {:?}", row.host_format, row.color_profile));
                        ui.table_next_column();
                        ui.text(format!("{:?}", row.wgpu_format));
                        ui.table_next_column();
                        ui.text(format!(
                            "{:?} aniso={} wrap={:?}/{:?} bias={:.2}",
                            row.filter_mode,
                            row.aniso_level,
                            row.wrap_u,
                            row.wrap_v,
                            row.mipmap_bias
                        ));
                        ui.table_next_column();
                        ui.text(if row.used_by_current_view {
                            "current"
                        } else {
                            ""
                        });
                    }
                }
            });
    }

    /// Full [`wgpu::AllocatorReport`] from [`FrameDiagnosticsSnapshot::gpu_allocator_report`], refreshed on a timer.
    ///
    /// Row labels mirror wgpu `label` strings on buffers and textures (often empty).
    pub(super) fn gpu_memory_tab(ui: &imgui::Ui, frame: Option<&FrameDiagnosticsSnapshot>) {
        let Some(d) = frame else {
            ui.text("Waiting for frame diagnostics…");
            return;
        };

        ui.text_disabled(format!(
            "Next full report refresh in ~{:.1} s (detail lags Stats totals by up to one interval).",
            d.gpu_allocator_report_next_refresh_in_secs
        ));

        let Some(hud) = &d.gpu_allocator_report else {
            ui.separator();
            ui.text_wrapped(
                "Full allocator report unavailable: unsupported backend, or not yet collected. \
                 The Stats tab still shows totals when the device reports them.",
            );
            return;
        };

        let r = hud.report.as_ref();

        ui.separator();
        ui.text("Summary (wgpu device allocator)");
        ui.text(format!(
            "{} / {}  allocated / reserved  |  {} blocks  |  {} sub-allocations",
            hud_fmt::bytes_compact(r.total_allocated_bytes),
            hud_fmt::bytes_compact(r.total_reserved_bytes),
            r.blocks.len(),
            r.allocations.len(),
        ));
        ui.text_disabled(
            "Sizes are device-local sub-allocations; Vulkan memory-type names are not exposed here.",
        );

        ui.separator();
        ui.text("Sub-allocations (by size, largest first)");
        let n = hud.allocation_indices_by_size.len();
        let table_flags = TableFlags::BORDERS
            | TableFlags::ROW_BG
            | TableFlags::SCROLL_Y
            | TableFlags::RESIZABLE
            | TableFlags::SIZING_STRETCH_PROP;
        if let Some(_table) =
            ui.begin_table_with_sizing("gpu_alloc_rows", 3, table_flags, [0.0, 360.0], 0.0)
        {
            ui.table_setup_column("Size");
            ui.table_setup_column("Offset");
            ui.table_setup_column("Label");
            ui.table_headers_row();
            let clip = ListClipper::new(n as i32);
            let tok = clip.begin(ui);
            for row_i in tok.iter() {
                let idx = hud.allocation_indices_by_size[row_i as usize];
                let a = &r.allocations[idx];
                ui.table_next_row();
                ui.table_next_column();
                ui.text(hud_fmt::bytes_compact(a.size));
                ui.table_next_column();
                ui.text(format!("{}", a.offset));
                let name = if a.name.is_empty() {
                    "(no label)"
                } else {
                    a.name.as_str()
                };
                ui.table_next_column();
                ui.text_wrapped(name);
            }
        }

        ui.separator();
        if let Some(_node) = ui.tree_node("Memory blocks") {
            let nb = r.blocks.len();
            if let Some(_table) = ui.begin_table_with_sizing(
                "gpu_mem_blocks",
                3,
                TableFlags::BORDERS | TableFlags::ROW_BG | TableFlags::SIZING_STRETCH_PROP,
                [0.0, 200.0],
                0.0,
            ) {
                ui.table_setup_column("Block");
                ui.table_setup_column("Size");
                ui.table_setup_column("Sub-allocs");
                ui.table_headers_row();
                for bi in 0..nb {
                    let b = &r.blocks[bi];
                    let sub = b.allocations.end.saturating_sub(b.allocations.start);
                    ui.table_next_row();
                    ui.table_next_column();
                    ui.text(format!("{bi}"));
                    ui.table_next_column();
                    ui.text(hud_fmt::bytes_compact(b.size));
                    ui.table_next_column();
                    ui.text(format!("{sub}"));
                }
            }
        }
    }

    /// Third overlay window: editable [`crate::config::RendererSettings`] with immediate disk sync.
    pub(super) fn renderer_config_window(
        ui: &imgui::Ui,
        settings: &RendererSettingsHandle,
        save_path: &std::path::Path,
        open: &mut bool,
    ) {
        ui.window("Renderer config")
            .opened(open)
            .position(
                [overlay_layout::MARGIN, overlay_layout::MARGIN],
                Condition::FirstUseEver,
            )
            .size(
                [
                    overlay_layout::RENDERER_CONFIG_W,
                    overlay_layout::RENDERER_CONFIG_H,
                ],
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

                renderer_config_panel_body(ui, &mut g, save_path);
            });
    }

    /// Second overlay window: one tab per render space and a clipped table of world TRS rows.
    pub(super) fn scene_transforms_window(
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
    pub(super) fn scene_transform_space_tab(ui: &imgui::Ui, space: &RenderSpaceTransformsSnapshot) {
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
