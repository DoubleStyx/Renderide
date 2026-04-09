//! Dear ImGui overlay for developer diagnostics (feature `debug-hud`).
//!
//! The **Renderide debug** window uses a **Stats** tab (unified renderer + frame diagnostics) and a
//! **Shader routes** tab for host shader → material family mappings.

#[cfg(feature = "debug-hud")]
use super::frame_diagnostics_snapshot::FrameDiagnosticsSnapshot;
use super::renderer_info_snapshot::RendererInfoSnapshot;
#[cfg(feature = "debug-hud")]
use super::scene_transforms_snapshot::RenderSpaceTransformsSnapshot;
use super::DebugHudInput;
use super::SceneTransformsSnapshot;

#[cfg(feature = "debug-hud")]
use std::path::PathBuf;
#[cfg(feature = "debug-hud")]
use std::time::{Duration, Instant};

#[cfg(feature = "debug-hud")]
use crate::config::{save_renderer_settings, PowerPreferenceSetting, RendererSettingsHandle};

#[cfg(feature = "debug-hud")]
use imgui::{
    Condition, Context, Drag, FontConfig, FontSource, Io, ListClipper,
    MouseButton as ImGuiMouseButton, TableFlags, TreeNodeFlags, WindowFlags,
};
#[cfg(feature = "debug-hud")]
use imgui_wgpu::{Renderer as ImguiWgpuRenderer, RendererConfig};

#[cfg(feature = "debug-hud")]
/// Right-aligned numeric [`format!`] helpers so HUD columns keep a stable width.
mod hud_fmt {
    /// Formats `value` as a right-aligned decimal with `decimals` places and total width `width`.
    pub fn f64_field(width: usize, decimals: usize, value: f64) -> String {
        format!("{value:>w$.d$}", w = width, d = decimals)
    }

    /// Human-readable gibibytes from bytes (numeric part only; caller adds `GiB` suffix).
    pub fn gib_value(width: usize, decimals: usize, bytes: u64) -> String {
        let g = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        f64_field(width, decimals, g)
    }
}

#[cfg(feature = "debug-hud")]
/// Per-property implementation status for the **Shader detail** tab.
mod shader_props {
    /// Implementation status of a single material property / feature.
    #[derive(Clone, Copy)]
    pub enum Status {
        /// Fully implemented and active in the shader.
        Ok,
        /// Code path exists but behaviour is restricted (e.g. a second UV that always mirrors UV0).
        Partial,
        /// Struct field or texture binding is present; shader logic not yet written.
        NotYet,
    }

    /// One row in the per-shader property table.
    pub struct Row {
        pub name: &'static str,
        pub status: Status,
        pub note: &'static str,
    }

    /// Returns `(short description, property rows)` for a given embedded shader stem,
    /// or `None` for unknown stems.
    pub fn rows_for_stem(stem: &str) -> Option<(&'static str, Vec<Row>)> {
        if stem.starts_with("unlit_") {
            return Some((
                "World Unlit — texture × tint, optional alpha test",
                vec![
                    Row {
                        name: "_Color",
                        status: Status::Ok,
                        note: "base tint; multiplied with texture",
                    },
                    Row {
                        name: "_Tex / _Tex_ST",
                        status: Status::Ok,
                        note: "albedo; UV scale+offset; V-flip applied so imagery matches host",
                    },
                    Row {
                        name: "_Cutoff",
                        status: Status::Ok,
                        note: "alpha discard (flag bit 1)",
                    },
                    Row {
                        name: "Vertex color",
                        status: Status::NotYet,
                        note: "no color vertex stream; treated as white",
                    },
                ],
            ));
        }
        if stem.starts_with("pbsmetallic_") {
            return Some((
                "PBS Metallic — Cook–Torrance BRDF, clustered forward lighting",
                vec![
                    Row { name: "_Color",                                      status: Status::Ok,      note: "base tint" },
                    Row { name: "_MainTex",                                    status: Status::Ok,      note: "albedo texture" },
                    Row { name: "_Cutoff",                                     status: Status::Ok,      note: "alpha discard" },
                    Row { name: "_Metallic / _MetallicGlossMap",               status: Status::Ok,      note: "metallic from uniform × map.r" },
                    Row { name: "_Glossiness / _GlossMapScale",                status: Status::Ok,      note: "smoothness → roughness (clamped min 0.045)" },
                    Row { name: "_SmoothnessTextureChannel",                   status: Status::Ok,      note: "selects metallic map alpha vs albedo alpha for smoothness" },
                    Row { name: "_BumpScale / _BumpMap",                       status: Status::Ok,      note: "normal map via orthonormal TBN; scale applied to XY" },
                    Row { name: "_Parallax",                                   status: Status::NotYet,  note: "in uniform struct; no parallax offset logic in vs_main" },
                    Row { name: "_OcclusionStrength / _OcclusionMap",          status: Status::Ok,      note: "AO from red channel; lerped with strength" },
                    Row { name: "_EmissionColor / _EmissionMap",               status: Status::Ok,      note: "additive emission; map.rgb × EmissionColor" },
                    Row { name: "_DetailAlbedoMap",                            status: Status::Partial, note: "sampled × 2 and multiplied into base color; UV1 is always a copy of UV0" },
                    Row { name: "_DetailNormalMap / _DetailNormalMapScale",     status: Status::NotYet,  note: "texture bound at group(1); not applied in fs_main" },
                    Row { name: "_UVSec",                                      status: Status::Partial, note: "select-UV1 logic present; TEXCOORD1 is always a copy of TEXCOORD0" },
                    Row { name: "_SpecularHighlights",                         status: Status::Ok,      note: "toggles Cook–Torrance specular term off (uses diffuse-only path)" },
                    Row { name: "_GlossyReflections",                          status: Status::Partial, note: "toggles 0.03 ambient factor off; no actual reflection probes" },
                    Row { name: "ForwardAdd passes",                           status: Status::NotYet,  note: "clustered forward only; ForwardAdd multi-pass not implemented" },
                    Row { name: "Lightmaps",                                   status: Status::NotYet,  note: "not implemented" },
                    Row { name: "Reflection probes",                           status: Status::NotYet,  note: "not implemented" },
                ],
            ));
        }
        if stem.starts_with("pbsspecular_") {
            return Some((
                "PBS Specular — Cook–Torrance BRDF with tinted specular, clustered forward",
                vec![
                    Row {
                        name: "_Color",
                        status: Status::Ok,
                        note: "base tint",
                    },
                    Row {
                        name: "_MainTex",
                        status: Status::Ok,
                        note: "albedo texture",
                    },
                    Row {
                        name: "_Cutoff",
                        status: Status::Ok,
                        note: "alpha discard",
                    },
                    Row {
                        name: "_SpecColor / _SpecGlossMap",
                        status: Status::Ok,
                        note: "specular tint × map.rgb; diffuse energy via oneMinusReflectivity",
                    },
                    Row {
                        name: "_Glossiness / _GlossMapScale",
                        status: Status::Ok,
                        note: "smoothness → roughness (clamped min 0.045)",
                    },
                    Row {
                        name: "_SmoothnessTextureChannel",
                        status: Status::Ok,
                        note: "selects spec-gloss map alpha vs albedo alpha for smoothness",
                    },
                    Row {
                        name: "_BumpScale / _BumpMap",
                        status: Status::Ok,
                        note: "normal map via orthonormal TBN",
                    },
                    Row {
                        name: "_Parallax",
                        status: Status::NotYet,
                        note: "in uniform struct; no parallax offset logic",
                    },
                    Row {
                        name: "_OcclusionStrength / _OcclusionMap",
                        status: Status::Ok,
                        note: "AO from red channel",
                    },
                    Row {
                        name: "_EmissionColor / _EmissionMap",
                        status: Status::Ok,
                        note: "additive emission",
                    },
                    Row {
                        name: "_DetailAlbedoMap",
                        status: Status::Partial,
                        note: "sampled × 2 and multiplied; UV1 is always a copy of UV0",
                    },
                    Row {
                        name: "_DetailNormalMap / _DetailNormalMapScale",
                        status: Status::NotYet,
                        note: "texture bound at group(1); not applied in fs_main",
                    },
                    Row {
                        name: "_UVSec",
                        status: Status::Partial,
                        note: "select-UV1 logic present; TEXCOORD1 = TEXCOORD0 copy",
                    },
                    Row {
                        name: "_SpecularHighlights",
                        status: Status::Ok,
                        note:
                            "toggles Cook–Torrance specular term (uses diffuse-only specular path)",
                    },
                    Row {
                        name: "_GlossyReflections",
                        status: Status::Partial,
                        note: "toggles 0.03 ambient factor; no actual reflection probes",
                    },
                    Row {
                        name: "ForwardAdd passes",
                        status: Status::NotYet,
                        note: "clustered forward only; ForwardAdd not implemented",
                    },
                    Row {
                        name: "Lightmaps",
                        status: Status::NotYet,
                        note: "not implemented",
                    },
                    Row {
                        name: "Reflection probes",
                        status: Status::NotYet,
                        note: "not implemented",
                    },
                ],
            ));
        }
        if stem.starts_with("debug_world_normals_") {
            return Some((
                "Debug World Normals — fallback; outputs world-space normals as RGB",
                vec![
                    Row {
                        name: "(no material properties)",
                        status: Status::Ok,
                        note: "world_n * 0.5 + 0.5 → RGB; used when no shader route is mapped for a host shader id",
                    },
                ],
            ));
        }
        if stem.starts_with("ui_unlit_") {
            return Some((
                "UI Unlit — canvas sprite, tint, mask, rect clip",
                vec![
                    Row { name: "_Tint",                                       status: Status::Ok,      note: "base color tint" },
                    Row { name: "_MainTex / _MainTex_ST",                      status: Status::Ok,      note: "sprite texture; UV scale+offset; V-flip (enabled by flag bit 0)" },
                    Row { name: "_Cutoff",                                     status: Status::Ok,      note: "alpha clip on final alpha (flag bit 1) or mask grayscale (flag bit 5)" },
                    Row { name: "_Rect",                                       status: Status::Ok,      note: "object-space rect discard; fragments outside XY range discarded (flag bit 2)" },
                    Row { name: "_MaskTex / _MaskTex_ST",                      status: Status::Ok,      note: "grayscale mask; multiply alpha into color.a (flag bit 4); clip vs _Cutoff (flag bit 5)" },
                    Row { name: "_OverlayTint",                                status: Status::Partial, note: "alpha-weighted RGB tint (flag bit 3); no scene depth composite implemented" },
                    Row { name: "_SrcBlend / _DstBlend",                       status: Status::Partial, note: "pipeline uses fixed alpha blending for UI stems; does not honor per-material blend factors" },
                    Row { name: "_ZWrite / _Cull",                             status: Status::NotYet,  note: "in uniform struct; pipeline does not honor per-material rasterizer state" },
                    Row { name: "Stencil ops (_Stencil, _StencilOp, …)",       status: Status::NotYet,  note: "in uniform struct; pipeline stencil state is fixed per pass" },
                    Row { name: "_ColorMask",                                  status: Status::NotYet,  note: "in uniform struct; color write mask is fixed" },
                    Row { name: "Vertex color",                                status: Status::Ok,      note: "float4 vertex color stream multiplied by _Tint; defaults to opaque white when mesh color is absent" },
                ],
            ));
        }
        if stem.starts_with("ui_textunlit_") {
            return Some((
                "UI Text Unlit — MSDF / SDF / Raster font atlas, outline, rect clip",
                vec![
                    Row {
                        name: "_TintColor",
                        status: Status::Ok,
                        note: "text face color multiplied by vertex color",
                    },
                    Row {
                        name: "_FontAtlas",
                        status: Status::Ok,
                        note: "font texture atlas",
                    },
                    Row {
                        name: "_TextMode",
                        status: Status::Ok,
                        note:
                            "0 = MSDF (median RGB), 1 = RASTER (atlas × tint), 2 = SDF (alpha dist)",
                    },
                    Row {
                        name: "_Range",
                        status: Status::Ok,
                        note: "pixel range for SDF/MSDF AA scale (from fwidth)",
                    },
                    Row {
                        name: "_FaceDilate / _FaceSoftness",
                        status: Status::Ok,
                        note: "SDF/MSDF face expansion and edge softness",
                    },
                    Row {
                        name: "_OutlineColor / _OutlineSize",
                        status: Status::Ok,
                        note: "SDF/MSDF outline; per-vertex scale from extra_data.y",
                    },
                    Row {
                        name: "_BackgroundColor",
                        status: Status::Ok,
                        note: "glyph background fill in SDF/MSDF modes",
                    },
                    Row {
                        name: "_Rect / _RectClip",
                        status: Status::Ok,
                        note: "object-space rect discard when _RectClip > 0.5 and rect has area",
                    },
                    Row {
                        name: "_OverlayTint",
                        status: Status::Partial,
                        note: "alpha-weighted RGB tint; no scene depth composite",
                    },
                    Row {
                        name: "_SrcBlend / _DstBlend / _ZWrite / _Cull / _ZTest",
                        status: Status::Partial,
                        note: "pipeline uses fixed alpha blending for UI stems; depth/cull/ztest are still fixed",
                    },
                    Row {
                        name: "Stencil ops (_Stencil, _StencilOp, …)",
                        status: Status::NotYet,
                        note: "in uniform struct; pipeline stencil state is fixed",
                    },
                    Row {
                        name: "_ColorMask",
                        status: Status::NotYet,
                        note: "in uniform struct; write mask is fixed",
                    },
                    Row {
                        name: "Vertex color",
                        status: Status::Ok,
                        note: "float4 vertex color stream is provided; defaults to opaque white when absent",
                    },
                ],
            ));
        }
        None
    }
}

#[cfg(feature = "debug-hud")]
pub struct DebugHud {
    imgui: Context,
    renderer: ImguiWgpuRenderer,
    last_frame_at: Instant,
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

#[cfg(feature = "debug-hud")]
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

#[cfg(feature = "debug-hud")]
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
            latest: None,
            frame_diagnostics: None,
            scene_transforms: SceneTransformsSnapshot::default(),
            scene_transforms_open: true,
            renderer_settings,
            config_save_path,
            renderer_config_open: true,
        }
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

    /// Records ImGui into `encoder` as a load-on-top pass over `backbuffer`.
    pub fn encode_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        (width, height): (u32, u32),
        input: &DebugHudInput,
    ) -> Result<(), String> {
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.update_delta_time(delta);
        apply_input(io, input);

        let snapshot = self.latest.clone();
        let frame_diag = self.frame_diagnostics.clone();
        let scene_transforms = self.scene_transforms.clone();
        let ui = self.imgui.frame();
        const PANEL_WIDTH: f32 = 760.0;
        let panel_x = (width as f32 - PANEL_WIDTH - 12.0).max(12.0);
        let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
            | WindowFlags::NO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV;

        ui.window("Renderide debug")
            .position([panel_x, 12.0], Condition::FirstUseEver)
            .size_constraints([PANEL_WIDTH, 0.0], [PANEL_WIDTH, 1.0e9])
            .bg_alpha(0.72)
            .flags(window_flags)
            .build(|| {
                if let Some(_tab_bar) = ui.tab_bar("debug_tabs") {
                    if let Some(_tab) = ui.tab_item("Stats") {
                        Self::main_debug_panel(ui, snapshot.as_ref(), frame_diag.as_ref());
                    }
                    if let Some(_tab) = ui.tab_item("Shader routes") {
                        Self::shader_mappings_tab(ui, frame_diag.as_ref());
                    }
                    if let Some(_tab) = ui.tab_item("Shader detail") {
                        Self::shader_detail_tab(ui);
                    }
                }
            });

        Self::scene_transforms_window(ui, &scene_transforms, &mut self.scene_transforms_open);

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
        Ok(())
    }

    /// Unified IPC, timing, adapter, scene, draws, and resources (no shader route list).
    fn main_debug_panel(
        ui: &imgui::Ui,
        renderer: Option<&RendererInfoSnapshot>,
        frame: Option<&FrameDiagnosticsSnapshot>,
    ) {
        if renderer.is_none() && frame.is_none() {
            ui.text("Waiting for snapshot…");
            return;
        }

        // Summary: FPS and time between redraws (`wall_frame_time_ms` / `frame_time_ms` — winit tick spacing).
        if let Some(f) = frame {
            let fps = f.fps_from_wall();
            ui.text(format!(
                "FPS {}  |  frame interval {} ms",
                hud_fmt::f64_field(8, 2, fps),
                hud_fmt::f64_field(8, 3, f.wall_frame_time_ms)
            ));
        } else if let Some(r) = renderer {
            let fps = if r.frame_time_ms > f64::EPSILON {
                1000.0 / r.frame_time_ms
            } else {
                0.0
            };
            ui.text(format!(
                "FPS {:.1}  |  frame interval {:.2} ms",
                fps, r.frame_time_ms
            ));
        }

        if let Some(r) = renderer {
            ui.text(format!(
                "Frame index {}  |  viewport {}×{}",
                r.last_frame_index, r.viewport_px.0, r.viewport_px.1
            ));
        } else if frame.is_some() {
            ui.text_disabled("Frame index / viewport: (need renderer snapshot)");
        }

        // Wall time for `execute_frame_graph` (encode + submit). GPU mesh pass ms appears when the
        // device supports timestamp queries and a readback has completed (throttled).
        if let Some(f) = frame {
            ui.text(format!(
                "CPU render graph: {} ms",
                hud_fmt::f64_field(8, 3, f.unified_cpu_frame_ms),
            ));
            if let Some(ms) = f.gpu_mesh_pass_ms {
                ui.text(format!(
                    "GPU mesh pass: {} ms",
                    hud_fmt::f64_field(8, 3, ms)
                ));
            }
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

    /// Per-property implementation table for every known embedded shader stem (**Shader detail** tab).
    ///
    /// Each shader gets a collapsible section; rows are color-coded: green = ok, yellow = partial,
    /// red = not yet.
    fn shader_detail_tab(ui: &imgui::Ui) {
        const STEMS: &[&str] = &[
            "unlit_default",
            "pbsmetallic_default",
            "pbsspecular_default",
            "debug_world_normals_default",
            "ui_unlit_default",
            "ui_textunlit_default",
        ];

        ui.text_disabled("Property implementation status per embedded shader");
        ui.text_disabled("ok (green)  |  partial (yellow)  |  not yet (red)");
        ui.separator();

        for &stem in STEMS {
            let Some((desc, rows)) = shader_props::rows_for_stem(stem) else {
                continue;
            };

            let header_label = format!("{stem}  —  {desc}");
            if ui.collapsing_header(&header_label, TreeNodeFlags::empty()) {
                let table_id = format!("##shprop_{stem}");
                let table_flags =
                    TableFlags::BORDERS | TableFlags::ROW_BG | TableFlags::SIZING_STRETCH_PROP;
                if let Some(_t) =
                    ui.begin_table_with_sizing(&table_id, 3, table_flags, [0.0, 0.0], 0.0)
                {
                    ui.table_setup_column("Property");
                    ui.table_setup_column("Status");
                    ui.table_setup_column("Notes");
                    ui.table_headers_row();

                    for row in &rows {
                        ui.table_next_row();
                        ui.table_next_column();
                        ui.text(row.name);
                        ui.table_next_column();
                        match row.status {
                            shader_props::Status::Ok => {
                                ui.text_colored([0.35, 1.0, 0.35, 1.0], "ok");
                            }
                            shader_props::Status::Partial => {
                                ui.text_colored([1.0, 0.88, 0.25, 1.0], "partial");
                            }
                            shader_props::Status::NotYet => {
                                ui.text_colored([1.0, 0.45, 0.45, 1.0], "not yet");
                            }
                        }
                        ui.table_next_column();
                        ui.text_wrapped(row.note);
                    }
                }
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
            .position([12.0, 12.0], Condition::FirstUseEver)
            .size([440.0, 360.0], Condition::FirstUseEver)
            .bg_alpha(0.88)
            .build(|| {
                ui.text_wrapped(
                    "This file is owned by the renderer. Do not edit config.ini manually while \
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
                if Drag::new("Exposure (reserved)")
                    .range(0.0, 16.0)
                    .speed(0.02)
                    .build(ui, &mut g.rendering.exposure)
                {
                    dirty = true;
                }
                ui.unindent();

                ui.text("Debug");
                ui.indent();
                if ui.checkbox("Log verbose (reserved)", &mut g.debug.log_verbose) {
                    dirty = true;
                }
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
        ui.window("Scene transforms")
            .opened(open)
            .position([12.0, 390.0], Condition::FirstUseEver)
            .size([720.0, 420.0], Condition::FirstUseEver)
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

#[cfg(all(test, feature = "debug-hud"))]
mod hud_fmt_tests {
    #[test]
    fn hud_fmt_produces_stable_field_width() {
        assert_eq!(super::hud_fmt::f64_field(8, 2, 1.0).len(), 8);
        assert_eq!(super::hud_fmt::f64_field(8, 2, 123.456).len(), 8);
    }
}

#[cfg(not(feature = "debug-hud"))]
/// Stub when `debug-hud` is disabled.
#[derive(Debug, Default)]
pub struct DebugHud;

#[cfg(not(feature = "debug-hud"))]
impl DebugHud {
    /// No-op without ImGui.
    pub fn new(
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self
    }

    /// No-op without ImGui.
    pub fn set_snapshot(&mut self, _sample: RendererInfoSnapshot) {}

    /// No-op without ImGui.
    pub fn set_scene_transforms_snapshot(&mut self, _sample: SceneTransformsSnapshot) {}

    /// No-op without ImGui.
    pub fn encode_overlay(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _backbuffer: &wgpu::TextureView,
        _extent: (u32, u32),
        _input: &DebugHudInput,
    ) -> Result<(), String> {
        Ok(())
    }
}
