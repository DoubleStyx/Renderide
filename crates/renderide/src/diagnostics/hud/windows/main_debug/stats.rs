//! **Stats** tab -- frame index, GPU adapter, host allocator, IPC, draw stats, resources.
//!
//! Density redesign:
//! * Each section is a `CollapsingHeader` (auto-styled, click to fold).
//! * Inside each header, key/value pairs go through a borderless 2-column table so labels stack
//!   in a left rail and values align in a right rail -- much easier to scan than freeform text.
//! * Color tags surface the bits worth caring about at a glance: overlay batch / draw counts,
//!   error counters, percentages near critical thresholds, "active/inactive" booleans.

use imgui::{TableFlags, TreeNodeFlags};

use crate::diagnostics::{FrameDiagnosticsSnapshot, RendererInfoSnapshot};

use super::super::super::fmt as hud_fmt;
use super::super::super::state::HudUiState;
use super::super::super::view::TabView;
use super::super::labels::device_type_label;

/// Bright cyan: stable headline values (frame index, viewport).
const TAG_HEADLINE: [f32; 4] = [0.55, 0.85, 1.00, 1.00];
/// Bright green: "active / healthy / connected" state, non-zero overlay counts.
const TAG_OK: [f32; 4] = [0.40, 1.00, 0.55, 1.00];
/// Dim gray: zero / inactive / not-applicable values.
const TAG_DIM: [f32; 4] = [0.55, 0.55, 0.55, 1.00];
/// Yellow: warning-zone values (high RAM use, recent drops, etc).
const TAG_WARN: [f32; 4] = [1.00, 0.90, 0.40, 1.00];
/// Red: error / failure counters when non-zero.
const TAG_ERR: [f32; 4] = [1.00, 0.45, 0.45, 1.00];

/// Borrowed snapshots fed to every [`StatsSection`].
struct StatsContext<'a> {
    renderer: Option<&'a RendererInfoSnapshot>,
    frame: Option<&'a FrameDiagnosticsSnapshot>,
}

/// One section of the **Stats** tab body. Each is rendered as a `CollapsingHeader`.
trait StatsSection {
    /// Header label (also the tree-node id).
    fn label(&self) -> &str;
    /// Default-open?
    fn default_open(&self) -> bool {
        true
    }
    /// Render the section's body once the header is expanded.
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>);
}

struct FrameLineSection;
struct GpuAdapterSection;
struct ProcessMemorySection;
struct HostCpuRamSection;
struct IpcSceneSection;
struct DrawStatsSection;
struct HealthSection;
struct ResourcesSection;
struct MaterialsSection;
struct FrameGraphSection;

const SECTIONS: &[&dyn StatsSection] = &[
    &FrameLineSection,
    &GpuAdapterSection,
    &ProcessMemorySection,
    &HostCpuRamSection,
    &IpcSceneSection,
    &DrawStatsSection,
    &HealthSection,
    &ResourcesSection,
    &MaterialsSection,
    &FrameGraphSection,
];

/// **Stats** tab dispatched from [`super::MainDebugWindow`].
pub struct StatsTab;

impl TabView for StatsTab {
    type Data<'a> = (
        Option<&'a RendererInfoSnapshot>,
        Option<&'a FrameDiagnosticsSnapshot>,
    );
    type State = HudUiState;

    fn render(&self, ui: &imgui::Ui, data: Self::Data<'_>, _state: &mut Self::State) {
        let (renderer, frame) = data;
        if renderer.is_none() && frame.is_none() {
            ui.text_disabled("Waiting for snapshot...");
            return;
        }
        let ctx = StatsContext { renderer, frame };
        for section in SECTIONS {
            let flags = if section.default_open() {
                TreeNodeFlags::DEFAULT_OPEN
            } else {
                TreeNodeFlags::empty()
            };
            if ui.collapsing_header(section.label(), flags) {
                ui.indent_by(8.0);
                section.body(ui, &ctx);
                ui.unindent_by(8.0);
                ui.spacing();
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Section bodies.
// -------------------------------------------------------------------------------------------------

impl StatsSection for FrameLineSection {
    fn label(&self) -> &str {
        "Frame"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            ui.text_disabled("(need renderer snapshot)");
            return;
        };
        kv_table(ui, "frame_kv", |ui| {
            kv_colored(
                ui,
                "Frame index",
                TAG_HEADLINE,
                format!("{}", r.last_frame_index),
            );
            kv_colored(
                ui,
                "Viewport",
                TAG_HEADLINE,
                format!("{} x {}", r.viewport_px.0, r.viewport_px.1),
            );
        });
    }
}

impl StatsSection for GpuAdapterSection {
    fn label(&self) -> &str {
        "GPU adapter"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            return;
        };
        kv_table(ui, "gpu_kv", |ui| {
            kv(ui, "Name", &r.adapter_name);
            kv(
                ui,
                "Class",
                &format!(
                    "{}  ({:?})",
                    device_type_label(r.adapter_device_type),
                    r.adapter_backend
                ),
            );
            kv(
                ui,
                "Driver",
                &format!("{} ({})", r.adapter_driver, r.adapter_driver_info),
            );
            kv(
                ui,
                "Surface",
                &format!("{:?}  |  present {:?}", r.surface_format, r.present_mode),
            );
            kv(
                ui,
                "MSAA",
                &format!(
                    "req {}x  eff {}x  max {}x",
                    r.msaa_requested_samples, r.msaa_effective_samples, r.msaa_max_samples
                ),
            );
            kv(
                ui,
                "MSAA (stereo)",
                &format!(
                    "eff {}x  max {}x",
                    r.msaa_effective_samples_stereo, r.msaa_max_samples_stereo
                ),
            );
            kv(
                ui,
                "Limits",
                &format!(
                    "tex2d<={}  max_buf={}  storage_bind={}",
                    r.gpu_max_texture_dim_2d, r.gpu_max_buffer_size, r.gpu_max_storage_binding
                ),
            );
            kv(
                ui,
                "Features",
                &format!(
                    "base_instance={}  multiview={}  f32_filter={}",
                    r.gpu_supports_base_instance,
                    r.gpu_supports_multiview,
                    r.gpu_supports_float32_filterable
                ),
            );
            kv(
                ui,
                "Texture compression",
                &format!("{:?}", r.gpu_texture_compression_features),
            );
        });
    }
}

impl StatsSection for ProcessMemorySection {
    fn label(&self) -> &str {
        "Process GPU memory"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(f) = ctx.frame else {
            return;
        };
        kv_table(ui, "mem_kv", |ui| {
            match (
                f.gpu_allocator.totals.allocated_bytes,
                f.gpu_allocator.totals.reserved_bytes,
            ) {
                (Some(alloc), Some(resv)) => {
                    kv(
                        ui,
                        "Allocated",
                        &format!(
                            "{} GiB / {} GiB reserved",
                            hud_fmt::gib_value(7, 2, alloc).trim(),
                            hud_fmt::gib_value(7, 2, resv).trim()
                        ),
                    );
                }
                _ => {
                    kv_dim(ui, "Allocated", "not reported for this backend");
                }
            }
        });
    }
}

impl StatsSection for HostCpuRamSection {
    fn label(&self) -> &str {
        "Host CPU / RAM"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(f) = ctx.frame else {
            return;
        };
        kv_table(ui, "host_kv", |ui| {
            if f.host.cpu_model.is_empty() {
                kv_dim(ui, "CPU model", "(unknown)");
            } else {
                kv(ui, "CPU model", &f.host.cpu_model);
            }
            let cpu_pct = f64::from(f.host.cpu_usage_percent);
            kv_colored(
                ui,
                "CPU usage",
                pct_color(cpu_pct),
                format!("{:>5.1} %  ({} logical)", cpu_pct, f.host.logical_cpus),
            );
            let ram_pct = if f.host.ram_total_bytes > 0 {
                100.0 * f.host.ram_used_bytes as f64 / f.host.ram_total_bytes as f64
            } else {
                0.0
            };
            kv_colored(
                ui,
                "RAM",
                pct_color(ram_pct),
                format!(
                    "{} / {} GiB  ({:>4.1} %)",
                    hud_fmt::gib_value(7, 2, f.host.ram_used_bytes).trim(),
                    hud_fmt::gib_value(7, 2, f.host.ram_total_bytes).trim(),
                    ram_pct
                ),
            );
        });
    }
}

impl StatsSection for IpcSceneSection {
    fn label(&self) -> &str {
        "IPC / scene"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            return;
        };
        kv_table(ui, "ipc_kv", |ui| {
            kv_colored(
                ui,
                "Connected",
                if r.ipc_connected { TAG_OK } else { TAG_ERR },
                format!("{}", r.ipc_connected),
            );
            kv(ui, "Init", &format!("{:?}", r.init_state));
            kv(ui, "Render spaces", &format!("{}", r.render_space_count));
            kv(
                ui,
                "Mesh renderables (CPU)",
                &format!("{}", r.mesh_renderable_count),
            );
        });
    }
}

impl StatsSection for DrawStatsSection {
    fn label(&self) -> &str {
        "Draws & batches"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(f) = ctx.frame else {
            return;
        };
        let m = &f.mesh_draw.stats;
        kv_table(ui, "draws_kv", |ui| {
            kv_split3(
                ui,
                "Batches",
                [m.batch_total, m.batch_main, m.batch_overlay],
                ["total", "main", "overlay"],
            );
            kv_split3(
                ui,
                "Draws",
                [m.draws_total, m.draws_main, m.draws_overlay],
                ["total", "main", "overlay"],
            );
            kv(
                ui,
                "GPU instance batches",
                &format!(
                    "{} indexed  ({} intersect / {} grab)",
                    m.instance_batch_total, m.intersect_pass_batches, m.transparent_pass_batches
                ),
            );
            kv(
                ui,
                "Transparent classes",
                &format!(
                    "ordered={}  zwrite={}  grab={}  comm={}  two-sided={}  fallback={}",
                    m.transparent_class_stats.ordered_alpha_draws,
                    m.transparent_class_stats.depth_writing_draws,
                    m.transparent_class_stats.grab_pass_filter_draws,
                    m.transparent_class_stats.commutative_blend_draws,
                    m.transparent_class_stats.known_two_sided_draws,
                    m.transparent_class_stats.compatibility_fallback_draws
                ),
            );
            kv(
                ui,
                "Depth prepass",
                &format!(
                    "{} batches  /  {} instances",
                    m.depth_prepass_batches, m.depth_prepass_instances
                ),
            );
            let compression = if m.instance_batch_total > 0 {
                m.gpu_instances_emitted as f32 / m.instance_batch_total as f32
            } else {
                0.0
            };
            kv(
                ui,
                "GPU instances emitted",
                &format!(
                    "{}  (avg {:.2} / batch)",
                    m.gpu_instances_emitted, compression
                ),
            );
            kv(
                ui,
                "Pipeline pass submits",
                &format!("{}", m.submitted_pipeline_pass_total),
            );
            kv(
                ui,
                "Frustum cull",
                &format!(
                    "{} considered  /  {} culled  /  Hi-Z {} culled  /  {} submitted",
                    m.draws_pre_cull, m.draws_culled, m.draws_hi_z_culled, m.draws_total
                ),
            );
            kv(
                ui,
                "Prep",
                &format!("rigid {}  skinned {}", m.rigid_draws, m.skinned_draws),
            );
        });
    }
}

impl StatsSection for HealthSection {
    fn label(&self) -> &str {
        "Health / errors"
    }
    fn default_open(&self) -> bool {
        false
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(f) = ctx.frame else {
            return;
        };
        let q = &f.ipc_health.queues;
        kv_table(ui, "health_kv", |ui| {
            kv_err_or_dim(
                ui,
                "IPC drops this tick",
                format!(
                    "primary={}  background={}",
                    q.ipc_primary_outbound_drop_this_tick, q.ipc_background_outbound_drop_this_tick
                ),
                u64::from(q.ipc_primary_outbound_drop_this_tick)
                    + u64::from(q.ipc_background_outbound_drop_this_tick),
            );
            kv_err_or_dim(
                ui,
                "IPC fail streak",
                format!(
                    "primary={}  background={}",
                    q.ipc_primary_consecutive_fail_streak, q.ipc_background_consecutive_fail_streak,
                ),
                u64::from(q.ipc_primary_consecutive_fail_streak)
                    + u64::from(q.ipc_background_consecutive_fail_streak),
            );
            kv_err_or_dim(
                ui,
                "Frame submit apply failures",
                format!("{}", f.ipc_health.frame_submit_apply_failures),
                f.ipc_health.frame_submit_apply_failures,
            );
            kv_err_or_dim(
                ui,
                "OpenXR wait_frame errs",
                format!("{}", f.xr_health.xr_wait_frame_failures),
                f.xr_health.xr_wait_frame_failures,
            );
            kv_err_or_dim(
                ui,
                "OpenXR locate_views errs",
                format!("{}", f.xr_health.xr_locate_views_failures),
                f.xr_health.xr_locate_views_failures,
            );
            kv_err_or_dim(
                ui,
                "Unhandled IPC cmds",
                format!("{}", f.ipc_health.unhandled_ipc_command_event_total),
                f.ipc_health.unhandled_ipc_command_event_total,
            );
            kv(
                ui,
                "Last submit render_tasks",
                &format!("{}", f.mesh_draw.last_submit_render_task_count),
            );
            kv(
                ui,
                "Camera readbacks",
                &format!(
                    "pending={}  completed={}  failed={}",
                    f.mesh_draw.pending_camera_readbacks,
                    f.mesh_draw.completed_camera_readbacks,
                    f.mesh_draw.failed_camera_readbacks
                ),
            );
        });
    }
}

impl StatsSection for ResourcesSection {
    fn label(&self) -> &str {
        "Resources"
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        if ctx.renderer.is_none() && ctx.frame.is_none() {
            return;
        }
        let mesh_pool = ctx
            .frame
            .map(|f| f.mesh_draw.mesh_pool_entry_count)
            .or_else(|| ctx.renderer.map(|r| r.resident_mesh_count));
        let texture_pool = ctx
            .renderer
            .map(|r| r.resident_texture_count)
            .or_else(|| ctx.frame.map(|f| f.mesh_draw.textures_gpu_resident));
        let render_texture_pool = ctx
            .frame
            .map(|f| f.mesh_draw.render_textures_gpu_resident)
            .or_else(|| ctx.renderer.map(|r| r.resident_render_texture_count));

        kv_table(ui, "res_kv", |ui| {
            if let Some(n) = mesh_pool {
                kv(ui, "Mesh pool", &format!("{n}"));
            }
            if let Some(n) = texture_pool {
                if let Some(f) = ctx.frame {
                    kv(
                        ui,
                        "Textures",
                        &format!(
                            "{n} GPU resident  /  {} CPU registered  /  {} mip0 ready",
                            f.mesh_draw.textures_cpu_registered,
                            f.mesh_draw.textures_cpu_mip0_ready
                        ),
                    );
                } else {
                    kv(ui, "Textures (pool)", &format!("{n}"));
                }
            }
            if let Some(n) = render_texture_pool {
                kv(ui, "Render textures (pool)", &format!("{n}"));
            }
        });
    }
}

impl StatsSection for MaterialsSection {
    fn label(&self) -> &str {
        "Materials"
    }
    fn default_open(&self) -> bool {
        false
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            return;
        };
        kv_table(ui, "materials_kv", |ui| {
            kv(
                ui,
                "Property maps",
                &format!("{}", r.material_property_slots),
            );
            kv(
                ui,
                "Property blocks",
                &format!("{}", r.property_block_slots),
            );
            kv(
                ui,
                "Shader bindings",
                &format!("{}", r.material_shader_bindings),
            );
            kv(
                ui,
                "Shader graph",
                &format!(
                    "{} routes / {} sources / {} globals",
                    r.material_shader_graph.shader_nodes,
                    r.material_shader_graph.embedded_source_nodes,
                    r.material_shader_graph.global_uniforms
                ),
            );
            kv(
                ui,
                "Pipeline cache",
                &format!(
                    "{} ready / {} pending / {} failed",
                    r.material_pipeline_cache.ready_entries,
                    r.material_pipeline_cache.pending_entries,
                    r.material_pipeline_cache.failed_entries
                ),
            );
            kv(
                ui,
                "Pipeline cache stats",
                &format!(
                    "{} hits / {} misses / {} evictions",
                    r.material_pipeline_cache.hits,
                    r.material_pipeline_cache.misses,
                    r.material_pipeline_cache.evictions
                ),
            );
            kv(
                ui,
                "WGSL hot reload",
                if r.material_shader_graph.dev_hot_reload_enabled {
                    "enabled"
                } else {
                    "disabled"
                },
            );
            if let Some(stem) = r.material_shader_graph.last_dev_reload_stem.as_deref() {
                kv(ui, "Last WGSL reload", stem);
            }
            if let Some(error) = r.material_shader_graph.last_dev_reload_error.as_deref() {
                kv(ui, "Last WGSL reload error", error);
            }
        });
    }
}

impl StatsSection for FrameGraphSection {
    fn label(&self) -> &str {
        "Frame graph"
    }
    fn default_open(&self) -> bool {
        false
    }
    fn body(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            return;
        };
        kv_table(ui, "graph_kv", |ui| {
            kv(
                ui,
                "Render graph passes",
                &format!(
                    "{}  (compile DAG waves {})",
                    r.frame_graph_pass_count, r.frame_graph_topo_levels
                ),
            );
            kv(ui, "GPU lights (packed)", &format!("{}", r.gpu_light_count));
            let (label, color) = if r.signed_scene_color_active {
                ("active", TAG_OK)
            } else {
                ("inactive", TAG_DIM)
            };
            kv_colored(ui, "Signed scene color", color, label.into());
        });
    }
}

// -------------------------------------------------------------------------------------------------
// Section helpers: 2-column key/value table, colored variants, threshold colors.
// -------------------------------------------------------------------------------------------------

fn kv_table(ui: &imgui::Ui, id: &str, body: impl FnOnce(&imgui::Ui)) {
    let flags = TableFlags::SIZING_STRETCH_PROP | TableFlags::PAD_OUTER_X;
    if let Some(_t) = ui.begin_table_with_sizing(id, 2, flags, [0.0, 0.0], 0.0) {
        ui.table_setup_column_with(imgui::TableColumnSetup {
            name: "key",
            flags: imgui::TableColumnFlags::WIDTH_FIXED,
            init_width_or_weight: 200.0,
            user_id: imgui::Id::default(),
        });
        ui.table_setup_column_with(imgui::TableColumnSetup {
            name: "val",
            flags: imgui::TableColumnFlags::WIDTH_STRETCH,
            init_width_or_weight: 1.0,
            user_id: imgui::Id::default(),
        });
        body(ui);
    }
}

fn kv(ui: &imgui::Ui, key: &str, value: &str) {
    ui.table_next_row();
    ui.table_next_column();
    ui.text_disabled(key);
    ui.table_next_column();
    ui.text(value);
}

fn kv_dim(ui: &imgui::Ui, key: &str, value: &str) {
    ui.table_next_row();
    ui.table_next_column();
    ui.text_disabled(key);
    ui.table_next_column();
    ui.text_disabled(value);
}

fn kv_colored(ui: &imgui::Ui, key: &str, color: [f32; 4], value: String) {
    ui.table_next_row();
    ui.table_next_column();
    ui.text_disabled(key);
    ui.table_next_column();
    ui.text_colored(color, value);
}

/// 3-segment value: `<a> <a_label>  |  <b> <b_label>  |  <c> <c_label>`, with the third segment
/// (typically "overlay") highlighted when non-zero so the user can spot overlay-layer presence at
/// a glance.
fn kv_split3(ui: &imgui::Ui, key: &str, values: [usize; 3], labels: [&str; 3]) {
    let [a, b, c] = values;
    let [a_label, b_label, c_label] = labels;
    ui.table_next_row();
    ui.table_next_column();
    ui.text_disabled(key);
    ui.table_next_column();
    ui.text(format!("{a:>5} {a_label}  |  {b:>5} {b_label}  |  "));
    ui.same_line();
    if c > 0 {
        ui.text_colored(TAG_OK, format!("{c:>5} {c_label}"));
    } else {
        ui.text_disabled(format!("{c:>5} {c_label}"));
    }
}

fn kv_err_or_dim(ui: &imgui::Ui, key: &str, value: String, counter: u64) {
    let color = if counter > 0 { TAG_ERR } else { TAG_DIM };
    kv_colored(ui, key, color, value);
}

fn pct_color(pct: f64) -> [f32; 4] {
    if pct >= 90.0 {
        TAG_ERR
    } else if pct >= 70.0 {
        TAG_WARN
    } else if pct >= 1.0 {
        TAG_OK
    } else {
        TAG_DIM
    }
}
