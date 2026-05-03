//! **Stats** tab -- frame index, GPU adapter, host allocator, IPC, draw stats, resources.
//!
//! The body composes a fixed list of [`StatsSection`] impls so each section borrows exactly the
//! sub-fragment(s) it consumes and the tab's render path is one `for section in SECTIONS` loop.

use crate::diagnostics::{FrameDiagnosticsSnapshot, RendererInfoSnapshot};

use super::super::super::fmt as hud_fmt;
use super::super::super::state::HudUiState;
use super::super::super::view::TabView;
use super::super::labels::device_type_label;

/// Borrowed snapshots fed to every [`StatsSection`].
struct StatsContext<'a> {
    renderer: Option<&'a RendererInfoSnapshot>,
    frame: Option<&'a FrameDiagnosticsSnapshot>,
}

/// One section of the **Stats** tab body.
trait StatsSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>);
}

struct FrameLineSection;
struct GpuAdapterSection;
struct HostAndAllocatorSection;
struct IpcAndSceneSection;
struct DrawStatsSection;
struct ResourcesAndGraphSection;

/// Static section list. Each section implements [`StatsSection`] and reads only the fragments it
/// needs; sections gracefully no-op when their inputs are absent.
const SECTIONS: &[&dyn StatsSection] = &[
    &FrameLineSection,
    &GpuAdapterSection,
    &HostAndAllocatorSection,
    &IpcAndSceneSection,
    &DrawStatsSection,
    &ResourcesAndGraphSection,
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
            ui.text("Waiting for snapshot...");
            return;
        }
        let ctx = StatsContext { renderer, frame };
        for section in SECTIONS {
            section.render(ui, &ctx);
        }
    }
}

impl StatsSection for FrameLineSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        if let Some(r) = ctx.renderer {
            ui.text(format!(
                "Frame index {}  |  viewport {}x{}",
                r.last_frame_index, r.viewport_px.0, r.viewport_px.1
            ));
        } else if ctx.frame.is_some() {
            ui.text_disabled("Frame index / viewport: (need renderer snapshot)");
        }
    }
}

impl StatsSection for GpuAdapterSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            return;
        };
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
            "MSAA: requested {}x  |  effective {}x  |  max {}x",
            r.msaa_requested_samples, r.msaa_effective_samples, r.msaa_max_samples
        ));
        ui.text(format!(
            "MSAA (VR stereo): effective {}x  |  max {}x",
            r.msaa_effective_samples_stereo, r.msaa_max_samples_stereo
        ));
        ui.text(format!(
            "Limits: tex2d<={}  max_buf={}  storage_bind={}  |  base_instance={}  multiview={}",
            r.gpu_max_texture_dim_2d,
            r.gpu_max_buffer_size,
            r.gpu_max_storage_binding,
            r.gpu_supports_base_instance,
            r.gpu_supports_multiview
        ));
    }
}

impl StatsSection for HostAndAllocatorSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(f) = ctx.frame else {
            return;
        };
        ui.separator();
        ui.text("Process GPU memory (wgpu allocator)");
        match (
            f.gpu_allocator.totals.allocated_bytes,
            f.gpu_allocator.totals.reserved_bytes,
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
}

impl StatsSection for IpcAndSceneSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(r) = ctx.renderer else {
            return;
        };
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
}

impl StatsSection for DrawStatsSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
        let Some(f) = ctx.frame else {
            return;
        };
        ui.separator();
        ui.text("Batches");
        let m = &f.mesh_draw.stats;
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
            "GPU instance batches (indexed submits): {:>5}  ({:>5} intersect)",
            m.instance_batch_total, m.intersect_pass_batches
        ));
        let compression = if m.instance_batch_total > 0 {
            m.gpu_instances_emitted as f32 / m.instance_batch_total as f32
        } else {
            0.0
        };
        ui.text(format!(
            "GPU instances emitted: {:>5}  |  avg instances/batch: {:>5.2}",
            m.gpu_instances_emitted, compression
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
            f.mesh_draw.last_submit_render_task_count
        ));
        let q = &f.ipc_health.queues;
        ui.text(format!(
            "IPC outbound drops this tick: primary={} background={}  |  consecutive fail streak: primary={} background={}",
            q.ipc_primary_outbound_drop_this_tick,
            q.ipc_background_outbound_drop_this_tick,
            q.ipc_primary_consecutive_fail_streak,
            q.ipc_background_consecutive_fail_streak
        ));
        ui.text(format!(
            "Frame submit apply failures: {}  |  OpenXR wait_frame errs: {}  locate_views errs: {}  |  unhandled IPC cmds (total events): {}",
            f.ipc_health.frame_submit_apply_failures,
            f.xr_health.xr_wait_frame_failures,
            f.xr_health.xr_locate_views_failures,
            f.ipc_health.unhandled_ipc_command_event_total
        ));
    }
}

impl StatsSection for ResourcesAndGraphSection {
    fn render(&self, ui: &imgui::Ui, ctx: &StatsContext<'_>) {
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

        ui.separator();
        ui.text("Resources");
        if let Some(n) = mesh_pool {
            ui.text(format!("Mesh pool: {n}"));
        }
        if let Some(n) = texture_pool {
            ui.text(format!("Textures (pool): {n}"));
        }
        if let Some(n) = render_texture_pool {
            ui.text(format!("Render textures (pool): {n}"));
        }

        if let Some(r) = ctx.renderer {
            ui.separator();
            ui.text("Materials (property store)");
            ui.text(format!(
                "Material property maps: {}  |  property blocks: {}  |  shader bindings: {}",
                r.material_property_slots, r.property_block_slots, r.material_shader_bindings
            ));

            ui.separator();
            ui.text("Frame graph");
            ui.text(format!(
                "Render graph passes: {}  (compile DAG waves: {})  |  GPU lights (packed): {}",
                r.frame_graph_pass_count, r.frame_graph_topo_levels, r.gpu_light_count
            ));
        }
    }
}
