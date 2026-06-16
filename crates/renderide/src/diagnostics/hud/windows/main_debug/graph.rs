//! **Stats / Graph** section -- render-graph scheduling, command recording, transient, and upload diagnostics.

use imgui::TableFlags;

use crate::diagnostics::FrameDiagnosticsSnapshot;
use crate::render_graph::CommandEncodingHudSnapshot;

use super::super::super::fmt as hud_fmt;
use super::super::sections::collapsible_section;

const TAG_OK: [f32; 4] = [0.40, 1.00, 0.55, 1.00];
const TAG_DIM: [f32; 4] = [0.55, 0.55, 0.55, 1.00];
const TAG_WARN: [f32; 4] = [1.00, 0.90, 0.40, 1.00];

/// Renders render graph and command recording diagnostics inside the **Stats** tab.
pub(super) fn render_graph_diagnostics(
    ui: &imgui::Ui,
    snapshot: Option<&FrameDiagnosticsSnapshot>,
) {
    let Some(snapshot) = snapshot else {
        ui.text_disabled("Waiting for graph diagnostics...");
        return;
    };
    let graph = &snapshot.graph;
    render_command_recording(ui, graph);
    render_scheduler(ui, graph);
    render_runtime_commands(ui, graph);
    render_transients_and_uploads(ui, graph);
    render_cpu_timings(ui, graph);
}

fn render_command_recording(ui: &imgui::Ui, graph: &CommandEncodingHudSnapshot) {
    collapsible_section(ui, "Command recording", true, |ui| {
        kv_table(ui, "graph_recording", |ui| {
            kv(
                ui,
                "Path",
                &format!(
                    "{} / {} (requested {}, single {})",
                    graph.recording_path,
                    graph.recording_strategy,
                    graph.requested_recording_mode,
                    graph.single_swapchain_encoder_status
                ),
            );
            kv_colored(
                ui,
                "Across-view admission",
                if graph.per_view_record_admitted {
                    TAG_OK
                } else {
                    TAG_DIM
                },
                format!(
                    "auto={} effective={} work={} draws={}",
                    graph.auto_per_view_record_admitted,
                    graph.per_view_record_admitted,
                    graph.estimated_per_view_record_work,
                    graph.estimated_per_view_draw_count
                ),
            );
            kv_colored(
                ui,
                "In-view admission",
                if graph.in_view_record_admitted {
                    TAG_OK
                } else {
                    TAG_DIM
                },
                format!(
                    "auto={} effective={}",
                    graph.auto_in_view_record_admitted, graph.in_view_record_admitted
                ),
            );
            kv(
                ui,
                "Views / commands",
                &format!(
                    "{} views / {} command buffers ({} per-view) / swapchain={}",
                    graph.view_count,
                    graph.command_buffers,
                    graph.per_view_command_buffers,
                    graph.target_is_swapchain
                ),
            );
        });
    });
}

fn render_scheduler(ui: &imgui::Ui, graph: &CommandEncodingHudSnapshot) {
    collapsible_section(ui, "Scheduler", true, |ui| {
        kv_table(ui, "graph_scheduler", |ui| {
            kv(
                ui,
                "Passes",
                &format!(
                    "{} retained / {} registered / {} culled / {} compile skipped",
                    graph.scheduler_passes,
                    graph.scheduler_registered_passes,
                    graph.scheduler_culled_passes,
                    graph.scheduler_compile_skipped_passes
                ),
            );
            kv(
                ui,
                "Waves",
                &format!(
                    "{} waves / {} largest / {} dependency edges",
                    graph.scheduler_waves,
                    graph.scheduler_largest_wave,
                    graph.scheduler_dependency_edges
                ),
            );
            kv(
                ui,
                "Parallel recording",
                &format!(
                    "{} units / {} batches / {} async-compute capable",
                    graph.scheduler_parallel_recording_units,
                    graph.scheduler_parallel_recording_batches,
                    graph.scheduler_async_compute_capable
                ),
            );
            kv(
                ui,
                "Materialized passes",
                &format!(
                    "{} merge groups / {} materialized groups",
                    graph.scheduler_merge_groups, graph.scheduler_materialized_groups
                ),
            );
            kv(
                ui,
                "Schedule metadata",
                &format!(
                    "{} submit steps / {} upload phases / {} resource events / {} import finals",
                    graph.scheduler_submit_steps,
                    graph.scheduler_upload_phases,
                    graph.scheduler_resource_events,
                    graph.scheduler_import_final_accesses
                ),
            );
            kv_err_or_dim(
                ui,
                "Validation diagnostics",
                graph.validation_diagnostics,
                format!(
                    "{} diagnostics / {} parameter schemas",
                    graph.validation_diagnostics, graph.pass_parameter_schemas
                ),
            );
        });
    });
}

fn render_runtime_commands(ui: &imgui::Ui, graph: &CommandEncodingHudSnapshot) {
    collapsible_section(ui, "Runtime commands", true, |ui| {
        let commands = graph.command_stats;
        kv_table(ui, "graph_commands", |ui| {
            kv(
                ui,
                "World mesh",
                &format!(
                    "{} draws / {} instance batches / {} pipeline submits",
                    commands.draw_items, commands.instance_batches, commands.pipeline_pass_submits
                ),
            );
            kv(
                ui,
                "Recorded passes",
                &format!(
                    "{} raster / {} compute / {} encoder / {} render-pass opens",
                    commands.recorded_raster_passes,
                    commands.recorded_compute_passes,
                    commands.recorded_encoder_passes,
                    commands.opened_render_passes
                ),
            );
            kv_err_or_dim(
                ui,
                "Runtime skipped passes",
                commands.skipped_passes,
                format!("{}", commands.skipped_passes),
            );
            kv(
                ui,
                "Copies / resolves",
                &format!(
                    "{} copies ({} skipped) / {} resolves ({} skipped)",
                    commands.copy_count,
                    commands.skipped_copy_count,
                    commands.resolve_count,
                    commands.skipped_resolve_count
                ),
            );
            kv(
                ui,
                "Runtime bandwidth",
                &hud_fmt::bytes_compact(
                    graph
                        .scheduler_estimated_bandwidth_bytes
                        .saturating_add(commands.estimated_bandwidth_bytes),
                ),
            );
        });
    });
}

fn render_transients_and_uploads(ui: &imgui::Ui, graph: &CommandEncodingHudSnapshot) {
    collapsible_section(ui, "Transients & uploads", true, |ui| {
        kv_table(ui, "graph_transients_uploads", |ui| {
            kv(
                ui,
                "Transient resources",
                &format!(
                    "{} textures / {} slots / {} texture lanes / {} buffer lanes",
                    graph.transient_texture_count,
                    graph.transient_texture_slots,
                    graph.transient_texture_lanes,
                    graph.transient_buffer_lanes
                ),
            );
            kv_colored(
                ui,
                "Transient misses",
                miss_color(graph.transient_texture_misses + graph.transient_buffer_misses),
                format!(
                    "{} texture / {} buffer / views {} hit {} miss",
                    graph.transient_texture_misses,
                    graph.transient_buffer_misses,
                    graph.transient_texture_view_hits,
                    graph.transient_texture_view_misses
                ),
            );
            kv(
                ui,
                "Upload batch",
                &format!(
                    "{} writes / {} / {} staged / {} fallback",
                    graph.upload_stats.writes,
                    hud_fmt::bytes_compact(graph.upload_stats.bytes as u64),
                    graph.upload_stats.staged_writes,
                    graph.upload_stats.fallback_writes
                ),
            );
            kv(
                ui,
                "Upload arena",
                &format!(
                    "{} persistent / {} temporary / {} reuse / {} grow",
                    hud_fmt::bytes_compact(graph.upload_stats.persistent_staging_bytes),
                    hud_fmt::bytes_compact(graph.upload_stats.temporary_staging_bytes),
                    graph.upload_stats.persistent_slot_reuses,
                    graph.upload_stats.persistent_slot_grows
                ),
            );
        });
    });
}

fn render_cpu_timings(ui: &imgui::Ui, graph: &CommandEncodingHudSnapshot) {
    collapsible_section(ui, "CPU timings", false, |ui| {
        kv_table(ui, "graph_timings", |ui| {
            kv(
                ui,
                "Prep",
                &format!(
                    "{:.3} ms pre-resolve / {:.3} ms resources",
                    graph.pre_resolve_ms, graph.prepare_resources_ms
                ),
            );
            kv(
                ui,
                "Encode",
                &format!(
                    "{:.3} ms frame-global / {:.3} ms per-view / {:.3} ms single-swapchain",
                    graph.frame_global_encode_ms,
                    graph.per_view_encode_ms,
                    graph.single_swapchain_encode_ms
                ),
            );
            kv(
                ui,
                "Finish",
                &format!(
                    "{:.3} ms frame-global / {:.3} ms per-view max / {:.3} ms upload / {:.3} ms single",
                    graph.frame_global_finish_ms,
                    graph.per_view_max_finish_ms,
                    graph.upload_finish_ms,
                    graph.single_swapchain_finish_ms
                ),
            );
            kv(
                ui,
                "Submit",
                &format!(
                    "{:.3} ms upload drain / {:.3} ms assemble / {:.3} ms enqueue",
                    graph.upload_drain_ms, graph.command_batch_assembly_ms, graph.submit_enqueue_ms
                ),
            );
        });
    });
}

fn kv_table(ui: &imgui::Ui, id: &str, body: impl FnOnce(&imgui::Ui)) {
    if let Some(_table) = ui.begin_table_with_sizing(
        id,
        2,
        TableFlags::SIZING_STRETCH_PROP | TableFlags::PAD_OUTER_X,
        [0.0, 0.0],
        0.0,
    ) {
        ui.table_setup_column_with(imgui::TableColumnSetup {
            name: "key",
            flags: imgui::TableColumnFlags::WIDTH_FIXED,
            init_width_or_weight: 200.0,
            user_id: imgui::Id::default(),
        });
        ui.table_setup_column("value");
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

fn kv_colored(ui: &imgui::Ui, key: &str, color: [f32; 4], value: String) {
    ui.table_next_row();
    ui.table_next_column();
    ui.text_disabled(key);
    ui.table_next_column();
    ui.text_colored(color, value);
}

fn kv_err_or_dim(ui: &imgui::Ui, key: &str, counter: usize, value: String) {
    kv_colored(ui, key, if counter > 0 { TAG_WARN } else { TAG_DIM }, value);
}

fn miss_color(count: usize) -> [f32; 4] {
    if count > 0 { TAG_WARN } else { TAG_DIM }
}
