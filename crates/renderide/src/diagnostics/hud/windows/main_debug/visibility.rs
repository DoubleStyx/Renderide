//! **Stats / Visibility** section -- per-view world-mesh culling, broadphase, and light-volume diagnostics.

use imgui::TableFlags;

use crate::diagnostics::FrameDiagnosticsSnapshot;
use crate::world_mesh::WorldMeshDrawStats;
use crate::world_mesh::draw_prep::WorldMeshVisibilityStats;

use super::super::sections::collapsible_section;

const TAG_OK: [f32; 4] = [0.40, 1.00, 0.55, 1.00];
const TAG_DIM: [f32; 4] = [0.55, 0.55, 0.55, 1.00];
const TAG_WARN: [f32; 4] = [1.00, 0.90, 0.40, 1.00];

/// Renders world-mesh visibility and culling diagnostics inside the **Stats** tab.
pub(super) fn render_visibility_diagnostics(
    ui: &imgui::Ui,
    snapshot: Option<&FrameDiagnosticsSnapshot>,
) {
    let Some(snapshot) = snapshot else {
        ui.text_disabled("Waiting for visibility diagnostics...");
        return;
    };
    render_world_mesh_culling(ui, snapshot);
    render_per_view_world_mesh(ui, snapshot);
    render_light_visibility(ui, snapshot);
}

fn render_world_mesh_culling(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    let stats = &snapshot.mesh_draw.stats;
    collapsible_section(ui, "World mesh culling", true, |ui| {
        kv_table(ui, "visibility_world_mesh_culling", |ui| {
            cull_summary_rows(ui, stats);
            visibility_index_rows(ui, &stats.visibility_stats);
            prep_rows(ui, stats);
        });
    });
}

fn render_per_view_world_mesh(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    collapsible_section(ui, "Per-view world mesh", true, |ui| {
        if snapshot.mesh_draw.per_view_stats.is_empty() {
            ui.text_disabled("No per-view world-mesh stats captured yet.");
            return;
        }
        if let Some(_table) = ui.begin_table_with_sizing(
            "visibility_per_view_table",
            7,
            table_flags(),
            [0.0, 0.0],
            0.0,
        ) {
            for label in [
                "View",
                "Viewport",
                "Target",
                "Draws",
                "Cull",
                "Broadphase",
                "Batches",
            ] {
                ui.table_setup_column(label);
            }
            ui.table_headers_row();
            for row in &snapshot.mesh_draw.per_view_stats {
                let stats = &row.stats;
                ui.table_next_row();
                cell(ui, format!("{:?}", row.view_id));
                cell(ui, format!("{} x {}", row.viewport_px.0, row.viewport_px.1));
                cell(
                    ui,
                    offscreen_label(row.offscreen_write_target, row.multiview_stereo),
                );
                cell(
                    ui,
                    format!(
                        "{} total / {} overlay",
                        stats.draws_total, stats.draws_overlay
                    ),
                );
                cell(
                    ui,
                    format!(
                        "{} frustum / {} Hi-Z",
                        stats.draws_culled, stats.draws_hi_z_culled
                    ),
                );
                cell(
                    ui,
                    format!(
                        "{} candidates / {} fallback",
                        stats.visibility_stats.candidate_runs, stats.visibility_stats.fallback_runs
                    ),
                );
                cell(
                    ui,
                    format!(
                        "{} batches / {} instances",
                        stats.instance_batch_total, stats.gpu_instances_emitted
                    ),
                );
            }
        }
    });
}

fn render_light_visibility(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    collapsible_section(ui, "Light visibility", true, |ui| {
        let lights = snapshot.lights;
        kv_table(ui, "visibility_lights", |ui| {
            kv(
                ui,
                "Packed lights",
                &format!(
                    "{} default / {} view packs / {} max per view",
                    lights.packed_default_lights,
                    lights.per_view_light_packs,
                    lights.max_per_view_lights
                ),
            );
            kv_colored(
                ui,
                "Signed scene color",
                if lights.signed_scene_color_active {
                    TAG_OK
                } else {
                    TAG_DIM
                },
                if lights.signed_scene_color_active {
                    "active".to_owned()
                } else {
                    "inactive".to_owned()
                },
            );
            kv(
                ui,
                "Influence volumes",
                &format!(
                    "{} indexed / {} fallback / {} rejected",
                    lights.visibility_indexed_lights,
                    lights.visibility_fallback_lights,
                    lights.visibility_rejected_lights
                ),
            );
            kv(
                ui,
                "Light cull traversal",
                &format!(
                    "{} BVH / {} linear",
                    lights.visibility_bvh_queries, lights.visibility_linear_queries
                ),
            );
        });
    });
}

fn cull_summary_rows(ui: &imgui::Ui, stats: &WorldMeshDrawStats) {
    let tested = stats.draws_pre_cull;
    let rejected = stats.draws_culled.saturating_add(stats.draws_hi_z_culled);
    kv_colored(
        ui,
        "Cull rejection",
        ratio_color(rejected, tested),
        format!(
            "{} rejected / {} tested ({})",
            rejected,
            tested,
            percent_text(rejected, tested)
        ),
    );
    kv(
        ui,
        "Cull stages",
        &format!(
            "{} frustum / {} Hi-Z / {} submitted",
            stats.draws_culled, stats.draws_hi_z_culled, stats.draws_total
        ),
    );
}

fn visibility_index_rows(ui: &imgui::Ui, stats: &WorldMeshVisibilityStats) {
    kv(
        ui,
        "Renderer runs",
        &format!(
            "{} indexed / {} fallback / {} candidates",
            stats.indexed_runs, stats.fallback_runs, stats.candidate_runs
        ),
    );
    kv(
        ui,
        "Candidate marks",
        &format!(
            "{} raw / {} duplicate / {} unique",
            stats.raw_candidate_marks, stats.duplicate_candidate_marks, stats.candidate_runs
        ),
    );
    kv_colored(
        ui,
        "Broadphase rejected",
        ratio_color(stats.broadphase_culled_runs, stats.indexed_runs),
        format!(
            "{} runs / {} draws ({})",
            stats.broadphase_culled_runs,
            stats.broadphase_culled_draws,
            percent_text(stats.broadphase_culled_runs, stats.indexed_runs)
        ),
    );
    kv(
        ui,
        "Traversal fallback",
        &format!("{} linear runs", stats.linear_fallback_runs),
    );
}

fn prep_rows(ui: &imgui::Ui, stats: &WorldMeshDrawStats) {
    kv(
        ui,
        "Draw classes",
        &format!(
            "{} rigid / {} skinned / {} overlay",
            stats.rigid_draws, stats.skinned_draws, stats.draws_overlay
        ),
    );
    kv(
        ui,
        "GPU preprocess admission",
        &format!(
            "{} eligible / {} ordered CPU / {} unsupported",
            stats.gpu_preprocess_stats.eligible_draws,
            stats.gpu_preprocess_stats.ordered_cpu_only_draws,
            stats.gpu_preprocess_stats.unsupported_cpu_only_draws
        ),
    );
    kv(
        ui,
        "Instancing blockers",
        &format!(
            "{} candidates / {} skinned / {} strict / {} grab / {} base-instance",
            stats.instancing_blocker_stats.candidate_draws,
            stats.instancing_blocker_stats.skinned_stream_draws,
            stats.instancing_blocker_stats.strict_order_draws,
            stats.instancing_blocker_stats.grab_pass_draws,
            stats
                .instancing_blocker_stats
                .base_instance_unsupported_draws
        ),
    );
}

fn offscreen_label(
    target: crate::frame_contract::OffscreenWriteTarget,
    multiview_stereo: bool,
) -> String {
    if multiview_stereo {
        return "multiview".to_owned();
    }
    match target {
        crate::frame_contract::OffscreenWriteTarget::None => "surface".to_owned(),
        crate::frame_contract::OffscreenWriteTarget::Untracked => "offscreen".to_owned(),
        crate::frame_contract::OffscreenWriteTarget::HostRenderTexture { asset_id, .. } => {
            format!("rt {asset_id}")
        }
    }
}

fn table_flags() -> TableFlags {
    TableFlags::SIZING_STRETCH_PROP | TableFlags::BORDERS_INNER_V | TableFlags::ROW_BG
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

fn cell(ui: &imgui::Ui, value: impl AsRef<str>) {
    ui.table_next_column();
    ui.text(value.as_ref());
}

fn percent_text(n: usize, d: usize) -> String {
    if d == 0 {
        return "n/a".to_owned();
    }
    format!("{:>4.1} %", 100.0 * n as f64 / d as f64)
}

fn ratio_color(n: usize, d: usize) -> [f32; 4] {
    if d == 0 || n == 0 {
        TAG_DIM
    } else if n.saturating_mul(4) >= d {
        TAG_OK
    } else {
        TAG_WARN
    }
}
