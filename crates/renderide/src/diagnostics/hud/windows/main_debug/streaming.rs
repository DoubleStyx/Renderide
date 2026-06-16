//! **Stats / Assets** section -- asset streaming, worker pressure, deferred work, and resident pools.

use imgui::TableFlags;

use crate::diagnostics::FrameDiagnosticsSnapshot;

use super::super::sections::collapsible_section;

const TAG_OK: [f32; 4] = [0.40, 1.00, 0.55, 1.00];
const TAG_DIM: [f32; 4] = [0.55, 0.55, 0.55, 1.00];
const TAG_WARN: [f32; 4] = [1.00, 0.90, 0.40, 1.00];

/// Renders asset streaming diagnostics inside the **Stats** tab.
pub(super) fn render_asset_diagnostics(
    ui: &imgui::Ui,
    snapshot: Option<&FrameDiagnosticsSnapshot>,
) {
    let Some(snapshot) = snapshot else {
        ui.text_disabled("Waiting for asset diagnostics...");
        return;
    };
    render_integration_queues(ui, snapshot);
    render_deferred_uploads(ui, snapshot);
    render_asset_workers(ui, snapshot);
    render_materials_and_pools(ui, snapshot);
}

fn render_integration_queues(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    let assets = &snapshot.assets;
    collapsible_section(ui, "Integration queues", true, |ui| {
        kv_table(ui, "assets_integration_queues", |ui| {
            kv_colored(
                ui,
                "Total queued",
                pressure_color(assets.total_queued),
                format!(
                    "{} queued / {} peak",
                    assets.total_queued, assets.peak_queued
                ),
            );
            kv(
                ui,
                "Queue lanes",
                &format!(
                    "{} main / {} high / {} render / {} normal / {} particle",
                    assets.main_queued,
                    assets.high_priority_queued,
                    assets.render_queued,
                    assets.normal_priority_queued,
                    assets.particle_queued
                ),
            );
        });
    });
}

fn render_deferred_uploads(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    let assets = &snapshot.assets;
    collapsible_section(ui, "Deferred uploads", true, |ui| {
        kv_table(ui, "assets_deferred_uploads", |ui| {
            kv_colored(
                ui,
                "Deferred payloads",
                pressure_color(deferred_upload_total(snapshot)),
                format!(
                    "{} mesh / {} tex2D / {} tex3D / {} cube / {} video",
                    assets.pending_mesh_uploads,
                    assets.pending_texture_uploads,
                    assets.pending_texture3d_uploads,
                    assets.pending_cubemap_uploads,
                    assets.pending_video_texture_loads
                ),
            );
            kv(
                ui,
                "Camera readbacks",
                &format!(
                    "{} pending / {} completed / {} failed",
                    snapshot.mesh_draw.pending_camera_readbacks,
                    snapshot.mesh_draw.completed_camera_readbacks,
                    snapshot.mesh_draw.failed_camera_readbacks
                ),
            );
            kv(
                ui,
                "Host render tasks",
                &format!("{}", snapshot.mesh_draw.last_submit_render_task_count),
            );
        });
    });
}

fn render_asset_workers(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    let assets = &snapshot.assets;
    collapsible_section(ui, "Asset workers", true, |ui| {
        kv_table(ui, "assets_workers", |ui| {
            kv_colored(
                ui,
                "Worker pressure",
                pressure_color(assets.worker_queued + assets.worker_running),
                format!(
                    "{} queued / {} running / {} max queued",
                    assets.worker_queued, assets.worker_running, assets.worker_max_queued
                ),
            );
            kv(
                ui,
                "Worker throughput",
                &format!(
                    "{} spawned / {} completed / {} inline / {} saturated",
                    assets.worker_spawned,
                    assets.worker_completed,
                    assets.worker_inline_executed,
                    assets.worker_saturated
                ),
            );
        });
    });
}

fn render_materials_and_pools(ui: &imgui::Ui, snapshot: &FrameDiagnosticsSnapshot) {
    let assets = &snapshot.assets;
    collapsible_section(ui, "Materials & resident pools", true, |ui| {
        kv_table(ui, "assets_materials_pools", |ui| {
            kv_colored(
                ui,
                "Deferred materials",
                pressure_color(assets.pending_material_batches + assets.pending_shader_routes),
                format!(
                    "{} batches / {} shader routes",
                    assets.pending_material_batches, assets.pending_shader_routes
                ),
            );
            kv_colored(
                ui,
                "Material GPU attachment",
                if assets.material_registry_attached && assets.embedded_bind_attached {
                    TAG_OK
                } else {
                    TAG_WARN
                },
                format!(
                    "registry={} embedded_bind={}",
                    assets.material_registry_attached, assets.embedded_bind_attached
                ),
            );
            kv(
                ui,
                "Resident meshes",
                &format!("{}", snapshot.mesh_draw.mesh_pool_entry_count),
            );
            kv(
                ui,
                "Resident textures",
                &format!(
                    "{} GPU / {} CPU registered / {} mip0 ready",
                    snapshot.mesh_draw.textures_gpu_resident,
                    snapshot.mesh_draw.textures_cpu_registered,
                    snapshot.mesh_draw.textures_cpu_mip0_ready
                ),
            );
            kv(
                ui,
                "Resident render textures",
                &format!("{}", snapshot.mesh_draw.render_textures_gpu_resident),
            );
        });
    });
}

fn deferred_upload_total(snapshot: &FrameDiagnosticsSnapshot) -> usize {
    let assets = &snapshot.assets;
    assets
        .pending_mesh_uploads
        .saturating_add(assets.pending_texture_uploads)
        .saturating_add(assets.pending_texture3d_uploads)
        .saturating_add(assets.pending_cubemap_uploads)
        .saturating_add(assets.pending_video_texture_loads)
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

fn pressure_color(count: usize) -> [f32; 4] {
    if count > 0 { TAG_WARN } else { TAG_DIM }
}
