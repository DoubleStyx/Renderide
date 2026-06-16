//! Shadow atlas Tracy plots.

use super::tracy_plot::tracy_plot;

/// Emits per-frame shadow atlas CPU work counters.
pub fn plot_shadow_atlas(
    layers: usize,
    caster_sets: usize,
    caster_draw_slots: usize,
    visible_groups: usize,
    visible_group_draws: usize,
    upload_bytes: usize,
) {
    tracy_plot!("shadow_atlas::layers", layers as f64);
    tracy_plot!("shadow_atlas::caster_sets", caster_sets as f64);
    tracy_plot!("shadow_atlas::caster_draw_slots", caster_draw_slots as f64);
    tracy_plot!("shadow_atlas::visible_groups", visible_groups as f64);
    tracy_plot!(
        "shadow_atlas::visible_group_draws",
        visible_group_draws as f64
    );
    tracy_plot!("shadow_atlas::upload_bytes", upload_bytes as f64);
}

/// Emits split frame-global command-recording counters for the shadow atlas path.
pub fn plot_frame_global_split(unit_count: usize, command_buffers: usize, chunk_size: usize) {
    tracy_plot!("frame_global_split::units", unit_count as f64);
    tracy_plot!(
        "frame_global_split::command_buffers",
        command_buffers as f64
    );
    tracy_plot!("frame_global_split::chunk_size", chunk_size as f64);
}
