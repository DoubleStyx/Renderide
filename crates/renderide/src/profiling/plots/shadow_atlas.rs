//! Shadow atlas Tracy plots.

use super::tracy_plot::tracy_plot;

/// CPU planning and indirect-upload work avoided by exact shadow cache hits.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowCacheProfileSample {
    pub caster_plan_hits: usize,
    pub caster_plan_misses: usize,
    pub visibility_hits: usize,
    pub visibility_misses: usize,
    pub content_hash_hits: usize,
    pub avoided_caster_draw_scans: usize,
    pub avoided_visibility_group_tests: usize,
    pub avoided_visibility_draw_tests: usize,
    pub avoided_content_hash_draws: usize,
    pub indirect_hit: bool,
    pub avoided_indirect_layers: usize,
    pub avoided_indirect_commands: usize,
    pub avoided_indirect_upload_bytes: usize,
}

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

/// Emits how the static/dynamic split resolved this frame.
///
/// `dynamic_over_static` is the count that matters: those layers restored cached static depth and
/// redrew only their dynamic casters instead of every visible caster. `skipped_static_draws` is the
/// draw submissions that avoided the encoder because of it.
pub fn plot_shadow_static_split(
    full_layers: usize,
    dynamic_over_static_layers: usize,
    static_refresh_layers: usize,
    skipped_static_draws: usize,
) {
    tracy_plot!("shadow_atlas::layers_full_redraw", full_layers as f64);
    tracy_plot!(
        "shadow_atlas::layers_dynamic_over_static",
        dynamic_over_static_layers as f64
    );
    tracy_plot!(
        "shadow_atlas::layers_static_refresh",
        static_refresh_layers as f64
    );
    tracy_plot!(
        "shadow_atlas::skipped_static_draws",
        skipped_static_draws as f64
    );
}

/// Emits exact retained shadow-planning cache counters.
pub fn plot_shadow_cache(sample: ShadowCacheProfileSample) {
    tracy_plot!(
        "shadow_atlas::caster_plan_cache_hits",
        sample.caster_plan_hits as f64
    );
    tracy_plot!(
        "shadow_atlas::caster_plan_cache_misses",
        sample.caster_plan_misses as f64
    );
    tracy_plot!(
        "shadow_atlas::visibility_cache_hits",
        sample.visibility_hits as f64
    );
    tracy_plot!(
        "shadow_atlas::visibility_cache_misses",
        sample.visibility_misses as f64
    );
    tracy_plot!(
        "shadow_atlas::content_hash_cache_hits",
        sample.content_hash_hits as f64
    );
    tracy_plot!(
        "shadow_atlas::avoided_caster_draw_scans",
        sample.avoided_caster_draw_scans as f64
    );
    tracy_plot!(
        "shadow_atlas::avoided_visibility_group_tests",
        sample.avoided_visibility_group_tests as f64
    );
    tracy_plot!(
        "shadow_atlas::avoided_visibility_draw_tests",
        sample.avoided_visibility_draw_tests as f64
    );
    tracy_plot!(
        "shadow_atlas::avoided_content_hash_draws",
        sample.avoided_content_hash_draws as f64
    );
    tracy_plot!(
        "shadow_atlas::indirect_plan_cache_hit",
        f64::from(u8::from(sample.indirect_hit))
    );
    tracy_plot!(
        "shadow_atlas::avoided_indirect_layers",
        sample.avoided_indirect_layers as f64
    );
    tracy_plot!(
        "shadow_atlas::avoided_indirect_commands",
        sample.avoided_indirect_commands as f64
    );
    tracy_plot!(
        "shadow_atlas::avoided_indirect_upload_bytes",
        sample.avoided_indirect_upload_bytes as f64
    );
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
