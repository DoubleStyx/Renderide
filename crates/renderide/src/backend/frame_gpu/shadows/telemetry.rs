//! Shadow-atlas planning and reuse telemetry.

use crate::world_mesh::WorldMeshPhase;

use super::{
    ShadowCasterSet, ShadowFramePlan, ShadowIndirectCacheResult, ShadowRenderScope,
    ShadowRenderView,
};

pub(super) fn plot_shadow_atlas(
    plan: &ShadowFramePlan,
    indirect: ShadowIndirectCacheResult,
    upload_bytes: usize,
) {
    let (visible_groups, visible_group_draws) = shadow_visible_group_stats(plan);
    crate::profiling::plot_shadow_atlas(
        plan.render_views.len(),
        plan.caster_sets.len(),
        plan.requested_draw_slots,
        visible_groups,
        visible_group_draws,
        upload_bytes,
    );
    plot_shadow_static_split_stats(plan);
    let stats = plan.cache_stats;
    crate::profiling::plot_shadow_cache(crate::profiling::ShadowCacheProfileSample {
        caster_plan_hits: stats.caster_plan_hits,
        caster_plan_misses: stats.caster_plan_misses,
        visibility_hits: stats.visibility_hits,
        visibility_misses: stats.visibility_misses,
        content_hash_hits: stats.content_hash_hits,
        avoided_caster_draw_scans: stats.avoided_caster_draw_scans,
        avoided_visibility_group_tests: stats.avoided_visibility_group_tests,
        avoided_visibility_draw_tests: stats.avoided_visibility_draw_tests,
        avoided_content_hash_draws: stats.avoided_content_hash_draws,
        cascade_hold_hits: stats.cascade_hold_hits,
        cascade_hold_age_misses: stats.cascade_hold_age_misses,
        cascade_hold_coverage_misses: stats.cascade_hold_coverage_misses,
        indirect_hit: indirect.hit,
        avoided_indirect_layers: indirect.avoided_layers,
        avoided_indirect_commands: indirect.avoided_commands,
        avoided_indirect_upload_bytes: indirect.avoided_upload_bytes,
    });
}

/// Counts how each rendering layer resolved its static/dynamic split this frame.
fn plot_shadow_static_split_stats(plan: &ShadowFramePlan) {
    let mut full = 0usize;
    let mut dynamic_over_static = 0usize;
    let mut static_refresh = 0usize;
    let mut skipped_static_draws = 0usize;
    let mut full_reasons = [0usize; 5];
    for &layer_idx in &plan.rendering_layer_indices {
        let Some(view) = plan.render_views.get(layer_idx as usize) else {
            continue;
        };
        let Some(caster_set) = plan.caster_sets.get(view.caster_set_index) else {
            continue;
        };
        match view.render_scope {
            ShadowRenderScope::DynamicOverStatic { refresh_static } => {
                dynamic_over_static = dynamic_over_static.saturating_add(1);
                if refresh_static {
                    static_refresh = static_refresh.saturating_add(1);
                } else {
                    skipped_static_draws =
                        skipped_static_draws.saturating_add(static_visible_draws(view, caster_set));
                }
            }
            ShadowRenderScope::Full => {
                full = full.saturating_add(1);
                if let Some(slot) = full_reasons.get_mut(usize::from(view.full_redraw_reason)) {
                    *slot = slot.saturating_add(1);
                }
            }
            ShadowRenderScope::Reuse => {}
        }
    }
    crate::profiling::plot_shadow_static_split(
        full,
        dynamic_over_static,
        static_refresh,
        skipped_static_draws,
    );
    crate::profiling::plot_shadow_full_redraw_reasons(full_reasons);
}

fn static_visible_draws(view: &ShadowRenderView, caster_set: &ShadowCasterSet) -> usize {
    let mut draws = 0usize;
    for phase in WorldMeshPhase::PRIMARY_FORWARD {
        for group in view.groups(phase) {
            if caster_set.group_is_dynamic(group) {
                break;
            }
            draws = draws
                .saturating_add((group.instance_range.end - group.instance_range.start) as usize);
        }
    }
    draws
}

pub(super) fn shadow_visible_group_stats(plan: &ShadowFramePlan) -> (usize, usize) {
    let mut groups = 0usize;
    let mut draws = 0usize;
    for &layer_idx in &plan.rendering_layer_indices {
        let Some(view) = plan.render_views.get(layer_idx as usize) else {
            continue;
        };
        groups = groups.saturating_add(view.visible_group_count);
        draws = draws.saturating_add(view.visible_group_draw_count);
    }
    (groups, draws)
}
