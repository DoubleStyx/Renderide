//! Renderer ordinal assignment for prepared draw rows.

use hashbrown::HashMap;

use crate::render_contract::ParticleDrawKind;
use crate::scene::{RenderSpaceId, SceneMeshRendererRead};

use super::{FramePreparedDraw, FramePreparedRun};

/// Assigns stable scene-table renderer ordinals to every prepared draw row.
pub(super) fn populate_renderer_ordinals_from_scene(
    draws: &mut [FramePreparedDraw],
    scene: &(impl SceneMeshRendererRead + ?Sized),
) {
    let mut renderer_counts = HashMap::new();
    for draw in draws {
        let (static_count, skinned_count) =
            *renderer_counts.entry(draw.space_id).or_insert_with(|| {
                (
                    scene
                        .static_mesh_renderers(draw.space_id)
                        .map_or(0, |renderers| renderers.len()),
                    scene
                        .skinned_mesh_renderers(draw.space_id)
                        .map_or(0, |renderers| renderers.len()),
                )
            });
        draw.renderer_ordinal = if draw.particle_draw.kind != ParticleDrawKind::None {
            // LOD visibility bitsets occupy the static+skinned scene-table range. Generated rows
            // live above it so a particle renderer index cannot alias a grouped mesh renderer.
            static_count
                .saturating_add(skinned_count)
                .saturating_add(draw.renderable_index)
        } else if draw.skinned {
            static_count.saturating_add(draw.renderable_index)
        } else {
            draw.renderable_index
        };
    }
}

/// Assigns dense renderer ordinals per render space when no scene table is available.
pub(super) fn populate_renderer_ordinals_from_runs(
    draws: &mut [FramePreparedDraw],
    runs: &[FramePreparedRun],
) {
    let mut next_by_space: HashMap<RenderSpaceId, usize> = HashMap::new();
    for run in runs {
        let start = run.start as usize;
        let end = run.end as usize;
        let Some(first) = draws.get(start) else {
            continue;
        };
        let ordinal = *next_by_space
            .entry(first.space_id)
            .and_modify(|next| *next += 1)
            .or_insert(0);
        for draw in &mut draws[start..end] {
            draw.renderer_ordinal = ordinal;
        }
    }
}
