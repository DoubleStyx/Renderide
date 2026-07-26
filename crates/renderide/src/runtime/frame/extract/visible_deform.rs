//! Visible mesh-deform key extraction from sorted world-mesh draw plans.

use rayon::prelude::*;

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy};
use crate::mesh_deform::SkinCacheKey;
use crate::world_mesh::WorldMeshDrawPlan;

use super::ViewWorldMeshDrawPlans;

/// View draw plans assigned to one visible-deform-key scan worker.
const VISIBLE_DEFORM_KEYS_PARALLEL_CHUNK_VIEWS: usize = 1;

/// Collects skin-cache keys for visible deformed world and overlay draw items.
pub(super) fn visible_mesh_deform_keys_from_draw_plans(
    draw_plans: &[ViewWorldMeshDrawPlans],
) -> hashbrown::HashSet<SkinCacheKey> {
    if should_parallelize_visible_deform_keys(draw_plans) {
        return visible_mesh_deform_keys_from_draw_plans_parallel(draw_plans);
    }
    visible_mesh_deform_keys_from_draw_plans_serial(draw_plans)
}

fn should_parallelize_visible_deform_keys(draw_plans: &[ViewWorldMeshDrawPlans]) -> bool {
    let total_draws = draw_plans
        .iter()
        .map(ViewWorldMeshDrawPlans::draw_count)
        .sum::<usize>();
    FrameParallelPolicy::for_current_thread_pool()
        .admit_draw_heavy_views(
            FrameCpuWorkload::view_draws(draw_plans.len(), total_draws),
            VISIBLE_DEFORM_KEYS_PARALLEL_CHUNK_VIEWS,
        )
        .is_parallel()
}

fn visible_mesh_deform_keys_from_draw_plans_serial(
    draw_plans: &[ViewWorldMeshDrawPlans],
) -> hashbrown::HashSet<SkinCacheKey> {
    let mut keys = hashbrown::HashSet::new();
    for plan in draw_plans {
        extend_visible_mesh_deform_keys_for_plan(&mut keys, &plan.world);
        if let Some(overlay) = &plan.desktop_overlay {
            extend_visible_mesh_deform_keys_for_plan(&mut keys, overlay);
        }
    }
    keys
}

fn visible_mesh_deform_keys_from_draw_plans_parallel(
    draw_plans: &[ViewWorldMeshDrawPlans],
) -> hashbrown::HashSet<SkinCacheKey> {
    draw_plans
        .par_iter()
        .with_min_len(VISIBLE_DEFORM_KEYS_PARALLEL_CHUNK_VIEWS)
        .map(|plan| {
            let mut keys = hashbrown::HashSet::new();
            extend_visible_mesh_deform_keys_for_plan(&mut keys, &plan.world);
            if let Some(overlay) = &plan.desktop_overlay {
                extend_visible_mesh_deform_keys_for_plan(&mut keys, overlay);
            }
            keys
        })
        .reduce(hashbrown::HashSet::new, |mut keys, partial| {
            keys.extend(partial);
            keys
        })
}

fn extend_visible_mesh_deform_keys_for_plan(
    keys: &mut hashbrown::HashSet<SkinCacheKey>,
    draw_plan: &WorldMeshDrawPlan,
) {
    let Some(collection) = draw_plan.as_prefetched() else {
        return;
    };
    for item in collection.items.iter() {
        if item.world_space_deformed || item.blendshape_deformed {
            keys.insert(SkinCacheKey::from_draw_parts(
                item.space_id,
                item.render_context,
                item.skinned,
                item.instance_id,
            ));
        }
    }
}
