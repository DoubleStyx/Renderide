//! Per-space transform filter mask construction for world-mesh draw collection.

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::cpu_parallelism::{
    ParallelAdmission, ParallelAdmissionSite, current_reference_worker_count,
    has_visibility_parallel_work, record_parallel_admission,
};
use crate::scene::RenderSpaceId;

use super::DrawCollectionContext;

/// Render spaces assigned to one filter-mask construction worker.
const FILTER_MASK_PARALLEL_CHUNK_SPACES: usize = 1;
/// Render-space count at which per-space filter mask construction uses Rayon.
const FILTER_MASK_PARALLEL_MIN_SPACES: usize = FILTER_MASK_PARALLEL_CHUNK_SPACES * 2;

/// Returns the filter-mask admission decision for a known worker count.
#[inline]
fn filter_mask_admission(
    space_count: usize,
    work_units: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if space_count >= FILTER_MASK_PARALLEL_MIN_SPACES
        && has_visibility_parallel_work(work_units, worker_count)
    {
        ParallelAdmission::Parallel {
            chunk_size: FILTER_MASK_PARALLEL_CHUNK_SPACES,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Estimates transform rows scanned by per-space filter-mask construction.
#[inline]
fn filter_mask_work_units(space_ids: &[RenderSpaceId], ctx: &DrawCollectionContext<'_>) -> usize {
    space_ids
        .iter()
        .filter_map(|sid| {
            ctx.scene
                .space(*sid)
                .map(|space| space.local_transforms().len())
        })
        .sum()
}

/// Builds per-space `Vec<bool>` masks from [`DrawCollectionContext::transform_filter`].
///
/// Returns an empty map when no transform filter was provided.
pub(super) fn build_per_space_filter_masks(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> HashMap<RenderSpaceId, Vec<bool>> {
    let Some(transform_filter) = ctx.transform_filter else {
        return HashMap::new();
    };

    let work_units = filter_mask_work_units(space_ids, ctx);
    let admission = filter_mask_admission(
        space_ids.len(),
        work_units,
        current_reference_worker_count(),
    );
    record_parallel_admission(
        ParallelAdmissionSite::FilterMasks,
        work_units,
        space_ids.len(),
        admission,
    );

    let pairs = if admission.is_parallel() {
        space_ids
            .par_iter()
            .with_min_len(FILTER_MASK_PARALLEL_CHUNK_SPACES)
            .copied()
            .filter_map(|sid| {
                let mask = transform_filter.build_pass_mask(ctx.scene, sid)?;
                Some((sid, mask))
            })
            .collect::<Vec<_>>()
    } else {
        space_ids
            .iter()
            .copied()
            .filter_map(|sid| {
                let mask = transform_filter.build_pass_mask(ctx.scene, sid)?;
                Some((sid, mask))
            })
            .collect::<Vec<_>>()
    };
    pairs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};
    use hashbrown::HashSet;

    use super::*;
    use crate::gpu_pools::MeshPool;
    use crate::materials::host_data::{
        MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry,
    };
    use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
    use crate::scene::SceneCoordinator;
    use crate::shared::{RenderTransform, RenderingContext};
    use crate::world_mesh::draw_prep::filter::CameraTransformDrawFilter;

    fn with_draw_context(
        scene: &SceneCoordinator,
        transform_filter: Option<&CameraTransformDrawFilter>,
        test: impl FnOnce(&DrawCollectionContext<'_>),
    ) {
        let mesh_pool = MeshPool::default_pool();
        let store = MaterialPropertyStore::new();
        let material_dict = MaterialDictionary::new(&store);
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let registry = PropertyIdRegistry::new();
        let property_ids = MaterialPipelinePropertyIds::new(&registry);
        let ctx = DrawCollectionContext {
            scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            material_router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: Default::default(),
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            mesh_lod_bias: 2.0,
            transform_filter,
            render_space_filter: None,
            material_cache: None,
            reflection_probes: None,
            prepared: None,
        };
        test(&ctx);
    }

    fn seed_space(scene: &mut SceneCoordinator, id: RenderSpaceId, parents: Vec<i32>) {
        let transforms = vec![RenderTransform::default(); parents.len()];
        scene.test_seed_space_identity_worlds(id, transforms, parents);
    }

    #[test]
    fn no_transform_filter_returns_empty_map() {
        let mut scene = SceneCoordinator::new();
        let space_id = RenderSpaceId(1);
        seed_space(&mut scene, space_id, vec![-1, 0]);

        with_draw_context(&scene, None, |ctx| {
            let masks = build_per_space_filter_masks(&[space_id], ctx);
            assert!(masks.is_empty());
        });
    }

    #[test]
    fn missing_spaces_produce_no_filter_entries() {
        let scene = SceneCoordinator::new();
        let filter = CameraTransformDrawFilter::default();

        with_draw_context(&scene, Some(&filter), |ctx| {
            let masks = build_per_space_filter_masks(&[RenderSpaceId(1), RenderSpaceId(2)], ctx);
            assert!(masks.is_empty());
        });
    }

    #[test]
    fn existing_spaces_forward_filter_masks_and_skip_missing_spaces() {
        let mut scene = SceneCoordinator::new();
        let first = RenderSpaceId(1);
        let second = RenderSpaceId(2);
        let missing = RenderSpaceId(3);
        seed_space(&mut scene, first, vec![-1, 0, 1]);
        seed_space(&mut scene, second, vec![-1, 0]);
        let filter = CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([1])),
            exclude: HashSet::new(),
        };

        with_draw_context(&scene, Some(&filter), |ctx| {
            let masks = build_per_space_filter_masks(&[first, missing, second], ctx);

            assert_eq!(masks.len(), 2);
            assert_eq!(masks.get(&first), Some(&vec![false, true, true]));
            assert_eq!(masks.get(&second), Some(&vec![false, true]));
            assert!(!masks.contains_key(&missing));
        });
    }

    #[test]
    fn parallel_filter_mask_builds_every_existing_space() {
        let mut scene = SceneCoordinator::new();
        let space_ids = [
            RenderSpaceId(1),
            RenderSpaceId(2),
            RenderSpaceId(3),
            RenderSpaceId(4),
        ];
        for &space_id in &space_ids {
            seed_space(&mut scene, space_id, vec![-1, 0, 1, 2]);
        }
        let filter = CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([1])),
            exclude: HashSet::new(),
        };

        with_draw_context(&scene, Some(&filter), |ctx| {
            let masks = build_per_space_filter_masks(&space_ids, ctx);

            assert_eq!(masks.len(), space_ids.len());
            for space_id in space_ids {
                assert_eq!(masks.get(&space_id), Some(&vec![false, true, true, true]));
            }
        });
    }
}
