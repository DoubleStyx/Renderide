//! Prepared snapshot rebuild and per-space reuse for retained render-world state.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::ops::Range;

use crate::cpu_parallelism::FrameParallelPolicy;
use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

use super::super::prepared_renderables::{FramePreparedDraw, expand_render_buffer_renderers_into};
use super::state::RenderWorldSpace;
use super::{
    RenderWorld, SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES, snapshot_rebuild_admission,
};

/// Shared inputs used while rebuilding a prepared snapshot.
struct SnapshotRebuildInputs<'a> {
    /// Scene mirror queried for active spaces and particle render buffers.
    scene: &'a SceneCoordinator,
    /// Mesh pool used when expanding generated particle render-buffer draws.
    mesh_pool: &'a MeshPool,
    /// Point render-buffer assets available for generated particle draws.
    point_render_buffers: &'a HashMap<i32, crate::particles::PointRenderBufferAsset>,
    /// Rendering context used for material override selection.
    render_context: RenderingContext,
}

/// Renderer table selected by a prepared-snapshot rebuild task.
#[derive(Clone, Copy)]
pub(super) enum SnapshotRendererTable {
    /// Static renderer templates.
    Static,
    /// Skinned renderer templates.
    Skinned,
}

/// One deterministic chunk of retained renderer templates copied into the prepared snapshot.
#[derive(Clone)]
pub(super) struct SnapshotRebuildTask<'a> {
    /// Index of the active render space in frame iteration order.
    pub(super) space_index: usize,
    /// Retained render space borrowed by this task.
    space: &'a RenderWorldSpace,
    /// Renderer table copied by this task.
    table: SnapshotRendererTable,
    /// Renderer index range copied by this task.
    pub(super) range: Range<usize>,
}

impl SnapshotRebuildTask<'_> {
    /// Returns the number of retained draw templates this task will emit.
    pub(super) fn retained_template_count(&self) -> usize {
        match self.table {
            SnapshotRendererTable::Static => self
                .space
                .retained_static_template_count_for_range(self.range.clone()),
            SnapshotRendererTable::Skinned => self
                .space
                .retained_skinned_template_count_for_range(self.range.clone()),
        }
    }

    /// Copies this task's retained draw templates into `draws`.
    fn append_draws_to(&self, draws: &mut Vec<FramePreparedDraw>) {
        match self.table {
            SnapshotRendererTable::Static => self
                .space
                .append_static_draws_range_to(self.range.clone(), draws),
            SnapshotRendererTable::Skinned => self
                .space
                .append_skinned_draws_range_to(self.range.clone(), draws),
        }
    }
}

/// Rebuilds the per-view-consumable prepared snapshot from retained renderer templates.
pub(super) fn rebuild_prepared_snapshot(
    render_world: &mut RenderWorld,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
    render_context: RenderingContext,
    dirty_spaces: Option<&HashSet<RenderSpaceId>>,
) {
    profiling::scope!("mesh::render_world::rebuild_prepared_snapshot");
    let inputs = SnapshotRebuildInputs {
        scene,
        mesh_pool,
        point_render_buffers,
        render_context,
    };
    let active_space_ids = scene
        .render_space_ids()
        .filter(|id| {
            render_world
                .spaces
                .get(id)
                .is_some_and(|space| space.active)
        })
        .collect::<Vec<_>>();
    let reused_space_draws =
        reusable_prepared_space_draws(render_world, &active_space_ids, dirty_spaces);
    let active_spaces = active_space_ids
        .iter()
        .enumerate()
        .filter_map(|(space_index, id)| {
            if reused_space_draws.contains_key(id) {
                return None;
            }
            render_world
                .spaces
                .get(id)
                .map(|space| (space_index, *id, space))
        })
        .collect::<Vec<_>>();
    let retained_draw_count = active_spaces
        .iter()
        .map(|(_, _, space)| *space)
        .map(RenderWorldSpace::retained_template_count)
        .sum::<usize>();
    let tasks = build_snapshot_rebuild_tasks(&active_spaces);
    let policy = FrameParallelPolicy::for_current_thread_pool();
    let parallel_outputs = snapshot_rebuild_admission(policy, tasks.len(), retained_draw_count)
        .chunk_size()
        .map(|chunk_size| {
            profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::parallel");
            tasks
                .par_iter()
                .with_min_len(chunk_size)
                .map(|task| {
                    profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::worker");
                    let mut draws = Vec::with_capacity(task.retained_template_count());
                    task.append_draws_to(&mut draws);
                    (task.space_index, draws)
                })
                .collect::<Vec<_>>()
        });
    drop(tasks);
    drop(active_spaces);
    render_world.prepared.begin_cached_rebuild(render_context);
    if let Some(outputs) = parallel_outputs {
        rebuild_snapshot_parallel(
            render_world,
            &inputs,
            &active_space_ids,
            &reused_space_draws,
            outputs,
        );
    } else {
        rebuild_snapshot_serial(render_world, &inputs, active_space_ids, &reused_space_draws);
    }
    render_world.prepared.finish_cached_rebuild();
}

/// Builds deterministic snapshot-copy tasks from active render spaces.
pub(super) fn build_snapshot_rebuild_tasks<'a>(
    active_spaces: &[(usize, RenderSpaceId, &'a RenderWorldSpace)],
) -> Vec<SnapshotRebuildTask<'a>> {
    let mut tasks = Vec::new();
    for (space_index, _, space) in active_spaces {
        extend_snapshot_table_tasks(
            &mut tasks,
            *space_index,
            space,
            SnapshotRendererTable::Static,
            space.static_renderers.len(),
        );
        extend_snapshot_table_tasks(
            &mut tasks,
            *space_index,
            space,
            SnapshotRendererTable::Skinned,
            space.skinned_renderers.len(),
        );
    }
    tasks
}

/// Appends chunked snapshot-copy tasks for one renderer table.
fn extend_snapshot_table_tasks<'a>(
    tasks: &mut Vec<SnapshotRebuildTask<'a>>,
    space_index: usize,
    space: &'a RenderWorldSpace,
    table: SnapshotRendererTable,
    renderer_count: usize,
) {
    let mut range_start = None;
    let mut range_template_count = 0usize;
    for renderer_index in 0..renderer_count {
        let template_count = retained_renderer_template_count(space, table, renderer_index);
        if template_count == 0 && range_start.is_none() {
            continue;
        }
        let start = *range_start.get_or_insert(renderer_index);
        range_template_count = range_template_count.saturating_add(template_count);
        if range_template_count >= SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES {
            tasks.push(SnapshotRebuildTask {
                space_index,
                space,
                table,
                range: start..renderer_index + 1,
            });
            range_start = None;
            range_template_count = 0;
        }
    }
    if let Some(start) = range_start {
        tasks.push(SnapshotRebuildTask {
            space_index,
            space,
            table,
            range: start..renderer_count,
        });
    }
}

/// Returns the retained template count for one renderer record in a snapshot source table.
fn retained_renderer_template_count(
    space: &RenderWorldSpace,
    table: SnapshotRendererTable,
    renderer_index: usize,
) -> usize {
    match table {
        SnapshotRendererTable::Static => space
            .static_renderers
            .get(renderer_index)
            .map_or(0, |renderer| renderer.draws.len()),
        SnapshotRendererTable::Skinned => space
            .skinned_renderers
            .get(renderer_index)
            .map_or(0, |renderer| renderer.draws.len()),
    }
}

fn rebuild_snapshot_parallel(
    render_world: &mut RenderWorld,
    inputs: &SnapshotRebuildInputs<'_>,
    active_space_ids: &[RenderSpaceId],
    reused_space_draws: &HashMap<RenderSpaceId, Vec<FramePreparedDraw>>,
    outputs: Vec<(usize, Vec<FramePreparedDraw>)>,
) {
    let mut output_index = 0usize;
    for (space_index, &id) in active_space_ids.iter().enumerate() {
        render_world.prepared.push_cached_space(id);
        if let Some(reused) = reused_space_draws.get(&id) {
            while outputs
                .get(output_index)
                .is_some_and(|(task_space_index, _)| *task_space_index == space_index)
            {
                output_index += 1;
            }
            render_world.prepared.extend_cached_draws(reused);
        } else {
            while outputs
                .get(output_index)
                .is_some_and(|(task_space_index, _)| *task_space_index == space_index)
            {
                render_world
                    .prepared
                    .extend_cached_draws(&outputs[output_index].1);
                output_index += 1;
            }
            append_particle_draws(render_world, inputs, id);
        }
    }
}

fn rebuild_snapshot_serial(
    render_world: &mut RenderWorld,
    inputs: &SnapshotRebuildInputs<'_>,
    active_space_ids: Vec<RenderSpaceId>,
    reused_space_draws: &HashMap<RenderSpaceId, Vec<FramePreparedDraw>>,
) {
    profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::serial");
    for id in active_space_ids {
        render_world.prepared.push_cached_space(id);
        if let Some(reused) = reused_space_draws.get(&id) {
            render_world.prepared.extend_cached_draws(reused);
        } else if let Some(space) = render_world.spaces.get(&id) {
            space.append_to_prepared(&mut render_world.prepared);
            append_particle_draws(render_world, inputs, id);
        }
    }
}

fn reusable_prepared_space_draws(
    render_world: &RenderWorld,
    active_space_ids: &[RenderSpaceId],
    dirty_spaces: Option<&HashSet<RenderSpaceId>>,
) -> HashMap<RenderSpaceId, Vec<FramePreparedDraw>> {
    let Some(dirty_spaces) = dirty_spaces else {
        return HashMap::new();
    };
    let mut reused = HashMap::new();
    for &id in active_space_ids {
        if dirty_spaces.contains(&id) {
            continue;
        }
        if let Some(draws) = render_world.prepared.cached_draws_for_space(id) {
            reused.insert(id, draws.to_vec());
        }
    }
    reused
}

fn append_particle_draws(
    render_world: &mut RenderWorld,
    inputs: &SnapshotRebuildInputs<'_>,
    id: RenderSpaceId,
) {
    expand_render_buffer_renderers_into(
        render_world.prepared.draws_mut_for_cached_rebuild(),
        inputs.scene,
        inputs.mesh_pool,
        inputs.point_render_buffers,
        inputs.render_context,
        id,
    );
}
