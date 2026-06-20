//! Prepared snapshot rebuild and per-space reuse for retained render-world state.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::ops::Range;

use crate::cpu_parallelism::FrameParallelPolicy;
use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, WorldMeshSceneRead};
use crate::shared::RenderingContext;

use super::super::prepared_renderables::{FramePreparedDraw, expand_render_buffer_renderers_into};
use super::state::RenderWorldSpace;
use super::{
    RenderWorld, SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES, snapshot_rebuild_admission,
};

/// Shared inputs used while rebuilding a prepared snapshot.
struct SnapshotRebuildInputs<'a, S: WorldMeshSceneRead + ?Sized> {
    /// Scene mirror queried for active spaces and particle render buffers.
    scene: &'a S,
    /// Mesh pool used when expanding generated particle render-buffer draws.
    mesh_pool: &'a MeshPool,
    /// Point render-buffer assets available for generated particle draws.
    point_render_buffers: &'a HashMap<i32, crate::particles::PointRenderBufferAsset>,
    /// Rendering context used for material override selection.
    render_context: RenderingContext,
}

/// Summary counters emitted by one prepared-snapshot rebuild.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SnapshotRebuildStats {
    /// Snapshot copy tasks built for dirty retained renderer templates.
    pub(super) task_count: usize,
    /// Retained draw templates considered by this snapshot rebuild.
    pub(super) retained_draw_count: usize,
    /// Active render spaces copied from the previous prepared snapshot.
    pub(super) reused_space_count: usize,
}

/// Renderer table selected by a prepared-snapshot rebuild task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SnapshotRendererTable {
    /// Static renderer templates.
    Static,
    /// Skinned renderer templates.
    Skinned,
}

/// Source region copied by one prepared-snapshot rebuild task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum SnapshotRebuildSource {
    /// A contiguous renderer row range.
    RendererRange {
        /// Renderer table copied by this task.
        table: SnapshotRendererTable,
        /// Renderer index range copied by this task.
        range: Range<usize>,
    },
    /// A contiguous draw-template range inside one oversized renderer row.
    RendererDrawRange {
        /// Renderer table copied by this task.
        table: SnapshotRendererTable,
        /// Renderer row copied by this task.
        renderer_index: usize,
        /// Draw-template range copied by this task.
        range: Range<usize>,
    },
    /// Generated particle render-buffer draws for one render space.
    ParticleRenderBuffers {
        /// Render space whose generated particle draws are expanded by this task.
        space_id: RenderSpaceId,
    },
}

/// One deterministic chunk of retained renderer templates copied into the prepared snapshot.
#[derive(Clone)]
pub(super) struct SnapshotRebuildTask<'a> {
    /// Index of the active render space in frame iteration order.
    pub(super) space_index: usize,
    /// Retained render space borrowed by this task.
    space: &'a RenderWorldSpace,
    /// Source region copied by this task.
    pub(super) source: SnapshotRebuildSource,
}

impl SnapshotRebuildTask<'_> {
    /// Returns the number of retained draw templates this task will emit.
    pub(super) fn retained_template_count(&self) -> usize {
        match &self.source {
            SnapshotRebuildSource::RendererRange {
                table: SnapshotRendererTable::Static,
                range,
            } => self
                .space
                .retained_static_template_count_for_range(range.clone()),
            SnapshotRebuildSource::RendererRange {
                table: SnapshotRendererTable::Skinned,
                range,
            } => self
                .space
                .retained_skinned_template_count_for_range(range.clone()),
            SnapshotRebuildSource::RendererDrawRange { range, .. } => range.len(),
            SnapshotRebuildSource::ParticleRenderBuffers { .. } => 0,
        }
    }

    /// Copies this task's retained draw templates into `draws`.
    fn append_draws_to<S>(
        &self,
        draws: &mut Vec<FramePreparedDraw>,
        inputs: &SnapshotRebuildInputs<'_, S>,
    ) where
        S: WorldMeshSceneRead + ?Sized,
    {
        match &self.source {
            SnapshotRebuildSource::RendererRange {
                table: SnapshotRendererTable::Static,
                range,
            } => self
                .space
                .append_static_draws_range_to(range.clone(), draws),
            SnapshotRebuildSource::RendererRange {
                table: SnapshotRendererTable::Skinned,
                range,
            } => self
                .space
                .append_skinned_draws_range_to(range.clone(), draws),
            SnapshotRebuildSource::RendererDrawRange {
                table: SnapshotRendererTable::Static,
                renderer_index,
                range,
            } => self.space.append_static_renderer_draws_range_to(
                *renderer_index,
                range.clone(),
                draws,
            ),
            SnapshotRebuildSource::RendererDrawRange {
                table: SnapshotRendererTable::Skinned,
                renderer_index,
                range,
            } => self.space.append_skinned_renderer_draws_range_to(
                *renderer_index,
                range.clone(),
                draws,
            ),
            SnapshotRebuildSource::ParticleRenderBuffers { space_id } => {
                expand_render_buffer_renderers_into(
                    draws,
                    inputs.scene,
                    inputs.mesh_pool,
                    inputs.point_render_buffers,
                    inputs.render_context,
                    *space_id,
                );
            }
        }
    }
}

/// Rebuilds the per-view-consumable prepared snapshot from retained renderer templates.
pub(super) fn rebuild_prepared_snapshot<S>(
    render_world: &mut RenderWorld,
    scene: &S,
    mesh_pool: &MeshPool,
    point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
    render_context: RenderingContext,
    dirty_spaces: Option<&HashSet<RenderSpaceId>>,
) -> SnapshotRebuildStats
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
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
    render_world.prepared.begin_cached_rebuild(render_context);
    let reused_space_ids =
        reusable_prepared_space_ids(&render_world.prepared, &active_space_ids, dirty_spaces);
    let active_spaces = active_space_ids
        .iter()
        .enumerate()
        .filter_map(|(space_index, id)| {
            if reused_space_ids.contains(id) {
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
    let mut tasks = build_snapshot_rebuild_tasks(&active_spaces);
    extend_snapshot_particle_tasks(&mut tasks, &active_spaces, scene);
    let stats = SnapshotRebuildStats {
        task_count: tasks.len(),
        retained_draw_count,
        reused_space_count: reused_space_ids.len(),
    };
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
                    task.append_draws_to(&mut draws, &inputs);
                    (task.space_index, draws)
                })
                .collect::<Vec<_>>()
        });
    drop(tasks);
    drop(active_spaces);
    if let Some(outputs) = parallel_outputs {
        rebuild_snapshot_parallel(render_world, &active_space_ids, &reused_space_ids, outputs);
    } else {
        rebuild_snapshot_serial(render_world, &inputs, active_space_ids, &reused_space_ids);
    }
    render_world.prepared.finish_cached_rebuild(scene);
    stats
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

/// Appends generated particle render-buffer tasks for active spaces that need them.
fn extend_snapshot_particle_tasks<'a, S>(
    tasks: &mut Vec<SnapshotRebuildTask<'a>>,
    active_spaces: &[(usize, RenderSpaceId, &'a RenderWorldSpace)],
    scene: &S,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    for (space_index, space_id, space) in active_spaces {
        if !space_has_render_buffer_renderers(scene, *space_id) {
            continue;
        }
        tasks.push(SnapshotRebuildTask {
            space_index: *space_index,
            space,
            source: SnapshotRebuildSource::ParticleRenderBuffers {
                space_id: *space_id,
            },
        });
    }
}

/// Returns whether a render space has generated particle render-buffer rows.
fn space_has_render_buffer_renderers<S>(scene: &S, space_id: RenderSpaceId) -> bool
where
    S: WorldMeshSceneRead + ?Sized,
{
    scene
        .billboard_render_buffers(space_id)
        .is_some_and(|renderers| !renderers.is_empty())
        || scene
            .trail_render_buffers(space_id)
            .is_some_and(|renderers| !renderers.is_empty())
        || scene
            .mesh_render_buffers(space_id)
            .is_some_and(|renderers| !renderers.is_empty())
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
        if template_count > SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES {
            if let Some(start) = range_start.take() {
                push_snapshot_renderer_range_task(
                    tasks,
                    space_index,
                    space,
                    table,
                    start..renderer_index,
                );
                range_template_count = 0;
            }
            extend_snapshot_renderer_draw_tasks(
                tasks,
                space_index,
                space,
                table,
                renderer_index,
                template_count,
            );
            continue;
        }
        let start = *range_start.get_or_insert(renderer_index);
        range_template_count = range_template_count.saturating_add(template_count);
        if range_template_count >= SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES {
            push_snapshot_renderer_range_task(
                tasks,
                space_index,
                space,
                table,
                start..renderer_index + 1,
            );
            range_start = None;
            range_template_count = 0;
        }
    }
    if let Some(start) = range_start {
        push_snapshot_renderer_range_task(tasks, space_index, space, table, start..renderer_count);
    }
}

/// Appends one renderer-range snapshot-copy task.
fn push_snapshot_renderer_range_task<'a>(
    tasks: &mut Vec<SnapshotRebuildTask<'a>>,
    space_index: usize,
    space: &'a RenderWorldSpace,
    table: SnapshotRendererTable,
    range: Range<usize>,
) {
    if range.is_empty() {
        return;
    }
    tasks.push(SnapshotRebuildTask {
        space_index,
        space,
        source: SnapshotRebuildSource::RendererRange { table, range },
    });
}

/// Appends draw-range snapshot-copy tasks for one oversized renderer row.
fn extend_snapshot_renderer_draw_tasks<'a>(
    tasks: &mut Vec<SnapshotRebuildTask<'a>>,
    space_index: usize,
    space: &'a RenderWorldSpace,
    table: SnapshotRendererTable,
    renderer_index: usize,
    template_count: usize,
) {
    let mut start = 0usize;
    while start < template_count {
        let end = start
            .saturating_add(SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES)
            .min(template_count);
        tasks.push(SnapshotRebuildTask {
            space_index,
            space,
            source: SnapshotRebuildSource::RendererDrawRange {
                table,
                renderer_index,
                range: start..end,
            },
        });
        start = end;
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
    active_space_ids: &[RenderSpaceId],
    reused_space_ids: &HashSet<RenderSpaceId>,
    outputs: Vec<(usize, Vec<FramePreparedDraw>)>,
) {
    let mut output_index = 0usize;
    for (space_index, &id) in active_space_ids.iter().enumerate() {
        render_world.prepared.push_cached_space(id);
        if reused_space_ids.contains(&id) {
            while outputs
                .get(output_index)
                .is_some_and(|(task_space_index, _)| *task_space_index == space_index)
            {
                output_index += 1;
            }
            render_world
                .prepared
                .extend_previous_cached_draws_for_space(id);
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
        }
    }
}

fn rebuild_snapshot_serial<S>(
    render_world: &mut RenderWorld,
    inputs: &SnapshotRebuildInputs<'_, S>,
    active_space_ids: Vec<RenderSpaceId>,
    reused_space_ids: &HashSet<RenderSpaceId>,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::serial");
    for id in active_space_ids {
        render_world.prepared.push_cached_space(id);
        if reused_space_ids.contains(&id) {
            render_world
                .prepared
                .extend_previous_cached_draws_for_space(id);
        } else if let Some(space) = render_world.spaces.get(&id) {
            space.append_to_prepared(&mut render_world.prepared);
            append_particle_draws(render_world, inputs, id);
        }
    }
}

fn reusable_prepared_space_ids(
    prepared: &super::super::prepared_renderables::FramePreparedRenderables,
    active_space_ids: &[RenderSpaceId],
    dirty_spaces: Option<&HashSet<RenderSpaceId>>,
) -> HashSet<RenderSpaceId> {
    let Some(dirty_spaces) = dirty_spaces else {
        return HashSet::new();
    };
    let mut reused = HashSet::new();
    for &id in active_space_ids {
        if dirty_spaces.contains(&id) {
            continue;
        }
        if prepared.has_previous_cached_draws_for_space(id) {
            reused.insert(id);
        }
    }
    reused
}

fn append_particle_draws<S>(
    render_world: &mut RenderWorld,
    inputs: &SnapshotRebuildInputs<'_, S>,
    id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    expand_render_buffer_renderers_into(
        render_world.prepared.draws_mut_for_cached_rebuild(),
        inputs.scene,
        inputs.mesh_pool,
        inputs.point_render_buffers,
        inputs.render_context,
        id,
    );
}
